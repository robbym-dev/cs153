"""Prototype: deterministic callout-circle detection + tiny-crop OCR.

For one page of a vector PDF:
  1. PyMuPDF finds every closed curve-only path whose bounding box looks like
     a keynote callout (~27pt diameter, ~0.84pt stroke, near-square).
  2. For each, render an 80x80pt region around the center at 400 DPI directly
     to PNG via fitz.Page.get_pixmap(clip=...).
  3. Ask Claude (low-effort, no thinking, tiny max_tokens) to read the code
     inside the circle. Parallel via ThreadPoolExecutor.
  4. Tally counts per code and compare to a reference spreadsheet.

Cache the per-callout OCR result JSON so the script is rerunnable without
hitting the API again.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz
import openpyxl
from anthropic import Anthropic

logger = logging.getLogger("probe_vector_callouts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
DEFAULT_SPREADSHEET = PROJECT_ROOT / "test_data" / "Park_Avenue_Elementary_School.xlsx"

# Callout-circle signature on PA original page 8 (see vector-geometry exploration).
TARGET_STROKE_WIDTH = 0.84
STROKE_WIDTH_TOL = 0.05
TARGET_DIAMETER = 27.0  # pt
DIAMETER_TOL = 3.0      # pt (accept 24..30 pt diameter)
ASPECT_RATIO_MAX = 1.15

CROP_HALF_PT = 40       # 80x80pt total crop
CROP_DPI = 400
OCR_MODEL = "claude-opus-4-7"
OCR_PROMPT = (
    "What keynote code is written inside this circle? Return ONLY the code, "
    "like WS5 or R2. Nothing else."
)
PARALLEL_WORKERS = 10
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")


@dataclass
class Callout:
    cx: float
    cy: float
    radius: float
    rect: tuple[float, float, float, float]


@dataclass
class OCRResult:
    cx: float
    cy: float
    raw: str
    code: str | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Geometric detection
# ---------------------------------------------------------------------------


def find_callouts(page) -> list[Callout]:
    out: list[Callout] = []
    for p in page.get_drawings():
        rect = p["rect"]
        w = rect.x1 - rect.x0
        h = rect.y1 - rect.y0
        if min(w, h) <= 0:
            continue
        diameter = max(w, h)
        if abs(diameter - TARGET_DIAMETER) > DIAMETER_TOL:
            continue
        if max(w, h) / min(w, h) > ASPECT_RATIO_MAX:
            continue
        width = p.get("width") or 0.0
        if abs(width - TARGET_STROKE_WIDTH) > STROKE_WIDTH_TOL:
            continue
        items = p.get("items", [])
        if not items:
            continue
        if any(it[0] not in ("c", "h", "re") for it in items):
            continue
        if sum(1 for it in items if it[0] == "c") < 2:
            continue
        out.append(
            Callout(
                cx=(rect.x0 + rect.x1) / 2,
                cy=(rect.y0 + rect.y1) / 2,
                radius=(w + h) / 4,
                rect=(rect.x0, rect.y0, rect.x1, rect.y1),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Per-callout OCR
# ---------------------------------------------------------------------------


def render_crop_png(page, callout: Callout) -> bytes:
    clip = fitz.Rect(
        callout.cx - CROP_HALF_PT, callout.cy - CROP_HALF_PT,
        callout.cx + CROP_HALF_PT, callout.cy + CROP_HALF_PT,
    ) & page.rect
    if clip.is_empty or clip.is_infinite or clip.width < 1 or clip.height < 1:
        raise ValueError(f"degenerate crop rect for callout at ({callout.cx:.1f},{callout.cy:.1f})")
    # Use a transform matrix instead of dpi= keyword to dodge the bandwriter
    # error fitz raises on certain clip + dpi combinations.
    scale = CROP_DPI / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return pix.tobytes("png")


def ocr_one(client: Anthropic, png_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    msg = client.messages.create(
        model=OCR_MODEL,
        max_tokens=50,
        output_config={"effort": "low"},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                },
                {"type": "text", "text": OCR_PROMPT},
            ],
        }],
    )
    for block in msg.content:
        if block.type == "text":
            return block.text.strip()
    return ""


def _clean_code(raw: str) -> str | None:
    """Normalize OCR output to a clean code, or None if unrecognizable."""
    s = raw.strip().upper().strip("`'\".,:;")
    # Sometimes the model returns 'The code is WS5.' — pull the last token that
    # matches our code pattern, if any.
    for token in reversed(re.split(r"[\s/,]+", s)):
        token = token.strip(".,;:()[]")
        m = CODE_RE.match(token)
        if m:
            return m.group(1)
    return None


def run_ocr_parallel(page, callouts: list[Callout]) -> list[OCRResult]:
    client = Anthropic()
    results: list[OCRResult | None] = [None] * len(callouts)
    crops: list[bytes | None] = []
    skipped = 0
    for c in callouts:
        try:
            crops.append(render_crop_png(page, c))
        except ValueError as exc:
            crops.append(None)
            skipped += 1
            logger.warning("  skipping off-page callout: %s", exc)
    logger.info("rendered %d crops at %d DPI (skipped %d degenerate)",
                len(crops) - skipped, CROP_DPI, skipped)

    def worker(idx: int) -> tuple[int, OCRResult]:
        if crops[idx] is None:
            return idx, OCRResult(cx=callouts[idx].cx, cy=callouts[idx].cy,
                                  raw="", code=None, error="degenerate crop")
        try:
            raw = ocr_one(client, crops[idx])
            code = _clean_code(raw)
            return idx, OCRResult(cx=callouts[idx].cx, cy=callouts[idx].cy,
                                  raw=raw, code=code)
        except Exception as exc:  # noqa: BLE001
            return idx, OCRResult(cx=callouts[idx].cx, cy=callouts[idx].cy,
                                  raw="", code=None, error=str(exc))

    completed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in range(len(crops))]
        for f in as_completed(futures):
            idx, res = f.result()
            results[idx] = res
            completed += 1
            if completed % 20 == 0 or completed == len(crops):
                logger.info("  OCR progress: %d/%d", completed, len(crops))
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Spreadsheet comparison
# ---------------------------------------------------------------------------


SPREADSHEET_CODE_RE = re.compile(r"^([A-Z]+\d+)\s*:")


def load_tyler_totals(path: Path) -> dict[str, dict[str, float]]:
    """code -> {unit: qty}. Aggregates across all DETAIL rows."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["DETAIL"]
    out: dict = defaultdict(lambda: defaultdict(float))
    for r in range(28, ws.max_row + 1):
        item = ws.cell(r, 1).value
        try:
            int(item)
        except (TypeError, ValueError):
            continue
        desc = ws.cell(r, 5).value
        qty = ws.cell(r, 6).value
        unit = ws.cell(r, 7).value
        if not desc or qty is None or not unit:
            continue
        m = SPREADSHEET_CODE_RE.match(str(desc).strip())
        if not m:
            continue
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            continue
        u = str(unit).strip().upper().replace(".", "")
        if u in ("FT", "LIN FT", "LINEAR FT"):
            u = "LF"
        elif u in ("SQ FT", "SQFT"):
            u = "SF"
        out[m.group(1)][u] += qty_f
    return {k: dict(v) for k, v in out.items()}


def print_summary(counts: Counter, tyler: dict, n_callouts: int,
                  n_recognized: int, raw_distribution: Counter):
    print()
    print("=" * 80)
    print("CALLOUT COUNTS")
    print("=" * 80)
    print(f"Geometric circles found:    {n_callouts}")
    print(f"OCR returned valid code:    {n_recognized}/{n_callouts} "
          f"({n_recognized/n_callouts*100:.1f}%)")
    if n_recognized < n_callouts:
        bad = raw_distribution.copy()
        for code in counts:
            bad[code] = 0
        non_codes = sum(v for k, v in raw_distribution.items()
                        if _clean_code(k) is None)
        print(f"Unrecognized OCR outputs:   {non_codes}")

    print()
    print(f"{'CODE':6} {'COUNT':>6}  Tyler quantities (any unit)")
    print("-" * 80)
    all_codes = sorted(set(counts) | set(tyler))
    matched_ea = 0
    matched_ea_total = 0
    for code in all_codes:
        c = counts.get(code, 0)
        tyler_row = tyler.get(code, {})
        tyler_summary = " + ".join(f"{q:g} {u}" for u, q in tyler_row.items()) or "—"
        marker = ""
        if "EA" in tyler_row and c > 0:
            matched_ea_total += 1
            ta = tyler_row["EA"]
            delta = c - ta
            pct = (delta / ta * 100) if ta else 0
            marker = (f"  Δ={delta:+.0f} ({pct:+.0f}%)"
                      + (" ✓" if abs(pct) <= 15 else " ✗"))
            if abs(pct) <= 15:
                matched_ea += 1
        print(f"{code:6} {c:>6}   {tyler_summary}{marker}")

    print()
    if matched_ea_total:
        print(f"EA-direct comparisons within ±15%: "
              f"{matched_ea}/{matched_ea_total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", default=str(DEFAULT_PDF))
    ap.add_argument("--page", type=int, default=8, help="1-indexed page number")
    ap.add_argument("--spreadsheet", default=str(DEFAULT_SPREADSHEET))
    ap.add_argument(
        "--results-json",
        default=str(PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
                    / "vector_callouts" / "page8_ocr.json"),
        help="Path to save per-callout OCR results (or load from if --reuse).",
    )
    ap.add_argument("--reuse", action="store_true",
                    help="Reuse cached OCR JSON if present (no API calls).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    results_path = Path(args.results_json)

    doc = fitz.open(pdf_path)
    page = doc.load_page(args.page - 1)
    logger.info("page %d: %.0f x %.0f pts", args.page, page.rect.width, page.rect.height)

    callouts = find_callouts(page)
    logger.info("geometric filter → %d candidate callout circle(s)", len(callouts))

    if args.reuse and results_path.exists():
        logger.info("reusing cached OCR results from %s", results_path)
        cached = json.loads(results_path.read_text())
        ocr_results = [OCRResult(**r) for r in cached["results"]]
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            return 1
        ocr_results = run_ocr_parallel(page, callouts)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({
            "pdf": str(pdf_path), "page": args.page,
            "n_callouts": len(callouts),
            "results": [asdict(r) for r in ocr_results],
        }, indent=2))
        logger.info("wrote %s", results_path)

    raw_distribution = Counter(r.raw for r in ocr_results)
    counts = Counter(r.code for r in ocr_results if r.code)
    n_recognized = sum(1 for r in ocr_results if r.code)

    tyler = load_tyler_totals(Path(args.spreadsheet))
    print_summary(counts, tyler, len(callouts), n_recognized, raw_distribution)

    doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

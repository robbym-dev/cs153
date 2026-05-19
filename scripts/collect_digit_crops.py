"""Collect labeled digit crops from confidently-LLM-read callout circles.

For each LLM-labeled callout across pages 7-10 of the PA original plans:
  1. Get the paths inside the circle (sorted top-to-bottom by y).
  2. Parse the code (e.g. "WS10" -> prefix "WS", digits "10").
  3. Sanity check: skip when path-count != character-count.
  4. The trailing N paths (where N = digit count) are the digit glyphs.
     The user's prompt said "x-position clustering" but our prior
     exploration established that characters are stacked vertically,
     not horizontally — so y-order is what we use.
  5. Render each digit path's bounding box (+1pt margin) at 600 DPI and
     save as a labeled PNG: data/digit_crops/{label}/crop_{page}_{idx}.png

Reuses the existing v2 OCR caches; zero new API calls.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
OCR_DIR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original" / "vector_callouts"
)
OUT_DIR = PROJECT_ROOT / "data" / "digit_crops"
PAGES = [7, 8, 9, 10]
DPI = 600
CIRCLE_DIAMETER = 27.0
INSIDE_MARGIN = 1.0
MIN_PATH_SIZE = 0.2
CROP_MARGIN_PT = 1.5  # render padding around each digit path bbox

CODE_RE = re.compile(r"^([A-Z]+)(\d+)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("digit_crops")


def paths_inside_circle(cx: float, cy: float, all_paths) -> list:
    r = CIRCLE_DIAMETER / 2
    out = []
    for p in all_paths:
        pr = p["rect"]
        if (
            pr.x0 >= cx - r + INSIDE_MARGIN
            and pr.x1 <= cx + r - INSIDE_MARGIN
            and pr.y0 >= cy - r + INSIDE_MARGIN
            and pr.y1 <= cy + r - INSIDE_MARGIN
            and max(pr.x1 - pr.x0, pr.y1 - pr.y0) >= MIN_PATH_SIZE
        ):
            out.append(p)
    return out


def split_code(code: str) -> tuple[str, str] | tuple[None, None]:
    """'WS10' -> ('WS', '10'); 'R2' -> ('R', '2'); 'A430' -> ('A', '430')."""
    m = CODE_RE.match(code)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def render_path_region(page, path_rect: fitz.Rect, margin_pt: float = CROP_MARGIN_PT) -> Image.Image:
    clip = fitz.Rect(
        path_rect.x0 - margin_pt, path_rect.y0 - margin_pt,
        path_rect.x1 + margin_pt, path_rect.y1 + margin_pt,
    ) & page.rect
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main() -> int:
    if not PDF.exists():
        print(f"error: PDF not found: {PDF}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    skipped_no_code = 0
    skipped_path_mismatch = 0
    skipped_bad_code = 0
    per_page: dict[int, int] = defaultdict(int)
    manifest: list[dict] = []

    doc = fitz.open(PDF)
    for page_no in PAGES:
        cache_path = OCR_DIR / f"page{page_no}_ocr_v2.json"
        if not cache_path.exists():
            log.warning(f"  no cache for page {page_no}: {cache_path}")
            continue
        callouts = json.loads(cache_path.read_text())["ocr_results"]
        page = doc.load_page(page_no - 1)
        all_paths = page.get_drawings()
        log.info(f"page {page_no}: {len(callouts)} callouts cached, "
                 f"{sum(1 for c in callouts if c.get('code'))} LLM-labeled")

        for idx, c in enumerate(callouts):
            code = c.get("code")
            if not code:
                skipped_no_code += 1
                continue
            prefix, digits = split_code(code)
            if not digits:
                skipped_bad_code += 1
                continue
            inside = paths_inside_circle(c["cx"], c["cy"], all_paths)
            if not inside:
                continue
            by_y = sorted(inside, key=lambda p: (p["rect"].y0 + p["rect"].y1) / 2)
            if len(by_y) != len(code):
                skipped_path_mismatch += 1
                continue

            # Trailing N paths are the digits in code order
            n_digits = len(digits)
            digit_paths = by_y[-n_digits:]
            for digit_char, dp in zip(digits, digit_paths):
                out_dir = OUT_DIR / digit_char
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"crop_p{page_no}_c{idx}_d{digit_char}.png"
                crop = render_path_region(page, dp["rect"])
                crop.save(out_dir / fname)
                counts[digit_char] += 1
                per_page[page_no] += 1
                manifest.append({
                    "page": page_no, "callout_idx": idx, "code": code,
                    "digit_char": digit_char,
                    "rel_path": str((out_dir / fname).relative_to(PROJECT_ROOT)),
                    "path_bbox": [round(dp["rect"].x0, 2), round(dp["rect"].y0, 2),
                                  round(dp["rect"].x1, 2), round(dp["rect"].y1, 2)],
                    "in_legend": c.get("in_legend", False),
                })
    doc.close()

    # Persist manifest
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "source_pdf": str(PDF.name),
        "pages": PAGES,
        "dpi": DPI,
        "crop_margin_pt": CROP_MARGIN_PT,
        "counts_per_class": dict(counts),
        "total": sum(counts.values()),
        "items": manifest,
    }, indent=2))

    # Report
    print()
    print("=" * 60)
    print(f"Digit crop collection — total: {sum(counts.values())}")
    print("=" * 60)
    print(f"Saved under: {OUT_DIR.relative_to(PROJECT_ROOT)}/")
    print()
    print(f"  Class  count")
    for d in "0123456789":
        bar = "█" * counts.get(d, 0)
        print(f"  {d}      {counts.get(d, 0):>4}  {bar}")
    print(f"  TOTAL  {sum(counts.values()):>4}")

    print(f"\n  Per-page contribution:")
    for p in sorted(per_page):
        print(f"    page {p}: {per_page[p]} digit crops")

    print(f"\n  Skipped:")
    print(f"    callouts with no LLM read:      {skipped_no_code}")
    print(f"    LLM reads with non-digit code:  {skipped_bad_code}")
    print(f"    path-count != char-count:       {skipped_path_mismatch}")

    print(f"\n  Manifest: {(OUT_DIR / 'manifest.json').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

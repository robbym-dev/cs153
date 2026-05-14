"""One-off: re-OCR page 8 callouts using crops from a single full-page raster.

The vector-callout v2 pipeline renders each 120x120pt crop independently from
the PDF vector source via fitz. The hypothesis: rasterizing the full page
once at 600 DPI (with PDF's anti-aliased text rendering applied to the whole
page in one pass) produces thicker / more readable glyphs than per-crop
vector rendering. This experiment reuses the same callout positions saved
in page8_ocr_v2.json so the result is a direct A/B against the prior run.

Quick experiment, no pipeline refactor.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from anthropic import Anthropic
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PAGE = 8
PRIOR_OCR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_raster_crop.json"
)
DPI = 600
CROP_HALF_PT = 60       # 120x120pt
CROP_HALF_PX = int(CROP_HALF_PT * DPI / 72)
PARALLEL = 10
MODEL = "claude-opus-4-7"
PROMPT = (
    "What keynote code is written inside this circle? Return ONLY the code, "
    "like WS5 or R2. Nothing else."
)
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("raster_crop")


def clean_code(raw: str) -> str | None:
    s = raw.strip().upper().strip("`'\".,:;")
    for token in reversed(re.split(r"[\s/,]+", s)):
        token = token.strip(".,;:()[]")
        m = CODE_RE.match(token)
        if m:
            return m.group(1)
    return None


def main() -> int:
    if not PRIOR_OCR.exists():
        print(f"error: prior cache not found: {PRIOR_OCR}", file=sys.stderr)
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    prior = json.loads(PRIOR_OCR.read_text())
    callouts = prior["ocr_results"]  # has cx, cy, raw, code, in_legend
    log.info("loaded %d callouts from prior v2 cache", len(callouts))

    log.info("rendering full page at %d DPI...", DPI)
    doc = fitz.open(PDF)
    page = doc.load_page(PAGE - 1)
    scale = DPI / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    log.info("  rendered %d x %d px (%d KB)",
             pix.width, pix.height, len(pix.samples) // 1024)

    # Convert to PIL once
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    def crop_for(cx: float, cy: float) -> bytes | None:
        x0 = int(cx * scale) - CROP_HALF_PX
        y0 = int(cy * scale) - CROP_HALF_PX
        x1 = x0 + 2 * CROP_HALF_PX
        y1 = y0 + 2 * CROP_HALF_PX
        # Clip to image bounds
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(img.width, x1), min(img.height, y1)
        if x1c - x0c < 50 or y1c - y0c < 50:
            return None
        crop = img.crop((x0c, y0c, x1c, y1c))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    crops = [crop_for(c["cx"], c["cy"]) for c in callouts]
    log.info("cropped %d / %d (skipped %d as off-page)",
             sum(1 for c in crops if c is not None), len(crops),
             sum(1 for c in crops if c is None))

    client = Anthropic()

    def worker(idx: int):
        c = callouts[idx]
        if crops[idx] is None:
            return idx, {"raw": "", "code": None, "error": "off-page",
                         "cx": c["cx"], "cy": c["cy"], "in_legend": c["in_legend"]}
        b64 = base64.standard_b64encode(crops[idx]).decode("ascii")
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=50,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/png",
                                                  "data": b64}},
                    {"type": "text", "text": PROMPT},
                ]}],
            )
            raw = ""
            for block in msg.content:
                if block.type == "text":
                    raw = block.text.strip(); break
            return idx, {"raw": raw, "code": clean_code(raw), "error": None,
                         "cx": c["cx"], "cy": c["cy"], "in_legend": c["in_legend"]}
        except Exception as exc:  # noqa: BLE001
            return idx, {"raw": "", "code": None, "error": str(exc),
                         "cx": c["cx"], "cy": c["cy"], "in_legend": c["in_legend"]}

    results: list[dict] = [None] * len(callouts)  # type: ignore[list-item]
    done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futures = [ex.submit(worker, i) for i in range(len(callouts))]
        for f in as_completed(futures):
            i, r = f.result()
            results[i] = r
            done += 1
            if done % 25 == 0 or done == len(callouts):
                log.info("  OCR %d/%d", done, len(callouts))

    OUT.write_text(json.dumps({"page": PAGE, "dpi": DPI,
                                "crop_half_pt": CROP_HALF_PT,
                                "crop_source": "single full-page raster",
                                "results": results}, indent=2))
    log.info("wrote %s", OUT)

    # Compare
    print()
    print("=" * 80)
    print(f"PAGE 8: per-crop vector render  vs  single full-page raster crop")
    print("=" * 80)

    prior_results = prior["ocr_results"]
    prior_valid = sum(1 for r in prior_results if r.get("code"))
    new_valid = sum(1 for r in results if r.get("code"))
    print(f"\nValid OCR reads:")
    print(f"  prior (per-crop vector @ 600 DPI):    {prior_valid}/{len(prior_results)} ({prior_valid/len(prior_results)*100:.1f}%)")
    print(f"  new   (full-page raster @ 600 DPI):   {new_valid}/{len(results)} ({new_valid/len(results)*100:.1f}%)")

    # Per-callout outcome diff
    flipped_to_valid = 0
    flipped_to_invalid = 0
    same_code = 0
    diff_code = 0
    for prev, new in zip(prior_results, results):
        if prev.get("code") and new.get("code"):
            if prev["code"] == new["code"]:
                same_code += 1
            else:
                diff_code += 1
        elif new.get("code"):
            flipped_to_valid += 1
        elif prev.get("code"):
            flipped_to_invalid += 1
    print(f"\nPer-callout outcome:")
    print(f"  both valid + agree:        {same_code}")
    print(f"  both valid + disagree:     {diff_code}")
    print(f"  invalid → valid (gained):  {flipped_to_valid}")
    print(f"  valid → invalid (lost):    {flipped_to_invalid}")

    # Code counts (placed only)
    prior_placed = Counter(r["code"] for r in prior_results
                           if r.get("code") and not r.get("in_legend"))
    new_placed = Counter(r["code"] for r in results
                         if r.get("code") and not r.get("in_legend"))
    all_codes = sorted(set(prior_placed) | set(new_placed))
    print(f"\nPlaced-callout counts per code (on page 8):")
    print(f"  {'CODE':6}  {'prior':>5}  {'new':>5}  Δ")
    for code in all_codes:
        p = prior_placed.get(code, 0)
        n = new_placed.get(code, 0)
        d = n - p
        marker = " ✓ better" if d > 0 else (" ✗ worse" if d < 0 else "")
        print(f"  {code:6}  {p:>5}  {n:>5}  {d:+d}{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

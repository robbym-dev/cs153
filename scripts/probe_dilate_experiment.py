"""One-off: apply morphological dilation + softer OCR prompt on page 8 crops.

Builds on probe_raster_crop_experiment.py (full-page raster, then PIL crops).
Adds two changes:
  1. PIL ImageFilter.MinFilter(3) to thicken thin black strokes by ~1 px.
     (User-requested filter was MaxFilter; on black-on-white drawings the
     correct dilator-of-dark-pixels is MinFilter — same intent.)
  2. New OCR prompt that solicits low-confidence reads instead of
     blank-when-uncertain behavior.
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
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PAGE = 8
PRIOR_OCR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_dilate.json"
)
DPI = 600
CROP_HALF_PT = 60
CROP_HALF_PX = int(CROP_HALF_PT * DPI / 72)
PARALLEL = 10
MODEL = "claude-opus-4-7"
PROMPT = (
    "This circle contains a construction keynote code made of thin outlined "
    "letterforms. The text may appear very faint. Report your best guess at "
    "the 2-5 character code (letters followed by digits, like WS5 or R2), "
    "even if you're not confident. If you can make out any partial letters "
    "or digits, report them. Return ONLY the code, nothing else."
)
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dilate")


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
    callouts = prior["ocr_results"]
    log.info("loaded %d callouts from prior v2 cache", len(callouts))

    log.info("rendering full page at %d DPI...", DPI)
    doc = fitz.open(PDF)
    page = doc.load_page(PAGE - 1)
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    log.info("  rendered %d x %d px", pix.width, pix.height)

    def crop_for(cx: float, cy: float) -> bytes | None:
        x0 = int(cx * scale) - CROP_HALF_PX
        y0 = int(cy * scale) - CROP_HALF_PX
        x1 = x0 + 2 * CROP_HALF_PX
        y1 = y0 + 2 * CROP_HALF_PX
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(img.width, x1), min(img.height, y1)
        if x1c - x0c < 50 or y1c - y0c < 50:
            return None
        crop = img.crop((x0c, y0c, x1c, y1c))
        # Thicken thin black strokes (MinFilter expands dark pixels into
        # surrounding white).
        crop = crop.filter(ImageFilter.MinFilter(3))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    crops = [crop_for(c["cx"], c["cy"]) for c in callouts]
    skipped = sum(1 for c in crops if c is None)
    log.info("cropped %d / %d (skipped %d off-page)",
             len(crops) - skipped, len(crops), skipped)

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
                                "filter": "PIL ImageFilter.MinFilter(3)",
                                "prompt": PROMPT,
                                "results": results}, indent=2))
    log.info("wrote %s", OUT)

    # Compare against the v2 baseline directly
    print()
    print("=" * 80)
    print("PAGE 8 OCR VALID RATE")
    print("=" * 80)
    prior_valid = sum(1 for r in prior["ocr_results"] if r.get("code"))
    new_valid = sum(1 for r in results if r.get("code"))
    print(f"  v2 baseline (per-crop, no dilate, terse prompt):  "
          f"{prior_valid}/{len(callouts)} ({prior_valid/len(callouts)*100:.1f}%)")
    print(f"  this run  (raster + MinFilter(3) + soft prompt):  "
          f"{new_valid}/{len(callouts)} ({new_valid/len(callouts)*100:.1f}%)")
    print(f"  delta:                                            "
          f"{new_valid - prior_valid:+d} ({(new_valid - prior_valid)/len(callouts)*100:+.1f}pp)")

    flipped_to_valid = sum(
        1 for prev, new in zip(prior["ocr_results"], results)
        if not prev.get("code") and new.get("code")
    )
    flipped_to_invalid = sum(
        1 for prev, new in zip(prior["ocr_results"], results)
        if prev.get("code") and not new.get("code")
    )
    print(f"\nPer-callout outcome:")
    print(f"  invalid → valid (gained):  {flipped_to_valid}")
    print(f"  valid → invalid (lost):    {flipped_to_invalid}")

    prior_placed = Counter(r["code"] for r in prior["ocr_results"]
                           if r.get("code") and not r.get("in_legend"))
    new_placed = Counter(r["code"] for r in results
                         if r.get("code") and not r.get("in_legend"))
    all_codes = sorted(set(prior_placed) | set(new_placed))
    print(f"\nPlaced-callout counts per code (page 8):")
    print(f"  {'CODE':6}  {'v2':>4}  {'new':>4}  Δ")
    for code in all_codes:
        p = prior_placed.get(code, 0)
        n = new_placed.get(code, 0)
        d = n - p
        marker = " ✓" if d > 0 else (" ✗" if d < 0 else "")
        print(f"  {code:6}  {p:>4}  {n:>4}  {d:+d}{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

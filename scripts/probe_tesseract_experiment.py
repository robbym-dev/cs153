"""One-off: run Tesseract OCR on the same 92 page-8 callout crops.

Same crop pipeline as the prior dilate experiments (full-page raster at
600 DPI + MinFilter(3) dilation). Only the OCR engine differs — uses
local Tesseract via pytesseract instead of the Claude vision API.

Tesseract config:
  --psm 7   single text line
  --oem 1   LSTM-only engine
  tessedit_char_whitelist  uppercase letters + digits
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PAGE = 8
PRIOR_V2 = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_tesseract.json"
)

DPI = 600
CROP_HALF_PT = 60
CROP_HALF_PX = int(CROP_HALF_PT * DPI / 72)
TESS_CONFIG = (
    "--psm 7 --oem 1 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("tesseract")


def clean_code(raw: str) -> str | None:
    """Same logic as the LLM experiments — pull the last code-shaped token."""
    s = raw.strip().upper().strip("`'\".,:;")
    for token in reversed(re.split(r"[\s/,]+", s)):
        token = token.strip(".,;:()[]")
        m = CODE_RE.match(token)
        if m:
            return m.group(1)
    return None


def main() -> int:
    if not PRIOR_V2.exists():
        print(f"error: missing {PRIOR_V2}", file=sys.stderr)
        return 1

    callouts = json.loads(PRIOR_V2.read_text())["ocr_results"]
    log.info("loaded %d callouts from prior v2 cache", len(callouts))

    log.info("rendering full page at %d DPI...", DPI)
    doc = fitz.open(PDF)
    page = doc.load_page(PAGE - 1)
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    log.info("  rendered %d x %d px", pix.width, pix.height)

    results: list[dict] = []
    for i, c in enumerate(callouts):
        cx, cy = c["cx"], c["cy"]
        x0 = int(cx * scale) - CROP_HALF_PX
        y0 = int(cy * scale) - CROP_HALF_PX
        x1 = x0 + 2 * CROP_HALF_PX
        y1 = y0 + 2 * CROP_HALF_PX
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(img.width, x1), min(img.height, y1)
        if x1c - x0c < 50 or y1c - y0c < 50:
            results.append({"cx": cx, "cy": cy, "raw": "", "code": None,
                            "error": "off-page", "in_legend": c["in_legend"]})
            continue
        crop = img.crop((x0c, y0c, x1c, y1c))
        crop = crop.filter(ImageFilter.MinFilter(3))
        try:
            raw = pytesseract.image_to_string(crop, config=TESS_CONFIG)
        except Exception as exc:  # noqa: BLE001
            results.append({"cx": cx, "cy": cy, "raw": "", "code": None,
                            "error": str(exc), "in_legend": c["in_legend"]})
            continue
        results.append({
            "cx": cx, "cy": cy, "raw": raw.strip(),
            "code": clean_code(raw),
            "error": None, "in_legend": c["in_legend"],
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(callouts):
            log.info("  Tesseract %d/%d", i + 1, len(callouts))

    OUT.write_text(json.dumps({
        "page": PAGE, "dpi": DPI, "crop_half_pt": CROP_HALF_PT,
        "ocr_engine": "tesseract 5.5.0 (LSTM)",
        "tess_config": TESS_CONFIG,
        "filter": "PIL ImageFilter.MinFilter(3)",
        "results": results,
    }, indent=2))
    log.info("wrote %s", OUT)

    # Compare against the 38% LLM baseline (v2)
    print()
    print("=" * 90)
    print("PAGE 8 — TESSERACT vs LLM BASELINE")
    print("=" * 90)
    baseline = json.loads(PRIOR_V2.read_text())["ocr_results"]
    base_valid = sum(1 for r in baseline if r.get("code"))
    new_valid = sum(1 for r in results if r.get("code"))
    n = len(callouts)
    print(f"  v2 baseline (Opus 4.7, terse prompt, no dilate)  {base_valid:>4}/{n:<4}  "
          f"{base_valid/n*100:>6.1f}%")
    print(f"  Tesseract LSTM (dilate + char whitelist)         {new_valid:>4}/{n:<4}  "
          f"{new_valid/n*100:>6.1f}%")

    # Per-callout outcome
    gained = lost = same_agree = same_disagree = 0
    flips: list[tuple[str, str]] = []
    for prev, new in zip(baseline, results):
        pc, nc = prev.get("code"), new.get("code")
        if pc and nc:
            if pc == nc:
                same_agree += 1
            else:
                same_disagree += 1
                flips.append((pc, nc))
        elif nc:
            gained += 1
        elif pc:
            lost += 1
    print(f"\nPer-callout outcome:")
    print(f"  both valid + agree:        {same_agree}")
    print(f"  both valid + disagree:     {same_disagree}")
    print(f"  invalid → valid (gained):  {gained}")
    print(f"  valid → invalid (lost):    {lost}")
    if flips:
        print(f"  Disagreements (LLM → Tesseract):")
        for prev, new in flips[:20]:
            print(f"    {prev} → {new}")

    # Per-code
    base_pl = Counter(r["code"] for r in baseline
                      if r.get("code") and not r.get("in_legend"))
    new_pl = Counter(r["code"] for r in results
                     if r.get("code") and not r.get("in_legend"))
    print(f"\nPlaced-callout counts (page 8 only):")
    print(f"  {'CODE':6}  {'LLM v2':>7}  {'Tesseract':>10}  Δ")
    for code in sorted(set(base_pl) | set(new_pl)):
        b, n2 = base_pl.get(code, 0), new_pl.get(code, 0)
        d = n2 - b
        marker = " ✓" if d > 0 else (" ✗" if d < 0 else "")
        print(f"  {code:6}  {b:>7}  {n2:>10}  {d:+d}{marker}")

    # Top raw outputs
    raws = Counter(r["raw"] for r in results)
    print(f"\nTop 15 Tesseract raw outputs:")
    for raw, cnt in raws.most_common(15):
        short = raw[:60].replace("\n", " | ")
        print(f"  {cnt:3} × {short!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

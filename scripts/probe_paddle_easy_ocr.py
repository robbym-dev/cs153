"""Quick experiment: PaddleOCR + EasyOCR vs Claude LLM and Tesseract.

Same crop pipeline as the prior dilate experiments (page-8 callouts,
full-page raster at 600 DPI, 120x120pt crops, PIL ImageFilter.MinFilter(3)).
Run both engines, filter with CODE_RE, compare against the cached LLM
v2 baseline (38.0%) and Tesseract (2.2%).
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PRIOR_OCR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_paddle_easy.json"
)

DPI = 600
CROP_HALF_PT = 60
CROP_HALF_PX = int(CROP_HALF_PT * DPI / 72)
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")
log = logging.getLogger("paddle_easy")


def clean_code(raw: str) -> str | None:
    s = (raw or "").strip().upper().strip("`'\".,:;")
    for tok in reversed(re.split(r"[\s/,;|]+", s)):
        tok = tok.strip(".,;:()[]")
        m = CODE_RE.match(tok)
        if m:
            return m.group(1)
    return None


def render_crops():
    """Render page once, build the 92 dilated PIL crops in memory."""
    callouts = json.loads(PRIOR_OCR.read_text())["ocr_results"]
    doc = fitz.open(PDF)
    page = doc.load_page(7)
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    crops = []
    for c in callouts:
        x0 = int(c["cx"] * scale) - CROP_HALF_PX
        y0 = int(c["cy"] * scale) - CROP_HALF_PX
        x1 = x0 + 2 * CROP_HALF_PX
        y1 = y0 + 2 * CROP_HALF_PX
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(img.width, x1), min(img.height, y1)
        if x1c - x0c < 50 or y1c - y0c < 50:
            crops.append(None)
            continue
        cr = img.crop((x0c, y0c, x1c, y1c)).filter(ImageFilter.MinFilter(3))
        crops.append(cr)
    return callouts, crops


def run_easyocr(crops):
    import easyocr
    log.info("loading EasyOCR (CPU, English)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = []
    for i, crop in enumerate(crops):
        if crop is None:
            results.append({"raw": "", "code": None, "error": "off-page"})
            continue
        arr = np.array(crop)
        try:
            out = reader.readtext(arr, detail=1, paragraph=False, allowlist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ))
        except Exception as exc:  # noqa: BLE001
            results.append({"raw": "", "code": None, "error": str(exc)})
            continue
        # Concat all detected text fragments
        raw = " ".join(str(t[1]) for t in out)
        results.append({"raw": raw, "code": clean_code(raw)})
        if (i + 1) % 20 == 0 or (i + 1) == len(crops):
            log.info("  EasyOCR %d/%d", i + 1, len(crops))
    return results


def run_paddleocr(crops):
    from paddleocr import PaddleOCR
    log.info("loading PaddleOCR (CPU, English)...")
    # PaddleOCR 3.x: use predict(); legacy ocr() also works
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    results = []
    for i, crop in enumerate(crops):
        if crop is None:
            results.append({"raw": "", "code": None, "error": "off-page"})
            continue
        arr = np.array(crop)
        try:
            out = ocr.predict(arr)
        except Exception as exc:  # noqa: BLE001
            results.append({"raw": "", "code": None, "error": str(exc)})
            continue
        # out is a list of dicts in v3.x with 'rec_texts' key
        raw_parts: list[str] = []
        for page_res in out:
            texts = page_res.get("rec_texts", []) if hasattr(page_res, "get") else []
            if not texts and isinstance(page_res, dict):
                texts = page_res.get("rec_texts", [])
            raw_parts.extend(str(t) for t in texts)
        raw = " ".join(raw_parts)
        results.append({"raw": raw, "code": clean_code(raw)})
        if (i + 1) % 20 == 0 or (i + 1) == len(crops):
            log.info("  PaddleOCR %d/%d", i + 1, len(crops))
    return results


def summarize(label: str, results, callouts, llm_baseline):
    n = len(results)
    valid = sum(1 for r in results if r.get("code"))
    log.info(f"\n=== {label} ===")
    log.info(f"  valid codes: {valid}/{n}  ({valid/n*100:.1f}%)")
    # Agreement with the 35 LLM-labeled
    llm_labeled = [(c, r) for c, r in zip(callouts, results) if c.get("code")]
    agree = sum(1 for c, r in llm_labeled if r.get("code") == c["code"])
    log.info(f"  agreement with LLM (on the 35 LLM-labeled): {agree}/{len(llm_labeled)}")
    # Per-code counts (placed only)
    counts = Counter()
    for c, r in zip(callouts, results):
        if c.get("in_legend") or not r.get("code"):
            continue
        counts[r["code"]] += 1
    log.info(f"  distinct codes: {len(counts)}")
    log.info(f"  per-code (top 15):")
    for code, n_ in counts.most_common(15):
        log.info(f"    {code:>8}  {n_:>3}")
    # Sample raw outputs
    raws = Counter(r["raw"] for r in results if r["raw"])
    log.info(f"  top 5 raw outputs:")
    for raw, c_ in raws.most_common(5):
        snip = raw[:60].replace("\n", " ")
        log.info(f"    {c_:>3} × {snip!r}")


def main() -> int:
    callouts, crops = render_crops()
    n = len(callouts)
    llm_valid = sum(1 for c in callouts if c.get("code"))
    log.info(f"Loaded {n} callouts; {llm_valid} have valid LLM codes ({llm_valid/n*100:.1f}%)")
    log.info(f"Rendered {sum(1 for c in crops if c is not None)} crops (120x120pt @ {DPI} DPI, MinFilter(3))")

    paddle_results = run_paddleocr(crops)
    easy_results = run_easyocr(crops)

    summarize("PaddleOCR", paddle_results, callouts, llm_valid)
    summarize("EasyOCR", easy_results, callouts, llm_valid)

    log.info("\n" + "=" * 70)
    log.info("HEADLINE — valid OCR rate on page 8 (92 callouts)")
    log.info("=" * 70)
    log.info(f"  LLM v2 baseline (Opus 4.7, terse prompt):  35/92  (38.0%)")
    log.info(f"  Tesseract LSTM:                             2/92  ( 2.2%)")
    p_valid = sum(1 for r in paddle_results if r.get("code"))
    e_valid = sum(1 for r in easy_results if r.get("code"))
    log.info(f"  PaddleOCR:                                 {p_valid:>2}/92  ({p_valid/n*100:>4.1f}%)")
    log.info(f"  EasyOCR:                                   {e_valid:>2}/92  ({e_valid/n*100:>4.1f}%)")

    OUT.write_text(json.dumps({
        "page": 8, "dpi": DPI, "crop_half_pt": CROP_HALF_PT,
        "paddle": paddle_results, "easy": easy_results,
        "llm_baseline": [{"code": c.get("code"), "in_legend": c.get("in_legend")}
                         for c in callouts],
    }, indent=2))
    log.info(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

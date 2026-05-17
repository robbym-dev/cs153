"""One-off: de-anchor the OCR prompt by adding diverse code examples.

Same setup as probe_dilate_experiment.py (full-page raster + MinFilter(3))
but the prompt now lists four diverse example codes instead of two — to
reduce the model's bias toward "WS5" when uncertain.
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
PRIOR_V2 = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
PRIOR_DILATE = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_dilate.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_dilate_v2.json"
)
DPI = 600
CROP_HALF_PT = 60
CROP_HALF_PX = int(CROP_HALF_PT * DPI / 72)
PARALLEL = 10
MODEL = "claude-opus-4-7"
PROMPT = (
    "This circle contains a construction keynote code made of thin outlined "
    "letterforms. The text may appear very faint. Report your best guess at "
    "the 2-5 character code (one or more letters followed by digits, like "
    "WS5, R2, F03, or E12), even if you're not confident. If you can make "
    "out any partial letters or digits, report them. Return ONLY the code, "
    "nothing else."
)
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dilate_v2")


def clean_code(raw: str) -> str | None:
    s = raw.strip().upper().strip("`'\".,:;")
    for token in reversed(re.split(r"[\s/,]+", s)):
        token = token.strip(".,;:()[]")
        m = CODE_RE.match(token)
        if m:
            return m.group(1)
    return None


def main() -> int:
    if not PRIOR_V2.exists():
        print(f"error: missing {PRIOR_V2}", file=sys.stderr); return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr); return 1

    callouts = json.loads(PRIOR_V2.read_text())["ocr_results"]
    log.info("loaded %d callouts from prior v2 cache", len(callouts))

    log.info("rendering full page at %d DPI...", DPI)
    doc = fitz.open(PDF)
    page = doc.load_page(PAGE - 1)
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

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
        crop = crop.filter(ImageFilter.MinFilter(3))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    crops = [crop_for(c["cx"], c["cy"]) for c in callouts]
    log.info("cropped %d / %d", sum(1 for c in crops if c is not None), len(crops))

    client = Anthropic()

    def worker(idx: int):
        c = callouts[idx]
        if crops[idx] is None:
            return idx, {"raw": "", "code": None, "error": "off-page",
                         "cx": c["cx"], "cy": c["cy"], "in_legend": c["in_legend"]}
        b64 = base64.standard_b64encode(crops[idx]).decode("ascii")
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=50,
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
                                "filter": "PIL ImageFilter.MinFilter(3)",
                                "prompt": PROMPT,
                                "results": results}, indent=2))
    log.info("wrote %s", OUT)

    # Three-way compare
    print()
    print("=" * 90)
    print("PAGE 8 — THREE-WAY OCR COMPARISON")
    print("=" * 90)

    baseline = json.loads(PRIOR_V2.read_text())["ocr_results"]
    biased = json.loads(PRIOR_DILATE.read_text())["results"] if PRIOR_DILATE.exists() else []

    def valid_count(rs):
        return sum(1 for r in rs if r.get("code"))

    n = len(callouts)
    print(f"{'approach':50}  {'valid':>10}  {'rate':>7}")
    print(f"  {'v2 baseline (no dilate, terse prompt)':50}  "
          f"{valid_count(baseline):>4}/{n:<4}  "
          f"{valid_count(baseline)/n*100:>6.1f}%")
    if biased:
        print(f"  {'dilate + WS5/R2-anchored prompt (prior run)':50}  "
              f"{valid_count(biased):>4}/{n:<4}  "
              f"{valid_count(biased)/n*100:>6.1f}%")
    print(f"  {'dilate + WS5/R2/F03/E12 de-anchored (this run)':50}  "
          f"{valid_count(results):>4}/{n:<4}  "
          f"{valid_count(results)/n*100:>6.1f}%")

    def placed_counter(rs):
        return Counter(r["code"] for r in rs
                       if r.get("code") and not r.get("in_legend"))

    base_pl = placed_counter(baseline)
    bias_pl = placed_counter(biased) if biased else Counter()
    new_pl = placed_counter(results)

    all_codes = sorted(set(base_pl) | set(bias_pl) | set(new_pl))
    print(f"\nPlaced-callout counts (page 8 only):")
    print(f"  {'CODE':6}  {'v2':>4}  {'biased':>6}  {'de-anchored':>11}  Reference")
    reference = {
        "WS5": "23 EA", "WS7": "7 EA", "R1": "2 EA", "R2": "8 EA",
        "R3": "2 EA", "R4": "2 EA", "R5": "2 EA",
        "WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
    }
    for code in all_codes:
        b = base_pl.get(code, 0); x = bias_pl.get(code, 0); n_ = new_pl.get(code, 0)
        ty = reference.get(code, "—")
        print(f"  {code:6}  {b:>4}  {x:>6}  {n_:>11}  {ty}")

    # WS5 specifically
    ws5_base = base_pl.get("WS5", 0)
    ws5_bias = bias_pl.get("WS5", 0)
    ws5_new = new_pl.get("WS5", 0)
    print(f"\nWS5 trajectory: v2={ws5_base}, biased={ws5_bias}, de-anchored={ws5_new}  "
          f"(target 13-20, Reference full-bldg total 23)")

    # Top unrecognized this run
    bad = Counter(r["raw"] for r in results if not r.get("code"))
    if bad:
        print(f"\nTop unrecognized OCR outputs from this run (top 10 of {len(bad)}):")
        for raw, cnt in bad.most_common(10):
            short = raw[:80].replace("\n", " ")
            print(f"  {cnt:3} × {short!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

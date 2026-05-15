"""One-off: pure abstract pattern prompt with no example codes.

Same crop pipeline (full-page raster + MinFilter(3) dilation) as the prior
two dilate experiments. Only difference: the prompt describes the code
pattern abstractly without naming any concrete codes — testing whether
the example-code anchor really was the bias source.
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
PRIOR_BIASED = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_dilate.json"
)
PRIOR_DEANCHORED = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_dilate_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_dilate_v3.json"
)
DPI = 600
CROP_HALF_PT = 60
CROP_HALF_PX = int(CROP_HALF_PT * DPI / 72)
PARALLEL = 10
MODEL = "claude-opus-4-7"
PROMPT = (
    "This circle contains a construction keynote code: one to three "
    "uppercase letters followed by one to three digits. Report the code "
    "exactly as you see it. Return ONLY the code, nothing else."
)
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dilate_v3")


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

    # Four-way comparison
    print()
    print("=" * 100)
    print("PAGE 8 — FOUR-WAY OCR COMPARISON")
    print("=" * 100)

    runs = [
        ("v2 baseline (no dilate, terse prompt)", PRIOR_V2, "ocr_results"),
        ("dilate + WS5/R2 anchored", PRIOR_BIASED, "results"),
        ("dilate + WS5/R2/F03/E12 de-anchored", PRIOR_DEANCHORED, "results"),
    ]

    n = len(callouts)
    print(f"{'approach':50}  {'valid':>10}  {'rate':>7}")
    for label, path, key in runs:
        if not path.exists():
            continue
        rs = json.loads(path.read_text())[key]
        v = sum(1 for r in rs if r.get("code"))
        print(f"  {label:50}  {v:>4}/{n:<4}  {v/n*100:>6.1f}%")
    new_v = sum(1 for r in results if r.get("code"))
    print(f"  {'dilate + abstract pattern (this run)':50}  "
          f"{new_v:>4}/{n:<4}  {new_v/n*100:>6.1f}%")

    def placed_counter(rs):
        return Counter(r["code"] for r in rs
                       if r.get("code") and not r.get("in_legend"))

    counters = []
    counter_labels = []
    for label, path, key in runs:
        if not path.exists():
            continue
        rs = json.loads(path.read_text())[key]
        counters.append(placed_counter(rs))
        counter_labels.append(label.split(" (")[0])
    counters.append(placed_counter(results))
    counter_labels.append("abstract")

    all_codes = sorted(set().union(*counters))
    tyler = {
        "WS5": "23 EA", "WS7": "7 EA", "R1": "2 EA", "R2": "8 EA",
        "R3": "2 EA", "R4": "2 EA", "R5": "2 EA",
        "WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
        "WS19": "131 LF",
    }

    print(f"\nPlaced-callout counts (page 8 only):")
    header = "  " + f"{'CODE':6}  " + "  ".join(f"{lab[:9]:>9}" for lab in counter_labels) + "  Tyler"
    print(header)
    for code in all_codes:
        row = f"  {code:6}  " + "  ".join(f"{c.get(code, 0):>9}" for c in counters)
        ty = tyler.get(code, "")
        print(f"{row}  {ty}")

    # Phantom-code summary: codes from prior runs' bias examples that don't
    # plausibly belong on this drawing
    phantoms = ("F02", "F03", "E12", "B3", "S2", "C1", "R10")
    print(f"\nPhantom example-list codes:")
    for code in phantoms:
        cells = [c.get(code, 0) for c in counters]
        if any(cells):
            print(f"  {code:6}  " + "  ".join(f"{v:>9}" for v in cells))

    # Top unrecognized
    bad = Counter(r["raw"] for r in results if not r.get("code"))
    if bad:
        print(f"\nTop unrecognized OCR outputs (top 10 of {len(bad)}):")
        for raw, cnt in bad.most_common(10):
            short = raw[:80].replace("\n", " ")
            print(f"  {cnt:3} × {short!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

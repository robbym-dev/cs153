"""Auto-label digit crops via sequential keynote-legend matching.

Idea: the LLM keynote extraction returns a list of codes in legend order
(WS1, WS2, ..., WS19, R1, ..., R5). The legend-column detector finds the
bullet circles for those codes. Matching them sequentially by y-position
gives a position-based label for each circle — no per-circle OCR needed.
Then take the trailing N paths inside each circle as digit glyphs and
save them as labeled training data.

Validation: each position match is sanity-checked two ways before being
accepted as training data:
  (a) the path count inside the circle must equal the character count of
      the assumed code (e.g. "WS10" -> 4 paths)
  (b) if the circle also has an LLM-OCR read from the v2 baseline, that
      read must agree with the assumed code

Either check failing rejects the match. This prevents propagating bad
labels when the legend's actual contents don't match the sequential
assumption.

Uses cached page8_full.json (LLM keynote extraction) and
page8_ocr_v2.json (per-circle LLM-OCR and in_legend flags). Zero new
API calls.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import fitz
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
TAKEOFF_DIR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original" / "takeoff"
)
OCR_DIR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original" / "vector_callouts"
)
OUT_DIR = PROJECT_ROOT / "data" / "digit_crops_auto"
PRIOR_DIR = PROJECT_ROOT / "data" / "digit_crops"

PAGE = 8
DPI = 600
CIRCLE_DIAMETER = 27.0
INSIDE_MARGIN = 1.0
MIN_PATH_SIZE = 0.2
CROP_MARGIN_PT = 1.5

CODE_RE = re.compile(r"^([A-Z]+)(\d+)$")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("auto_label")


def split_code(code: str) -> tuple[str | None, str | None]:
    m = CODE_RE.match(code)
    return (m.group(1), m.group(2)) if m else (None, None)


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


def render_path_region(page, path_rect, margin_pt: float = CROP_MARGIN_PT) -> Image.Image:
    clip = fitz.Rect(
        path_rect.x0 - margin_pt, path_rect.y0 - margin_pt,
        path_rect.x1 + margin_pt, path_rect.y1 + margin_pt,
    ) & page.rect
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main() -> int:
    keynotes_cache = TAKEOFF_DIR / f"page{PAGE}_full.json"
    ocr_cache = OCR_DIR / f"page{PAGE}_ocr_v2.json"
    if not keynotes_cache.exists() or not ocr_cache.exists():
        print("error: required caches missing", file=sys.stderr)
        return 1

    # 1. LLM keynote list, in legend order
    keynotes = json.loads(keynotes_cache.read_text())["keynotes"]
    keynote_codes = [kn["code"] for kn in keynotes]
    log.info(f"LLM keynote extraction on page {PAGE}: {len(keynote_codes)} codes")
    log.info(f"  {keynote_codes}")

    # 2. Legend circles (in_legend=True), sorted top-to-bottom
    callouts = json.loads(ocr_cache.read_text())["ocr_results"]
    legend = [c for c in callouts if c.get("in_legend")]
    legend.sort(key=lambda c: c["cy"])
    log.info(f"\nLegend circles on page {PAGE} (auto-detected, top-to-bottom): {len(legend)}")

    # 3. Match position-wise & validate
    doc = fitz.open(PDF)
    page = doc.load_page(PAGE - 1)
    all_paths = page.get_drawings()

    rows = []   # per-legend-circle audit
    crops_to_save: list[tuple[str, int, fitz.Rect, str]] = []
    for i, lc in enumerate(legend):
        if i >= len(keynote_codes):
            rows.append((i, lc, None, None, "no-code-at-position"))
            continue
        assumed = keynote_codes[i]
        prefix, digits = split_code(assumed)
        if not digits:
            rows.append((i, lc, assumed, None, "non-digit-code"))
            continue
        inside = paths_inside_circle(lc["cx"], lc["cy"], all_paths)
        by_y = sorted(inside, key=lambda p: (p["rect"].y0 + p["rect"].y1) / 2)
        path_ok = len(by_y) == len(assumed)
        llm_read = lc.get("code")
        llm_ok = llm_read is None or llm_read == assumed
        verdict = "ACCEPT" if (path_ok and llm_ok) else "REJECT"
        reasons = []
        if not path_ok:
            reasons.append(f"path-count={len(by_y)} vs len({assumed})={len(assumed)}")
        if not llm_ok:
            reasons.append(f"LLM-read={llm_read!r}")
        rows.append((i, lc, assumed, by_y, verdict + (" — " + ", ".join(reasons) if reasons else "")))
        if verdict == "ACCEPT":
            for digit_char, dp in zip(digits, by_y[-len(digits):]):
                crops_to_save.append((digit_char, i, dp["rect"], assumed))

    # 4. Save verified crops
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    manifest = []
    for digit_char, legend_idx, rect, code in crops_to_save:
        out_subdir = OUT_DIR / digit_char
        out_subdir.mkdir(exist_ok=True)
        fname = f"legend{legend_idx}_{code}_d{digit_char}.png"
        crop = render_path_region(page, rect)
        crop.save(out_subdir / fname)
        counts[digit_char] += 1
        manifest.append({
            "page": PAGE, "legend_position": legend_idx, "assumed_code": code,
            "digit_char": digit_char,
            "rel_path": str((out_subdir / fname).relative_to(PROJECT_ROOT)),
            "path_bbox": [round(rect.x0, 2), round(rect.y0, 2),
                          round(rect.x1, 2), round(rect.y1, 2)],
        })
    doc.close()

    # Audit print
    print()
    print("=" * 90)
    print(f"Per-legend-circle audit (page {PAGE})")
    print("=" * 90)
    print(f"  {'pos':>3}  {'y':>5}  {'assumed':>8}  {'paths':>5}  "
          f"{'LLM read':>10}  verdict")
    for i, lc, assumed, by_y, verdict in rows:
        n_paths = len(by_y) if by_y else 0
        llm_read = lc.get("code")
        print(f"  {i:>3}  {int(lc['cy']):>5}  {str(assumed):>8}  "
              f"{n_paths:>5}  {str(llm_read):>10}  {verdict}")

    # Class coverage comparison
    prior_manifest = json.loads((PRIOR_DIR / "manifest.json").read_text())
    prior_counts = prior_manifest["counts_per_class"]

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Legend circles found:           {len(legend)}")
    print(f"  Keynote codes in LLM list:      {len(keynote_codes)}")
    accepted = sum(1 for r in rows if r[4].startswith("ACCEPT"))
    rejected = len(rows) - accepted
    print(f"  Position matches ACCEPTED:      {accepted}/{len(rows)}")
    print(f"  Position matches REJECTED:      {rejected}")
    print(f"  Digit crops saved:              {sum(counts.values())}")
    print()
    print(f"  Class coverage:")
    print(f"  {'digit':>5}  {'prior 21-crop set':>18}  {'this legend-auto':>17}  total")
    new_classes = []
    for d in "0123456789":
        prior_n = prior_counts.get(d, 0)
        new_n = counts.get(d, 0)
        marker = ""
        if prior_n == 0 and new_n > 0:
            new_classes.append(d)
            marker = "  ← NEW"
        elif new_n > 0:
            marker = ""
        print(f"  {d:>5}  {prior_n:>18}  {new_n:>17}  {prior_n + new_n}{marker}")

    print()
    print(f"  Prior dataset had: {len(prior_counts)} classes covered "
          f"({sorted(int(k) for k, v in prior_counts.items() if v > 0)})")
    print(f"  Missing in prior:  {[d for d in '0123456789' if prior_counts.get(d, 0) == 0]}")
    if new_classes:
        print(f"  Newly covered:     {new_classes}")
    else:
        print(f"  Newly covered:     (none — sequential matching produced no usable labels)")

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "page": PAGE,
        "method": "sequential keynote-legend matching",
        "n_legend_circles": len(legend),
        "n_keynote_codes": len(keynote_codes),
        "matches_accepted": accepted, "matches_rejected": rejected,
        "counts_per_class": dict(counts),
        "audit": [
            {"position": r[0], "cy": r[1]["cy"], "assumed_code": r[2],
             "n_paths": len(r[3]) if r[3] else 0,
             "llm_read": r[1].get("code"), "verdict": r[4]}
            for r in rows
        ],
        "items": manifest,
    }, indent=2))
    print(f"\n  Manifest: {(OUT_DIR / 'manifest.json').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

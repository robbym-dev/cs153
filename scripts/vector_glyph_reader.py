"""Self-supervised vector glyph reader for callout circles.

Where the original spec proposed labeling templates by their position in a
sequential keynote legend, page 8's detected "legend" column turned out to
be a stack of repeated callouts rather than a sequential listing. The
keynote-description legend itself is rendered as text-as-paths elsewhere
on the page and isn't reliably detected as a column of bullets.

Same self-supervised principle works with a different oracle: the LLM-OCR
v2 baseline (38% rate on page 8) reads a subset of callouts confidently.
Treat those confident reads as labels, decompose each labeled circle into
character-ordered path features (paths sorted top-to-bottom — characters
are stacked vertically inside each circle on this drawing), and aggregate
features per character to build a template library.

Then for the un-labeled callouts (and for sanity-checking the labeled
ones), match each path against the templates by normalized Euclidean
distance on (n_lines, n_endpoints, bbox_height).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PAGE = 8
PRIOR_OCR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_glyph_match.json"
)

CIRCLE_DIAMETER = 27.0
INSIDE_MARGIN = 1.0
MIN_PATH_SIZE = 0.2


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathFeatures:
    bbox_w: float
    bbox_h: float
    n_lines: int
    n_endpoints: int
    length: float

    def to_vec(self) -> tuple[float, float, float]:
        # Width is essentially constant (5.6pt) across all in-circle glyph
        # paths on this page, so we drop it from the discriminative vector.
        return (float(self.n_lines), float(self.n_endpoints), self.bbox_h)


def extract_features(path) -> PathFeatures:
    pr = path["rect"]
    items = path.get("items", [])
    pts: set[tuple[float, float]] = set()
    total = 0.0
    n_lines = 0
    for it in items:
        if it[0] == "l" and len(it) >= 3:
            n_lines += 1
            p1, p2 = it[1], it[2]
            pts.add((round(p1.x, 1), round(p1.y, 1)))
            pts.add((round(p2.x, 1), round(p2.y, 1)))
            total += math.hypot(p2.x - p1.x, p2.y - p1.y)
    return PathFeatures(
        bbox_w=pr.x1 - pr.x0,
        bbox_h=pr.y1 - pr.y0,
        n_lines=n_lines,
        n_endpoints=len(pts),
        length=total,
    )


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


def paths_in_reading_order(cx: float, cy: float, all_paths) -> list:
    """Characters are stacked vertically inside each circle — sort by y."""
    inside = paths_inside_circle(cx, cy, all_paths)
    return sorted(inside, key=lambda p: (p["rect"].y0 + p["rect"].y1) / 2)


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------


@dataclass
class Template:
    character: str
    n_examples: int
    avg_lines: float
    avg_endpoints: float
    avg_h: float
    std_lines: float
    std_endpoints: float
    std_h: float


def build_templates(callouts_with_codes, all_paths) -> tuple[dict[str, Template], int, int]:
    """Each LLM-labeled callout contributes one feature vector per character.

    Skipped when len(paths_in_circle) != len(code) — that's a sanity check
    against accidentally labeling the wrong path with the wrong character.
    """
    by_char: dict[str, list[PathFeatures]] = defaultdict(list)
    used = 0
    skipped = 0
    for c in callouts_with_codes:
        code = c.get("code")
        if not code:
            continue
        paths = paths_in_reading_order(c["cx"], c["cy"], all_paths)
        if len(paths) != len(code):
            skipped += 1
            continue
        used += 1
        for ch, p in zip(code, paths):
            by_char[ch].append(extract_features(p))

    templates: dict[str, Template] = {}
    for ch, feats in by_char.items():
        vecs = [(f.n_lines, f.n_endpoints, f.bbox_h) for f in feats]
        templates[ch] = Template(
            character=ch,
            n_examples=len(feats),
            avg_lines=statistics.fmean(v[0] for v in vecs),
            avg_endpoints=statistics.fmean(v[1] for v in vecs),
            avg_h=statistics.fmean(v[2] for v in vecs),
            std_lines=statistics.pstdev([v[0] for v in vecs]) if len(vecs) > 1 else 0.0,
            std_endpoints=statistics.pstdev([v[1] for v in vecs]) if len(vecs) > 1 else 0.0,
            std_h=statistics.pstdev([v[2] for v in vecs]) if len(vecs) > 1 else 0.0,
        )
    return templates, used, skipped


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _feature_scale(templates: dict[str, Template]) -> tuple[float, float, float]:
    """Normalize each feature dimension by its range across templates so
    Euclidean distance isn't dominated by raw magnitude.
    """
    if not templates:
        return (1.0, 1.0, 1.0)
    lines = [t.avg_lines for t in templates.values()]
    endpoints = [t.avg_endpoints for t in templates.values()]
    heights = [t.avg_h for t in templates.values()]
    return (
        max(max(lines) - min(lines), 1.0),
        max(max(endpoints) - min(endpoints), 1.0),
        max(max(heights) - min(heights), 0.5),
    )


def match_path(f: PathFeatures, templates: dict[str, Template], scale) -> tuple[str | None, float, float]:
    """Return (best character, distance, margin-over-second-best)."""
    sl, se, sh = scale
    target = (f.n_lines / sl, f.n_endpoints / se, f.bbox_h / sh)
    ranked: list[tuple[float, str]] = []
    for ch, t in templates.items():
        tvec = (t.avg_lines / sl, t.avg_endpoints / se, t.avg_h / sh)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, tvec)))
        ranked.append((d, ch))
    if not ranked:
        return None, float("inf"), 0.0
    ranked.sort()
    best_d, best_ch = ranked[0]
    second_d = ranked[1][0] if len(ranked) > 1 else best_d * 2
    return best_ch, best_d, second_d - best_d


def predict_callout(c, all_paths, templates, scale) -> dict:
    paths = paths_in_reading_order(c["cx"], c["cy"], all_paths)
    code_chars: list[str] = []
    per_char: list[dict] = []
    for p in paths:
        f = extract_features(p)
        ch, d, margin = match_path(f, templates, scale)
        code_chars.append(ch or "?")
        per_char.append({
            "char": ch, "distance": round(d, 3), "margin": round(margin, 3),
            "n_lines": f.n_lines, "bbox_h": round(f.bbox_h, 2),
        })
    return {
        "cx": c["cx"], "cy": c["cy"],
        "n_paths": len(paths),
        "predicted_code": "".join(code_chars) if code_chars else None,
        "per_char": per_char,
        "llm_code": c.get("code"),
        "in_legend": c.get("in_legend"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not PRIOR_OCR.exists():
        print(f"error: missing {PRIOR_OCR}", file=sys.stderr)
        return 1
    callouts = json.loads(PRIOR_OCR.read_text())["ocr_results"]
    labeled = [c for c in callouts if c.get("code")]
    unlabeled = [c for c in callouts if not c.get("code")]
    print(f"Callouts on page {PAGE}: {len(callouts)}  "
          f"(LLM-labeled: {len(labeled)}, unlabeled: {len(unlabeled)})")

    doc = fitz.open(PDF)
    page = doc.load_page(PAGE - 1)
    all_paths = page.get_drawings()

    templates, used, skipped = build_templates(labeled, all_paths)
    print(f"\nTemplate library built from {used} labeled callouts "
          f"(skipped {skipped} where path-count ≠ char-count):\n")
    print(f"  {'char':4} {'#examples':>9} {'avg_lines':>9} "
          f"{'avg_endpts':>10} {'avg_h':>6}  ±std")
    for ch in sorted(templates):
        t = templates[ch]
        print(f"  {ch:4} {t.n_examples:>9}  "
              f"{t.avg_lines:>9.1f} {t.avg_endpoints:>10.1f} {t.avg_h:>6.2f}  "
              f"±({t.std_lines:.1f},{t.std_endpoints:.1f},{t.std_h:.2f})")

    scale = _feature_scale(templates)
    print(f"\nFeature scale (n_lines, n_endpoints, h): {tuple(round(s, 2) for s in scale)}")

    # Predict on every callout
    predictions = [predict_callout(c, all_paths, templates, scale) for c in callouts]
    OUT.write_text(json.dumps({
        "page": PAGE,
        "templates": {ch: t.__dict__ for ch, t in templates.items()},
        "predictions": predictions,
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")

    # Verify: does predictor agree with LLM on LLM-labeled callouts?
    agree = disagree = 0
    disagreements: list[tuple[str, str]] = []
    for p in predictions:
        if p["llm_code"] is None:
            continue
        if p["predicted_code"] == p["llm_code"]:
            agree += 1
        else:
            disagree += 1
            disagreements.append((p["llm_code"], p["predicted_code"]))
    print(f"\n--- Self-validation on LLM-labeled callouts ---")
    print(f"  agree:    {agree}/{agree + disagree}  "
          f"({agree / (agree + disagree) * 100:.1f}%)" if (agree + disagree) else "n/a")
    if disagreements:
        c_dis = Counter(disagreements)
        print(f"  Disagreements (top 10):")
        for (llm, pred), n in c_dis.most_common(10):
            print(f"    {n:3} × LLM={llm!r:8} predicted={pred!r}")

    # New predictions on previously-unlabeled callouts
    new_preds = [p for p in predictions if p["llm_code"] is None]
    new_codes = Counter(p["predicted_code"] for p in new_preds if p["predicted_code"])
    print(f"\n--- Predictions on the {len(new_preds)} LLM-unlabeled callouts ---")
    print(f"  Distinct codes predicted: {len(new_codes)}")
    print(f"  Top codes (top 15):")
    for code, n in new_codes.most_common(15):
        print(f"    {n:3} × {code}")

    # Combine LLM (high-confidence) + glyph predictions (for unlabeled) for
    # a final placed-callout count
    final: Counter = Counter()
    for p in predictions:
        if p.get("in_legend"):
            continue  # skip legend bullets per the pipeline convention
        code = p["llm_code"] or p["predicted_code"]
        if code:
            final[code] += 1
    print(f"\n--- Combined placed counts (LLM + glyph-match fallback) ---")
    # Reference quantities — Tyler/Reference spreadsheet (full building)
    tyler = {
        "WS5": "23 EA", "WS7": "7 EA", "R1": "2 EA", "R2": "8 EA",
        "R3": "2 EA", "R4": "2 EA", "R5": "2 EA",
        "WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
        "WS19": "131 LF",
    }
    print(f"  {'CODE':6}  {'count':>5}  Reference (full bldg)")
    for code in sorted(final):
        ref = tyler.get(code, "")
        print(f"  {code:6}  {final[code]:>5}  {ref}")

    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

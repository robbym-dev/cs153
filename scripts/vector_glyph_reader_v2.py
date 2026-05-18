"""Self-supervised vector glyph reader v2 — cross-page clustering edition.

Empirical reality on this PDF: the sequential keynote-description legend
isn't a column of bullets. Page 8's auto-detected legend column is a stack
of repeated callouts; page 10's 19-circle column has only 13 unique digit
shapes across its 19 entries — clearly not WS1..WS19 in order.

What IS true: a callout's internal-path signature (per-character
endpoint patterns) uniquely identifies which code it carries. Two
callouts of the same code have byte-identical glyph paths. So:

  1. Across all four elevation pages (7, 8, 9, 10) collect every
     callout's internal-path "fingerprint" — a hashable tuple of each
     character's normalized endpoint coordinates.
  2. Cluster callouts by fingerprint equality. Each cluster is one
     unique code.
  3. Label each cluster by majority LLM-OCR vote across its members —
     consensus over many examples beats any single LLM read.
  4. For labeled clusters, decompose paths into character-labeled
     templates. Aggregate features per character.
  5. For unlabeled page-8 callouts, match each path against templates
     by normalized Euclidean distance on (n_lines, n_endpoints,
     bbox_height).
  6. Report agreement with the LLM 38% baseline.
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
OCR_CACHE_DIR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original" / "vector_callouts"
)
OCR_CACHES = {
    7: OCR_CACHE_DIR / "page7_ocr_v2.json",
    8: OCR_CACHE_DIR / "page8_ocr_v2.json",
    9: OCR_CACHE_DIR / "page9_ocr_v2.json",
    10: OCR_CACHE_DIR / "page10_ocr_v2.json",
}
OUT = OCR_CACHE_DIR / "page8_glyph_match_v2.json"

CIRCLE_DIAMETER = 27.0
INSIDE_MARGIN = 1.0
MIN_PATH_SIZE = 0.2


# ---------------------------------------------------------------------------
# Path features and fingerprints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathFeatures:
    bbox_w: float
    bbox_h: float
    n_lines: int
    n_endpoints: int


def extract_features(path) -> PathFeatures:
    pr = path["rect"]
    items = path.get("items", [])
    pts: set[tuple[float, float]] = set()
    n_lines = 0
    for it in items:
        if it[0] == "l" and len(it) >= 3:
            n_lines += 1
            for pt in (it[1], it[2]):
                pts.add((round(pt.x, 1), round(pt.y, 1)))
    return PathFeatures(
        bbox_w=pr.x1 - pr.x0,
        bbox_h=pr.y1 - pr.y0,
        n_lines=n_lines,
        n_endpoints=len(pts),
    )


def path_fingerprint(path) -> tuple:
    """Endpoint set normalized to the path's own bbox top-left — invariant
    under translation, sensitive to the actual shape outline."""
    pr = path["rect"]
    pts = set()
    for it in path.get("items", []):
        if it[0] == "l":
            for pt in (it[1], it[2]):
                pts.add((round(pt.x - pr.x0, 1), round(pt.y - pr.y0, 1)))
    return tuple(sorted(pts))


def paths_in_circle(cx: float, cy: float, all_paths) -> list:
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
    return sorted(
        paths_in_circle(cx, cy, all_paths),
        key=lambda p: (p["rect"].y0 + p["rect"].y1) / 2,
    )


def callout_fingerprint(cx: float, cy: float, all_paths) -> tuple:
    """Tuple of per-character fingerprints, in vertical reading order.
    Two callouts of the same code have equal fingerprints."""
    return tuple(path_fingerprint(p) for p in paths_in_reading_order(cx, cy, all_paths))


# ---------------------------------------------------------------------------
# Cross-page clustering
# ---------------------------------------------------------------------------


def gather_callouts():
    """Across pages 7-10, return list of dicts:
       {page, cx, cy, in_legend, llm_code, fingerprint, paths}"""
    doc = fitz.open(PDF)
    rows = []
    for page_no, ocr_cache in OCR_CACHES.items():
        if not ocr_cache.exists():
            print(f"  warning: missing {ocr_cache.name}", file=sys.stderr)
            continue
        callouts = json.loads(ocr_cache.read_text())["ocr_results"]
        page = doc.load_page(page_no - 1)
        all_paths = page.get_drawings()
        for c in callouts:
            paths_ord = paths_in_reading_order(c["cx"], c["cy"], all_paths)
            fp = tuple(path_fingerprint(p) for p in paths_ord)
            rows.append({
                "page": page_no,
                "cx": c["cx"],
                "cy": c["cy"],
                "in_legend": c.get("in_legend", False),
                "llm_code": c.get("code"),
                "fingerprint": fp,
                "n_paths": len(paths_ord),
                "_paths": paths_ord,  # underscore-prefixed: not JSON-serializable
            })
    doc.close()
    return rows


# ---------------------------------------------------------------------------
# Label clusters by LLM vote
# ---------------------------------------------------------------------------


def label_clusters(rows):
    """Group rows by fingerprint, vote majority LLM read within each cluster.
    Returns: {fingerprint -> {"label": str|None, "size": int,
                              "llm_reads": Counter, "members": [row_indices]}}"""
    by_fp = defaultdict(list)
    for i, r in enumerate(rows):
        by_fp[r["fingerprint"]].append(i)

    clusters = {}
    for fp, member_indices in by_fp.items():
        if not fp:
            continue  # empty (no paths inside — skip)
        llm_reads = Counter(rows[i]["llm_code"] for i in member_indices
                            if rows[i]["llm_code"])
        # Majority label only if (a) at least one LLM read AND (b) the top
        # read is supported by >= 2 callouts OR is the only read in cluster
        label = None
        if llm_reads:
            top_code, top_n = llm_reads.most_common(1)[0]
            if top_n >= 2 or len(llm_reads) == 1:
                label = top_code
        clusters[fp] = {
            "label": label,
            "size": len(member_indices),
            "members": member_indices,
            "llm_reads": dict(llm_reads),
            "n_paths": len(fp),
        }
    return clusters


# ---------------------------------------------------------------------------
# Build per-character templates from labeled clusters
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


def build_templates(rows, clusters):
    """For each labeled cluster where path-count matches label-length, take
    each callout in the cluster, decompose into character/path pairs, and
    aggregate features per character."""
    by_char: dict[str, list[PathFeatures]] = defaultdict(list)
    clusters_used = 0
    callouts_used = 0
    for fp, info in clusters.items():
        label = info["label"]
        if not label:
            continue
        if len(label) != info["n_paths"]:
            continue  # path-count must match character count
        clusters_used += 1
        for idx in info["members"]:
            r = rows[idx]
            paths = r["_paths"]
            callouts_used += 1
            for ch, p in zip(label, paths):
                by_char[ch].append(extract_features(p))

    templates: dict[str, Template] = {}
    for ch, feats in by_char.items():
        ls = [f.n_lines for f in feats]
        es = [f.n_endpoints for f in feats]
        hs = [f.bbox_h for f in feats]
        templates[ch] = Template(
            character=ch, n_examples=len(feats),
            avg_lines=statistics.fmean(ls),
            avg_endpoints=statistics.fmean(es),
            avg_h=statistics.fmean(hs),
            std_lines=statistics.pstdev(ls) if len(ls) > 1 else 0.0,
            std_endpoints=statistics.pstdev(es) if len(es) > 1 else 0.0,
            std_h=statistics.pstdev(hs) if len(hs) > 1 else 0.0,
        )
    return templates, clusters_used, callouts_used


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def feature_scale(templates):
    if not templates:
        return (1.0, 1.0, 1.0)
    ls = [t.avg_lines for t in templates.values()]
    es = [t.avg_endpoints for t in templates.values()]
    hs = [t.avg_h for t in templates.values()]
    return (
        max(max(ls) - min(ls), 1.0),
        max(max(es) - min(es), 1.0),
        max(max(hs) - min(hs), 0.5),
    )


def match_path(f, templates, scale, candidates=None):
    """Find nearest template by normalized Euclidean distance, optionally
    restricting to a candidate-character set (for position priors)."""
    sl, se, sh = scale
    target = (f.n_lines / sl, f.n_endpoints / se, f.bbox_h / sh)
    pool = templates if candidates is None else {
        ch: t for ch, t in templates.items() if ch in candidates
    }
    if not pool:
        return None, float("inf")
    best_ch, best_d = None, float("inf")
    for ch, t in pool.items():
        tvec = (t.avg_lines / sl, t.avg_endpoints / se, t.avg_h / sh)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, tvec)))
        if d < best_d:
            best_ch, best_d = ch, d
    return best_ch, best_d


# Position priors: by total path count, restrict each position's alphabet.
# Codes on this drawing are R1-R5 (2-char) and WS1-WS19 (3- or 4-char).
DIGITS = set("0123456789")
POSITION_CANDIDATES = {
    2: [{"R"}, DIGITS],                     # R + digit
    3: [{"W"}, {"S"}, DIGITS],              # W, S, digit
    4: [{"W"}, {"S"}, {"1"}, DIGITS],       # W, S, "1" (tens), units digit
}


def predict_with_priors(row, templates, scale, use_priors=True):
    paths = row["_paths"]
    n = len(paths)
    candidates_per_position = (
        POSITION_CANDIDATES.get(n, [None] * n) if use_priors else [None] * n
    )
    code_chars = []
    distances = []
    for p, cands in zip(paths, candidates_per_position):
        f = extract_features(p)
        ch, d = match_path(f, templates, scale, candidates=cands)
        code_chars.append(ch or "?")
        distances.append(round(d, 3))
    return "".join(code_chars), distances


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    rows = gather_callouts()
    pages_seen = sorted({r["page"] for r in rows})
    print(f"Loaded {len(rows)} callouts across pages {pages_seen}")
    by_page = Counter(r["page"] for r in rows)
    for p in sorted(by_page):
        labeled = sum(1 for r in rows if r["page"] == p and r["llm_code"])
        print(f"  page {p}: {by_page[p]} callouts, {labeled} LLM-labeled")

    clusters = label_clusters(rows)
    print(f"\nUnique fingerprints (≈ unique codes): {len(clusters)}")
    labeled_clusters = [c for c in clusters.values() if c["label"]]
    print(f"Labeled clusters (>=2 LLM votes or unanimous):   {len(labeled_clusters)}")
    print(f"Total callouts covered by labeled clusters:      "
          f"{sum(c['size'] for c in labeled_clusters)}")

    print(f"\nLargest clusters (top 15):")
    print(f"  {'size':>4}  {'label':>8}  {'n_paths':>7}  "
          f"LLM reads in cluster")
    for fp, info in sorted(clusters.items(), key=lambda kv: -kv[1]["size"])[:15]:
        reads = ", ".join(f"{k}×{v}" for k, v in
                          sorted(info["llm_reads"].items(), key=lambda x: -x[1]))
        print(f"  {info['size']:>4}  {info['label'] or '—':>8}  "
              f"{info['n_paths']:>7}  {reads or '(no LLM reads)'}")

    templates, n_clusters_used, n_callouts_used = build_templates(rows, clusters)
    print(f"\nTemplates built from {n_clusters_used} clusters / "
          f"{n_callouts_used} callouts:")
    print(f"  {'ch':3} {'#ex':>4} {'lines':>6} {'endpts':>6} {'h':>5}  ±std")
    for ch in sorted(templates):
        t = templates[ch]
        print(f"  {ch:3} {t.n_examples:>4}  "
              f"{t.avg_lines:>5.1f} {t.avg_endpoints:>6.1f} {t.avg_h:>4.2f}  "
              f"±({t.std_lines:.1f},{t.std_endpoints:.1f},{t.std_h:.2f})")

    scale = feature_scale(templates)

    # Predict every page-8 callout
    p8 = [r for r in rows if r["page"] == 8]
    predictions = []
    for r in p8:
        pred, dists = predict_with_priors(r, templates, scale, use_priors=True)
        predictions.append({
            "cx": r["cx"], "cy": r["cy"],
            "in_legend": r["in_legend"],
            "n_paths": r["n_paths"],
            "llm_code": r["llm_code"],
            "predicted_code": pred,
            "distances": dists,
        })

    OUT.write_text(json.dumps({
        "page": 8,
        "templates": {ch: t.__dict__ for ch, t in templates.items()},
        "predictions": predictions,
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")

    # Validation
    agree = disagree = 0
    flips = []
    for p in predictions:
        if p["llm_code"] is None:
            continue
        if p["predicted_code"] == p["llm_code"]:
            agree += 1
        else:
            disagree += 1
            flips.append((p["llm_code"], p["predicted_code"]))
    total = agree + disagree
    print(f"\n--- Page 8 self-validation against LLM ---")
    print(f"  agree: {agree}/{total}  ({agree/total*100:.1f}%)" if total
          else "  (no LLM labels)")
    if flips:
        c = Counter(flips)
        print(f"  Top disagreements:")
        for (l, pr), n in c.most_common(8):
            print(f"    {n:3} × LLM={l!r:8} predicted={pr!r}")

    # New predictions on previously-unlabeled page 8 callouts
    new_preds = [p for p in predictions if p["llm_code"] is None]
    new_codes = Counter(p["predicted_code"] for p in new_preds if p["predicted_code"])
    print(f"\n--- Predictions on the {len(new_preds)} LLM-unlabeled page-8 callouts ---")
    for code, n in new_codes.most_common(15):
        print(f"  {n:3} × {code}")

    # Combined counts
    final = Counter()
    for p in predictions:
        if p["in_legend"]:
            continue
        code = p["llm_code"] or p["predicted_code"]
        if code:
            final[code] += 1
    print(f"\n--- Combined placed counts (LLM + glyph-match fallback) ---")
    ref = {
        "WS5": "23 EA", "WS7": "7 EA", "R1": "2 EA", "R2": "8 EA",
        "R3": "2 EA", "R4": "2 EA", "R5": "2 EA",
        "WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
        "WS19": "131 LF",
    }
    for code in sorted(final):
        print(f"  {code:6}  {final[code]:>4}  {ref.get(code, '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

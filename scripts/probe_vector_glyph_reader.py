"""Exploration: extract glyph paths inside each callout circle on page 8.

For each of the 92 callouts identified by the vector-callout v2 detector:
  1. Find all vector paths whose bounding box sits strictly inside the
     circle's bbox (these are candidate letter outlines).
  2. Cluster those paths into character groups by x-coordinate proximity.
  3. Compute a feature vector per character group: bbox, aspect ratio,
     stroke count, curve/line mix, total path length.

Just exploration — print features for the first 10 circles so we can see
whether the vector data is rich enough to distinguish characters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PRIOR_V2 = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
PAGE = 8
CIRCLE_DIAMETER = 27.0  # pt, from auto-detected signature
INSIDE_MARGIN = 1.0     # pt — exclude paths touching the circle outline
MIN_PATH_SIZE = 0.2     # pt — exclude stray points
CHAR_GAP_THRESHOLD = 1.5  # pt — gap between distinct characters


def is_glyph_path(path, circle_rect):
    pr = path["rect"]
    return (
        pr.x0 >= circle_rect.x0 + INSIDE_MARGIN
        and pr.x1 <= circle_rect.x1 - INSIDE_MARGIN
        and pr.y0 >= circle_rect.y0 + INSIDE_MARGIN
        and pr.y1 <= circle_rect.y1 - INSIDE_MARGIN
        and max(pr.x1 - pr.x0, pr.y1 - pr.y0) >= MIN_PATH_SIZE
    )


def cluster_by_x(paths, gap_threshold: float):
    """Sort by x-center, group consecutive paths whose left edge is within
    gap_threshold of the previous group's right edge.
    """
    if not paths:
        return []
    by_x = sorted(paths, key=lambda p: (p["rect"].x0 + p["rect"].x1) / 2)
    groups: list[list] = [[by_x[0]]]
    for p in by_x[1:]:
        prev_right = max(g["rect"].x1 for g in groups[-1])
        if p["rect"].x0 - prev_right <= gap_threshold:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def path_length(path) -> tuple[float, int, int]:
    """Total geometric length, # line ops, # curve ops."""
    length = 0.0
    n_l = n_c = 0
    for item in path.get("items", []):
        op = item[0]
        if op == "l" and len(item) >= 3:
            n_l += 1
            p1, p2 = item[1], item[2]
            length += math.hypot(p2.x - p1.x, p2.y - p1.y)
        elif op == "c" and len(item) >= 5:
            n_c += 1
            # crude length estimate: chord p1→p4
            p1, p4 = item[1], item[4]
            length += math.hypot(p4.x - p1.x, p4.y - p1.y)
        elif op == "re" and len(item) >= 2:
            n_l += 4
            r = item[1]
            length += 2 * (r.width + r.height)
        elif op == "qu" and len(item) >= 2:
            # filled quad — treat as 4 edges
            n_l += 4
            quad = item[1]
            length += (
                math.hypot(quad.ll.x - quad.ul.x, quad.ll.y - quad.ul.y)
                + math.hypot(quad.lr.x - quad.ll.x, quad.lr.y - quad.ll.y)
                + math.hypot(quad.ur.x - quad.lr.x, quad.ur.y - quad.lr.y)
                + math.hypot(quad.ul.x - quad.ur.x, quad.ul.y - quad.ur.y)
            )
    return length, n_l, n_c


def features_of_group(group):
    x0 = min(p["rect"].x0 for p in group)
    y0 = min(p["rect"].y0 for p in group)
    x1 = max(p["rect"].x1 for p in group)
    y1 = max(p["rect"].y1 for p in group)
    w, h = x1 - x0, y1 - y0
    total_len = 0.0
    total_l = total_c = 0
    for p in group:
        L, nl, nc = path_length(p)
        total_len += L
        total_l += nl
        total_c += nc
    return {
        "bbox": (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
        "w": round(w, 2),
        "h": round(h, 2),
        "aspect": round(w / h, 2) if h > 0 else None,
        "n_paths": len(group),
        "n_lines": total_l,
        "n_curves": total_c,
        "length": round(total_len, 1),
    }


def main() -> int:
    callouts = json.loads(PRIOR_V2.read_text())["ocr_results"]
    doc = fitz.open(PDF)
    page = doc.load_page(PAGE - 1)
    all_paths = page.get_drawings()
    print(f"Page 8: {len(callouts)} callouts, {len(all_paths)} total vector paths\n")

    summary_glyph_counts = []
    summary_char_counts = []

    for i, c in enumerate(callouts[:10]):
        cx, cy = c["cx"], c["cy"]
        cr = fitz.Rect(cx - CIRCLE_DIAMETER / 2, cy - CIRCLE_DIAMETER / 2,
                       cx + CIRCLE_DIAMETER / 2, cy + CIRCLE_DIAMETER / 2)
        # gather paths inside the circle's bbox (excluding the circle itself)
        inside = [p for p in all_paths if is_glyph_path(p, cr)]
        groups = cluster_by_x(inside, CHAR_GAP_THRESHOLD)

        ocr_label = c.get("code") or (c.get("raw")[:30] if c.get("raw") else "(blank)")
        legend = "[LEGEND]" if c.get("in_legend") else ""
        print(f"--- circle {i+1}: center=({cx:.0f},{cy:.0f}) {legend} ---")
        print(f"  prior LLM OCR: {ocr_label!r}")
        print(f"  glyph paths inside: {len(inside)}, "
              f"clustered into {len(groups)} character group(s)")
        for j, g in enumerate(groups):
            f = features_of_group(g)
            print(f"    char {j+1}: bbox={f['bbox']}  "
                  f"{f['w']:>4} x {f['h']:>4}  aspect={f['aspect']}  "
                  f"paths={f['n_paths']:>2}  l={f['n_lines']:>2} c={f['n_curves']:>2}  "
                  f"len={f['length']:>4}")
        print()

        summary_glyph_counts.append(len(inside))
        summary_char_counts.append(len(groups))

    # Aggregate stats over all 92
    glyph_counts = []
    char_counts = []
    for c in callouts:
        cx, cy = c["cx"], c["cy"]
        cr = fitz.Rect(cx - CIRCLE_DIAMETER / 2, cy - CIRCLE_DIAMETER / 2,
                       cx + CIRCLE_DIAMETER / 2, cy + CIRCLE_DIAMETER / 2)
        inside = [p for p in all_paths if is_glyph_path(p, cr)]
        groups = cluster_by_x(inside, CHAR_GAP_THRESHOLD)
        glyph_counts.append(len(inside))
        char_counts.append(len(groups))

    from collections import Counter
    print("=" * 78)
    print(f"Aggregate stats across all {len(callouts)} callouts:")
    print(f"  glyph-path counts:  min={min(glyph_counts)}  median={sorted(glyph_counts)[len(glyph_counts)//2]}  max={max(glyph_counts)}")
    print(f"  character-group counts:  {dict(Counter(char_counts))}")
    # Cross-tabulate by whether the LLM read a code
    valid_chars = [n for n, c in zip(char_counts, callouts) if c.get("code")]
    invalid_chars = [n for n, c in zip(char_counts, callouts) if not c.get("code")]
    if valid_chars:
        print(f"\n  Char-group count distribution where LLM returned a valid code "
              f"({len(valid_chars)} circles):")
        print(f"    {dict(Counter(valid_chars))}")
    if invalid_chars:
        print(f"  Char-group count distribution where LLM returned no code "
              f"({len(invalid_chars)} circles):")
        print(f"    {dict(Counter(invalid_chars))}")

    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

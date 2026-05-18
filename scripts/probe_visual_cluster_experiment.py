"""Quick experiment: cluster page-8 callouts by visual similarity of their
rendered crops, then label clusters by LLM-OCR majority vote.

  - Render the page once at 300 DPI
  - Crop each of the 92 callouts (cached positions from page8_ocr_v2.json)
  - Downscale each crop to 32x32 grayscale → flattened 1024-dim vector
  - Agglomerative clustering with Euclidean linkage, target ~15 clusters
  - Vote-label each cluster from members' LLM reads
  - Compare resulting counts to Reference (Tyler) EA totals
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PRIOR_OCR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_visual_cluster.json"
)
DPI = 300
CROP_HALF_PT = 30        # 60x60pt crop (tighter than the 120pt OCR crops)
THUMB_SIZE = 32          # 32x32 grayscale
TARGET_CLUSTERS = 15     # rough target — Park Ave has ~15-20 distinct codes


def main() -> int:
    if not PRIOR_OCR.exists():
        print(f"error: missing {PRIOR_OCR}", file=sys.stderr)
        return 1
    callouts = json.loads(PRIOR_OCR.read_text())["ocr_results"]
    print(f"Loaded {len(callouts)} callouts")

    # Render page 8 once
    doc = fitz.open(PDF)
    page = doc.load_page(7)
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    doc.close()
    print(f"Rendered page at {DPI} DPI: {img.size[0]} × {img.size[1]} px")

    # Crop, downscale, flatten
    vectors = []
    valid_idx = []
    crop_half_px = int(CROP_HALF_PT * scale)
    for i, c in enumerate(callouts):
        x0 = int(c["cx"] * scale) - crop_half_px
        y0 = int(c["cy"] * scale) - crop_half_px
        x1 = x0 + 2 * crop_half_px
        y1 = y0 + 2 * crop_half_px
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width, x1), min(img.height, y1)
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        crop = img.crop((x0, y0, x1, y1)).resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        arr = np.asarray(crop, dtype=np.float32) / 255.0  # normalize to [0,1]
        vectors.append(arr.flatten())
        valid_idx.append(i)
    X = np.stack(vectors)
    print(f"Built feature matrix: {X.shape} (callouts × pixels)")

    # Agglomerative clustering — target ~TARGET_CLUSTERS
    clusterer = AgglomerativeClustering(n_clusters=TARGET_CLUSTERS, linkage="ward")
    labels = clusterer.fit_predict(X)
    print(f"\nClustering: {TARGET_CLUSTERS} requested clusters with Ward linkage")

    # Build per-cluster info
    clusters: dict[int, dict] = defaultdict(lambda: {"members": [], "llm_reads": Counter()})
    for cluster_id, vi in zip(labels, valid_idx):
        c = callouts[vi]
        clusters[int(cluster_id)]["members"].append(vi)
        if c.get("code"):
            clusters[int(cluster_id)]["llm_reads"][c["code"]] += 1

    # Vote-label each cluster (label only when at least one LLM read exists)
    for cid, info in clusters.items():
        if info["llm_reads"]:
            label, _ = info["llm_reads"].most_common(1)[0]
            info["label"] = label
        else:
            info["label"] = None

    # Print cluster summary
    print(f"\n{'cluster':>7}  {'size':>4}  {'label':>8}  LLM reads (within cluster)")
    print("-" * 75)
    cluster_items = sorted(clusters.items(), key=lambda kv: -len(kv[1]["members"]))
    for cid, info in cluster_items:
        reads = info["llm_reads"]
        reads_str = ", ".join(f"{k}×{v}" for k, v in reads.most_common())
        print(f"  {cid:>5}  {len(info['members']):>4}  "
              f"{info['label'] or '—':>8}  {reads_str or '(no LLM reads)'}")

    # Coverage stats
    labeled_callouts = sum(len(info["members"]) for info in clusters.values()
                           if info["label"])
    unlabeled_callouts = sum(len(info["members"]) for info in clusters.values()
                             if not info["label"])
    distinct_codes = {info["label"] for info in clusters.values() if info["label"]}
    print(f"\nCallouts in labeled clusters:   {labeled_callouts}/{len(valid_idx)}  "
          f"({labeled_callouts/len(valid_idx)*100:.1f}%)")
    print(f"Callouts in unlabeled clusters: {unlabeled_callouts}/{len(valid_idx)}")
    print(f"Distinct codes identified:      {len(distinct_codes)}")
    print(f"  {sorted(distinct_codes)}")

    # Aggregate counts per code (placed callouts only — exclude in_legend)
    code_counts: Counter = Counter()
    for cid, info in clusters.items():
        if not info["label"]:
            continue
        for vi in info["members"]:
            if callouts[vi].get("in_legend"):
                continue
            code_counts[info["label"]] += 1

    print(f"\nPer-code placed counts (after visual clustering):")
    reference_ea = {"R1": 2, "R2": 8, "R3": 2, "R4": 2, "R5": 2,
                    "WS5": 23, "WS7": 7}
    reference_other = {"WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
                       "WS9": "211 LF", "WS19": "131 LF",
                       "WS6": "30 SF", "WS10": "200 LF + 208 SF",
                       "WS12": "6 SF", "WS13": "48 LF", "WS14": "17 SF",
                       "WS15": "127 SF", "WS16": "26 SF", "WS17": "11 LF",
                       "WS18": "53 SF"}
    print(f"  {'code':6}  {'count':>5}   Reference   Δ%")
    ea_matched = 0
    ea_total = 0
    for code in sorted(code_counts):
        cnt = code_counts[code]
        if code in reference_ea:
            target = reference_ea[code]
            ea_total += 1
            delta_pct = (cnt - target) / target * 100
            within = abs(delta_pct) <= 15
            if within:
                ea_matched += 1
            mark = " ✓" if within else " ✗"
            print(f"  {code:6}  {cnt:>5}   {target} EA       {delta_pct:+.0f}%{mark}")
        elif code in reference_other:
            print(f"  {code:6}  {cnt:>5}   {reference_other[code]}")
        else:
            print(f"  {code:6}  {cnt:>5}   —")

    print(f"\nEA-direct comparisons within ±15%: {ea_matched}/{ea_total}")

    # Compare against the 38% LLM baseline
    llm_labeled = sum(1 for c in callouts if c.get("code"))
    print(f"\n--- Headline ---")
    print(f"LLM v2 baseline:          {llm_labeled}/{len(callouts)} "
          f"({llm_labeled/len(callouts)*100:.1f}% labeled)")
    print(f"Visual cluster + vote:    {labeled_callouts}/{len(valid_idx)} "
          f"({labeled_callouts/len(valid_idx)*100:.1f}% labeled)")

    # Save
    OUT.write_text(json.dumps({
        "page": 8,
        "n_clusters": TARGET_CLUSTERS,
        "dpi": DPI, "crop_half_pt": CROP_HALF_PT, "thumb_size": THUMB_SIZE,
        "clusters": [
            {
                "id": cid,
                "size": len(info["members"]),
                "label": info["label"],
                "llm_reads": dict(info["llm_reads"]),
                "members": [
                    {"cx": callouts[vi]["cx"], "cy": callouts[vi]["cy"],
                     "in_legend": callouts[vi].get("in_legend"),
                     "llm_code": callouts[vi].get("code")}
                    for vi in info["members"]
                ],
            }
            for cid, info in clusters.items()
        ],
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

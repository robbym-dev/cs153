"""Quick experiment: DINOv2-small embeddings + agglomerative clustering.

Replaces the raw-pixel clustering experiment with a learned representation.
Same crops (page-8 callouts, 120x120pt @ 600 DPI, MinFilter(3) dilation),
but each is encoded by DINOv2-small's CLS token (384-dim) instead of
being flattened raw.

Then: pairwise cosine distance, agglomerative clustering with n=15/20/25,
vote-label clusters from the 35 confident LLM reads, compare per-code
counts to Reference (Tyler) EA totals.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PRIOR_OCR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_dinov2_cluster.json"
)
DPI = 600
CROP_HALF_PT = 60      # 120x120pt crops, same as prior dilate experiments
N_CLUSTERS_TO_TRY = (15, 20, 25)

# Reference (Tyler) full-building totals
REF_EA = {"R1": 2, "R2": 8, "R3": 2, "R4": 2, "R5": 2, "WS5": 23, "WS7": 7}
REF_OTHER = {
    "WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
    "WS9": "211 LF", "WS19": "131 LF",
    "WS6": "30 SF", "WS10": "200 LF + 208 SF",
    "WS12": "6 SF", "WS13": "48 LF", "WS14": "17 SF",
    "WS15": "127 SF", "WS16": "26 SF", "WS17": "11 LF", "WS18": "53 SF",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dinov2")


def render_crops():
    callouts = json.loads(PRIOR_OCR.read_text())["ocr_results"]
    doc = fitz.open(PDF)
    page = doc.load_page(7)
    scale = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    crop_px = int(CROP_HALF_PT * scale)
    crops = []
    for c in callouts:
        x0 = int(c["cx"] * scale) - crop_px
        y0 = int(c["cy"] * scale) - crop_px
        x1, y1 = x0 + 2 * crop_px, y0 + 2 * crop_px
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(img.width, x1), min(img.height, y1)
        if x1c - x0c < 50 or y1c - y0c < 50:
            crops.append(None)
            continue
        cr = img.crop((x0c, y0c, x1c, y1c)).filter(ImageFilter.MinFilter(3))
        crops.append(cr)
    return callouts, crops


def embed_with_dinov2(crops):
    log.info("loading facebook/dinov2-small (CPU)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()

    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    pil_crops = [crops[i].convert("RGB") for i in valid_idx]
    log.info(f"  encoding {len(pil_crops)} crops...")
    embeddings = []
    BATCH = 16
    with torch.no_grad():
        for i in range(0, len(pil_crops), BATCH):
            batch = pil_crops[i:i + BATCH]
            inputs = processor(images=batch, return_tensors="pt")
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].numpy()  # (B, 384)
            embeddings.append(cls)
    X = np.concatenate(embeddings, axis=0)
    log.info(f"  embedding matrix: {X.shape}  (callouts × {X.shape[1]} dims)")
    return X, valid_idx


def cluster_and_label(X, valid_idx, callouts, n_clusters):
    dist = cosine_distances(X)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(dist)

    clusters = defaultdict(lambda: {"members": [], "llm_reads": Counter()})
    for cluster_id, vi in zip(labels, valid_idx):
        c = callouts[vi]
        clusters[int(cluster_id)]["members"].append(vi)
        if c.get("code"):
            clusters[int(cluster_id)]["llm_reads"][c["code"]] += 1
    for info in clusters.values():
        info["label"] = (
            info["llm_reads"].most_common(1)[0][0]
            if info["llm_reads"] else None
        )
    return clusters


def report(label, clusters, callouts, n_valid):
    labeled = sum(len(c["members"]) for c in clusters.values() if c["label"])
    distinct = {c["label"] for c in clusters.values() if c["label"]}
    counts = Counter()
    for info in clusters.values():
        if not info["label"]:
            continue
        for k in info["members"]:
            if callouts[k].get("in_legend"):
                continue
            counts[info["label"]] += 1

    print(f"\n=== {label} ===")
    print(f"  callouts in labeled clusters: {labeled}/{n_valid}  ({labeled/n_valid*100:.0f}%)")
    print(f"  distinct codes identified:    {len(distinct)}")

    # Top 8 clusters
    big = sorted(clusters.items(), key=lambda kv: -len(kv[1]["members"]))[:8]
    print(f"  top 8 clusters:")
    for cid, info in big:
        reads = ", ".join(f"{k}×{v}" for k, v in info["llm_reads"].most_common(4))
        print(f"    [size {len(info['members']):>2}  →  {str(info['label']):>6}]"
              f"   LLM in cluster: {reads or '(none)'}")

    # Per-code counts + EA comparison
    ea_ok = ea_total = 0
    print(f"\n  Per-code placed counts:")
    print(f"    {'CODE':6}  {'count':>5}   Reference   Δ%")
    for code in sorted(counts):
        cnt = counts[code]
        if code in REF_EA:
            tgt = REF_EA[code]
            ea_total += 1
            d = (cnt - tgt) / tgt * 100
            within = abs(d) <= 15
            if within:
                ea_ok += 1
            mark = " ✓" if within else " ✗"
            print(f"    {code:6}  {cnt:>5}   {tgt} EA       {d:+.0f}%{mark}")
        elif code in REF_OTHER:
            print(f"    {code:6}  {cnt:>5}   {REF_OTHER[code]}")
        else:
            print(f"    {code:6}  {cnt:>5}   (not in reference)")
    print(f"  EA-direct matches within ±15%: {ea_ok}/{ea_total}")
    return {
        "labeled_callouts": labeled, "distinct_codes": len(distinct),
        "ea_matches": ea_ok, "ea_total": ea_total, "counts": dict(counts),
    }


def main() -> int:
    callouts, crops = render_crops()
    n = len(callouts)
    llm_valid = sum(1 for c in callouts if c.get("code"))
    log.info(f"Loaded {n} callouts ({llm_valid} LLM-labeled, {n - llm_valid} unlabeled)")
    log.info(f"Rendered {sum(1 for c in crops if c is not None)} crops "
             f"(120x120pt @ {DPI} DPI, MinFilter(3))")

    X, valid_idx = embed_with_dinov2(crops)

    all_runs = []
    for k in N_CLUSTERS_TO_TRY:
        clusters = cluster_and_label(X, valid_idx, callouts, k)
        info = report(f"n_clusters = {k}", clusters, callouts, len(valid_idx))
        info["n_clusters"] = k
        all_runs.append({"n_clusters": k, "info": info, "clusters": [
            {"id": cid, "size": len(c["members"]), "label": c["label"],
             "llm_reads": dict(c["llm_reads"]),
             "members": [{"cx": callouts[m]["cx"], "cy": callouts[m]["cy"],
                          "in_legend": callouts[m].get("in_legend"),
                          "llm_code": callouts[m].get("code")}
                         for m in c["members"]]}
            for cid, c in clusters.items()
        ]})

    print("\n" + "=" * 70)
    print("HEADLINE — page-8 callout reading approaches")
    print("=" * 70)
    print(f"  LLM v2 baseline (Opus 4.7):              35/92  (38.0%)  EA match —")
    print(f"  Raw-pixel cluster (60pt × 32px × 15):    76/83  (92.0%)  EA 0/2")
    for r in all_runs:
        info = r["info"]
        print(f"  DINOv2-small + agglomerative (n={r['n_clusters']:>2}):     "
              f"{info['labeled_callouts']:>2}/{len(valid_idx):>2}  "
              f"({info['labeled_callouts']/len(valid_idx)*100:>4.0f}%)  "
              f"EA {info['ea_matches']}/{info['ea_total']}, "
              f"{info['distinct_codes']} distinct codes")

    OUT.write_text(json.dumps({
        "page": 8, "dpi": DPI, "crop_half_pt": CROP_HALF_PT,
        "model": "facebook/dinov2-small",
        "embedding_dim": int(X.shape[1]),
        "runs": all_runs,
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

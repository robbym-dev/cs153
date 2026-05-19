"""Constrained agglomerative clustering: DINOv2 embeddings + must-not-link.

Same DINOv2-small CLS embeddings of the 92 page-8 callout crops as the
prior experiment. Difference: enforce cannot-link constraints between
any two callouts whose LLM reads disagree. Implemented by inflating the
pairwise cosine distance between forbidden pairs to a sentinel value;
with average linkage, that prevents any merge that would put two
disagreeing LLM-labeled callouts in the same cluster.

Target: n=20 clusters. Compare per-code counts vs Reference (Tyler).
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
    / "vector_callouts" / "page8_dinov2_constrained.json"
)

DPI = 600
CROP_HALF_PT = 60
N_CLUSTERS = 20
FORBIDDEN_DIST = 100.0  # >> any cosine distance (which lives in [0, 2])

REF_EA = {"R1": 2, "R2": 8, "R3": 2, "R4": 2, "R5": 2, "WS5": 23, "WS7": 7}
REF_OTHER = {
    "WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
    "WS9": "211 LF", "WS19": "131 LF",
    "WS6": "30 SF", "WS10": "200 LF + 208 SF",
    "WS12": "6 SF", "WS13": "48 LF", "WS14": "17 SF",
    "WS15": "127 SF", "WS16": "26 SF", "WS17": "11 LF", "WS18": "53 SF",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("constrained")


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
        crops.append(cr.convert("RGB"))
    return callouts, crops


def embed(crops):
    log.info("loading facebook/dinov2-small...")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()
    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    embs = []
    BATCH = 16
    with torch.no_grad():
        for i in range(0, len(valid_idx), BATCH):
            batch = [crops[valid_idx[j]] for j in range(i, min(i + BATCH, len(valid_idx)))]
            out = model(**proc(images=batch, return_tensors="pt"))
            embs.append(out.last_hidden_state[:, 0, :].numpy())
    X = np.concatenate(embs, axis=0)
    log.info(f"  embedded {X.shape[0]} crops → dim {X.shape[1]}")
    return X, valid_idx


def apply_cannot_link(D, valid_idx, callouts):
    """Inflate D[i,j] to FORBIDDEN_DIST when both have LLM labels that disagree.

    Returns: (D_modified, n_forbidden_pairs)."""
    D = D.copy()
    n = len(valid_idx)
    labels = [callouts[valid_idx[i]].get("code") for i in range(n)]
    n_forbidden = 0
    for i in range(n):
        if not labels[i]:
            continue
        for j in range(i + 1, n):
            if labels[j] and labels[j] != labels[i]:
                D[i, j] = FORBIDDEN_DIST
                D[j, i] = FORBIDDEN_DIST
                n_forbidden += 1
    return D, n_forbidden


def cluster_and_label(D, valid_idx, callouts, n_clusters):
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(D)
    clusters = defaultdict(lambda: {"members": [], "llm_reads": Counter()})
    for cid, vi in zip(labels, valid_idx):
        clusters[int(cid)]["members"].append(vi)
        if callouts[vi].get("code"):
            clusters[int(cid)]["llm_reads"][callouts[vi]["code"]] += 1
    for info in clusters.values():
        info["label"] = (info["llm_reads"].most_common(1)[0][0]
                         if info["llm_reads"] else None)
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

    # Verify constraints honored
    purity_violations = 0
    for info in clusters.values():
        if len(info["llm_reads"]) > 1:
            purity_violations += 1
    print(f"  clusters with mixed LLM reads: {purity_violations}  (should be 0)")

    big = sorted(clusters.items(), key=lambda kv: -len(kv[1]["members"]))[:10]
    print(f"  top 10 clusters:")
    for cid, info in big:
        reads = ", ".join(f"{k}×{v}" for k, v in info["llm_reads"].most_common(4))
        print(f"    [size {len(info['members']):>2}  →  {str(info['label']):>6}]   "
              f"LLM: {reads or '(none)'}")

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
            if within: ea_ok += 1
            mark = " ✓" if within else " ✗"
            print(f"    {code:6}  {cnt:>5}   {tgt} EA       {d:+.0f}%{mark}")
        elif code in REF_OTHER:
            print(f"    {code:6}  {cnt:>5}   {REF_OTHER[code]}")
        else:
            print(f"    {code:6}  {cnt:>5}   (not in reference)")
    print(f"  EA-direct matches within ±15%: {ea_ok}/{ea_total}")
    return {"labeled": labeled, "distinct": len(distinct),
            "ea_ok": ea_ok, "ea_total": ea_total, "counts": dict(counts),
            "mixed_clusters": purity_violations}


def main() -> int:
    callouts, crops = render_crops()
    n = len(callouts)
    llm_valid = sum(1 for c in callouts if c.get("code"))
    log.info(f"Loaded {n} callouts; {llm_valid} LLM-labeled")
    log.info(f"Rendered {sum(1 for c in crops if c is not None)} crops "
             f"(120x120pt @ {DPI} DPI, MinFilter(3))")

    X, valid_idx = embed(crops)
    D = cosine_distances(X)

    # Unconstrained baseline (recompute to confirm parity with prior run)
    log.info("\n--- Unconstrained DINOv2 clustering @ n=20 (sanity check) ---")
    cl_unc = cluster_and_label(D, valid_idx, callouts, N_CLUSTERS)
    info_unc = report("unconstrained (baseline)", cl_unc, callouts, len(valid_idx))

    # Constrained
    D_constrained, n_forbidden = apply_cannot_link(D, valid_idx, callouts)
    log.info(f"\n--- Constrained DINOv2 clustering @ n=20 ---")
    log.info(f"  Forbidden pairs (different LLM labels): {n_forbidden}")
    cl_con = cluster_and_label(D_constrained, valid_idx, callouts, N_CLUSTERS)
    info_con = report("constrained (must-not-link from LLM)", cl_con, callouts,
                      len(valid_idx))

    print("\n" + "=" * 70)
    print("HEADLINE — page 8 clustering (n=20)")
    print("=" * 70)
    print(f"  {'approach':40}  {'labeled':>10}  {'EA':>5}  {'codes':>5}")
    print(f"  {'raw-pixel':40}  {'76/83':>10}  {'0/2':>5}  {'4':>5}")
    print(f"  {'DINOv2 (unconstrained)':40}  "
          f"{info_unc['labeled']}/{len(valid_idx):<2}     "
          f"{info_unc['ea_ok']}/{info_unc['ea_total']:<2}    {info_unc['distinct']:>5}")
    print(f"  {'DINOv2 + cannot-link from LLM':40}  "
          f"{info_con['labeled']}/{len(valid_idx):<2}     "
          f"{info_con['ea_ok']}/{info_con['ea_total']:<2}    {info_con['distinct']:>5}")

    OUT.write_text(json.dumps({
        "page": 8, "dpi": DPI, "n_clusters": N_CLUSTERS,
        "model": "facebook/dinov2-small",
        "n_forbidden_pairs": n_forbidden,
        "unconstrained": info_unc,
        "constrained": info_con,
        "constrained_clusters": [
            {"id": cid, "size": len(c["members"]), "label": c["label"],
             "llm_reads": dict(c["llm_reads"]),
             "members": [{"cx": callouts[m]["cx"], "cy": callouts[m]["cy"],
                          "in_legend": callouts[m].get("in_legend"),
                          "llm_code": callouts[m].get("code")}
                         for m in c["members"]]}
            for cid, c in cl_con.items()
        ],
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

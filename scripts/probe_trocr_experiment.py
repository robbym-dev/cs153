"""Quick experiment: TrOCR on the 92 page-8 callout crops.

Two tests:
  1. Direct OCR — VisionEncoderDecoderModel('microsoft/trocr-base-handwritten').
     Feed each crop, decode the generated text, filter with CODE_RE.
  2. If direct OCR doesn't beat the LLM (38%), pull just the encoder
     CLS embedding (the same encoder used internally by TrOCR), cluster
     with agglomerative at n=20, label by LLM-vote, compare against
     DINOv2's 1/2 EA match.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF = PROJECT_ROOT / "test_data" / "PA_Exterior_Renovations_Bid_Apr_6_2026.pdf"
PRIOR_OCR = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_ocr_v2.json"
)
OUT = (
    PROJECT_ROOT / "tests" / "extractions" / "park_ave_original"
    / "vector_callouts" / "page8_trocr.json"
)
MODEL_NAME = "microsoft/trocr-base-handwritten"
DPI = 600
CROP_HALF_PT = 60
N_CLUSTERS = 20
CODE_RE = re.compile(r"^([A-Z]+\d+[A-Z]*)$")

REF_EA = {"R1": 2, "R2": 8, "R3": 2, "R4": 2, "R5": 2, "WS5": 23, "WS7": 7}
REF_OTHER = {
    "WS1": "250 LF", "WS4": "506 LF", "WS8": "114 LF",
    "WS9": "211 LF", "WS19": "131 LF",
    "WS6": "30 SF", "WS10": "200 LF + 208 SF",
    "WS12": "6 SF", "WS13": "48 LF", "WS14": "17 SF",
    "WS15": "127 SF", "WS16": "26 SF", "WS17": "11 LF", "WS18": "53 SF",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")
log = logging.getLogger("trocr")


def clean_code(raw: str) -> str | None:
    s = (raw or "").strip().upper().strip("`'\".,:;")
    for tok in reversed(re.split(r"[\s/,;|]+", s)):
        tok = tok.strip(".,;:()[]")
        m = CODE_RE.match(tok)
        if m:
            return m.group(1)
    return None


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


def test1_direct_ocr(processor, model, valid_crops):
    log.info("\n--- Test 1: TrOCR direct OCR ---")
    raws = []
    BATCH = 4  # decoder generation is heavy; small batch on CPU
    with torch.no_grad():
        for i in range(0, len(valid_crops), BATCH):
            batch = valid_crops[i:i + BATCH]
            inputs = processor(images=batch, return_tensors="pt")
            gen = model.generate(inputs.pixel_values, max_new_tokens=10)
            text = processor.batch_decode(gen, skip_special_tokens=True)
            raws.extend(text)
            log.info(f"  TrOCR generate {min(i+BATCH, len(valid_crops))}/{len(valid_crops)}")
    return raws


def test2_cluster_embeddings(processor, model, valid_crops, valid_idx, callouts):
    log.info("\n--- Test 2: cluster TrOCR encoder CLS embeddings ---")
    embs = []
    BATCH = 16
    with torch.no_grad():
        for i in range(0, len(valid_crops), BATCH):
            batch = valid_crops[i:i + BATCH]
            inputs = processor(images=batch, return_tensors="pt")
            enc_out = model.encoder(pixel_values=inputs.pixel_values)
            cls = enc_out.last_hidden_state[:, 0, :].numpy()  # (B, hidden)
            embs.append(cls)
    X = np.concatenate(embs, axis=0)
    log.info(f"  encoder embedding matrix: {X.shape}")

    dist = cosine_distances(X)
    labels = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, metric="precomputed", linkage="average"
    ).fit_predict(dist)

    clusters: dict[int, dict] = defaultdict(
        lambda: {"members": [], "llm_reads": Counter()}
    )
    for cid, vi in zip(labels, valid_idx):
        clusters[int(cid)]["members"].append(vi)
        if callouts[vi].get("code"):
            clusters[int(cid)]["llm_reads"][callouts[vi]["code"]] += 1
    for info in clusters.values():
        info["label"] = (
            info["llm_reads"].most_common(1)[0][0]
            if info["llm_reads"] else None
        )

    counts: Counter = Counter()
    for info in clusters.values():
        if not info["label"]:
            continue
        for k in info["members"]:
            if callouts[k].get("in_legend"):
                continue
            counts[info["label"]] += 1

    labeled = sum(len(c["members"]) for c in clusters.values() if c["label"])
    distinct = {c["label"] for c in clusters.values() if c["label"]}
    log.info(f"  callouts in labeled clusters: {labeled}/{len(valid_idx)}")
    log.info(f"  distinct codes:               {len(distinct)}")

    log.info(f"  top 8 clusters:")
    big = sorted(clusters.items(), key=lambda kv: -len(kv[1]["members"]))[:8]
    for cid, info in big:
        reads = ", ".join(f"{k}×{v}" for k, v in info["llm_reads"].most_common(4))
        log.info(f"    [size {len(info['members']):>2}  →  {str(info['label']):>6}]   "
                 f"LLM: {reads or '(none)'}")

    ea_ok = ea_total = 0
    log.info(f"\n  Per-code placed counts:")
    log.info(f"    {'CODE':6}  {'count':>5}   Reference   Δ%")
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
            log.info(f"    {code:6}  {cnt:>5}   {tgt} EA       {d:+.0f}%{mark}")
        elif code in REF_OTHER:
            log.info(f"    {code:6}  {cnt:>5}   {REF_OTHER[code]}")
        else:
            log.info(f"    {code:6}  {cnt:>5}   (not in reference)")
    log.info(f"  EA-direct matches within ±15%: {ea_ok}/{ea_total}")
    return {
        "labeled": labeled, "distinct": len(distinct),
        "ea_ok": ea_ok, "ea_total": ea_total, "counts": dict(counts),
        "clusters": [
            {"id": cid, "size": len(c["members"]), "label": c["label"],
             "llm_reads": dict(c["llm_reads"]),
             "members": [{"cx": callouts[m]["cx"], "cy": callouts[m]["cy"],
                          "in_legend": callouts[m].get("in_legend"),
                          "llm_code": callouts[m].get("code")}
                         for m in c["members"]]}
            for cid, c in clusters.items()
        ],
    }


def main() -> int:
    callouts, crops = render_crops()
    n = len(callouts)
    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    valid_crops = [crops[i] for i in valid_idx]
    log.info(f"Loaded {n} callouts, rendered {len(valid_crops)} crops "
             f"(120x120pt @ {DPI} DPI, MinFilter(3))")
    llm_valid = sum(1 for c in callouts if c.get("code"))

    log.info(f"Loading {MODEL_NAME} (CPU)...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.eval()

    # ---------- Test 1: direct OCR ----------
    raws = test1_direct_ocr(processor, model, valid_crops)
    # Build results list aligned with all 92 callouts (None for off-page)
    results1 = []
    j = 0
    for i, c in enumerate(crops):
        if c is None:
            results1.append({"raw": "", "code": None, "error": "off-page"})
        else:
            raw = raws[j]; j += 1
            results1.append({"raw": raw, "code": clean_code(raw)})

    valid_count = sum(1 for r in results1 if r["code"])
    log.info(f"\n  TrOCR direct OCR valid: {valid_count}/{n}  ({valid_count/n*100:.1f}%)")
    log.info(f"  Top raw outputs:")
    raw_counts = Counter(r["raw"] for r in results1 if r["raw"])
    for raw, c_ in raw_counts.most_common(8):
        short = raw[:60].replace("\n", " ")
        log.info(f"    {c_:>3} × {short!r}")
    # Agreement with LLM
    agree = sum(1 for c, r in zip(callouts, results1)
                if c.get("code") and r.get("code") == c["code"])
    log.info(f"  agreement with LLM (35 labels): {agree}/{llm_valid}")

    # ---------- Test 2: cluster encoder embeddings ----------
    clustering_result = test2_cluster_embeddings(
        processor, model, valid_crops, valid_idx, callouts
    )

    # ---------- Headline ----------
    print("\n" + "=" * 70)
    print("HEADLINE — page-8 callout reading (92 callouts)")
    print("=" * 70)
    print(f"  LLM v2 baseline (Opus 4.7):               35/92 (38.0%)")
    print(f"  PaddleOCR:                                18/92 (19.6%)")
    print(f"  EasyOCR:                                  15/92 (16.3%)")
    print(f"  Tesseract LSTM:                            2/92 ( 2.2%)")
    print(f"  TrOCR-base-handwritten (direct):         {valid_count:>3}/{n} "
          f"({valid_count/n*100:>4.1f}%)")
    print()
    print(f"  Clustering @ n={N_CLUSTERS} — coverage, EA matches:")
    print(f"    DINOv2-small:               80/87 (92%)  EA 1/2 (R2 exact)")
    print(f"    TrOCR encoder embeddings:   {clustering_result['labeled']}/{len(valid_idx)} "
          f"({clustering_result['labeled']/len(valid_idx)*100:.0f}%)  "
          f"EA {clustering_result['ea_ok']}/{clustering_result['ea_total']}, "
          f"{clustering_result['distinct']} distinct codes")

    OUT.write_text(json.dumps({
        "page": 8, "model": MODEL_NAME, "dpi": DPI,
        "crop_half_pt": CROP_HALF_PT, "n_clusters": N_CLUSTERS,
        "direct_ocr": results1,
        "clustering": clustering_result,
    }, indent=2, default=str))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

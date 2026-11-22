"""
Generates deterministic, image_id-grouped nested stratified few-shot manifests
for TextVQA NEW_train under ALT_THREE_SPLITS (1%, 5%, 10%, 25%, 100%).

Usage (from project root):
    python TRAIN/ALT_THREE_SPLITS/scripts/generate_few_shot_splits.py
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

MANIFEST_IN = Path("DATA_PIPELINE/ALT_THREE_SPLITS/splits/crossvqa_split_manifest_pooled.parquet")
MANIFEST_OUT = Path("DATA_PIPELINE/ALT_THREE_SPLITS/splits/textvqa_few_shot_manifest.parquet")
SUMMARY_OUT = Path("DATA_PIPELINE/ALT_THREE_SPLITS/splits/textvqa_few_shot_summary.json")
FRACTIONS = [1, 5, 10, 25, 100]
SEED = 42


def main() -> None:
    assert MANIFEST_IN.exists(), f"Missing pooled split manifest: {MANIFEST_IN}"
    df = pl.read_parquet(MANIFEST_IN)

    tv_train = df.filter((pl.col("dataset") == "textvqa") & (pl.col("NEW_split") == "NEW_train"))
    assert tv_train.height > 0, "No textvqa NEW_train rows found in manifest."

    # Group strictly at image_id level to prevent paired-question leakage
    img_groups = (
        tv_train.group_by("image_id")
        .agg(
            [
                pl.len().alias("question_count"),
                pl.col("ocr_token_count").mean().alias("mean_ocr_tokens"),
            ]
        )
        .sort("image_id")
    )

    # Create discrete stratification bins: OCR density quartile + question_count (1 vs 2+)
    ocr_vals = img_groups["mean_ocr_tokens"].to_numpy()
    q_bins = np.digitize(ocr_vals, np.quantile(ocr_vals, [0.25, 0.5, 0.75]))
    q_cnt_bin = np.minimum(img_groups["question_count"].to_numpy(), 2)
    strata = np.array([f"{qb}_{qc}" for qb, qc in zip(q_bins, q_cnt_bin)])

    all_indices = np.arange(img_groups.height)
    image_ids = img_groups["image_id"].to_list()

    # Nested sampling via successive stratified subsets from largest fraction down (or cumulative incremental)
    # Build 25% first from 100%, then 10% from 25%, 5% from 10%, 1% from 5% to guarantee exact nesting
    selected_idx_by_pct = {100: set(all_indices.tolist())}
    current_pool = all_indices

    for pct in [25, 10, 5, 1]:
        target_n = max(1, int(round((pct / 100.0) * len(all_indices))))
        if target_n >= len(current_pool):
            chosen = current_pool
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=target_n, random_state=SEED + pct)
            pool_strata = strata[current_pool]
            # Merge rare strata if any stratum has < 2 samples in current pool
            unique, counts = np.unique(pool_strata, return_counts=True)
            rare = set(unique[counts < 2].tolist())
            if rare:
                pool_strata = np.array(["rare" if s in rare else s for s in pool_strata])
            sub_idx, _ = next(sss.split(current_pool, pool_strata))
            chosen = current_pool[sub_idx]
        selected_idx_by_pct[pct] = set(chosen.tolist())
        current_pool = np.sort(chosen)

    # Assert strict nested property
    for small, large in [(1, 5), (5, 10), (10, 25), (25, 100)]:
        assert selected_idx_by_pct[small].issubset(
            selected_idx_by_pct[large]
        ), f"Nesting violated between {small}% and {large}%"

    membership_rows = []
    for idx, img_id in enumerate(image_ids):
        membership_rows.append(
            {
                "image_id": img_id,
                "in_1pct": idx in selected_idx_by_pct[1],
                "in_5pct": idx in selected_idx_by_pct[5],
                "in_10pct": idx in selected_idx_by_pct[10],
                "in_25pct": idx in selected_idx_by_pct[25],
                "in_100pct": True,
            }
        )
    img_membership_df = pl.DataFrame(membership_rows)

    out_df = tv_train.select(["sample_id", "image_id", "group_key", "ocr_token_count"]).join(
        img_membership_df, on="image_id", how="left"
    )
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(MANIFEST_OUT)

    sha256 = hashlib.sha256(MANIFEST_OUT.read_bytes()).hexdigest()
    summary = {
        "source_manifest": str(MANIFEST_IN.as_posix()),
        "output_manifest": str(MANIFEST_OUT.as_posix()),
        "sha256": sha256,
        "seed": SEED,
        "total_queries": int(out_df.height),
        "total_unique_images": int(img_groups.height),
        "subsets": {},
    }
    for pct in FRACTIONS:
        col = f"in_{pct}pct"
        sub = out_df.filter(pl.col(col))
        summary["subsets"][f"{pct}pct"] = {
            "queries": int(sub.height),
            "unique_images": int(sub["image_id"].n_unique()),
            "mean_ocr_tokens": round(float(sub["ocr_token_count"].mean()), 3),
        }

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated few-shot manifest at {MANIFEST_OUT} (SHA-256: {sha256[:12]}...)")
    print(json.dumps(summary["subsets"], indent=2))


if __name__ == "__main__":
    main()

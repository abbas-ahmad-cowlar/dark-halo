#!/usr/bin/env python
"""
Step 3  –  Leakage-Safe Train / Val / Test Split
=================================================

Splits the modeling dataset by **simulation realization** to prevent
data leakage.  Halos from the same realization share cosmological
parameters and initial conditions, so they must never appear in
different splits.

Allocation (50 realizations → 35 / 8 / 7 = 70% / 16% / 14%):
    train : LH realizations  0..34  (35 realizations)
    val   : LH realizations 35..42  ( 8 realizations)
    test  : LH realizations 43..49  ( 7 realizations)

The realization ordering is shuffled with a fixed seed before
assignment so the split is deterministic but not ordered by the
original LH index.

Outputs
-------
- data/camels/{suite}_LH/dataset_split.parquet  (dataset with 'split' column)
- artifacts/step3_splits/split_manifest.json     (realization → split mapping + stats)
- artifacts/step3_splits/split_output.txt         (console log)

Usage
-----
    python scripts/step3_split.py
    python scripts/step3_split.py --suite tng
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lines: list[str] = []

def log(msg: str = "") -> None:
    _log_lines.append(msg)
    print(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: Leakage-safe splits.")
    p.add_argument("--suite", default="simba",
                   help="Suite tag (default: simba).")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path,
                   default=Path("artifacts/step3_splits"))
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for realization shuffle (default: 42).")
    p.add_argument("--train-frac", type=float, default=0.70,
                   help="Fraction of realizations for training (default: 0.70).")
    p.add_argument("--val-frac", type=float, default=0.16,
                   help="Fraction of realizations for validation (default: 0.16).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_dir = (args.data_dir or Path(f"data/camels/{args.suite}_LH")).resolve()
    out_dir = args.out_dir.resolve()
    test_frac = 1.0 - args.train_frac - args.val_frac

    log("=" * 60)
    log("Step 3: Leakage-Safe Train / Val / Test Split")
    log("=" * 60)
    log(f"  Seed       : {args.seed}")
    log(f"  Fractions  : train={args.train_frac:.0%}  val={args.val_frac:.0%}  test={test_frac:.0%}")
    log()

    t_start = time.time()

    # ── Load ──────────────────────────────────────────────────────────────
    dataset_path = data_dir / "dataset.parquet"
    log(f"  Loading: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    log(f"  Total rows: {len(df):,}")

    realization_ids = sorted(df["simulation_set_id"].unique())
    n_real = len(realization_ids)
    log(f"  Realizations: {n_real}  (IDs: {realization_ids[0]}..{realization_ids[-1]})")
    log()

    # ── Shuffle realizations with fixed seed ──────────────────────────────
    rng = np.random.RandomState(args.seed)
    shuffled = np.array(realization_ids)
    rng.shuffle(shuffled)

    n_train = int(np.round(n_real * args.train_frac))
    n_val = int(np.round(n_real * args.val_frac))
    n_test = n_real - n_train - n_val

    train_ids = sorted(shuffled[:n_train].tolist())
    val_ids = sorted(shuffled[n_train:n_train + n_val].tolist())
    test_ids = sorted(shuffled[n_train + n_val:].tolist())

    log(f"── Split allocation ──")
    log(f"  train : {n_train} realizations → IDs {train_ids}")
    log(f"  val   : {n_val} realizations → IDs {val_ids}")
    log(f"  test  : {n_test} realizations → IDs {test_ids}")
    log()

    # ── Assign split labels ───────────────────────────────────────────────
    split_map: dict[int, str] = {}
    for rid in train_ids:
        split_map[rid] = "train"
    for rid in val_ids:
        split_map[rid] = "val"
    for rid in test_ids:
        split_map[rid] = "test"

    df["split"] = df["simulation_set_id"].map(split_map)

    # Verify no unmapped realizations
    unmapped = df["split"].isna().sum()
    if unmapped > 0:
        log(f"  WARNING: {unmapped} rows have no split assignment!")
    else:
        log("  All rows assigned to a split ✓")
    log()

    # ── Per-split statistics ──────────────────────────────────────────────
    log("── Per-split statistics ──")
    stats: dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        n_rows = len(subset)
        n_reals = subset["simulation_set_id"].nunique()
        target = subset["log10_M_halo"]
        stats[split_name] = {
            "rows": n_rows,
            "realizations": n_reals,
            "target_mean": round(float(target.mean()), 4),
            "target_std": round(float(target.std()), 4),
            "target_min": round(float(target.min()), 4),
            "target_max": round(float(target.max()), 4),
        }
        log(f"  {split_name:5s}: {n_rows:>8,} rows  ({n_reals:>2} reals)  "
            f"log10_M mean={target.mean():.4f}  std={target.std():.4f}  "
            f"range=[{target.min():.2f}, {target.max():.2f}]")
    log()

    # ── Save ──────────────────────────────────────────────────────────────
    log("── Saving ──")
    split_path = data_dir / "dataset_split.parquet"
    df.to_parquet(split_path, index=False)
    log(f"  Dataset with split: {split_path}  ({split_path.stat().st_size / 1e6:.1f} MB)")

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": test_frac,
        },
        "realization_count": n_real,
        "split_realization_ids": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "split_stats": stats,
        "total_rows": len(df),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    manifest_path = out_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"  Manifest: {manifest_path}")
    log()

    # ── Leakage verification ──────────────────────────────────────────────
    log("── Leakage verification ──")
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    tv_overlap = train_set & val_set
    tt_overlap = train_set & test_set
    vt_overlap = val_set & test_set
    if tv_overlap or tt_overlap or vt_overlap:
        log(f"  LEAK DETECTED: train∩val={tv_overlap} train∩test={tt_overlap} val∩test={vt_overlap}")
    else:
        log("  No realization overlap between any splits ✓")
    log()

    log(f"  Elapsed: {manifest['elapsed_seconds']}s")
    log()
    log("Step 3 complete. ✓")

    # Write log
    output_path = out_dir / "split_output.txt"
    output_path.write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

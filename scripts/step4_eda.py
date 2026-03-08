#!/usr/bin/env python
"""
Step 4  –  EDA and Sanity Checks
=================================

Runs exploratory data analysis on the split dataset:
1. Distribution checks for target and all features
2. Missingness and NaN/inf audit
3. Correlation matrix (feature–feature + feature–target)
4. Outlier detection via Isolation Forest (OOD flagging)
5. Generates diagnostic plots saved to artifacts/step4_eda/

Outputs
-------
- artifacts/step4_eda/eda_report.txt        (full console log)
- artifacts/step4_eda/eda_metadata.json     (numeric stats)
- artifacts/step4_eda/target_distribution.png
- artifacts/step4_eda/feature_distributions.png
- artifacts/step4_eda/correlation_matrix.png
- artifacts/step4_eda/feature_vs_target.png
- artifacts/step4_eda/isolation_forest_ood.png

Usage
-----
    python scripts/step4_eda.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

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
    p = argparse.ArgumentParser(description="Step 4: EDA and Sanity Checks.")
    p.add_argument("--suite", default="simba")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step4_eda"))
    return p.parse_args()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "log10_M_halo"
FEATURES = [
    "stellar_mass", "gas_mass", "vel_disp", "sfr",
    "half_mass_rad", "pos_x", "pos_y", "pos_z", "env_dist_5nn",
]

# ---------------------------------------------------------------------------
# EDA Functions
# ---------------------------------------------------------------------------
def missingness_audit(df: pd.DataFrame) -> dict[str, Any]:
    log("── 1. Missingness & Non-Finite Audit ──")
    total = len(df)
    results: dict[str, Any] = {}
    all_clean = True
    for col in [TARGET] + FEATURES:
        nan_count = int(df[col].isna().sum())
        inf_count = int(np.isinf(df[col].values).sum()) if df[col].dtype != object else 0
        results[col] = {"nan": nan_count, "inf": inf_count}
        status = "✓" if nan_count == 0 and inf_count == 0 else "✗"
        if nan_count > 0 or inf_count > 0:
            all_clean = False
        log(f"  {col:20s}  NaN={nan_count:>6}  Inf={inf_count:>6}  {status}")
    log(f"  Overall: {'ALL CLEAN ✓' if all_clean else 'ISSUES FOUND ✗'}")
    log()
    return results


def distribution_stats(df: pd.DataFrame) -> dict[str, Any]:
    log("── 2. Distribution Statistics ──")
    stats: dict[str, Any] = {}
    for col in [TARGET] + FEATURES:
        s = df[col]
        d = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "q25": round(float(s.quantile(0.25)), 6),
            "median": round(float(s.median()), 6),
            "q75": round(float(s.quantile(0.75)), 6),
            "max": round(float(s.max()), 6),
            "skew": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }
        stats[col] = d
        log(f"  {col:20s}  mean={d['mean']:>14.4f}  std={d['std']:>14.4f}  "
            f"skew={d['skew']:>7.2f}  kurt={d['kurtosis']:>7.2f}")
    log()
    return stats


def plot_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    log("── 3. Target Distribution Plot ──")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    axes[0].hist(df[TARGET], bins=80, color="#4A90D9", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("log₁₀(M_halo) [Msun/h]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Target Distribution")

    # Per-split histogram
    for split, color in [("train", "#4A90D9"), ("val", "#D9534F"), ("test", "#5CB85C")]:
        subset = df[df["split"] == split]
        axes[1].hist(subset[TARGET], bins=60, alpha=0.5, label=split, color=color)
    axes[1].set_xlabel("log₁₀(M_halo)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Target by Split")
    axes[1].legend()

    # QQ-like: sorted target vs normal quantiles
    sorted_vals = np.sort(df[TARGET].values)
    norm_quantiles = np.random.randn(len(sorted_vals))
    norm_quantiles.sort()
    axes[2].scatter(norm_quantiles[::100], sorted_vals[::100], s=2, alpha=0.5, c="#4A90D9")
    axes[2].set_xlabel("Normal Quantiles")
    axes[2].set_ylabel("log₁₀(M_halo)")
    axes[2].set_title("Q-Q Plot (subsampled)")

    plt.tight_layout()
    path = out_dir / "target_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")
    log()


def plot_feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    log("── 4. Feature Distribution Plots ──")
    n = len(FEATURES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        vals = df[feat].values
        # Use log scale for mass columns (very skewed)
        if feat in ("stellar_mass", "gas_mass"):
            vals_plot = np.log10(vals + 1)  # +1 to handle zeros
            ax.set_xlabel(f"log₁₀({feat} + 1)")
        else:
            vals_plot = vals
            ax.set_xlabel(feat)
        ax.hist(vals_plot, bins=60, color="#6C5B7B", edgecolor="white", alpha=0.8)
        ax.set_ylabel("Count")
        ax.set_title(feat)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = out_dir / "feature_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")
    log()


def correlation_analysis(df: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    log("── 5. Correlation Analysis ──")
    cols = [TARGET] + FEATURES
    corr = df[cols].corr()

    # Feature-target correlations
    log("  Feature → Target correlations:")
    ft_corr: dict[str, float] = {}
    for feat in FEATURES:
        r = corr.loc[feat, TARGET]
        ft_corr[feat] = round(float(r), 4)
        bar = "█" * int(abs(r) * 30)
        sign = "+" if r > 0 else "-"
        log(f"    {feat:20s}  r={r:>+.4f}  {sign}{bar}")
    log()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix (lower triangle)")
    plt.tight_layout()
    path = out_dir / "correlation_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")
    log()
    return ft_corr


def plot_feature_vs_target(df: pd.DataFrame, out_dir: Path) -> None:
    log("── 6. Feature vs Target Scatter Plots ──")
    n = len(FEATURES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    # Subsample for scatter efficiency
    sample = df.sample(min(10000, len(df)), random_state=42)

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        x = sample[feat].values
        y = sample[TARGET].values
        if feat in ("stellar_mass", "gas_mass"):
            x = np.log10(x + 1)
            ax.set_xlabel(f"log₁₀({feat} + 1)")
        else:
            ax.set_xlabel(feat)
        ax.scatter(x, y, s=1, alpha=0.3, c="#355C7D")
        ax.set_ylabel("log₁₀(M_halo)")
        ax.set_title(feat)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = out_dir / "feature_vs_target.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")
    log()


def isolation_forest_ood(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """
    Train an Isolation Forest on the *training* split features to flag
    physically anomalous objects (OOD halos) across all splits.
    """
    log("── 7. Isolation Forest OOD Detection ──")

    train_mask = df["split"] == "train"
    X_train = df.loc[train_mask, FEATURES].values
    X_all = df[FEATURES].values

    # Log-transform mass features for better separation
    for i, feat in enumerate(FEATURES):
        if feat in ("stellar_mass", "gas_mass"):
            X_train[:, i] = np.log10(X_train[:, i] + 1)
            X_all[:, i] = np.log10(X_all[:, i] + 1)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,  # expect ~2% anomalies
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    preds = iso.predict(X_all)  # 1 = inlier, -1 = outlier
    scores = iso.decision_function(X_all)

    n_outliers = int((preds == -1).sum())
    n_total = len(preds)
    pct = n_outliers / n_total * 100

    log(f"  Outliers flagged: {n_outliers:,} / {n_total:,} ({pct:.2f}%)")

    # Per-split breakdown
    ood_stats: dict[str, Any] = {"total_outliers": n_outliers, "total": n_total, "pct": round(pct, 2)}
    for split in ["train", "val", "test"]:
        mask = df["split"].values == split
        n_out = int((preds[mask] == -1).sum())
        n_tot = int(mask.sum())
        log(f"    {split:5s}: {n_out:>5} / {n_tot:>7} outliers ({n_out/n_tot*100:.2f}%)")
        ood_stats[split] = {"outliers": n_out, "total": n_tot}
    log()

    # Plot: anomaly score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(scores, bins=80, color="#8E44AD", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0, color="red", linestyle="--", label="Threshold")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Isolation Forest Score Distribution")
    axes[0].legend()

    # Scatter: outliers in target vs strongest feature
    sample_idx = np.random.RandomState(42).choice(len(df), min(15000, len(df)), replace=False)
    ax = axes[1]
    inlier_mask = preds[sample_idx] == 1
    outlier_mask = preds[sample_idx] == -1
    ax.scatter(df[TARGET].values[sample_idx[inlier_mask]],
               scores[sample_idx[inlier_mask]], s=1, alpha=0.3, c="#2C3E50", label="Inlier")
    ax.scatter(df[TARGET].values[sample_idx[outlier_mask]],
               scores[sample_idx[outlier_mask]], s=5, alpha=0.8, c="#E74C3C", label="Outlier")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("log₁₀(M_halo)")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("OOD vs Target Mass")
    ax.legend()

    plt.tight_layout()
    path = out_dir / "isolation_forest_ood.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")
    log()

    return ood_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_dir = (args.data_dir or Path(f"data/camels/{args.suite}_LH")).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log("=" * 60)
    log("Step 4: EDA and Sanity Checks")
    log("=" * 60)
    log()

    # Load
    dataset_path = data_dir / "dataset_split.parquet"
    log(f"  Loading: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    log(f"  Rows: {len(df):,}  Columns: {df.shape[1]}")
    log(f"  Splits: {dict(df['split'].value_counts())}")
    log()

    metadata: dict[str, Any] = {"suite": args.suite, "total_rows": len(df)}

    # 1. Missingness
    metadata["missingness"] = missingness_audit(df)

    # 2. Distribution stats
    metadata["distribution_stats"] = distribution_stats(df)

    # 3. Target distribution plot
    plot_target_distribution(df, out_dir)

    # 4. Feature distributions
    plot_feature_distributions(df, out_dir)

    # 5. Correlation analysis
    metadata["feature_target_correlations"] = correlation_analysis(df, out_dir)

    # 6. Feature vs target scatter
    plot_feature_vs_target(df, out_dir)

    # 7. Isolation Forest OOD
    metadata["ood_detection"] = isolation_forest_ood(df, out_dir)

    # Summary
    elapsed = round(time.time() - t_start, 1)
    metadata["elapsed_seconds"] = elapsed
    log(f"  Total elapsed: {elapsed}s")
    log()
    log("Step 4 complete. ✓")

    # Save metadata
    meta_path = out_dir / "eda_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save log
    output_path = out_dir / "eda_report.txt"
    output_path.write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

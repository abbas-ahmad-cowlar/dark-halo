#!/usr/bin/env python
"""
Step 5  –  Baseline 1: Linear / Ridge Regression
==================================================

Trains Linear and Ridge regression models as the lower-bound baseline.
Uses StandardScaler preprocessing and log-transforms for skewed mass
features (informed by Step 4 EDA).

Outputs
-------
- artifacts/step5_linear/linear_results.json   (metrics)
- artifacts/step5_linear/pred_vs_true.png      (scatter)
- artifacts/step5_linear/residual_plot.png      (residual diagnostics)
- artifacts/step5_linear/step5_output.txt       (console log)

Usage
-----
    python scripts/step5_linear_baseline.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lines: list[str] = []

def log(msg: str = "") -> None:
    _log_lines.append(msg)
    print(msg)

# ---------------------------------------------------------------------------
TARGET = "log10_M_halo"
RAW_FEATURES = [
    "stellar_mass", "gas_mass", "vel_disp", "sfr",
    "half_mass_rad", "env_dist_5nn",
]
# Positions excluded — EDA showed r ≈ 0 with target

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 5: Linear/Ridge baseline.")
    p.add_argument("--suite", default="simba")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step5_linear"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Log-transform highly skewed mass features, keep others as-is."""
    out = df.copy()
    for feat in features:
        if feat in ("stellar_mass", "gas_mass"):
            out[f"log_{feat}"] = np.log10(out[feat] + 1)
    return out


def get_feature_names() -> list[str]:
    """Return feature column names after preprocessing."""
    cols = []
    for f in RAW_FEATURES:
        if f in ("stellar_mass", "gas_mass"):
            cols.append(f"log_{f}")
        else:
            cols.append(f)
    return cols


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    log(f"  {label:20s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6)}


def plot_pred_vs_true(
    results: dict[str, dict], y_true: np.ndarray,
    predictions: dict[str, np.ndarray], out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_true, y_pred, s=1, alpha=0.2, c="#355C7D")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect")
        metrics = results[name]["test"]
        ax.set_xlabel("True log₁₀(M_halo)")
        ax.set_ylabel("Predicted log₁₀(M_halo)")
        ax.set_title(f"{name}\nRMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")
        ax.legend()

    plt.tight_layout()
    path = out_dir / "pred_vs_true.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


def plot_residuals(
    results: dict, y_true: np.ndarray,
    predictions: dict[str, np.ndarray], out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, len(predictions), figsize=(6 * len(predictions), 8))
    if len(predictions) == 1:
        axes = axes.reshape(-1, 1)

    for col, (name, y_pred) in enumerate(predictions.items()):
        residuals = y_true - y_pred

        # Residual vs predicted
        axes[0, col].scatter(y_pred, residuals, s=1, alpha=0.2, c="#6C5B7B")
        axes[0, col].axhline(y=0, color="red", linestyle="--", alpha=0.7)
        axes[0, col].set_xlabel("Predicted log₁₀(M_halo)")
        axes[0, col].set_ylabel("Residual")
        axes[0, col].set_title(f"{name}: Residual vs Predicted")

        # Residual histogram
        axes[1, col].hist(residuals, bins=80, color="#C06C84", edgecolor="white", alpha=0.8)
        axes[1, col].axvline(x=0, color="red", linestyle="--", alpha=0.7)
        axes[1, col].set_xlabel("Residual")
        axes[1, col].set_ylabel("Count")
        axes[1, col].set_title(f"{name}: Residual Distribution\nstd={residuals.std():.4f}")

    plt.tight_layout()
    path = out_dir / "residual_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


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
    log("Step 5: Baseline 1 — Linear / Ridge Regression")
    log("=" * 60)
    log()

    # Load
    df = pd.read_parquet(data_dir / "dataset_split.parquet")
    log(f"  Loaded: {len(df):,} rows")

    # Preprocess
    df = preprocess(df, RAW_FEATURES)
    feat_cols = get_feature_names()
    log(f"  Features: {feat_cols}")
    log()

    # Split
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_train = train[feat_cols].values
    y_train = train[TARGET].values
    X_val = val[feat_cols].values
    y_val = val[TARGET].values
    X_test = test[feat_cols].values
    y_test = test[TARGET].values

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    log(f"  Train: {len(train):>8,}    Val: {len(val):>7,}    Test: {len(test):>7,}")
    log()

    # ── Train models ──────────────────────────────────────────────────────
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(α=1.0)": Ridge(alpha=1.0),
        "Ridge(α=10.0)": Ridge(alpha=10.0),
    }

    all_results: dict[str, dict] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        log(f"── {name} ──")
        model.fit(X_train_s, y_train)

        model_results: dict[str, Any] = {}
        for split_name, X_s, y in [("train", X_train_s, y_train),
                                    ("val", X_val_s, y_val),
                                    ("test", X_test_s, y_test)]:
            y_pred = model.predict(X_s)
            model_results[split_name] = evaluate(y, y_pred, split_name)

        # Feature coefficients
        coefs = dict(zip(feat_cols, model.coef_.tolist()))
        model_results["coefficients"] = coefs
        log("  Coefficients:")
        for f, c in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            bar = "█" * int(abs(c) * 10)
            sign = "+" if c > 0 else "-"
            log(f"    {f:20s}  {c:>+.4f}  {sign}{bar}")

        all_results[name] = model_results
        test_predictions[name] = model.predict(X_test_s)
        log()

    # ── Plots ─────────────────────────────────────────────────────────────
    log("── Diagnostic Plots ──")
    plot_pred_vs_true(all_results, y_test, test_predictions, out_dir)
    plot_residuals(all_results, y_test, test_predictions, out_dir)
    log()

    # ── Save ──────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    all_results["_meta"] = {
        "features": feat_cols,
        "target": TARGET,
        "preprocessing": "log10(mass+1) for stellar_mass, gas_mass; StandardScaler on all",
        "elapsed_seconds": elapsed,
    }

    results_path = out_dir / "linear_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    log(f"  Results: {results_path}")

    log()
    log(f"  Elapsed: {elapsed}s")
    log()
    log("Step 5 complete. ✓")

    (out_dir / "step5_output.txt").write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

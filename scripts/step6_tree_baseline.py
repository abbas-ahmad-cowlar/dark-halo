#!/usr/bin/env python
"""
Step 6  –  Baseline 2: Random Forest / XGBoost
================================================

Trains tree ensemble models with hyperparameter search on
validation set, evaluates on test set, and runs SHAP importance.

Outputs
-------
- artifacts/step6_trees/tree_results.json
- artifacts/step6_trees/pred_vs_true.png
- artifacts/step6_trees/residual_plot.png
- artifacts/step6_trees/shap_importance.png
- artifacts/step6_trees/step6_output.txt

Usage
-----
    python scripts/step6_tree_baseline.py
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ---------------------------------------------------------------------------
_log_lines: list[str] = []
def log(msg: str = "") -> None:
    _log_lines.append(msg)
    print(msg)

TARGET = "log10_M_halo"
RAW_FEATURES = [
    "stellar_mass", "gas_mass", "vel_disp", "sfr",
    "half_mass_rad", "env_dist_5nn",
]

# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 6: RF/XGBoost baseline.")
    p.add_argument("--suite", default="simba")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step6_trees"))
    return p.parse_args()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feat in RAW_FEATURES:
        if feat in ("stellar_mass", "gas_mass"):
            out[f"log_{feat}"] = np.log10(out[feat] + 1)
    return out

def get_feature_names() -> list[str]:
    return [f"log_{f}" if f in ("stellar_mass", "gas_mass") else f for f in RAW_FEATURES]

def evaluate(y_true, y_pred, label):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    log(f"  {label:20s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6)}


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_dir = (args.data_dir or Path(f"data/camels/{args.suite}_LH")).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log("=" * 60)
    log("Step 6: Baseline 2 — Random Forest / XGBoost")
    log("=" * 60)
    log()

    df = pd.read_parquet(data_dir / "dataset_split.parquet")
    df = preprocess(df)
    feat_cols = get_feature_names()
    log(f"  Loaded: {len(df):,} rows   Features: {feat_cols}")
    log()

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    X_train, y_train = train[feat_cols].values, train[TARGET].values
    X_val, y_val = val[feat_cols].values, val[TARGET].values
    X_test, y_test = test[feat_cols].values, test[TARGET].values
    log(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    log()

    all_results: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    best_model = None
    best_model_name = None
    best_r2 = -999

    # ── Random Forest (quick grid on val) ─────────────────────────────────
    log("── Random Forest Hyperparameter Search ──")
    rf_configs = [
        {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 3},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
    ]
    best_rf = None
    best_rf_val_r2 = -999
    best_rf_cfg = None

    for cfg in rf_configs:
        label = f"RF(n={cfg['n_estimators']},d={cfg['max_depth']},ml={cfg['min_samples_leaf']})"
        log(f"  Training {label} ...")
        rf = RandomForestRegressor(**cfg, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_v = rf.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_v)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_v))
        log(f"    Val: RMSE={val_rmse:.4f}  R²={val_r2:.4f}")
        if val_r2 > best_rf_val_r2:
            best_rf_val_r2 = val_r2
            best_rf = rf
            best_rf_cfg = cfg
    log(f"  Best RF config: {best_rf_cfg}")
    log()

    # Evaluate best RF
    log("── Best Random Forest — Full Evaluation ──")
    rf_results: dict[str, Any] = {"config": best_rf_cfg}
    for sname, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        rf_results[sname] = evaluate(y, best_rf.predict(X), sname)
    all_results["RandomForest"] = rf_results
    test_predictions["RandomForest"] = best_rf.predict(X_test)
    if rf_results["val"]["r2"] > best_r2:
        best_r2 = rf_results["val"]["r2"]
        best_model = best_rf
        best_model_name = "RandomForest"

    # Permutation importance from RF
    log("  Feature importances (RF built-in):")
    for f, imp in sorted(zip(feat_cols, best_rf.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 60)
        log(f"    {f:20s}  {imp:.4f}  {bar}")
    log()

    # ── XGBoost ───────────────────────────────────────────────────────────
    if HAS_XGB:
        log("── XGBoost Hyperparameter Search ──")
        xgb_configs = [
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
            {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8},
            {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.9},
        ]
        best_xgb = None
        best_xgb_val_r2 = -999
        best_xgb_cfg = None

        for cfg in xgb_configs:
            label = f"XGB(n={cfg['n_estimators']},d={cfg['max_depth']},lr={cfg['learning_rate']})"
            log(f"  Training {label} ...")
            model = xgb.XGBRegressor(**cfg, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)
            y_pred_v = model.predict(X_val)
            val_r2 = r2_score(y_val, y_pred_v)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_v))
            log(f"    Val: RMSE={val_rmse:.4f}  R²={val_r2:.4f}")
            if val_r2 > best_xgb_val_r2:
                best_xgb_val_r2 = val_r2
                best_xgb = model
                best_xgb_cfg = cfg
        log(f"  Best XGB config: {best_xgb_cfg}")
        log()

        log("── Best XGBoost — Full Evaluation ──")
        xgb_results: dict[str, Any] = {"config": best_xgb_cfg}
        for sname, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            xgb_results[sname] = evaluate(y, best_xgb.predict(X), sname)
        all_results["XGBoost"] = xgb_results
        test_predictions["XGBoost"] = best_xgb.predict(X_test)
        if xgb_results["val"]["r2"] > best_r2:
            best_r2 = xgb_results["val"]["r2"]
            best_model = best_xgb
            best_model_name = "XGBoost"
        log()
    else:
        log("  XGBoost not installed — skipping.")
        log()

    # ── Plots ─────────────────────────────────────────────────────────────
    log("── Diagnostic Plots ──")
    # Pred vs true
    n_models = len(test_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, test_predictions.items()):
        ax.scatter(y_test, y_pred, s=1, alpha=0.2, c="#355C7D")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.7)
        m = all_results[name]["test"]
        ax.set_xlabel("True log₁₀(M_halo)")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nRMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")
    plt.tight_layout()
    fig.savefig(out_dir / "pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: pred_vs_true.png")

    # Residuals
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    for col, (name, y_pred) in enumerate(test_predictions.items()):
        res = y_test - y_pred
        axes[0, col].scatter(y_pred, res, s=1, alpha=0.2, c="#6C5B7B")
        axes[0, col].axhline(0, color="red", linestyle="--")
        axes[0, col].set_xlabel("Predicted"); axes[0, col].set_ylabel("Residual")
        axes[0, col].set_title(f"{name}: Residual vs Predicted")
        axes[1, col].hist(res, bins=80, color="#C06C84", edgecolor="white", alpha=0.8)
        axes[1, col].axvline(0, color="red", linestyle="--")
        axes[1, col].set_xlabel("Residual"); axes[1, col].set_title(f"std={res.std():.4f}")
    plt.tight_layout()
    fig.savefig(out_dir / "residual_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: residual_plot.png")

    # ── SHAP ──────────────────────────────────────────────────────────────
    if HAS_SHAP and best_model is not None:
        log()
        log(f"── SHAP Analysis on {best_model_name} ──")
        # Use a subsample for SHAP efficiency
        shap_sample_idx = np.random.RandomState(42).choice(len(X_test), min(3000, len(X_test)), replace=False)
        X_shap = X_test[shap_sample_idx]

        if best_model_name == "XGBoost":
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap)

        # Mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        log("  Mean |SHAP| values:")
        for f, v in sorted(zip(feat_cols, mean_abs_shap), key=lambda x: -x[1]):
            bar = "█" * int(v * 30)
            log(f"    {f:20s}  {v:.4f}  {bar}")

        # SHAP summary plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_shap, feature_names=feat_cols, show=False)
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(out_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        log(f"  Saved: shap_importance.png")
    else:
        if not HAS_SHAP:
            log("  SHAP not installed — skipping. Install with: pip install shap")
    log()

    # ── Save ──────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    all_results["_meta"] = {
        "features": feat_cols,
        "best_model": best_model_name,
        "elapsed_seconds": elapsed,
    }
    with (out_dir / "tree_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    log(f"  Elapsed: {elapsed}s")
    log()
    log("Step 6 complete. ✓")
    (out_dir / "step6_output.txt").write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

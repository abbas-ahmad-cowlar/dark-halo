#!/usr/bin/env python
"""
Step 9  –  Cross-Simulation Transfer Test
===========================================

Downloads IllustrisTNG LH data, builds the same dataset,
then tests models trained on SIMBA against TNG data.

Pipeline:
1. Download TNG LH data from FlatHUB (reuses Step 1 logic)
2. Build TNG dataset (reuses Step 2 logic)
3. Train models on SIMBA, evaluate on TNG
4. Compare SIMBA-only vs TNG performance

Outputs
-------
- data/camels/tng_LH/groups.parquet, subhalos.parquet, dataset.parquet
- artifacts/step9_transfer/transfer_results.json
- artifacts/step9_transfer/transfer_comparison.png
- artifacts/step9_transfer/step9_output.txt

Usage
-----
    python scripts/step9_transfer.py
    python scripts/step9_transfer.py --skip-download   # if TNG data already exists
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
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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

# FlatHUB API constants (from step1)
FLATHUB_DATA = "https://flathub.flatironinstitute.org/api/camels/data/csv"
SNAPSHOT_Z0 = 33
SUITE_CODE = {"SIMBA": 1, "IllustrisTNG": 0}
SET_CODE = {"LH": 0}
TYPE_CODE = {"Group": 0, "Subhalo": 1}

GROUP_FIELDS = [
    "Group_M_Crit200", "Group_FirstSub", "Group_Nsubs",
    "Group_Mass", "Group_LenType_dm",
    "Group_Pos_x", "Group_Pos_y", "Group_Pos_z",
]
SUBHALO_FIELDS = [
    "Subhalo_id", "Subhalo_GrNr",
    "Subhalo_MassType_gas", "Subhalo_MassType_stars",
    "Subhalo_VelDisp", "Subhalo_SFR", "Subhalo_HalfmassRad",
    "Subhalo_Pos_x", "Subhalo_Pos_y", "Subhalo_Pos_z",
    "Subhalo_Mass",
]

# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Step 9: Cross-simulation transfer.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip TNG download if data already exists.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=49)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step9_transfer"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Download TNG (reusing step1 logic)
# ---------------------------------------------------------------------------
def download_tng(start: int, end: int, data_dir: Path) -> bool:
    """Download IllustrisTNG LH data from FlatHUB."""
    import requests
    from io import StringIO

    data_dir.mkdir(parents=True, exist_ok=True)
    timeout = 600

    for catalog_type, fields in [("Group", GROUP_FIELDS), ("Subhalo", SUBHALO_FIELDS)]:
        log(f"\n  Downloading {catalog_type} catalog...")
        all_dfs = []

        for real_id in range(start, end + 1):
            params = {
                "simulation_suite": SUITE_CODE["IllustrisTNG"],
                "simulation_set": SET_CODE["LH"],
                "simulation_set_id": real_id,
                "snapshot": SNAPSHOT_Z0,
                "type": TYPE_CODE[catalog_type],
                "fields": " ".join(fields),
            }
            try:
                t0 = time.time()
                resp = requests.get(FLATHUB_DATA, params=params, timeout=timeout)
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text))
                df["simulation_set_id"] = real_id
                all_dfs.append(df)
                elapsed = time.time() - t0
                log(f"    [{real_id+1:>3}/{end-start+1}]  {catalog_type} LH_{real_id}"
                    f"  ✓  {len(df)} rows  ({elapsed:.1f}s)")
            except Exception as e:
                log(f"    [{real_id+1:>3}/{end-start+1}]  {catalog_type} LH_{real_id}"
                    f"  ✗  {e}")
                continue

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            fname = "groups.parquet" if catalog_type == "Group" else "subhalos.parquet"
            combined.to_parquet(data_dir / fname, index=False)
            log(f"  Total {catalog_type} rows: {len(combined):,}")
        else:
            log(f"  ERROR: No {catalog_type} data downloaded!")
            return False

    return True


# ---------------------------------------------------------------------------
# Build TNG dataset (reusing step2 logic)
# ---------------------------------------------------------------------------
def build_tng_dataset(data_dir: Path) -> pd.DataFrame:
    log("\n── Building TNG dataset ──")
    groups = pd.read_parquet(data_dir / "groups.parquet")
    subhalos = pd.read_parquet(data_dir / "subhalos.parquet")
    log(f"  Raw: {len(groups):,} groups, {len(subhalos):,} subhalos")

    # Quality filter
    groups = groups[groups["Group_LenType_dm"] >= 100].copy()
    log(f"  After DM>=100 filter: {len(groups):,} groups")

    # Central subhalo merge
    sub_cols = [
        "simulation_set_id", "Subhalo_id", "Subhalo_GrNr",
        "Subhalo_MassType_gas", "Subhalo_MassType_stars",
        "Subhalo_VelDisp", "Subhalo_SFR", "Subhalo_HalfmassRad",
        "Subhalo_Pos_x", "Subhalo_Pos_y", "Subhalo_Pos_z",
    ]
    subs = subhalos[sub_cols].copy()
    merged = groups.merge(
        subs,
        left_on=["simulation_set_id", "Group_FirstSub"],
        right_on=["simulation_set_id", "Subhalo_id"],
        how="inner",
    )
    merged.drop(columns=["Subhalo_id", "Subhalo_GrNr"], inplace=True)
    log(f"  After central merge: {len(merged):,} rows")

    # Unit conversions
    for col in ["Group_M_Crit200", "Group_Mass", "Subhalo_MassType_stars", "Subhalo_MassType_gas"]:
        if col in merged.columns:
            merged[col] = merged[col] * 1e10
    for col in ["Subhalo_Pos_x", "Subhalo_Pos_y", "Subhalo_Pos_z"]:
        if col in merged.columns:
            merged[col] = merged[col] / 1e3

    # Log target
    merged = merged[merged["Group_M_Crit200"] > 0].copy()
    merged["log10_M_halo"] = np.log10(merged["Group_M_Crit200"])

    # Environment proxy
    merged["env_dist_5nn"] = np.nan
    for real_id in merged["simulation_set_id"].unique():
        mask = merged["simulation_set_id"] == real_id
        pos = merged.loc[mask, ["Subhalo_Pos_x", "Subhalo_Pos_y", "Subhalo_Pos_z"]].values
        if len(pos) > 5:
            tree = cKDTree(pos)
            dists, _ = tree.query(pos, k=6)
            merged.loc[mask, "env_dist_5nn"] = dists[:, 5]

    # Rename
    merged = merged.rename(columns={
        "Subhalo_MassType_stars": "stellar_mass",
        "Subhalo_MassType_gas": "gas_mass",
        "Subhalo_VelDisp": "vel_disp",
        "Subhalo_SFR": "sfr",
        "Subhalo_HalfmassRad": "half_mass_rad",
        "Subhalo_Pos_x": "pos_x",
        "Subhalo_Pos_y": "pos_y",
        "Subhalo_Pos_z": "pos_z",
    })

    # Drop invalids
    feat_cols = ["stellar_mass", "gas_mass", "vel_disp", "sfr",
                 "half_mass_rad", "pos_x", "pos_y", "pos_z", "env_dist_5nn"]
    check = feat_cols + ["log10_M_halo"]
    mask = merged[check].apply(lambda s: np.isfinite(s)).all(axis=1)
    merged = merged[mask].copy()
    log(f"  Final TNG dataset: {len(merged):,} rows")

    merged.to_parquet(data_dir / "dataset.parquet", index=False)
    return merged


# ---------------------------------------------------------------------------
# Transfer evaluation
# ---------------------------------------------------------------------------
def preprocess(df):
    out = df.copy()
    for feat in RAW_FEATURES:
        if feat in ("stellar_mass", "gas_mass"):
            out[f"log_{feat}"] = np.log10(out[feat] + 1)
    return out

def get_feature_names():
    return [f"log_{f}" if f in ("stellar_mass", "gas_mass") else f for f in RAW_FEATURES]

def metrics(y_true, y_pred):
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def transfer_evaluation(simba_df: pd.DataFrame, tng_df: pd.DataFrame,
                         out_dir: Path) -> dict:
    log("\n" + "=" * 60)
    log("Transfer Evaluation: SIMBA-trained → TNG-tested")
    log("=" * 60)

    simba_df = preprocess(simba_df)
    tng_df = preprocess(tng_df)
    feat_cols = get_feature_names()

    # SIMBA train data (all of it since TNG is the test)
    simba_train = simba_df[simba_df["split"] == "train"]
    simba_test = simba_df[simba_df["split"] == "test"]

    X_simba_train = simba_train[feat_cols].values
    y_simba_train = simba_train[TARGET].values
    X_simba_test = simba_test[feat_cols].values
    y_simba_test = simba_test[TARGET].values
    X_tng = tng_df[feat_cols].values
    y_tng = tng_df[TARGET].values

    log(f"\n  SIMBA train: {len(simba_train):,}  SIMBA test: {len(simba_test):,}")
    log(f"  TNG test:    {len(tng_df):,}")
    log()

    results: dict[str, dict] = {}

    # Ridge
    scaler = StandardScaler()
    X_st_s = scaler.fit_transform(X_simba_train)
    X_ss_s = scaler.transform(X_simba_test)
    X_tng_s = scaler.transform(X_tng)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_st_s, y_simba_train)
    results["Ridge"] = {
        "simba_test": metrics(y_simba_test, ridge.predict(X_ss_s)),
        "tng_test": metrics(y_tng, ridge.predict(X_tng_s)),
    }

    # RF
    rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_simba_train, y_simba_train)
    results["RandomForest"] = {
        "simba_test": metrics(y_simba_test, rf.predict(X_simba_test)),
        "tng_test": metrics(y_tng, rf.predict(X_tng)),
    }
    tng_pred_rf = rf.predict(X_tng)

    # XGBoost
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6,
                                       learning_rate=0.1, subsample=0.8,
                                       random_state=42, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_simba_train, y_simba_train)
        results["XGBoost"] = {
            "simba_test": metrics(y_simba_test, xgb_model.predict(X_simba_test)),
            "tng_test": metrics(y_tng, xgb_model.predict(X_tng)),
        }

    # Print table
    log(f"  {'Model':20s} │ {'SIMBA R²':>10s} │ {'TNG R²':>10s} │ {'Δ R²':>10s} │ {'SIMBA RMSE':>12s} │ {'TNG RMSE':>12s}")
    log("  " + "─" * 85)
    for model_name, res in results.items():
        sr2 = res["simba_test"]["r2"]
        tr2 = res["tng_test"]["r2"]
        delta = tr2 - sr2
        srmse = res["simba_test"]["rmse"]
        trmse = res["tng_test"]["rmse"]
        log(f"  {model_name:20s} │ {sr2:>10.4f} │ {tr2:>10.4f} │ {delta:>+10.4f} │ {srmse:>12.4f} │ {trmse:>12.4f}")
    log()

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pred vs true: RF on SIMBA test vs TNG
    axes[0].scatter(y_simba_test, rf.predict(X_simba_test), s=1, alpha=0.2, c="#2ECC71", label="SIMBA test")
    axes[0].scatter(y_tng, tng_pred_rf, s=1, alpha=0.2, c="#E74C3C", label="TNG test")
    lims = [min(y_simba_test.min(), y_tng.min()), max(y_simba_test.max(), y_tng.max())]
    axes[0].plot(lims, lims, "k--", alpha=0.5)
    axes[0].set_xlabel("True log₁₀(M_halo)")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("RF: SIMBA-trained → SIMBA vs TNG")
    axes[0].legend(markerscale=5)

    # R² comparison bar chart
    model_names = list(results.keys())
    simba_r2s = [results[m]["simba_test"]["r2"] for m in model_names]
    tng_r2s = [results[m]["tng_test"]["r2"] for m in model_names]
    x_pos = np.arange(len(model_names))
    width = 0.35
    axes[1].bar(x_pos - width/2, simba_r2s, width, label="SIMBA test", color="#2ECC71", alpha=0.8)
    axes[1].bar(x_pos + width/2, tng_r2s, width, label="TNG test", color="#E74C3C", alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=15)
    axes[1].set_ylabel("R²")
    axes[1].set_title("Transfer Degradation: SIMBA → TNG")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_dir / "transfer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: transfer_comparison.png")

    return results


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data_dir_tng = Path("data/camels/tng_LH").resolve()
    data_dir_simba = Path("data/camels/simba_LH").resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log("╔" + "═" * 58 + "╗")
    log("║  Step 9: Cross-Simulation Transfer Test" + " " * 17 + "║")
    log("╚" + "═" * 58 + "╝")
    log()

    # 1. Download TNG
    if args.skip_download and (data_dir_tng / "dataset.parquet").exists():
        log("  Skipping TNG download (--skip-download, data exists).")
        tng_df = pd.read_parquet(data_dir_tng / "dataset.parquet")
    else:
        log("── Downloading IllustrisTNG LH data ──")
        success = download_tng(args.start, args.end, data_dir_tng)
        if not success:
            log("  Download failed. Cannot proceed.")
            (out_dir / "step9_output.txt").write_text("\n".join(_log_lines), encoding="utf-8")
            return
        tng_df = build_tng_dataset(data_dir_tng)

    # 2. Load SIMBA
    simba_df = pd.read_parquet(data_dir_simba / "dataset_split.parquet")
    log(f"\n  SIMBA dataset: {len(simba_df):,} rows")
    log(f"  TNG dataset:   {len(tng_df):,} rows")

    # 3. Transfer evaluation
    results = transfer_evaluation(simba_df, tng_df, out_dir)

    # 4. Save
    elapsed = round(time.time() - t_start, 1)
    all_results = {
        "simba_rows": len(simba_df),
        "tng_rows": len(tng_df),
        "transfer_results": results,
        "elapsed_seconds": elapsed,
    }
    with (out_dir / "transfer_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    log(f"\n  Total elapsed: {elapsed}s")
    log("\nStep 9 complete. ✓")
    log("All steps (1-9) of the Dark Halo pipeline are now complete!")
    (out_dir / "step9_output.txt").write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

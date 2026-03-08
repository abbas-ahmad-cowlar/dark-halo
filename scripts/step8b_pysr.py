#!/usr/bin/env python
"""
Step 8b  –  PySR Symbolic Regression
=====================================

Uses PySR to discover compact analytic formulas that map baryonic
features to dark matter halo mass.  Runs on the top-3 SHAP-ranked
features from Step 6: vel_disp, log_stellar_mass, half_mass_rad.

Outputs
-------
- artifacts/step8_final/pysr_equations.txt     (discovered equations)
- artifacts/step8_final/pysr_results.json      (best-fit metrics)
- artifacts/step8_final/pysr_pareto.png        (complexity vs accuracy)
- artifacts/step8_final/pysr_output.txt        (console log)

Usage
-----
    python scripts/step8b_pysr.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

# ---------------------------------------------------------------------------
_log_lines: list[str] = []
def log(msg: str = "") -> None:
    _log_lines.append(msg)
    print(msg)

TARGET = "log10_M_halo"
# Top-3 features by SHAP importance
TOP_FEATURES = ["vel_disp", "log_stellar_mass", "half_mass_rad"]

def parse_args():
    p = argparse.ArgumentParser(description="Step 8b: PySR Symbolic Regression.")
    p.add_argument("--suite", default="simba")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step8_final"))
    p.add_argument("--niterations", type=int, default=40,
                   help="Number of PySR iterations (higher=better but slower).")
    p.add_argument("--sample-size", type=int, default=10000,
                   help="Subsample size for PySR training (speed).")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = (args.data_dir or Path(f"data/camels/{args.suite}_LH")).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log("=" * 60)
    log("Step 8b: PySR Symbolic Regression")
    log("=" * 60)
    log()

    if not HAS_PYSR:
        log("  ERROR: pysr not installed. Run: pip install pysr")
        log("  Skipping symbolic regression.")
        (out_dir / "pysr_output.txt").write_text("\n".join(_log_lines), encoding="utf-8")
        return

    # Load
    df = pd.read_parquet(data_dir / "dataset_split.parquet")
    df["log_stellar_mass"] = np.log10(df["stellar_mass"] + 1)
    df["log_gas_mass"] = np.log10(df["gas_mass"] + 1)

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    # Subsample training for PySR speed
    if len(train) > args.sample_size:
        train_sub = train.sample(args.sample_size, random_state=42)
    else:
        train_sub = train
    log(f"  Train sample: {len(train_sub):,}  (from {len(train):,})")
    log(f"  Test: {len(test):,}")
    log(f"  Features: {TOP_FEATURES}")
    log()

    X_train = train_sub[TOP_FEATURES].values
    y_train = train_sub[TARGET].values
    X_test = test[TOP_FEATURES].values
    y_test = test[TARGET].values

    # Configure PySR
    model = PySRRegressor(
        niterations=args.niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "square"],
        populations=15,
        population_size=40,
        maxsize=20,
        variable_names=TOP_FEATURES,
        random_state=42,
        parallelism="serial",
        progress=True,
        verbosity=1,
        temp_equation_file=True,
    )

    log("── Running PySR ──")
    log(f"  Iterations: {args.niterations}")
    log(f"  Max equation size: 25 nodes")
    log()

    model.fit(X_train, y_train)

    log()
    log("── PySR Results ──")

    # Get equations
    equations_df = model.equations_
    if equations_df is not None and len(equations_df) > 0:
        log(f"  Found {len(equations_df)} equations on Pareto front")
        log()

        # Print Pareto front
        eq_lines = []
        log(f"  {'#':>3s}  {'Complexity':>10s}  {'Loss':>12s}  {'Score':>8s}  Equation")
        log("  " + "─" * 80)
        for idx, row in equations_df.iterrows():
            line = (f"  {idx:>3d}  {row['complexity']:>10d}  "
                    f"{row['loss']:>12.6f}  {row.get('score', 0):>8.4f}  "
                    f"{row['equation']}")
            log(line)
            eq_lines.append(line)

        # Save equations to file
        eq_path = out_dir / "pysr_equations.txt"
        eq_path.write_text("\n".join(eq_lines), encoding="utf-8")
        log(f"\n  Equations saved: {eq_path}")

        # Best equation evaluation on test (NaN-safe)
        try:
            y_pred = model.predict(X_test)
            # Some equations may produce NaN for edge-case inputs
            valid = np.isfinite(y_pred)
            if valid.sum() < len(y_pred):
                log(f"  Warning: {(~valid).sum()} NaN predictions masked out")
            y_pred_clean = y_pred[valid]
            y_test_clean = y_test[valid]
            rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
            r2 = r2_score(y_test_clean, y_pred_clean)
            log(f"\n  Best equation test:  RMSE={rmse:.4f}  R²={r2:.4f}  ({valid.sum():,}/{len(y_test):,} valid)")
            log(f"  Best equation: {model.sympy()}")
        except Exception as e:
            log(f"  Could not evaluate best equation on test: {e}")
            rmse = float('nan')
            r2 = float('nan')

        # Pareto plot: complexity vs loss
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(equations_df["complexity"], equations_df["loss"],
                "o-", color="#8E44AD", markersize=6)
        ax.set_xlabel("Complexity (# nodes)")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("PySR Pareto Front: Complexity vs Accuracy")
        ax.set_yscale("log")
        plt.tight_layout()
        fig.savefig(out_dir / "pysr_pareto.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: pysr_pareto.png")

        # Save results JSON
        results = {
            "best_equation": str(model.sympy()) if r2 == r2 else "evaluation_failed",
            "best_test_rmse": round(rmse, 6) if rmse == rmse else None,
            "best_test_r2": round(r2, 6) if r2 == r2 else None,
            "n_equations": len(equations_df),
            "features_used": TOP_FEATURES,
            "pareto_front": [
                {"complexity": int(row["complexity"]),
                 "loss": round(float(row["loss"]), 6),
                 "equation": str(row["equation"])}
                for _, row in equations_df.iterrows()
            ],
        }
        with (out_dir / "pysr_results.json").open("w") as f:
            json.dump(results, f, indent=2)
    else:
        log("  No equations found. Try increasing niterations.")

    elapsed = round(time.time() - t_start, 1)
    log(f"\n  Elapsed: {elapsed}s")
    log("\nStep 8b (PySR) complete. ✓")
    (out_dir / "pysr_output.txt").write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

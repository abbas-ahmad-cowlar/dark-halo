#!/usr/bin/env python
"""
Step 7  –  Core MLP Model with MC Dropout
==========================================

Trains a feed-forward MLP to predict log10(M_halo) from baryonic
features.  Uses MC Dropout for uncertainty quantification.

Outputs
-------
- artifacts/step7_mlp/mlp_results.json
- artifacts/step7_mlp/pred_vs_true.png
- artifacts/step7_mlp/residual_plot.png
- artifacts/step7_mlp/training_curve.png
- artifacts/step7_mlp/uncertainty_plot.png
- artifacts/step7_mlp/step7_output.txt

Usage
-----
    python scripts/step7_mlp.py
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
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
    p = argparse.ArgumentParser(description="Step 7: MLP with MC Dropout.")
    p.add_argument("--suite", default="simba")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step7_mlp"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mc-samples", type=int, default=50,
                   help="Number of MC Dropout forward passes for uncertainty.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HaloMLP(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
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
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        n += len(X_batch)
    return total_loss / n

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
            n += len(X_batch)
    return total_loss / n

def predict(model, X_tensor, device):
    model.eval()
    with torch.no_grad():
        return model(X_tensor.to(device)).cpu().numpy()

def mc_dropout_predict(model, X_tensor, device, n_samples=50):
    """MC Dropout: keep dropout active at inference, sample n_samples predictions."""
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor.to(device)).cpu().numpy()
            preds.append(pred)
    preds = np.array(preds)  # (n_samples, n_points)
    return preds.mean(axis=0), preds.std(axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_dir = (args.data_dir or Path(f"data/camels/{args.suite}_LH")).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.time()
    log("=" * 60)
    log("Step 7: Core MLP with MC Dropout")
    log("=" * 60)
    log(f"  Device: {device}")
    log(f"  Architecture: {args.hidden}  dropout={args.dropout}")
    log(f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}")
    log()

    # Load & preprocess
    df = pd.read_parquet(data_dir / "dataset_split.parquet")
    df = preprocess(df)
    feat_cols = get_feature_names()

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[TARGET].values.astype(np.float32)
    X_val = val_df[feat_cols].values.astype(np.float32)
    y_val = val_df[TARGET].values.astype(np.float32)
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_test = test_df[TARGET].values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    log(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    log()

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = HaloMLP(len(feat_cols), args.hidden, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model parameters: {n_params:,}")
    log()

    # ── Training loop ─────────────────────────────────────────────────────
    log("── Training ──")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience_limit = 20

    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  Epoch {epoch:>3}/{args.epochs}  "
                f"train_loss={t_loss:.6f}  val_loss={v_loss:.6f}  "
                f"lr={lr_now:.1e}  best@{best_epoch}")

        if patience_counter >= patience_limit:
            log(f"  Early stopping at epoch {epoch} (patience={patience_limit})")
            break

    log(f"  Best epoch: {best_epoch}  val_loss={best_val_loss:.6f}")
    log()

    # Load best model
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))

    # ── Evaluate ──────────────────────────────────────────────────────────
    log("── Evaluation (best model) ──")
    X_train_t = torch.from_numpy(X_train_s)
    X_val_t = torch.from_numpy(X_val_s)
    X_test_t = torch.from_numpy(X_test_s)

    results: dict[str, Any] = {}
    results["train"] = evaluate(y_train, predict(model, X_train_t, device), "train")
    results["val"] = evaluate(y_val, predict(model, X_val_t, device), "val")
    results["test"] = evaluate(y_test, predict(model, X_test_t, device), "test")
    log()

    y_test_pred = predict(model, X_test_t, device)

    # ── MC Dropout Uncertainty ────────────────────────────────────────────
    log(f"── MC Dropout ({args.mc_samples} samples) ──")
    mc_mean, mc_std = mc_dropout_predict(model, X_test_t, device, args.mc_samples)
    results["mc_dropout"] = {
        "n_samples": args.mc_samples,
        "mean_uncertainty": round(float(mc_std.mean()), 6),
        "median_uncertainty": round(float(np.median(mc_std)), 6),
        "p90_uncertainty": round(float(np.percentile(mc_std, 90)), 6),
    }
    log(f"  Mean uncertainty (std):   {mc_std.mean():.6f}")
    log(f"  Median uncertainty:       {np.median(mc_std):.6f}")
    log(f"  90th percentile:          {np.percentile(mc_std, 90):.6f}")
    log()

    # ── Plots ─────────────────────────────────────────────────────────────
    log("── Plots ──")

    # Training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", color="#355C7D")
    ax.plot(val_losses, label="Val Loss", color="#E74C3C")
    ax.axvline(best_epoch - 1, color="green", linestyle="--", alpha=0.5, label=f"Best @ {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curve")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: training_curve.png")

    # Pred vs true
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_test_pred, s=1, alpha=0.2, c="#355C7D")
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    ax.plot(lims, lims, "r--", alpha=0.7)
    ax.set_xlabel("True log₁₀(M_halo)")
    ax.set_ylabel("Predicted")
    m = results["test"]
    ax.set_title(f"MLP: RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")
    plt.tight_layout()
    fig.savefig(out_dir / "pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: pred_vs_true.png")

    # Residuals
    residuals = y_test - y_test_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test_pred, residuals, s=1, alpha=0.2, c="#6C5B7B")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(residuals, bins=80, color="#C06C84", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual"); axes[1].set_title(f"std={residuals.std():.4f}")
    plt.tight_layout()
    fig.savefig(out_dir / "residual_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: residual_plot.png")

    # Uncertainty plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test, mc_std, s=1, alpha=0.2, c="#8E44AD")
    axes[0].set_xlabel("True log₁₀(M_halo)")
    axes[0].set_ylabel("MC Dropout std")
    axes[0].set_title("Predictive Uncertainty vs Mass")

    # Calibration: error vs uncertainty
    abs_error = np.abs(y_test - mc_mean)
    n_bins = 20
    bin_edges = np.percentile(mc_std, np.linspace(0, 100, n_bins + 1))
    bin_means_unc = []
    bin_means_err = []
    for i in range(n_bins):
        mask = (mc_std >= bin_edges[i]) & (mc_std < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means_unc.append(mc_std[mask].mean())
            bin_means_err.append(abs_error[mask].mean())
    axes[1].scatter(bin_means_unc, bin_means_err, c="#E74C3C", s=30)
    if bin_means_unc:
        lims = [0, max(max(bin_means_unc), max(bin_means_err)) * 1.1]
        axes[1].plot(lims, lims, "k--", alpha=0.5, label="Ideal")
    axes[1].set_xlabel("Mean MC std (uncertainty)")
    axes[1].set_ylabel("Mean |error|")
    axes[1].set_title("Uncertainty Calibration")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / "uncertainty_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: uncertainty_plot.png")
    log()

    # ── Save ──────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    results["_meta"] = {
        "architecture": args.hidden,
        "dropout": args.dropout,
        "epochs_run": len(train_losses),
        "best_epoch": best_epoch,
        "features": feat_cols,
        "device": str(device),
        "n_params": n_params,
        "elapsed_seconds": elapsed,
    }
    with (out_dir / "mlp_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log(f"  Elapsed: {elapsed}s")
    log()
    log("Step 7 complete. ✓")
    (out_dir / "step7_output.txt").write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

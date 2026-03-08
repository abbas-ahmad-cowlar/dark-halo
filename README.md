# Dark Halo — Predicting Dark Matter Halo Mass from Baryonic Galaxy Properties

**Author:** Syed Abbas Ahmad — Department of Physics and Applied Sciences, PIEAS
**Course:** CIS-452 Fundamentals of Machine Learning

An end-to-end ML pipeline that predicts dark matter halo mass (M₂₀₀c) from observable baryonic galaxy features using the [CAMELS](https://www.camel-simulations.org/) cosmological simulation suite.

## Why This Matters

Dark matter halos are invisible gravitational scaffolds that host galaxies, but their mass cannot be measured directly. This project demonstrates that baryonic observables — especially **velocity dispersion** (σᵥ) and **stellar mass** (M★) — can predict halo mass to **R² ≈ 0.98** using tree-based models, recovering the virial theorem and stellar-to-halo mass relation from data alone.

## Results Summary

| Model                   | Test RMSE | Test R²   |
| ----------------------- | --------- | --------- |
| Ridge Regression        | 0.364     | 0.754     |
| **Random Forest**       | **0.104** | **0.980** |
| XGBoost                 | 0.104     | 0.980     |
| MLP (256→128→64)        | 0.108     | 0.978     |
| PySR (analytic formula) | 0.162     | 0.951     |

**Cross-simulation transfer (SIMBA → IllustrisTNG):** RF degrades only ~3% (R² 0.98 → 0.95), confirming the learned mapping generalizes across different galaxy formation physics.

**Best symbolic formula discovered by PySR:**

```
log₁₀(M_halo) ≈ 0.833·log(σᵥ − 3.64) + √(√r + 53.85)
```

## Project Structure

```
dark-halo/
├── scripts/                          # Pipeline scripts (Steps 1-9)
│   ├── step1_download_camels.py      # Download CAMELS LH data from FlatHUB
│   ├── step2_build_dataset.py        # Build modeling-ready dataset
│   ├── step3_split.py                # Leakage-safe train/val/test split
│   ├── step4_eda.py                  # EDA, distributions, Isolation Forest
│   ├── step5_linear_baseline.py      # Linear/Ridge baseline
│   ├── step6_tree_baseline.py        # RF/XGBoost + SHAP analysis
│   ├── step7_mlp.py                  # PyTorch MLP + MC Dropout uncertainty
│   ├── step8_final_evaluation.py     # Unified comparison & scaling analysis
│   ├── step8b_pysr.py                # PySR symbolic regression
│   └── step9_transfer.py             # Cross-simulation transfer (SIMBA→TNG)
├── data/                             # Raw & processed data (gitignored)
│   └── camels/
│       ├── simba_LH/                 # SIMBA 50 realizations
│       └── tng_LH/                   # IllustrisTNG 50 realizations
├── artifacts/                        # Outputs: plots, metrics, reports
│   ├── step1_inventory/              # Download manifests
│   ├── step2_construction/           # Dataset build metadata
│   ├── step3_splits/                 # Split manifest
│   ├── step4_eda/                    # EDA diagnostic plots
│   ├── step5_linear/                 # Linear model diagnostics
│   ├── step6_trees/                  # Tree model + SHAP plots
│   ├── step7_mlp/                    # MLP training curves + uncertainty
│   ├── step8_final/                  # Final comparison + PySR Pareto
│   └── step9_transfer/               # Transfer evaluation plots
├── docs/                             # Report
│   ├── report.tex                    # LaTeX source (30 pages)
│   └── report.pdf                    # Compiled report
├── requirements.txt                  # Python dependencies
└── .gitignore
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download CAMELS SIMBA LH data (50 realizations, ~7 min)
python scripts/step1_download_camels.py

# 3. Build dataset (228K rows, ~30 sec)
python scripts/step2_build_dataset.py

# 4. Create leakage-safe splits (by realization ID)
python scripts/step3_split.py

# 5. Run EDA
python scripts/step4_eda.py

# 6-7. Train models
python scripts/step5_linear_baseline.py
python scripts/step6_tree_baseline.py
python scripts/step7_mlp.py

# 8. Final evaluation + symbolic regression
python scripts/step8_final_evaluation.py
python scripts/step8b_pysr.py

# 9. (Optional) Cross-simulation transfer test
python scripts/step9_transfer.py
```

## Report

A 30-page LaTeX report is included at `docs/report.tex`. To compile:

```bash
cd docs
pdflatex report.tex
pdflatex report.tex   # second pass for cross-references
```

## Data Source

[CAMELS FlatHUB](https://flathub.flatironinstitute.org/) — the publicly available cosmological simulation suite. We use the **SIMBA** and **IllustrisTNG** Latin Hypercube (LH) sets with 50 realizations each at z=0 (snapshot 33).

## Key Design Decisions

- **Leakage-safe splitting**: Split by simulation realization ID (not random rows) to prevent data leakage from halos sharing the same cosmological parameters.
- **Quality filter**: Only halos with ≥100 DM particles are included.
- **Central subhalos only**: Satellite galaxies are excluded via `Group_FirstSub` indexing.
- **Environment feature**: 5th-nearest-neighbor distance computed via `scipy.spatial.cKDTree`.
- **Uncertainty quantification**: MC Dropout (50 forward passes) on the MLP model.

## License

MIT

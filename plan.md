# Project Charter: Predicting Dark Matter Halo Mass from Baryonic Proxies

## 1) Goal and Physics Context

### Project Goal

Build and evaluate a machine learning pipeline that predicts total dark matter halo mass in log-space from baryonic, observationally motivated catalog features.

- **Primary task:** single-target regression
- **Target variable:** `log10(M_halo)` using the closest CAMELS halo-mass definition (priority: `M200c` equivalent)
- **Redshift:** `z = 0` only (snapshot `090` under the updated CAMELS numbering convention; formerly snapshot 033)
- **Data modality:** tabular halo/subhalo catalog data only

### Scientific Motivation

The project tests how well luminous/baryonic proxies (stellar/gas content, kinematics, size, SFR) encode information about the underlying dark matter halo potential. This mirrors real astrophysical inference, where dark matter is not directly observed.

### Strict Scope Boundaries

Out of scope (do not implement):

- 3D map/image/voxel CNN workflows
- temporal or merger-tree modeling
- generative modeling

---

## 2) Dataset Source and Data Dictionaries

## Dataset Decision (Locked)

- **Dataset family:** CAMELS
- **Subset type:** tabular halo/subhalo catalogs (not CMD 2D/3D map products)
- **Suites of interest:** SIMBA and IllustrisTNG (for stretch transfer test)
- **Primary development suite:** SIMBA
- **Snapshot:** `z=0` — uniformly snapshot index `090` across all suites (per 2024 CAMELS renumbering)

### Access and Files

Canonical references:

- CAMELS docs: [https://camels.readthedocs.io/](https://camels.readthedocs.io/)
- CAMELS data portal: [https://www.camel-simulations.org/data](https://www.camel-simulations.org/data)
- Public release context: [https://arxiv.org/abs/2201.01300](https://arxiv.org/abs/2201.01300)

Expected catalog organization includes group/subfind HDF5 catalogs (recent naming in docs uses `groups_###.hdf5`), organized by suite/generation/set/realization.

### Target and Candidate Columns (Planned Dictionary)

Final column names will be confirmed against the exact CAMELS suite version used in Step 1.

#### Target (single)

- **Halo mass target (preferred):** `Group_M_Crit200`
- **Fallbacks if needed:** nearest physically equivalent halo mass definition available in selected files
- **Model target used for training/eval:** `log10(target_mass)`

#### Candidate Input Features (baryonic observable proxies)

Feature family is locked; final exact field names mapped during data audit.

- **Stellar mass proxy:** stellar component mass (e.g., from `SubhaloMassType[..., stars]` or suite-equivalent)
- **Gas mass proxy:** gas component mass (e.g., `SubhaloMassType[..., gas]` or suite-equivalent)
- **Velocity dispersion proxy:** subhalo velocity dispersion field (suite-equivalent)
- **Star formation rate:** subhalo/group SFR field (suite-equivalent)
- **Half-mass radius proxy:** stellar half-mass radius field (suite-equivalent)

#### Linking Logic (Locked)

- **Central subhalos only:** use only the central (most massive) subhalo per FOF group to avoid satellite contamination. This is the standard astrophysical choice.
- **Linking method:** use the `GroupFirstSub` array from the FOF group catalog to identify each group's central subhalo index. Cross-verify using the subhalo's `SubhaloGrNr` field.
- Enforce one consistent unit system and transformation policy before modeling (see Unit Conversions below).

#### Unit Conversions (Locked)

All raw HDF5 fields are converted to a consistent unit system before any downstream use:

- **Masses:** multiply raw HDF5 mass fields by `1e10` → units become `Msun/h`.
- **Velocities:** retain native units (`km/s`). Apply scale-factor division (`v / a`) if the snapshot is not exactly z=0 (for z=0, `a=1`, so no correction needed).
- **Positions:** stored as `ckpc/h` in the catalog. Divide by `1e3` to convert to `Mpc/h` when spatial features (e.g., cKDTree environment proxy) are computed.

### Quality Controls and Filters

- **Minimum particle-count threshold (hard):** `GroupLenType[:, 1] >= 100` (at least 100 dark matter particles per FOF group). Groups below this threshold are excluded.
- Drop invalid/non-physical rows (missing, non-finite, non-positive where log transform is required).
- Record all exclusion criteria in metadata for reproducibility.

### Data Advisory Note

The CAMELS SIMBA 1P and CV sets had a known FOF/Subfind bug affecting global DM and gas properties (now fixed). This project explicitly uses the **LH (Latin Hypercube)** sets, which are unaffected. Nevertheless, the data pipeline enforces sanity checks (particle thresholds, non-finite filtering) regardless.

---

## 3) Model Architecture Plan (Locked Ladder)

## Tier 1: Baseline Linear Model

- **Model:** Linear Regression and/or Ridge Regression
- **Purpose:** lower-bound performance and calibration sanity check

## Tier 2: Tree Ensemble Baseline

- **Model:** Random Forest and/or XGBoost
- **Purpose:** strong non-linear baseline; feature importance and interpretability anchor

## Tier 3: Core Neural Model

- **Model:** MLP (dense feed-forward network)
- **Purpose:** compare neural performance against strong tabular baselines
- **Constraint:** implementation should be modular/portable to GPU runtime later (e.g., Colab) without data-pipeline rewrite

---

## 4) Data Splitting and Leakage Prevention (Hard Rule)

- No random row-wise split.
- Split by **simulation realization/seed and/or parameter-set grouping** so that train/val/test do not share leakage pathways.
- Validation and test sets must be physically disjoint from training realizations.
- Maintain a fixed split manifest file for full reproducibility.

---

## 5) Evaluation and Diagnostics (Locked)

### Metrics

- RMSE
- MAE
- R^2

### Diagnostics

- Predicted vs true scatter plots
- Residual plots (including trend vs target mass)
- Error summary by mass bins (recommended for astrophysical interpretability)

### Interpretability

- SHAP and/or permutation importance on tree model(s)
- Report the top predictive baryonic proxies and physical interpretation caveats

---

## 6) Stretch Goal (Selected)

### Cross-Simulation Transfer Test

Train on SIMBA, test generalization on IllustrisTNG under matched feature/target definitions.

- No new architecture required
- Focus on domain-shift robustness and scientific credibility

---

## 7) Compute, Reproducibility, and Engineering Constraints

- CPU-first for extraction, EDA, and baseline modeling
- MLP training path must be portable to optional GPU environment
- Keep first milestone data volume in local-RAM-friendly range (~1e5 to 3e5 halos/subhalos)
- Preserve deterministic seeds and configuration logs
- Keep all decisions and assumptions synchronized with this `plan.md`

---

## 8) Step-by-Step Roadmap (Execution Order)

## Step 1: Data Access and Inventory

- Download CAMELS LH catalog HDF5 files via Python `requests` from the public URL endpoint
- **Local storage paths:** `./data/camels/simba_LH/` and `./data/camels/tng_LH/`
- **Scale:** target realizations `LH_0` through `LH_49` (~50 realizations), yielding ~100,000+ high-quality central halos
- **Snapshot target:** `groups_090.hdf5` (z=0)
- Enumerate available group/subhalo fields and units
- Lock final target column and feature columns in a data dictionary table

## Step 2: Dataset Construction

- Build a clean tabular dataset with linked subhalo features -> parent halo target
- Apply particle-count and quality filters
- Apply log transform to target; document all transforms
- Spatial feature engineering: use `scipy.spatial.cKDTree` on 3D subhalo positions to compute a local environment proxy (e.g., distance to Nth nearest neighbor)

## Step 3: Leakage-Safe Splits

- Construct train/val/test split by realization/seed (or parameter-set grouping)
- Save split manifest and summary statistics

## Step 4: EDA and Sanity Checks

- Distribution checks for target/features
- Missingness and outlier handling policy
- Correlation scan and baseline feature diagnostics
- Out-of-distribution (OOD) detection: train an Isolation Forest (or equivalent) to flag physically anomalous objects before predictive modeling

## Step 5: Baseline 1 (Linear/Ridge)

- Train and evaluate with locked metrics/plots
- Store artifacts and result table

## Step 6: Baseline 2 (RF/XGBoost)

- Hyperparameter search within local compute budget
- Evaluate and compare to linear baseline
- Run SHAP/permutation importance

## Step 7: Core MLP Model

- Train/evaluate MLP with same split and metric protocol
- Compare against tree baseline and document trade-offs
- Uncertainty quantification via Monte Carlo Dropout: keep dropout active at inference and report predictive mean/std confidence summaries

## Step 8: Final Evaluation Package

- Produce unified comparison table and diagnostic figures
- Write physics interpretation of feature importance and residual behavior
- Symbolic regression (`pysr`) on top SHAP-ranked features to search for compact analytic mappings to halo mass
- Data scaling analysis: train baselines on subset fractions (e.g., 10%, 50%, 100%) and fit/plot RMSE scaling behavior

## Step 9: Stretch Goal (if time permits)

- Train on SIMBA and test on IllustrisTNG
- Quantify transfer degradation/improvement and discuss causes

---

## 9) Definition of Done (Milestone Gates)

### Minimum Success Criteria

- End-to-end pipeline from CAMELS tabular catalogs to evaluated models
- Leakage-safe split demonstrated and reproducible
- At least one tree model and one MLP benchmarked against linear baseline
- Required metrics and diagnostic plots produced
- Interpretability analysis completed for tree model

### Stretch Completion

- Cross-simulation transfer test completed and documented

---

## 10) Finalized Implementation Decisions

These decisions are now locked and active:

1. **CAMELS generation/set paths:** use the standard **LH (Latin Hypercube)** sets from the original public release for both SIMBA and IllustrisTNG.
2. **Feature-name mapping policy:** map exact column keys dynamically during Step 1 exploration, assuming standard Subfind indexing, then enforce a cross-suite standardization wrapper.
3. **Environment/package standard:** use `requirements.txt` with the core stack: `pandas`, `numpy`, `h5py`, `scipy`, `scikit-learn`, `xgboost`, `shap`, `pysr`, `matplotlib`, `seaborn`, `torch`.

---

## 11) Long-Term Plan (Phase 2 & Beyond)

These items are explicitly deferred until the core project is 100% complete.

- Model deployment: wrap the final model in a FastAPI or Streamlit interface for live inference.
- Cosmological parameter conditioning: add `Omega_m` and `sigma_8` as tabular features to model cross-universe rules.
- Multi-task learning: expand the MLP to jointly predict halo mass and concentration.
- Causal machine learning: explore counterfactual scenarios (e.g., reduced gas mass and resulting halo-mass estimate shifts).
- Physics-informed neural networks (PINNs): long-horizon option to encode astrophysical scaling constraints directly in the loss.

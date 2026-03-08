#!/usr/bin/env python
"""Quick validation of Step 1 downloaded data. Writes results to validation_output.txt."""
import pyarrow.parquet as pq
import json
from pathlib import Path

lines = []
def log(msg=""):
    lines.append(msg)
    print(msg)

log("=== STEP 1 VALIDATION ===")
log()

gm = pq.read_metadata("data/camels/simba_LH/groups.parquet")
sm = pq.read_metadata("data/camels/simba_LH/subhalos.parquet")
gs = pq.read_schema("data/camels/simba_LH/groups.parquet")
ss = pq.read_schema("data/camels/simba_LH/subhalos.parquet")

log(f"Groups:   {gm.num_rows:>10,} rows, {gm.num_columns} cols")
log(f"Subhalos: {sm.num_rows:>10,} rows, {sm.num_columns} cols")
log()
log("Group columns:   " + str(gs.names))
log("Subhalo columns: " + str(ss.names))
log()

g_expected = [
    "simulation_set_id", "Group_M_Crit200", "Group_FirstSub",
    "Group_Nsubs", "Group_Mass", "Group_LenType_dm",
    "Group_Pos_x", "Group_Pos_y", "Group_Pos_z",
]
s_expected = [
    "simulation_set_id", "Subhalo_id", "Subhalo_GrNr",
    "Subhalo_MassType_gas", "Subhalo_MassType_stars",
    "Subhalo_VelDisp", "Subhalo_SFR", "Subhalo_HalfmassRad",
    "Subhalo_Pos_x", "Subhalo_Pos_y", "Subhalo_Pos_z", "Subhalo_Mass",
]
g_miss = [c for c in g_expected if c not in gs.names]
s_miss = [c for c in s_expected if c not in ss.names]
log(f"Group missing cols:   {g_miss if g_miss else 'NONE'}")
log(f"Subhalo missing cols: {s_miss if s_miss else 'NONE'}")
log()

gt = pq.read_table(
    "data/camels/simba_LH/groups.parquet",
    columns=["simulation_set_id", "Group_M_Crit200", "Group_LenType_dm"],
).slice(0, 5).to_pydict()
log("First 5 Group rows:")
for i in range(5):
    log(f"  real={gt['simulation_set_id'][i]}  M_Crit200={gt['Group_M_Crit200'][i]:.4f}  DM_particles={gt['Group_LenType_dm'][i]}")
log()

st = pq.read_table(
    "data/camels/simba_LH/subhalos.parquet",
    columns=["simulation_set_id", "Subhalo_id", "Subhalo_GrNr",
             "Subhalo_MassType_stars", "Subhalo_VelDisp"],
).slice(0, 5).to_pydict()
log("First 5 Subhalo rows:")
for i in range(5):
    log(f"  real={st['simulation_set_id'][i]}  id={st['Subhalo_id'][i]}  GrNr={st['Subhalo_GrNr'][i]}  M_stars={st['Subhalo_MassType_stars'][i]:.6f}  VelDisp={st['Subhalo_VelDisp'][i]:.2f}")
log()

with open("artifacts/step1_inventory/simba_lh_manifest.json") as f:
    manifest = json.load(f)
log(f"Manifest total_groups:   {manifest['total_groups']:>10,}")
log(f"Manifest total_subhalos: {manifest['total_subhalos']:>10,}")
match = manifest["total_groups"] == gm.num_rows and manifest["total_subhalos"] == sm.num_rows
log(f"Parquet matches manifest: {'YES' if match else 'NO'}")
log()
log("=== VALIDATION PASSED ===" if not g_miss and not s_miss and match else "=== VALIDATION FAILED ===")

Path("artifacts/step1_inventory/validation_output.txt").write_text("\n".join(lines), encoding="utf-8")

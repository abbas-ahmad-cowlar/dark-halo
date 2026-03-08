#!/usr/bin/env python
"""
Step 1  –  CAMELS LH Catalog Download via FlatHUB API
=====================================================

Downloads FOF Group and Subfind Subhalo catalog data for SIMBA (or
IllustrisTNG) Latin-Hypercube realizations from the FlatHUB REST API at
https://flathub.flatironinstitute.org.

The primary HTTP file-server (users.flatironinstitute.org) may be
intermittently unavailable; FlatHUB is an alternative that serves the
same underlying Subfind catalog data via a queryable REST interface.

Outputs
-------
- Parquet files:
    data/camels/{suite}_LH/groups.parquet      (all Group/halo rows)
    data/camels/{suite}_LH/subhalos.parquet    (all Subhalo rows)
- JSON manifest:
    artifacts/step1_inventory/{suite}_lh_manifest.json
- Field lists:
    artifacts/step1_inventory/{suite}_lh_group_fields.txt
    artifacts/step1_inventory/{suite}_lh_subhalo_fields.txt

Usage
-----
    python scripts/step1_download_camels.py                      # SIMBA LH_0..LH_49
    python scripts/step1_download_camels.py --suite IllustrisTNG  # TNG LH_0..LH_49
    python scripts/step1_download_camels.py --start 0 --end 99   # wider range
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FLATHUB_API = "https://flathub.flatironinstitute.org/api/camels"
FLATHUB_DATA = f"{FLATHUB_API}/data/csv"

# FlatHUB uses integer codes for enum-like filters
SUITE_CODE = {"SIMBA": 1, "IllustrisTNG": 0}
SET_CODE = {"LH": 0, "1P": 1, "CV": 2, "EX": 3}
TYPE_CODE = {"Group": 0, "Subhalo": 1}

# FlatHUB still uses old snapshot numbering: z=0 → snapshot 33
SNAPSHOT_Z0 = 33

# Fields we need from each catalog type.
# Multi-dimensional HDF5 fields are decomposed in FlatHUB
# e.g.  SubhaloMassType → Subhalo_MassType_gas, _dm, _x2, _x3, _stars, _bh
GROUP_FIELDS = [
    "simulation_set_id",          # LH realization index (0..999)
    "Group_M_Crit200",            # target: halo mass (1e10 Msun/h)
    "Group_FirstSub",             # index of central subhalo
    "Group_Nsubs",                # number of subhalos in group
    "Group_Mass",                 # total FoF mass
    "Group_LenType_dm",           # DM particle count (for quality cut)
    "Group_Pos_x",                # group position x
    "Group_Pos_y",                # group position y
    "Group_Pos_z",                # group position z
]

SUBHALO_FIELDS = [
    "simulation_set_id",
    "Subhalo_id",                 # subhalo index within realization
    "Subhalo_GrNr",               # parent group index (linking key)
    "Subhalo_MassType_gas",       # gas mass
    "Subhalo_MassType_stars",     # stellar mass
    "Subhalo_VelDisp",            # velocity dispersion
    "Subhalo_SFR",                # star formation rate
    "Subhalo_HalfmassRad",        # half-mass radius
    "Subhalo_Pos_x",              # position x  (for cKDTree)
    "Subhalo_Pos_y",              # position y
    "Subhalo_Pos_z",              # position z
    "Subhalo_Mass",               # total subhalo mass
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CAMELS LH catalogs via FlatHUB API.",
    )
    p.add_argument("--suite", default="SIMBA", choices=list(SUITE_CODE),
                   help="Simulation suite (default: SIMBA).")
    p.add_argument("--start", type=int, default=0,
                   help="First LH realization index (inclusive, default: 0).")
    p.add_argument("--end", type=int, default=49,
                   help="Last LH realization index (inclusive, default: 49).")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Override base data directory.")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/step1_inventory"),
                   help="Directory for inventory output files.")
    p.add_argument("--timeout", type=int, default=600,
                   help="HTTP timeout per query in seconds (default: 600).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of realizations per API request (default: 1).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FlatHUB query helpers
# ---------------------------------------------------------------------------
def _query_flathub_csv(
    suite: str,
    obj_type: str,
    fields: list[str],
    realization_ids: list[int],
    timeout: int = 300,
) -> pd.DataFrame:
    """
    Query the FlatHUB CSV endpoint for one or more LH realizations.

    Parameters
    ----------
    suite : str          "SIMBA" or "IllustrisTNG"
    obj_type : str       "Group" or "Subhalo"
    fields : list[str]   FlatHUB column names to fetch
    realization_ids : list[int]  LH realization indices
    timeout : int        HTTP timeout

    Returns
    -------
    pd.DataFrame with the requested columns
    """
    params: dict[str, Any] = {
        "simulation_suite": SUITE_CODE[suite],
        "simulation_set": SET_CODE["LH"],
        "snapshot": SNAPSHOT_Z0,
        "type": TYPE_CODE[obj_type],
        "fields": " ".join(fields),
        "limit": 0,   # no limit — return all rows
    }

    # If specific realizations requested, add as filter
    # FlatHUB accepts comma-separated ranges for simulation_set_id
    if len(realization_ids) == 1:
        params["simulation_set_id"] = realization_ids[0]
    else:
        params["simulation_set_id"] = ",".join(str(i) for i in realization_ids)

    resp = requests.get(FLATHUB_DATA, params=params, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    return df


def download_catalog(
    suite: str,
    obj_type: str,
    fields: list[str],
    start: int,
    end: int,
    batch_size: int = 10,
    timeout: int = 300,
) -> pd.DataFrame:
    """
    Download a full catalog in batches of realizations.
    Returns a concatenated DataFrame.
    """
    all_ids = list(range(start, end + 1))
    total = len(all_ids)
    frames: list[pd.DataFrame] = []

    for batch_start in range(0, total, batch_size):
        batch_ids = all_ids[batch_start : batch_start + batch_size]
        tag = f"  [{batch_start + 1:>4}..{min(batch_start + batch_size, total):>4}/{total}]"

        label = f"LH_{batch_ids[0]}" if len(batch_ids) == 1 else f"LH_{batch_ids[0]}..LH_{batch_ids[-1]}"
        print(f"{tag}  {obj_type} {label} …", end="", flush=True)

        retries = 3
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                df = _query_flathub_csv(
                    suite, obj_type, fields, batch_ids, timeout=timeout,
                )
                elapsed = time.time() - t0
                frames.append(df)
                print(f"  ✓ {len(df):>6} rows  ({elapsed:.1f}s)")
                break
            except (requests.RequestException, pd.errors.EmptyDataError) as exc:
                print(f"\n    [attempt {attempt}/{retries}] {exc}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    print(f"    ✗ FAILED batch {label}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def build_manifest(
    suite: str,
    start: int,
    end: int,
    groups_df: pd.DataFrame,
    subhalos_df: pd.DataFrame,
    group_fields: list[str],
    subhalo_fields: list[str],
) -> dict[str, Any]:

    manifest: dict[str, Any] = {
        "suite": suite,
        "source": "FlatHUB API",
        "snapshot": SNAPSHOT_Z0,
        "snapshot_note": "FlatHUB uses old numbering; snapshot 33 = z=0",
        "set": "LH",
        "realization_range": [start, end],
        "total_groups": len(groups_df),
        "total_subhalos": len(subhalos_df),
        "group_fields": group_fields,
        "subhalo_fields": subhalo_fields,
        "column_mapping": {
            "target_mass": "Group_M_Crit200  (× 1e10 → Msun/h)",
            "group_first_sub": "Group_FirstSub",
            "group_dm_particles": "Group_LenType_dm  (quality cut: ≥ 100)",
            "stellar_mass_proxy": "Subhalo_MassType_stars  (× 1e10 → Msun/h)",
            "gas_mass_proxy": "Subhalo_MassType_gas  (× 1e10 → Msun/h)",
            "velocity_dispersion": "Subhalo_VelDisp  (km/s)",
            "star_formation_rate": "Subhalo_SFR  (Msun/yr)",
            "half_mass_radius": "Subhalo_HalfmassRad  (ckpc/h)",
            "subhalo_position": "Subhalo_Pos_{x,y,z}  (ckpc/h; ÷1e3 → Mpc/h)",
            "linking_key": "Subhalo_GrNr → Group index within realization",
        },
        "unit_notes": {
            "masses": "raw × 1e10 → Msun/h",
            "positions": "ckpc/h  (÷1e3 → Mpc/h)",
            "velocities": "km/s  (÷a if z≠0; a=1 at z=0)",
        },
    }

    if not groups_df.empty:
        # Per-realization stats
        per_real = groups_df.groupby("simulation_set_id").size()
        manifest["groups_per_realization"] = {
            "min": int(per_real.min()),
            "max": int(per_real.max()),
            "mean": round(float(per_real.mean()), 1),
        }

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    suite_tag = args.suite.lower()
    if suite_tag == "illustristng":
        suite_tag = "tng"

    data_dir = (args.data_dir or Path(f"data/camels/{suite_tag}_LH")).resolve()
    out_dir = args.out_dir.resolve()

    print(f"=== Step 1: Download CAMELS {args.suite} LH catalogs via FlatHUB ===")
    print(f"    Realizations : LH_{args.start} .. LH_{args.end}")
    print(f"    Data dir     : {data_dir}")
    print(f"    Inventory dir: {out_dir}")
    print()

    # ── Download Groups ───────────────────────────────────────────────────
    print("── Downloading Group (halo) catalog ──")
    groups_df = download_catalog(
        args.suite, "Group", GROUP_FIELDS,
        args.start, args.end,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    print(f"   Total Group rows: {len(groups_df)}")
    print()

    # ── Download Subhalos ─────────────────────────────────────────────────
    print("── Downloading Subhalo catalog ──")
    subhalos_df = download_catalog(
        args.suite, "Subhalo", SUBHALO_FIELDS,
        args.start, args.end,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    print(f"   Total Subhalo rows: {len(subhalos_df)}")
    print()

    if groups_df.empty or subhalos_df.empty:
        print("ERROR: Failed to download data.", file=sys.stderr)
        sys.exit(1)

    # ── Save Parquet ──────────────────────────────────────────────────────
    data_dir.mkdir(parents=True, exist_ok=True)
    groups_path = data_dir / "groups.parquet"
    subhalos_path = data_dir / "subhalos.parquet"
    groups_df.to_parquet(groups_path, index=False)
    subhalos_df.to_parquet(subhalos_path, index=False)
    print(f"  Saved: {groups_path}  ({groups_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Saved: {subhalos_path}  ({subhalos_path.stat().st_size / 1e6:.1f} MB)")
    print()

    # ── Write inventory artifacts ─────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        args.suite, args.start, args.end,
        groups_df, subhalos_df,
        GROUP_FIELDS, SUBHALO_FIELDS,
    )

    manifest_path = out_dir / f"{suite_tag}_lh_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    gf_path = out_dir / f"{suite_tag}_lh_group_fields.txt"
    with gf_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(GROUP_FIELDS) + "\n")

    sf_path = out_dir / f"{suite_tag}_lh_subhalo_fields.txt"
    with sf_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(SUBHALO_FIELDS) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print("─── Inventory Summary ───")
    print(f"  Suite             : {args.suite}")
    print(f"  Realizations      : LH_{args.start} .. LH_{args.end}")
    print(f"  Snapshot          : {SNAPSHOT_Z0}  (z ≈ 0)")
    print(f"  Total Groups      : {len(groups_df)}")
    print(f"  Total Subhalos    : {len(subhalos_df)}")
    print(f"  Group fields      : {len(GROUP_FIELDS)}")
    print(f"  Subhalo fields    : {len(SUBHALO_FIELDS)}")
    print()
    print("  Column mapping (locked):")
    for role, field in manifest["column_mapping"].items():
        print(f"    {role:30s} → {field}")
    print()
    print(f"  Manifest  : {manifest_path}")
    print(f"  Groups    : {groups_path}")
    print(f"  Subhalos  : {subhalos_path}")
    print()
    print("Step 1 complete. ✓")


if __name__ == "__main__":
    main()

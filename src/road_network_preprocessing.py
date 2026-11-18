"""
road_network_preprocessing.py
==============================
Amrithaa – Road network data processing
Handles .nodes and .arcs files for all trips (train + test).

Cleaning rules applied
-----------------------
Nodes
  1. Invalid coordinates  : lon not in [-180, 180] or lat not in [-90, 90]
                            → flagged and dropped from the cleaned set.
  2. Duplicate node rows  : exact (lon, lat) duplicates
                            → first occurrence kept, rest dropped.
                            Node IDs in arc references are remapped to the
                            surviving representative.

Arcs
  1. Invalid endpoints    : source or target index >= n_nodes or < 0
                            → dropped; logged in cleaning summary.
  2. Self-loops           : source == target
                            → dropped (zero-length, non-traversable).
  3. Duplicate arcs       : identical (source, target) pairs after the
                            steps above → first occurrence kept.

Arc lengths
  Computed in metres using pyproj (EPSG:32640, UTM zone 40N, which covers
  the Tatarstan / Kazan region where the data sit).  If pyproj is not
  available the column is left as NaN.

Outputs per trip  (written to  <out_dir>/<trip_id>/)
  <trip_id>_nodes_clean.csv      – validated node table with original node_id
  <trip_id>_arcs_clean.csv       – validated arc table with arc_id & length_m
  <trip_id>_cleaning_log.csv     – per-trip cleaning counts

Aggregate outputs  (written to  <out_dir>/)
  gps_cleaning_summary.csv       – one row per trip, all cleaning counts
  arc_length_summary.csv         – per-trip arc-length statistics
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── optional pyproj import ────────────────────────────────────────────────────
try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False
    print("[WARN] pyproj not found – arc lengths will be NaN.")

# ─────────────────────────────────────────────────────────────────────────────
# CRS: EPSG:4326 (WGS-84) → EPSG:32640 (UTM zone 40N)
# Justified by Harshitha's CRS analysis; Kazan sits in zone 40N.
# ─────────────────────────────────────────────────────────────────────────────
_CRS_SRC = "EPSG:4326"
_CRS_DST = "EPSG:32640"

if _HAS_PYPROJ:
    _TRANSFORMER = Transformer.from_crs(_CRS_SRC, _CRS_DST, always_xy=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_nodes(path: Path) -> pd.DataFrame:
    """
    Read a .nodes file.

    The file has two whitespace-separated columns with no header.
    Based on observed coordinate ranges (~54 lon, ~56 lat for Kazan) the
    first column is longitude and the second is latitude.

    Returns
    -------
    DataFrame with columns: node_id (int), lon (float), lat (float)
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"])
    df.insert(0, "node_id", range(len(df)))
    return df


def load_arcs(path: Path) -> pd.DataFrame:
    """
    Read a .arcs file.

    The file has two whitespace-separated integer columns with no header:
    source node index and target node index (0-based).

    Returns
    -------
    DataFrame with columns: arc_id (int), source (int), target (int)
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["source", "target"],
                     dtype=int)
    df.insert(0, "arc_id", range(len(df)))
    return df


def haversine_m(lon1: np.ndarray, lat1: np.ndarray,
                lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Fallback: haversine distance in metres (used when pyproj absent).
    Accurate to < 0.5 % for distances < 100 km.
    """
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_arc_lengths(nodes_clean: pd.DataFrame,
                        arcs_clean: pd.DataFrame) -> np.ndarray:
    """
    Return arc lengths in metres for every row of *arcs_clean*.
    Uses pyproj UTM projection when available, haversine otherwise.
    """
    src_lon = nodes_clean.loc[arcs_clean["source"].values, "lon"].values
    src_lat = nodes_clean.loc[arcs_clean["source"].values, "lat"].values
    tgt_lon = nodes_clean.loc[arcs_clean["target"].values, "lon"].values
    tgt_lat = nodes_clean.loc[arcs_clean["target"].values, "lat"].values

    if _HAS_PYPROJ:
        x1, y1 = _TRANSFORMER.transform(src_lon, src_lat)
        x2, y2 = _TRANSFORMER.transform(tgt_lon, tgt_lat)
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    else:
        return haversine_m(src_lon, src_lat, tgt_lon, tgt_lat)


# ─────────────────────────────────────────────────────────────────────────────
# Core cleaning function
# ─────────────────────────────────────────────────────────────────────────────

def clean_trip_network(
    trip_id: str,
    nodes_raw: pd.DataFrame,
    arcs_raw: pd.DataFrame,
) -> dict:
    """
    Apply all cleaning rules to a single trip's nodes and arcs.

    Returns a dict with keys:
      nodes_clean, arcs_clean, log (Series of counts), arc_len_stats (Series)
    """
    log: dict = {"trip_id": trip_id}

    # ── 1. Node coordinate validation ────────────────────────────────────────
    valid_lon = nodes_raw["lon"].between(-180.0, 180.0)
    valid_lat = nodes_raw["lat"].between(-90.0, 90.0)
    coord_mask = valid_lon & valid_lat

    log["n_nodes_raw"] = len(nodes_raw)
    log["n_invalid_coords"] = int((~coord_mask).sum())

    nodes_valid = nodes_raw[coord_mask].copy()

    # ── 2. Duplicate node rows ────────────────────────────────────────────────
    # Build a remap table: duplicate node IDs → their representative ID.
    # Steps: sort by node_id, mark duplicates on (lon, lat), keep first.
    dup_mask = nodes_valid.duplicated(subset=["lon", "lat"], keep="first")
    log["n_duplicate_nodes"] = int(dup_mask.sum())

    # For each duplicate, find which earlier node_id it duplicates.
    # We build a lookup: old_node_id → canonical_node_id
    nodes_dedup = nodes_valid[~dup_mask].copy()
    nodes_dedup = nodes_dedup.reset_index(drop=True)
    # re-index node_ids in a fresh integer sequence
    nodes_dedup["node_id"] = range(len(nodes_dedup))

    # Map original node_id → new node_id via (lon, lat) key
    coord_to_new_id: dict[tuple, int] = {
        (row.lon, row.lat): row.node_id
        for row in nodes_dedup.itertuples(index=False)
    }
    old_to_new: dict[int, int] = {}
    for row in nodes_valid.itertuples(index=False):
        key = (row.lon, row.lat)
        if key in coord_to_new_id:
            old_to_new[row.node_id] = coord_to_new_id[key]

    log["n_nodes_clean"] = len(nodes_dedup)

    # ── 3. Arc endpoint validation ────────────────────────────────────────────
    # Remap source/target through old_to_new; drop arcs whose endpoints
    # were removed (invalid coords or unmapped).
    arcs_work = arcs_raw.copy()
    arcs_work["source"] = arcs_work["source"].map(old_to_new)
    arcs_work["target"] = arcs_work["target"].map(old_to_new)

    n_invalid_ep = int(arcs_work["source"].isna().sum() +
                       arcs_work["target"].isna().sum())
    # Count uniquely invalid arcs (NaN in either column)
    invalid_ep_mask = arcs_work["source"].isna() | arcs_work["target"].isna()
    log["n_arcs_raw"] = len(arcs_work)
    log["n_invalid_endpoints"] = int(invalid_ep_mask.sum())

    arcs_work = arcs_work[~invalid_ep_mask].copy()
    arcs_work["source"] = arcs_work["source"].astype(int)
    arcs_work["target"] = arcs_work["target"].astype(int)

    # ── 4. Self-loops ─────────────────────────────────────────────────────────
    self_loop_mask = arcs_work["source"] == arcs_work["target"]
    log["n_self_loops"] = int(self_loop_mask.sum())
    arcs_work = arcs_work[~self_loop_mask].copy()

    # ── 5. Duplicate arcs ─────────────────────────────────────────────────────
    dup_arc_mask = arcs_work.duplicated(subset=["source", "target"], keep="first")
    log["n_duplicate_arcs"] = int(dup_arc_mask.sum())
    arcs_clean = arcs_work[~dup_arc_mask].copy()
    arcs_clean = arcs_clean.reset_index(drop=True)
    arcs_clean["arc_id"] = range(len(arcs_clean))

    log["n_arcs_clean"] = len(arcs_clean)

    # ── 6. Arc lengths ────────────────────────────────────────────────────────
    # Build an index on node_id for fast lookup
    nodes_indexed = nodes_dedup.set_index("node_id")
    lengths = compute_arc_lengths(nodes_indexed, arcs_clean)
    arcs_clean["length_m"] = lengths

    arc_len_stats = {
        "trip_id": trip_id,
        "mean_length_m": float(np.nanmean(lengths)),
        "median_length_m": float(np.nanmedian(lengths)),
        "min_length_m": float(np.nanmin(lengths)) if len(lengths) else np.nan,
        "max_length_m": float(np.nanmax(lengths)) if len(lengths) else np.nan,
        "p95_length_m": float(np.nanpercentile(lengths, 95)) if len(lengths) else np.nan,
        "total_length_km": float(np.nansum(lengths) / 1000),
        "n_zero_length": int((lengths == 0).sum()),
    }

    return {
        "nodes_clean": nodes_dedup,
        "arcs_clean": arcs_clean,
        "log": log,
        "arc_len_stats": arc_len_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_all_trips(
    data_root: Path,
    out_dir: Path,
    write_per_trip: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterate over every trip folder under *data_root* (train + test),
    clean each network, and write per-trip and aggregate outputs.

    Parameters
    ----------
    data_root   : root that contains 'train/' and/or 'student_test/' sub-dirs
    out_dir     : where to write outputs
    write_per_trip : if True, write per-trip cleaned CSV files

    Returns
    -------
    cleaning_summary_df, arc_length_summary_df
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaning_rows: list[dict] = []
    arc_len_rows: list[dict] = []

    # Collect all trip directories across splits
    split_dirs = []
    for split in ("train", "student_test"):
        sd = data_root / split
        if sd.exists():
            split_dirs.append(sd)

    if not split_dirs:
        raise FileNotFoundError(
            f"No 'train' or 'student_test' directories found under {data_root}"
        )

    for split_dir in split_dirs:
        trip_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for trip_dir in trip_dirs:
            trip_id = trip_dir.name

            nodes_path = trip_dir / f"{trip_id}.nodes"
            arcs_path = trip_dir / f"{trip_id}.arcs"

            if not nodes_path.exists() or not arcs_path.exists():
                print(f"[SKIP] {trip_id}: missing .nodes or .arcs")
                continue

            try:
                nodes_raw = load_nodes(nodes_path)
                arcs_raw = load_arcs(arcs_path)
                result = clean_trip_network(trip_id, nodes_raw, arcs_raw)
            except Exception as exc:
                print(f"[ERROR] {trip_id}: {exc}")
                continue

            if write_per_trip:
                trip_out = out_dir / trip_id
                trip_out.mkdir(parents=True, exist_ok=True)

                result["nodes_clean"].to_csv(
                    trip_out / f"{trip_id}_nodes_clean.csv", index=False
                )
                result["arcs_clean"].to_csv(
                    trip_out / f"{trip_id}_arcs_clean.csv", index=False
                )
                log_df = pd.DataFrame([result["log"]])
                log_df.to_csv(
                    trip_out / f"{trip_id}_cleaning_log.csv", index=False
                )

            cleaning_rows.append(result["log"])
            arc_len_rows.append(result["arc_len_stats"])
            print(f"[OK] {trip_id}  nodes {result['log']['n_nodes_clean']}/"
                  f"{result['log']['n_nodes_raw']}  arcs "
                  f"{result['log']['n_arcs_clean']}/{result['log']['n_arcs_raw']}")

    cleaning_df = pd.DataFrame(cleaning_rows)
    arc_len_df = pd.DataFrame(arc_len_rows)

    cleaning_df.to_csv(out_dir / "gps_cleaning_summary.csv", index=False)
    arc_len_df.to_csv(out_dir / "arc_length_summary.csv", index=False)

    print(f"\n[DONE] Processed {len(cleaning_df)} trips.")
    print(f"  gps_cleaning_summary.csv  → {out_dir}")
    print(f"  arc_length_summary.csv    → {out_dir}")

    return cleaning_df, arc_len_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Paths relative to the repo root (run from PathSnap-ML/).
    DATA_ROOT = Path("data/Student folder/Student folder")
    OUT_DIR = Path("results/network_cleaned")

    cleaning_summary, arc_len_summary = process_all_trips(
        data_root=DATA_ROOT,
        out_dir=OUT_DIR,
        write_per_trip=True,
    )

    print("\n── Cleaning summary (first 5 rows) ──")
    print(cleaning_summary.head().to_string(index=False))

    print("\n── Arc-length summary (first 5 rows) ──")
    print(arc_len_summary.head().to_string(index=False))

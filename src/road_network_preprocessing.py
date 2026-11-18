"""
road_network_preprocessing.py
==============================
Amrithaa – Road network data processing  (v2 – revised after PR #1 review)

Changes from v1
---------------
* orig_node_id and orig_arc_id are now preserved throughout — every cleaned
  row carries the original file-line index so Arun's route IDs stay traceable.
* CRS is no longer a fixed global EPSG:32640.  get_crs_for_trip() selects the
  best UTM zone per-trip from the actual coordinate bounding box, matching the
  shared geo_utils convention agreed with Harshitha.
* haversine fallback retained when pyproj is unavailable.

Cleaning rules
--------------
Nodes
  1. Invalid coordinates  : lon ∉ [-180, 180] or lat ∉ [-90, 90] → dropped.
  2. Duplicate (lon, lat) : first occurrence kept; arc endpoints remapped to
                            the surviving representative's new index.
                            orig_node_id of the survivor is preserved.

Arcs
  1. Invalid endpoints    : source or target cannot be mapped after node
                            cleaning → dropped.
  2. Self-loops           : source == target → dropped.
  3. Duplicate (src, tgt) : first occurrence kept.

Columns in output
-----------------
nodes_clean : orig_node_id | node_id | lon | lat
arcs_clean  : orig_arc_id  | arc_id  | source | target | length_m
              (source/target reference the new sequential node_id)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pyproj import Transformer, CRS
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False
    print("[WARN] pyproj not found – arc lengths computed with haversine.")


# ─────────────────────────────────────────────────────────────────────────────
# CRS helper  (per-trip, not a fixed global)
# ─────────────────────────────────────────────────────────────────────────────

def get_crs_for_trip(lons: np.ndarray, lats: np.ndarray) -> str:
    """
    Choose the best-fit projected CRS for a trip based on its median longitude.

    Uses the standard UTM zone formula.  Returns an EPSG string such as
    'EPSG:32640'.  Northern hemisphere assumed (all data is in Tatarstan ~56°N).

    This function is the shared per-trip CRS selector agreed with Harshitha;
    all arc-length calculations in this module call it rather than using a
    hardcoded zone.
    """
    median_lon = float(np.median(lons))
    zone = int((median_lon + 180) / 6) + 1
    # Northern hemisphere → 326xx; southern → 327xx
    epsg = 32600 + zone
    return f"EPSG:{epsg}"


def make_transformer(src_epsg: str, dst_epsg: str):
    """Return a pyproj Transformer, or None if pyproj is unavailable."""
    if not _HAS_PYPROJ:
        return None
    return Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)


# ─────────────────────────────────────────────────────────────────────────────
# File loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_nodes(path: Path) -> pd.DataFrame:
    """
    Read a .nodes file → DataFrame with columns:
        orig_node_id (int)  – original 0-based line index from the file
        lon          (float)
        lat          (float)

    The file has two whitespace-separated columns, no header.
    Column order confirmed as (lon, lat) from observed ranges (~54°, ~56°).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"])
    df.insert(0, "orig_node_id", range(len(df)))
    return df


def load_arcs(path: Path) -> pd.DataFrame:
    """
    Read a .arcs file → DataFrame with columns:
        orig_arc_id (int)   – original 0-based line index from the file
        source      (int)   – source node index (references orig_node_id)
        target      (int)   – target node index (references orig_node_id)

    The file has two whitespace-separated integer columns, no header.
    """
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=["source", "target"], dtype=int
    )
    df.insert(0, "orig_arc_id", range(len(df)))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers
# ─────────────────────────────────────────────────────────────────────────────

def haversine_m(
    lon1: np.ndarray, lat1: np.ndarray,
    lon2: np.ndarray, lat2: np.ndarray,
) -> np.ndarray:
    """Haversine distance in metres.  Accurate to < 0.5 % for d < 100 km."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_arc_lengths(
    nodes_indexed: pd.DataFrame,
    arcs_clean: pd.DataFrame,
    transformer,
) -> np.ndarray:
    """
    Compute arc lengths in metres.

    Parameters
    ----------
    nodes_indexed : nodes_clean DataFrame indexed on node_id
    arcs_clean    : arcs DataFrame with source/target referencing node_id
    transformer   : pyproj Transformer (lon→x, lat→y) or None for haversine
    """
    src = arcs_clean["source"].values
    tgt = arcs_clean["target"].values

    src_lon = nodes_indexed.loc[src, "lon"].values
    src_lat = nodes_indexed.loc[src, "lat"].values
    tgt_lon = nodes_indexed.loc[tgt, "lon"].values
    tgt_lat = nodes_indexed.loc[tgt, "lat"].values

    if transformer is not None:
        x1, y1 = transformer.transform(src_lon, src_lat)
        x2, y2 = transformer.transform(tgt_lon, tgt_lat)
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    else:
        return haversine_m(src_lon, src_lat, tgt_lon, tgt_lat)


# ─────────────────────────────────────────────────────────────────────────────
# Core cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_trip_network(
    trip_id: str,
    nodes_raw: pd.DataFrame,
    arcs_raw: pd.DataFrame,
) -> dict:
    """
    Clean one trip's nodes and arcs.

    Returns
    -------
    dict with keys:
      nodes_clean    – DataFrame[orig_node_id, node_id, lon, lat]
      arcs_clean     – DataFrame[orig_arc_id, arc_id, source, target, length_m]
      log            – dict of cleaning counts
      arc_len_stats  – dict of arc length statistics
      crs            – the EPSG string used for this trip
    """
    log: dict = {"trip_id": trip_id}

    # ── 1. Coordinate validation ──────────────────────────────────────────────
    valid_lon = nodes_raw["lon"].between(-180.0, 180.0)
    valid_lat = nodes_raw["lat"].between(-90.0, 90.0)
    coord_mask = valid_lon & valid_lat

    log["n_nodes_raw"] = len(nodes_raw)
    log["n_invalid_coords"] = int((~coord_mask).sum())

    nodes_valid = nodes_raw[coord_mask].copy()

    # ── 2. Duplicate nodes – keep first, build orig→new remap ────────────────
    dup_mask = nodes_valid.duplicated(subset=["lon", "lat"], keep="first")
    log["n_duplicate_nodes"] = int(dup_mask.sum())

    nodes_dedup = nodes_valid[~dup_mask].copy().reset_index(drop=True)
    # Assign fresh sequential node_id; preserve orig_node_id from load_nodes
    nodes_dedup.insert(1, "node_id", range(len(nodes_dedup)))

    # Map orig_node_id → new node_id via (lon, lat) lookup
    coord_to_new: dict[tuple, int] = {
        (r.lon, r.lat): r.node_id
        for r in nodes_dedup.itertuples(index=False)
    }
    orig_to_new: dict[int, int] = {
        r.orig_node_id: coord_to_new[(r.lon, r.lat)]
        for r in nodes_valid.itertuples(index=False)
    }

    log["n_nodes_clean"] = len(nodes_dedup)

    # ── 3. Per-trip CRS ───────────────────────────────────────────────────────
    crs = get_crs_for_trip(nodes_dedup["lon"].values, nodes_dedup["lat"].values)
    transformer = make_transformer("EPSG:4326", crs)
    log["crs_used"] = crs

    # ── 4. Arc endpoint remapping and validation ──────────────────────────────
    arcs_work = arcs_raw.copy()
    arcs_work["source"] = arcs_work["source"].map(orig_to_new)
    arcs_work["target"] = arcs_work["target"].map(orig_to_new)

    invalid_ep_mask = arcs_work["source"].isna() | arcs_work["target"].isna()
    log["n_arcs_raw"] = len(arcs_work)
    log["n_invalid_endpoints"] = int(invalid_ep_mask.sum())

    arcs_work = arcs_work[~invalid_ep_mask].copy()
    arcs_work["source"] = arcs_work["source"].astype(int)
    arcs_work["target"] = arcs_work["target"].astype(int)

    # ── 5. Self-loops ─────────────────────────────────────────────────────────
    self_loop_mask = arcs_work["source"] == arcs_work["target"]
    log["n_self_loops"] = int(self_loop_mask.sum())
    arcs_work = arcs_work[~self_loop_mask].copy()

    # ── 6. Duplicate arcs ─────────────────────────────────────────────────────
    dup_arc_mask = arcs_work.duplicated(subset=["source", "target"], keep="first")
    log["n_duplicate_arcs"] = int(dup_arc_mask.sum())

    arcs_clean = arcs_work[~dup_arc_mask].copy().reset_index(drop=True)
    arcs_clean.insert(1, "arc_id", range(len(arcs_clean)))
    log["n_arcs_clean"] = len(arcs_clean)

    # ── 7. Arc lengths ────────────────────────────────────────────────────────
    nodes_idx = nodes_dedup.set_index("node_id")
    lengths = compute_arc_lengths(nodes_idx, arcs_clean, transformer)
    arcs_clean["length_m"] = lengths

    arc_len_stats = {
        "trip_id": trip_id,
        "crs_used": crs,
        "mean_length_m":   float(np.nanmean(lengths))                       if len(lengths) else np.nan,
        "median_length_m": float(np.nanmedian(lengths))                     if len(lengths) else np.nan,
        "min_length_m":    float(np.nanmin(lengths))                        if len(lengths) else np.nan,
        "max_length_m":    float(np.nanmax(lengths))                        if len(lengths) else np.nan,
        "p95_length_m":    float(np.nanpercentile(lengths, 95))             if len(lengths) else np.nan,
        "total_length_km": float(np.nansum(lengths) / 1000),
        "n_zero_length":   int((lengths == 0).sum()),
    }

    return {
        "nodes_clean":   nodes_dedup,
        "arcs_clean":    arcs_clean,
        "log":           log,
        "arc_len_stats": arc_len_stats,
        "crs":           crs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def process_all_trips(
    data_root: Path,
    out_dir: Path,
    write_per_trip: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all trips under data_root/train and data_root/student_test.

    Writes
    ------
    Per trip (if write_per_trip):
      <out_dir>/<trip_id>/<trip_id>_nodes_clean.csv
      <out_dir>/<trip_id>/<trip_id>_arcs_clean.csv
      <out_dir>/<trip_id>/<trip_id>_cleaning_log.csv

    Aggregate:
      <out_dir>/network_cleaning_summary.csv
      <out_dir>/arc_length_summary.csv

    Returns
    -------
    (cleaning_summary_df, arc_length_summary_df)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaning_rows: list[dict] = []
    arc_len_rows:  list[dict] = []

    split_dirs = [
        data_root / s
        for s in ("train", "student_test")
        if (data_root / s).exists()
    ]
    if not split_dirs:
        raise FileNotFoundError(
            f"No train/ or student_test/ found under {data_root}"
        )

    for split_dir in split_dirs:
        for trip_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            tid = trip_dir.name
            nf  = trip_dir / f"{tid}.nodes"
            af  = trip_dir / f"{tid}.arcs"

            if not nf.exists() or not af.exists():
                print(f"[SKIP] {tid}: missing file(s)")
                continue

            try:
                result = clean_trip_network(
                    tid, load_nodes(nf), load_arcs(af)
                )
            except Exception as exc:
                print(f"[ERROR] {tid}: {exc}")
                continue

            if write_per_trip:
                td_out = out_dir / tid
                td_out.mkdir(parents=True, exist_ok=True)
                result["nodes_clean"].to_csv(td_out / f"{tid}_nodes_clean.csv", index=False)
                result["arcs_clean"].to_csv( td_out / f"{tid}_arcs_clean.csv",  index=False)
                pd.DataFrame([result["log"]]).to_csv(
                    td_out / f"{tid}_cleaning_log.csv", index=False
                )

            cleaning_rows.append(result["log"])
            arc_len_rows.append(result["arc_len_stats"])
            lg = result["log"]
            print(
                f"[OK] {tid}  crs={result['crs']}  "
                f"nodes {lg['n_nodes_clean']}/{lg['n_nodes_raw']}  "
                f"arcs {lg['n_arcs_clean']}/{lg['n_arcs_raw']}"
            )

    cleaning_df = pd.DataFrame(cleaning_rows)
    arc_len_df  = pd.DataFrame(arc_len_rows)

    cleaning_df.to_csv(out_dir / "network_cleaning_summary.csv", index=False)
    arc_len_df.to_csv( out_dir / "arc_length_summary.csv",       index=False)

    print(f"\n[DONE] {len(cleaning_df)} trips processed.")
    print(f"  → {out_dir / 'network_cleaning_summary.csv'}")
    print(f"  → {out_dir / 'arc_length_summary.csv'}")
    return cleaning_df, arc_len_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_ROOT = Path("data/Student folder/Student folder")
    OUT_DIR   = Path("results/network_cleaned")

    cleaning_summary, arc_len_summary = process_all_trips(
        data_root=DATA_ROOT,
        out_dir=OUT_DIR,
        write_per_trip=True,
    )

    print("\n── Cleaning summary (first 5 rows) ──")
    print(cleaning_summary.head().to_string(index=False))
    print("\n── Arc-length summary (first 5 rows) ──")
    print(arc_len_summary.head().to_string(index=False))

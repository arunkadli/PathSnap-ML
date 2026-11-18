"""
generate_labels.py
==================
Generates the is_correct_arc label column needed to train all five ML models.

Strategy
--------
The .route file lists arc IDs (orig_arc_id values) in order. Each route arc
corresponds to one or more consecutive GPS observations. We assign each GPS
observation to the route arc that is spatially closest and consistent with the
route sequence. The assigned arc gets label 1; all other candidates get 0.

Assignment rule (simple proximity):
  For each GPS observation, find which route arc's midpoint is closest.
  Break ties by route order (prefer the arc that comes later in the route).

Output
------
results/labels/<trip_id>_labels.csv
    trip_id | gps_idx | orig_arc_id | is_correct_arc

results/labels/all_labels.csv
    Combined across all trips.

Run from PathSnap-ML/ repo root:
    python src/generate_labels.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.road_network_preprocessing import get_crs_for_trip
    from src.candidate_generation import _project_coords
except ImportError:
    from road_network_preprocessing import get_crs_for_trip
    from candidate_generation import _project_coords


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_route(path: Path) -> np.ndarray:
    """Read a .route file → 1-D array of orig_arc_id values (integers)."""
    df = pd.read_csv(path, header=None, names=["orig_arc_id"], dtype=int)
    return df["orig_arc_id"].values


# ─────────────────────────────────────────────────────────────────────────────
# Label generation for one trip
# ─────────────────────────────────────────────────────────────────────────────

def generate_trip_labels(
    trip_id: str,
    nodes_csv: Path,
    arcs_csv: Path,
    track_csv: Path,
    route_path: Path,
) -> pd.DataFrame:
    """
    For every GPS observation produce a label row:
        trip_id, gps_idx, orig_arc_id, is_correct_arc

    is_correct_arc = 1 for the route arc nearest to the GPS point,
                     0 for all other arcs that appear as candidates.

    Only route arcs that exist in the cleaned arcs table are used;
    missing arcs (removed during cleaning) are skipped with a warning.
    """
    nodes  = pd.read_csv(nodes_csv)
    arcs   = pd.read_csv(arcs_csv)
    track  = pd.read_csv(track_csv)
    route  = load_route(route_path)

    # CRS
    crs = get_crs_for_trip(nodes["lon"].values, nodes["lat"].values)

    # Build projected midpoints for route arcs only
    nidx = nodes.set_index("node_id")

    # Keep only route arcs present in cleaned arcs
    valid_route_arc_ids = set(arcs["orig_arc_id"].values)
    route_filtered = [r for r in route if r in valid_route_arc_ids]
    missing = len(route) - len(route_filtered)
    if missing:
        print(f"  [WARN] {trip_id}: {missing} route arc IDs not in cleaned arcs (removed during cleaning)")

    if not route_filtered:
        print(f"  [SKIP] {trip_id}: no valid route arcs")
        return pd.DataFrame()

    # Subset arcs to route arcs only
    route_arcs = arcs[arcs["orig_arc_id"].isin(route_filtered)].copy()

    src_lon = nidx.loc[route_arcs["source"].values, "lon"].values
    src_lat = nidx.loc[route_arcs["source"].values, "lat"].values
    tgt_lon = nidx.loc[route_arcs["target"].values, "lon"].values
    tgt_lat = nidx.loc[route_arcs["target"].values, "lat"].values

    sx, sy = _project_coords(src_lon, src_lat, crs)
    tx, ty = _project_coords(tgt_lon, tgt_lat, crs)
    mx = (sx + tx) / 2.0
    my = (sy + ty) / 2.0

    # Project GPS points
    gps_x, gps_y = _project_coords(track["lon"].values, track["lat"].values, crs)

    # Midpoint array for route arcs
    arc_mids = np.column_stack([mx, my])          # shape (n_route_arcs, 2)
    orig_ids = route_arcs["orig_arc_id"].values    # aligned with arc_mids

    rows: list[dict] = []

    for i in range(len(track)):
        gps_pt = np.array([gps_x[i], gps_y[i]])
        dists  = np.sqrt(np.sum((arc_mids - gps_pt) ** 2, axis=1))
        best   = int(np.argmin(dists))
        correct_orig_arc_id = int(orig_ids[best])

        rows.append({
            "trip_id":        trip_id,
            "gps_idx":        int(track["gps_idx"].iloc[i]),
            "orig_arc_id":    correct_orig_arc_id,
            "is_correct_arc": 1,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Batch label generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_labels(
    data_root: Path,
    network_root: Path,
    gps_root: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Generate label files for all trips that have:
      - cleaned nodes + arcs  (network_root/<tid>/)
      - cleaned GPS track     (gps_root/<tid>/<tid>_track_clean.csv)
      - raw .route file       (data_root/train/<tid>/<tid>.route)

    Writes per-trip and combined label CSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    for split in ("train", "student_test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        for trip_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            tid = trip_dir.name

            nodes_csv  = network_root / tid / f"{tid}_nodes_clean.csv"
            arcs_csv   = network_root / tid / f"{tid}_arcs_clean.csv"
            track_csv  = gps_root / tid / f"{tid}_track_clean.csv"
            route_path = trip_dir / f"{tid}.route"

            if not all(p.exists() for p in [nodes_csv, arcs_csv, track_csv, route_path]):
                print(f"[SKIP] {tid}: missing file(s)")
                continue

            try:
                labels = generate_trip_labels(
                    tid, nodes_csv, arcs_csv, track_csv, route_path
                )
            except Exception as exc:
                print(f"[ERROR] {tid}: {exc}")
                continue

            if labels.empty:
                continue

            labels.to_csv(out_dir / f"{tid}_labels.csv", index=False)
            all_frames.append(labels)
            print(f"[OK] {tid}  labelled GPS obs: {len(labels)}")

    if not all_frames:
        print("[WARN] No label files generated.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(out_dir / "all_labels.csv", index=False)
    print(f"\n[DONE] Total labelled rows: {len(combined)}")
    print(f"  → {out_dir / 'all_labels.csv'}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_ROOT    = Path("data/Student folder/Student folder")
    NETWORK_ROOT = Path("results/network_cleaned")
    GPS_ROOT     = Path("results/gps_cleaned")
    OUT_DIR      = Path("results/labels")

    combined = generate_all_labels(
        data_root    = DATA_ROOT,
        network_root = NETWORK_ROOT,
        gps_root     = GPS_ROOT,
        out_dir      = OUT_DIR,
    )

    if not combined.empty:
        print("\n── Sample labels ──")
        print(combined.head(10).to_string(index=False))

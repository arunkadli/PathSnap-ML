"""
candidate_generation.py
========================
Amrithaa – Spatial candidate arc generation for GPS observations.

For each GPS point in a trip's .track file this module finds the N nearest
road arcs and returns a candidate table that downstream models (KNN, etc.)
consume directly.

Design
------
1. Load the cleaned nodes and arcs for the trip.
2. Build a midpoint lookup for every arc  (midpoint = average of endpoints).
   For short arcs (< ~200 m) this is an adequate proxy; for longer arcs the
   foot-of-perpendicular is also computed and used when it falls on the segment.
3. Use a KD-tree (scipy) on projected midpoints for fast radius/kNN search.
4. For each GPS observation return the K nearest arcs with:
   - perpendicular distance from GPS point to arc (metres)
   - heading difference between GPS bearing and arc bearing (degrees)
   - arc length, source/target degree features
   - orig_arc_id so Arun's route labels remain traceable

Dependencies: numpy, pandas, scipy  (all in requirements.txt)
Optional:     pyproj  (for UTM projection; haversine fallback used otherwise)

Usage
-----
    from src.candidate_generation import generate_candidates

    candidates = generate_candidates(
        trip_id   = "00000001",
        nodes_csv = Path("results/network_cleaned/00000001/00000001_nodes_clean.csv"),
        arcs_csv  = Path("results/network_cleaned/00000001/00000001_arcs_clean.csv"),
        track_csv = Path("results/gps_cleaned/00000001/00000001_track_clean.csv"),
        k         = 5,
        radius_m  = 50.0,
    )
    # candidates: DataFrame with one row per (gps_obs_index, candidate_arc)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

# Local import of the per-trip CRS helper
try:
    from src.road_network_preprocessing import get_crs_for_trip, haversine_m
except ImportError:
    from road_network_preprocessing import get_crs_for_trip, haversine_m


# ─────────────────────────────────────────────────────────────────────────────
# Projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _project_coords(
    lons: np.ndarray, lats: np.ndarray, crs: str
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lon, lat) → (x_m, y_m) using the given EPSG CRS."""
    if _HAS_PYPROJ:
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        return t.transform(lons, lats)
    # Fallback: equirectangular approximation centred on data bbox
    lat0 = np.mean(lats)
    R = 6_371_000.0
    x = np.radians(lons) * R * np.cos(np.radians(lat0))
    y = np.radians(lats) * R
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Arc geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _foot_of_perpendicular(
    px: np.ndarray, py: np.ndarray,
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each GPS point P and arc segment A→B, compute:
      - the clamped foot of perpendicular F on AB (clamp t to [0,1])
      - the distance |PF| in the same units as the input (metres if projected)

    All inputs are 1-D arrays of the same length (one per candidate pair).
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    # t: scalar projection of AP onto AB, clamped to [0, 1]
    t = np.where(
        seg_len_sq > 0,
        np.clip(((px - ax) * dx + (py - ay) * dy) / seg_len_sq, 0.0, 1.0),
        0.0,
    )
    fx = ax + t * dx
    fy = ay + t * dy
    dist = np.sqrt((px - fx) ** 2 + (py - fy) ** 2)
    return fx, fy, dist


def _bearing_deg(x1, y1, x2, y2) -> np.ndarray:
    """
    Bearing of segment (x1,y1)→(x2,y2) in degrees [0, 360).
    Works in projected space; angles are measured from +y (north).
    """
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dx, dy)) % 360.0
    return angle


def _heading_diff(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Smallest absolute angular difference between two bearing arrays [0, 180]."""
    diff = np.abs(h1 - h2) % 360.0
    return np.minimum(diff, 360.0 - diff)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_arc_index(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    crs: str,
) -> dict:
    """
    Pre-compute projected arc geometry and build a KD-tree on midpoints.

    Parameters
    ----------
    nodes : DataFrame[node_id, lon, lat]  (node_id = new sequential index)
    arcs  : DataFrame[arc_id, orig_arc_id, source, target, length_m]
    crs   : EPSG string for the projected CRS

    Returns
    -------
    dict with keys: tree, arc_data, crs
      tree     : cKDTree on midpoint (x, y) in metres
      arc_data : DataFrame aligned with the tree rows, containing projected
                 src/tgt coords, midpoint, bearing, length_m, orig_arc_id
    """
    nidx = nodes.set_index("node_id")

    src_lon = nidx.loc[arcs["source"].values, "lon"].values
    src_lat = nidx.loc[arcs["source"].values, "lat"].values
    tgt_lon = nidx.loc[arcs["target"].values, "lon"].values
    tgt_lat = nidx.loc[arcs["target"].values, "lat"].values

    sx, sy = _project_coords(src_lon, src_lat, crs)
    tx, ty = _project_coords(tgt_lon, tgt_lat, crs)

    mx = (sx + tx) / 2.0
    my = (sy + ty) / 2.0

    bearing = _bearing_deg(sx, sy, tx, ty)

    arc_data = pd.DataFrame({
        "arc_id":       arcs["arc_id"].values,
        "orig_arc_id":  arcs["orig_arc_id"].values,
        "source":       arcs["source"].values,
        "target":       arcs["target"].values,
        "length_m":     arcs["length_m"].values,
        "sx": sx, "sy": sy,
        "tx": tx, "ty": ty,
        "mx": mx, "my": my,
        "arc_bearing":  bearing,
    })

    tree = cKDTree(np.column_stack([mx, my]))
    return {"tree": tree, "arc_data": arc_data, "crs": crs}


def generate_candidates(
    trip_id: str,
    nodes_csv: Path,
    arcs_csv: Path,
    track_csv: Path,
    k: int = 5,
    radius_m: float = 50.0,
    degree_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For every GPS observation in track_csv find the K nearest road arcs
    within radius_m metres and return a flat candidate DataFrame.

    Parameters
    ----------
    trip_id   : trip identifier string
    nodes_csv : path to <trip>_nodes_clean.csv
    arcs_csv  : path to <trip>_arcs_clean.csv
    track_csv : path to cleaned GPS track CSV; must have columns
                lon, lat, timestamp (and optionally heading_deg, speed_mps)
    k         : maximum candidates per GPS point
    radius_m  : search radius in metres; arcs outside this radius are excluded
    degree_csv: optional path to per-node degree CSV for feature enrichment

    Returns
    -------
    DataFrame with one row per (gps_idx, candidate_arc):
        trip_id, gps_idx, timestamp, gps_lon, gps_lat,
        orig_arc_id, arc_id, source, target, length_m,
        perp_dist_m, heading_diff_deg, arc_bearing_deg,
        gps_heading_deg, gps_speed_mps,
        in_degree_src, out_degree_src, in_degree_tgt, out_degree_tgt,
        candidate_rank
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    nodes = pd.read_csv(nodes_csv)
    arcs  = pd.read_csv(arcs_csv)
    track = pd.read_csv(track_csv)

    # Normalise track column names (handle both Architha's output and raw format)
    track = _normalise_track(track)

    # ── CRS ──────────────────────────────────────────────────────────────────
    crs = get_crs_for_trip(nodes["lon"].values, nodes["lat"].values)

    # ── Degree lookup (optional enrichment) ──────────────────────────────────
    degree_lookup: dict[int, dict] = {}
    if degree_csv is not None and Path(degree_csv).exists():
        deg_df = pd.read_csv(degree_csv)
        for row in deg_df.itertuples(index=False):
            degree_lookup[int(row.node_id)] = {
                "in":  int(getattr(row, "in_degree",  0)),
                "out": int(getattr(row, "out_degree", 0)),
            }

    # ── Build arc index ───────────────────────────────────────────────────────
    idx = build_arc_index(nodes, arcs, crs)
    tree     = idx["tree"]
    arc_data = idx["arc_data"]

    # ── Project GPS points ────────────────────────────────────────────────────
    gps_x, gps_y = _project_coords(
        track["lon"].values, track["lat"].values, crs
    )

    # ── GPS heading and speed (if available) ──────────────────────────────────
    gps_heading = track["heading_deg"].values  if "heading_deg" in track.columns  else np.full(len(track), np.nan)
    gps_speed   = track["speed_mps"].values    if "speed_mps"   in track.columns  else np.full(len(track), np.nan)

    # ── Candidate search ─────────────────────────────────────────────────────
    rows: list[dict] = []

    for i in range(len(track)):
        pt = np.array([[gps_x[i], gps_y[i]]])

        # Query KD-tree: up to k neighbours within radius_m
        dists_mid, idxs = tree.query(pt, k=k, distance_upper_bound=radius_m + 1e-6)
        dists_mid = dists_mid[0]
        idxs      = idxs[0]

        # Filter out dummy "not found" entries (index == n_arcs)
        valid = (idxs < len(arc_data)) & (dists_mid <= radius_m)
        idxs      = idxs[valid]
        dists_mid = dists_mid[valid]

        if len(idxs) == 0:
            # Expand to 2× radius when nothing found (keep at most k)
            dists_mid, idxs = tree.query(pt, k=min(k, len(arc_data)),
                                          distance_upper_bound=radius_m * 2 + 1e-6)
            dists_mid = dists_mid[0]
            idxs      = idxs[0]
            valid = idxs < len(arc_data)
            idxs      = idxs[valid]
            dists_mid = dists_mid[valid]

        cands = arc_data.iloc[idxs]

        # Perpendicular distance from GPS point to each arc segment
        _, _, perp_dists = _foot_of_perpendicular(
            np.full(len(cands), gps_x[i]),
            np.full(len(cands), gps_y[i]),
            cands["sx"].values,
            cands["sy"].values,
            cands["tx"].values,
            cands["ty"].values,
        )

        # Heading difference
        h_diff = _heading_diff(
            np.full(len(cands), gps_heading[i]),
            cands["arc_bearing"].values,
        )

        for rank, (ci, perp, hdiff) in enumerate(
            zip(cands.itertuples(index=False), perp_dists, h_diff), start=1
        ):
            src_deg = degree_lookup.get(int(ci.source), {"in": 0, "out": 0})
            tgt_deg = degree_lookup.get(int(ci.target), {"in": 0, "out": 0})

            rows.append({
                "trip_id":          trip_id,
                "gps_idx":          i,
                "timestamp":        track["timestamp"].iloc[i],
                "gps_lon":          track["lon"].iloc[i],
                "gps_lat":          track["lat"].iloc[i],
                "orig_arc_id":      int(ci.orig_arc_id),
                "arc_id":           int(ci.arc_id),
                "source":           int(ci.source),
                "target":           int(ci.target),
                "length_m":         float(ci.length_m),
                "perp_dist_m":      float(perp),
                "heading_diff_deg": float(hdiff),
                "arc_bearing_deg":  float(ci.arc_bearing),
                "gps_heading_deg":  float(gps_heading[i]),
                "gps_speed_mps":    float(gps_speed[i]),
                "in_degree_src":    src_deg["in"],
                "out_degree_src":   src_deg["out"],
                "in_degree_tgt":    tgt_deg["in"],
                "out_degree_tgt":   tgt_deg["out"],
                "candidate_rank":   rank,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Batch candidate generation across all trips
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_candidates(
    network_root: Path,
    track_root: Path,
    out_dir: Path,
    k: int = 5,
    radius_m: float = 50.0,
) -> pd.DataFrame:
    """
    Run generate_candidates for every trip that has cleaned network + track.

    Parameters
    ----------
    network_root : results/network_cleaned/
    track_root   : results/gps_cleaned/   (or data/ if using raw tracks)
    out_dir      : where to write per-trip and combined candidate CSVs

    Returns
    -------
    Combined candidates DataFrame
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    trip_ids = sorted(d.name for d in network_root.iterdir() if d.is_dir())

    for tid in trip_ids:
        nodes_csv = network_root / tid / f"{tid}_nodes_clean.csv"
        arcs_csv  = network_root / tid / f"{tid}_arcs_clean.csv"

        # Try cleaned GPS first; fall back to raw track
        track_candidates = [
            track_root / tid / f"{tid}_track_clean.csv",
            track_root / tid / f"{tid}.track",
        ]
        track_csv = next((p for p in track_candidates if p.exists()), None)

        if not (nodes_csv.exists() and arcs_csv.exists() and track_csv):
            print(f"[SKIP] {tid}: missing network or track file")
            continue

        try:
            cands = generate_candidates(
                trip_id=tid,
                nodes_csv=nodes_csv,
                arcs_csv=arcs_csv,
                track_csv=track_csv,
                k=k,
                radius_m=radius_m,
            )
        except Exception as exc:
            print(f"[ERROR] {tid}: {exc}")
            continue

        per_trip_path = out_dir / f"{tid}_candidates.csv"
        cands.to_csv(per_trip_path, index=False)
        all_frames.append(cands)
        print(f"[OK] {tid}  GPS obs: {cands['gps_idx'].nunique()}  "
              f"candidates: {len(cands)}")

    if not all_frames:
        print("[WARN] No candidate files generated.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(out_dir / "all_candidates.csv", index=False)
    print(f"\n[DONE] Combined candidates: {len(combined)} rows")
    print(f"  → {out_dir / 'all_candidates.csv'}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_track(track: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either Architha's cleaned GPS output or the raw .track format and
    return a DataFrame with columns: lon, lat, timestamp.
    """
    # Raw .track: three columns, no header
    if list(track.columns) == [0, 1, 2] or list(track.columns) == ["0", "1", "2"]:
        track = track.copy()
        track.columns = ["lon", "lat", "timestamp"]
        return track

    # Already named
    rename = {}
    for col in track.columns:
        cl = col.lower()
        if cl in ("longitude", "lon_deg"):
            rename[col] = "lon"
        elif cl in ("latitude", "lat_deg"):
            rename[col] = "lat"
        elif cl in ("time", "ts", "time_s", "time_sec"):
            rename[col] = "timestamp"
    if rename:
        track = track.rename(columns=rename)

    return track


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    NET_ROOT   = Path("results/network_cleaned")
    TRACK_ROOT = Path("data/Student folder/Student folder/train")
    OUT_DIR    = Path("results/candidates")

    combined = generate_all_candidates(
        network_root=NET_ROOT,
        track_root=TRACK_ROOT,
        out_dir=OUT_DIR,
        k=5,
        radius_m=50.0,
    )

    if not combined.empty:
        print("\n── Sample candidates ──")
        print(combined.head(10).to_string(index=False))

"""
gps_preprocessing.py
=====================
GPS track cleaning and feature extraction from .track files.

This module processes raw .track files (lon, lat, timestamp) and produces:
  - Cleaned GPS track CSVs  (used by candidate_generation.py)
  - GPS quality summary      (gps_quality_summary.csv)
  - GPS cleaning summary     (gps_cleaning_summary.csv)

Cleaning rules applied
----------------------
1. Invalid coordinates   : lon ∉ [-180,180] or lat ∉ [-90,90]  → dropped
2. Duplicate timestamps  : keep first occurrence
3. Non-monotonic timestamps: re-sort by timestamp
4. Large GPS gaps        : flag rows where time diff > 30 s
5. Implied speed outliers: flag rows where implied speed > 50 m/s (~180 km/h)
6. Add derived columns   : dist_m, time_diff_s, speed_mps, heading_deg

Output columns in <trip>_track_clean.csv
-----------------------------------------
  gps_idx, lon, lat, timestamp,
  dist_m, time_diff_s, speed_mps, heading_deg,
  large_gap_flag, speed_outlier_flag

Run from PathSnap-ML/ repo root:
    python src/gps_preprocessing.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

try:
    from src.road_network_preprocessing import get_crs_for_trip, haversine_m
except ImportError:
    from road_network_preprocessing import get_crs_for_trip, haversine_m


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_SPEED_MPS   = 50.0   # ~180 km/h — anything faster flagged as outlier
MAX_GAP_S       = 30.0   # seconds — gaps longer than this are flagged


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_track(path: Path) -> pd.DataFrame:
    """
    Read a .track file → DataFrame[lon, lat, timestamp].
    Three whitespace-separated columns, no header.
    """
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["lon", "lat", "timestamp"]
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Derived geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_distances(lons: np.ndarray, lats: np.ndarray,
                       crs: str) -> np.ndarray:
    """
    Compute distance in metres between consecutive GPS points.
    Returns array of length N (first element is 0.0).
    """
    if len(lons) < 2:
        return np.zeros(len(lons))

    if _HAS_PYPROJ:
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = t.transform(lons, lats)
        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.sqrt(dx**2 + dy**2)
    else:
        dists = haversine_m(lons[:-1], lats[:-1], lons[1:], lats[1:])

    return np.concatenate([[0.0], dists])


def _compute_headings(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Compute bearing in degrees [0, 360) between consecutive GPS points.
    Uses the forward bearing formula. First point gets same heading as second.
    """
    if len(lons) < 2:
        return np.zeros(len(lons))

    dx = np.diff(lons)
    dy = np.diff(lats)
    headings = np.degrees(np.arctan2(dx, dy)) % 360.0
    return np.concatenate([[headings[0]], headings])


# ─────────────────────────────────────────────────────────────────────────────
# Core cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_track(trip_id: str, track_raw: pd.DataFrame) -> dict:
    """
    Clean one trip's GPS track.

    Returns dict:
        track_clean    – cleaned DataFrame ready for candidate_generation.py
        log            – cleaning count dict
        quality_stats  – quality stats dict
    """
    log: dict = {"trip_id": trip_id}
    log["n_raw"] = len(track_raw)

    # ── 1. Invalid coordinates ────────────────────────────────────────────────
    valid_lon = track_raw["lon"].between(-180.0, 180.0)
    valid_lat = track_raw["lat"].between(-90.0, 90.0)
    coord_mask = valid_lon & valid_lat
    log["n_invalid_coords"] = int((~coord_mask).sum())
    df = track_raw[coord_mask].copy()

    # ── 2. Missing values ─────────────────────────────────────────────────────
    missing = df.isna().any(axis=1).sum()
    log["n_missing"] = int(missing)
    df = df.dropna()

    # ── 3. Duplicate timestamps ───────────────────────────────────────────────
    dup_ts = df.duplicated(subset=["timestamp"], keep="first").sum()
    log["n_duplicate_timestamps"] = int(dup_ts)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # ── 4. Sort by timestamp ─────────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "gps_idx", range(len(df)))

    # ── 5. Per-trip CRS ───────────────────────────────────────────────────────
    crs = get_crs_for_trip(df["lon"].values, df["lat"].values)

    # ── 6. Derived columns ────────────────────────────────────────────────────
    dist_m      = _compute_distances(df["lon"].values, df["lat"].values, crs)
    time_diff_s = np.concatenate([[0.0], np.diff(df["timestamp"].values)])
    # Avoid division by zero for identical timestamps
    speed_mps   = np.where(
        time_diff_s > 0,
        dist_m / time_diff_s,
        0.0
    )
    heading_deg = _compute_headings(df["lon"].values, df["lat"].values)

    df["dist_m"]      = dist_m
    df["time_diff_s"] = time_diff_s
    df["speed_mps"]   = speed_mps
    df["heading_deg"] = heading_deg

    # ── 7. Flag large gaps ────────────────────────────────────────────────────
    df["large_gap_flag"] = (df["time_diff_s"] > MAX_GAP_S).astype(int)
    log["n_large_gaps"]  = int(df["large_gap_flag"].sum())

    # ── 8. Flag speed outliers (do NOT drop — just flag for ML) ──────────────
    df["speed_outlier_flag"] = (df["speed_mps"] > MAX_SPEED_MPS).astype(int)
    log["n_speed_outliers"]  = int(df["speed_outlier_flag"].sum())

    log["n_clean"] = len(df)

    # ── Quality stats ─────────────────────────────────────────────────────────
    quality_stats = {
        "trip_id":          trip_id,
        "n_observations":   len(df),
        "duration_s":       float(df["timestamp"].max() - df["timestamp"].min()),
        "total_dist_m":     float(dist_m.sum()),
        "mean_speed_mps":   float(np.nanmean(speed_mps[speed_mps > 0])) if (speed_mps > 0).any() else 0.0,
        "max_speed_mps":    float(np.nanmax(speed_mps)),
        "mean_interval_s":  float(np.nanmean(time_diff_s[time_diff_s > 0])) if (time_diff_s > 0).any() else 0.0,
        "pct_1s_intervals": float(np.mean(time_diff_s[1:] == 1.0) * 100) if len(time_diff_s) > 1 else 0.0,
        "n_large_gaps":     log["n_large_gaps"],
        "n_speed_outliers": log["n_speed_outliers"],
        "lon_min":          float(df["lon"].min()),
        "lon_max":          float(df["lon"].max()),
        "lat_min":          float(df["lat"].min()),
        "lat_max":          float(df["lat"].max()),
        "crs_used":         crs,
    }

    return {
        "track_clean":   df,
        "log":           log,
        "quality_stats": quality_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def process_all_tracks(
    data_root: Path,
    out_dir: Path,
    write_per_trip: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all .track files under data_root/train and data_root/student_test.

    Writes
    ------
    Per trip (if write_per_trip=True):
        <out_dir>/<trip_id>/<trip_id>_track_clean.csv

    Aggregate:
        <out_dir>/gps_quality_summary.csv
        <out_dir>/gps_cleaning_summary.csv

    Returns
    -------
    (quality_summary_df, cleaning_summary_df)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_rows:  list[dict] = []
    cleaning_rows: list[dict] = []

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
            tid  = trip_dir.name
            trkf = trip_dir / f"{tid}.track"

            if not trkf.exists():
                print(f"[SKIP] {tid}: no .track file")
                continue

            try:
                raw    = load_track(trkf)
                result = clean_track(tid, raw)
            except Exception as exc:
                print(f"[ERROR] {tid}: {exc}")
                continue

            if write_per_trip:
                td_out = out_dir / tid
                td_out.mkdir(parents=True, exist_ok=True)
                result["track_clean"].to_csv(
                    td_out / f"{tid}_track_clean.csv", index=False
                )

            quality_rows.append(result["quality_stats"])
            cleaning_rows.append(result["log"])

            lg = result["log"]
            print(
                f"[OK] {tid}  obs {lg['n_clean']}/{lg['n_raw']}  "
                f"gaps={lg['n_large_gaps']}  speed_outliers={lg['n_speed_outliers']}"
            )

    quality_df  = pd.DataFrame(quality_rows)
    cleaning_df = pd.DataFrame(cleaning_rows)

    quality_df.to_csv( out_dir / "gps_quality_summary.csv",  index=False)
    cleaning_df.to_csv(out_dir / "gps_cleaning_summary.csv", index=False)

    print(f"\n[DONE] {len(quality_df)} trips processed.")
    print(f"  → {out_dir / 'gps_quality_summary.csv'}")
    print(f"  → {out_dir / 'gps_cleaning_summary.csv'}")

    return quality_df, cleaning_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_ROOT = Path("data/Student folder/Student folder")
    OUT_DIR   = Path("results/gps_cleaned")

    quality_summary, cleaning_summary = process_all_tracks(
        data_root=DATA_ROOT,
        out_dir=OUT_DIR,
        write_per_trip=True,
    )

    print("\n── Quality summary (first 5 rows) ──")
    print(quality_summary.head().to_string(index=False))
    print("\n── Cleaning summary (first 5 rows) ──")
    print(cleaning_summary.head().to_string(index=False))

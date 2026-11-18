"""
gps_features.py -- GPS-derived features for every observation of a trip (PathSnap).

STANDARD LIBRARY ONLY, same as gps_preprocessing.py.

Input : cleaned rows from gps_preprocessing.py (data/cleaned/<partition>/<trip>.csv
        or the list returned by clean_trip). One output row per input row, in the
        same order, so no observation is dropped and orig_idx still maps back to
        the raw .track row.
Output: data/features/<partition>/<trip_id>.csv

Run from the repo root (after gps_preprocessing.py):
    python src/gps_features.py

Only the trip's own GPS track is used (its neighbouring fixes in time). No road
network, no labels, nothing learned from other trips, so these features are
identical for train, student_test and evaluator trips and cannot leak labels.
Features that need the road network (distance to an arc, heading difference to
an arc, arc length, ...) belong in the candidate table, which uses heading_deg
from here.

KEY_COLS identify the row; do not feed lon/lat/t to the model directly.
FEATURE_COLS are the model inputs.
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

try:
    from gps_preprocessing import haversine_m, DT_CAP_S
except ImportError:                                   # imported as src.gps_features
    from src.gps_preprocessing import haversine_m, DT_CAP_S

PARTITIONS = ["train", "student_test"]
MIN_MOVE_M = 3.0          # a step shorter than this has no reliable heading (GPS noise)
STATIONARY_KMH = 1.0      # smoothed speed below this = stationary
SPIKE_KMH = 200.0         # implied speed above this = speed spike (kept, only flagged)
SMOOTH_HALF = 2           # speed median window = 2*SMOOTH_HALF+1 fixes
SINUOUS_HALF = 3          # sinuosity window = 2*SINUOUS_HALF+1 fixes
SINUOUS_CAP = 10.0
ACCEL_CAP = 20.0          # m/s^2, features are clipped, points are never dropped

KEY_COLS = ["orig_idx", "lon", "lat", "t"]
FEATURE_COLS = [
    "dt_prev_s", "dt_next_s", "step_prev_m", "step_next_m",
    "speed_prev_kmh", "speed_next_kmh", "speed_smooth_kmh", "accel_ms2",
    "heading_deg", "heading_sin", "heading_cos", "heading_valid",
    "turn_deg", "abs_turn_deg", "sinuosity",
    "is_stationary", "is_speed_spike", "is_repeat_coord",
    "gap_before", "gap_after", "segment_id", "is_segment_start", "is_segment_end",
    "pos_in_segment", "segment_len", "cum_dist_m", "progress", "t_rel_s",
]


# ----------------------------------------------------------------------------
# geometry helpers (Arun can reuse these for arc headings in the candidate table)
# ----------------------------------------------------------------------------
def bearing_deg(lon1, lat1, lon2, lat2):
    """Initial compass bearing 0..360 (0 = north, 90 = east) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_diff_deg(a, b):
    """Signed smallest rotation from bearing a to bearing b, in [-180, 180)."""
    return (b - a + 180) % 360 - 180


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
def compute_features(clean):
    """clean: list of dicts with orig_idx, lon, lat, t, gap_flag, segment_id,
    is_repeat_coord. Returns one feature dict per row, same order."""
    n = len(clean)
    if n == 0:
        return []
    lon = [float(r["lon"]) for r in clean]
    lat = [float(r["lat"]) for r in clean]
    t = [float(r["t"]) for r in clean]
    gap = [int(r["gap_flag"]) for r in clean]
    seg = [int(r["segment_id"]) for r in clean]
    rep = [int(r["is_repeat_coord"]) for r in clean]

    # step i = movement from fix i-1 to fix i (step[0] = 0)
    step = [0.0] + [haversine_m(lon[i - 1], lat[i - 1], lon[i], lat[i]) for i in range(1, n)]
    dt = [0.0] + [t[i] - t[i - 1] for i in range(1, n)]
    spd = [step[i] / dt[i] * 3.6 if dt[i] > 0 else 0.0 for i in range(n)]
    if n > 1:
        spd[0] = spd[1]

    # movement bearings; None where the step is too short to trust
    b_in = [None] + [bearing_deg(lon[i - 1], lat[i - 1], lon[i], lat[i])
                     if step[i] >= MIN_MOVE_M else None for i in range(1, n)]
    b_out = b_in[1:] + [None]
    head = [b_in[i] if b_in[i] is not None else b_out[i] for i in range(n)]
    valid = [1 if h is not None else 0 for h in head]
    last = None                                        # forward fill, then back fill
    for i in range(n):
        if head[i] is None:
            head[i] = last
        else:
            last = head[i]
    nxt = None
    for i in range(n - 1, -1, -1):
        if head[i] is None:
            head[i] = nxt
        else:
            nxt = head[i]
    head = [h if h is not None else 0.0 for h in head]

    # position inside the segment (a new segment starts after every dropout)
    seg_first, seg_count = {}, {}
    for i, s in enumerate(seg):
        seg_first.setdefault(s, i)
        seg_count[s] = seg_count.get(s, 0) + 1

    total = sum(step)
    cum = 0.0
    out = []
    for i in range(n):
        cum += step[i]
        dt_prev = min(dt[i], DT_CAP_S)
        dt_next = min(dt[i + 1], DT_CAP_S) if i < n - 1 else 0.0
        v_prev = spd[i]
        v_next = spd[i + 1] if i < n - 1 else spd[i]

        # speed smoothed with a median over neighbours of the same segment
        lo, hi = max(0, i - SMOOTH_HALF), min(n, i + SMOOTH_HALF + 1)
        vals = [spd[j] for j in range(lo, hi) if seg[j] == seg[i] and gap[j] == 0]
        v_smooth = statistics.median(vals) if vals else v_prev

        # acceleration across the fix; 0 when a gap makes the speeds meaningless
        dts = [d for d in (dt_prev, dt_next) if d > 0]
        den = sum(dts) / len(dts) if dts else 0.0
        gap_next = gap[i + 1] if i < n - 1 else 0
        if den > 0 and gap[i] == 0 and gap_next == 0:
            accel = max(-ACCEL_CAP, min(ACCEL_CAP, (v_next - v_prev) / 3.6 / den))
        else:
            accel = 0.0

        # turn angle between the incoming and outgoing movement
        if b_in[i] is not None and b_out[i] is not None:
            turn = angle_diff_deg(b_in[i], b_out[i])
        else:
            turn = 0.0

        # sinuosity: path length / straight distance over a small window
        lo, hi = max(0, i - SINUOUS_HALF), min(n, i + SINUOUS_HALF + 1)
        idx = [j for j in range(lo, hi) if seg[j] == seg[i]]
        if len(idx) >= 2:
            path = sum(step[j] for j in idx[1:])
            disp = haversine_m(lon[idx[0]], lat[idx[0]], lon[idx[-1]], lat[idx[-1]])
            sinu = min(path / max(disp, 1.0), SINUOUS_CAP)
        else:
            sinu = 1.0

        pos = i - seg_first[seg[i]]
        out.append({
            "orig_idx": clean[i]["orig_idx"], "lon": lon[i], "lat": lat[i], "t": t[i],
            "dt_prev_s": dt_prev, "dt_next_s": dt_next,
            "step_prev_m": step[i], "step_next_m": step[i + 1] if i < n - 1 else 0.0,
            "speed_prev_kmh": v_prev, "speed_next_kmh": v_next,
            "speed_smooth_kmh": v_smooth, "accel_ms2": accel,
            "heading_deg": head[i],
            "heading_sin": math.sin(math.radians(head[i])),
            "heading_cos": math.cos(math.radians(head[i])),
            "heading_valid": valid[i],
            "turn_deg": turn, "abs_turn_deg": abs(turn), "sinuosity": sinu,
            "is_stationary": 1 if v_smooth < STATIONARY_KMH else 0,
            "is_speed_spike": 1 if v_prev > SPIKE_KMH else 0,
            "is_repeat_coord": rep[i],
            "gap_before": gap[i], "gap_after": gap_next,
            "segment_id": seg[i],
            "is_segment_start": 1 if pos == 0 else 0,
            "is_segment_end": 1 if pos == seg_count[seg[i]] - 1 else 0,
            "pos_in_segment": pos, "segment_len": seg_count[seg[i]],
            "cum_dist_m": cum, "progress": cum / total if total > 0 else 0.0,
            "t_rel_s": t[i] - t[0],
        })
    return out


def features_from_track(track_path):
    """Raw .track file -> (clean rows, feature rows). Use this in the API so a
    trip goes through exactly the same steps as in training."""
    try:
        from gps_preprocessing import read_raw, clean_trip
    except ImportError:
        from src.gps_preprocessing import read_raw, clean_trip
    clean, _ = clean_trip(read_raw(track_path))
    return clean, compute_features(clean)


# ----------------------------------------------------------------------------
# csv io + pipeline
# ----------------------------------------------------------------------------
def load_clean(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{"orig_idx": int(r["orig_idx"]), "lon": float(r["lon"]), "lat": float(r["lat"]),
                 "t": float(r["t"]), "gap_flag": int(r["gap_flag"]),
                 "segment_id": int(r["segment_id"]), "is_repeat_coord": int(r["is_repeat_coord"])}
                for r in csv.DictReader(f)]


def write_features(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLS + FEATURE_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def run_all(clean_dir="data/cleaned", out_dir="data/features", verbose=True):
    clean_dir, out_dir = Path(clean_dir), Path(out_dir)
    for part in PARTITIONS:
        files = sorted((clean_dir / part).glob("*.csv"))
        n_rows = 0
        for p in files:
            clean = load_clean(p)
            feats = compute_features(clean)
            assert len(feats) == len(clean), f"{p.name}: row count changed"
            write_features(out_dir / part / p.name, feats)
            n_rows += len(feats)
        if verbose:
            print(f"[{part}] {len(files)} trips, {n_rows} observations -> {out_dir / part}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPS-derived features (stdlib only)")
    ap.add_argument("--clean-dir", default="data/cleaned")
    ap.add_argument("--out", default="data/features")
    a = ap.parse_args()
    run_all(a.clean_dir, a.out)

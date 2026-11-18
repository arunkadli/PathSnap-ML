"""
gps_preprocessing.py -- GPS track quality checks, cleaning and plots (PathSnap).

STANDARD LIBRARY ONLY (no numpy / pandas / matplotlib), so it runs on laptops
where Application Control blocks compiled packages.

Run from the repo root:
    python src/gps_preprocessing.py --data data/student_folder

Writes:
    results/gps_quality_summary.csv    RAW data quality, one row per trip (before cleaning)
    results/gps_cleaning_summary.csv   what the cleaning did, one row per trip
    data/cleaned/<partition>/<trip_id>.csv   cleaned GPS files (gitignored, re-creatable)
    eda/figures/gps_*.svg              plots (open in a browser)

CLEANING RULES (agreed in data_quality_log.md)
  1. Invalid observation (NaN / unparseable, lat outside +-90, lon outside +-180,
     or exactly (0,0))                       -> removed, counted, orig_idx kept.
  2. Exact duplicate row (same lon, lat, t)  -> later copies removed, counted.
     Same coordinates at a different time (parked car) is NOT a duplicate: kept and
     flagged is_repeat_coord.
  3. Chronological order                     -> stable sort by time if needed; the
     original row number is kept in orig_idx so predictions map back to raw rows.
  4. Time-difference info                    -> dt_s (raw), dt_capped_s (capped at 60 s),
     step_m (haversine metres), speed_kmh.
  5. Large gaps (dt > 5 s)                   -> gap_flag: 0 = none,
     1 = pause (moved < 50 m: parked, clock jumped), 2 = dropout (moved >= 50 m).
     segment_id increases after every dropout (the HMM may reset there).
  6. Nothing else is deleted: GPS speed spikes are kept (that noise is what
     map-matching has to correct).

Only the training partition may be used to learn anything from this data; the
cleaning itself uses fixed rules, so it is applied identically to every partition.
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

PARTITIONS = ["train", "student_test"]
RAW_EXT = ["track", "nodes", "arcs", "route"]

BIG_GAP_S = 5          # dt above this = "large gap"
PAUSE_JUMP_M = 50      # moved less than this during a gap = pause, else dropout
DT_CAP_S = 60          # time-difference feature is capped here
MAX_PLAUSIBLE_KMH = 200

CLEAN_COLS = ["orig_idx", "lon", "lat", "t", "dt_s", "dt_capped_s", "step_m",
              "speed_kmh", "gap_flag", "gap_type", "segment_id", "is_repeat_coord"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (raw degrees are not metres)."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _to_float(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_raw(path):
    """Read a .track file -> list of (orig_idx, lon, lat, t).
    Whitespace separated, no header, CRLF safe. Short or unparseable rows become NaN
    rows (so they are counted as invalid instead of crashing)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            p = line.split()
            if len(p) < 3:
                rows.append((i, float("nan"), float("nan"), float("nan")))
            else:
                rows.append((i, _to_float(p[0]), _to_float(p[1]), _to_float(p[2])))
    return rows


def is_invalid(lon, lat, t):
    if lon != lon or lat != lat or t != t:                 # NaN
        return True
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        return True
    return lon == 0 and lat == 0                           # classic dropout signature


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


# ----------------------------------------------------------------------------
# raw data quality (BEFORE cleaning)
# ----------------------------------------------------------------------------
def quality_trip(rows, partition, trip_id, trip_dir):
    q = {"partition": partition, "trip_id": trip_id}
    for e in RAW_EXT:
        f = trip_dir / f"{trip_id}.{e}"
        q[f"has_{e}"] = f.exists()
        q[f"size_{e}_kb"] = round(f.stat().st_size / 1024, 1) if f.exists() else ""
    q["n_raw"] = len(rows)
    q["n_nan_or_unparseable"] = sum(1 for _, a, b, c in rows if a != a or b != b or c != c)
    ok = [r for r in rows if r[1] == r[1] and r[2] == r[2] and r[3] == r[3]]
    q["n_lat_invalid"] = sum(1 for _, _, b, _ in ok if not -90 <= b <= 90)
    q["n_lon_invalid"] = sum(1 for _, a, _, _ in ok if not -180 <= a <= 180)
    q["n_zero_coord"] = sum(1 for _, a, b, _ in ok if a == 0 and b == 0)
    q["n_lon_looks_like_lat"] = sum(1 for _, a, b, _ in ok if abs(a) <= 90 and abs(b) > 90)
    q["n_exact_duplicates"] = len(ok) - len({(a, b, c) for _, a, b, c in ok})
    q["n_repeat_coords"] = sum(1 for i in range(1, len(ok))
                               if ok[i][1] == ok[i - 1][1] and ok[i][2] == ok[i - 1][2])
    good = [r for r in ok if not is_invalid(r[1], r[2], r[3])]
    if len(good) < 2:
        q["flags"] = "too_few_valid_points"
        return q
    lon = [r[1] for r in good]; lat = [r[2] for r in good]; t = [r[3] for r in good]
    dts = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    steps = [haversine_m(lon[i], lat[i], lon[i + 1], lat[i + 1]) for i in range(len(t) - 1)]
    speeds = [s / d * 3.6 for s, d in zip(steps, dts) if d > 0]
    q["n_t_decreasing"] = sum(1 for d in dts if d < 0)
    q["n_t_equal"] = sum(1 for d in dts if d == 0)
    q["t_monotonic"] = all(d > 0 for d in dts)
    q["duration_s"] = t[-1] - t[0]
    q["pct_dt_eq_1s"] = round(100 * sum(1 for d in dts if d == 1) / len(dts), 2)
    q["median_dt_s"] = statistics.median(dts)
    q["n_gaps_gt1s"] = sum(1 for d in dts if d > 1)
    q["n_gaps_big"] = sum(1 for d in dts if d > BIG_GAP_S)
    q["max_gap_s"] = max(dts)
    q["lon_min"], q["lon_max"] = min(lon), max(lon)
    q["lat_min"], q["lat_max"] = min(lat), max(lat)
    q["extent_ns_km"] = round(haversine_m(q["lon_min"], q["lat_min"], q["lon_min"], q["lat_max"]) / 1000, 3)
    q["extent_ew_km"] = round(haversine_m(q["lon_min"], q["lat_min"], q["lon_max"], q["lat_min"]) / 1000, 3)
    q["utm_epsg"] = utm_epsg(sum(lon) / len(lon), sum(lat) / len(lat))
    q["path_length_km"] = round(sum(steps) / 1000, 3)
    q["mean_speed_kmh"] = round(sum(speeds) / len(speeds), 2) if speeds else ""
    q["max_speed_kmh"] = round(max(speeds), 2) if speeds else ""
    q["n_speed_gt200kmh"] = sum(1 for v in speeds if v > MAX_PLAUSIBLE_KMH)
    fl = []
    if q["n_nan_or_unparseable"] or q["n_lat_invalid"] or q["n_lon_invalid"] or q["n_zero_coord"]:
        fl.append("invalid_obs")
    if q["n_exact_duplicates"]: fl.append("duplicates")
    if not q["t_monotonic"]: fl.append("time_not_increasing")
    if q["n_gaps_big"]: fl.append(f"gaps_gt{BIG_GAP_S}s")
    if q["n_speed_gt200kmh"]: fl.append("speed_spikes")
    q["flags"] = ";".join(fl)
    return q


# ----------------------------------------------------------------------------
# cleaning
# ----------------------------------------------------------------------------
def clean_trip(rows):
    """rows: raw (orig_idx, lon, lat, t). Returns (clean_rows, stats)."""
    st = {"n_raw": len(rows)}

    # 1. invalid observations
    valid = [r for r in rows if not is_invalid(r[1], r[2], r[3])]
    st["n_invalid_removed"] = len(rows) - len(valid)

    # 3. chronological order (stable sort; orig_idx is the tie-breaker)
    st["n_out_of_order_steps"] = sum(1 for i in range(len(valid) - 1) if valid[i + 1][3] < valid[i][3])
    valid.sort(key=lambda r: (r[3], r[0]))
    st["reordered"] = st["n_out_of_order_steps"] > 0

    # 2. exact duplicates (same lon, lat, t) -> keep first
    seen, dedup = set(), []
    for r in valid:
        key = (r[1], r[2], r[3])
        if key in seen:
            continue
        seen.add(key); dedup.append(r)
    st["n_duplicates_removed"] = len(valid) - len(dedup)

    # 4 + 5. time differences and gap flags
    out, seg, n_pause, n_drop, n_repeat, n_same_t = [], 0, 0, 0, 0, 0
    for i, (idx, lon, lat, t) in enumerate(dedup):
        if i == 0:
            dt = step = spd = None; gap = 0; gtype = "none"; rep = 0
        else:
            _, plon, plat, pt = dedup[i - 1]
            dt = t - pt
            step = haversine_m(plon, plat, lon, lat)
            spd = step / dt * 3.6 if dt > 0 else None
            rep = 1 if (lon == plon and lat == plat) else 0
            n_same_t += 1 if dt == 0 else 0
            if dt > BIG_GAP_S:
                if step < PAUSE_JUMP_M:
                    gap, gtype = 1, "pause"; n_pause += 1
                else:
                    gap, gtype = 2, "dropout"; n_drop += 1; seg += 1
            else:
                gap, gtype = 0, "none"
            n_repeat += rep
        out.append({
            "orig_idx": idx, "lon": lon, "lat": lat, "t": t,
            "dt_s": "" if dt is None else dt,
            "dt_capped_s": "" if dt is None else min(dt, DT_CAP_S),
            "step_m": "" if step is None else round(step, 2),
            "speed_kmh": "" if spd is None else round(spd, 2),
            "gap_flag": gap, "gap_type": gtype, "segment_id": seg, "is_repeat_coord": rep,
        })
    st.update({"n_clean": len(out), "n_pause_gaps": n_pause, "n_dropout_gaps": n_drop,
               "n_segments": seg + 1 if out else 0, "n_repeat_coords_flagged": n_repeat,
               "n_same_timestamp_diff_xy": n_same_t})
    st["pct_points_retained"] = round(100 * len(out) / len(rows), 3) if rows else ""
    return out, st


# ----------------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------------
def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _svg_open(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'font-family="sans-serif" font-size="11">',
            f'<rect width="{w}" height="{h}" fill="white"/>',
            f'<text x="{w/2}" y="22" text-anchor="middle" font-size="14">{title}</text>']


def svg_hist(path, values, bins, title, xlabel, logy=False, color="#4477aa"):
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / (hi - lo) * bins), bins - 1)] += 1
    f = (lambda c: math.log10(c + 1)) if logy else (lambda c: c)
    top = max(f(c) for c in counts) or 1
    W, H, L, B, T = 640, 360, 60, 50, 40
    bw = (W - L - 20) / bins
    o = _svg_open(W, H, title)
    for i, c in enumerate(counts):
        h = f(c) / top * (H - B - T)
        o.append(f'<rect x="{L + i*bw:.1f}" y="{H-B-h:.1f}" width="{bw-1:.1f}" height="{h:.1f}" fill="{color}"/>')
    o += [f'<line x1="{L}" y1="{H-B}" x2="{W-20}" y2="{H-B}" stroke="black"/>',
          f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="black"/>',
          f'<text x="{L}" y="{H-B+16}">{lo:.4g}</text>',
          f'<text x="{W-20}" y="{H-B+16}" text-anchor="end">{hi:.4g}</text>',
          f'<text x="{W/2}" y="{H-8}" text-anchor="middle">{xlabel}</text>',
          f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle">'
          f'count ({"log10(c+1)" if logy else "linear"}), max={max(counts)}</text>', "</svg>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(o), encoding="utf-8")


def svg_bars(path, labels, values, title, ylabel, color="#cc6677"):
    W, H, L, B, T = 560, 340, 70, 60, 40
    top = max(values) or 1
    bw = (W - L - 20) / len(values)
    o = _svg_open(W, H, title)
    for i, (lab, v) in enumerate(zip(labels, values)):
        h = v / top * (H - B - T)
        x = L + i * bw + bw * 0.15
        o.append(f'<rect x="{x:.1f}" y="{H-B-h:.1f}" width="{bw*0.7:.1f}" height="{h:.1f}" fill="{color}"/>')
        o.append(f'<text x="{x + bw*0.35:.1f}" y="{H-B-h-4:.1f}" text-anchor="middle">{v}</text>')
        o.append(f'<text x="{x + bw*0.35:.1f}" y="{H-B+16}" text-anchor="middle">{lab}</text>')
    o += [f'<line x1="{L}" y1="{H-B}" x2="{W-20}" y2="{H-B}" stroke="black"/>',
          f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle">{ylabel}</text>', "</svg>"]
    path.write_text("\n".join(o), encoding="utf-8")


def svg_locations(path, quality):
    W = H = 560
    rows = [q for q in quality if "lon_min" in q]
    x0 = min(q["lon_min"] for q in rows); x1 = max(q["lon_max"] for q in rows)
    y0 = min(q["lat_min"] for q in rows); y1 = max(q["lat_max"] for q in rows)
    sx = lambda x: 30 + (x - x0) / (x1 - x0) * (W - 60)
    sy = lambda y: H - 30 - (y - y0) / (y1 - y0) * (H - 60)
    o = _svg_open(W, H, "Trip locations (blue = train, orange = student_test)")
    for q in rows:
        c = "#4477aa" if q["partition"] == "train" else "#ee8822"
        o.append(f'<circle cx="{sx((q["lon_min"]+q["lon_max"])/2):.1f}" '
                 f'cy="{sy((q["lat_min"]+q["lat_max"])/2):.1f}" r="4" fill="{c}" fill-opacity="0.7"/>')
    o.append(f'<text x="30" y="{H-8}">lon {x0:.1f} .. {x1:.1f}</text>'
             f'<text x="{W-30}" y="{H-8}" text-anchor="end">lat {y0:.1f} .. {y1:.1f}</text></svg>')
    path.write_text("\n".join(o), encoding="utf-8")


def svg_trajectory(path, clean_rows, title):
    """One trip: blue = normal points, orange = after a pause, red = after a dropout."""
    W = H = 560
    lon0 = sum(r["lon"] for r in clean_rows) / len(clean_rows)
    lat0 = sum(r["lat"] for r in clean_rows) / len(clean_rows)
    k = math.cos(math.radians(lat0))                   # shrink longitude to keep shape true
    xs = [(r["lon"] - lon0) * k for r in clean_rows]
    ys = [r["lat"] - lat0 for r in clean_rows]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1e-6
    sx = lambda x: 40 + (x - min(xs)) / span * (W - 80)
    sy = lambda y: H - 40 - (y - min(ys)) / span * (H - 80)
    o = _svg_open(W, H, title)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    o.append(f'<polyline points="{pts}" fill="none" stroke="#bbbbbb" stroke-width="1"/>')
    col = {0: "#4477aa", 1: "#ee8822", 2: "#cc3311"}
    for x, y, r in zip(xs, ys, clean_rows):
        rad = 1.5 if r["gap_flag"] == 0 else 6
        o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{rad}" fill="{col[r["gap_flag"]]}"/>')
    for i, (lab, c) in enumerate([("normal point", "#4477aa"), ("after pause", "#ee8822"),
                                  ("after dropout", "#cc3311")]):
        o.append(f'<circle cx="{40}" cy="{44 + i*16}" r="5" fill="{c}"/>'
                 f'<text x="52" y="{48 + i*16}">{lab}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


# ----------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------
def run_all(data_root, out_root=".", write_cleaned=True, make_plots=True, verbose=True):
    data_root, out_root = Path(data_root), Path(out_root)
    quality, cleaning, all_dts, all_speeds, gap_sizes = [], [], [], [], []
    best_demo, best_score = None, -1

    for part in PARTITIONS:
        trips = sorted(d for d in (data_root / part).iterdir() if d.is_dir())
        if verbose:
            print(f"[{part}] {len(trips)} trips")
        for d in trips:
            tid = d.name
            raw = read_raw(d / f"{tid}.track")
            q = quality_trip(raw, part, tid, d)
            clean, st = clean_trip(raw)
            quality.append(q)
            cleaning.append({"partition": part, "trip_id": tid, **st})
            if write_cleaned:
                write_csv(out_root / "data" / "cleaned" / part / f"{tid}.csv", clean, CLEAN_COLS)
            for r in clean:
                if r["dt_s"] != "":
                    all_dts.append(r["dt_s"])
                    if r["dt_s"] > 1:
                        gap_sizes.append(r["dt_s"])
                if r["speed_kmh"] != "":
                    all_speeds.append(r["speed_kmh"])
            # demo trip for the trajectory plot: a readable one, with a few (not hundreds
            # of) flagged points so the gaps stand out against normal points
            flagged = st["n_dropout_gaps"] + st["n_pause_gaps"]
            score = st["n_dropout_gaps"] if (500 <= st["n_clean"] <= 3500 and 3 <= st["n_dropout_gaps"] <= 12
                                             and flagged <= 25) else -1
            if part == "train" and score > best_score:
                best_demo, best_score = (tid, clean), score

    write_csv(out_root / "results" / "gps_quality_summary.csv", quality)
    write_csv(out_root / "results" / "gps_cleaning_summary.csv", cleaning)

    if make_plots:
        fig = out_root / "eda" / "figures"
        fig.mkdir(parents=True, exist_ok=True)
        svg_hist(fig / "gps_obs_per_trip.svg", [q["n_raw"] for q in quality], 30,
                 "GPS observations per trip (train + student_test)", "observations")
        svg_hist(fig / "gps_sampling_interval.svg", [min(x, 30) for x in all_dts], 30,
                 "Sampling interval, clipped at 30 s (log counts)", "seconds between fixes",
                 logy=True, color="#228833")
        svg_hist(fig / "gps_gap_sizes.svg", [min(x, 60) for x in gap_sizes], 30,
                 "Gaps longer than 1 s, clipped at 60 s (log counts)", "gap (s)",
                 logy=True, color="#cc3311")
        svg_hist(fig / "gps_speed_distribution.svg", [min(x, 300) for x in all_speeds], 30,
                 "Implied speed, clipped at 300 km/h (log counts)", "km/h", logy=True, color="#aa3377")
        svg_hist(fig / "gps_path_length_km.svg", [q["path_length_km"] for q in quality if "path_length_km" in q],
                 25, "Path length per trip", "km")
        svg_bars(fig / "gps_gap_types.svg", ["pause", "dropout"],
                 [sum(c["n_pause_gaps"] for c in cleaning), sum(c["n_dropout_gaps"] for c in cleaning)],
                 f"Large gaps (> {BIG_GAP_S} s) by type", "number of gaps")
        svg_locations(fig / "gps_trip_locations.svg", quality)
        if best_demo:
            svg_trajectory(fig / "gps_sample_trajectory.svg", best_demo[1],
                           f"train/{best_demo[0]}: raw GPS with gap flags")

    if verbose:
        for part in PARTITIONS:
            c = [x for x in cleaning if x["partition"] == part]
            print(f"[{part}] raw={sum(x['n_raw'] for x in c)} clean={sum(x['n_clean'] for x in c)} "
                  f"invalid_removed={sum(x['n_invalid_removed'] for x in c)} "
                  f"dups_removed={sum(x['n_duplicates_removed'] for x in c)} "
                  f"pause_gaps={sum(x['n_pause_gaps'] for x in c)} "
                  f"dropout_gaps={sum(x['n_dropout_gaps'] for x in c)}")
    return {"quality": quality, "cleaning": cleaning,
            "pooled_pct_dt_eq_1s": round(100 * sum(1 for x in all_dts if x == 1) / len(all_dts), 2),
            "n_steps": len(all_dts)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPS quality checks + cleaning (stdlib only)")
    ap.add_argument("--data", default="data/student_folder")
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-cleaned-files", action="store_true", help="skip writing data/cleaned/")
    a = ap.parse_args()
    res = run_all(a.data, a.out, write_cleaned=not a.no_cleaned_files)
    print(f"Pooled share of steps exactly 1 s apart: {res['pooled_pct_dt_eq_1s']}%")
    print("Wrote results/gps_quality_summary.csv, results/gps_cleaning_summary.csv, "
          "data/cleaned/, eda/figures/gps_*.svg")

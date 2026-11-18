"""
Architha's EDA (GPS tracks) -- STANDARD LIBRARY ONLY (no numpy / pandas / matplotlib).
Use this when compiled packages are blocked by Application Control.

    python eda/architha_track_eda_stdlib.py --data data/student_folder

Outputs (CSV + simple SVG charts, no extra packages needed):
    results/track_inventory.csv, track_gap_events.csv, track_flagged_trips.csv,
    results/track_summary_by_partition.csv
    eda/figures/*.svg   (open in a browser; screenshot them for the report)

Inspects only; it does not modify or drop data.
"""
import argparse, csv, math, statistics
from pathlib import Path

PARTS = ["train", "student_test"]
EXTS = ["track", "nodes", "arcs", "route"]
BIG_GAP_S = 5
MAX_KMH = 200
PAUSE_JUMP_M = 50


def read_track(path):
    lon, lat, t = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()          # splits on tabs/spaces, strips CRLF
            if len(p) != 3:
                continue
            lon.append(float(p[0])); lat.append(float(p[1])); t.append(float(p[2]))
    return lon, lat, t


def hav_m(lo1, la1, lo2, la2):
    R = 6371000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def analyse(trip_dir, part):
    tid = trip_dir.name
    row = {"partition": part, "trip_id": tid}
    for e in EXTS:
        f = trip_dir / f"{tid}.{e}"
        row[f"has_{e}"] = f.exists()
        row[f"size_{e}_kb"] = round(f.stat().st_size / 1024, 1) if f.exists() else ""
    lon, lat, t = read_track(trip_dir / f"{tid}.track")
    n = len(t)
    row["n_obs"] = n
    row["n_nan"] = sum(1 for v in lon + lat + t if v != v)
    row["n_lat_invalid"] = sum(1 for v in lat if not -90 <= v <= 90)
    row["n_lon_invalid"] = sum(1 for v in lon if not -180 <= v <= 180)
    row["n_zero_coord"] = sum(1 for a, b in zip(lon, lat) if a == 0 and b == 0)
    row["n_duplicate_rows"] = n - len(set(zip(lon, lat, t)))

    dts = [t[i + 1] - t[i] for i in range(n - 1)]
    row["t_start"], row["t_end"] = t[0], t[-1]
    row["duration_s"] = t[-1] - t[0]
    row["t_monotonic"] = all(d > 0 for d in dts)
    row["n_dt_nonpositive"] = sum(1 for d in dts if d <= 0)
    row["pct_dt_eq_1s"] = round(100 * sum(1 for d in dts if d == 1) / len(dts), 2) if dts else ""
    row["n_gaps_gt1s"] = sum(1 for d in dts if d > 1)
    row["n_gaps_big"] = sum(1 for d in dts if d > BIG_GAP_S)
    row["max_gap_s"] = max(dts) if dts else ""
    row["median_dt_s"] = statistics.median(dts) if dts else ""

    steps = [hav_m(lon[i], lat[i], lon[i + 1], lat[i + 1]) for i in range(n - 1)]
    speeds = [s / d * 3.6 for s, d in zip(steps, dts) if d > 0]
    row["lon_min"], row["lon_max"] = min(lon), max(lon)
    row["lat_min"], row["lat_max"] = min(lat), max(lat)
    c_lon, c_lat = sum(lon) / n, sum(lat) / n
    zone = int((c_lon + 180) // 6) + 1
    row["centroid_lon"], row["centroid_lat"] = round(c_lon, 5), round(c_lat, 5)
    row["utm_epsg"] = (32600 if c_lat >= 0 else 32700) + zone   # project PER TRIP
    row["path_length_km"] = round(sum(steps) / 1000, 3)
    row["extent_ns_km"] = round(hav_m(row["lon_min"], row["lat_min"], row["lon_min"], row["lat_max"]) / 1000, 3)
    row["extent_ew_km"] = round(hav_m(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_min"]) / 1000, 3)
    row["mean_speed_kmh"] = round(sum(speeds) / len(speeds), 2) if speeds else ""
    row["max_speed_kmh"] = round(max(speeds), 2) if speeds else ""
    row["n_speed_implausible"] = sum(1 for v in speeds if v > MAX_KMH)

    gaps = []
    for i, d in enumerate(dts):
        if d > 1:
            kind = "pause_(no_movement)" if steps[i] < PAUSE_JUMP_M else "dropout_(position_jumped)"
            gaps.append({"partition": part, "trip_id": tid, "after_obs_index": i,
                         "gap_s": d, "jump_m": round(steps[i], 2), "kind": kind})
    row["n_gaps_pause"] = sum(1 for g in gaps if g["gap_s"] > BIG_GAP_S and g["kind"].startswith("pause"))
    row["n_gaps_dropout"] = sum(1 for g in gaps if g["gap_s"] > BIG_GAP_S and g["kind"].startswith("dropout"))

    fl = []
    if row["n_nan"]: fl.append("nan_values")
    if row["n_lat_invalid"] or row["n_lon_invalid"]: fl.append("invalid_coords")
    if row["n_zero_coord"]: fl.append("zero_coords")
    if not row["t_monotonic"]: fl.append("timestamps_not_strictly_increasing")
    if row["n_gaps_big"]: fl.append(f"big_gaps(>{BIG_GAP_S}s)")
    if row["n_speed_implausible"]: fl.append("implausible_speed")
    if n < 30: fl.append("very_short_trip")
    row["flags"] = ";".join(fl)
    return row, gaps, dts


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def svg_hist(path, values, bins, title, xlabel, logy=False, color="#4477aa"):
    """Minimal SVG histogram (no dependencies)."""
    lo, hi = min(values), max(values)
    if hi == lo: hi = lo + 1
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / (hi - lo) * bins), bins - 1)] += 1
    f = (lambda c: math.log10(c + 1)) if logy else (lambda c: c)
    top = max(f(c) for c in counts) or 1
    W, H, L, B, T = 640, 360, 60, 50, 40
    bw = (W - L - 20) / bins
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14">{title}</text>']
    for i, c in enumerate(counts):
        h = f(c) / top * (H - B - T)
        out.append(f'<rect x="{L + i*bw:.1f}" y="{H - B - h:.1f}" width="{bw-1:.1f}" height="{h:.1f}" fill="{color}"/>')
    out.append(f'<line x1="{L}" y1="{H-B}" x2="{W-20}" y2="{H-B}" stroke="black"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="black"/>')
    out.append(f'<text x="{L}" y="{H-B+16}">{lo:.4g}</text><text x="{W-20}" y="{H-B+16}" text-anchor="end">{hi:.4g}</text>')
    out.append(f'<text x="{W/2}" y="{H-8}" text-anchor="middle">{xlabel}</text>')
    ylab = f"count ({'log10(c+1)' if logy else 'linear'}), max={max(counts)}"
    out.append(f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle">{ylab}</text>')
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def svg_extents(path, rows):
    W = H = 560
    lo_x = min(r["lon_min"] for r in rows); hi_x = max(r["lon_max"] for r in rows)
    lo_y = min(r["lat_min"] for r in rows); hi_y = max(r["lat_max"] for r in rows)
    sx = lambda x: 30 + (x - lo_x) / (hi_x - lo_x) * (W - 60)
    sy = lambda y: H - 30 - (y - lo_y) / (hi_y - lo_y) * (H - 60)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W/2}" y="18" text-anchor="middle" font-size="13">Trip locations (blue=train, orange=student_test)</text>']
    for r in rows:
        c = "#4477aa" if r["partition"] == "train" else "#ee8822"
        out.append(f'<circle cx="{sx((r["lon_min"]+r["lon_max"])/2):.1f}" cy="{sy((r["lat_min"]+r["lat_max"])/2):.1f}" r="4" fill="{c}" fill-opacity="0.7"/>')
    out.append(f'<text x="30" y="{H-8}">lon {lo_x:.1f} .. {hi_x:.1f}</text><text x="{W-30}" y="{H-8}" text-anchor="end">lat {lo_y:.1f} .. {hi_y:.1f}</text></svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/student_folder")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)

    inv, gaps, all_dts = [], [], []
    for part in PARTS:
        trips = sorted(d for d in (root / part).iterdir() if d.is_dir())
        print(f"[{part}] {len(trips)} trip folders")
        for d in trips:
            r, g, dts = analyse(d, part)
            inv.append(r); gaps += g; all_dts += dts

    write_csv(out / "results/track_inventory.csv", inv)
    write_csv(out / "results/track_gap_events.csv", gaps,
              ["partition", "trip_id", "after_obs_index", "gap_s", "jump_m", "kind"])
    write_csv(out / "results/track_flagged_trips.csv", [r for r in inv if r["flags"]], list(inv[0].keys()))

    summ = []
    for part in PARTS:
        g = [r for r in inv if r["partition"] == part]
        obs = [r["n_obs"] for r in g]
        summ.append({
            "partition": part, "trips": len(g), "obs_total": sum(obs), "obs_min": min(obs),
            "obs_median": statistics.median(obs), "obs_max": max(obs),
            "nan_total": sum(r["n_nan"] for r in g), "dup_rows": sum(r["n_duplicate_rows"] for r in g),
            "non_monotonic_trips": sum(1 for r in g if not r["t_monotonic"]),
            "mean_pct_dt_eq_1s": round(statistics.mean(r["pct_dt_eq_1s"] for r in g), 2),
            "trips_with_big_gaps": sum(1 for r in g if r["n_gaps_big"] > 0),
            "max_gap_s": max(r["max_gap_s"] for r in g),
            "max_speed_kmh": max(r["max_speed_kmh"] for r in g),
            "trips_flagged": sum(1 for r in g if r["flags"]),
        })
    write_csv(out / "results/track_summary_by_partition.csv", summ)

    fig = out / "eda/figures"
    svg_hist(fig / "track_obs_per_trip.svg", [r["n_obs"] for r in inv], 30,
             "Observations per trip (train + student_test)", "GPS observations", logy=False)
    svg_hist(fig / "track_sampling_interval.svg", [min(d, 30) for d in all_dts], 30,
             "Sampling interval (clipped at 30 s)", "seconds between fixes", logy=True, color="#228833")
    if gaps:
        svg_hist(fig / "track_gap_sizes.svg", [min(g["gap_s"], 60) for g in gaps], 30,
                 "Gap sizes (>1 s, clipped at 60 s)", "gap (s)", logy=True, color="#cc3311")
    svg_hist(fig / "track_path_length_km.svg", [r["path_length_km"] for r in inv], 25,
             "Path length per trip", "km")
    svg_extents(fig / "track_trip_locations.svg", inv)

    print("\n=== FILE AVAILABILITY ===")
    miss = [r for r in inv if not all(r[f"has_{e}"] for e in EXTS)]
    print("none missing" if not miss else [(r["partition"], r["trip_id"]) for r in miss])
    print("\n=== SUMMARY BY PARTITION ===")
    for s in summ:
        print(s)
    zones = {}
    for r in inv: zones[r["utm_epsg"]] = zones.get(r["utm_epsg"], 0) + 1
    print(f"\n=== UTM zones across trips: {len(zones)} distinct -> project PER TRIP ===")
    print(dict(sorted(zones.items(), key=lambda kv: -kv[1])))
    big = [g for g in gaps if g["gap_s"] > BIG_GAP_S]
    print(f"\n=== BIG GAPS (>{BIG_GAP_S}s): {len(big)} total ===")
    print("pause:", sum(1 for g in big if g["kind"].startswith("pause")),
          "| dropout:", sum(1 for g in big if g["kind"].startswith("dropout")))
    fl = [r for r in inv if r["flags"]]
    print(f"\n=== FLAGGED TRIPS: {len(fl)} (see results/track_flagged_trips.csv) ===")
    print("\nCLEANING DECISIONS TO MAKE: pauses, dropouts, speed spikes -> log in data_quality_log.md")


if __name__ == "__main__":
    main()
"""Spatial proximity and route-topology analysis for student PathSnap trips.

Distances are nearest distances to the supplied reference-route segments in a
trip-local WGS 84 / UTM CRS. They are spatial proximity proxies, not temporal
GPS-to-arc matches or validated GPS measurement error.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import find_dataset_root, load_trip
from src.geospatial import (
    project_coordinates,
    utm_crs_for_trip,
    utm_zone_for_location,
)

PARTITIONS = ("train", "student_test")
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "eda" / "figures"
TRAIN_PERCENTILES = (50, 75, 90, 95, 99)
DISTANCE_COLUMNS = [
    "partition",
    "trip_id",
    "gps_row",
    "timestamp_seconds",
    "longitude",
    "latitude",
    "nearest_route_distance_m",
    "crs_epsg",
    "route_arc_occurrences",
    "unique_route_arc_count",
    "route_continuity_break_count",
    "route_geometry_type",
]
CYCLE_COLUMNS = [
    "partition",
    "trip_id",
    "continuous_block",
    "node_id",
    "first_route_pos",
    "last_route_pos",
    "cycle_start_pos",
    "cycle_end_pos",
    "cycle_arc_count",
]
REVISIT_COLUMNS = [
    "partition",
    "trip_id",
    "arc_id",
    "previous_route_pos",
    "route_pos",
    "gap_positions",
]
BREAK_COLUMNS = [
    "partition",
    "trip_id",
    "route_pos",
    "arc_a",
    "arc_b",
    "to_node_a",
    "from_node_b",
]


def _validate_trip_id(trip_id):
    if len(trip_id) != 8 or not trip_id.isdigit():
        raise ValueError(f"Unexpected trip folder name {trip_id!r}; expected 8 digits.")


def _point_to_segment_distances(points, starts, ends, chunk_size=128):
    """Compute each point's minimum planar distance to a set of line segments."""
    if len(points) == 0:
        return np.empty(0, dtype=float)
    if len(starts) == 0:
        raise ValueError("Cannot measure distance to an empty route geometry.")

    vectors = ends - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    result = np.empty(len(points), dtype=float)

    for offset in range(0, len(points), chunk_size):
        point_chunk = points[offset : offset + chunk_size]
        displacement = point_chunk[:, None, :] - starts[None, :, :]
        projections = np.einsum("ijk,jk->ij", displacement, vectors)
        parameters = np.zeros_like(projections)
        np.divide(
            projections,
            squared_lengths[None, :],
            out=parameters,
            where=squared_lengths[None, :] > 0,
        )
        np.clip(parameters, 0.0, 1.0, out=parameters)
        nearest = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
        deltas = point_chunk[:, None, :] - nearest
        squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        result[offset : offset + len(point_chunk)] = np.sqrt(
            squared_distances.min(axis=1)
        )
    return result


def _route_topology(partition, trip_id, route, arcs, nodes):
    route_arc_ids = route["arc_id"].to_numpy(dtype=np.int64)
    if len(route_arc_ids) == 0:
        raise ValueError(f"{partition}/{trip_id} has an empty reference route.")

    arc_table = arcs.set_index("arc_id", verify_integrity=True)
    missing_route_ids = np.setdiff1d(np.unique(route_arc_ids), arc_table.index)
    if len(missing_route_ids):
        raise ValueError(
            f"{partition}/{trip_id} route contains invalid arc IDs: "
            f"{missing_route_ids[:10].tolist()}"
        )

    route_edges = arc_table.loc[route_arc_ids]
    starts = route_edges["from_node_id"].to_numpy(dtype=np.int64)
    ends = route_edges["to_node_id"].to_numpy(dtype=np.int64)
    invalid_endpoints = (
        (starts < 0)
        | (starts >= len(nodes))
        | (ends < 0)
        | (ends >= len(nodes))
    )
    if invalid_endpoints.any():
        raise ValueError(
            f"{partition}/{trip_id} route uses arcs with invalid node references."
        )

    break_positions = np.flatnonzero(ends[:-1] != starts[1:])
    revisit_rows = []
    last_arc_position = {}
    for route_pos, arc_id in enumerate(route_arc_ids):
        previous = last_arc_position.get(int(arc_id))
        if previous is not None:
            revisit_rows.append(
                {
                    "partition": partition,
                    "trip_id": trip_id,
                    "arc_id": int(arc_id),
                    "previous_route_pos": previous,
                    "route_pos": route_pos,
                    "gap_positions": route_pos - previous,
                }
            )
        last_arc_position[int(arc_id)] = route_pos

    cycle_rows = []
    block_ranges = []
    block_start = 0
    for break_pos in break_positions:
        block_ranges.append((block_start, int(break_pos) + 1))
        block_start = int(break_pos) + 1
    block_ranges.append((block_start, len(route_arc_ids)))

    for block_id, (block_start, block_end) in enumerate(block_ranges):
        if block_start >= block_end:
            continue
        block_nodes = np.concatenate(
            ([starts[block_start]], ends[block_start:block_end])
        )
        last_node_position = {}
        for node_position, node_id in enumerate(block_nodes):
            node_id = int(node_id)
            previous = last_node_position.get(node_id)
            if previous is not None:
                cycle_rows.append(
                    {
                        "partition": partition,
                        "trip_id": trip_id,
                        "continuous_block": block_id,
                        "node_id": node_id,
                        "first_route_pos": block_start + previous,
                        "last_route_pos": block_start + node_position,
                        "cycle_start_pos": block_start + previous,
                        "cycle_end_pos": block_start + node_position - 1,
                        "cycle_arc_count": node_position - previous,
                    }
                )
            last_node_position[node_id] = node_position

    return {
        "route_arc_ids": route_arc_ids,
        "route_edges": route_edges,
        "from_node_ids": starts,
        "to_node_ids": ends,
        "break_positions": break_positions,
        "block_ranges": block_ranges,
        "arc_revisits": revisit_rows,
        "node_cycles": cycle_rows,
    }


def _validate_projection(track, nodes, crs, projected_track):
    epsg = crs.to_epsg()
    if epsg is None or not crs.is_projected:
        raise ValueError(f"Expected a resolvable projected CRS, got {crs}.")
    if not crs.axis_info or crs.axis_info[0].unit_name.lower() not in {
        "metre",
        "meter",
    }:
        raise ValueError(f"CRS EPSG:{epsg} is not expressed in metres.")
    if not np.isclose(crs.axis_info[0].unit_conversion_factor, 1.0):
        raise ValueError(f"CRS EPSG:{epsg} does not use a metre scale factor.")

    valid = np.isfinite(projected_track[["x", "y"]].to_numpy()).all(axis=1)
    sample_rows = np.flatnonzero(valid)
    if len(sample_rows) == 0:
        raise ValueError("No track coordinates were successfully projected.")
    sample_rows = np.unique(
        np.linspace(
            0, len(sample_rows) - 1, num=min(9, len(sample_rows)), dtype=int
        )
    )
    source_rows = np.flatnonzero(valid)[sample_rows]
    inverse = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon_back, lat_back = inverse.transform(
        projected_track["x"].to_numpy()[source_rows],
        projected_track["y"].to_numpy()[source_rows],
    )
    source_lon = track["longitude"].to_numpy()[source_rows]
    source_lat = track["latitude"].to_numpy()[source_rows]
    _, _, errors_m = Geod(ellps="WGS84").inv(
        source_lon, source_lat, lon_back, lat_back
    )
    max_error_m = float(np.max(np.abs(errors_m)))
    if not np.isfinite(max_error_m) or max_error_m > 0.05:
        raise ValueError(
            f"EPSG:{epsg} inverse-transform check exceeded 0.05 m "
            f"(observed {max_error_m:.6f} m)."
        )

    return {
        "crs_epsg": int(epsg),
        "crs_name": crs.name,
        "crs_unit": crs.axis_info[0].unit_name,
        "max_roundtrip_error_m": max_error_m,
        "projected_x_min_m": float(projected_track.loc[valid, "x"].min()),
        "projected_x_max_m": float(projected_track.loc[valid, "x"].max()),
        "projected_y_min_m": float(projected_track.loc[valid, "y"].min()),
        "projected_y_max_m": float(projected_track.loc[valid, "y"].max()),
    }


def _analyse_trip(partition, trip_id, data_root):
    _validate_trip_id(trip_id)
    track, nodes, arcs, route = load_trip(
        trip_id, partition, data_dir=data_root
    )
    topology = _route_topology(partition, trip_id, route, arcs, nodes)

    crs = utm_crs_for_trip(track)
    projected_track = project_coordinates(track, crs)
    projected_nodes = project_coordinates(nodes, crs)
    projection_row = _validate_projection(track, nodes, crs, projected_track)

    node_coordinates = projected_nodes.set_index("node_id")[["x", "y"]]
    unique_route_arc_ids = pd.unique(topology["route_arc_ids"])
    unique_route_edges = arcs.set_index("arc_id").loc[unique_route_arc_ids]
    from_ids = unique_route_edges["from_node_id"].to_numpy(dtype=np.int64)
    to_ids = unique_route_edges["to_node_id"].to_numpy(dtype=np.int64)
    starts = node_coordinates.loc[from_ids].to_numpy(dtype=float)
    ends = node_coordinates.loc[to_ids].to_numpy(dtype=float)
    valid_segments = (
        np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1)
    )
    starts, ends = starts[valid_segments], ends[valid_segments]
    if len(starts) == 0:
        raise ValueError(
            f"{partition}/{trip_id} has no route segments with valid coordinates."
        )

    point_xy = projected_track[["x", "y"]].to_numpy(dtype=float)
    valid_points = np.isfinite(point_xy).all(axis=1)
    distances = np.full(len(track), np.nan, dtype=float)
    distances[valid_points] = _point_to_segment_distances(
        point_xy[valid_points], starts, ends
    )

    route_arc_ids = topology["route_arc_ids"]
    route_arc_repeat_count = len(topology["arc_revisits"])
    cycle_rows = topology["node_cycles"]
    route_row = {
        "partition": partition,
        "trip_id": trip_id,
        "track_points": len(track),
        "route_arc_occurrences": len(route_arc_ids),
        "unique_route_arc_count": len(unique_route_arc_ids),
        "repeated_arc_occurrences": route_arc_repeat_count,
        "consecutive_duplicate_arc_count": int(
            np.count_nonzero(route_arc_ids[:-1] == route_arc_ids[1:])
        ),
        "continuity_break_count": len(topology["break_positions"]),
        "continuous_block_count": len(topology["block_ranges"]),
        "node_return_cycle_count": len(cycle_rows),
        "max_cycle_arc_count": max(
            (row["cycle_arc_count"] for row in cycle_rows), default=0
        ),
        "invalid_projected_track_points": int((~valid_points).sum()),
        "route_arcs_with_invalid_node_geometry": int((~valid_segments).sum()),
        "utm_zone_count_for_track": len(
            {
                utm_zone_for_location(lon, lat)
                for lon, lat in zip(
                    track["longitude"].to_numpy(),
                    track["latitude"].to_numpy(),
                )
                if np.isfinite(lon)
                and np.isfinite(lat)
                and -80 <= lat <= 84
                and -180 <= lon <= 180
            }
        ),
    }
    projection_row.update(
        {
            "partition": partition,
            "trip_id": trip_id,
            "track_points": len(track),
            "projected_track_points": int(valid_points.sum()),
            "projected_route_segments": len(starts),
        }
    )

    distance_rows = pd.DataFrame(
        {
            "partition": partition,
            "trip_id": trip_id,
            "gps_row": track.index.to_numpy(dtype=np.int64),
            "timestamp_seconds": track["timestamp_seconds"].to_numpy(),
            "longitude": track["longitude"].to_numpy(),
            "latitude": track["latitude"].to_numpy(),
            "nearest_route_distance_m": distances,
            "crs_epsg": projection_row["crs_epsg"],
            "route_arc_occurrences": len(route_arc_ids),
            "unique_route_arc_count": len(unique_route_arc_ids),
            "route_continuity_break_count": len(topology["break_positions"]),
            "route_geometry_type": "straight_from_node_to_node",
        }
    )
    return (
        route_row,
        projection_row,
        distance_rows,
        topology["arc_revisits"],
        cycle_rows,
        topology["break_positions"],
        topology["route_edges"],
    )


def _verify_existing_route_tables(route_rows, break_rows):
    validation_path = RESULTS_DIR / "arun_route_validation.csv"
    breaks_path = RESULTS_DIR / "arun_route_breaks.csv"
    if not validation_path.is_file() or not breaks_path.is_file():
        raise FileNotFoundError(
            "Expected Arun route-validation baselines were not found in results/."
        )

    baseline_validation = pd.read_csv(
        validation_path, dtype={"trip_id": str}
    ).set_index(["partition", "trip_id"])
    computed_validation = pd.DataFrame(route_rows).set_index(
        ["partition", "trip_id"]
    )
    for column, computed_column in (
        ("n_route", "route_arc_occurrences"),
        ("n_breaks", "continuity_break_count"),
    ):
        if not baseline_validation[column].sort_index().equals(
            computed_validation[computed_column].sort_index()
        ):
            raise ValueError(
                f"Regenerated route values differ from existing {column} baseline."
            )

    baseline_breaks = pd.read_csv(breaks_path, dtype={"trip_id": str})
    baseline_keys = baseline_breaks[BREAK_COLUMNS].sort_values(
        ["partition", "trip_id", "route_pos"]
    ).reset_index(drop=True)
    computed_keys = pd.DataFrame(break_rows, columns=BREAK_COLUMNS).sort_values(
        ["partition", "trip_id", "route_pos"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        baseline_keys, computed_keys, check_dtype=False, check_exact=True
    )
    return "Existing route counts and continuity-break rows matched."


def _distance_summaries(distance_frame):
    finite = distance_frame.loc[
        np.isfinite(distance_frame["nearest_route_distance_m"])
    ].copy()
    rows = []
    partition_bounds = {}

    for partition, group in finite.groupby("partition", sort=False):
        values = group["nearest_route_distance_m"].to_numpy()
        q1, q3 = np.percentile(values, [25, 75])
        lower = max(0.0, float(q1 - 1.5 * (q3 - q1)))
        upper = float(q3 + 1.5 * (q3 - q1))
        partition_bounds[partition] = (lower, upper)

    def add_summary(level, partition, trip_id, group):
        values = group["nearest_route_distance_m"].to_numpy()
        lower, upper = partition_bounds[partition]
        percentiles = np.percentile(values, [25, 50, 75, 90, 95, 99])
        outlier_count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "summary_level": level,
                "partition": partition,
                "trip_id": trip_id,
                "n_gps_points": len(values),
                "min_distance_m": float(values.min()),
                "mean_distance_m": float(values.mean()),
                "std_distance_m": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "p25_distance_m": float(percentiles[0]),
                "median_distance_m": float(percentiles[1]),
                "p75_distance_m": float(percentiles[2]),
                "p90_distance_m": float(percentiles[3]),
                "p95_distance_m": float(percentiles[4]),
                "p99_distance_m": float(percentiles[5]),
                "max_distance_m": float(values.max()),
                "iqr_outlier_lower_fence_m": lower,
                "iqr_outlier_upper_fence_m": upper,
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_fraction": outlier_count / len(values),
                "measurement_interpretation": (
                    "nearest route geometry proxy; not GPS error"
                ),
            }
        )

    for partition, group in finite.groupby("partition", sort=False):
        add_summary("partition", partition, "", group)
        for trip_id, trip_group in group.groupby("trip_id", sort=False):
            add_summary("trip", partition, trip_id, trip_group)
    return pd.DataFrame(rows), partition_bounds


def _candidate_coverage(distance_frame):
    train_values = distance_frame.loc[
        (distance_frame["partition"] == "train")
        & np.isfinite(distance_frame["nearest_route_distance_m"]),
        "nearest_route_distance_m",
    ].to_numpy()
    if len(train_values) == 0:
        raise ValueError("No finite training distances are available for radii.")

    radius_options = [
        (
            percentile,
            float(math.ceil(np.percentile(train_values, percentile))),
            "ceiling of training nearest-route distance percentile",
        )
        for percentile in TRAIN_PERCENTILES
    ]
    radius_options.append(
        (
            np.nan,
            50.0,
            "existing data-quality-log radius; comparison only, not a recommendation",
        )
    )
    rows = []
    for percentile, radius, basis in radius_options:
        for partition, group in distance_frame.groupby("partition", sort=False):
            values = group["nearest_route_distance_m"].to_numpy()
            values = values[np.isfinite(values)]
            within = int(np.count_nonzero(values <= radius))
            outside = len(values) - within
            rows.append(
                {
                    "radius_source_percentile_train": percentile,
                    "radius_m": radius,
                    "radius_basis": basis,
                    "partition": partition,
                    "n_valid_gps_points": len(values),
                    "within_reference_route_geometry": within,
                    "route_geometry_coverage_fraction": within / len(values),
                    "outside_reference_route_geometry": outside,
                    "route_geometry_outside_fraction": outside / len(values),
                    "interpretation": (
                        "proximity coverage only; not true candidate recall or miss rate"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_distance_distribution(distance_frame, output_path):
    colors = {"train": "#2369a1", "student_test": "#d97925"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    all_values = distance_frame["nearest_route_distance_m"].dropna().to_numpy()
    plot_max = max(float(np.percentile(all_values, 99)), 1.0)
    bins = np.linspace(0.0, plot_max, 81)

    for partition in PARTITIONS:
        values = distance_frame.loc[
            distance_frame["partition"] == partition,
            "nearest_route_distance_m",
        ].dropna().to_numpy()
        axes[0].hist(
            values,
            bins=bins,
            alpha=0.55,
            color=colors[partition],
            label=partition,
        )
        ordered = np.sort(values)
        axes[1].plot(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            color=colors[partition],
            label=partition,
        )
    axes[0].set(
        title="Nearest-route distance (through pooled P99)",
        xlabel="Distance to nearest reference-route segment (m)",
        ylabel="GPS observations",
    )
    axes[0].text(
        0.98,
        0.96,
        "Histogram omits values above pooled P99;\nsummary statistics retain all observations.",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    axes[1].set(
        title="Empirical cumulative distribution",
        xlabel="Distance to nearest reference-route segment (m)",
        ylabel="Fraction of GPS observations",
    )
    axes[1].set_xscale("symlog", linthresh=1.0)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Spatial proximity proxy, not temporal GPS-to-arc error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_trip_distances(distance_frame, output_path):
    trip_rows = []
    for (partition, trip_id), group in distance_frame.groupby(
        ["partition", "trip_id"], sort=True
    ):
        values = group["nearest_route_distance_m"].dropna().to_numpy()
        trip_rows.append(
            {
                "partition": partition,
                "trip_id": trip_id,
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
        )
    trip_stats = pd.DataFrame(trip_rows)
    trip_stats = trip_stats.sort_values(["partition", "trip_id"]).reset_index(drop=True)
    colors = {"train": "#2369a1", "student_test": "#d97925"}
    fig, axis = plt.subplots(figsize=(13, 5))
    for partition in PARTITIONS:
        selected = trip_stats.loc[trip_stats["partition"] == partition]
        x = np.arange(len(selected))
        axis.scatter(
            x,
            selected["median"],
            color=colors[partition],
            marker="o",
            label=f"{partition} median",
        )
        axis.scatter(
            x,
            selected["p95"],
            color=colors[partition],
            marker="^",
            alpha=0.8,
            label=f"{partition} P95",
        )
    axis.set(
        title="Nearest-reference-route distance by trip",
        xlabel="Trip position within partition (sorted by trip ID)",
        ylabel="Distance (m)",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run_analysis(data_root=None, overwrite=False):
    data_root = find_dataset_root(data_root)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = [
        RESULTS_DIR / "arun_spatial_analysis.csv",
        RESULTS_DIR / "arun_distance_summary.csv",
        RESULTS_DIR / "arun_candidate_coverage.csv",
        RESULTS_DIR / "arun_route_cycle_summary.csv",
        RESULTS_DIR / "arun_route_node_cycles.csv",
        RESULTS_DIR / "arun_route_arc_revisits.csv",
        RESULTS_DIR / "arun_projection_validation.csv",
        FIGURES_DIR / "arun_route_distance_distribution.png",
        FIGURES_DIR / "arun_route_distance_by_trip.png",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; use --overwrite only after "
            "reviewing them:\n" + "\n".join(str(path) for path in existing)
        )

    route_rows = []
    projection_rows = []
    distance_frames = []
    all_arc_revisits = []
    all_node_cycles = []
    all_break_rows = []
    for partition in PARTITIONS:
        partition_dir = data_root / partition
        if not partition_dir.is_dir():
            raise FileNotFoundError(f"Required partition is missing: {partition_dir}")
        trip_dirs = sorted(path for path in partition_dir.iterdir() if path.is_dir())
        for trip_dir in trip_dirs:
            result = _analyse_trip(partition, trip_dir.name, data_root)
            (
                route_row,
                projection_row,
                distances,
                arc_revisits,
                node_cycles,
                break_positions,
                route_edges,
            ) = result
            route_rows.append(route_row)
            projection_rows.append(projection_row)
            distance_frames.append(distances)
            all_arc_revisits.extend(arc_revisits)
            all_node_cycles.extend(node_cycles)
            for break_pos in break_positions:
                all_break_rows.append(
                    {
                        "partition": partition,
                        "trip_id": trip_dir.name,
                        "route_pos": int(break_pos),
                        "arc_a": int(route_edges.index[break_pos]),
                        "arc_b": int(route_edges.index[break_pos + 1]),
                        "to_node_a": int(
                            route_edges.iloc[break_pos]["to_node_id"]
                        ),
                        "from_node_b": int(
                            route_edges.iloc[break_pos + 1]["from_node_id"]
                        ),
                    }
                )

    distance_frame = pd.concat(distance_frames, ignore_index=True)
    route_summary = pd.DataFrame(route_rows)
    projection_validation = pd.DataFrame(projection_rows)
    arc_revisits = pd.DataFrame(all_arc_revisits, columns=REVISIT_COLUMNS)
    node_cycles = pd.DataFrame(all_node_cycles, columns=CYCLE_COLUMNS)
    baseline_check = _verify_existing_route_tables(route_rows, all_break_rows)
    distance_summary, _ = _distance_summaries(distance_frame)
    candidate_coverage = _candidate_coverage(distance_frame)

    distance_frame = distance_frame[DISTANCE_COLUMNS]
    data_frames = {
        "arun_spatial_analysis.csv": distance_frame,
        "arun_distance_summary.csv": distance_summary,
        "arun_candidate_coverage.csv": candidate_coverage,
        "arun_route_cycle_summary.csv": route_summary,
        "arun_route_node_cycles.csv": node_cycles,
        "arun_route_arc_revisits.csv": arc_revisits,
        "arun_projection_validation.csv": projection_validation,
    }
    _plot_distance_distribution(
        distance_frame, FIGURES_DIR / "arun_route_distance_distribution.png"
    )
    _plot_trip_distances(
        distance_frame, FIGURES_DIR / "arun_route_distance_by_trip.png"
    )
    for name, frame in data_frames.items():
        frame.to_csv(RESULTS_DIR / name, index=False)

    print(f"Dataset root: {data_root}")
    print(f"Trips analyzed: {len(route_summary)}")
    print(baseline_check)
    print(
        "Route revisits:",
        int(route_summary["repeated_arc_occurrences"].sum()),
        "repeated arc occurrences across",
        int((route_summary["repeated_arc_occurrences"] > 0).sum()),
        "trips",
    )
    print(
        "Node-return cycle candidates:",
        int(route_summary["node_return_cycle_count"].sum()),
        "within continuous route blocks",
    )
    print(
        "GPS-to-route distance interpretation: nearest supplied route geometry "
        "proxy; no temporal arc assignment."
    )
    print(distance_summary.loc[
        distance_summary["summary_level"] == "partition",
        [
            "partition",
            "n_gps_points",
            "mean_distance_m",
            "median_distance_m",
            "p95_distance_m",
            "p99_distance_m",
            "max_distance_m",
        ],
    ].to_string(index=False))
    print("New tables:", ", ".join(data_frames))
    print("New figures: arun_route_distance_distribution.png, arun_route_distance_by_trip.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Dataset root containing train/ and student_test/.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs previously generated by this script.",
    )
    args = parser.parse_args()
    run_analysis(args.data_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

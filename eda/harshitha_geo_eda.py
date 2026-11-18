"""Harshitha's CRS, coordinate, trajectory, and interactive-map EDA."""

import argparse
import sys
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import find_dataset_root, load_trip
from src.geospatial import utm_crs_for_trip, utm_zone_for_location

PARTITIONS = ("train", "student_test")
FILE_EXTENSIONS = ("track", "nodes", "arcs", "route")
DEFAULT_MAP_TRIPS = (
    ("train", "00000002"),
    ("train", "00000046"),
    ("student_test", "00000000"),
)
FIGURES_DIR = REPO_ROOT / "eda" / "figures"
RESULTS_DIR = REPO_ROOT / "results"


def _valid_coordinates(frame):
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & longitude.between(-180, 180)
        & latitude.between(-90, 90)
    )
    return longitude, latitude, valid


def _trip_files(data_root, partition, trip_id):
    trip_dir = data_root / partition / trip_id
    return {ext: trip_dir / f"{trip_id}.{ext}" for ext in FILE_EXTENSIONS}


def _summarize_trip(data_root, partition, trip_id):
    track, nodes, arcs, route = load_trip(trip_id, partition, data_dir=data_root)
    files = _trip_files(data_root, partition, trip_id)
    longitude, latitude, valid = _valid_coordinates(track)
    valid_lon = longitude[valid]
    valid_lat = latitude[valid]
    node_longitude, node_latitude, valid_nodes = _valid_coordinates(nodes)
    valid_node_lon = node_longitude[valid_nodes]
    valid_node_lat = node_latitude[valid_nodes]

    zone_by_point = {
        utm_zone_for_location(lon, lat)
        for lon, lat in zip(valid_lon, valid_lat)
        if -80 <= lat <= 84
    }
    node_zones = {
        utm_zone_for_location(lon, lat)
        for lon, lat in zip(valid_node_lon, valid_node_lat)
        if -80 <= lat <= 84
    }
    crs = utm_crs_for_trip(track)
    record = {
        "partition": partition,
        "trip_id": trip_id,
        "track_points": len(track),
        "node_count": len(nodes),
        "arc_count": len(arcs),
        "route_arc_count": len(route),
        "invalid_track_coordinates": int((~valid).sum()),
        "longitude_min": float(valid_lon.min()) if len(valid_lon) else np.nan,
        "longitude_max": float(valid_lon.max()) if len(valid_lon) else np.nan,
        "latitude_min": float(valid_lat.min()) if len(valid_lat) else np.nan,
        "latitude_max": float(valid_lat.max()) if len(valid_lat) else np.nan,
        "invalid_node_coordinates": int((~valid_nodes).sum()),
        "node_longitude_min": (
            float(valid_node_lon.min()) if len(valid_node_lon) else np.nan
        ),
        "node_longitude_max": (
            float(valid_node_lon.max()) if len(valid_node_lon) else np.nan
        ),
        "node_latitude_min": (
            float(valid_node_lat.min()) if len(valid_node_lat) else np.nan
        ),
        "node_latitude_max": (
            float(valid_node_lat.max()) if len(valid_node_lat) else np.nan
        ),
        "utm_epsg": crs.to_epsg(),
        "utm_zone": int(crs.to_epsg() % 100),
        "track_utm_zone_count": len(zone_by_point),
        "track_crosses_utm_zone": len(zone_by_point) > 1,
        "node_utm_zone_count": len(node_zones),
        "nodes_cross_utm_zone": len(node_zones) > 1,
    }
    for extension, path in files.items():
        record[f"{extension}_bytes"] = path.stat().st_size
    record["total_bytes"] = sum(record[f"{ext}_bytes"] for ext in FILE_EXTENSIONS)
    return record


def _write_tables(data_root):
    trip_rows = []
    for partition in PARTITIONS:
        partition_dir = data_root / partition
        if not partition_dir.is_dir():
            raise FileNotFoundError(f"Dataset partition not found: {partition_dir}")
        trip_dirs = sorted(path for path in partition_dir.iterdir() if path.is_dir())
        trip_rows.extend(
            _summarize_trip(data_root, partition, trip_dir.name)
            for trip_dir in trip_dirs
        )
    if not trip_rows:
        raise FileNotFoundError(f"No trips found under {data_root}.")

    inventory = pd.DataFrame(trip_rows)
    inventory.to_csv(RESULTS_DIR / "harshitha_trip_inventory.csv", index=False)

    summaries = []
    for partition, rows in inventory.groupby("partition", sort=False):
        summaries.append({
            "partition": partition,
            "trip_count": len(rows),
            "total_track_points": int(rows["track_points"].sum()),
            "total_nodes": int(rows["node_count"].sum()),
            "total_arcs": int(rows["arc_count"].sum()),
            "total_route_arcs": int(rows["route_arc_count"].sum()),
            "total_file_bytes": int(rows["total_bytes"].sum()),
            "invalid_track_coordinates": int(
                rows["invalid_track_coordinates"].sum()
            ),
            "invalid_node_coordinates": int(
                rows["invalid_node_coordinates"].sum()
            ),
            "trips_crossing_utm_zone": int(rows["track_crosses_utm_zone"].sum()),
        })
    pd.DataFrame(summaries).to_csv(
        RESULTS_DIR / "harshitha_dataset_summary.csv", index=False
    )

    crs_summary = (
        inventory.groupby(["partition", "utm_epsg"], as_index=False)
        .agg(trip_count=("trip_id", "count"))
        .sort_values(["partition", "utm_epsg"])
    )
    crs_summary.to_csv(RESULTS_DIR / "harshitha_crs_summary.csv", index=False)
    return inventory


def _load_all_tracks(data_root):
    track_frames = []
    for partition in PARTITIONS:
        for trip_dir in sorted((data_root / partition).iterdir()):
            if not trip_dir.is_dir():
                continue
            track, _, _, _ = load_trip(
                trip_dir.name, partition, data_dir=data_root
            )
            longitude, latitude, valid = _valid_coordinates(track)
            coordinates = pd.DataFrame({
                "longitude": longitude[valid],
                "latitude": latitude[valid],
                "partition": partition,
            })
            track_frames.append(coordinates)
    return pd.concat(track_frames, ignore_index=True)


def _plot_coordinate_distribution(tracks):
    colors = {"train": "#2369a1", "student_test": "#d97925"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for partition in PARTITIONS:
        values = tracks.loc[tracks["partition"] == partition]
        axes[0].hist(
            values["longitude"], bins=70, alpha=0.55,
            color=colors[partition], label=partition,
        )
        axes[1].hist(
            values["latitude"], bins=70, alpha=0.55,
            color=colors[partition], label=partition,
        )
    axes[0].set(title="Longitude distribution", xlabel="Longitude (degrees)")
    axes[1].set(title="Latitude distribution", xlabel="Latitude (degrees)")
    for axis in axes:
        axis.set_ylabel("GPS observations")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("GPS coordinate distributions (WGS 84)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "harshitha_coordinate_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    for partition in PARTITIONS:
        values = tracks.loc[tracks["partition"] == partition]
        axis.scatter(
            values["longitude"], values["latitude"], s=2, alpha=0.18,
            color=colors[partition], label=partition, rasterized=True,
        )
    axis.set(
        title="GPS observations across the supplied dataset (WGS 84)",
        xlabel="Longitude (degrees)",
        ylabel="Latitude (degrees)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "harshitha_coordinate_coverage.png", dpi=180)
    plt.close(fig)


def _node_segments(nodes, arcs):
    if nodes.empty or arcs.empty:
        return np.array([]), np.array([])
    node_coordinates = nodes.set_index("node_id")[["longitude", "latitude"]]
    valid = (
        arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    )
    valid_arcs = arcs.loc[valid]
    if valid_arcs.empty:
        return np.array([]), np.array([])
    starts = node_coordinates.loc[valid_arcs["from_node_id"]].to_numpy()
    ends = node_coordinates.loc[valid_arcs["to_node_id"]].to_numpy()
    x = np.column_stack((starts[:, 0], ends[:, 0], np.full(len(starts), np.nan)))
    y = np.column_stack((starts[:, 1], ends[:, 1], np.full(len(starts), np.nan)))
    return x.ravel(), y.ravel()


def _plot_trajectory_examples(data_root):
    examples = (("train", "00000002"), ("student_test", "00000000"))
    fig, axes = plt.subplots(1, len(examples), figsize=(14, 6))
    for axis, (partition, trip_id) in zip(axes, examples):
        track, nodes, arcs, route = load_trip(
            trip_id, partition, data_dir=data_root
        )
        x, y = _node_segments(nodes, arcs)
        if len(x):
            axis.plot(x, y, color="#a6a6a6", linewidth=0.45, alpha=0.55,
                      label="Road network")
        route_arcs = route["arc_id"].to_numpy()
        valid_route = route_arcs[(route_arcs >= 0) & (route_arcs < len(arcs))]
        if len(valid_route):
            route_x, route_y = _node_segments(
                nodes, arcs.set_index("arc_id").loc[valid_route].reset_index()
            )
            axis.plot(route_x, route_y, color="#c43c39", linewidth=1.5,
                      alpha=0.9, label="Reference route")
        axis.plot(
            track["longitude"], track["latitude"],
            color="#2369a1", linewidth=0.9, alpha=0.9, label="GPS track",
        )
        axis.scatter(
            track["longitude"].iloc[[0, -1]],
            track["latitude"].iloc[[0, -1]],
            c=["#2a8c54", "#2a8c54"], s=18, zorder=5, label="Start / end",
        )
        axis.set_title(f"{partition} / {trip_id}")
        axis.set_xlabel("Longitude (degrees)")
        axis.set_ylabel("Latitude (degrees)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle("Example GPS trajectories and supplied road/reference layers")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "harshitha_trajectory_examples.png", dpi=180)
    plt.close(fig)


def _geojson_lines(nodes, arcs, arc_ids=None):
    node_coordinates = nodes.set_index("node_id")[["longitude", "latitude"]]
    selected_arcs = (
        arcs if arc_ids is None else arcs.set_index("arc_id").reindex(arc_ids)
    )
    features = []
    for arc in selected_arcs.itertuples(index=False):
        if pd.isna(arc.from_node_id) or pd.isna(arc.to_node_id):
            continue
        if not (0 <= arc.from_node_id < len(nodes)
                and 0 <= arc.to_node_id < len(nodes)):
            continue
        start = node_coordinates.loc[int(arc.from_node_id)].to_numpy()
        end = node_coordinates.loc[int(arc.to_node_id)].to_numpy()
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        features.append({
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(start[0]), float(start[1])],
                    [float(end[0]), float(end[1])],
                ],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _write_map(data_root, partition, trip_id):
    track, nodes, arcs, route = load_trip(
        trip_id, partition, data_dir=data_root
    )
    longitude, latitude, valid = _valid_coordinates(track)
    coordinates = list(
        zip(latitude[valid].astype(float), longitude[valid].astype(float))
    )
    if not coordinates:
        raise ValueError(f"{partition}/{trip_id} has no valid GPS coordinates.")
    center = [float(np.median(latitude[valid])), float(np.median(longitude[valid]))]
    map_view = folium.Map(
        location=center, zoom_start=13, control_scale=True,
        tiles="OpenStreetMap",
    )

    road_layer = folium.FeatureGroup(name="Road network", show=True)
    folium.GeoJson(
        _geojson_lines(nodes, arcs),
        name="Road network arcs",
        style_function=lambda _: {
            "color": "#808080", "weight": 1, "opacity": 0.55,
        },
    ).add_to(road_layer)
    road_layer.add_to(map_view)

    route_layer = folium.FeatureGroup(name="Reference route", show=True)
    route_ids = route["arc_id"].to_numpy()
    folium.GeoJson(
        _geojson_lines(nodes, arcs, route_ids),
        name="Reference route arcs",
        style_function=lambda _: {
            "color": "#d62728", "weight": 4, "opacity": 0.9,
        },
    ).add_to(route_layer)
    route_layer.add_to(map_view)

    gps_layer = folium.FeatureGroup(name="Raw GPS", show=True)
    folium.PolyLine(
        coordinates, color="#1675c1", weight=3, opacity=0.9,
        tooltip="Raw GPS trajectory",
    ).add_to(gps_layer)
    folium.CircleMarker(
        coordinates[0], radius=5, color="#238b45", fill=True,
        tooltip="Start",
    ).add_to(gps_layer)
    folium.CircleMarker(
        coordinates[-1], radius=5, color="#762a83", fill=True,
        tooltip="End",
    ).add_to(gps_layer)
    gps_layer.add_to(map_view)

    legend = """
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background: white; padding: 8px 11px; border: 1px solid #777;
                border-radius: 4px; font-size: 13px;">
      <b>Map layers</b><br>
      <span style="color:#1675c1;">&#9632;</span> Raw GPS<br>
      <span style="color:#808080;">&#9632;</span> Road network<br>
      <span style="color:#d62728;">&#9632;</span> Reference route
    </div>
    """
    map_view.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(map_view)
    output = FIGURES_DIR / f"harshitha_map_{partition}_{trip_id}.html"
    map_view.save(str(output))
    html = output.read_text(encoding="utf-8")
    output.write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n",
        encoding="utf-8",
    )


def run(data_root):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_root = find_dataset_root(data_root)
    inventory = _write_tables(data_root)
    tracks = _load_all_tracks(data_root)
    _plot_coordinate_distribution(tracks)
    _plot_trajectory_examples(data_root)

    known_trips = set(zip(inventory["partition"], inventory["trip_id"]))
    for partition, trip_id in DEFAULT_MAP_TRIPS:
        if (partition, trip_id) not in known_trips:
            raise FileNotFoundError(
                f"Required map example {partition}/{trip_id} is not in the dataset."
            )
        _write_map(data_root, partition, trip_id)

    all_lon = tracks["longitude"]
    all_lat = tracks["latitude"]
    zones = {
        utm_zone_for_location(lon, lat)
        for lon, lat in zip(all_lon, all_lat)
        if -80 <= lat <= 84
    }
    print(f"Dataset root: {data_root}")
    print(f"Trips: {len(inventory)}; valid GPS observations: {len(tracks)}")
    print(
        "GPS bounds (WGS 84): "
        f"lon {all_lon.min():.6f}..{all_lon.max():.6f}, "
        f"lat {all_lat.min():.6f}..{all_lat.max():.6f}"
    )
    print(f"UTM zones represented by GPS points: {len(zones)} ({sorted(zones)})")
    print(
        "Trips whose GPS points cross UTM zone boundaries: "
        f"{int(inventory['track_crosses_utm_zone'].sum())}"
    )
    print(f"Tables: {RESULTS_DIR}")
    print(f"Figures and maps: {FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        help="Dataset root or its parent directory; otherwise PATHSNAP_DATA and "
        "standard project/download locations are checked.",
    )
    args = parser.parse_args()
    run(args.data_root)


if __name__ == "__main__":
    main()

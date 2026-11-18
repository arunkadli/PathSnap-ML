"""
route_checks.py  --  Arun: reference-route validation helpers for PathSnap-ML.

Scope of this module (CRS-INDEPENDENT only):
    * reading one trip's raw files
    * validating the .route file against .arcs / .nodes
    * checking route continuity (to_node of arc i == from_node of arc i+1)
    * comparing route length with the number of GPS points

Nothing here converts coordinates or measures distances in metres. Those parts
need the team's projected CRS (Harshitha) and live in a later section.

NOTE: Harshitha is writing the shared loader (load_trip). `read_trip_raw` below
is a deliberately small stand-in; swap it for the shared loader when ready.

Raw formats (verified against the actual files, see notebook section 1):
    .track : lon, lat, timestamp_s   (whitespace separated, no header, CRLF)
    .nodes : lon, lat                (row index = node_id)
    .arcs  : from_node, to_node      (row index = arc_id)
    .route : arc_id                  (one integer per line)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test")  # evaluator data is never touched here
DEFAULT_DATA_ROOT = Path.home() / "Downloads" / "Student folder" / "Student folder"


def resolve_data_root(data_root=None):
    """Resolve an explicit path, PATHSNAP_DATA, or the user's downloaded data."""
    if data_root is not None:
        root = Path(data_root).expanduser()
    elif os.environ.get("PATHSNAP_DATA"):
        root = Path(os.environ["PATHSNAP_DATA"]).expanduser()
    else:
        root = DEFAULT_DATA_ROOT

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Set PATHSNAP_DATA to the "
            "directory containing train/ and student_test/."
        )
    return root


def list_trips(data_root=None, partitions=PARTITIONS):
    """Return [(partition, trip_id), ...] for every trip folder found."""
    data_root = resolve_data_root(data_root)
    unsupported = set(partitions) - set(PARTITIONS)
    if unsupported:
        raise ValueError(
            f"Unsupported partition(s): {sorted(unsupported)}. "
            f"Allowed partitions: {PARTITIONS}."
        )

    trips = []
    for part in partitions:
        partition_dir = data_root / part
        if not partition_dir.is_dir():
            raise FileNotFoundError(f"Dataset partition not found: {partition_dir}")
        for d in sorted(partition_dir.iterdir()):
            if d.is_dir():
                trips.append((part, d.name))
    return trips


def read_trip_raw(data_root=None, partition=None, trip_id=None):
    """
    Read the four raw files of one trip. Original files are only READ, never
    modified. sep=r'\\s+' copes with both tab separators and CRLF line endings.
    Returns (track, nodes, arcs, route) where route is a 1-D int numpy array.
    """
    if partition not in PARTITIONS:
        raise ValueError(
            f"Unsupported partition: {partition!r}. Allowed partitions: {PARTITIONS}."
        )
    if trip_id is None:
        raise ValueError("trip_id is required.")

    base = resolve_data_root(data_root) / partition / trip_id / trip_id
    track = pd.read_csv(f"{base}.track", sep=r"\s+", header=None,
                        names=["lon", "lat", "ts"])
    nodes = pd.read_csv(f"{base}.nodes", sep=r"\s+", header=None,
                        names=["lon", "lat"])
    arcs = pd.read_csv(f"{base}.arcs", sep=r"\s+", header=None,
                       names=["from_node", "to_node"])
    # Read the route as raw text tokens first, so a non-integer token would be
    # detected (not silently coerced to NaN/float by pandas).
    tokens = Path(f"{base}.route").read_text().split()
    try:
        route = np.array([int(t) for t in tokens], dtype=np.int64)
    except ValueError:
        route = None  # caller reports this as "route not all integers"
    return track, nodes, arcs, route


def validate_route(route, arcs, nodes):
    """
    Structural checks on one route. Returns a dict of counts.
      route_all_int     : every token parsed as an integer
      n_route           : number of arc ids in the route
      n_out_of_range    : ids < 0 or >= number of arcs (i.e. NOT in .arcs)
      n_consecutive_dup : arc id immediately repeated (i, i+1 identical)
      n_unique          : distinct arc ids in the route
      n_repeated_ids    : n_route - n_unique (arcs visited more than once)
      n_arc_node_oob    : arcs in .arcs pointing at a node id not in .nodes
    """
    out = {"route_all_int": route is not None}
    n_arcs, n_nodes = len(arcs), len(nodes)
    out["n_arc_node_oob"] = int(
        ((arcs.from_node < 0) | (arcs.from_node >= n_nodes) |
         (arcs.to_node < 0) | (arcs.to_node >= n_nodes)).sum())
    if route is None:
        return out
    out["n_route"] = len(route)
    out["n_out_of_range"] = int(((route < 0) | (route >= n_arcs)).sum())
    out["n_consecutive_dup"] = int((route[1:] == route[:-1]).sum())
    out["n_unique"] = len(np.unique(route))
    out["n_repeated_ids"] = out["n_route"] - out["n_unique"]
    return out


def find_continuity_breaks(route, arcs):
    """
    Continuity rule: to_node(route[i]) must equal from_node(route[i+1]).
    Returns a DataFrame with one row per break (empty if none):
        route_pos, arc_a, arc_b, to_node_a, from_node_b
    Only pairs where BOTH ids are valid arc ids are tested.
    """
    n = len(arcs)
    if n == 0 or len(route) < 2:
        return pd.DataFrame(
            columns=["route_pos", "arc_a", "arc_b", "to_node_a", "from_node_b"]
        )

    a, b = route[:-1], route[1:]
    valid = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    a_c, b_c = np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)
    to_a = arcs.to_node.values[a_c]
    from_b = arcs.from_node.values[b_c]
    broken = valid & (to_a != from_b)
    idx = np.where(broken)[0]
    return pd.DataFrame({
        "route_pos": idx,
        "arc_a": a[idx], "arc_b": b[idx],
        "to_node_a": to_a[idx], "from_node_b": from_b[idx],
    })


def repeat_gaps(route):
    """
    For arc ids that appear more than once, return the distance (in route
    positions) between successive occurrences. Small gaps (2, 3, ...) mean the
    vehicle goes back and forth; large gaps mean a genuine later revisit.
    """
    last_seen, gaps = {}, []
    for pos, arc in enumerate(route):
        if arc in last_seen:
            gaps.append(pos - last_seen[arc])
        last_seen[arc] = pos
    return np.array(gaps, dtype=int)

"""
Shared PathSnap trip loader.

Expected dataset layout (after extracting Student folder.zip):
    data/
      student_folder/
        train/<trip_id>/<trip_id>.track
        train/<trip_id>/<trip_id>.nodes
        train/<trip_id>/<trip_id>.arcs
        train/<trip_id>/<trip_id>.route
        student_test/<trip_id>/...

The loader returns four pandas DataFrames with meaningful column names.
Coordinates in the supplied files are stored as longitude, latitude.
No projection or coordinate transformation is performed here.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, Path]

# Raw-file schemas from the PathSnap Dataset Exploration guide.
TRACK_COLUMNS = ["longitude", "latitude", "timestamp_seconds"]
NODE_COLUMNS = ["longitude", "latitude"]
ARC_COLUMNS = ["from_node_id", "to_node_id"]


def find_dataset_root(data_dir: Optional[PathLike] = None) -> Path:
    """Find the directory containing partition folders (train/, student_test/)."""
    configured_root = (
        data_dir if data_dir is not None else os.environ.get("PATHSNAP_DATA")
    )
    if configured_root is not None:
        root = Path(configured_root).expanduser().resolve()
        candidates = [
            root,
            root / "student_folder",
            root / "Student folder",
            root / "Student_folder",
            root / "Student folder" / "Student folder",
        ]
    else:
        repo_root = Path(__file__).resolve().parents[1]
        data = repo_root / "data"
        candidates = [
            data / "student_folder",
            data / "Student folder",
            data / "Student_folder",
            data,
            repo_root.parent / "Student folder" / "Student folder",
            Path.home() / "Downloads" / "Student folder" / "Student folder",
        ]

    for candidate in candidates:
        if (candidate / "train").is_dir() or (candidate / "student_test").is_dir():
            return candidate

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find the PathSnap dataset root. Expected a directory containing "
        f"'train/' and/or 'student_test/'. Checked:\n{checked}\n"
        "Extract Student folder.zip into data/ and preferably rename the extracted "
        "folder to data/student_folder/."
    )


def _normalise_partition(partition: str) -> str:
    value = str(partition).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "training": "train",
        "train": "train",
        "studenttest": "student_test",
        "student_testing": "student_test",
        "student_test": "student_test",
        "test": "student_test",
        "evaluator": "evaluator_test",
        "evaluator_testing": "evaluator_test",
        "evaluator_test": "evaluator_test",
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown partition {partition!r}. Use 'train', 'student_test', "
            "or 'evaluator_test'."
        )
    return aliases[value]


def load_trip(
    trip_id: Union[str, int],
    partition: str = "train",
    data_dir: Optional[PathLike] = None,
):
    """
    Load one trip and return (track, nodes, arcs, route) as DataFrames.

    Parameters
    ----------
    trip_id:
        Trip folder/file identifier, e.g. "00000001". Leading zeroes are preserved
        when a string is supplied. Integer 1 is formatted as "00000001".
    partition:
        "train", "student_test", or "evaluator_test".
    data_dir:
        Optional path to the dataset root OR its parent data/ directory.
        If omitted, PATHSNAP_DATA and standard project/download locations are checked.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        track: longitude, latitude, timestamp_seconds
        nodes: node_id, longitude, latitude
        arcs: arc_id, from_node_id, to_node_id
        route: sequence, arc_id

    Notes
    -----
    The row order of .nodes and .arcs is significant: zero-based row position is
    the node_id/arc_id. The route file contains arc_id values in traversal order.
    """
    part = _normalise_partition(partition)
    if isinstance(trip_id, int):
        trip_name = f"{trip_id:08d}"
    else:
        trip_name = str(trip_id).strip()
        if not trip_name:
            raise ValueError("trip_id cannot be empty.")

    root = find_dataset_root(data_dir)
    trip_dir = root / part / trip_name
    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip directory not found: {trip_dir}\n"
            f"Check trip_id={trip_name!r} and partition={part!r}."
        )

    file_paths = {ext: trip_dir / f"{trip_name}.{ext}"
                  for ext in ("track", "nodes", "arcs", "route")}
    missing = [str(path) for path in file_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "This trip is missing one or more required files:\n  - "
            + "\n  - ".join(missing)
        )

    track = pd.read_csv(
        file_paths["track"], sep=r"\s+", header=None, names=TRACK_COLUMNS,
        dtype={"longitude": "float64", "latitude": "float64",
               "timestamp_seconds": "float64"},
    )
    nodes = pd.read_csv(
        file_paths["nodes"], sep=r"\s+", header=None, names=NODE_COLUMNS,
        dtype={"longitude": "float64", "latitude": "float64"},
    )
    arcs = pd.read_csv(
        file_paths["arcs"], sep=r"\s+", header=None, names=ARC_COLUMNS,
        dtype={"from_node_id": "int64", "to_node_id": "int64"},
    )
    route = pd.read_csv(
        file_paths["route"], sep=r"\s+", header=None, names=["arc_id"],
        dtype={"arc_id": "int64"},
    )

    # The raw row positions are the identifiers defined by the dataset guide.
    nodes.insert(0, "node_id", range(len(nodes)))
    arcs.insert(0, "arc_id", range(len(arcs)))
    route.insert(0, "sequence", range(len(route)))

    return track, nodes, arcs, route

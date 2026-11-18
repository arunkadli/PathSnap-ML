# PathSnap-ML

PathSnap-ML is a machine-learning project for map-matching noisy GPS observations to a road network and reconstructing routes.

## Current status

The current EDA work validates reference-route arc IDs, node references, route continuity, repeated arcs, and GPS-observation counts across the supplied `train` and `student_test` partitions. GPS-to-reference-arc alignment and metre-based noise analysis are pending the team's projected CRS decision and an agreed alignment methodology.

## Dataset setup

The original dataset is kept outside this repository. Prefer extracting the archive as:

```text
data\student_folder\train\
data\student_folder\student_test\
```

The shared loader also checks the dataset location documented by the instructor and the standard Downloads location. To use another location, set `PATHSNAP_DATA` to the dataset root (the directory containing `train\` and `student_test\`) or its parent in PowerShell:

```powershell
$env:PATHSNAP_DATA = 'C:\path\to\Student folder\Student folder'
```

Only the student-provided `train` and `student_test` partitions are read by the EDA; evaluator data is not accessed.

## Environment and EDA

From the project root, install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the shared geospatial EDA from the project root:

```powershell
python eda\harshitha_geo_eda.py
```

It writes trip/CRS tables to `results\`, coordinate and trajectory plots, and interactive Folium maps to `eda\figures\`. The loader in `src\data_loader.py` exposes `load_trip(trip_id, partition)` and returns `(track, nodes, arcs, route)` DataFrames. `src\geospatial.py` selects a WGS 84 / UTM CRS per trip from its median valid GPS position and projects coordinates when needed. The dataset spans multiple continents and UTM zones, so projected coordinates are local to each trip; do not compare metre coordinates across trips without transforming them to a common CRS.

Open `eda\arun_reference_route_analysis.ipynb` in VS Code and run all cells for route validation.

The route-check helpers are in `eda\route_checks.py`. No metre-based distances or search radii are computed until the team finalizes the projected CRS.

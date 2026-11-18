# PathSnap-ML

PathSnap-ML is a machine-learning project for map-matching noisy GPS observations to a road network and reconstructing routes.

## Current status

The EDA validates reference-route arc IDs and node references, route continuity and revisits, route cycles, and GPS-observation counts across the supplied `train` and `student_test` partitions. The spatial analysis also reports nearest-reference-route distances using the shared trip-local CRS utilities. These distances are spatial-proximity proxies, not temporal GPS-to-arc matches or measured GPS noise. No operational search radius is recommended.

## Dataset setup

The original dataset is kept outside this repository. Prefer extracting the archive as:

```text
data\student_folder\train\
data\student_folder\student_test\
```

The shared loader also checks the dataset location documented by the instructor and the standard Downloads location. To use another location, set `PATHSNAP_DATA` to the dataset root (the directory containing `train\` and `student_test\`) or its parent in PowerShell:

On the current development machine, the dataset was found at `C:\Users\arunr\Downloads\Student folder\Student folder`.

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

Open `eda\arun_reference_route_analysis.ipynb` in VS Code and run all cells for route validation. From the project root, run `py -3.10 eda\arun_spatial_analysis.py` for the separate spatial-proximity analysis. It writes tables to `results\` and figures to `eda\figures\`; it will refuse to overwrite existing outputs unless `--overwrite` is specified.

The route-check helpers in `eda\route_checks.py` reuse the shared loader in `src\data_loader.py`. Per-trip projection is provided by `src\geospatial.py`. The spatial-analysis method and limitations are documented in `eda\arun_spatial_analysis_report.md`; `.route` files do not provide temporal GPS-point associations, so the distance output must not be interpreted as actual GPS noise or candidate recall.

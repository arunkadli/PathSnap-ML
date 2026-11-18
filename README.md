# PathSnap-ML

PathSnap-ML is a machine-learning project for map-matching noisy GPS observations to a road network and reconstructing routes.

## Current status

The current EDA work validates reference-route arc IDs, node references, route continuity, repeated arcs, and GPS-observation counts across the supplied `train` and `student_test` partitions. GPS-to-reference-arc alignment and metre-based noise analysis are pending the team's projected CRS decision and an agreed alignment methodology.

## Dataset setup

The original dataset is kept outside this repository. The loader uses the following local path by default:

```text
C:\Users\arunr\Downloads\Student folder\Student folder
```

The root must contain `train\` and `student_test\`. To use another location, set `PATHSNAP_DATA` to that root in PowerShell:

```powershell
$env:PATHSNAP_DATA = 'C:\path\to\Student folder'
```

Only the student-provided `train` and `student_test` partitions are read by the route EDA; evaluator data is not accessed.

## Environment and EDA

From the project root, install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Open `eda\arun_reference_route_analysis.ipynb` in VS Code and run all cells. The notebook locates the workspace root from either the project root or `eda\`, and writes route tables to `results\` and figures to `eda\figures\`.

The route-check helpers are in `eda\route_checks.py`. No metre-based distances or search radii are computed until the team finalizes the projected CRS.

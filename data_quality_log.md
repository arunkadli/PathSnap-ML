# Data Quality Log

Partition used for all findings: train (80 trips) and student_test (10 trips).
Source: eda/architha_track_eda_stdlib.py -> results/track_*.csv

## Findings
| # | Issue | Evidence | Decision |
|---|-------|----------|----------|
| 1 | Missing / duplicate / backward timestamps | nan=0, dup=0, non-monotonic trips=0 | No cleaning needed |
| 2 | Sampling not always 1 Hz | train 93.8% and student_test 98.2% of steps are exactly 1 s | Keep all points; add time_gap_s feature |
| 3 | Pauses (gap > 5 s, moved < 50 m) | 801 events | Keep points; add `after_pause` flag; HMM may stay on same segment |
| 4 | Gaps with movement (gap > 5 s, moved >= 50 m) | 459 events | Keep points; add `gap_s` and `jump_m` features; relax HMM transition distance |
| 5 | Huge time jumps (max 153,064 s, trip 90) | timestamp jumps, little movement | Cap time-gap feature at 60 s |
| 6 | Implausible speeds (max 648 km/h) | 57 trips flagged | Keep (this is GPS noise to be corrected); add implied_speed feature |
| 7 | GPS tracks span 23 UTM zones; 2 trips cross a zone boundary | results/harshitha_crs_summary.csv and results/harshitha_trip_inventory.csv | Select one local UTM CRS per trip from the median valid GPS position; keep track and network in the same trip CRS |
| 8 | Candidate search radius | median GPS error about 5 m; 50 m leaves about 1.3% candidate miss rate | Radius = 50 m |
| 9 | student_test differs from train | max 3,958 obs vs 19,313; fewer gaps | Report both; do not tune on student_test |

## Rules
- No GPS observation is deleted; every point gets a prediction.
- Fit any scaling on train only.
- Evaluator data is never used for any decision.

## Arun — Phase 2 route validation (verified 2026-11-18)

The notebook was executed against the local student dataset at `C:\Users\arunr\Downloads\Student folder\Student folder`; it read only `train` and `student_test` (80 and 10 trips). The original dataset was not modified.

- **Verified:** all 90 `.route` files parsed as integer arc-ID sequences. The routes contain 34,813 arc-ID occurrences; 0 IDs are outside their trip's `.arcs` table.
- **Verified:** 0 arcs reference missing node IDs. There are 34,723 consecutive route-arc pairs and 14 continuity breaks (0.0403%) across 11 trips: 13 breaks in 10 training trips and 1 break in student-test trip `00000076`. The existing network analysis found directed bridge paths of 1–41 hops for 13 breaks and no path for the break in training trip `00000086`.
- **Verified:** 322 repeated arc-ID occurrences across 10 trips. Student-test trip `00000056` has 255 repeated occurrences. No immediate consecutive arc-ID duplicates were found.
- **Verified:** 227,577 GPS observations versus 34,813 route arc occurrences overall. Training has 208,417 observations and 31,336 route arcs; student-test has 19,160 observations and 3,477 route arcs. Every trip has more GPS observations than route arcs. The median per-trip GPS-observation/route-arc ratio is 5.25 (range 1.26–56.98); this is not a point-to-arc assignment.
- **Verified comparison:** the regenerated `arun_route_validation.csv`, `arun_route_breaks.csv`, and `arun_route_repeats.csv` matched their pre-existing versions (row/column values matched, with floating-point tolerance). The prior reported route-validation counts were reproduced.
- **Still unresolved:** no source file links GPS observations to specific route arcs. The later spatial analysis computes a nearest-reference-route geometry proxy, but does not validate temporal point-to-arc assignments or true GPS error.
- **Superseded CRS status:** Harshitha's per-trip UTM utility is reused for the spatial-proximity analysis below. No temporal point-to-arc mapping, GPS-noise estimate, or operational candidate radius has been validated or recommended.
- **Limitation:** the supplied reference route is not independently verified ground truth; any future alignment or noise conclusions will inherit that uncertainty.
## Harshitha — shared loader and geospatial EDA (verified 2026-11-18)

- The shared `load_trip(trip_id, partition, data_dir=None)` loader returns named `track`, `nodes`, `arcs`, and `route` DataFrames. It honors `PATHSNAP_DATA` and accepts a dataset root or its parent directory.
- Scanned all 80 training and 10 student-test trips: 227,577 GPS observations, 0 invalid longitude/latitude coordinates, and WGS 84 track bounds of longitude -100.418538..137.237029 and latitude -34.640291..61.699590.
- GPS positions and road-network nodes occupy 23 WGS 84 UTM zones. Two training trips cross UTM zone boundaries, so the shared geospatial helper selects each trip's UTM zone from its median valid GPS position, then uses that same CRS for its track and network. Southern-hemisphere trips use the matching EPSG:327xx CRS. Projected metre coordinates are not directly comparable across trips.
- Generated `results/harshitha_trip_inventory.csv`, `results/harshitha_dataset_summary.csv`, `results/harshitha_crs_summary.csv`, coordinate/trajectory PNGs, and three Folium maps with GPS, road-network, and reference-route layers plus a legend and layer control under `eda/figures/`.

## Arun — spatial proximity and route-topology analysis (verified 2026-11-18)

This section records a new analysis using only `train` and `student_test`. The earlier track-EDA findings remain; route findings are consolidated under Phase 2 to avoid duplicate entries. The run used `eda/arun_spatial_analysis.py` and the shared loader and projection helpers copied unchanged from Harshitha's `origin/eda-harshitha` branch. The analysis did not access evaluator data or modify the original dataset.

- **Verified:** all 90 trip route counts and continuity-break rows regenerated by the script match the existing `arun_route_validation.csv` and `arun_route_breaks.csv`. The existing validation records 34,813 route arc occurrences, 0 invalid route IDs, 0 invalid node references, and 14 continuity breaks (13 training, 1 student-test).
- **Verified:** 322 repeated arc occurrences occur in 10 trips; the new revisit rows reproduce the existing repeated-arc counts and gap summaries. Repeated arcs are not counted multiple times in the route-geometry union.
- **Verified:** 1,383 node-return cycle candidates were found within continuous route blocks. Every emitted cycle was checked against its source route arcs for closure and internal continuity. A repeated node is a topological return, not proof of an intentional semantic loop.
- **CRS reused and checked:** the existing `utm_crs_for_trip` selects a trip-local WGS 84 / UTM CRS from median valid GPS coordinates; `project_coordinates` uses EPSG:4326 and `always_xy=True`. The 90 trips use 25 distinct EPSG codes representing 23 numeric UTM zone numbers across both hemispheres, with metre axes. Maximum sample inverse-transform error was `3.54e-9 m`. Trips `00000011` and `00000097` cross UTM zone boundaries.
- **Spatial proxy only:** each of 227,577 GPS observations was measured to the nearest unique straight endpoint-to-endpoint segment among that trip's `.route` arcs. This is not a temporal GPS-to-arc alignment, GPS error, or true candidate miss rate. No missing geometry was invented across route breaks.
- **Partition summaries (metres):** train mean 21.43, median 5.61, standard deviation 46.25, P75 10.79, P90 74.73, P95 140.27, P99 203.67, max 569.66; student_test mean 6.55, median 5.16, standard deviation 6.18, P75 8.94, P90 13.37, P95 16.78, P99 27.96, max 105.91.
- **Coverage only:** training-percentile-derived radii of 6, 11, 75, 141, and 204 m yielded training route-geometry coverage of 52.65%, 75.60%, 90.01%, 95.10%, and 99.01%, respectively. Student-test coverage was 56.28%, 83.46%, 99.92%, 100%, and 100%. The existing 50 m log value yields 89.17% train and 99.86% student-test coverage under this proxy. These figures are not candidate recall/miss rates, and no search radius is recommended.
- **Notable tail:** the highest trip-level P95 proxies are trips `00000002` (242.36 m), `00000054` (210.05 m), and `00000033` (142.06 m). Trip `00000054` has no route continuity break but a 138.59 m median proxy; the available files do not establish the cause. IQR outliers are statistical flags only, not invalid GPS observations.
- **Unresolved:** `.route` has no timestamps or GPS-point indices; route segments have endpoint coordinates but no road centerline shapes; reference routes are not independently verified ground truth. Therefore the distance distribution must not be interpreted as measured GPS noise, and the older `50 m / 1.3% miss` entry is not confirmed or refuted by this differently defined proxy.

Outputs are documented in [eda/arun_spatial_analysis_report.md](eda/arun_spatial_analysis_report.md); route validation baseline files were not overwritten.

# Arun Spatial Proximity and Route Topology Analysis

Analysis run: 2026-11-18

## Scope and reproducibility

The analysis ran against the local dataset at `C:\Users\arunr\Downloads\Student folder\Student folder`, reading only the `train` and `student_test` partitions (80 and 10 trips). The original dataset and existing Arun route-validation outputs were not modified. The new script also confirmed that its per-trip route arc counts and continuity-break rows match the existing route-validation baselines.

Run from the repository root with Python 3.10.11:

```powershell
py -3.10 eda\arun_spatial_analysis.py
```

The script refuses to replace its outputs by default. To regenerate these specific outputs after reviewing them:

```powershell
py -3.10 eda\arun_spatial_analysis.py --overwrite
```

It imports the shared `load_trip`, `utm_crs_for_trip`, and `project_coordinates` helpers. The latter selects a trip-local WGS 84 / UTM CRS using the median valid GPS location and transforms longitude/latitude with `always_xy=True`. This is not a single global CRS: 25 EPSG codes are used across the 90 trips. All selected CRSs have metre axes. An inverse-transform check on sample track points from each trip had a maximum geodesic round-trip error of `3.54e-9 m`; projected track coordinate ranges were easting `241,883–822,267 m` and northing `33,786–7,100,494 m`. Trips `00000011` and `00000097` cross UTM zone boundaries while retaining their median-selected trip CRS.

## Route topology findings

- The 90 routes contain 34,813 arc-ID occurrences. All route IDs resolve in their trip's arc table, and the prior validation reports no missing node references.
- There are 14 continuity breaks among 34,723 consecutive route-arc pairs (`0.0403%`), affecting 11 trips: 10 training trips and student-test trip `00000076`. The existing break analysis found a directed bridge path for 13 breaks and none for the break in training trip `00000086`.
- There are 322 repeated arc occurrences in 10 trips; student-test trip `00000056` has 255. No adjacent repeated arc IDs occur.
- The new analysis detects 1,383 node-return cycle candidates. A candidate is a route walk that returns to a previously visited node within a continuous block. All reported cycles were checked against the source route arcs for closure and within-cycle continuity. These are topological returns, not proof of an intended semantic loop.
- Route walks are split at continuity breaks for cycle detection; no missing bridge arcs or geometries are fabricated.

The dataset has 227,577 GPS observations and 34,813 route arc occurrences: 208,417 observations / 31,336 arcs in `train`, and 19,160 / 3,477 in `student_test`. Every trip has more observations than route arcs; the median per-trip count ratio is 5.25. This ratio is not an observation-to-arc assignment.

## Distance methodology and interpretation

For each GPS observation, the script computes its minimum planar distance to the union of the unique arc segments listed in that trip's reference route, using the trip-local projected CRS. Arc geometry is represented by a straight segment between the `.nodes` coordinates referenced by the `.arcs` endpoints. Repeated route arcs do not get extra weight in the geometric union. Disconnected route portions remain separate; no geometry is inserted across a break.

The `.route` files provide ordered arc IDs but no GPS timestamps or point indices. Therefore, the analysis does **not** make a temporal GPS-to-arc association. The saved per-observation distance is a nearest-reference-route **spatial proximity proxy**, not actual GPS measurement error, confirmed ground truth, or candidate recall. The supplied reference routes are not independently validated, and the files do not provide curved road shapes between endpoint nodes.

## Nearest-reference-route distance summary

All distances are in metres. Percentiles use all valid observations within each partition. IQR outliers use partition-specific Tukey fences and are not classified as invalid GPS points.

| Partition | GPS points | Mean | Median | Std. dev. | P75 | P90 | P95 | P99 | Maximum | IQR outliers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 208,417 | 21.43 | 5.61 | 46.25 | 10.79 | 74.73 | 140.27 | 203.67 | 569.66 | 27,656 (13.27%) |
| student_test | 19,160 | 6.55 | 5.16 | 6.18 | 8.94 | 13.37 | 16.78 | 27.96 | 105.91 | 679 (3.54%) |

The training distribution has a long tail and is strongly affected by a few trips. The largest trip-level P95 values are trip `00000002` (242.36 m), `00000054` (210.05 m), and `00000033` (142.06 m). Trip `00000054` has no route continuity break but a 138.59 m median proxy distance; trip `00000033` has one break and a 122.41 m median. This demonstrates that continuity alone does not explain the large distances. The analysis does not infer a cause or label these values as GPS error.

## Exploratory coverage radii

The following integer-metre options are ceilings of training-distance percentiles. Coverage is the share of observations whose nearest supplied reference-route segment lies within the radius; the student-test partition is reported separately and was not used to derive these options.

| Basis | Radius | Train coverage | Student-test coverage |
|---|---:|---:|---:|
| Training P50 | 6 m | 52.65% | 56.28% |
| Training P75 | 11 m | 75.60% | 83.46% |
| Training P90 | 75 m | 90.01% | 99.92% |
| Training P95 | 141 m | 95.10% | 100.00% |
| Training P99 | 204 m | 99.01% | 100.00% |
| Existing log's 50 m value (comparison only) | 50 m | 89.17% | 99.86% |

These are geometry-proximity coverage figures, **not** true candidate recall or miss rates. The existing data-quality log's `50 m / 1.3% miss` claim is not reproduced by this proxy metric: at 50 m, 10.83% of training and 0.14% of student-test observations lie outside the supplied route geometry. The measures have different definitions, so this comparison neither validates nor disproves the prior claim. No operational search radius is recommended: a point's correct road is unknown, the route is not independent ground truth, and the train/test distance distributions differ substantially.

## Outputs

- Reproducible analysis: [arun_spatial_analysis.py](./arun_spatial_analysis.py)
- Per-observation proxy distances: [arun_spatial_analysis.csv](../results/arun_spatial_analysis.csv)
- Partition and trip statistics: [arun_distance_summary.csv](../results/arun_distance_summary.csv)
- Train-derived radius coverage: [arun_candidate_coverage.csv](../results/arun_candidate_coverage.csv)
- Per-trip topology counts: [arun_route_cycle_summary.csv](../results/arun_route_cycle_summary.csv)
- Node-return cycle details: [arun_route_node_cycles.csv](../results/arun_route_node_cycles.csv)
- Arc revisit details: [arun_route_arc_revisits.csv](../results/arun_route_arc_revisits.csv)
- Projection validation: [arun_projection_validation.csv](../results/arun_projection_validation.csv)
- Figures: [distance distribution](./figures/arun_route_distance_distribution.png), [distance by trip](./figures/arun_route_distance_by_trip.png)

## Limitations and remaining work

1. A defensible point-to-reference-arc assignment needs a temporal/sequential alignment method and validation; this analysis deliberately does not supply one.
2. The distance proxy may reflect route-label errors, route gaps, road-shape simplification, parallel roads, or GPS displacement; these cannot be distinguished from the available files alone.
3. The per-trip UTM choice is reused from Harshitha's helper. Boundary-crossing trips should be reviewed if high-precision cross-zone distances are needed.
4. Do not treat the reported proximity distribution or IQR outliers as true GPS noise or use its coverage numbers as model candidate recall without independent route/position validation.

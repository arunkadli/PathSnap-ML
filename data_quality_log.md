HEAD
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
| 7 | Trips span 25 UTM zones | results/track_inventory.csv `utm_epsg` | Project each trip with its own UTM zone |
| 8 | Candidate search radius | median GPS error about 5 m; 50 m leaves about 1.3% candidate miss rate | Radius = 50 m |
| 9 | student_test differs from train | max 3,958 obs vs 19,313; fewer gaps | Report both; do not tune on student_test |

## Rules
- No GPS observation is deleted; every point gets a prediction.
- Fit any scaling on train only.
- Evaluator data is never used for any decision.



- [Arun | route] `.route` is a plain integer sequence of arc_ids in all 90 trips; 0 ids fall outside the trip's `.arcs`; 0 arcs reference a missing node id.
- [Arun | route] Continuity: 14 breaks in 34,723 consecutive arc pairs (0.04%), in 11 of 90 trips (13 train, 1 student_test: trip 00000076). Each break needs 1-41 missing arcs to bridge (trip 00000086: no directed path). Reference routes are therefore not guaranteed fully continuous; see results/arun_route_breaks.csv.
- [Arun | route] 10 trips revisit arc ids (322 repeats total; student_test trip 00000056 has 255). Revisits are separated by long gaps in the route (loops/returns), not back-and-forth, except trip 00000086 (gap 3). Alignment must not assume each arc appears once.
- [Arun | route] No file links GPS points to route arcs: `.route` has no timestamps or point indices. Points-per-arc and point-to-true-arc distance must come from our own alignment (not yet implemented; blocked on CRS and method decision).
- [Arun | route] Every route has fewer arcs than its trip has GPS points; trip-level average is a median of 5.25 points per arc (range 1.26-56.98). This is an average only, not a per-arc count.
- [Arun | limitation] Reference route is supplied reference information, not independently verified ground truth (Dataset PDF s.6); noise/miss-rate figures derived from it inherit that uncertainty.


- [Arun | route] `.route` is a plain integer sequence of arc_ids in all 90 trips; 0 ids fall outside the trip's `.arcs`; 0 arcs reference a missing node id.
- [Arun | route] Continuity: 14 breaks in 34,723 consecutive arc pairs (0.04%), in 11 of 90 trips (13 train, 1 student_test: trip 00000076). Each break needs 1-41 missing arcs to bridge (trip 00000086: no directed path). Reference routes are therefore not guaranteed fully continuous; see results/arun_route_breaks.csv.
- [Arun | route] 10 trips revisit arc ids (322 repeats total; student_test trip 00000056 has 255). Revisits are separated by long gaps in the route (loops/returns), not back-and-forth, except trip 00000086 (gap 3). Alignment must not assume each arc appears once.
- [Arun | route] No file links GPS points to route arcs: `.route` has no timestamps or point indices. Points-per-arc and point-to-true-arc distance must come from our own alignment (not yet implemented; blocked on CRS and method decision).
- [Arun | route] Every route has fewer arcs than its trip has GPS points; trip-level average is a median of 5.25 points per arc (range 1.26-56.98). This is an average only, not a per-arc count.
- [Arun | limitation] Reference route is supplied reference information, not independently verified ground truth (Dataset PDF s.6); noise/miss-rate figures derived from it inherit that uncertainty.

## Arun — Phase 2 route validation (verified 2026-11-18)

The notebook was executed against the local student dataset at `C:\Users\arunr\Downloads\Student folder\Student folder`; it read only `train` and `student_test` (80 and 10 trips). The original dataset was not modified.

- **Verified:** all 90 `.route` files parsed as integer arc-ID sequences. The routes contain 34,813 arc-ID occurrences; 0 IDs are outside their trip's `.arcs` table.
- **Verified:** 0 arcs reference missing node IDs. There are 34,723 consecutive route-arc pairs and 14 continuity breaks (0.0403%) across 11 trips: 13 breaks in 10 training trips and 1 break in student-test trip `00000076`. The directed network has no path for the break in training trip `00000086`.
- **Verified:** 322 repeated arc-ID occurrences across 10 trips. Student-test trip `00000056` has 255 repeated occurrences. No immediate consecutive arc-ID duplicates were found.
- **Verified:** 227,577 GPS observations versus 34,813 route arc occurrences overall. Training has 208,417 observations and 31,336 route arcs; student-test has 19,160 observations and 3,477 route arcs. Every trip has more GPS observations than route arcs. The median of per-trip GPS-observation/route-arc ratios is 5.25; this is not a point-to-arc assignment.
- **Verified comparison:** the regenerated `arun_route_validation.csv`, `arun_route_breaks.csv`, and `arun_route_repeats.csv` matched their pre-existing versions (row/column values matched, with floating-point tolerance). The prior reported route-validation counts were reproduced.
- **Pending:** no source file links GPS observations to specific route arcs. Alignment must account for uneven sampling, loops/repeated arcs, and route discontinuities; point-to-arc mappings and distances are not validated yet.
- **Pending CRS decision:** projected CRS selection is Harshitha's responsibility. No metre-based GPS-to-arc distances, noise distribution, or candidate search radius was calculated.
- **Limitation:** the supplied reference route is not independently verified ground truth; any future alignment or noise conclusions will inherit that uncertainty.
 c7159c3 (Complete Arun reference route validation EDA)

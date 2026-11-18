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


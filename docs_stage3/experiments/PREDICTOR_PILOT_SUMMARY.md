# CA / PPLNM paired pilot (seed 7)

These are eight actually executed REMUS 6DOF episodes. Each configuration has one episode; this is a descriptive pilot, not a statistical superiority claim. Navigation arrival and finite-horizon avoidance acceptance are reported separately. All shared parameters, scene generation, original vehicle dynamics, collision geometry, 1 m safety threshold and 5 m arrival tolerance are unchanged between each pair.

CA and PPLNM use the same 3 s history sampling. PPLNM is frozen and trained for 9 s from the latest measurement; older measurement age consumes part of that range. The tail is explicitly CV after 9 s. Calibrated P25 radii are exposed, not applied. Offline obstacle RMSE by absolute forecast horizon is pending in this run CSV, represented as NaN; deterministic execution trajectory error is a different metric.

Wall-clock planning was measured while independent experiment workers shared CPU. It was not injected as physical command latency. P95/deadline observations therefore cannot be interpreted as isolated runtime benchmarks.

| Scenario | Predictor | Arrival | Collision | Actual margin violation | Min actual net clearance m | Emergency / unsafe forecast | Acceptance | P95 ms | Deadline misses |
|---|---|---|---|---|---:|---:|---|---:|---:|
| static3d | ca | True | False | False | 2.342522 | 1 / 1 | False | 2903.84 | 18 |
| static3d | pplnm | True | False | False | 3.342797 | 0 / 0 | True | 2065.38 | 9 |
| crossing3d | ca | True | False | False | 13.532717 | 0 / 0 | True | 1968.30 | 1 |
| crossing3d | pplnm | True | False | False | 11.325517 | 0 / 0 | True | 1943.70 | 0 |
| multi_crossing3d | ca | True | False | False | 1.910302 | 9 / 0 | True | 1976.17 | 2 |
| multi_crossing3d | pplnm | True | False | False | 1.927652 | 15 / 5 | False | 2162.19 | 12 |
| maneuvering3d | ca | True | False | False | 9.374990 | 0 / 0 | True | 1935.05 | 0 |
| maneuvering3d | pplnm | True | False | False | 4.681684 | 3 / 0 | True | 1945.73 | 0 |

Static CA arrived with one unsafe emergency forecast, whereas paired PPLNM had none. Multi-crossing PPLNM had five unsafe emergency forecasts despite arriving with actual net clearance above 1 m; paired CA had none. PPLNM reduced the observed minimum clearance in crossing and maneuvering cases. These mixed outcomes do not establish a general benefit or harm; prediction accuracy, especially trained-range versus extension error, must be assessed independently at common causal anchors.

All raw states, controls, observations, selected plans, planning diagnostics, obstacle trajectories, figures and replays remain under cases/. No failure evidence was removed.

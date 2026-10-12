# Candidate and emergency pilot (seed 7)

Sixteen preregistered, actually executed REMUS 6DOF episodes; zero program errors. Each configuration has one episode. Static speed scale is 1; multi-crossing speed scale is 1.5. The original vehicle physics, scene geometry, 1 m safety requirement and 5 m arrival tolerance are unchanged. Results are descriptive; there is no mixed overall success rate across configurations.

Timing was measured with two experiment workers plus other independent experiments sharing CPU. Planning time was not injected as physical execution delay. Offline full-grid checks run after navigation and are excluded from planning P95/deadline counts.

| Scenario | Grid / policy / budget / emergency | Arrival | Collision | Timeout | Min actual net gap m | Emergency / unsafe | Acceptance | P95 ms |
|---|---|---|---|---|---:|---:|---|---:|
| static3d | baseline45 / diverse / 9 / coast | True | False | False | 2.342522 | 1 / 1 | False | 2076.34 |
| static3d | baseline45 / diverse / 18 / coast | True | False | False | 2.342522 | 1 / 1 | False | 4085.75 |
| static3d | baseline45 / diverse / 45 / coast | True | False | False | 1.835173 | 2 / 2 | False | 9643.85 |
| static3d | baseline45 / ranked / 9 / coast | True | False | False | 2.487398 | 0 / 0 | True | 1965.21 |
| static3d | baseline45 / coverage / 9 / coast | False | False | True | 2.722619 | 0 / 0 | False | 1862.68 |
| static3d | expanded / coverage / 9 / coast | False | False | True | 2.755937 | 0 / 0 | False | 1997.02 |
| static3d | expanded / coverage / 45 / coast | True | False | False | 1.835173 | 2 / 2 | False | 9113.44 |
| static3d | baseline45 / diverse / 9 / verified_backup | True | False | False | 2.342522 | 1 / 1 | False | 1888.75 |
| multi_crossing3d | baseline45 / diverse / 9 / coast | True | False | False | 0.508115 | 3 / 3 | False | 1921.97 |
| multi_crossing3d | baseline45 / diverse / 18 / coast | True | False | False | 0.459421 | 3 / 3 | False | 3685.76 |
| multi_crossing3d | baseline45 / diverse / 45 / coast | True | False | False | 0.459344 | 3 / 3 | False | 8812.82 |
| multi_crossing3d | baseline45 / ranked / 9 / coast | True | False | False | 0.472502 | 3 / 3 | False | 1831.56 |
| multi_crossing3d | baseline45 / coverage / 9 / coast | True | False | False | 2.050077 | 0 / 0 | True | 1634.74 |
| multi_crossing3d | expanded / coverage / 9 / coast | False | False | True | 2.023764 | 0 / 0 | False | 1832.34 |
| multi_crossing3d | expanded / coverage / 45 / coast | True | False | False | 0.459344 | 3 / 3 | False | 8925.25 |
| multi_crossing3d | baseline45 / diverse / 9 / verified_backup | True | False | False | 0.508115 | 3 / 3 | False | 2311.36 |

## Same-state first-failure diagnostic

Fixed first-failure frames only, at most one per episode. Every row separately compares all policies and budgets at its own fully captured actual REMUS/controller/reference/actuator state and causal observation histories. Different episode rows are not identical vehicle states. No captured frame does not mean no omissions at other times. Initial obstacle clearance excludes terrain and later motion. An initial predicted gap below 1 m is sufficient to reject every candidate, not proof of physically unavoidable collision.

The capture rule was fixed before navigation: the first planning call with feasible=False. All original-grid or expanded-grid references were then integrated by the original REMUS 6DOF plant, controller, reference model and actuators from that isolated snapshot. The previous selected action, planning subgoal, causal tracks and applied memory uncertainty are retained. Each compact CSV row includes safe counts for diverse, ranked and coverage at budgets 9/18/45. Empty diagnostic cells mean no failed planning call was captured, not a zero safe count.

| Case | First failure s | Full grid size | Full grid safe | Configured shortlist missed safe | Initial predicted obstacle gap m |
|---|---:|---:|---:|---|---:|
| candidate__static3d__diverse9__seed7 | 38.0 | 45 | 0 | False | 0.829306 |
| candidate__static3d__diverse18__seed7 | 38.0 | 45 | 0 | False | 0.829306 |
| candidate__static3d__diverse45__seed7 | 38.0 | 45 | 0 | False | 0.220100 |
| candidate__static3d__expanded_coverage45__seed7 | 38.0 | 68 | 0 | False | 0.220100 |
| candidate__static3d__verified_backup9__seed7 | 38.0 | 45 | 0 | False | 0.829306 |
| candidate__multi_crossing3d__diverse9__seed7 | 30.0 | 45 | 0 | False | 0.312337 |
| candidate__multi_crossing3d__diverse18__seed7 | 30.0 | 45 | 0 | False | 0.307883 |
| candidate__multi_crossing3d__diverse45__seed7 | 30.0 | 45 | 0 | False | 0.307883 |
| candidate__multi_crossing3d__ranked9__seed7 | 30.0 | 45 | 0 | False | 0.317002 |
| candidate__multi_crossing3d__expanded_coverage45__seed7 | 30.0 | 68 | 0 | False | 0.307883 |
| candidate__multi_crossing3d__verified_backup9__seed7 | 30.0 | 45 | 0 | False | 0.312337 |

## Interpretation and limits

Increasing the full-rollout budget did not monotonically improve the observed result. Ranked9 reached the static target without emergency, while coverage9 timed out in the static scene despite continuously finding feasible local trajectories. Coverage9 reached the multi-crossing target with the actual gap above 1 m and no emergency, while ranked9 violated the actual margin there. This is evidence of scene-dependent selection/navigation behavior, not universal superiority of a shortlist policy.

A full-grid zero at a captured frame rules out a shortlist omission only for that specific state, causal forecast and finite reference grid. It does not establish that no physically feasible action exists. A shared initial inflated obstacle constraint already below 1 m cannot be repaired by selecting a different reference at that same planning instant. Earlier navigation decisions and prediction/memory margins remain separate causal mechanisms to investigate.

All original observations, truth snapshots, states, actuator/control logs, selected plans, failed planning calls, figures and replays remain under cases/. Emergency RPM0 is a physical coasting action, not instantaneous stopping. Every tested backup uses the same full 6DOF validation and safety criterion. No unsafe forecast or navigation failure was removed. Offline CA/PPLNM common-anchor prediction errors are supplied separately without rerunning navigation or rewriting old metadata. The paired offline variants use common 3 s history sampling; original candidate navigation used the retained CA default of 1 s history sampling. Therefore ca_dt3 offline RMSE is a separately labelled common-history comparison, not the measured prediction error of the actual ca_dt1 navigation predictor.

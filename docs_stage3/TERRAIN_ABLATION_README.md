# Independent guidance-progress bugfix ablation

The original negative runs remain unchanged. No scene, seed, 6 m waypoint tolerance, 25 m lookahead, 5 m final-goal criterion, collision check, 1 m margin, RPM grid or physical parameter was changed. This ablation changes only local progression from a missed corner to its immediately adjacent route segment after an actual local pass/association and an original swept terrain check.

Old direct results are reused explicitly; they were not rerun for this ablation. All root-scene configs normalize to identical actual parameter dictionaries within each A* before/after comparison. Wall-clock latency is descriptive under concurrent jobs, not an isolated benchmark or injected execution delay.

| Scene | Version | Arrival | Collision | Timeout | Minimum actual clearance / m | Emergencies / unsafe | Final goal distance / m |
|---|---|---|---|---|---:|---:|---:|
| perlin_cruise_constrained3d | direct, original result reused | False | False | True | 1.001706 | 37 / 0 | 9.481118 |
| perlin_cruise_constrained3d | A*, before progress fix | False | False | True | 17.098811 | 0 / 0 | 49.876432 |
| perlin_cruise_constrained3d | A*, after isolated progress fix | True | False | False | 17.098811 | 0 / 0 | 4.983715 |
| terrain3d | direct, original result reused | False | False | True | 20.897314 | 85 / 0 | 108.204463 |
| terrain3d | A*, before progress fix | True | False | False | 6.383011 | 0 / 0 | 4.993618 |
| terrain3d | A*, after isolated progress fix | True | False | False | 6.383011 | 0 / 0 | 4.993618 |

The progress mechanism was reproduced by a new recorded-state unit test before the fix: 28 pass / 1 fail. After the fix, 29 terrain plus 2 episode-serialization tests passed. The fix checks only the immediately adjacent route segment within the unchanged lookahead neighborhood and requires its original swept terrain connection. A self-near distant route leg cannot skip unvisited remote corners; unsafe seabed states cannot certify the transition.

This is a program correction, not retuning of scene difficulty or a new risk algorithm. Remaining arrival or safety failures are retained and must be analyzed separately. Neutral-cruise initialization admission remains finite-horizon evidence, not a proof that every tested navigation method can complete the task.

Source CSV contains relative raw-result paths and an explicit physical_run_reused_for_this_ablation flag. Only the two after-fix A* rows are new physical reruns.

The four distinct latest direct/A* cases contain no observed obstacle objects. Offline paired CA/PPLNM analysis produced zero object anchors and zero RMSE samples: obstacle prediction RMSE is undefined, not zero. The vehicle prediction/execution RMSE reported in original summaries is a different deterministic-plant metric.

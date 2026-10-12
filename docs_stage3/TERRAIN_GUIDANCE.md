# Known-seabed guidance baseline

`auv3d/stage3_terrain.py` implements ordinary coarse-grid three-dimensional A*.
It is a shared navigation aid, not a new research contribution and not a
six-DOF safety certificate. It has no obstacle schedule or future-position
input. The direct final-goal baseline is preserved as `mode="direct"`.

The alternative uses a 15 m horizontal grid, 3 m depth grid, Euclidean edge
cost and admissible Euclidean A* heuristic. Its fixed 60,000 expanded-node
budget, deterministic neighbor order and tie ordering make a repeated scene
reproducible. Pure vertical edges are omitted; every accepted edge must have
absolute depth change / horizontal distance at most 0.25. That mild geometric
grade is an aid to navigation; controller lag and actuator limits are still
validated by the original REMUS forward rollout.

Both grid edges and every line-of-sight shortcut call the existing
`HeightField.segment_clearance` using the original enclosing sphere and
1 m safety criterion. This retains the original global slope bound,
between-sample conservative correction, NED surface and map boundaries.
The original maximum-depth constraint is checked separately. Failure to find
a route or exhausting the node budget raises `TerrainRouteError`; a failed
search never silently returns an unchecked straight route.
An independent two-million-grid-node allocation guard rejects accidentally
tiny grid spacings before memory allocation; it does not certify a failed route.

`TerrainGuidance.current_target(actual_position)` advances along the route
using actual REMUS position and a 25 m lookahead. Intermediate waypoint
tolerance is 6 m. It rechecks the segment from the actual deviated position
to a proposed lookahead target and exposes `unreachable_target_count` when
no such shortcut is certified. The local planner must still independently
reject unsafe full REMUS trajectories. Reaching a subgoal does not mean the
task is complete: only distance to the unchanged final goal and the original
arrival criterion can determine navigation success.

`build_terrain_guidance(terrain, start, goal, body_radius_m=...,
safety_margin_m=..., depth_limit_m=..., config=TerrainGuideConfig(...))`
returns an object with `path`, `final_goal`, `metrics`, `progress_m` and
`current_target(actual_position)`. The route and final goal are read-only
arrays; `current_target` returns an independent point. The execution loop may
save `path` and `metrics` alongside actual trajectories and control logs.
Route metrics explicitly describe their validation scope as terrain geometry,
not dynamic-obstacle safety or REMUS execution feasibility.

# Fixed terrain-isolation scene

The existing `terrain3d` ridge and `paper_random3d` generator are unchanged.
The separate `make_constrained_perlin_scene(cfg)` returns a
`perlin_constrained3d` scene on the complete original 1000 × 1000 m Perlin map
(`TerrainConfig()` and the requested seed). No cropping, flattening, amplitude
change or seed retry occurs. Dynamic obstacles are omitted in this new scene
to isolate terrain guidance; dynamic and terrain performance remain separate
experiments.

Its start and goal are at a predeclared depth of 70 m and separated by 180 m
along either map axis. Centers lie on a fixed 25 m grid between 100 and 900 m.
Pairs are ordered by center distance to the map center, then north, east and
axis. The first pair with safe endpoints and an original swept straight-line
clearance below 1 m is selected. Selection depends only on seabed geometry,
never on the outcome of any tested controller or predictor. If that rule
produces no admissible pair, the scene fails explicitly without changing its
seed. The selection rule, examined-pair count, seeded terrain configuration
and straight-line clearance are recorded in scene metadata.

For seed 7, the fixed rule gives start `[550, 360, 70]`, goal
`[550, 540, 70]`. The direct line has conservative net clearance
approximately -21.65 m; this is an actual terrain constraint rather than a
merely visible distant seabed. A geometric A* route exists around the shallow
region. Whether the original six-DOF closed loop can execute that route is a
separate measured result; the unit tests do not claim task success.

# Actual six-DOF paired validation (seed 7)

The four original-model closed-loop runs below used identical settings within
each scene: 200 s limit, H=24 s, physical step 0.02 s, planning period 2 s,
original 45-reference grid with budget 9, CA, original coast emergency and
the original 1 m safety criterion. Only direct versus A* guidance changed.

| Scene | Guidance | Actual result | Minimum actual net clearance / m | Emergencies / unsafe forecasts |
|---|---|---|---:|---:|
| terrain3d | direct | Timeout at 200 s | 20.897314 | 85 / 0 |
| terrain3d | coarse3d_astar | Arrival at 133.32 s; finite-horizon acceptance passed | 6.383011 | 0 / 0 |
| perlin_constrained3d | direct | Terrain collision at 11.42 s | -0.003941 | 6 / 6 |
| perlin_constrained3d | coarse3d_astar | Same terrain collision at 11.42 s | -0.003941 | 6 / 6 |

The ridge result is a measured single-seed improvement in task completion,
not a statistical success-rate claim. Both Perlin runs are retained as
**initialization-limited diagnostics**, not evidence from a validated
dynamically reachable common task. Their geometric start clearance is
13.593868 m, but only about 8 m of straight motion intersects the original
conservative terrain boundary. Initial physical speed after the unchanged
warmup is about 1.84171 m/s. RPM-zero coasting cannot stop before this region.

Independent initial-state forward checks evaluated all 45 original references
at each corresponding navigation target. Both full grids had zero feasible
references (best predicted clearance -15.593049 m for direct and -13.278171 m
for A*). All eight configured backup references also violated 1 m. These
finite-grid findings establish that the original budget-9 shortlist was not
the initial failure mechanism; they do not prove physical impossibility under
arbitrary time-varying control. The geometric A* path alone does not validate
an initialization against vehicle/controller lag or stopping distance.

A future ordinary Perlin task protocol must predeclare and independently
validate common dynamically executable initialization conditions using the
unchanged plant. The current points, seed, terrain and failed trajectories were
preserved; no positions or thresholds were adjusted after seeing these results.

Raw configs, actual states, controls, predictions, swept clearance, original
plots, interactive replays, full-grid/backup diagnostics and paired plots are
saved under `results_stage3/terrain_closed_loop_20261011/`. Planning latency
was measured under concurrent CPU jobs, and physics paused during planning.
It must not be presented as isolated real-time performance or a physical
execution delay. These terrain-isolation scenes contain no obstacle targets;
obstacle forecast RMSE is undefined, not zero.

# Independent neutral-cruise-admitted Perlin protocol

`make_cruise_constrained_perlin_scene(cfg)` defines a separate
`perlin_cruise_constrained3d` scenario. It retains the old failed scene and
uses the same unchanged seeded Perlin map, fixed 70 m depth, 180 m axis chord,
25 m center grid and fixed pair ordering. Its protocol was saved in
`results_stage3/perlin_cruise_protocol_20261011/PROTOCOL.json` before selection
or running either guidance method.

After safe-endpoint and blocked-straight-line geometric admission, it calls
the original `initialize_plant` and predicts holding initial depth, initial
yaw and nominal RPM with the unchanged REMUS controller/actuators over a
**fixed 24 s initialization horizon**. Every physical interval uses the
original `HeightField.segment_clearance`, original enclosing sphere, 1 m
criterion and maximum-depth constraint. The fixed initialization horizon is
independent of the local planner horizon. The first pair to pass is selected;
all preceding physical-admission failures are retained in scene metadata.
No DWA, CA/PPLNM forecast, method-dependent score, actual navigation outcome,
seed retry or empirical stopping-distance constant is consulted.

For seed 7 and default common physical settings, the selected 93rd pair is
`[410, 400, 70]` to `[590, 400, 70]`. Its direct chord remains terrain-blocked
(net clearance -21.831367 m), while neutral cruise has at least 29.194379 m
clearance over 1,200 original-model physical intervals. The preceding three
geometrically admissible pairs failed physical admission. An ordinary A*
terrain route also exists (188.530504 m geometric length, 2.725951 m minimum
swept terrain clearance).

This protocol verifies that an original-model neutral initialization can
execute a complete 24 s horizon. It is **not proof of entire-task dynamic
reachability** or safety around future dynamic obstacles. These claims are
explicitly separated in scene metadata. The new scenario's later direct/A*
closed-loop results must be reported separately from the old initialization
failure; they must not be combined into a single unexplained success rate.

The initially measured new-task direct run timed out at 200 s with final
distance 9.481118 m, actual minimum clearance 1.001706 m and 37 forecast-safe
coast events. The initial A* run also timed out, with final distance 49.876432 m,
actual minimum clearance 17.098811 m and no emergencies. These negative results
remain in `results_stage3/perlin_cruise_closed_loop_20261011/`.

# Corrected local segment progression

The recorded initial A* run exposed a definite navigation-logic error. The
actual vehicle passed within 7.680021 m of the first corner and within
5.387188 m of the second corner, but the old rule required entry into the first
corner's 6 m ball forever. Its lookahead already pointed beyond that corner,
while route progress became locked at the first segment's 133.884792 m length.
The target stayed at `[565.365316, 416.492442, 70]` through 198 s; the vehicle
subsequently turned back toward that stale target. Prediction versus physical
execution RMSE was zero, so this failure is not evidence that the controller
failed to execute its verified prediction.

The correction retains the original 6 m waypoint tolerance, 25 m lookahead,
path, final-goal standard and safety margin. It also permits progression to
the immediately adjacent segment when the actual vehicle locally passes the
corner plane or associates more closely with that next segment. The corner
must lie within the existing lookahead neighborhood, and the segment from
actual position to its next-segment projection must pass the original swept
terrain check. It does not globally project onto later self-near route legs
or skip unvisited remote corners.

A recorded-state regression failed before the correction (28 tests passed,
1 failed). After correction, all 29 terrain tests and both serialization tests
passed. Additional regressions prevent jumping across a distant self-returning
route or certifying a transition from an unsafe seabed state. The original
negative navigation experiments are preserved, and the independent rerun is
saved separately in `results_stage3/perlin_cruise_guidance_fixed_20261011/`.

The isolated corrected Perlin run **actually reached the unchanged final-goal
ball at 142.60 s**, final distance 4.983715 m. Actual minimum swept clearance
was 17.098811 m, with zero emergencies, unsafe forecasts, margin violations or
collisions. Every planning decision had nine verified feasible candidates.
The saved config normalizes identically to the retained pre-fix A* run; the
scene, original geometric route and safety criteria are unchanged. The final
result is evidence that this one common task is executable by the original
six-DOF plant with the corrected ordinary guidance. It is not a statistical
success-rate claim, and the old initialization-limited Perlin task remains a
different explicitly failed diagnostic.

The unchanged original ridge task was also rerun after the isolated correction:
it reached at 133.32 s, final distance 4.993618 m, with minimum actual clearance
6.383011 m and zero emergencies or unsafe forecasts. These trajectory metrics
match its retained pre-fix run. Both reruns preserve the same normalized
configuration dictionaries; planning P95 wall times (1842.482 ms for Perlin,
1738.558 ms for the ridge) were measured under shared CPU load and are not
isolated real-time benchmarks or delays inserted into the physics loop.

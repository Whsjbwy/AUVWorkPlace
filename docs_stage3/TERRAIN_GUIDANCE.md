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

# Causal perception-memory interface

`auv3d.stage3_memory.make_track_memory(cfg)` creates an independent perception
memory for the REMUS experiment. It does not access a scenario, obstacle truth,
the vehicle plant, or a future plan. Integration supplies only already delivered
observations and the current actual NED vehicle position.

```python
memory = make_track_memory(cfg)
memory.update(delivered_observations, now, ego_position=actual_eta[:3])
tracks = memory.current(now)
extra_radius_by_id = {t.object_id: memory.uncertainty_for(t, now) for t in tracks}
diagnostics = memory.diagnostics(now)
```

Histories retain acquisition timestamps, contain at most the same 32 measured
points as Stage2, and are returned as isolated copies. No predicted point is
appended to a measurement history. A malformed, future, duplicate or reversed
measurement rejects the whole batch without a partial observation write.

| Configuration | Behavior |
| --- | --- |
| `memory_policy="hard_expiry"` | Original Stage2 age filtering, with `track_max_age_s` of 4, 8 or 12 seconds. Extra radius is zero. |
| `memory_policy="retain_uncertain"` | Keep lost tracks until episode end; inflate the forecast envelope without changing physical obstacle radius. |
| `memory_policy="conditional"` | Same uncertain retention, but delete a lost object if the declared-speed reachable balls cannot interact over the planning horizon plus control and command-delay buffer. |

For retention policies, extra radius is
`memory_uncertainty_growth_mps * max(0, now-last_observation_time-observation_period_s-observation_delay_s)`.
The default growth rate is 2.5 metres/second. This is an explicit ablation
parameter; it is not a calibrated probability, certified physical coverage,
or replacement for a predictor's separate calibration ball. Forecast integration
adds this radius to the common nominal forecast envelope. Original collision
geometry and the common one-metre safety criterion remain unchanged.

Conditional removal uses last observed object point `p`, actual ego position
`x`, observation age `a`, and `T=horizon_s+control_dt+command_delay_s`:

```
||p-x|| > memory_obstacle_speed_bound_mps*(a+T)
          + memory_ego_speed_bound_mps*T
          + interaction_distance_m + body_radius_m
          + observed_obstacle_radius_m + safety_margin_m
```

Both declared speed bounds default to 4 metres/second. They are assumptions to
be checked for a given experiment, not authenticated platform limits. The
conditional claim holds only when both bounds hold. The test is strict, uses no
future planned trajectory, and never removes a currently fresh target. Missing
actual ego position disables this deletion. No deletion in a crowded scene is
a valid result; the condition must not be loosened to increase apparent gains.
The rule also assumes exact measurement positions. The factory disables
conditional deletion when `sensor_noise_std_m > 0`: Gaussian position noise has
no finite deterministic bound, so a confidence multiple must not be presented
as a hard exclusion certificate. Such configurations keep uncertain tracks.

Diagnostics report active/lost/retained IDs, per-ID age and extra radius, maximum
and mean inflation, counts and immutable event records for additions, losses,
reacquisition and removal reasons. `lost_object_count` counts distinct objects
ever lost; `lost` counts transitions, so repeated loss of one object is visible.
Loss means acquisition age exceeds the configured observation period plus
delivery delay, not that a truth-based detector declared the object absent.

`hard_expiry` is numerically identical to the valid-input Stage2 memory; stronger
validation additionally rejects invalid radii, coordinates, clocks and replayed
packets. The original Stage2 memory implementation remains available for its
existing callers and tests.

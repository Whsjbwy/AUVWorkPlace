# Stage 5 paired risk experiments

Use PowerShell from the existing `AUVWorkPlace` checkout. The commands below
use the separate stage1 environment and CPU. Every output directory must be
fresh; the runner refuses to replace saved experiments.

```powershell
Set-Location 'D:\修改代码实验\AUVWorkPlace'
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
.\.venv-stage1\Scripts\python.exe run_stage5.py manifest --phase pilot
.\.venv-stage1\Scripts\python.exe -m pytest tests_stage5/test_stage5_experiments.py -q
```

`pilot` and `formal` use new, disjoint whole-scene registrations. Each has
`dev`, `tune`, `calibration` and `test` roles. Noise replicas share `scene_id`;
three noise settings do not triple the independent scene count. The default
CLI experiment selects noise 0.02 m and a single CPU worker. `--limit` and
`--case` select an explicitly reported subset; such a run is not the complete
registered study. The old stage4 independent test set is not reused.

## Calibration and the A1 fixed-margin baseline

Build an independent pool for every predictor/noise pair being compared:

```powershell
.\.venv-stage1\Scripts\python.exe run_stage5.py build-calibration --phase pilot --predictors ca cv filtered_cv --noise 0 .02 .1 --out results_stage5\pilot_calibration
.\.venv-stage1\Scripts\python.exe run_stage5.py build-calibration --phase formal --obstacle-only --predictors ca --noise .02 --out results_stage5\formal_calibration
```

The builder writes `ARTIFACT_REGISTRY.json`. Its entries explicitly identify
predictor, noise and path. Relative paths in a manually prepared registry are
resolved against that registry's directory. There is no fallback to another
predictor, noise or phase. Formal calibration always uses the four predeclared
obstacle families; `--obstacle-only` makes that scope explicit and can also be
used for pilot data. All empty terrain controls remain in registration and
navigation evaluation, with factory failures reported separately. This excludes
families from obstacle-score estimation, without retrying a bad seed or treating
empty scenes as zero-error observations. The registry records that scope.
The risk loader additionally checks actual model
parameters, query times, shapes, source role and predictor contract.

These calibration trajectories come from a fixed causal sensing carrier,
not completed REMUS missions. Empty scenes are recorded but excluded from
obstacle-score quantiles. Missing speed groups are unknown; insufficient
independent samples cause rejection, never a finite radius cap. Carrier to
closed-loop covariate shift remains a research limitation.

The preregistered A1 scan is separate from independent test execution:

```powershell
.\.venv-stage1\Scripts\python.exe run_stage5.py tune-margin --phase formal --predictors ca --noise .02 --combination add replace --margins 0 .75 3 6 --workers 1 --out results_stage5\formal_A1_tune
.\.venv-stage1\Scripts\python.exe run_stage5.py freeze-margin --root results_stage5\formal_A1_tune --out results_stage5\formal_A1_frozen.json
```

Every margin must cover all six registered tune scenes with the same paired
case set. Program errors and incomplete physical outcomes block freezing.
Selection maximizes arrival with the actual 1 m margin, then minimizes
collisions, margin violations, timeouts, summed no-feasible decisions and the
margin itself. The frozen file retains all scan points and case IDs, grouped
by predictor, noise and `add`/`replace`. Own risk-inflated emergency flags are
reported but are not the tuning objective. Test results are never read by
this command. Frozen tuning and execution phases must match; a pilot tune
record cannot substitute for the formal tune record.

## Paired execution

Methods are A0 original engineering envelope, A1 independently tuned fixed
margin, A2 time/speed-group balls, A3 whole-scene joint ball, A4 whole-scene
candidate ECP score and A5 time/speed-group candidate ECP score. A3/A4 use new
matched calibration data while retaining their stage4 score definitions.
A0 and A1 are not redundantly executed for multiple alpha levels. Pilot uses
alpha 0.2 and formal uses alpha 0.1 unless an explicit separate sensitivity
run requests another value.

```powershell
.\.venv-stage1\Scripts\python.exe run_stage5.py run-pilot --methods A0 A2 A3 A4 A5 --predictors ca --noise .02 --combination replace --artifacts results_stage5\pilot_calibration\ARTIFACT_REGISTRY.json --workers 1 --out results_stage5\pilot_CA_replace
.\.venv-stage1\Scripts\python.exe run_stage5.py run-study --phase formal --methods A0 A1 A2 A3 A4 A5 --predictors ca --noise .02 --combination replace --alpha .1 --artifacts results_stage5\formal_calibration\ARTIFACT_REGISTRY.json --tuned-margin results_stage5\formal_A1_frozen.json --workers 1 --out results_stage5\formal_CA_replace
```

The complete primary formal run is 12 independent test geometries × 6 methods
= 72 episodes. Additional predictors, noise levels, `add` comparisons and
delay settings are separately labeled factors. Do not pool their outcomes
into a single success rate. The primary `replace` setting removes only the
named fixed/age engineering prediction margins before applying the selected
risk treatment. Physical vehicle/obstacle radii, retained memory uncertainty,
the 1 m actual safety criterion and the original REMUS parameters remain
unchanged. An actual collision or unsafe coast is preserved as a failure.

To restrict an ablation, first read the manifest and pass its exact `--case`
IDs. Do not select a favorable subset after reading independent outcomes.
`--no-plot` skips figures explicitly; default execution saves the usual actual
REMUS trajectory figures and raw logs. `--animation` additionally requests
animation. All outputs are local under `results_stage5`, not source assets.

## Physical command-latency experiments

```powershell
.\.venv-stage1\Scripts\python.exe run_stage5.py run-study --phase formal --methods A0 --predictors ca --noise .02 --delay-mode fixed --delay-s .5 --workers 1 --out results_stage5\delay_fixed_halfsecond
.\.venv-stage1\Scripts\python.exe run_stage5.py run-study --phase formal --methods A0 --predictors ca --noise .02 --delay-mode measured --workers 1 --out results_stage5\delay_measured
```

`none` retains the historical synchronous execution. `fixed` advances the
physical plant, obstacle clock and sensing under the previous command until
the specified computation delay has elapsed. `measured` uses that decision's
measured planning wall time, rounded upward to a physical tick. Both add any
registered transport delay; commands are not applied early. Decisions use the
observation history delivered at their start. No safe instant stop is assumed.

Use single-worker runs for paired latency comparisons and schedule them without
other CPU-intensive experiments. The CLI limits numerical threads before
importing numerical runtimes and records those settings. Other OS load is not
controlled. Multi-worker outputs remain useful research runs but their wall
latency does not establish comparative real-time superiority.

## Offline labels and same-state decisions

```powershell
.\.venv-stage1\Scripts\python.exe run_stage5.py analyze --root results_stage5\formal_CA_replace --anchors 20 40 60 --shadow --out results_stage5\formal_CA_replace_analysis
```

Analysis restores original controller, reference and actuator state by replaying
every archived physical control interval and checking every saved state. It
uses a fresh registered scene, not the old stage4 geometry. A0 recomputation
must match its archived action, feasibility and clearance before same-state
attribution is accepted. That initial archive check is excluded from measured
method samples. Timed method order rotates by registered scene seed and planning
cycle to reduce fixed-order cache bias. Complete nested vehicle/controller state
is checked for mutation, including integrator and reference memory.
Every risk method uses the same fully verified
shortlist; predictor/noise/delay factors are held fixed. Shadow timing is a
separate offline measurement and is not added to online planning time.

Future obstacle truth is available only to this post-execution analysis. Fixed
anchor times without a recorded plan or delivered tracks are not replaced
with convenient later times. The CSV keeps its actual evaluated denominator.
False-feasible labels concern the selected reference held for its full horizon,
including true unobserved obstacles; they are not all-candidate error rates or
actual task collisions. False-reject rates remain unknown without exhaustive
counterfactual evaluation. A2/A5 coverage compares their actual per-time/bin
score quantiles; the scalar `calibration_coverage` is the intersection across
all labeled times and observed bins of that selected reference, rather than a
pointwise rate. `calibration_coverage_by_horizon` retains covered/label counts
for each query time as JSON. A3/A4 use the exact joint definitions. A0/A1 have no calibrated
score coverage. Missing target bins are never zero-error samples. These
conditional empirical diagnostics do not assert mission-wide safety.

`EXPERIMENT_SUMMARY.csv` separates physical arrival, safe arrival (arrival with
no actual collision or 1 m violation), collision, 1 m violations,
timeouts, own unsafe-emergency forecasts, fully verified feasible fraction,
no-feasible decision fraction, planning P95/deadlines, actual delay and offline labels. `EXPERIMENT_CONFIG.json`,
`EXPERIMENT_RECORDS.json`, `GROUPED_SUMMARY.json`, per-case raw files and all
failure traces remain available for review. Exit code 0 means no program error;
it does not mean every navigation task succeeded.

## Render saved executions without rerunning a mission

Choose a completed `cases/<job_id>` directory and a fresh output directory:

```powershell
$env:PYTHONUTF8='1'
$env:MPLBACKEND='Agg'
.\.venv-stage1\Scripts\python.exe -m scripts_stage5.render_saved --case results_stage5/<study>/cases/<job_id> --out results_stage5/<new-figures> --frames 40
```

This exports the original seabed, recorded actual six-DOF trajectory, separate
selected predictions, error/control plots, a GIF and a self-contained
`replay_3d.html`. The renderer reads saved data and never executes a new mission
or overwrites the source. GIF frames use recorded obstacle snapshot times;
HTML allows rotation, zoom and time replay. `FIGURE_SOURCES.json` identifies
the original episode and actual injected predictor. Display downsampling is
independent of full-resolution collision verification.

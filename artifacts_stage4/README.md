# Frozen CA risk inference fixture

`ca_scene_calibration_v1.npz` is the approximately 0.49 MB inference library
constructed from the 24 pre-registered **calibration** B0 episodes on 2026-10-12.
It contains 16 obstacle scenes and 8 explicitly empty terrain scenes, 72 fixed
observed windows and 127 matched target windows. No tune or test episode enters
this library. Collision and timeout calibration episodes are retained.

The 24-second CA histories/labels use 0.2-second offsets. The original 0.02-second
REMUS model and obstacle replay provide the offline labels. Arrays are unchanged
from the local source library; only source-directory metadata is made relative
for distribution. Full raw episodes and build diagnostics stay in the independent
review archive, rather than Git.

Both B1 and B2 can load this file. At alpha 0.1/0.2/0.3, B1 adds approximately
63.0408/46.4346/45.1002 metres to the unchanged engineering prediction envelope.
B2 recomputes its geometry-dependent scene scores for each candidate. These are
uncapped empirical scene-block calibrations, not a closed-loop safety guarantee.
Empty terrain scenes are included in the registered pool; this is not 24
independent dynamic-obstacle trials or a conditional dynamic-scene guarantee.

See [risk methods](../docs_stage4/RISK_METHODS.md) and
[protocol](../docs_stage4/PROTOCOL.md) for the frame-transport assumption,
correlated windows, original ECP attribution and frozen offline limitations.

Run an independent registered test comparison without rebuilding calibration:

```powershell
.\.venv-stage1\Scripts\python.exe run_stage4.py run-paired --split test --alpha .1 --library artifacts_stage4/ca_scene_calibration_v1.npz --workers 2 --out results_stage4/my_fixture_test
```

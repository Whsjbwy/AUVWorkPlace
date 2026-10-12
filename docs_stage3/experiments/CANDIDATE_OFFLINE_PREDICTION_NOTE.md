# Completed same-anchor offline prediction evaluation

All 16 completed candidate episodes were evaluated by `scripts_stage3/analyze_prediction_errors.py` without rerunning navigation. The command exited 0. Results are in `results_stage3/candidate_offline_prediction_20261011`.

```powershell
$env:PYTHONUTF8='1'
$candidateSources = Get-ChildItem results_stage3/common_candidates_20261011/cases -Directory | Sort-Object Name | Select-Object -ExpandProperty FullName
& .venv-stage1\Scripts\python.exe scripts_stage3/analyze_prediction_errors.py --source $candidateSources --out results_stage3/candidate_offline_prediction_20261011 2>&1 | Tee-Object results_stage3/batch_console_20261011/candidate_offline_prediction_console.log
if ($LASTEXITCODE -ne 0) { throw 'candidate offline paired prediction analysis failed; inspect preserved console log' }
```

The existing output is intentionally protected from overwrite; reproducing the analysis requires a new output directory.

- 12,708 paired raw error rows, 369 grouped statistics rows and 16 figures.
- Variants: `ca_dt3`, `pplnm_cv_after9`, `pplnm_ca_after9`.
- Common planning horizons: 3, 6, 9, 12, 18 and 24 s.
- Each predictor receives exactly the same delivered measurement history and absolute target timestamps at each anchor. Support groups and sample counts are identical across the three variants.
- Trained support ends at the latest measurement time plus 9 s. Stale measurements can therefore place a planning-relative 9 s target outside trained support; those rows remain in a separate extension group.
- Offline regenerated truth matched every saved truth snapshot numerically with zero position difference. Truth was read only after the original navigation episodes and never fed back to a planner.
- Zero new vehicle rollouts, parameter fitting, risk margin adjustment or calibration refitting.
- Original candidate navigation used CA with the original 1 s history interval. The offline `ca_dt3` variant is a separate fair common-history comparison, not the actual navigation predictor's RMSE. Original run metadata and NaN placeholders were not rewritten.

Examples from the original diverse9 episodes at planning horizon 9 s:

| Scene | Common support group | Object-anchor samples | CA dt3 RMSE m | PPLNM RMSE m |
|---|---|---:|---:|---:|
| static3d | within 9 s from latest measurement | 19 | 0.000000 | 0.561916 |
| multi_crossing3d, speed scale 1.5 | within 9 s from latest measurement | 53 | 2.624110 | 5.363191 |

These examples do not pool different budgets or scenes, and do not establish prediction superiority. The full CSV preserves each case, horizon, variant and trained-support group. PPLNM's navigation improvement in one static episode is not evidence of more accurate static-obstacle prediction: this offline comparison shows the static CA reference is exact and the learned predictor has nonzero position error.

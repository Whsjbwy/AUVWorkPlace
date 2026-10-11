# 本轮失败归因结论

15组是五类已知困难工况的9/18/45预算对照，不是全部场景的成功率抽样。实际12到达、3地形超时、0碰撞，0/15通过既定有限时域安全验收；同一标准下另7组敏感性实验6组验收通过。没有通过降低安全余量或修改障碍位置取得成功。

- **static3d：**9/18都到达且实际净间隙2.343米，但t=38秒全部45参考无安全解，RPM0滑行预测仅0.829米，因此验收失败。静态障碍CA误差为0，不能归因于CA运动预测；预算45仍有两次不安全滑行预测。
- **terrain3d：**三种预算均超时，实际净间隙20.920米且没有不安全应急。t=30秒全部45参考最大预测净间隙0.930米，滑行24.052米；安全保守滑行维持至超时。场景存在旁侧/上方通道，当前目标深度网格、24秒时域与局部引导不足是待独立消融的根因假设，未证明物理不可达。
- **paper_random3d：**9预算到达但实际0.723米、19次不安全应急；18/45实际余量达标，仍有3/8次不安全应急预测。9预算曾漏掉一个同状态全网格可行参考。24秒CA误差很大，但不能仅凭该误差把所有实际失败判为预测误判。
- **multi_crossing_speed15：**三预算均到达且无碰撞，实际净间隙0.508/0.459/0.459米，全网格未解决失败；还存在可行决策后实际低于1米的区间。已验证一个直接信息缺口：t=30秒障碍离开水平视场，最后t=29秒观测正常交付并保留至t=33秒，33.02秒按4秒记忆过期；t=34秒当前tracks不含该历史已见障碍。原声呐和TrackMemory逐时刻重放与日志一致，未发现ID、交付或删除程序错误；属于感知/过期策略限制。
- **maneuvering_turn8_speed14：**三预算到达且实际余量达标，但仍9/9/10次不安全应急。9/18各有一次筛选遗漏；预算45消除该类遗漏却未通过验收。24秒CA误差约44.5米；需区分未执行预测未来反例与真正执行保持段的反例。

A类仅指**当前45个离散参考中无安全解**，不是物理无解证明。B类是同状态全网格安全参考被入围筛选遗漏，本轮共3次。D类RPM0应急不能保证保持1米余量。E类地形引导和局部网格局限仍待单因素实验。C类实际预测误判用后面的同ID、同绝对时刻、执行保持段检查限定，不能由总体RMSE推断。

执行能力单独计量：全部15例完整同模型保持参考的位置预测误差为0，但实际深度参考RMSE为0.0149–4.4514米、航向参考RMSE为0.00405–46.4031度，实际执行器RPM与施加RPM的滞后RMSE为54.263–225.528。预测一致不代表瞬间兑现参考，也不代表模型已由实艇标定。

以下完整表格及原始逐事件证据来自本轮真实执行；英文报告中success只表示到达，Safety accepted列单独决定验收。所有原始证据在交付ZIP的`raw/budget/`，派生表在`raw/budget/derived_reports/`。

---

# REMUS candidate-budget failure analysis

All 15 listed conditions were newly executed for this study. Seed 7, horizon 24 s, physical step 0.02 s, control period 2 s; zero configured command delay. Original dynamics and the 1 m safety margin were unchanged.

Arrival and safety acceptance are separate. An unsafe emergency forecast or actual sub-margin interval fails safety acceptance, even when the endpoint is reached. A process exit of zero means that the experiment and its diagnostics completed.

| Condition | Budget | Result | Safety accepted | Min clearance (m) | Actual sub-margin intervals | Unsafe emergencies | Missed safe-grid frames | No safe-grid frames | P95 planning (ms) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| static3d | 9 | success | False | 2.343 | 0 | 1 | 0 | 1 | 1862.5 |
| static3d | 18 | success | False | 2.343 | 0 | 1 | 0 | 1 | 3441.9 |
| static3d | 45 | success | False | 1.835 | 0 | 2 | 0 | 2 | 9095.1 |
| terrain3d | 9 | timeout | False | 20.920 | 0 | 0 | 0 | 60 | 2044.8 |
| terrain3d | 18 | timeout | False | 20.920 | 0 | 0 | 0 | 60 | 3675.7 |
| terrain3d | 45 | timeout | False | 20.920 | 0 | 0 | 0 | 60 | 9105.3 |
| paper_random3d | 9 | success | False | 0.723 | 466 | 19 | 1 | 19 | 2018.9 |
| paper_random3d | 18 | success | False | 1.795 | 0 | 3 | 0 | 4 | 3540.3 |
| paper_random3d | 45 | success | False | 1.110 | 0 | 8 | 0 | 8 | 8208.4 |
| multi_speed15 | 9 | success | False | 0.508 | 165 | 3 | 0 | 3 | 1948.3 |
| multi_speed15 | 18 | success | False | 0.459 | 171 | 3 | 0 | 3 | 3598.7 |
| multi_speed15 | 45 | success | False | 0.459 | 171 | 3 | 0 | 3 | 7809.0 |
| maneuver_turn8speed14 | 9 | success | False | 2.172 | 0 | 9 | 1 | 10 | 2040.6 |
| maneuver_turn8speed14 | 18 | success | False | 1.942 | 0 | 9 | 1 | 10 | 3536.1 |
| maneuver_turn8speed14 | 45 | success | False | 1.939 | 0 | 10 | 0 | 11 | 7980.2 |

## Actual execution and obstacle forecast evidence

| Condition | Budget | Sub-margin intervals after feasible plan | CA 2 s RMSE (m) | CA 24 s RMSE (m) | Actual interacted IDs | Max observed / tracked |
|---|---:|---:|---:|---:|---:|---:|
| static3d | 9 | 0 | 0.000 | 0.000 | 1 | 1 / 1 |
| static3d | 18 | 0 | 0.000 | 0.000 | 1 | 1 / 1 |
| static3d | 45 | 0 | 0.000 | 0.000 | 1 | 1 / 1 |
| terrain3d | 9 | 0 | no tracks | no tracks | 0 | 0 / 0 |
| terrain3d | 18 | 0 | no tracks | no tracks | 0 | 0 / 0 |
| terrain3d | 45 | 0 | no tracks | no tracks | 0 | 0 / 0 |
| paper_random3d | 9 | 0 | 1.953 | 87.055 | 3 | 3 / 3 |
| paper_random3d | 18 | 0 | 2.004 | 86.323 | 3 | 3 / 3 |
| paper_random3d | 45 | 0 | 1.315 | 91.102 | 3 | 3 / 3 |
| multi_speed15 | 9 | 96 | 0.553 | 6.632 | 3 | 3 / 3 |
| multi_speed15 | 18 | 98 | 0.553 | 6.632 | 3 | 3 / 3 |
| multi_speed15 | 45 | 98 | 0.548 | 6.577 | 3 | 3 / 3 |
| maneuver_turn8speed14 | 9 | 0 | 0.692 | 44.485 | 3 | 3 / 3 |
| maneuver_turn8speed14 | 18 | 0 | 0.692 | 44.485 | 3 | 3 / 3 |
| maneuver_turn8speed14 | 45 | 0 | 0.711 | 44.769 | 3 | 3 / 3 |

An actual sub-margin interval after a feasible decision is retained as a safety failure. Its presence alongside forecast error does not, by itself, establish which observation, model or decision caused that interval; the per-ID forecast rows and captured planning-state records provide the evidence for a closer audit. Total obstacle population is not used as a substitute for actual interaction or observation density.

Terrain example at t=30 s (budget 9): all 45 tested moving references fail the unchanged clearance requirement. The largest predicted clearance is 0.930176 m at depth/heading/RPM reference [27.0, 0.9597254484673876, 800.0]; emergency coasting predicts 24.052444 m. This records a limitation of the finite local reference grid while the reachable scene also contains side/upper passages. Keeping a safe coast can still end in task timeout.

## Interpretation limits

- Failed shortlisted plans were checked offline against all 45 references using a complete controller/actuator/reference-state clone captured at that exact planning call. Budget 45 records reuse the actual live full-grid evaluations. Previous selected references reproduce history cost. No offline result changed the executed action.
- A shortlist omission means a safe reference existed in that finite grid under the same causal CA forecast. No safe grid reference means only that these 45 references failed this model check; it does not prove global unavoidability. Different budgets subsequently visit different states, so their episode differences are not single-decision counterfactuals.
- Actual clearance comes from the same swept physical-interval collision evaluator as execution. The violation CSV includes every interval below the unchanged 1 m margin, including intervals following a feasible plan. Summed interval duration is the duration of intervals containing a breach, not an exact measurement of continuous exposure time.
- Vehicle prediction error compares only a selected held reference until the next planning decision. Small or zero deterministic clone error does not mean immediate reference tracking. Separate control_response JSON reports actual depth/heading reference errors and actuator RPM lag.
- Obstacle forecast error compares CA positions with the same obstacle ID at the same absolute simulation time. All recorded obstacle truth snapshots were numerically checked against an independent same-seed generator before extending truth through the 24 s forecast. These offline obstacle errors are distinct from vehicle execution error and never enter planning.
- Planner P95 values are actual wall-clock timers under up to three concurrent study processes and other shared-machine activity. Observer and offline diagnostics are excluded from those timers. Physics pauses during planning; measured latency was not injected as execution delay. These values are not isolated real-time guarantees.
- The results identify baseline limitations without adding ECP, reinforcement learning, or a new control policy. Collision thresholds, acceptance rules, controller parameters, and stored original experiments were not relaxed.

## Files

- `budget_comparison.csv`: every requested actual condition/budget and safety, interaction, latency, reference-response, vehicle-error and obstacle-error metrics.
- `failure_event_comparison.csv`: all failure/collision-context events, full-grid counts and 9/18/45 shortlist membership evaluated at the same captured state.
- `candidate_grid_diagnostics.csv`: individual references, fast and physical costs/clearances, feasibility and numerical failure evidence.
- Per-case `failure_diagnostics/event_*.json`: complete captured plant/controller state, causal tracks and all 45 evaluations. Original state/control/planning/error CSV, selected plans, obstacles and PNGs remain alongside them.
- Per-case `actual_margin_diagnostics/`: every violating actual physical interval, contiguous groups and clearance plot.
- Per-case `obstacle_forecast_errors.csv` and summary: same-ID/time errors, ranges, ages and 2 s/24 s RMSE/max, separate from actual-control response.

## 同时间反事实检查与C类归因边界

另实际执行 `scripts_stage2/analyze_forecast_decisions.py --source results_stage2/review_20261011_budget --require-complete --out results_stage2/review_20261011_counterfactual/all_complete`，退出0，6.216秒。完整分析664个可行决策、83732个当前跟踪预测点、1854条记录真值快照；真值重放和已执行车辆点均与记录一致。

在0.2秒保存点上，已执行保持区间没有发现**当前已跟踪对象**的CA假安全反例；不能据此证明0.02秒连续区间无反例，更不能否定真实扫掠记录的安全失败。另31个决策的已跟踪对象反例只在随后未执行的24秒预测未来出现：maneuver预算9/18/45为3/3/4，paper为7/7/7；它们显示长时CA可信度有限，不能直接归因为实际事故。

多障碍三预算各10个真实保持段子阈值点都对应**历史已交付、当前跟踪已丢失**的对象，而非当前已跟踪对象CA误差；点上最小间隙0.508560/0.459696/0.459610米。原0.02秒扫掠结果仍为主安全依据。预算9详细重放证据在counterfactual根目录的tracking_gap_multi_speed15_budget9.json，全部原始归因表在交付ZIP的raw/additional/review_20261011_counterfactual/。

这轮没有据未完成消融改造跟踪过期、备用动作或全局地形引导；它们是下一轮应单独验证的基线改进位置。

# 第二阶段实际测试报告（Windows / CPU，2026-10-11）

公开版本仅将历史命令的本机工作目录前缀替换为 `<repository>`；实际命令和数值原日志仍完整保存在独立实验交付中。
最终全量 **251 passed，0 failed，0 errors，0 skipped**：原147项完整保留，新增104项。
正式REMUS六自由度14个工况：**13到达，9通过有限时域避障验收，0实际碰撞，1超时；4项到达但应急安全验证失败**。这些是单种子工程检查，不能当作统计成功率或实艇安全证明。

工作目录 `<repository>`，独立 `.venv-stage1`，Python3.12.14、NumPy2.3.5、Matplotlib3.10.8、pytest9.0.2、PyTorch2.10.0 CPU。仿真和现成LSTM均未使用GPU；未新建或破坏原 `.venv`。README给出独立 `.venv-stage2` 安装方式。

## 执行证据与判定

- [最终JUnit](../results_stage2/final_tests_20261011/pytest.xml)、[测试命令/退出码/环境](../results_stage2/final_tests_20261011/regression_report.json)、[pytest日志](../results_stage2/final_tests_20261011/pytest.log)。
- [原功能回归16项完整记录](../results_stage2/regression_20261011/regression_report.json)。较早全量235项也实际通过，随后增加适配修复、回放、加载及安全分类测试，最终重跑251项。
- [14项实际命令、日志、摘要](../results_stage2/acceptance_20261011/acceptance_results.json)、[统一安全判定与媒体复核](../results_stage2/acceptance_20261011/acceptance_analysis.json)。
- 首批Windows子进程日志含GBK输出；原日志保留，另有[UTF-8可读副本及转换说明](../results_stage2/acceptance_20261011/logs_utf8/encoding_notes.json)。新验收脚本统一子进程UTF-8。
- [独立数值与媒体复查](../results_stage2/inspection_final_20261011.json)。

子进程退出0只代表程序完成。导航 `status/reached_goal` 与 `avoidance_acceptance_passed` 分开：后者要求到达、实际净间隙≥配置1米且无不安全应急预测。碰撞阈值仍为包络球或地形净间隙≤0，未放宽。早期正式摘要未含新分类字段，原文件保持不动，最终报告与analysis从原间隙和事件派生同一判定。

## 全量单元与功能测试

| 测试组 | 实际通过数 |
|---|---:|
| 原PythonVehicleSimulator | 18 |
| 原阶段1 | 129 |
| 阶段2 `test_control_prediction` | 20 |
| 阶段2 `test_scenarios` | 45 |
| 阶段2 `test_diagnostics` | 4 |
| 阶段2 `test_closed_loop` | 7 |
| 阶段2 `test_shared_p25` | 11 |
| 阶段2 `test_stage2_visualization` | 7 |
| 阶段2 `test_replay` | 8 |
| 阶段2 `test_saved_outputs` | 2 |

新增测试覆盖RPM/速度、深度/航向跟踪、原限幅与执行器滞后、非零横滚俯仰变换、海流不重复积分、完整控制器/参考/执行器状态隔离、预测与执行一致、3D碰撞与共同时间轴、观测延迟、无可行候选滑行、安全分类、场景可达条件、日志/图片/GIF/HTML与离线加载。没有删除原测试或用放宽碰撞制造通过。

实际最终命令：

```powershell
.\.venv-stage1\Scripts\python.exe scripts_stage2/run_regression.py --tests-only --out results_stage2/final_tests_20261011
```

脚本将pytest工作目录和动画放入新结果目录，保存完整子命令、退出码及JUnit。

## 原功能回归与独立控制响应

| 命令项 | 结果 | 实际墙钟s | 日志 |
|---|---|---:|---|
| versions | PASS | 0.68 | [versions.log](../results_stage2/regression_20261011/versions.log) |
| pip_check | PASS | 0.83 | [pip_check.log](../results_stage2/regression_20261011/pip_check.log) |
| audit | PASS | 0.61 | [audit.log](../results_stage2/regression_20261011/audit.log) |
| pytest | PASS | 43.28 | [pytest.log](../results_stage2/regression_20261011/pytest.log) |
| original_remus | PASS | 14.95 | [original_remus.log](../results_stage2/regression_20261011/original_remus.log) |
| regression | PASS | 1.40 | [regression.log](../results_stage2/regression_20261011/regression.log) |
| responses | PASS | 6.80 | [responses.log](../results_stage2/regression_20261011/responses.log) |
| diagnose | PASS | 0.99 | [diagnose.log](../results_stage2/regression_20261011/diagnose.log) |
| stage1_ca_empty3d | PASS | 1.96 | [stage1_ca_empty3d.log](../results_stage2/regression_20261011/stage1_ca_empty3d.log) |
| stage1_ca_static3d | PASS | 1.95 | [stage1_ca_static3d.log](../results_stage2/regression_20261011/stage1_ca_static3d.log) |
| stage1_ca_crossing3d | PASS | 1.99 | [stage1_ca_crossing3d.log](../results_stage2/regression_20261011/stage1_ca_crossing3d.log) |
| stage1_ca_paper_random3d | PASS | 3.25 | [stage1_ca_paper_random3d.log](../results_stage2/regression_20261011/stage1_ca_paper_random3d.log) |
| stage1_pplnm_empty3d | PASS | 3.02 | [stage1_pplnm_empty3d.log](../results_stage2/regression_20261011/stage1_pplnm_empty3d.log) |
| stage1_pplnm_static3d | PASS | 3.14 | [stage1_pplnm_static3d.log](../results_stage2/regression_20261011/stage1_pplnm_static3d.log) |
| stage1_pplnm_crossing3d | PASS | 3.23 | [stage1_pplnm_crossing3d.log](../results_stage2/regression_20261011/stage1_pplnm_crossing3d.log) |
| stage1_pplnm_paper_random3d | PASS | 4.79 | [stage1_pplnm_paper_random3d.log](../results_stage2/regression_20261011/stage1_pplnm_paper_random3d.log) |

原REMUS 200秒仿真含GIF完成；数值回归与原README舍入值一致，深度/航向/航程绝对差分别约4.74×10⁻⁶米、1.82×10⁻⁶度、5.23×10⁻⁷米。原PVS、锁文件和cmd入口未修改；阶段2没有修改run_remus.py。参数审计保留原质量、惯性、阻尼、舵面和推进器模型。

独立控制响应9/9通过：从静止启动，RPM600→1400使相对纵向速度约1.001→2.344m/s；深度35→40米阶跃在100秒仍差−0.975米，航向40度阶跃仍差−0.881度。四个正交流向0.3m/s与东北0.1/西南0.5m/s均实际运行，完整流坐标恒等式残差最大约6.66×10⁻¹⁶m/s。慢参考动态保留，不能假设瞬时跟踪。

已有150轮PPLNM权重按原配置实际加载并预测，CA与PPLNM各四类阶段1场景均到达；无需重新训练。REMUS阶段2使用因果CA障碍预测，未把三步/9秒LSTM直接外推为24秒可信预测。

## REMUS六自由度正式场景结果

所有工况同一默认：物理步长0.02秒、规划周期2秒、预测24秒、45个深度/航向/RPM参考、最多9个完整预测，必要时另加1个应急预测；20秒物理预热后只平移位置回场景起点，保留实际速度/姿态/控制器/执行器状态。除表中参数外几何、原物理参数、评分和碰撞条件不变。

| 工况 | 导航结局/时长s | 最小实际净间隙m | 近距对象/最多同时 | 最大观测/跟踪 | 应急/不安全 | 避障验收 |
|---|---|---:|---:|---:|---:|---|
| [empty3d](../results_stage2/acceptance_20261011/empty3d/summary.json) | success / 84.92 | 29.17596 | 0/0 | 0/0 | 0/0 | 通过有限模型检查 |
| [static3d](../results_stage2/acceptance_20261011/static3d/summary.json) | success / 87.40 | 2.34252 | 1/1 | 1/1 | 1/1 | 失败：不安全应急/余量 |
| [crossing3d](../results_stage2/acceptance_20261011/crossing3d/summary.json) | success / 86.48 | 13.53272 | 1/1 | 1/1 | 0/0 | 通过有限模型检查 |
| [head_on3d](../results_stage2/acceptance_20261011/head_on3d/summary.json) | success / 86.16 | 1.75153 | 1/1 | 1/1 | 0/0 | 通过有限模型检查 |
| [multi_crossing3d](../results_stage2/acceptance_20261011/multi_crossing3d/summary.json) | success / 106.98 | 1.91030 | 3/3 | 3/3 | 9/0 | 通过有限模型检查 |
| [maneuvering3d](../results_stage2/acceptance_20261011/maneuvering3d/summary.json) | success / 86.96 | 11.43157 | 3/3 | 3/3 | 0/0 | 通过有限模型检查 |
| [terrain3d](../results_stage2/acceptance_20261011/terrain3d/summary.json) | timeout / 150.00 | 20.91985 | 0/0 | 0/0 | 60/0 | 失败：超时 |
| [tracking_mismatch3d](../results_stage2/acceptance_20261011/tracking_mismatch3d/summary.json) | success / 86.32 | 29.17596 | 0/0 | 0/0 | 0/0 | 通过有限模型检查 |
| [paper_random3d](../results_stage2/acceptance_20261011/paper_random3d/summary.json) | success / 191.66 | 0.72277 | 3/2 | 3/3 | 20/19 | 失败：不安全应急/余量 |
| [crossing_current03_dir90](../results_stage2/acceptance_20261011/crossing_current03_dir90/summary.json) | success / 86.58 | 6.36736 | 1/1 | 1/1 | 0/0 | 通过有限模型检查 |
| [multi_crossing_speed15](../results_stage2/acceptance_20261011/multi_crossing_speed15/summary.json) | success / 94.92 | 0.50811 | 3/3 | 3/3 | 3/3 | 失败：不安全应急/余量 |
| [maneuvering_turn8_speed14](../results_stage2/acceptance_20261011/maneuvering_turn8_speed14/summary.json) | success / 106.30 | 2.17212 | 3/3 | 3/3 | 11/9 | 失败：不安全应急/余量 |
| [crossing_range60_delay1](../results_stage2/acceptance_20261011/crossing_range60_delay1/summary.json) | success / 84.82 | 6.10849 | 1/1 | 1/1 | 0/0 | 通过有限模型检查 |
| [crossing_range35_delay2](../results_stage2/acceptance_20261011/crossing_range35_delay2/summary.json) | success / 105.46 | 3.86442 | 1/1 | 1/1 | 11/0 | 通过有限模型检查 |

近距以实际车体包络与障碍表面净间隙≤30米统计；碰撞按同一物理区间扫掠球与保守地形检查。多交叉和变速场景实际同时交互3个障碍，补足仅看总数的不足。原随机场景虽生成120个，但实际近距只有3个、最多同时2个、最大跟踪3个，不能据120这个总数宣称密集动态避障充分验证。

| 工况 | 规划中位数ms | P95 ms | 最大ms | 超2秒次数 |
|---|---:|---:|---:|---:|
| empty3d | 1111.69 | 1531.66 | 1710.35 | 0 |
| static3d | 1136.89 | 1393.12 | 1568.76 | 0 |
| crossing3d | 1057.67 | 1202.06 | 1308.53 | 0 |
| head_on3d | 1074.44 | 1134.19 | 1167.31 | 0 |
| multi_crossing3d | 1075.33 | 1219.88 | 1310.13 | 0 |
| maneuvering3d | 1074.27 | 1155.23 | 1187.76 | 0 |
| terrain3d | 1182.17 | 1248.55 | 1294.62 | 0 |
| tracking_mismatch3d | 1069.65 | 1103.56 | 1301.90 | 0 |
| paper_random3d | 1077.87 | 1225.86 | 1257.12 | 0 |
| crossing_current03_dir90 | 1078.51 | 1174.73 | 1308.12 | 0 |
| multi_crossing_speed15 | 1299.29 | 1597.84 | 1709.62 | 0 |
| maneuvering_turn8_speed14 | 1101.04 | 1321.29 | 1374.08 | 0 |
| crossing_range60_delay1 | 1072.57 | 1117.08 | 1183.02 | 0 |
| crossing_range35_delay2 | 1062.08 | 1204.86 | 1237.04 | 0 |

正式P95范围约1.104–1.598秒，全部实际单次规划耗时未超过2秒；部分工况与回归测试并行，结果不是空闲硬件的性能认证。当前同步离线仿真会暂停物理时钟等待规划，尚未把计算耗时注入真实命令延迟。

## 真实失败、修复与剩余问题

1. 开发12秒时域：静态场景150秒超时，最小间隙4.002米、60次应急；迎面场景51.98秒真实碰撞，间隙−0.01689米。相同几何统一改24秒后改善，原失败保存在 `results_stage2/development_*`，没有改碰撞阈值。
2. 正式静态场景虽到达，但38秒发生一次不安全应急预测。障碍离开前向视场后，带观测年龄的保守余量使有限候选不可行；车辆真实滑行未碰撞。此项明确不通过安全验收，不能用到达掩盖。
3. 海床脊150秒超时、距目标108.22米；实际停滞在约N=101.80、D=32.75米、u=0.00090m/s。慢深度参考、局部时域及减速后的舵效导致规划困境；需要更早的地形航路引导与可恢复的应急动作，而不是降低地形碰撞标准。
4. 随机场景、1.5倍速度多交叉、8度/秒曲线参数且1.4倍速度的场景均到达，但存在不安全应急预测；前两项实际最小间隙也小于1米配置余量，均记验收失败。CA加确定性余量不具备已校准的未知转向保证。
5. 共享P25的24秒候选曾触发Euler奇异，初次4项错误日志保留；在阶段2局部适配层按当前俯仰和时域筛除超原±30度的候选后修复。原9秒入口候选/评分一致性新增测试通过，原planner未改。
6. 初期保存仅含观测时刻的障碍快照，GIF可能早于实际终点结束。后续保存补存离线终点真值；早期crossing在新目录重绘，先核对87个原快照，最大位置差0米，再补86.48秒终点。原实验不改写，重绘不算新车辆运行或安全验收。

## P25共享场景与执行错配诊断

| P25对照 | 结局 | 时长s | 实际间隙m |
|---|---|---:|---:|
| empty3d | success | 108.22 | 29.17838 |
| static3d | success | 110.68 | 2.08075 |
| crossing3d | success | 109.34 | 3.39208 |
| multi_crossing3d | obstacle_collision | 42.44 | -0.00255 |

P25多交叉42.44秒真实碰撞，−0.00255米；未删除该结果。共享场景、感知、障碍和评价，但车辆动力学、动作及初速不同，不能作为只比较算法的公平胜率或REMUS成功证据。

8秒错配诊断：完整REMUS克隆预测与真实执行RMSE/最大误差均为0；单独理想P25速率对照RMSE3.372米、末端7.407米，实际仍距参考深度−7.475米、航向−65.416度。P25速率与深度/航向/RPM不是等价命令。所有正式已选参考在下一重规划前的预测误差也为0，只证明同模型、同完整初态、保持参考的数值一致，未验证真实平台扰动或硬件误差。

## 可视化与结果复核

- [三维地形](../results_stage2/visuals_final/crossing3d/terrain_3d.png)、[实际/预测航迹](../results_stage2/visuals_final/crossing3d/trajectory_local_3d.png)、[60帧GIF](../results_stage2/visuals_final/crossing3d/trajectory.gif)、[误差曲线](../results_stage2/visuals_final/crossing3d/prediction_error.png)、[控制曲线](../results_stage2/visuals_final/crossing3d/controls.png)、[离线三维HTML](../results_stage2/visuals_final/crossing3d/replay_3d.html)。
- [海床脊真实失败航迹](../results_stage2/acceptance_20261011/terrain3d/trajectory_3d.png)、[理想假设错配曲线](../results_stage2/regression_20261011/diagnose/tracking_diagnostic.png)。
- PNG/GIF经Pillow实际读取；代表性NED图、控制图和GIF末帧已目视检查。终止时间/真值/动画末帧均为86.48秒，见[重绘说明](../results_stage2/visuals_final/crossing3d/render_manifest.json)及[QA记录](../results_stage2/visuals_final/crossing3d/render_qa.json)。
- HTML数据/无未来显示/无网络资源等测试通过，Node语法检查退出0；未自动执行浏览器交互验收，不声称这项已测。

独立复查14目录结果：`all_passed=True`。复算有限值、物理/规划时钟、控制区间与执行器末值对齐、原15度/1525RPM限制、完整候选预算、预测起点、P95/RMSE与导航结局，并读取图片。这里的通过表示日志数值和媒体一致，不把导航/安全失败改成成功。

```powershell
.\.venv-stage1\Scripts\python.exe scripts_stage2/inspect_results.py results_stage2/acceptance_20261011 --out results_stage2/inspection_final_20261011.json
.\.venv-stage1\Scripts\python.exe scripts_stage2/render_saved.py --source results_stage2/acceptance_20261011/crossing3d --out results_stage2/visuals_final/crossing3d
```

以上是实际使用的路径；再次执行须换新输出目录/文件，禁止覆盖历史。

## 实际逐项命令

原功能回归：

```powershell
<repository>\.venv-stage1\Scripts\python.exe -m pip list --format=json
<repository>\.venv-stage1\Scripts\python.exe -m pip check
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py audit --out <repository>\results_stage2\regression_20261011\audit.json
<repository>\.venv-stage1\Scripts\python.exe -m pytest <repository>\PythonVehicleSimulator\tests <repository>\tests_stage1 <repository>\tests_stage2 -q --junitxml <repository>\results_stage2\regression_20261011\pytest.xml --basetemp <repository>\results_stage2\regression_20261011\pytest_temp
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_remus.py --duration 200 --out <repository>\results_stage2\regression_20261011\original_remus
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py regression --out <repository>\results_stage2\regression_20261011\regression
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py responses --out <repository>\results_stage2\regression_20261011\responses
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py diagnose --out <repository>\results_stage2\regression_20261011\diagnose
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario empty3d --out <repository>\results_stage2\regression_20261011\stage1_ca_empty3d
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario static3d --out <repository>\results_stage2\regression_20261011\stage1_ca_static3d
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario crossing3d --out <repository>\results_stage2\regression_20261011\stage1_ca_crossing3d
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario paper_random3d --out <repository>\results_stage2\regression_20261011\stage1_ca_paper_random3d
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario empty3d --out <repository>\results_stage2\regression_20261011\stage1_pplnm_empty3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario static3d --out <repository>\results_stage2\regression_20261011\stage1_pplnm_static3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario crossing3d --out <repository>\results_stage2\regression_20261011\stage1_pplnm_crossing3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario paper_random3d --out <repository>\results_stage2\regression_20261011\stage1_pplnm_paper_random3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
```

正式六自由度：

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario empty3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\empty3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario static3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\static3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario crossing3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\crossing3d
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario head_on3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\head_on3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario multi_crossing3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\multi_crossing3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario maneuvering3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\maneuvering3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario terrain3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\terrain3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario tracking_mismatch3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\tracking_mismatch3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario paper_random3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 200 --out <repository>\results_stage2\acceptance_20261011\paper_random3d --no-animation
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario crossing3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\crossing_current03_dir90 --no-animation --current-speed .3 --current-direction-deg 90
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario multi_crossing3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\multi_crossing_speed15 --no-animation --obstacle-speed-scale 1.5
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario maneuvering3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\maneuvering_turn8_speed14 --no-animation --obstacle-turn-rate-deg-s 8 --obstacle-speed-scale 1.4
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario crossing3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\crossing_range60_delay1 --no-animation --sensor-range-m 60 --observation-delay-s 1
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage2.py simulate --backend remus6dof --scenario crossing3d --config <repository>\configs\stage2_development.json --horizon-s 24 --control-dt 2 --candidate-budget 9 --max-time 150 --out <repository>\results_stage2\acceptance_20261011\crossing_range35_delay2 --no-animation --sensor-range-m 35 --observation-delay-s 2
```

P25对照：

```powershell
.\.venv-stage1\Scripts\python.exe run_stage2.py simulate --backend p25 --scenario empty3d --max-time 150 --no-animation --out results_stage2/p25_shared_20261011/empty3d
.\.venv-stage1\Scripts\python.exe run_stage2.py simulate --backend p25 --scenario static3d --max-time 150 --no-animation --out results_stage2/p25_shared_20261011/static3d
.\.venv-stage1\Scripts\python.exe run_stage2.py simulate --backend p25 --scenario crossing3d --max-time 150 --no-animation --out results_stage2/p25_shared_20261011/crossing3d
.\.venv-stage1\Scripts\python.exe run_stage2.py simulate --backend p25 --scenario multi_crossing3d --max-time 150 --no-animation --out results_stage2/p25_shared_20261011/multi_crossing3d
```

## 交付边界

六自由度闭环已实现并实际运行；静态安全应急、复杂动态余量与海床全局可达性仍未全部验收。后续接入点见 [ECP_INTERFACES.md](ECP_INTERFACES.md)，不能先用ECP名义覆盖这些基础问题。没有引入ECP/强化学习，没有额外REMUS源码，没有文件/模型/Git对象校验和检查。所有新结果在results_stage2，历史结果保留，未推送GitHub。

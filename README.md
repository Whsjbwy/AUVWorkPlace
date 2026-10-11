# AUVWorkPlace

同一仓库保留原 Fossen REMUS 100 六自由度仿真，整合阶段1三维海底、障碍、声呐、物理先验 LSTM 与三维 DWA，并新增使用原REMUS控制器和执行器的六自由度闭环避障基线。基础仿真与已有模型预测均支持CPU。

| 入口 | 执行模型 | 功能 |
|---|---|---|
| `run_remus.py` | 原 REMUS 六自由度水动力学 | 深度/航向控制、水平海流、三维轨迹与 GIF |
| `run_stage1.py regression / remus-characterize / remus-route` | 同一份原 REMUS 模型 | 原例回归、8组运行点检查、独立三维航点跟踪 |
| `run_stage1.py simulate` | P25 简化三维运动学 | 地形、静态/动态障碍、碰撞、DWA、CA/PPLNM 预测 |
| `run_stage2.py simulate --backend remus6dof` | 原 REMUS 六自由度水动力学 | 真实状态反馈、参考空间DWA、隔离六自由度预测、原控制器与执行器执行 |
| `run_stage2.py simulate --backend p25` | P25 简化三维运动学 | 阶段2共同场景、感知时序与碰撞评价下的运动学对照 |
| `run_stage2.py responses / diagnose` | 原 REMUS 六自由度水动力学 | 速度/深度/航向/海流响应，以及理想运动学与真实控制参考响应的错配诊断 |

**阶段2已经实现REMUS六自由度状态反馈与实际执行闭环。** 新后端采样目标深度、航向与RPM，经原自动驾驶控制器生成舵角，原模型以0.02秒积分并处理执行器滞后。P25的 `[u,q,r]` 不会直接当作舵角。第一阶段避障轨迹及 `--backend p25` 仍是简化运动学结果，图例和日志明确标注；它们不代表REMUS六自由度避障成功。

阶段2是有限时域工程基线，没有ECP或新型学习算法。任务完成、碰撞、超时和应急风险必须按真实结果判断；最新逐项执行证据见 [阶段2测试报告](docs_stage2/TEST_REPORT.md)。阶段1 [接口清单](docs_stage1/REMUS_INTERFACE.md) 保留为历史记录，当前接口见 [阶段2后续接口](docs_stage2/ECP_INTERFACES.md)。

阶段2六自由度后端的障碍预测当前仍为 **CA**，尚未接入PPLNM；已有PPLNM只用于阶段1P25后端。默认45个参考中最多9个进行完整REMUS验证，无可行参考时额外检查RPM0滑行。没有ECP或可靠电功率标定，不能报告真实电能节省。同步默认运行不把墙钟规划耗时加入物理时间。

本轮代码复审、9/18/45预算对照和失败归因见 [复审报告](docs_stage2/review_20261011/TEST_REPORT.md) 与 [失败分析](docs_stage2/review_20261011/FAILURE_ANALYSIS.md)。此前251项测试、14工况13到达/9有限时域验收通过属于历史阶段2结果，不能替代本轮记录。长期研究分支、提交边界和自动测试流程见 [版本管理](docs_stage2/VERSION_CONTROL.md)。原始实验目录和审查ZIP留在本地，Git只保存源码、配置、必要推理权重和小体积报告。

## Windows / PowerShell 安装

阶段2建议使用独立环境，保留原 `.venv` 和已有 `.venv-stage1`。在仓库根目录执行：

```powershell
py -3.12 -m venv .venv-stage2
.\.venv-stage2\Scripts\python.exe -m pip install -r requirements-stage1-core.txt
$env:MPLBACKEND = 'Agg'
$env:PYTHONUTF8 = '1'
.\.venv-stage2\Scripts\python.exe -m pip check
```

核心依赖足以运行REMUS闭环、P25对照、响应检查和图像/GIF。要执行已有PPLNM测试与阶段1权重预测，再安装可选学习依赖：

```powershell
.\.venv-stage2\Scripts\python.exe -m pip install -r requirements-stage1-learning.txt
```

不需要GPU。新依赖环境复用已验证的阶段1依赖清单，原 `requirements-lock.txt` 保持不变。若Windows未注册 `py -3.12`，用本机Python3.12解释器的绝对路径替换；本次实际验证仍使用已安装依赖的 `.venv-stage1`，新环境命令是可复制的独立安装方式。

以下保留阶段1安装和运行方式：

在仓库根目录使用 Python 3.12，新建环境，不替换原 `.venv`：

```powershell
py -3.12 -m venv .venv-stage1
.\.venv-stage1\Scripts\python.exe -m pip install -r requirements-stage1-learning.txt
$env:MPLBACKEND = 'Agg'
```

如果默认解释器已经是3.12，也可用 `python -m venv .venv-stage1`。不使用 LSTM 时只安装 `requirements-stage1-core.txt`。核心依赖包括本仓库 `PythonVehicleSimulator` 的可编辑安装；原 `requirements-lock.txt` 保持原样。

历史本机检查中默认 `python` 为3.13.5、`py -3.12`未注册，使用已安装的3.12.14解释器创建了独立环境。其他机器请用自己的Python3.12路径；仓库不依赖某个用户目录。

本次 Windows 实测环境：Python3.12.14、NumPy2.3.5、Matplotlib3.10.8、PyTorch2.10.0 CPU、pytest9.0.2。没有RTX5060/CUDA训练实测。实际测试记录见 [TEST_REPORT.md](docs_stage1/TEST_REPORT.md)。

## 阶段2：REMUS六自由度闭环避障

下列命令使用新环境；也可以将 `.venv-stage2` 替换为已安装完成的 `.venv-stage1`：

```powershell
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
.\.venv-stage2\Scripts\python.exe .\run_stage2.py simulate --backend remus6dof --scenario empty3d --config configs/stage2_development.json --out "results_stage2/empty_$runStamp"
.\.venv-stage2\Scripts\python.exe .\run_stage2.py simulate --backend remus6dof --scenario static3d --out "results_stage2/static_$runStamp"
.\.venv-stage2\Scripts\python.exe .\run_stage2.py simulate --backend remus6dof --scenario crossing3d --out "results_stage2/crossing_$runStamp"
Get-Content "results_stage2/crossing_$runStamp/summary.json"
Start-Process "results_stage2/crossing_$runStamp/trajectory_3d.png"
Start-Process "results_stage2/crossing_$runStamp/trajectory.gif"
Start-Process "results_stage2/crossing_$runStamp/prediction_error.png"
Start-Process "results_stage2/crossing_$runStamp/replay_3d.html"
```

可选场景为 `empty3d`、`static3d`、`head_on3d`、`crossing3d`、`multi_crossing3d`、`maneuvering3d`、`terrain3d`、`tracking_mismatch3d` 和 `paper_random3d`。场景定义、合理通道和感知条件见 [SCENARIOS.md](docs_stage2/SCENARIOS.md)。例如：

```powershell
.\.venv-stage2\Scripts\python.exe .\run_stage2.py simulate --backend remus6dof --scenario multi_crossing3d --out "results_stage2/multi_$runStamp"
.\.venv-stage2\Scripts\python.exe .\run_stage2.py simulate --backend remus6dof --scenario crossing3d --current-speed 0.3 --current-direction-deg 90 --sensor-range-m 80 --observation-delay-s 1 --out "results_stage2/current_delay_$runStamp"
.\.venv-stage2\Scripts\python.exe .\run_stage2.py simulate --backend remus6dof --scenario maneuvering3d --obstacle-speed-scale 1.2 --obstacle-turn-rate-deg-s 6 --out "results_stage2/turning_$runStamp"
```

默认物理步长0.02秒、规划周期2秒、预测时域24秒，每次最多对9个入围参考做完整六自由度验证；无可行参考时额外验证1个RPM0应急滑行参考。快速近似只排序候选，不提供六自由度安全验证。12秒开发时域在静态场景超时、迎面场景碰撞；正式默认统一延长到24秒，几何和碰撞条件保持不变。正式规划延迟中位数、P95和超预算次数直接来自各运行 `summary.json`，不要据程序能运行就推断满足实时要求。

默认先做20秒独立物理预热，再把位置平移到场景起点，保留实际速度、姿态、控制器积分、参考模型及执行器状态，任务时钟与障碍时钟从0开始。这是明确的巡航初始化；从静止启动使用 `--warmup-s 0`。动态障碍与REMUS按同一物理时刻更新，规划仅使用已经交付的观测和实际车辆状态。每个规划周期重新复制完整车辆状态，其他候选的预测不会推进真实车辆。

无可行参考时RPM0也需要原推进器减速，车辆仍会滑行或随流漂移。日志保留应急预测是否安全、实际应急次数、碰撞/超时原因；不能将应急状态解释为安全停止。实际碰撞使用0.02秒执行轨迹与同时间区间的障碍扫掠包络球，并保留原保守地形检查。

`status` / `reached_goal` 表示实际导航结局。`avoidance_acceptance_passed` 另要求到达、实际间隙满足配置余量且没有不安全应急预测；`safety_validation_status` 标明碰撞、余量不足、应急安全检查失败或未完成。到达不能消除此前的安全验收失败。当前为同步离线仿真，规划计算期间仿真物理时钟暂停；记录的墙钟耗时尚未作为执行延迟注入，因此P95不能单独证明实时平台安全。

输出包括 `states.csv`、`controls.csv`、`planning.csv`、`prediction_errors.csv`、`selected_plans.json`、`obstacles.json`、`events.json`、`terrain.npz`、配置与结果JSON，以及地形图、三维实际航迹图、局部图、控制曲线、误差曲线和GIF。`replay_3d.html` 是无需网络或额外依赖的三维回放，可拖动旋转、滚轮缩放、调整时间和播放速度。实际REMUS执行、当前预测、动态障碍快照时刻及应急事件分别标注；深度轴向下。显示抽稀保留末点，不参与碰撞检查。`--no-animation` 跳过GIF，`--no-plot` 只保留数值结果。非空结果目录拒绝覆盖。

`prediction_errors.csv` 比较已选参考在下一次重规划前的预测与真实执行。同一确定性REMUS模型、相同初态和保持参考时误差可为0，这说明状态隔离和数值一致，不能说明真实平台没有执行误差，也不能证明24秒后障碍预测仍然准确。原水动力学海流保留仅按航向转换的近似，坐标积分和独立诊断使用完整ZYX姿态变换，海流不会重复积分。

复审新增 `observations.json`（测量、计划交付、实际交付时间）、`planning_diagnostics.json`（每周期可行数/失败原因/使用的因果历史）及 `clearance.csv`（真实扫掠净间隙与1米余量之差）。可选 `--command-delay-s 0.5` 在物理时钟中延迟命令生效，期间继续执行前一参考；这是固定延迟敏感性对照，不等于实测墙钟耗时或完整异步实时调度。此时误差按计划原决策时刻对齐，保留延迟导致的执行偏差。P25后端拒绝此REMUS专用设置。

```powershell
.\.venv-stage2\Scripts\python.exe scripts_stage2/run_budget_study.py --workers 1 --out "results_stage2/budget_$runStamp"
.\.venv-stage2\Scripts\python.exe scripts_stage2/run_sensitivity_study.py --out "results_stage2/sensitivity_$runStamp"
.\.venv-stage2\Scripts\python.exe run_stage2.py simulate --scenario crossing3d --sensor-hfov-deg 90 --sensor-vfov-deg 6 --command-delay-s 0.5 --out "results_stage2/narrow_delay_$runStamp"
```

预算脚本对失败规划时刻捕获完整控制器/参考/执行器状态，执行结束后独立检查全部45个参考；离线结果不会改变已执行动作。9/18/45预算分别重新运行完整任务，不能用离线存在可行参考证明整条任务一定能成功。多进程模式的P95受共享CPU竞争影响，实时能力评估应使用单进程并隔离负载。

## 阶段2：共享场景的P25对照与响应诊断

```powershell
.\.venv-stage2\Scripts\python.exe .\run_stage2.py simulate --backend p25 --scenario crossing3d --out "results_stage2/p25_crossing_$runStamp"
.\.venv-stage2\Scripts\python.exe .\run_stage2.py responses --out "results_stage2/responses_$runStamp"
.\.venv-stage2\Scripts\python.exe .\run_stage2.py diagnose --out "results_stage2/diagnostic_$runStamp"
Start-Process "results_stage2/responses_$runStamp/control_responses.png"
Start-Process "results_stage2/diagnostic_$runStamp/tracking_diagnostic.png"
```

P25对照共享场景生成器、起终点、障碍真值、传感器参数、观测延迟及碰撞评价；它仍使用简化动力学和 `[u,q,r]`，初始对水速度固定1m/s，REMUS则采用配置的物理预热。车辆模型、动作空间和初始速度不同，因此这不是只改变算法的公平胜率比较；阶段2比较入口不加载9秒范围的阶段1PPLNM权重。已有LSTM预测继续使用下方阶段1命令。

`responses` 实际检查RPM600→1400、深度35→40米、航向0→40度和6种水平流。`diagnose` 在开放可达环境施加+8米深度、+70度航向、RPM1400参考8秒，对比完整REMUS预测/实际与单独P25理想速率轨迹，不声明避障成功。真实数值与解释见 [场景及响应记录](docs_stage2/SCENARIOS.md)。

## 原 REMUS 六自由度程序

```powershell
.\.venv-stage1\Scripts\python.exe .\run_remus.py --out results_stage1/my_remus
.\.venv-stage1\Scripts\python.exe .\run_remus.py --depth 20 --heading 90 --duration 200 --current 0.3 --no-animation --out results_stage1/my_remus_custom
Start-Process .\results_stage1\my_remus\trajectory.png
Start-Process .\results_stage1\my_remus\trajectory.gif
```

原 `.venv` 和 `run_remus.cmd` 可以继续使用。新增 `--out` 仅选择输出目录，不改默认参数；不提供时仍写入 `results/remus100/` 并覆盖同名文件。`--no-animation` 跳过GIF。建议每次选择不同目录。

默认时长200s、步长0.02s、深度30m、航向50°、1525rpm、水平流0.5m/s且流向170°，初始状态为零。输出 `simulation.csv`、`summary.json`、状态和控制曲线、三维PNG与GIF。NED坐标为北、东、深度向下，姿态采用ZYX欧拉角。

200s回归为10001点，最终深度约29.74975m、航向49.99890°、路径长453.55089m。保留原控制器约0.25m深度偏差。

## 三维环境与 DWA

```powershell
.\.venv-stage1\Scripts\python.exe .\run_stage1.py audit --out results_stage1/my_audit.json
.\.venv-stage1\Scripts\python.exe .\run_stage1.py simulate --scenario empty3d --out results_stage1/my_empty
.\.venv-stage1\Scripts\python.exe .\run_stage1.py simulate --scenario static3d --out results_stage1/my_static
.\.venv-stage1\Scripts\python.exe .\run_stage1.py simulate --scenario crossing3d --out results_stage1/my_crossing
.\.venv-stage1\Scripts\python.exe .\run_stage1.py simulate --scenario paper_random3d --out results_stage1/my_random
Start-Process .\results_stage1\my_random\terrain_3d.png
Start-Process .\results_stage1\my_random\trajectory_3d.png
Start-Process .\results_stage1\my_crossing\trajectory_local_3d.png
Get-Content .\results_stage1\my_crossing\summary.json
```

`empty3d` 为无障碍、`static3d` 为固定球体、`crossing3d` 为移动交叉障碍，前三种使用平海底。`paper_random3d` 为1000×1000m Perlin海底和120个PI动态障碍。默认使用CA；`--config configs/stage1_development.json` 显式加载开发配置。`paper_reference.json` 是来源记录，不能直接作为运行配置。

输出包括完整配置、`terrain.npz`、`obstacles.json`、`trajectory.csv`、`planning.csv`、`selected_plans.json`、`summary.json` 和五张诊断图。深度向下；蓝线为P25执行轨迹，橙色虚线为不同控制时刻的DWA预测片段，红虚线为不可行时的应急预测。局部图显示障碍运动及其快照时刻。

CLI正常退出只说明程序完成；到达与否应查看 `status`、`reached_goal`、`failure_reason`、最小间隙、应急减速和过期预测回退次数。非空目录默认拒绝覆盖；确实需要覆盖才加 `--overwrite`。

## 直接加载已有 LSTM 权重

无需首先重新训练150轮：

```powershell
.\.venv-stage1\Scripts\python.exe .\run_stage1.py simulate --scenario crossing3d --checkpoint artifacts_stage1/pplnm_150/pplnm.pt --calibration artifacts_stage1/pplnm_150/pplnm_calibration.json --out results_stage1/my_learned_crossing
Start-Process .\results_stage1\my_learned_crossing\trajectory_local_3d.png
```

网络使用6×9历史输入、两层256单元LSTM、三步×三坐标残差输出和恒加速度物理先验。权重来自48条合成轨迹的小规模150轮训练，不等于论文完整复现。预测间隔3s、范围9s，观测/控制周期1s；过期目标超出训练范围时显式回退CA并计数。

项目不再计算或比较文件、配置、模型或Git对象Hash。模型与校准按预测器类型、架构、输入输出维度、历史长度、采样间隔、预测步数、训练轮数及标签匹配，并检查文件存在、有限值、正尺度及严格权重维度。这些参数检查不证明文件内容相同。

现成 `.pt` 权重未改写，校准数值保留，仅迁移配置元数据。旧 `training_state.pt` 缺少新数据配置，不能直接按新规则续训；现成预测权重可以直接使用。新训练支持保存配置并续训，相关功能测试保留。

## 自动测试和完整验收

阶段2及全部原有测试：

```powershell
$env:MPLBACKEND = 'Agg'
.\.venv-stage2\Scripts\python.exe -m pytest PythonVehicleSimulator/tests tests_stage1 tests_stage2 -q
.\.venv-stage2\Scripts\python.exe -m pip check
```

历史阶段2为251项通过、14项REMUS工况13到达/9项有限时域验收通过。本轮复审后实际重跑351项全部通过：保留原147项及既有阶段2测试，新增100项复审回归，0失败、0跳过。另重新执行9/18/45预算矩阵和延迟/周期/声呐敏感性实验，任务与安全结局分别记录。完整学习测试需要上述可选PyTorch依赖；跳过不等于通过。新结果和实际命令见 [复审测试报告](docs_stage2/review_20261011/TEST_REPORT.md)，历史结果见 [原阶段2报告](docs_stage2/TEST_REPORT.md)。

推荐自动记录命令、日志、JUnit与结果；这些脚本把上游测试动画放入独立工作目录：

```powershell
$checkStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
.\.venv-stage2\Scripts\python.exe scripts_stage2/run_regression.py --out "results_stage2/regression_$checkStamp"
.\.venv-stage2\Scripts\python.exe scripts_stage2/run_acceptance.py --out "results_stage2/acceptance_$checkStamp"
.\.venv-stage2\Scripts\python.exe scripts_stage2/inspect_results.py "results_stage2/acceptance_$checkStamp" --out "results_stage2/inspection_$checkStamp.json"
.\.venv-stage2\Scripts\python.exe scripts_stage2/render_saved.py --source "results_stage2/acceptance_$checkStamp/crossing3d" --out "results_stage2/replot_$checkStamp"
```

`run_regression.py --tests-only` 只执行环境、依赖、audit与全量pytest；完整模式还执行原REMUS、响应、错配和CA/PPLNM回归。`run_acceptance.py` 分别记录程序退出、到达与安全状态，退出0不能当作14项安全验收全通过。

以下阶段1验收入口继续可用：

```powershell
$env:MPLBACKEND = 'Agg'
.\.venv-stage1\Scripts\python.exe -m pytest PythonVehicleSimulator/tests tests_stage1 -q
.\.venv-stage1\Scripts\python.exe -m pip check
# 推荐：所有测试动画、临时文件和仿真结果写到新目录
.\.venv-stage1\Scripts\python.exe .\scripts_stage1\run_acceptance.py
```

对本次已有结果单独复查数值、规划时间对齐和图片/GIF读取：

```powershell
.\.venv-stage1\Scripts\python.exe .\scripts_stage1\inspect_results.py results_stage1/acceptance_20261011_final
```

验收脚本默认创建带时间的独立目录，保存命令、工作目录、退出码、耗时、日志、JUnit及场景状态，不覆盖已有结果。执行全部原有/新增测试、audit、原REMUS含动画、200s回归、8组运行点、航点跟踪、CA和现成PPLNM各四场景。完整验收需要PyTorch；核心环境中的网络测试会跳过，不能记为完整通过。

```powershell
.\.venv-stage1\Scripts\python.exe .\run_stage1.py regression --out results_stage1/my_regression
.\.venv-stage1\Scripts\python.exe .\run_stage1.py remus-route --out results_stage1/my_route
.\.venv-stage1\Scripts\python.exe .\run_stage1.py remus-characterize --out results_stage1/my_characterize
```

## 项目结构与记录

```text
PythonVehicleSimulator/          原有唯一六自由度模型与测试
run_remus.py / run_remus.cmd     原程序及Windows入口
requirements-lock.txt           原锁文件，保持原样
auv3d/                          新增环境、预测、DWA、外部模型包装器
auv3d/adapters/remus_guidance.py 原控制器参考适配、完整状态隔离与预测
auv3d/remus_planner.py           参考空间采样与六自由度候选验证
auv3d/stage2_simulation.py       实际REMUS状态反馈执行循环
auv3d/stage2_p25.py              同场景P25运动学对照
configs/                        运行配置与参数来源
run_stage1.py                   阶段1入口
run_stage2.py                   阶段2REMUS/P25闭环与响应诊断入口
tests_stage1/                   功能和集成测试
tests_stage2/                   六自由度闭环、场景、诊断与可视化测试
scripts_stage1/                 PowerShell及自动验收
scripts_stage2/                 全量回归、场景验收、结果复查及离线重绘
docs_stage1/                    使用边界、变更、测试及后续接口
docs_stage2/                    当前架构、场景、变更、测试及后续ECP接口
artifacts_stage1/               随包权重和历史成果，保留失败日志
results_stage1/                 新生成结果，默认不提交Git
results_stage2/                 独立阶段2结果，保留真实失败与成功
```

[修改记录](docs_stage1/MERGE_CHANGELOG.md)、[实际测试报告](docs_stage1/TEST_REPORT.md)、[后续六自由度接口](docs_stage1/REMUS_INTERFACE.md)、[实现选择](docs_stage1/IMPLEMENTATION_CHOICES.md)。历史云端报告和日志仅为资料，不替代本次执行证据。

当前阶段2：[变更记录](docs_stage2/CHANGELOG.md)、[测试报告](docs_stage2/TEST_REPORT.md)、[场景说明](docs_stage2/SCENARIOS.md)、[后续ECP与执行误差接口](docs_stage2/ECP_INTERFACES.md)。

## 来源与许可

`PythonVehicleSimulator/` 为 Thor I. Fossen 的上游源码快照，保留原水动力、控制器和执行器参数。上游：https://github.com/cybergalactic/PythonVehicleSimulator ，MIT许可见 [LICENSE](PythonVehicleSimulator/LICENSE)。原 `examples/` 图片与原测试保留。

新增代码为工程重建，未实现ECP、声呐成像或实艇安全证明；`u³`积分仅为代理，`energy_joules=null`。

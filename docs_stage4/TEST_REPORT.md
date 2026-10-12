# 阶段4实际测试与实验执行报告

本报告核对本地原始JUnit、运行配置、实际结果和控制器回放记录。
最终完整回归实际**753/753通过，0 failures、0 errors、0 skipped，78.478秒**；
证据为`results_stage4/quality_publish2_20261012/junit.xml`。
2026-10-12 12:44（UTC+8）的批次状态只是快照，不能代替尚未完成的配对实验。
代码测试通过、程序正常结束、到达目标、实际1米裕量和有限时域验收分别记录。

## 自动测试的实际执行

运行环境为Windows、独立`.venv-stage1`环境，使用CPU测试和`MPLBACKEND=Agg`。
全量收集包含原`PythonVehicleSimulator/tests`和`tests_stage1`至`tests_stage4`，
没有删除旧测试或降低碰撞/安全标准。输出保存在独立`results_stage4`目录。

| 实际执行目录 | JUnit测试数 | failures | errors | skipped | JUnit秒数 | 解释 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `quality_20261012` | 696 | 0 | 106 | 0 | 41.278 | 首次运行的`--basetemp`父目录不存在，106项夹具setup报错；不能称为通过。 |
| `quality_verified_20261012` | 698 | 0 | 0 | 0 | 69.333 | 修正独立目录与绝对路径后的当时版本通过；后续仍增加测试。 |
| `quality_complete_20261012` | 1 | 0 | 1 | 0 | 0.944 | 新阶段4诊断测试与阶段2同名，pytest收集失败；此处1是收集错误记录，绝非仅运行了1项有效测试。 |
| `quality_complete2_20261012` | 741 | 0 | 0 | 0 | 67.197 | 将新增测试改名`test_stage4_diagnostics.py`，保留全部内容后通过。 |
| `quality_final_20261012` | 747 | 0 | 0 | 0 | 64.574 | 增加来源角色/身份和同状态归因校验后的全量通过。 |
| `quality_publish_20261012` | 753 | 2 | 0 | 0 | 83.174 | 加入6个实际小校准库加载测试后的首次发布回归；751通过、2项动画导出因Windows路径解析权限报错，须保留并重验。 |
| `quality_publish2_20261012` | 753 | 0 | 0 | 0 | 78.478 | 独立工作目录、测试绝对路径及所需文件系统权限下全量重新执行，全部通过；没有修改动画测试或原源码。 |

以上每个目录的`junit.xml`都是对应执行的真实结果；部分控制台配置不打印最终
数字，因此以JUnit计数核对。首次失败的详细控制台记录还保存在
`results_stage4/quality_20261012_full_tests.log`，没有清除失败执行。

最终753项全部通过版本的目录归属核对如下。原PVS 18项加阶段1 129项，共147项，
保留用户此前的147项基础回归；阶段2、3及4在其上继续增加测试。

| 测试组 | 实际收集并通过 |
| --- | ---: |
| 原PythonVehicleSimulator | 18 |
| 阶段1 | 129 |
| 阶段2 | 206 |
| 阶段3 | 203 |
| 阶段4 | 197 |
| 合计 | 753 |

`quality_publish_20261012`两项失败是原`test_3D_plot`及
`test_gif_animation_is_actually_exported_with_distinct_frames`，均在Matplotlib
动画写出前的`Path(outfile).parent.resolve(strict=True)`出现`WinError 5`。
这是执行权限问题，不是算法性能失败；没有删除两个测试或修改动画实现。
随后在独立工作目录`quality_publish2_20261012`重新执行全部753项，使用仓库
`PYTHONPATH`和测试绝对路径，在具有所需文件系统权限的执行上下文运行；旧源码、
旧测试及安全条件均保留，753项全部通过。新增6项实际库测试本身也全部通过。
独立JUnit核对保存于`test_report_review_20261012/final_junit_verified.json`。

## 其他实际验证及修复

| 检查 | 实际结果 | 原始证据 |
| --- | --- | --- |
| Python依赖 | `python -m pip check`退出0，`No broken requirements found.` | `quality_publish_20261012/pip_check.log`及独立`test_report_review_20261012/pip_check.log` |
| 原REMUS动力学回归 | 原自动驾驶执行20秒，步长0.02秒、1001个状态，全部有限；最终深度8.012034米、航向18.007489度。 | `remus_dynamics_regression_20261012/summary.json`、`states.csv`及同名控制台日志 |
| 协议和工厂预检 | 48/48工厂及起终点检查完成；16个海床场景A*几何检查、8个不同Perlin海床均通过原24秒初始化准入；不是闭环成功证明。 | `protocol_review_20261012/PREFLIGHT_REPORT.json` |
| 协议单元测试 | 69/69通过。 | `protocol_review_20261012/pytest_protocol.log`、XML |
| 协议及诊断阶段回归 | 89/89通过，属于当时版本的目标检查，不能与后续全量重复相加。 | `protocol_review_20261012/pytest_protocol_diagnostics_final.log`、XML |
| 独立风险/校准/分析/诊断检查 | 90/90通过，8.09秒；来源修复后相关回归37/37通过，4.29秒，两组存在重叠。 | `independent_review_artifacts_20261012/pytest_review.xml`及`pytest_guard_retest.xml` |
| 真实完整控制器回放 | `crossing3d_v04_s410104`的6544个状态、6543个控制区间逐项重放一致；20/40/60秒B0动作、可行性、净间隙均复现。 | `protocol_review_20261012/controller_replay_actual_case.json` |
| 候选隔离及归因检查 | 同一实际三锚点中整个vehicle字段、控制器记忆、执行器和时钟不变；故意修改归档RPM后明确拒绝同状态归因。 | `independent_review_artifacts_20261012/runtime_evidence.json` |
| 来源角色及身份错误 | 独立复制fixture中`split=test`或错误`case_id`最初可被接受，已修复；两项真实复测均拒绝，原复现保留。 | `independent_review_artifacts_20261012/metadata_guard_retest.json` |

原REMUS回归实际命令为：

```powershell
.\.venv-stage1\Scripts\python.exe run_stage1.py regression --duration 20 --out results_stage4/remus_dynamics_regression_20261012
```

此命令对应已经保存的历史输出；复现时请更换结果目录，避免覆盖。

确定的实现缺口还包括：必须用显式空场景标记区分真正无障碍与缺失标签；
校准构库/结果复用必须拒绝实际非B0来源、程序错误和共同配置不匹配；
同状态影子对照必须先证明重算B0决策与归档一致。
这些校验已加入，原B0默认行为、RPM0滑行及其危险标记保留。
B0没有新增校准项，不能把原工程包络记作零分位的覆盖成功。

## 源码保护与提交检查

独立执行正常Git内容差异检查：

```powershell
git diff --quiet origin/main -- PythonVehicleSimulator run_remus.py run_remus.cmd requirements-lock.txt auv3d/remus.py auv3d/adapters/remus_guidance.py
```

退出0：上述模型、原入口、依赖锁、已有车辆包装和控制适配文件相对本阶段基点
`origin/main`没有内容差异。使用已获取的`origin/main`，因为本地`main`仍是历史状态；
不能把过去阶段已合并的新增文件误认为本阶段改变了物理模型。
这是正常Git版本差异检查，没有执行额外文件摘要校验。
证据为`test_report_review_20261012/remote_base_source_checks.json`。

`quality_publish_20261012/prestage_audit.json`记录28个审查文件、19个Python语法检查、
无敏感模式匹配、无额外摘要校验实现、无受保护源码差异。最终暂存检查与远端提交结果
由发布阶段单独记录，不能仅凭预检查宣称已推送。

## 冻结校准的实际数据

24个B0 calibration角色场景全部完成后建立24秒CA库，包含16个有障碍场景及8个
真正无障碍海床负对照。库记录72个可用窗口、127个匹配目标窗口；离线回放核对
1498个已保存障碍快照，最大位置差0米。完整场景才是校准统计单位，不能把127个
相关窗口当作127个独立场景。

校准场景中的研究结果为19到达、2实际碰撞、3超时，失败没有被丢弃。
这些是校准角色数据，不是独立测试成绩。B1在alpha=0.1、0.2、0.3的额外球半径
分别约63.041、46.435、45.100米。大误差来自共同CA、噪声、机动和24秒全时域
场景最大值等实际因素；没有为了改善结果缩短时域、改CA或截断球半径。
详见`calibration_build_20261012/LIBRARY_STATISTICS.json`及`INTERPRETATION.md`。

小体积推断库`artifacts_stage4/ca_scene_calibration_v1.npz`约0.49MB，仅把元数据中的
本地绝对源路径改为仓库相对路径；预测、标签、掩码、半径和时间网格数值保留。
它支持独立复现B1/B2加载，不上传成批原始实验日志，也不绑定任何文件摘要。

## 正式实验批次与未完成状态

正式注册的是48个唯一场景，36开发、12独立测试；108是方法/风险水平执行次数，
不是108个独立场景，不能合成一个无法解释的总成功率。

| 正式批次 | 注册执行次数 | 数据角色与方法 | workers | 12:44快照：原始episode完整/未完 |
| --- | ---: | --- | ---: | --- |
| `risk_pilot_B0_20261012_1030` | 48 | B0：24 calibration、6 train、6 tune、12 test | 2 | 36/12 |
| `tune_risk_20261012` | 36 | 6 tune × B1/B2 × alpha 0.1/0.2/0.3 | 2 | 0/36 |
| `test_risk_20261012` | 24 | 12 test × B1/B2，alpha固定0.1 | 2 | 0/24 |
| 合计 | 108 | 不混合split、方法和alpha | 三批同时可达6个实验worker | 36/72 |

快照按`summary.json`及`selected_plans.json`存在性统计原始episode完整度；不是
统计安全通过数，也不等于批次已生成最终`STUDY_SUMMARY.json`。
快照时三批均未最终汇总，未发现`program_error.json`；未完成任务不能记成功，
零已完成的批次也不能宣称状态有限检查已通过。
完整配置及快照证据分别为各批次`EXPERIMENT_CONFIG.json`和
`test_report_review_20261012/document_audit_snapshot.json`。

额外接口冒烟`B2_interface_smoke_20261012`只有1次固定tune场景
`static3d_v04_s410004`、alpha=0.1，用于真实六自由度接口验证，**不计入正式108次，
也不额外计作独立统计样本**。实际结果是200秒超时、没有到达，最小实际净间隙
28.074米、87次应急、59次危险应急预测，有限时域验收失败；所有状态有限、
零程序异常及零候选数值预测异常。不能把运行完成或没有碰撞写成避障成功。
其P95约1943.357ms、3次超过2秒周期，属于该次并发调度下的观测。

## 计时、公平性与结论边界

`CONCURRENCY_TIMELINE_20261012.jsonl`记录B0、冒烟的开始/结束；正式tune/test
各自注册时间和worker数保存在其配置中。B0、tune、test批次同时运行，期间还有
pytest和桌面任务，因此P95可能包含CPU竞争、操作系统调度和其他任务影响。
每个方法规划候选预算共同为9，但不同批次的墙钟P95不能直接解释为风险方法的
纯计算加速或减速。正式时延比较还需要相同调度条件下的单worker配对复测。

物理仿真在同步规划期间暂停；墙钟耗时没有自动插入0.02秒执行时钟，也没有
宣称实时实现。实际超周期次数照实保留，不能用相同仿真时间掩盖计算延迟。

当前六自由度预测器为CA；B2是冻结的候选相关ECP工程适配，没有在线ACP、
新增网络或本轮PPLNM对照。0.2秒校准与0.02秒物理验证的支持点不同，16有障碍
加8空场景的混合覆盖也不是有障碍条件下的覆盖认证。不同闭环视场可能改变
传感噪声抽样顺序，同种子不等于全过程逐目标同一噪声场。

固定20/40/60秒同状态对照只说明相同真实控制器状态及交付历史下是否改变动作；
选中参考保持24秒的离线未来诊断只评价这条反事实参考，不能称为全部候选误判率
或实际闭环碰撞数。在线决定只用已交付观测，离线未来真值不反流规划。

本报告尚不回答ECP是否改善真实闭环；必须等待完整配对结果、失败分析、
同状态动作归因及覆盖标签，分别报告碰撞、裕量、限时到达、应急和保守性。
没有更改1米安全标准、原碰撞定义、5米到达半径或REMUS物理参数。

## 安全复现全量检查

以下PowerShell命令使用新的结果目录，避免覆盖上述历史执行记录：
最终`quality_publish2_20261012`运行采用同样的独立工作目录、仓库`PYTHONPATH`、
绝对测试路径和独立临时目录方式；不是在已有实验输出目录重复覆盖运行。

```powershell
Set-Location 'D:\修改代码实验\AUVWorkPlace'
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$env:CUDA_VISIBLE_DEVICES = ''
$taskRoot = (Get-Location).Path
$taskQuality = Join-Path $taskRoot ('results_stage4/quality_local_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $taskQuality | Out-Null
$env:PYTHONPATH = $taskRoot
& (Join-Path $taskRoot '.venv-stage1/Scripts/python.exe') -m pip check
Push-Location $taskQuality
try {
    & (Join-Path $taskRoot '.venv-stage1/Scripts/python.exe') -m pytest (Join-Path $taskRoot 'PythonVehicleSimulator/tests') (Join-Path $taskRoot 'tests_stage1') (Join-Path $taskRoot 'tests_stage2') (Join-Path $taskRoot 'tests_stage3') (Join-Path $taskRoot 'tests_stage4') -q --basetemp (Join-Path $taskQuality 'pytest') --junitxml (Join-Path $taskQuality 'junit.xml') -o ('cache_dir=' + (Join-Path $taskQuality 'cache'))
} finally {
    Pop-Location
}
```

每次执行检查实际退出码和JUnit，不能把示例命令当成已经执行的成功证据。

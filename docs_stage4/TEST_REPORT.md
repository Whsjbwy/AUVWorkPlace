# 阶段4实际测试与实验执行报告

本报告核对本地原始JUnit、运行配置、实际结果和控制器回放记录。
最终完整回归实际**753/753通过，0 failures、0 errors、0 skipped，78.478秒**；
证据为`results_stage4/quality_publish2_20261012/junit.xml`。
2026-10-12 12:44（UTC+8）的批次状态只是历史快照；正式108次现已全部执行完成，
零程序异常。该执行计数不等于108次安全避障成功，离线归因统计另外记录。
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

最终全量测试实际在`D:\修改代码实验\AUVWorkPlace\results_stage4\quality_publish2_20261012`
作为工作目录执行，`PYTHONPATH`指向仓库根目录，采用具有所需文件系统权限的
执行上下文。实际完整命令如下；这是已经运行的记录，后文另给新目录复现方法。

```powershell
$env:PYTHONPATH = 'D:\修改代码实验\AUVWorkPlace'
Set-Location 'D:\修改代码实验\AUVWorkPlace\results_stage4\quality_publish2_20261012'
& 'D:\修改代码实验\AUVWorkPlace\.venv-stage1\Scripts\python.exe' -m pytest 'D:\修改代码实验\AUVWorkPlace\PythonVehicleSimulator\tests' 'D:\修改代码实验\AUVWorkPlace\tests_stage1' 'D:\修改代码实验\AUVWorkPlace\tests_stage2' 'D:\修改代码实验\AUVWorkPlace\tests_stage3' 'D:\修改代码实验\AUVWorkPlace\tests_stage4' -q --basetemp 'D:\修改代码实验\AUVWorkPlace\results_stage4\quality_publish2_20261012\pytest' --junitxml 'D:\修改代码实验\AUVWorkPlace\results_stage4\quality_publish2_20261012\junit.xml' -o 'cache_dir=D:\修改代码实验\AUVWorkPlace\results_stage4\quality_publish2_20261012\cache'
```

实际退出0，JUnit记录753项、0失败、0错误、0跳过、78.478秒；
控制台保存在同目录`pytest.log`。权限上下文处理针对Windows文件系统访问，
没有改动测试内容、动画实现或安全判据。

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
| 完整45参考离线诊断 | 指定两个B0真实test状态各40秒，90条完整物理候选、270条方法评分；零预测数值异常，整个车辆/控制器/执行器状态未被修改。在线预算仍为9。 | `full_grid_offline_review_20261012/SUMMARY.json`、`METHOD_SUMMARY.csv`、`INTERPRETATION.md` |
| 海床负对照 | 正式批次完成后20/20配对数值完全相等：12 tune、8 test，6唯一case；全状态/控制/动作时刻/保存的候选轨迹/结局最大绝对差0。 | `negative_control_review_20261012.json` |

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

首次源码提交已推送至`research/stage4-risk-pilot`，并建立
[PR #3](https://github.com/Whsjbwy/AUVWorkPlace/pull/3)。该次
[push工作流](https://github.com/Whsjbwy/AUVWorkPlace/actions/runs/38205406737)及
[pull request工作流](https://github.com/Whsjbwy/AUVWorkPlace/actions/runs/38205444321)
均实际完成且结论为success；Windows与Ubuntu CPU检查各运行两次，共4项通过。
保存证据为`quality_publish2_20261012/GITHUB_CI.json`。
这是首次源码发布时点的CI，不提前代表后续最终实验文档提交的CI结果；后续提交
需要另保存对应检查证据。没有自动合并main。

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

实际构库由PowerShell here-string传入Python标准输入，调用
`auv3d.stage4_calibration.build_library`，不是运行`build-calibration` CLI。
工作目录为仓库根目录，`PYTHONUTF8=1`，实际核心调用如下：

```powershell
@'
from pathlib import Path
import json
from auv3d.stage4_calibration import build_library
r = Path('results_stage4/risk_pilot_B0_20261012_1030')
config = json.loads((r / 'EXPERIMENT_CONFIG.json').read_text(encoding='utf-8'))
rows = [dict(row, method='B0') for row in config['registered_cases']]
out = Path('results_stage4/risk_calibration_20261012.npz')
metadata = build_library(r, rows, out)
'@ | .\.venv-stage1\Scripts\python.exe -
```

同一次实际inline脚本在调用前检查24个校准场景的六类原始文件、无程序错误、
worker完成JSON中的REMUS后端/场景/种子；调用后保存
`calibration_build_20261012/BUILD_REPORT.json`。完整执行退出0，输出490514字节，
metadata包含24个完整场景。上述块记录真实核心API调用，外围检查以原BUILD_REPORT
为证据；编写本报告没有重新构库。不要在现有结果路径重新执行并覆盖历史产物。

## 正式实验批次与历史执行状态

正式注册的是48个唯一场景，36开发、12独立测试；108是方法/风险水平执行次数，
不是108个独立场景，不能合成一个无法解释的总成功率。

全部正式闭环批次现已正常结束，实际结果如下。原始碰撞、超时、危险应急与
裕量不足记录完整保留；`program_error_count=0`只表示没有阻断性程序异常。

| 正式批次 | 实际执行次数 | 程序错误 | 完成UTC时间 | 最终证据 |
| --- | ---: | ---: | --- | --- |
| `risk_pilot_B0_20261012_1030` | 48 | 0 | 2026-10-12 04:55:56 | 批次`STUDY_SUMMARY.json`、`B0_command_20261012_1030.log` |
| `tune_risk_20261012` | 36 | 0 | 2026-10-12 05:26:18 | 批次`STUDY_SUMMARY.json`、`tune_risk_command_20261012.log` |
| `test_risk_20261012` | 24 | 0 | 2026-10-12 05:12:48 | 批次`STUDY_SUMMARY.json`、`test_risk_command_20261012.log` |
| 合计 | 108 | 0 | 各批次独立完成 | 不合并方法、split或alpha计算总成功率 |

时间由`CONCURRENCY_TIMELINE_20261012.jsonl`的实际完成事件记录。
B0批次的实际参数由注册配置和完成日志核对为全48场景、2 workers、默认绘图、
不导出动画。下面按统一参数顺序记录相同已执行入口，非声称保存了逐字shell历史：

```powershell
.\.venv-stage1\Scripts\python.exe run_stage4.py run-baseline --out results_stage4/risk_pilot_B0_20261012_1030 --workers 2 2>&1 | Tee-Object -FilePath results_stage4/B0_command_20261012_1030.log
```

两组风险闭环的实际启动命令如下，均从仓库根目录执行，未使用`--no-plot`，
每个episode保留地形、航迹及交互HTML输出。两个进程均最终退出0：

```powershell
.\.venv-stage1\Scripts\python.exe run_stage4.py run-risk --methods B1 B2 --split tune --alpha .1 .2 .3 --library results_stage4/risk_calibration_20261012.npz --out results_stage4/tune_risk_20261012 --workers 2
.\.venv-stage1\Scripts\python.exe run_stage4.py run-risk --methods B1 B2 --split test --alpha .1 --library results_stage4/risk_calibration_20261012.npz --out results_stage4/test_risk_20261012 --workers 2
```

全部实验worker结束后，实际另启动单进程离线分析。线程环境
`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`及
`NUMEXPR_NUM_THREADS`均设1，其他代理暂停重数据处理。实际命令为：

```powershell
.\.venv-stage1\Scripts\python.exe run_stage4.py analyze --root results_stage4/risk_pilot_B0_20261012_1030 results_stage4/tune_risk_20261012 results_stage4/test_risk_20261012 --library results_stage4/risk_calibration_20261012.npz --baseline-root results_stage4/risk_pilot_B0_20261012_1030 --alpha .1 --anchors 20 40 60 --out results_stage4/risk_analysis_20261012 2>&1 | Tee-Object -FilePath results_stage4/risk_analysis_command_20261012.log
```

分析实际2026-10-12 05:27:31 UTC启动、05:33:19 UTC完成，进程退出0；
`risk_analysis_20261012/ANALYSIS_SUMMARY.json`记录108个episode、209个选中参考
锚点、66条同真实状态影子决策，其中38条相对B0改变。该66含各方法记录，不能
当作66个独立物理场景；209条仅是预登记20/40/60秒锚点中存在交付track的
选中参考诊断，不是全部规划时刻/全部候选。离线标签和未来真值仅用于事后评价，
不回流在线规划。完整覆盖和预测误差统计由同目录CSV保留，不以无标签记录填零。

等离线分析明确结束后，另实际执行只读海床负对照：

```powershell
.\.venv-stage1\Scripts\python.exe -m results_stage4.negative_control_review_runner_20261012
```

2026-10-12 13:33:42（UTC+8）完成，退出0，20/20严格相等；逐项数组形状及
零最大绝对差保存于`negative_control_review_20261012.json`。没有修改源码、配置
或结果，也没有使用文件摘要。海床没有观测障碍时B1/B2未影响真实状态和控制，
这是风险接口的负对照，不是风险方法提高地形导航性能的证据。

以下保留执行中的12:44（UTC+8）历史快照，不能拿它替代上表最终状态，也不能
把随后完成这一事实解释为原先未完成任务已在当时通过。

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

正式108次闭环执行已完成；ECP是否改善真实闭环仍需结合独立配对结果、失败分析、
同状态动作归因及覆盖标签，分别报告碰撞、裕量、限时到达、应急和保守性。
最终性能解释集中于`RESULTS.md`，不能从本报告的程序退出码推导安全结论。
没有更改1米安全标准、原碰撞定义、5米到达半径或REMUS物理参数。

最终交付审查还实际解析全部119个受版本管理Python文件，0语法错误；再次执行
`python -m pip check`退出0。受保护源码相对`origin/main`仍无差异，首次源码
发布后没有再改Python源码。第二次提交只补报告、小体积CSV/JSON及真实图片；
没有虚拟环境、ZIP或成批原始日志。凭据模式检查0匹配，最大新增图片530234字节。
原始审查记录为`results_stage4/quality_delivery_20261012/FINAL_AUDIT.json`。

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

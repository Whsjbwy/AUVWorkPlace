# 阶段5实际执行记录

此文件随正式实验完成更新。源码回归通过、程序正常结束、到达、实际碰撞、实际1米裕量以及方法自身预测包络下的危险应急分别记录。

| 已执行检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 原PVS及阶段1–4完整回归 | 753项通过，0失败/错误/跳过，55.365秒 | `results_stage5/quality_legacy_20261012/junit.xml` |
| 当时版本全部回归 | 889项通过，0失败/错误/跳过，77.715秒；后续增加来源与噪声等测试，最终计数另记 | `results_stage5/quality_complete_initial_20261012/junit.xml` |
| 首次源码提交完整回归 | **910项通过，0失败/错误/跳过，84.655秒**；包括原753项及新增157项 | `results_stage5/quality_source_verified_20261012/junit.xml` |
| 渲染与误差范围修正后完整回归 | **911项通过，0失败/错误/跳过，88.964秒**；包括原753项及新增158项 | `results_stage5/quality_final_source_20261012/junit.xml` |
| 历史冻结对照入口加入后完整回归 | **916项通过，0失败/错误/跳过，86.108秒**；包括原753项及新增163项 | `results_stage5/quality_with_historical_20261012/junit.xml` |
| 远端CPU CI | Python3.12的Windows及Ubuntu两项job均成功，包含依赖及原有/新增测试 | GitHub Actions run 38216111562，对应首次源码提交 |
| 小规模开发闭环 | 6个新dev场景 × A0/A2/A3/A4/A5，共30次实际REMUS执行，0程序异常；碰撞和超时保留，不是全安全通过 | `results_stage5/pilot_development_20261012/` |
| 原REMUS自动驾驶20秒回归 | 1001状态全部有限，最终深度8.012034米、航向18.007489度；仅为原响应回归，不宣称20秒已经稳态 | `results_stage5/remus_regression_20261012/` |
| 完整empty3d闭环 | 84.92秒到达，实际最小净间隙29.175962米，无应急，预测/执行位置误差0 | `results_stage5/empty_regression_20261012/` |
| 物理延迟与旧执行循环针对性回归 | 25项通过；计时完成前继续旧指令，障碍与观测继续，无worker重叠 | `tests_stage5/test_physical_latency.py`及原阶段2测试，首轮控制台记录 |
| 协议/校准目标回归 | 41项通过，0.81秒 | `results_stage5/protocol_calibration_verified_20261012/` |
| 原始来源与shadow独立审查 | 三项确定问题修复后均复验通过，完整vehicle数值状态不变 | `results_stage5/source_review_20261012/RETEST_EVIDENCE.json` |
| 预测噪声DEV诊断 | 18份来源、6独立几何、54预测条件，0程序错误；不是闭环或未见test结论 | `results_stage5/prediction_noise_pilot_dev_20261012/` |
| 依赖检查 | `No broken requirements found.` | `results_stage5/pip_check_20261012.log` |
| 原模型内容差异 | 对原PVS、REMUS、控制适配器及原入口执行普通Git差异检查，输出为空 | 原物理文件没有修改 |

初次新测试执行124项中118通过、6项新测试错误使用`result.summary`，改为实际字典接口后保留测试继续复验。图标题格式调整后一轮7项中2项因字符串拼接遗漏失败，修复后7/7通过。原失败XML和控制台日志均在`results_stage5`保留，没有删除失败测试或放宽安全标准。

正式构库预检：96个登记几何中95个通过原场景工厂。唯一失败为海床calibration种子525006，没有按原准入规则找到180米路线，原ID和错误保留、没有换种子。风险库统一只使用四个有障碍家族的48个完整calibration场景，所有24个无障碍海床calibration场景均不参与障碍分位数，避免按成败挑选。所有formal dev/tune/test工厂检查均通过。

实际最新完整回归命令由仓库独立结果目录执行，设置仓库PYTHONPATH和Agg后：

```powershell
& 'D:\修改代码实验\AUVWorkPlace\.venv-stage1\Scripts\python.exe' -m pytest 'D:\修改代码实验\AUVWorkPlace\PythonVehicleSimulator\tests' 'D:\修改代码实验\AUVWorkPlace\tests_stage1' 'D:\修改代码实验\AUVWorkPlace\tests_stage2' 'D:\修改代码实验\AUVWorkPlace\tests_stage3' 'D:\修改代码实验\AUVWorkPlace\tests_stage4' 'D:\修改代码实验\AUVWorkPlace\tests_stage5' -q --basetemp 'D:\修改代码实验\AUVWorkPlace\results_stage5\quality_with_historical_20261012\pytest' --junitxml 'D:\修改代码实验\AUVWorkPlace\results_stage5\quality_with_historical_20261012\junit.xml' -o 'cache_dir=D:\修改代码实验\AUVWorkPlace\results_stage5\quality_with_historical_20261012\cache'
```

每次复现使用新的结果目录，避免覆盖。长批次多worker的墙钟规划时间只能描述该负载；最终配对计算计时须单worker、统一线程并避开其他实验。同步none模式的成功不能当作实时控制能力。

日志渲染针对性测试首轮10项中2项setup错误来自pytest临时目录的父目录未创建；保留错误XML，先创建独立父目录后原测试10/10通过（3.130秒），未删除或放宽断言。

## 最终验证与实验清单

最终源码（含实际应用日志修正）全部**916项通过，0失败、错误、跳过，59.338秒**，证据`quality_command_log_corrected_20261012/junit.xml`及`pytest.log`。原753项全部保留，新增163项；反复运行不叠加为不同测试数。此前图例修正后的916项85.403秒也保留于`quality_final_delivery_20261012`。

| 实际批次/命令 | 新闭环执行次数 | 真实结果与目录 |
|---|---:|---|
| `run_stage5.py run-pilot --split dev --methods A0 A2 A3 A4 A5 --combination replace`，pilot库/噪声.02 | 30 | 0程序错误；A0碰撞及其他超时保留，`pilot_development_20261012` |
| `run_stage5.py tune-margin --phase formal --noise .02 --combination add replace --margins 0 .75 3 6 --workers 2` | 48 | 6完整独立tune几何×4阈值×2组合，0程序错误，`formal_A1_tune_20261012` |
| `run_stage5.py freeze-margin --root results_stage5/formal_A1_tune_20261012 --out results_stage5/formal_A1_frozen_20261012.json` | 0 | 仅tune数据选择3米；原始48项和完整选择曲线保留 |
| `run_stage5.py run-study --phase formal --split test --methods A0 A2 A3 A4 A5 --alpha .1 --noise .02 --combination replace --workers 4 --no-plot`，formal匹配库 | 60 | 0程序错误，`formal_primary_core_20261012` |
| 同上`--methods A1 --tuned-margin results_stage5/formal_A1_frozen_20261012.json` | 12 | 0程序错误，`formal_primary_A1_20261012`；主比较合计72 |
| 同4障碍g14的`--methods A1 A2 A3 A4 A5 --combination add --workers 4 --no-plot` | 20 | 0程序错误，`formal_add_ablation_20261012` |
| CV补充8、CA噪声0/.1补充4、filtered_cv补充8，逐case登记、各自独立校准库 | 20 | 3批全部exit0，`formal_predictor_{cv,ca_noise,filtered_cv}_20261012`；另复用主批4项，24行不是24次新执行 |
| `python -m scripts_stage5.historical_frozen --out results_stage5/formal_historical_frozen_20261012 --workers 2` | 4 | exit0；2静态超时/2交叉安全到达，旧库/原add协议独立报告 |
| `run_stage5.py run-study --phase formal --case s5_formal_multi_crossing3d_g14_s523014_n020 --methods A0 A1 A2 A3 A4 A5 --noise .02 --alpha .1 --combination replace --delay-mode measured --workers 1 --no-plot`，formal库+A1冻结 | 6 | exit0，六法均安全到达；实际延迟/失败预测分开，`formal_measured_delay_20261012` |
| `run_stage5.py analyze --root results_stage5/formal_primary_core_20261012 results_stage5/formal_primary_A1_20261012 --anchors 20 40 60 --shadow` | 0 | exit0；72原执行、141选中参考锚点、138同状态决策，`formal_primary_analysis_20261012` |
| `python results_stage5/run_supplemental_analysis_20261012.py` | 0 | exit0；严格筛选44记录/131锚点，不重跑mission，`supplemental_analysis_20261012` |
| `python -m scripts_stage5.prediction_noise --phase formal --split test --registry results_stage5/data_formal_obstacle_20261012/ARTIFACT_REGISTRY.json --out results_stage5/prediction_noise_formal_test_20261012`；另执行pilot/dev对应命令 | 0 | DEV18份来源/54预测条件，formal test36份来源/108预测条件，均0错误；原始历史、未来离线标签、分时域统计保留 |

合计200次登记风险闭环（30+48+60+12+20+20+4+6），不是200个独立几何。命令表省略重复`--out`/`--artifacts`等长路径；每批`EXPERIMENT_CONFIG.json`、登记manifest、命令log、独占ENV中的完整argv保留实际参数，精确复现使用新的输出目录。

原REMUS20秒及完整empty3d回归另列在前表，不计风险对照分母。日志修正后另执行真实2.4秒crossing短诊断，application为[0.5,1.5,null]、ready为[0.5,1.5,2.5]，实际应用事件[0.5,1.5]完全吻合，`command_log_live_diagnostic_20261012/LOG_SCHEMA_CHECK.json`；不是正式任务成功。

独立主72审查和补充46审查覆盖118个不同正式闭环记录；所有状态有限、物理时钟/障碍同步、历史因果、配对场景一致、候选不改变真实控制器。实测延迟48,501个控制区间没有提前执行命令，483次实际应用、5次终端pending。旧原始selected_plans的application字段预填ready时间是已发现日志语义问题，保留旧文件，以实际events核对；新源码分离ready与真实application，未应用为null，原测例加强而非删除。

三维正式地形、静态失败、动态多交叉、独立预测、执行误差曲线、GIF和交互HTML均从保存的真实数据导出，来源JSON记录，没有再运行或拼接理想P25航迹。首次沙箱GIF导出WinError5原错误保留；正常本地权限重试到新目录成功，未覆盖原文件。

代码质量通过不表示研究性能全通过。主比较各方法实际到达、安全到达、碰撞、1米违规、超时、无候选及自身危险应急见[正式结果](RESULTS.md)。没有修改碰撞或安全/到达标准，没有额外文件摘要校验。

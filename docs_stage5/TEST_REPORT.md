# 阶段5实际执行记录

此文件随正式实验完成更新。源码回归通过、程序正常结束、到达、实际碰撞、实际1米裕量以及方法自身预测包络下的危险应急分别记录。

| 已执行检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 原PVS及阶段1–4完整回归 | 753项通过，0失败/错误/跳过，55.365秒 | `results_stage5/quality_legacy_20261012/junit.xml` |
| 当时版本全部回归 | 889项通过，0失败/错误/跳过，77.715秒；后续增加来源与噪声等测试，最终计数另记 | `results_stage5/quality_complete_initial_20261012/junit.xml` |
| 当前源码完整回归 | **910项通过，0失败/错误/跳过，84.655秒**；包括原753项及新增157项 | `results_stage5/quality_source_verified_20261012/junit.xml` |
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
& 'D:\修改代码实验\AUVWorkPlace\.venv-stage1\Scripts\python.exe' -m pytest 'D:\修改代码实验\AUVWorkPlace\PythonVehicleSimulator\tests' 'D:\修改代码实验\AUVWorkPlace\tests_stage1' 'D:\修改代码实验\AUVWorkPlace\tests_stage2' 'D:\修改代码实验\AUVWorkPlace\tests_stage3' 'D:\修改代码实验\AUVWorkPlace\tests_stage4' 'D:\修改代码实验\AUVWorkPlace\tests_stage5' -q --basetemp 'D:\修改代码实验\AUVWorkPlace\results_stage5\quality_source_verified_20261012\pytest' --junitxml 'D:\修改代码实验\AUVWorkPlace\results_stage5\quality_source_verified_20261012\junit.xml' -o 'cache_dir=D:\修改代码实验\AUVWorkPlace\results_stage5\quality_source_verified_20261012\cache'
```

每次复现使用新的结果目录，避免覆盖。长批次多worker的墙钟规划时间只能描述该负载；最终配对计算计时须单worker、统一线程并避开其他实验。同步none模式的成功不能当作实时控制能力。

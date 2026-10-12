# 阶段5代码变更

本轮在最新main上建立`research/stage5-risk-validation`，保护阶段4代码、校准库、108次历史实验及失败记录。原REMUS动力学、控制器、执行器、碰撞定义、实际1米净间隙与5米到达标准均未改变。

| 文件 | 变化及原因 |
| --- | --- |
| `auv3d/stage5_risk.py` | A0–A5独立风险策略；区分物理半径、实际安全标准及预测裕量；add/replace独立因子；逐时刻与全时域校准明确区分。 |
| `auv3d/stage5_forecasting.py` | 原CA保持不变，新增因果CV及固定6点速度拟合CV；各预测器使用独立实际参数合同与校准库。 |
| `auv3d/stage5_protocol.py` | 新的整场景/种子划分、预登记共同布局变体及来源身份检查；旧test不再作为新未见测试。 |
| `auv3d/stage5_calibration.py` | 同声呐规则下固定因果观测载体获取历史，未来标签仅离线；按预测器、噪声、场景及速度组独立校准，不足样本拒绝而不截断分位。 |
| `auv3d/stage5_experiments.py` | 统一配对实验、独立A1扫描及冻结文件、严格来源角色核对、实际安全到达与无候选比例等指标；完整保留研究失败。 |
| `auv3d/stage5_analysis.py` | 原控制器逐步重放、独立未来标签、逐时刻/全时域覆盖、轮转顺序的同状态计时及完整控制器隔离检查。 |
| `auv3d/stage5_plots.py` | 明确显示海床、真实六自由度航迹、独立预测及执行误差；早停后不伪造继续航行或缺失预测。 |
| `auv3d/stage2_simulation.py` | 可选注入预测器和共同场景；新增none/fixed/measured规划延迟。延迟期间原模型、障碍及感知继续推进，命令按就绪时刻生效；原默认行为数值保留。 |
| `auv3d/stage4_diagnostics.py` | 仅新增可选scene/forecaster参数，以准确重放新几何及预测器；默认旧调用不变。 |
| `run_stage5.py`、`scripts_stage5/` | PowerShell可用的构库、调参、闭环、独立噪声诊断和分析入口。 |
| `configs/stage5_formal_study.json` | 在正式test前冻结主比较、A1选择规则、add消融、预测器子集、同状态计时及延迟实验。 |
| `tests_stage5/` | 风险几何、时间对齐、来源防错、延迟物理执行、控制器隔离、真实短闭环、图像与日志回归。所有旧测试保留。 |
| README、AGENTS、CPU CI、`docs_stage5/` | 更新入口、真实边界和可复制流程，CI包含阶段5。 |

确定的程序修复包括：A1 pilot调参文件误用于formal、CA原始数据误登记为CV、直接deepcopy含GNC模块的真实REMUS导致shadow异常；均有修复前复现和修复后拒绝/通路证据。开发中的测试字典访问和图标题字符串拼接错误也已修正，首次失败日志保留。

没有增加ECP在线ACP、PPLNM训练、新网络、应急控制策略或电能节省模型。A5是待验证工程适配；filtered CV只平滑速度估计。校准载体不是REMUS航迹，存在感知/ego协变量漂移，不能声称任务概率安全保证。

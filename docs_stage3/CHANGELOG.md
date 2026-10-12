# 阶段3修改记录

- `stage3_memory.py`：因果记忆、硬过期对照、保留并膨胀、声明速度界下条件删除与诊断。
- `stage3_candidates.py`、`remus_planner.py`：保留原45格，独立扩展/分层筛选；原完整六自由度验证备份；同状态全候选诊断。
- `stage3_terrain.py`、`scenarios.py`：普通已知海床粗网格三维A*；独立固定规则Perlin受限路线。原场景几何保留。
- `stage3_forecasting.py`、`remus_planner.py`：可插拔绝对时间预测协议、CA默认、显式围球分量、逐规划缓存。
- `stage2_config.py`、`stage2_simulation.py`：仅新增可选实验配置、实际状态子目标、记忆/备份/预测日志；原标准与默认策略保留。
- `run_stage3.py`、`scripts_stage3/`、`configs/stage3/`：阶段3入口和固定参数实验，不覆盖已有结果。
- `tests_stage3/`、CI：新增功能测试并同时执行原全部测试。
- README、AGENTS、`docs_stage3/`：使用方式、能力边界、原始证据与研究分支流程。

未修改原`PythonVehicleSimulator/`水动力学、控制器、执行器，未删除旧测试或实验，未引入ECP或重新训练模型。

独立PPLNM提交新增`stage3_pplnm.py`、配置和CLI切换：复用9秒已有权重与参数校验，明确CV/CA尾部或越界拒绝；暴露原P25经验球有效掩码，初步导航对照保持原共同工程包络。`scripts_stage3/analyze_prediction_errors.py`验证交付历史后进行同锚点离线误差评价，不把未来真值交给规划器。三维图标题补充预测器、9秒后延拓、记忆与地形方案。

本轮确定修复与最终交付：

- `stage3_forecasting.py`：负半径逐项验证、对象ID/请求时刻严格匹配，避免错误包络造成虚假可行。
- `stage2_simulation.py`：两项安全标志仅转原生bool修复JSON保存；补预测版本/历史dt日志。碰撞与安全数值未变。
- `stage3_pplnm.py`：经验球速度分箱使用原校准数据的dt3重采样特征，工程导航包络仍保持原公式；另独立提交便于区分PPLNM贡献。
- `stage3_terrain.py`、`scenarios.py`：独立巡航准入Perlin场景保留旧初始化失败；修复越过转角后局部子目标进度锁住，仅相邻段且须原扫掠安全检查，未调6米/25米引导参数或5米到达标准。
- `scripts_stage3/run_experiments.py`：先注册完整矩阵再执行，按配置分组保存程序错误/到达/实际安全/预测风险；共同物理dt从完整注册表派生，混合值不虚报。
- `scripts_stage3/candidate_diagnostics.py`：固定首次不可行帧隔离保存并在导航后同状态全格审查，离线开销不计在线P95。
- `scripts_stage3/build_delivery.py`：新目录保留全原始证据及失败，挂接明确来源的共同锚点RMSE；完整日志证明无目标才标未定义。普通ZIP压缩，不进行额外文件摘要检查。
- `tests_stage3/`：模型fixture路径支持仓库外cwd，新增校准、进度、注册、采样诊断、零样本和交付证据保护。最终原全部+新增556项通过。
- `docs_stage3/experiments/`、`examples/`：小体积39配置结果、同锚点误差与实际六自由度示例；批量原始日志、动画、虚拟环境和ZIP留本地。旧实验/旧代码测试未删除。

确认未修改原`PythonVehicleSimulator/`、`run_remus.py`、`run_remus.cmd`和`requirements-lock.txt`。无ECP、无24秒PPLNM新训练或校准、无能耗节省声明。

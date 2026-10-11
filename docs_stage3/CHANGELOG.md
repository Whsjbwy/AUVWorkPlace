# 阶段3修改记录

- `stage3_memory.py`：因果记忆、硬过期对照、保留并膨胀、声明速度界下条件删除与诊断。
- `stage3_candidates.py`、`remus_planner.py`：保留原45格，独立扩展/分层筛选；原完整六自由度验证备份；同状态全候选诊断。
- `stage3_terrain.py`、`scenarios.py`：普通已知海床粗网格三维A*；独立固定规则Perlin受限路线。原场景几何保留。
- `stage3_forecasting.py`、`remus_planner.py`：可插拔绝对时间预测协议、CA默认、显式围球分量、逐规划缓存。
- `stage2_config.py`、`stage2_simulation.py`：仅新增可选实验配置、实际状态子目标、记忆/备份/预测日志；原标准与默认策略保留。
- `run_stage3.py`、`scripts_stage3/`、`configs/stage3/`：阶段3入口和固定参数实验，不覆盖已有结果。
- `tests_stage3/`、CI：新增功能测试并同时执行原全部测试。
- README、AGENTS、`docs_stage3/`：使用方式、能力边界、原始证据与研究分支流程。

未修改原`PythonVehicleSimulator/`水动力学、控制器、执行器，未删除旧测试或实验，未引入ECP或重新训练模型。PPLNM六自由度接入将在独立提交中记录。

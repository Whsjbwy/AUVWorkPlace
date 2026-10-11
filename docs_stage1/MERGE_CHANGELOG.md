# 阶段1合并修改记录（2026-10-11）

合并基础为从用户指定 `Whsjbwy/AUVWorkPlace` 下载的原仓库，当前工作目录此前为空。阶段1ZIP在主仓库旁暂存，逐项确认根目录没有冲突后加入；没有覆盖已有同名实验目录，没有推送或提交远端。

## 原仓库

- 保持 `PythonVehicleSimulator/` 全部模型、控制器、执行器、测试和许可原样；质量、惯性、阻尼、舵角、RPM和时间常数没有修改。
- 保持 `run_remus.cmd`、`requirements-lock.txt`、`examples/` 原样。使用普通Git差异检查确认这些路径没有变更，未执行对象摘要核验。
- `run_remus.py` 仅新增 `--out`，并将有限值检查显式限定为数值参数；原默认输出、仿真调用、数学计算、绘图和动画流程保持。
- README更新为统一入口说明，`.gitignore`新增独立环境与结果目录。

## 从阶段1新增

`auv3d/` 全部环境、地形、几何、障碍、声呐、数据、预测、网络、训练、规划、REMUS包装器、仿真及可视化模块；`configs/`、`tests_stage1/`、`docs_stage1/`、`scripts_stage1/`、`run_stage1.py`、两份阶段1依赖、入口说明。

完整保留 `artifacts_stage1/` 的训练权重、数据、训练状态、成功/失败实验、图片、日志；根目录旧 `validation_manifest.json`移到 `artifacts_stage1/cloud_validation_manifest.json`，注明历史身份。旧结果中的摘要字段只作历史资料，不被当前代码读取执行。

## 阶段1代码修改

| 文件 | 修改与目的 |
|---|---|
| `auv3d/remus.py` | 去除源码和Git对象身份验证及fallback；正常导入唯一原模型；保留文件存在、模块来源、接口、维度、参数、矩阵与数值检查 |
| `auv3d/prediction.py` | 去除模型文件摘要；严格加载权重、归一化维度/尺度/有限值和架构/时间/预测范围检查；校准按实际参数匹配 |
| `auv3d/planner.py` | 去除checkpoint绑定；预测范围用于显式过期回退；碰撞与安全阈值原样保留 |
| `auv3d/train.py` | 去除数据/模型摘要；恢复训练改为可读的数据形状、类型、采样、划分、生成配置、归一化与训练参数匹配；新增非有限损失/梯度防护 |
| `auv3d/simulation.py` | 去除配置摘要；保存每次选中规划的时刻、动作、预测状态、可行性；输出明确执行模型和失败原因 |
| `auv3d/visualization.py` | 新增地形图，局部显示海底、起终点、障碍；固定NED深度轴；分别标注P25执行、预测与应急轨迹 |
| `requirements-stage1-core.txt` | 加入原模型可编辑安装，使用独立环境，不覆盖原锁文件 |
| `scripts_stage1/run_acceptance.py` | 新增完整验收，记录命令、日志、退出码、JUnit、耗时及场景状态；结果目录不可覆盖 |
| `scripts_stage1/inspect_results.py` | 数值核对原入口与包装器的物理状态、规划时间对齐、轨迹有限值、到达条件及图片/GIF实际解码 |
| `scripts_stage1/verify_stage1.*` | 委托完整验收入口，避免默认覆盖旧输出 |

保留三个原测试模块；替换两项源码身份检查、checkpoint绑定测试为缺失文件/正常导入/接口和模型参数检查；扩充非法维度、数值、NED、完整角速率、现成权重预测、恢复训练等测试。新增 `test_episode_records.py` 和 `test_visualization.py`，验证时序记录和深度显示。未删除失败测试、未放宽碰撞判定。

迁移三个校准文件的元数据：`pplnm_150/pplnm_calibration.json`、`pplnm_smoke/pplnm_calibration.json`、`pplnm_smoke/ca_calibration.json`。原权重和校准分位数没有改写。

## 不再使用

ZIP中的 `third_party/pvs_minimal/` 和文件摘要清单没有引入主仓库。原来源身份、checkpoint—calibration摘要绑定、配置/训练数据摘要均不再执行。

旧 `training_state.pt` 不含新训练配置，直接续训会明确报错；不影响随包 `pplnm.pt` 预测。需新增训练时用独立输出目录建立新状态。实际配置匹配不保证所有样本内容一致。

## 交付结果

本次所有验收和失败复现日志位于 `results_stage1/acceptance_20261011*`，详见 [TEST_REPORT.md](TEST_REPORT.md)。完整六自由度动态避障仍需 [REMUS_INTERFACE.md](REMUS_INTERFACE.md) 中的控制、预测与执行接口，未将P25轨迹冒充六自由度执行。

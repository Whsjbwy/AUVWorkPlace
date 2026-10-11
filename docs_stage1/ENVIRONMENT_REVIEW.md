# 三维环境与可视化接口审查

本文件记录合并时对三维环境的代码审查与可视化专项检查。完整仿真执行结果见同目录的测试报告及独立结果目录；代码审查不代替实际仿真验收。

## 坐标与数据含义

- 位置使用 NED：`[North, East, Depth]`，单位 m，深度向下为正。海面深度为 0，海底深度为正。
- P25 状态为 `[N, E, D, u, pitch, yaw]`；角度为 rad，命令为 `[u, q, r]`，速度为 m/s，角速度为 rad/s。
- `kinematic_derivative` 中 `D_dot = -u * sin(pitch)`，因此正俯仰角使深度减小。该模型没有横荡、垂荡或滚转动力学。
- 简化模型中 `pitch_dot = q`、`yaw_dot = r/cos(pitch)`；`r` 不是任意俯仰下的欧拉航向角变化率。
- `terrain.depths` 数组按 `[east_index, north_index]` 排列，与 `meshgrid(terrain.xs, terrain.ys)` 一致。
- 三维图显式设置 `zlim=(deepest, 0)`；二维深度图反转纵轴。局部图包含插值后的实际海底深度，避免只显示航迹而遗漏地形。

## 图像输出

`plot_episode` 保存五张图：

| 文件 | 内容 |
| --- | --- |
| `terrain_3d.png` | 完整地形、初始障碍分布、起点和目标点 |
| `trajectory_3d.png` | 完整环境与 P25 执行航迹、稀疏的 DWA 预测轨迹快照 |
| `trajectory_local_3d.png` | 航线附近地形、起终点、最终位置、执行航迹、预测轨迹、最接近时刻的障碍位置 |
| `depth.png` | P25 深度随时间变化，深度向下 |
| `speed.png` | P25 运动学速度状态，不能解释为 REMUS 推进器的动态响应 |

蓝色实线为 `P25 executed trajectory`，来自 `trajectory.csv` 所对应的逐积分步状态。橙色虚线为 `P25 planned rollout snapshots`，来自 `selected_plans` 的独立短时预测；最多展示五个控制时刻，不把不同未来轨迹拼接成一条实际航迹。红色虚线表示选中的应急退化预测。图标题明确声明当前是 P25 运动学，而不是 REMUS 六自由度避障。

较多障碍物使用中心散点显示空间分布，图例说明标记尺寸仅表示半径大小，不是严格按图中物理比例绘制的球体；局部少量障碍物按真实半径绘制球面线框。局部图的障碍位置选取所有记录中距离 AUV 最近的时刻，时间写在图例中。点线是记录的障碍物真实运动，仅用于离线诊断，不能解释为规划器收到的未来真值。

没有 `selected_plans` 的旧结果会明确显示没有保存预测轨迹，不凭空生成第二条轨迹。所有图同时显示场景名与真实终止状态，超时与碰撞不会改名为成功。

## 环境代码审查

- `HeightField.clearance` 同时检查海底、海面和水平场景边界；坡度地形采用保守球体安全距离。`segment_clearance` 额外检查步间线段及间隔误差界，没有为得到成功航迹放宽碰撞条件。
- `swept_sphere_clearance` 使用同一时间区间的 AUV 与障碍物相对线段，避免仅比较离散端点而漏掉交叉碰撞。
- `run_episode` 先记录当前状态和当前障碍物快照，然后同步推进二者，再检查该时间区间碰撞。图中障碍物最近接近时刻通过轨迹时间插值匹配，未用索引假设两种日志有相同采样频率。
- 声呐将 NED 目标相对位置旋转到机体系，根据三维范围/FOV/球体角度范围选择观测。地形遮挡是几何可见性近似，不是声学传感器仿真。
- 随机障碍物的 PI 参数属于障碍物模型；此审查与可视化修改没有改动 REMUS 质量、惯性、阻尼、执行器或控制器。

## 专项检查记录

新增 `tests_stage1/test_visualization.py`：

1. 实际导出五张 PNG，检查海面为 0、深度轴反向、海底可见，并检查显示的预测与执行曲线确实读取各自不同的数据。
2. 对旧结果不虚构预测轨迹。
3. 在绘图前拒绝非有限轨迹，避免生成看似正常的错误图。

初次尝试 `.venv-stage1\Scripts\python.exe -m pytest tests_stage1/test_visualization.py -q` 时新环境尚无 pytest，明确失败：`No module named pytest`。随后使用已有 `F:\Anaconda3\python.exe -m pytest tests_stage1/test_visualization.py -q` 实际执行，结果 **3 passed in 2.44s**。此次环境为 Python 3.13.5、NumPy 2.1.3、Matplotlib 3.10.0；Python 3.12 独立环境的最终复测由完整测试报告记录。

## 正式输出视觉复核（2026-10-11）

逐张打开并检查 `results_stage1/acceptance_20261011/` 中以下真实仿真输出：

- `ca_crossing3d/trajectory_local_3d.png`：起终点、最终位置、蓝色执行曲线、橙色预测片段、t=39s 障碍球体及真实运动点线可见；海面 0m 在上，平坦海底 200m 在下。
- `ca_paper_random3d/terrain_3d.png`：完整起伏海底、120 个初始障碍物中心分布及起终点可见，深度轴方向正确。
- `pplnm_crossing3d/trajectory_local_3d.png`：t=40s 障碍位置与三维绕行曲线清楚，预测与执行图例分开。
- `pplnm_paper_random3d/trajectory_local_3d.png`：局部真实海底、附近障碍球体、起终点、最终位置与航迹均可见，AUV 位于海底上方。

以上四张图均在标题明确标注 P25 运动学且不是 REMUS 六自由度避障，其对应 summary 的终止状态均为 `success`。两次 PPLNM 场景分别记录 3 次和 6 次陈旧轨迹回退，这些回退仍保留于日志，不能将结果解读为每个时刻均由 LSTM 给出有效预测。图像用于工程验收和离线诊断，不能作为论文精确复现或六自由度避障成功的证据。

## 尚未实现的六自由度闭环接口

当前执行曲线仍由 `geometry.kinematic_step` 产生，不能作为 REMUS 的实际六自由度运动轨迹。完整闭环需要外部控制适配层，将 DWA 速度/机体角速度参考转换为原模型支持的深度/航向/推力或舵角/RPM，并由 REMUS 真实状态反馈给下一次规划。还需要基于六自由度响应评估候选命令的可达性、执行器饱和和迟滞、海流相对速度、三维碰撞及模型预测误差。适配时应分别保存 `planned_states` 与 `remus_executed_states`，并在日志及图例中保留二者来源。

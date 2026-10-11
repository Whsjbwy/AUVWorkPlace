# Windows / Python 3.12 实际测试报告（2026-10-11）

公开版本仅将历史命令的本机工作目录前缀替换为 `<repository>`；实际命令和数值原日志仍完整保存在独立实验交付中。
本报告来自实际执行的命令、JUnit与仿真文件，不引用随ZIP云端结果作为本次通过证据。

工作目录：`<repository>`。环境：独立`.venv-stage1`，Python3.12.14、Windows、NumPy2.3.5、Matplotlib3.10.8、PyTorch2.10.0 CPU、pytest9.0.2。基础仿真与现成模型预测均未使用GPU。

## 结果

**18项原项目测试 + 129项阶段1测试 = 147 passed，0 failed，0 errors，0 skipped。** 原REMUS 200s含动画完成，audit/回归/8组运行点/航点跟踪均通过；CA与现成150轮PPLNM各四类场景均到达。

完整命令、工作目录、退出码、耗时、输出摘要：[acceptance_report.json](../results_stage1/acceptance_20261011_final/acceptance_report.json)。JUnit：[pytest.xml](../results_stage1/acceptance_20261011_final/pytest.xml)。

正式验收命令：

```powershell
.\.venv-stage1\Scripts\python.exe scripts_stage1/run_acceptance.py --out results_stage1/acceptance_20261011_final
```

该轮完成16个命令项。随后实际执行下述输出检查，退出码0；此项现已接入验收脚本末尾，未来运行会自动记录为第17项。

```powershell
.\.venv-stage1\Scripts\python.exe scripts_stage1/inspect_results.py results_stage1/acceptance_20261011_final
```

## 逐项执行记录

以下命令为日志中实际参数；所有退出码均为0。完整路径保留以供核查。pytest工作目录为结果目录下`pytest_work`，避免上游动画写入原示例。

| 项目 | 结果 | 耗时s | 日志 |
|---|---|---:|---|
| versions | PASS | 0.73 | [versions.log](../results_stage1/acceptance_20261011_final/versions.log) |
| pip_check | PASS | 0.87 | [pip_check.log](../results_stage1/acceptance_20261011_final/pip_check.log) |
| audit | PASS | 0.60 | [audit.log](../results_stage1/acceptance_20261011_final/audit.log) |
| pytest | 147通过 / 0跳过 | 23.30 | [pytest.log](../results_stage1/acceptance_20261011_final/pytest.log) |
| original_remus | PASS | 13.33 | [original_remus.log](../results_stage1/acceptance_20261011_final/original_remus.log) |
| regression | PASS | 1.42 | [regression.log](../results_stage1/acceptance_20261011_final/regression.log) |
| remus_characterize | PASS | 4.12 | [remus_characterize.log](../results_stage1/acceptance_20261011_final/remus_characterize.log) |
| remus_route | success，3/3航点 | 2.42 | [remus_route.log](../results_stage1/acceptance_20261011_final/remus_route.log) |
| ca_empty3d | success，到达目标 | 1.92 | [ca_empty3d.log](../results_stage1/acceptance_20261011_final/ca_empty3d.log) |
| ca_static3d | success，到达目标 | 2.00 | [ca_static3d.log](../results_stage1/acceptance_20261011_final/ca_static3d.log) |
| ca_crossing3d | success，到达目标 | 2.02 | [ca_crossing3d.log](../results_stage1/acceptance_20261011_final/ca_crossing3d.log) |
| ca_paper_random3d | success，到达目标 | 3.32 | [ca_paper_random3d.log](../results_stage1/acceptance_20261011_final/ca_paper_random3d.log) |
| pplnm_empty3d | success，到达目标 | 3.32 | [pplnm_empty3d.log](../results_stage1/acceptance_20261011_final/pplnm_empty3d.log) |
| pplnm_static3d | success，到达目标 | 3.37 | [pplnm_static3d.log](../results_stage1/acceptance_20261011_final/pplnm_static3d.log) |
| pplnm_crossing3d | success，到达目标 | 3.21 | [pplnm_crossing3d.log](../results_stage1/acceptance_20261011_final/pplnm_crossing3d.log) |
| pplnm_paper_random3d | success，到达目标 | 4.83 | [pplnm_paper_random3d.log](../results_stage1/acceptance_20261011_final/pplnm_paper_random3d.log) |

每项的实际调用：

**versions**

```powershell
<repository>\.venv-stage1\Scripts\python.exe -m pip list --format=json
```

**pip_check**

```powershell
<repository>\.venv-stage1\Scripts\python.exe -m pip check
```

**audit**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py audit --out <repository>\results_stage1\acceptance_20261011_final\audit.json
```

**pytest**

```powershell
<repository>\.venv-stage1\Scripts\python.exe -m pytest <repository>\PythonVehicleSimulator\tests <repository>\tests_stage1 -q --junitxml <repository>\results_stage1\acceptance_20261011_final\pytest.xml --basetemp <repository>\results_stage1\acceptance_20261011_final\pytest_temp
```

**original_remus**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_remus.py --duration 200 --out <repository>\results_stage1\acceptance_20261011_final\original_remus
```

**regression**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py regression --out <repository>\results_stage1\acceptance_20261011_final\regression
```

**remus_characterize**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py remus-characterize --out <repository>\results_stage1\acceptance_20261011_final\remus_characterize
```

**remus_route**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py remus-route --out <repository>\results_stage1\acceptance_20261011_final\remus_route
```

**ca_empty3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario empty3d --out <repository>\results_stage1\acceptance_20261011_final\ca_empty3d
```

**ca_static3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario static3d --out <repository>\results_stage1\acceptance_20261011_final\ca_static3d
```

**ca_crossing3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario crossing3d --out <repository>\results_stage1\acceptance_20261011_final\ca_crossing3d
```

**ca_paper_random3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario paper_random3d --out <repository>\results_stage1\acceptance_20261011_final\ca_paper_random3d
```

**pplnm_empty3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario empty3d --out <repository>\results_stage1\acceptance_20261011_final\pplnm_empty3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
```

**pplnm_static3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario static3d --out <repository>\results_stage1\acceptance_20261011_final\pplnm_static3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
```

**pplnm_crossing3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario crossing3d --out <repository>\results_stage1\acceptance_20261011_final\pplnm_crossing3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
```

**pplnm_paper_random3d**

```powershell
<repository>\.venv-stage1\Scripts\python.exe <repository>\run_stage1.py simulate --scenario paper_random3d --out <repository>\results_stage1\acceptance_20261011_final\pplnm_paper_random3d --checkpoint <repository>\artifacts_stage1\pplnm_150\pplnm.pt --calibration <repository>\artifacts_stage1\pplnm_150\pplnm_calibration.json
```

## 原REMUS数值与动画

200s、dt0.02s、10001条记录；全部有限。原`run_remus.py`和`RemusPlant`回归的全部时间/位置/姿态/体速度及实际执行器状态按实际数值比较，最大状态差 **0.0**。这是物理输出比较，没有文件摘要检查。

- 最终深度：29.7497547401m。
- 最终航向：49.9988981850°。
- 路径长：453.5508894768m。
- 与原README舍入值的误差小于1e-5。
- 原入口的三维GIF保存成功，实际解码全部100帧；上游test_3D_plot同样通过。
- 8组运行点全部有限；航点演示3/3到达，360.02s。它没有障碍，不代表闭环避障。

## 三维场景实测

默认seed7、相同未放宽碰撞规则。表中的间隙为当前包围球/地形模型下实际执行的最小间隙。

| 预测器 | 场景 | 状态 | 仿真时长s | 最小间隙m | 应急次数 | 过期学习回退 |
|---|---|---|---:|---:|---:|---:|
| CA | empty3d | success | 80.45 | 34.161879 | 0 | 0 |
| CA | static3d | success | 81.65 | 0.135997 | 3 | 0 |
| CA | crossing3d | success | 81.60 | 1.748965 | 3 | 0 |
| CA | paper_random3d | success | 145.85 | 2.838456 | 2 | 0 |
| PPLNM | empty3d | success | 80.45 | 34.161879 | 0 | 0 |
| PPLNM | static3d | success | 83.55 | 0.318603 | 5 | 6 |
| PPLNM | crossing3d | success | 81.55 | 8.457442 | 0 | 3 |
| PPLNM | paper_random3d | success | 145.45 | 2.659696 | 0 | 6 |

PPLNM静态场景5次应急、最小间隙约0.319m；CA静态约0.136m。到达不等于满足整个任务1m间隙或理论安全保证，未据此宣传可靠避障。过期学习预测回退CA是显式记录的机制，不能声称所有周期适用学习校准覆盖。

两个随包模型（150轮与smoke）实际加载、预测、校准参数兼容、超预测范围拒绝均有pytest证据；同时真实执行了短CPU训练、重载与分块恢复测试，无需重训150轮。

## 数值、碰撞与可视化复核

输出检查记录：[numeric_visual_checks.json](../results_stage1/acceptance_20261011_final/numeric_visual_checks.json)、[日志](../results_stage1/acceptance_20261011_final/numeric_visual_checks.log)。

- 八个场景CSV均为10列、全部有限、时间严格递增；到达误差在配置容限内且模型最小间隙为正。
- 所有已选预测片段为46×6且有限；起始位置与同一规划时刻的执行状态对齐。
- 八套共40张PNG实际解码成功；三维起点、目标、障碍、海底和规划/执行标签实际打开检查。
- paper_random3d海底深度50–200m，120动态障碍；平海底场景为200m。NED图海面0在上、正深度在下。
- pytest保留并执行扫掠交叉碰撞、不同时间不碰撞、静态穿透、海底边界、坐标转换和传感时序等检查。
- 三个可视化专项测试与规划日志保存测试通过；未删除原测试或放宽碰撞标准。

实际图片： [随机地形与障碍](../results_stage1/acceptance_20261011_final/pplnm_paper_random3d/terrain_3d.png)、[随机环境完整航迹](../results_stage1/acceptance_20261011_final/pplnm_paper_random3d/trajectory_3d.png)、[动态交叉局部航迹](../results_stage1/acceptance_20261011_final/pplnm_crossing3d/trajectory_local_3d.png)、[原REMUS动画](../results_stage1/acceptance_20261011_final/original_remus/trajectory.gif)。

## 首轮失败与处理

首次在受限Windows执行环境运行同一验收脚本，pytest为146通过、1失败（上游test_3D_plot）；原run_remus动画也失败。两个异常都出现在Matplotlib调用pathlib解析输出路径，报WinError5拒绝访问。独立调用Path.resolve(strict=True)也复现同样异常。数字仿真、回归与八个场景在首轮均已完成。

未删除测试、未改上游绘图库、未跳过动画；改为在已授权项目结果目录内使用正常权限执行同一完整验收，147项和动画全部成功。首次日志保留于 [首轮报告](../results_stage1/acceptance_20261011/acceptance_report.json)、[pytest失败详情](../results_stage1/acceptance_20261011/pytest.log)、[原动画失败详情](../results_stage1/acceptance_20261011/original_remus.log)。

依赖下载首次被网络沙箱阻止，随后获准在新建环境安装成功；pip check实际通过。原环境与原锁文件未改。

## 边界与未完成项

本次没有无法完成的必需测试。未完成的开发功能为DWA—REMUS六自由度避障闭环，具体接口见 [REMUS_INTERFACE.md](REMUS_INTERFACE.md)。P25仿真结果不代表REMUS实际执行。未做GPU训练、实艇标定、正式多种子成功率或ECP研究验证。

静态文本搜索确认当前auv3d、tests_stage1、run_stage1.py、scripts_stage1中不存在文件或Git对象摘要计算/校验调用；历史档案字段未参与执行。原模型目录、原锁文件、CMD与examples普通Git差异为空。

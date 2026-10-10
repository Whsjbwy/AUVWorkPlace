# AUVWorkPlace

首次跑通的 REMUS 100 基础仿真，采用 Fossen PythonVehicleSimulator 六自由度动力学、深度/航向自动驾驶和水平洋流模型。

本版本不包含后来添加的论文海底地形、动态障碍物、碰撞检测、Gymnasium 或强化学习训练。

## 安装与运行

使用 Python 3.12，在仓库根目录 PowerShell 执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe .\run_remus.py
```

安装后也可以双击 `run_remus.cmd`，运行完成自动打开状态图和动画。依赖固定为首次验证版本，仿真器源码已经包含在本仓库，无需再次下载。

自定义示例：

```powershell
.\.venv\Scripts\python.exe .\run_remus.py --depth 20 --heading 90 --duration 200 --current 0.3
```

参数：`--depth` 目标深度（0~100 m）、`--heading` 航向（度）、`--duration` 时长（秒）、`--rpm` 转速（0~1525）、`--current` 流速（m/s）、`--current-direction` 洋流流向（度）。`--no-animation` 跳过动画，已有 GIF 不更新。

## 默认实验

| 参数 | 设置 |
|---|---|
| 时长 / 步长 | 200 s / 0.02 s |
| 初始位置、姿态、速度 | 全零 |
| 目标深度 / 航向 | 30 m / 50° |
| 推进器 | 1525 rpm |
| 洋流 | 0.5 m/s，流向 170° |
| 坐标系 | NED：x 北、y 东、z 向下；ZYX 欧拉角 |

首次运行共 10001 个记录点，状态均为有限数；最终深度 29.74975 m，最终航向 49.99890°，路径长度 453.55089 m。深度约有 0.25 m 偏差，保留官方控制器行为。计算耗时与硬件有关。

![状态与跟踪曲线](examples/states.png)

输出位于 `results/remus100/`，再次运行覆盖同名文件：

- `states.png`、`controls.png`：状态及控制曲线。
- `trajectory.png`、`trajectory.gif`：三维轨迹及压缩回放动画，圆点表示 AUV 位置。
- `simulation.csv`：全部状态和控制量，单位见列名。
- `summary.json`：参数、误差、轨迹长度和运行时间。

本案例没有海底或地图边界，只验证基础动力学和自动驾驶运行，不代表导航避障或实艇验证。

## 验证

```powershell
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m pytest PythonVehicleSimulator/tests/test_propeller_advance_speed.py PythonVehicleSimulator/tests/test_crossflow_drag.py PythonVehicleSimulator/tests/test_simulate.py::test_remus100 -q
.\.venv\Scripts\python.exe -m pip check
```

首次验证时，上述 4 项测试通过，依赖检查通过。

## 来源与许可

- `run_remus.py`：独立启动、导出和绘图，调用官方 `remus100` 与 `simulate`。
- `run_remus.cmd`：Windows 一键启动。
- `PythonVehicleSimulator/`：上游源码快照，省略预生成 GIF，不包含其 `.git`，源码未修改。
- 上游：https://github.com/cybergalactic/PythonVehicleSimulator
- 固定提交：`a93d1938ba1f78f2580c24c73f003091b2f03cd0`
- 作者 Thor I. Fossen，上游 MIT 许可保留于 [LICENSE](PythonVehicleSimulator/LICENSE)。

独立启动程序按时间正序保存数据，没有沿用上游 `main.py` 绘图前反转全部状态数据的处理。动力学模型与控制器保持原样。

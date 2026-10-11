# 第二阶段交付入口

原仓库内已接通参考空间DWA—原REMUS控制器—六自由度动力学—真实状态反馈闭环。原PVS参数及阶段1功能保留，没有ECP实现，没有校验和检查，未推送远端。

实际全量测试 **251 passed，0失败、0跳过**，其中原147项＋新增104项。正式14个REMUS工况13到达；9项通过有限时域验收，4项因不安全应急未通过，1项海床脊超时。完整结果见 [测试报告](docs_stage2/TEST_REPORT.md)，到达不能代替安全验收。

本机已安装的Python3.12环境可直接运行：

```powershell
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
.\.venv-stage1\Scripts\python.exe run_stage2.py simulate --backend remus6dof --scenario crossing3d --out "results_stage2/my_crossing_$runStamp"
Get-Content "results_stage2/my_crossing_$runStamp/summary.json"
Start-Process "results_stage2/my_crossing_$runStamp/trajectory.gif"
Start-Process "results_stage2/my_crossing_$runStamp/replay_3d.html"
```

完整压缩包不包含虚拟环境；解压后按 [README](README.md) 创建 `.venv-stage2`。基础仿真不需要GPU，全部学习测试还需可选PyTorch依赖。

直接查看已保存的实际成果：

- [三维航迹](results_stage2/visuals_final/crossing3d/trajectory_local_3d.png)
- [动画](results_stage2/visuals_final/crossing3d/trajectory.gif)
- [离线旋转/缩放/时间回放](results_stage2/visuals_final/crossing3d/replay_3d.html)
- [计划与实际误差](results_stage2/visuals_final/crossing3d/prediction_error.png)
- [理想P25速率假设的错配](results_stage2/regression_20261011/diagnose/tracking_diagnostic.png)
- [修改记录](docs_stage2/CHANGELOG.md)、[场景说明](docs_stage2/SCENARIOS.md)、[后续接口](docs_stage2/ECP_INTERFACES.md)

`--backend p25` 是共享场景的简化运动学对照，不能作为REMUS实际执行。原LSTM继续使用阶段1入口；阶段2采用CA障碍预测＋完整REMUS候选验证。同模型保持参考时的0预测误差只证明数值一致，当前不含实艇扰动或规划计算延迟的执行影响。

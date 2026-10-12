# 保存的真实三维成果

以下是原始正式输出的小体积副本，未重新执行任务。完整来源JSON、真实状态、独立预测、误差、障碍时刻、GIF及可旋转/缩放/时间回放HTML在独立实验ZIP中。NED深度向下，海床显示在艇体下方。实线是原REMUS六自由度真实执行，虚线是单独候选前向预测。

- [静态失败配对](static_failure.png)：formal static g14，seed520014，六法共同真值；A0到达但实际不足1米，其余超时，全数保留。来源`results_stage5/figures_formal_static3d_g14_v2_20261012`。
- [该失败的实际间隙、执行误差与RPM](static_clearance_and_error.png)：真实1米判据与碰撞0线分别显示，没有放宽。
- [多障碍动态交叉](multi_crossing.png)：formal multi g14，seed523014，障碍位于40秒，虚线是各法40秒独立预测。来源`figures_formal_multi_crossing3d_g14_v2_20261012`。
- [山脊海床](seabed_ridge.png)：formal terrain g14，seed524014，A0真实六自由度航迹与海床；无动态障碍，六法完全相同，属于共同导航负对照。来源`figures_formal_terrain_v2_20261012`。

图像精简只影响展示采样。实际碰撞与净间隙始终使用完整0.02秒真实积分和采样间检查。

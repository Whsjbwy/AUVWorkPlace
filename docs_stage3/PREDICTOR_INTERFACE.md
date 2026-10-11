# CA与PPLNM共同预测接口

`make_obstacle_forecaster(cfg)`由`obstacle_predictor=ca|pplnm`选择。规划器只调用`forecast(track, absolute_times, planning_now, extra_uncertainty_m)`，不接收障碍真实运动表。返回`ObstacleForecast`包含对象编号、严格对应请求的绝对时间、NED位置、各时刻包络半径与来源元数据。协议检查形状、有限值、正半径、因果历史、时间递增、对象编号和时间对应。一次规划的缓存只比较实际历史与请求参数，没有文件摘要验证。

默认CA仍按原观测周期重采样，原`.75+age*max(1,observed_speed)`膨胀与原碰撞定义保留。`prediction_history_dt_s=3`是CA/PPLNM初步对照共同设置，不能把CA默认1秒与PPLNM3秒的差异冒充纯预测器贡献。

已有PPLNM版本为6个历史样本、9维输入、3维位置输出、dt3秒、3个未来头；支持范围从最后**测量**时刻起9秒。模型仍是已有150轮小规模合成数据版本，未在本轮重新训练，未用实验/测试轨迹拟合或调节模型、半径和校准。配置与权重/校准匹配检查使用结构、归一化维度、dt、步数、训练标签和轮数等实际字段。

REMUS默认规划24秒，必须显式选择延拓：

- `cv_after9`：9秒后以已学习6到9秒预测位置的割线速度继续，位置连续。
- `ca_after9`：在已学习9秒端点上，以因果历史CA在9秒的速度与加速度继续，位置连续。
- `reject`：请求超过测量时刻+9秒即明确拒绝；不能静默退回CA或钳住位置。默认24秒规划选择此模式会因范围不够而报错。

两个尾部都只是工程延拓，未经9秒外训练或校准，不代表同等预测可信度。`trained_valid_mask`、`trained_support_end_s`、`extension_point_count`逐时刻/逐预测记录；陈旧观测会减少从当前规划时刻起可用的训练支持，甚至整个请求均属于延拓。

`p25_calibration_radii_m`与`calibration_valid_mask`保留已有P25式经验球接口；超出9秒返回`None/false`。本轮预测器对照**不把这些经验球加入导航半径**，两者使用相同原工程包络，只隔离预测器变化。元数据明确`calibrated_radii_applied=false`。这些经验边际半径不是ECP概率、整个任务安全保证或独立24秒校准版本。

共同时间锚点评估由`scripts_stage3/analyze_prediction_errors.py`进行：先确认每个历史点确实在决策前交付，再按原种子离线重演真值并与保存快照逐数值核对。CA与PPLNM使用完全相同对象、历史、决策时间和绝对未来时间，分别报告3/6/9/12/18/24秒RMSE及测量起算是否在9秒内。真值只用于执行后评价，不反馈规划或校准。

```powershell
.\.venv-stage1\Scripts\python.exe run_stage3.py simulate --scenario crossing3d --obstacle-predictor pplnm --prediction-history-dt-s 3 --pplnm-extension-mode cv_after9 --out results_stage3/pplnm_run
.\.venv-stage1\Scripts\python.exe scripts_stage3/analyze_prediction_errors.py --source results_stage3/pplnm_run --out results_stage3/paired_errors
```

独立24秒多时域模型与独立训练/验证/校准拆分尚未建立；不得把尾部当作该版本已经实现。

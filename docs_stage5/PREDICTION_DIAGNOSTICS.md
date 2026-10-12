# 阶段5离线预测、噪声与校准诊断

`scripts_stage5/prediction_noise.py`在真实dev/tune/test角色上比较CA、CV和
filtered_cv。三个预测器接收同一已经交付的历史副本；未来真值仅在事后误差与
覆盖计算中使用，不用于选择预测器、更新校准、调整裕量或在线规划。
本文件记录已实际完成的pilot DEV诊断，不把它称为独立测试或REMUS闭环成功。

## 已执行命令及范围

从仓库根目录，在CPU环境实际执行：

```powershell
$env:PYTHONUTF8='1'
$env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.\.venv-stage1\Scripts\python.exe -m pytest tests_stage5/test_stage5_noise_analysis.py -q --junitxml results_stage5/prediction_noise_checks_20261012/junit.xml
.\.venv-stage1\Scripts\python.exe -m scripts_stage5.prediction_noise --phase pilot --split dev --registry results_stage5/data_pilot_20261012/ARTIFACT_REGISTRY.json --out results_stage5/prediction_noise_pilot_dev_20261012
```

新增9项测试全部通过，0.70秒。实际诊断退出0，2026-10-12 06:48:52 UTC完成：
6个唯一DEV几何×3种噪声=18份完整观测来源，三预测器共54个条件，零程序错误。
终端记录分别保存于`results_stage5/prediction_noise_checks_20261012/pytest.log`
与`results_stage5/prediction_noise_pilot_dev_20261012.log`。
运行时有其他pilot闭环worker，这次墙钟耗时不用于公平实时性能结论。

正式TEST尚未由本脚本执行。只有主参数冻结且阶段负责人明确开始评价后，才对
formal/test调用相同入口。新调用使用新输出目录；脚本拒绝覆盖已有目录。
脚本不接受把calibration角色当作未见评价，也不支持修改case角色后构库。

## 观测、标签及统计单元

采用协议中固定1米/秒直线观测载体、真实声呐视场/距离/延迟/噪声、4秒过期，
锚点20、40、60秒。每条预测保存0–24秒、0.2秒步长的121个时间点。
该载体不是实际REMUS执行轨迹；与真实闭环ego、视场和目标分布存在协变量漂移。
没有根据成功航迹挑样本，但也不能据此宣称场景交换性或闭环安全保证。

主要RMSE报告3、6、9、12、18、24秒，以三维欧氏位置误差平方和除以实际
目标-锚点标签数再开方。聚合加总平方和与标签数，**不平均各案例的RMSE**。
同场景多个目标/锚点相关，标签计数不是独立场景数。

独立库从同phase的calibration场景冻结，按预测器和噪声分别加载；验证实际
算法参数、噪声、身份、真实来源角色和时间网格。DEV评价没有重算或更改这些库。
海床两个无障碍族保存空掩码，但RMSE/覆盖未知，不能记作零误差或100%覆盖。

## DEV实际预测误差

下表每格有24个目标-锚点位置标签；两个海床DEV不贡献障碍标签。
完整六个预测时域见`RMSE_BY_HORIZON.csv`，逐案例数据见
`RMSE_BY_CASE_HORIZON.csv`。

| 噪声标准差（米） | 预测器 | 9秒RMSE（米） | 24秒RMSE（米） |
| --- | --- | ---: | ---: |
| 0 | CA | 0.169 | 2.397 |
| 0 | CV | 0.622 | 3.947 |
| 0 | filtered_cv | 0.806 | 4.358 |
| 0.02 | CA | 3.640 | 24.232 |
| 0.02 | CV | 0.668 | 3.851 |
| 0.02 | filtered_cv | 0.774 | 4.278 |
| 0.1 | CA | 18.247 | 120.780 |
| 0.1 | CV | 2.180 | 6.243 |
| 0.1 | filtered_cv | 0.823 | 4.111 |

这组DEV结果支持原CA在噪声下长时外推误差显著放大的诊断；零噪声CA反而最好。
filtered_cv在高噪声下改善本组预测误差，但这不是正式TEST、不是全部运动分布，
也不能推出其真实六自由度闭环必然优于CA或CV。
filtered_cv仅估计较平滑速度，位置仍锚定最新带噪测量，没有把位置去噪当作已实现。

## A2分时域/速度球与A3联合球

A2每速度组、每时间点在calibration完整场景间计算有限样本上分位；缺该组的
场景不填零。两组仍为已观测速度`[0,0.25)`和`[0.25,+∞)`米/秒，不能查询未来
真实速度。pilot alpha0.2的实测库n为：

| 噪声标准差（米） | 观测静止组完整场景n | 观测移动组完整场景n |
| --- | ---: | ---: |
| 0 | 4 | 12 |
| 0.02 | 4 | 12 |
| 0.1 | 4 | 16 |

噪声可让同一真实静态场景同时在多个观测组出现，因此两组n不能相加当独立样本数。
三预测器使用相同的历史和分组。所有121个时刻的q、n及阶数保存在
`A2_CALIBRATION_TIME_SPEED_QUANTILES.csv`。若阶数超过有效n，保存
`insufficient_samples`和未知q，不能截断成有限上界或记作覆盖成功。

A2分别报告两类经验量：

- `empirical_position_coverage`：该组/时间点实际位置标签中误差不超过q的比例。
- `empirical_scene_horizon_coverage`：该组/时间点完整评价场景内所有实际可用
  目标/锚点的最大误差是否不超过q，再按有标签的完整场景统计。

这些是不同分母。逐案例数见`A2_COVERAGE_BY_CASE.csv`，聚合见
`A2_EMPIRICAL_COVERAGE.csv`；0标签或缺有限q时覆盖为未知，不填零或一。
本DEV在24秒的所有实际有标签组都被相应A2球覆盖，但只有4个有障碍DEV场景，
不能解释为具有统计认证的80%条件覆盖，更不能解释为整个闭环安全率。

A3在calibration每个活跃完整场景内取全目标/锚点/121时刻最大欧氏误差，然后
用16个活跃完整场景作联合分位。DEV联合经验覆盖的评价单位也是有标签的完整
场景，不是位置点。该全时域q及本DEV结果为：

| 噪声标准差（米） | CA联合q（米） | CV联合q（米） | filtered_cv联合q（米） | 每法DEV联合覆盖 |
| --- | ---: | ---: | ---: | --- |
| 0 | 8.830 | 11.584 | 13.362 | 4/4 |
| 0.02 | 48.498 | 12.689 | 13.310 | 4/4 |
| 0.1 | 233.105 | 19.981 | 14.084 | 4/4 |

来源为`A3_JOINT_COVERAGE_BY_CASE.csv`及`A3_EMPIRICAL_JOINT_COVERAGE.csv`。
分组条件和有限样本阶数不同，A2某组q不保证小于A3联合q；例如零噪声移动组
24秒球可比联合球更大。不能只用某一个q大小代表整体效率或安全。

**逐位置覆盖、场景内某时刻最大误差覆盖、24秒场景联合覆盖及任务安全是四种
不同指标。**本脚本没有实际REMUS闭环、候选相关A5或任务碰撞评价，不能将这些
预测诊断计入导航成功数。DEV结果可以暴露机制，不能代替未见TEST。

## 原始交付文件

输出目录为`results_stage5/prediction_noise_pilot_dev_20261012/`：

| 文件 | 内容 |
| --- | --- |
| `EXPERIMENT_CONFIG.json`、`SUMMARY.json` | 真实角色、phase、配置、18来源/6几何/54条件、程序错误数与范围。 |
| 每case的`RAW_FORECASTS.npz` | 三预测器世界坐标完整预测、世界/局部标签、mask、ID、观测速度及时间网格。 |
| 每case的`ACQUISITION.json` | 原角色、全部已经交付的观测、时间、载体规则和缺失原因。 |
| 每case的`HISTORIES_AND_FORECAST_RECORDS.json` | 每锚点实际历史与预测器参数/记录。 |
| 各RMSE与覆盖CSV | 可按平方和/计数重算，空/未知明确保留。 |
| `PREDICTION_RMSE.png` | 各噪声下真实预测误差随时域。 |
| `A2_TIME_SPEED_QUANTILES.png` | 各速度组、噪声及预测器的独立校准球。 |

两个PNG已实际生成并查看，曲线及图例正常。未修改冻结协议、校准库、物理参数、
碰撞条件、1米安全间隙或到达标准，未计算额外文件Hash。

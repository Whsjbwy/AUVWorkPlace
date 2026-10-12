# 阶段5风险与预测接口

本轮将“预测器误差、校准形状、工程包络叠加”拆开。REMUS水动力、自动驾驶及执行器参数、物理包围球、实际碰撞检测、1米实际安全间隙和到达标准均不由本模块修改。阶段4源文件和原始结果保留；A3/A4复用其分数形式，在**本轮独立校准数据**上重新取分位，不把新分位冒充阶段4原数值。

## 三个共同预测器

`stage5_forecasting.make_forecaster(predictor, cfg)`返回原`ForecastContext`可注入的对象；正式名称是`ca`、`cv`、`filtered_cv`。`filtered`只是兼容别名，日志和校准契约归一为`filtered_cv`。

| 预测器 | 位置及导数估计 | 保持不变的事项 |
|---|---|---|
| CA | 直接返回原`CAObstacleForecaster`，沿用阶段3历史插值和二阶差分 | 原CA数值、历史间隔、年龄及记忆包络 |
| CV | 最后两个已交付观测的位置差除以实际采集时间差 | 最新测量位置作为位置锚点 |
| filtered CV | 至多6个已交付观测，以最新位置固定截距，按实际采集时差最小二乘拟合速度 | 最新位置没有去噪；无速度裁剪、无加速度、无未来数据 |

对filtered CV，令最新采集时刻为`t_n`，`δ_i=t_i−t_n`，则

```text
v_hat = argmin_v Σ_(i=n-5)^(n-1) ||(p_i−p_n) − δ_i v||²
p_hat(t) = p_n + (t−t_n) v_hat
```

只有一个观测时保持位置；两个观测时退化为CV。数值实现缩放时间矩阵改善条件数，但不改变该最小二乘目标。该模块仅减小速度估计的噪声敏感性，不能称为完整状态滤波，也不能保证机动预测更准确。所有预测使用绝对仿真时刻，失联年龄不会错误地重置为零。历史采集时间必须严格递增且不晚于当前规划时间。参数由`forecaster_contract(predictor,cfg)`完整记录；CA/CV/filtered CV各自校准，不能交换量化半径。

## 六种风险处理

设`P`为同一预测器的预测，`Y`为离线标签，`j`为完整独立几何场景，`w`为注册采集窗口，`o`为当时已经观测的目标，`k`为预测步。所有误差都在同一时间锚点、同一NED位置和相同航向局部坐标下计算。速度分组仅使用最后两次已交付观测的速度模长：`[0,0.25)`和`[0.25,+∞)`米/秒；不是未来真值速度。

| 方法 | 校准量 | 候选相关性 |
|---|---|---|
| A0 | 原固定工程包络，额外q为零 | 无 |
| A1 | 独立调参集选择并冻结的显式非负固定值 | 无 |
| A2 | 逐预测步、观测速度分组的位置误差球 | 无 |
| A3 | 全窗口、全目标、全24秒最大位置误差的统一球 | 无，保留阶段4joint形式 |
| A4 | 全窗口、全24秒最大候选扫掠间隙误差 | 有，保留阶段4冻结joint ECP形式 |
| A5 | 逐时间区间、观测速度分组的候选扫掠间隙误差 | 有，三维工程适配 |

A2恢复P25式“按预测步及速度组分别标定”的结构；为了避免把同一轨迹的相关帧冒充独立样本，本轮每个时间/速度组先在**同一完整场景**内对有效窗口和目标取最大值，再以场景为统计单位。它不是原P25训练数据、速度分组边界和全部校准协议的逐字复现，也不是PPLNM网络本身的验证。

```text
s_ball[j,b,k] = max_(w,o: observed_speed[w,o]∈b)
                    ||P[j,w,o,k]−Y[j,w,o,k]||
q_ball[b,k] = upper_order_statistic({s_ball[j,b,k] : scene j observed bin b})
```

完整无障碍场景单独标记为空并保留注册信息，**不贡献零误差分位样本**。某个场景没有观察到某个速度组时，该组分数为未知，不能填零；非空场景完全缺少必要标签则构库报错。场景、窗口、目标和时间采样数分别记录，不把窗口数当作独立n。

## ECP分数、几何与时间

ECP已有原理来自[作者论文公开版](https://arxiv.org/html/2504.00447v1)式(7)、(8)、(11)、(12)和[CORE-SNU作者实现](https://github.com/CORE-SNU/ECP-MPC)。本项目使用候选位置重评成熟预测/标签的思想，不宣称原创ECP，也没有实现作者在线自适应ACP。阶段4的核对及作者代码位置见[阶段4风险说明](../docs_stage4/RISK_METHODS.md)。

本轮继续使用同步三维球扫掠几何。对同一时间区间`I_k=[τ_k,τ_(k+1)]`和候选路径`φ`，在每个障碍扣除其实际半径及车辆半径后取最小值：

```text
g_b(φ,Z,I_k) = min_(o: observed_speed[o]∈b) min_(λ∈[0,1])
  ||lerp(φ_k,φ_(k+1),λ) − lerp(Z_o,k,Z_o,k+1,λ)|| − R_vehicle − R_o
s_ECP[j,b,k](φ) = max_w [g_b(φ,P[j,w],I_k) − g_b(φ,Y[j,w],I_k)]_+
```

预测与标签的最近目标可以不同，不能写成逐目标差的最大值。A5在每个速度组内单独执行上述最近集合计算；在线各目标使用所属速度组的q，最终取所有组及地形/边界约束的最小净间隙。A4沿用阶段4**所有目标共同取最近值后作差**，再对时间/窗口取最大；不能用A5各组分数最大值替代A4。

库的0.2秒网格只定义经验分数。候选车辆仍由隔离的原REMUS控制器和执行器在不大于0.02秒的物理步长前向积分。A2逐点q转换成相邻端点最大值作为整个风险区间半径；A5直接使用逐区间q，在区间交界取相邻两个区间较大值。随后在每个物理采样区间执行原同步扫掠验证，不以0.2秒端点距离代替碰撞检测。不把逐预测步经验覆盖宣称为所有时间共同覆盖，更不能推出任务无碰撞概率。

## add和replace的明确含义

原预测包络为

```text
R_base = R_obstacle + R_vehicle + m_fixed + m_age + m_memory
m_fixed = 原cfg.prediction_margin_m
m_age = observation_age × max(1, observed_speed)
```

除A0固定保持原包络外，A1–A5提供独立配置：

```text
add:     R_risk = R_base + q
replace: R_risk = R_obstacle + R_vehicle + m_memory + q
```

replace只去掉`fixed_prediction_margin_m`和`stage2_age_margin_m`两个白名单工程项。失联记忆额外外推界保留；本轮没有对删除该界做独立条件校准。实际1米安全阈值继续由原规划器和碰撞评价使用，不藏在q中，也不随风险方法改变。A1的`fixed_margin_m`必须由调用方明确给出，并由独立调参记录解释来源；模块没有偷偷给“最优值”。

快速45候选排序与有限完整动力学预算继续共享原预测包络；风险方法作用于入围候选的完整物理验证。这一固定短名单可能漏掉安全动作，不能用风险模块名称掩盖该限制。同一状态对比需重用相同控制器、参考和执行器状态以及相同已交付历史。

## 校准工件契约

每个phase、预测器和噪声条件保存独立NPZ。主要字段：

| 字段 | 形状/含义 |
|---|---|
| `pred_local`, `label_local` | `[scene,window,object,time,3]`，局部NED平移/航向旋转 |
| `mask`, `radii`, `observed_speeds_mps` | `[scene,window,object]`；半径为实际障碍半径 |
| `times_s` | 相对预测时刻，从零到共同24秒，严格递增 |
| `case_ids`, `scene_ids` | 唯一完整场景身份；同一几何的噪声副本不能扩充n |
| `empty_scene_mask` | 明确完成的真空场景；区别于缺失标签 |
| `metadata_json` | 下述实际参数及各场景来源记录 |

元数据必须包含`schema='stage5-risk-v1'`、`split='calibration'`、`predictor`、完整`predictor_contract`、`noise_std_m`、与数组同序的`case_ids`、`speed_bin_edges_mps=[0,0.25,'inf']`及逐场景`scenes`。加载器同时检查逐场景角色、case/scene身份和噪声条件，不以顶层声明掩盖测试来源。只检查这些实际参数、形状、时间和数值，不做文件摘要绑定。

完整场景量化采用`k=ceil((n+1)(1−α))`上侧次序统计量。若`k>n`，保持不可有限校准的事实；A2/A5会明确拒绝样本不足的速度组，不截断无穷，也不偷偷回退到另一组或另一预测器。A3/A4只要求其joint活跃场景池样本充足。注册的α、组内有效n、排除的空场景、不可校准请求均须报告。

本轮离线采集载体沿注册起点终点直线以固定速度获取因果声呐历史，它**不是REMUS实际执行轨迹**。因此校准不取决于某种方法是否成功到达，但存在采集载体到真实闭环FOV/失联历史的协变量偏移。局部航向刚体对齐只保留几何，不证明跨场景交换性。候选选择自适应、场景族相关性、失联目标、执行延迟、候选预算与动力学不可行都限制覆盖解释。

## 调用及离线评价

```python
forecaster = make_forecaster('filtered_cv', cfg)
policy = make_policy('A5', library_path, alpha=0.1,
                     combination='replace', cfg=cfg, predictor='filtered_cv')
result = run_remus_episode(cfg, scenario, forecaster=forecaster,
                          risk_policy=policy, scene=registered_scene)
```

风险hook只接收候选状态、已交付tracks、规划时间、配置及因果预测上下文；没有场景真值参数或在线更新入口。离线评价可通过纯函数`time_ball_scores`、`time_ecp_scores`及旧`ecp_scene_scores`重算测试误差，但其标签不能回传给规划决策或调参。

A2日志`q_by_speed_bin_m`为`[bin,time]`，A5为`[bin,interval]`；`score_kind`区分含义，两者均保存`q_interval_by_speed_bin_m`。每目标保存实际观测速度组和边界风险半径。q为零且`observed_set_empty=True`只表示没有被追踪障碍，不是误差为零保证。覆盖、反事实选中参考误判、实际碰撞、实际1米违规、到达、危险应急及计算超周期必须分别报告。危险应急使用各方法自己的风险包络，因此不能单独拿其次数作为统一真实安全指标。

本轮没有PPLNM新接入、在线ACP、闭环安全定理或电功率标定。若A5相对A1/A2没有稳定的独立闭环收益，应报告负结果，不追加网络复杂度掩盖原因。

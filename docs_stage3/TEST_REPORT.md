# 阶段3实际测试与实验报告

本轮在Windows/Python3.12.14、现有独立`.venv-stage1`、CPU PyTorch2.10.0环境实际执行。原碰撞定义、1米净间隙、5米最终到达标准及REMUS物理参数均未改。所有本轮原始结果位于新`results_stage3/`，旧失败和红灯测试保留。

## 最终验收范围

最终完整回归 **556通过，0失败、0错误、0跳过**，包含`PythonVehicleSimulator/tests`、`tests_stage1`、`tests_stage2`、`tests_stage3`，耗时45.974秒，原始证据`results_stage3/final_release_quality/pytest.log/xml`。此前校准速度/地形推进修复后546通过，加入零观测保护后外cwd554通过；随后注册元数据修复后再次全量获得556。各轮数量不能相加当成覆盖数量。

实际命令为`.venv-stage1/Scripts/python.exe -m pytest`加上述四组测试路径、`-q --junitxml=<独立结果目录>/pytest.xml`。仓库外回归使用解释器与测试的绝对路径、`PYTHONPATH`指向仓库，cwd为独立`pytest_work`，对应GitHub CI的工作目录语义。原全部测试保留，没有删除失败测试或跳过真实模型。

实际collection按路径计数：原PythonVehicleSimulator18、阶段1 129（合计原阶段1口径147）、阶段2 206、阶段3 203，共556。最终102个项目Python源文件内存语法编译通过；`pip check`无冲突，保护文件差异为空，见`final_release_quality/quality_checks.json`。

实际发现的问题均有复现/修复证据：

| 确定问题 | 修前 | 修复与实际复验 |
|---|---|---|
| 负预测半径被相邻最大值隐藏 | 33通过/1失败 | 先检查全部半径，再合并分段；34通过，`forecast_contract_tests/` |
| numpy.bool_安全结果不能保存JSON | 原异常堆栈保留 | 只转换原生bool，数值/阈值不变；新增2项结果保存回归 |
| PPLNM校准分箱误用原始观测速度 | 弯曲dt1历史与dt3校准特征不一致 | 使用原数据集dt3重采样速度；27模型契约+17分析测试通过；导航半径不变 |
| 地形子目标越过转角后进度锁住 | 28通过/1失败 | 保持6米/25米参数，局部相邻段推进须通过原扫掠检查；29地形+2保存测试通过，真实同配置复验见下 |
| 仓库外真实模型测试使用相对权重路径 | GitHub Ubuntu/Windows各3错误；本地24通过/3错误 | 仅fixture按源码解析绝对路径；外cwd27通过、阶段3全部201通过、全量554通过，`ci_path_fix_20261011/` |
| 自定义合法.01秒注册误报共同.02秒 | 新回归14通过/2失败 | 从完整注册配置派生共同步长，混合值明确列出；runner/capture19通过，`registration_metadata_fix_20261011/` |

原入口、动力学与依赖实际检查：`python -m pip check`无冲突；`run_stage1.py audit`通过；`run_stage1.py regression --out results_stage3/original_remus_regression`完成200秒/10001有限状态；`run_remus.py --duration 200 --out results_stage3/common_quality/original_remus`原图像/GIF生成，最终深度29.749755米、航向49.998898度；`run_stage2.py responses --out results_stage3/common_quality/responses`原响应检查全部通过。原始命令输出见`common_quality/`。保护文件相对main无变更。

## 真实闭环对照

下列是实际启动命令，默认seed7：

```powershell
python scripts_stage3/run_experiments.py --suite memory --output results_stage3/common_memory_20261011 --workers 2
python scripts_stage3/run_experiments.py --suite candidate --output results_stage3/common_candidates_20261011 --workers 2
python scripts_stage3/run_experiments.py --suite predictor --output results_stage3/common_predictor_20261011 --workers 1
python scripts_stage3/run_experiments.py --suite smoke --output results_stage3/unified_runner_smoke_20261011 --limit 1 --workers 1
```

memory15、candidate16、predictor8共39组全部完成，程序异常0。它们是不同独立配置，不能合成总成功率；各配置单seed单例仅为描述性初步对照。

| 独立实验 | 实际结果 | 精简资料 |
|---|---|---|
| 无障碍smoke | 84.92秒到达，间隙29.175962米，应急0；该命令仅执行登记3例中的1例 | 对应原始目录；静态/交叉另有真实完整实验 |
| 记忆15例 | 宽视场硬4/8/12均到达但1米违规；保留/条件删除超时；窄视场五方案全碰撞；条件删除0次 | [逐配置报告](experiments/MEMORY_PILOT_SUMMARY.md)、[CSV](experiments/MEMORY_RESULTS.csv) |
| 候选16例 | static仅ranked9、速度1.5多交叉仅coverage9通过有限时域验收；增加预算不单调改善 | [报告](experiments/CANDIDATE_PILOT_SUMMARY.md)、[CSV](experiments/CANDIDATE_RESULTS.csv) |
| 首失败同状态全格 | 11捕获帧完整45/68格均0安全；共同起点预测gap已小于1米；不是这些帧的预算遗漏，不能称物理无解 | 保存完整plant/controller/actuator/历史、各策略×9/18/45计数 |
| 验证备份 | 两例共32次完整6DOF备份验证，0安全备份；保留真实RPM0滑行，不假设瞬停 | 候选报告/原逐次日志 |
| CA/PPLNM公平8例 | 全到达且无碰撞/实际1米违规；CA静态unsafe1、PPLNM多交叉unsafe5，均验收失败 | [报告](experiments/PREDICTOR_PILOT_SUMMARY.md)、[CSV](experiments/PREDICTOR_RESULTS.csv) |
| 原paper_random3d | 191.66秒到达，间隙0.722766米，应急20/unsafe19，验收失败；120总障碍但实际交互/最大观测/跟踪均3 | `paper_random_retained/` |
| 山脊 | direct200超时；A*133.32秒到达/间隙6.383011米/应急0；修复后轨迹逐值相同 | [地形消融](TERRAIN_ABLATION_README.md)、[CSV](TERRAIN_ABLATION_SUMMARY.csv) |
| 旧Perlin初始化诊断 | direct/A*均11.42秒碰撞，初始完整候选/备份无安全参考；负结果保留 | `terrain_closed_loop_20261011/` |
| 新独立巡航准入Perlin | 初版direct/A*200超时；只修进度后的同A*142.60秒到达，间隙17.098811米，应急0 | `perlin_cruise_guidance_fixed_20261011/`，旧direct明确复用，非重复新执行 |

地形真实启动脚本、参数与日志保存在各目录`terrain_experiment.py`和`batch_command.log`；命令为`python <该目录>/terrain_experiment.py --scenario <场景> --guidance <模式>`。源码统一入口`run_stage3.py simulate`支持同样配置。原Perlin失败与新准入任务分别报告，没有改旧起点使其通过。

## 预测评价、图像及限制

最终27已完成目录的同锚点离线评价产生38268条误差/642共同支持统计；候选16例另产生12708条误差/369统计。每个保存真值快照重演最大差0，历史均核对为决策前已交付。模型、裕量、校准没有按测试结果调节。CA_dt3/PPLNM9+CV尾/PPLNM9+CA尾在同锚点、同历史、同绝对时间比较，并按最后测量起算的共同9秒支持分组。CA原dt1导航仍标dt1；离线CA_dt3不能冒充其在线预测误差。无目标RMSE明确未定义而非0；有目标未评价则pending。原矩阵CSV中的运行时pending保留，交付CSV另挂接完成的离线数据。

实际图像包含海床、起终点、真实6DOF航迹、当前预测、同时间障碍、事件与计划执行误差；HTML回放和Perlin/ridge真实30帧GIF已生成，NED深度向下已经目检。代表图见`examples/`，全部失败图与原日志在审查ZIP。

墙钟P95/期限次数为实测，但共享CPU且同步物理时间在规划时暂停；不作实时证书。确定模型下计划执行误差接近0不代表海试误差0。当前默认CA，PPLNM9已真正进入REMUS闭环但尾部仅显式工程延拓；没有24秒训练/校准版本、ECP或真实电能结论。详见[实验协议](EXPERIMENT_PROTOCOL.md)、[失败分析](FAILURE_ANALYSIS.md)。

## 历史开发记录（保留当时状态，以下待执行措辞不代表最终状态）

运行平台：Windows，Python3.12.14，现有独立`.venv-stage1`，CPU推理。结果只写入新的`results_stage3/`，此前阶段1/2实验和ZIP保留。

| 实际执行命令/检查 | 结果 | 原始记录 |
|---|---|---|
| `python -m pytest PythonVehicleSimulator/tests tests_stage1 tests_stage2 tests_stage3 -q --junitxml=results_stage3/common_integration_tests/pytest.xml` | 439通过、0失败/错误/跳过；包含原351项 | `common_integration_tests/pytest.log/xml` |
| `python -m pip check` | No broken requirements found | 本轮质量检查记录 |
| `python run_stage1.py audit` | 原源码可用，模型矩阵正定，原RPM/舵限/时间常数有效 | 本轮质量检查记录 |
| `python run_stage1.py regression --out results_stage3/original_remus_regression` | 原六自由度200秒、10001状态，全有限，与README四舍五入一致 | `original_remus_regression/summary.json` |
| 候选模块原规划/控制+新增测试 | 77通过，其中26新增 | `candidates_verified_tests.xml` |
| 感知记忆测试 | 38通过 | `memory_unit_final_20261011/pytest.log/xml` |
| 海床几何引导测试 | 23通过 | `terrain_module_20261011/tests_final.log/xml` |

完整闭环已实际执行：expanded/coverage/backup无障碍88.82秒到达、净间隙29.175962米、应急0；静态原coast与verified_backup均87.4秒到达、净间隙2.342522米、应急1且不安全预测1。后两者验收失败，不能把到达算安全通过。备份帧验证8条均不安全，并超过2秒规划期限一次。原coast的states/controls/clearance/prediction_errors与历史阶段2静态实验逐数值一致，最大差异0。

上述候选模块实验保存于`candidate_checks/`，先于最终统一日志集成；其中旧summary的rollout上限字段对8备份不充分，实际次数以`candidate_observer.json`为准。这一限制保留在原始证据中，最终代码已经分列预算上限、实际备份验证次数与安全备份事件。

共同基线提交前再次执行全量pytest：**473通过，0失败/错误/跳过**，原始`common_quality/pytest.log/xml`。新增预测契约34项首次33通过/1失败，发现可变半径负值会被分段最大值隐藏，修复为先验证每项半径，再分段归并；`forecast_contract_tests/initial.log/xml`及`fixed.log/xml`均保留。另补预测对象编号/请求时间戳匹配检查，末端记忆只推进诊断、不额外交付观测。

最终集成CLI实际运行`python run_stage3.py simulate --backend remus6dof --scenario crossing3d --max-time-s 120 --out results_stage3/common_crossing_cli --no-animation`：到达、最小间隙13.532717米、应急0、不安全预测0，有限时域验收通过；图像与HTML回放已生成。原`run_remus.py --duration 200`入口、独立控制响应检查也实际通过，详见`common_quality/quality_checks.json`。墙钟P95受同时地形任务CPU争用影响，不能作为隔离实时性能。

地形与统一矩阵仍在执行；未完成项不计为通过。后续提交将追加各配置的实际结果、CA/PPLNM共同时间锚点误差和最终全量测试数。

PPLNM接入后实际全量回归：`pplnm_quality/pytest.log/xml`记录**532通过，0失败/错误/跳过**。随后另补的2项结果保存回归通过：合法深度上限主导间隙时，原比较产生`numpy.bool_`导致JSON异常；现在只将两项安全结果转换为Python `bool`，数值/阈值不改。原失败堆栈与修复后smoke均保留，新测试`test_episode_serialization.py`覆盖保存与到达分支。

真实PPLNM权重/校准契约26项以及离线评价16项首次合计42通过；共同支持范围分组补强后43项通过。PPLNM CLI动态交叉85.4秒到达，净间隙11.325517米，应急0，验收通过。该独立跑通不能直接与历史CA(dt1)当作纯方法优劣；统一CA/PPLNM实验均dt3，正在执行。已有同锚点离线结果并不支持PPLNM总体优于CA，训练域外CV尾部误差明显，应单独报告。

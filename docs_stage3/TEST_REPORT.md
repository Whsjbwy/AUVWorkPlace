# 阶段3执行记录（开发中）

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

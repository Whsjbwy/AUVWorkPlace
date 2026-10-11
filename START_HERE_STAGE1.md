# 阶段1已合并到 AUVWorkPlace

请从根目录 [README.md](README.md) 开始安装、运行和查看结果。

当前仅调用仓库原有 `PythonVehicleSimulator/`；不需要重复依赖。源码身份和模型/校准摘要验证已移除，改为接口、维度、采样间隔、预测范围及训练配置检查。

原REMUS程序、动画和测试保留。三维DWA执行仍采用P25简化运动学，六自由度航点示例没有障碍，不能视为完整六自由度避障。

本次Windows/Python3.12验证见 [TEST_REPORT.md](docs_stage1/TEST_REPORT.md)，变更见 [MERGE_CHANGELOG.md](docs_stage1/MERGE_CHANGELOG.md)，后续接口见 [REMUS_INTERFACE.md](docs_stage1/REMUS_INTERFACE.md)。

`artifacts_stage1/` 与云端报告为随包历史成果，保留成功和失败记录。直接加载 `artifacts_stage1/pplnm_150/pplnm.pt`，无需首先重新训练。

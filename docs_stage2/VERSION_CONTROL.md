# 研究基线的版本管理与发布流程

主仓库仍为 `Whsjbwy/AUVWorkPlace`。本阶段代码、共同配置、种子、测试、原模型与准确结果说明应在同一研究分支审阅；不另建独立工程，不重置已有工作，不强推。当前研究分支约定为 `research/stage2-baseline`，是否推送及创建PR以用户授权为准。

## 提交边界

正常提交源码、测试、可复制PowerShell命令、依赖清单、共同场景配置和文档。保留原 `PythonVehicleSimulator/`、`run_remus.py`、`run_remus.cmd`、`requirements-lock.txt` 及既有示例。模型来源及参数合法性通过实际配置/形状/数值检查，不加入任何文件或Git对象校验和验证。

本机原 `artifacts_stage1/` 共228个文件、73,768,119字节，其中大部分是历史状态、数据、原始轨迹、图片及日志。它们继续留在本地和完整交付归档中；`.gitignore` 仅影响提交选择，不删除文件。普通Git提交精确保留以下推理夹具和说明：

| 路径 | 大小/作用 |
| --- | --- |
| `artifacts_stage1/README_ARCHIVE.md` | 历史成果来源和现成模型说明 |
| `artifacts_stage1/pplnm_150/pplnm.pt` | 3,215,431字节；150轮现成权重，供阶段1预测与功能测试使用 |
| `artifacts_stage1/pplnm_150/pplnm_calibration.json` | 预测器配置与校准分位数 |
| `artifacts_stage1/pplnm_150/ca_calibration.json` | CA对照的参数与校准 |
| `artifacts_stage1/pplnm_smoke/pplnm.pt` | 3,215,367字节；原有参数化加载测试所需的第二份权重 |
| `artifacts_stage1/pplnm_smoke/pplnm_calibration.json` | 第二份权重的实际配置/校准匹配测试 |
| `artifacts_stage1/pplnm_smoke/ca_calibration.json` | 小规模CA对照参数/校准 |

这两份约3.2MB权重无需额外大文件存储即可被普通Git版本管理。不能只保留150轮权重却删掉读取smoke权重的功能测试，或让远端缺少测试所需夹具。`training_state.pt` 约12.9MB，不是现成推理的必要输入；训练数据集、原始CSV、完整图像/GIF/HTML、历史日志、云端manifest及旧训练报告留作外部成果。旧历史文本可能包含未使用的摘要字段，不重新引入运行检查或绑定。

所有 `.venv*`、`results*`、ZIP/压缩包、训练状态、临时输出及本机 `DELIVERY_STAGE*.json` 默认忽略。不要使用 `git add -f` 将整个结果目录或环境打包进提交。需要保留小型正式实验摘要时，审阅后复制到文档下明确命名的证据目录，例如 `docs_stage2/evidence/<experiment>/summary.json`、`config.json` 和种子/命令记录；保留原失败状态和计量单位，并注明原始完整成果的独立存放位置。不要把原始结果目录改名伪装为源码。

## 本机路径、隐私与认证

原始运行日志和交付manifest可能带本机工作目录、用户名、解释器路径和临时目录。原始文件保持不动；面向GitHub的摘要使用仓库相对路径或明确的 `<repository>` / `<python3.12>` 占位符，注明仅做显示脱敏、数值未变，不改写实际运行事实。测试报告中的本机路径只是来源证据，不应成为CI或新用户安装的硬依赖。

不提交 `.env`、私钥、令牌、浏览器状态、凭据存储、个人运行时或代理状态。认证检查只确认CLI/凭据辅助器/连接器是否配置和是否可以进行授权的只读操作，不读取或输出Token/password。Git正常提交和推送不需要主动校验文件或对象摘要。

本次只读检查时，本机未发现可直接调用的 `gh` 命令，Git凭据辅助器配置为 `manager`，已连接的GitHub工具可完成认证用户只读查询。这只能说明当前认证途径存在，不证明本地HTTPS推送一定有权限；推送结果和PR创建结果需要实际确认。连接器可作为PR创建途径，但不能据“已经连接”声称代码已推送。

## 测试与研究失败的处理

提交前运行原PVS、阶段1和阶段2全部pytest，并执行依赖检查。单元/功能测试失败、导入异常、数值非有限或缺少必要夹具是代码问题，应先修复再推送；不得删测试或跳过失败测试制造通过。

研究场景中的碰撞、超时、应急滑行或规划延迟超预算是需要保留的实验结果。只要程序、评价和测试合同正确，可以将这些真实失败随代码与报告提交到研究分支；不能把研究失败改成成功，也不能放宽碰撞标准。涉及算法比较时应使用同一REMUS动力学、初始化、观测和评价，P25共享场景只是不同模型的工程对照。

建议流程：

1. 查看工作区改动和文件清单，确认旧成果与原模型保护范围，确保提交包含全部必要夹具。
2. 在已授权的研究分支完成代码与文档，保存共同配置、种子和实际结果摘要。
3. 使用独立环境运行全部测试与 `pip check`；将临时输出写入独立结果目录，记录真正执行的命令与结果。
4. 审阅将提交的文件列表和文本，确认没有本机环境、压缩包、凭据或大型原始数据；对路径仅做披露的显示脱敏。
5. 创建普通提交，推送已授权分支并创建PR；不得自动合并、强推、重置或删除用户工作。PR应写清真实REMUS闭环功能、已有失败、测试证据及后续接口。
6. 检查远端CI结果。CI失败时修复并以新提交更新PR；需要合并时由用户决定。

## GitHub CPU持续集成

`.github/workflows/tests.yml` 为pull request、main/master及research分支运行Python3.12 CPU测试，分别使用Windows和Ubuntu。先安装CPU版PyTorch，再安装 `requirements-stage1-learning.txt`，确保已有权重加载/预测测试真正执行。核心依赖清单继续复用，原锁文件不变。

CI首先加载两份必要推理夹具，然后执行：

```text
python -m pytest PythonVehicleSimulator/tests tests_stage1 tests_stage2 -q
```

实际workflow为pytest显式指定runner临时目录保存JUnit、测试临时文件和缓存，避免污染源码树；依赖检查与测试步骤不容许失败后假装成功。输出与测试计数保留在GitHub任务日志中。Windows/Linux的执行结果只有远端workflow实际运行后才能称为CI通过，本地Windows测试不能替代Ubuntu验证。仓库是否要求CI作为合并门槛由GitHub分支保护设置控制，该workflow不会自行修改权限或保护规则。

## 研究声明边界

本阶段实现的是原REMUS模型上的参考空间DWA闭环、有限时域候选预测和实际执行监视。没有ECP实现，没有24秒PPLNM六自由度避障，没有校准电能模型或实艇安全证明。`energy_joules=null` 应继续保留；同模型固定参考误差为0只证明数值一致与状态隔离。PR、版本说明和长期实验记录都应保留这些边界。

# 项目说明

<!-- 每次运行都会带给 agent。带尖括号占位的行由 intake/r1 按问卷回答写好（占位全部换掉才算写完）；之后每轮的 intake 和 close 只改有变化的行。 -->

## 目标与部署（问卷 A、B 段）

- 模型与精度：<待 intake 填写>（HF 名称、权重路径、权重 / KV / MTP draft 的精度；权重是否直读、禁止拷贝）
- 硬件与后端：<待 intake 填写>（GPU 型号和卡数；引擎 SGLang / vLLM / ATOM 及版本；基础镜像 tag@digest）
- 部署形态：<待 intake 填写>（agg 或 PD；TP/DP/EP/PP/DCP；MTP 的 K；kit：`repos/Infera/examples/<kit>`）
- 代码：<待 intake 填写>（`repos/Infera` 的分支或 commit；还需要的仓库）
- 负载：<待 intake 填写>（AgentX 数据集和客户端版本、强制接受长度；并发点）
- 目标口径：Y = 输入 tok/s/GPU（`input_tput_per_gpu`，含缓存命中，P+D 全部卡），X = P90 interactivity（`p90_intvty` = 1/p90 full-response ITL）；同时报告 e2e normalized interactivity（`p90_e2e_norm_intvty`）和 TTFT p90。台账的设置在 project.yaml 的 `ledger`。
- 工作点：<待 intake 填写>（例如 X ≥ 30 tok/s/user 下 Y 最大，或 C64）
- 成功标准与停止条件：<待 intake 填写>（例如本轮工作点 Y +10%；每轮 ≤ 4 天、≤ 100 节点·小时）
- 对标：<待 intake 填写>（InferenceX 哪条曲线；是否提交 InferenceX 及要遵守的规则）

## 范围与规则（问卷 C、D 段）

- 允许的改动：<待 intake 填写>（只调配置和拓扑 / router / 引擎 patch / kernel）
- 正确性门槛：<待 intake 填写>（GSM8K 题数和阈值；真实 MTP 接受率）
- 交付形态：<待 intake 填写>（配方和报告 / kit 与 patch / PR / 发布镜像）
- 集群：按 project.yaml 的 machines 用节点，操作手册见项目的 skill `cluster-<名>`；外来容器和坏节点：记录后换节点，不排查，要停别人的容器先问我。
- 不可逆和对外的操作（push、开 PR、发布、停别人的容器）一律先问我；对外文字不写集群名、机器名和 IP。

## 轮次与版本

- 一轮：intake/rN → survey/rN ∥（第 1 轮的）baseline → profile/rN → plan/rN →（experiment/rN-* × N）→ close/rN → 起草 intake/r(N+1)。任务名带轮次前缀（`experiment/r2-spf`）。
- 版本 vN = 可复现的栈（Infera commit、kit 和参数、镜像 digest 或补丁集、引擎和 router 版本、客户端版本）加它在集成里测出的曲线。v0 是基线；close/rN 集成通过才封版为 vN，否则版本不变。
- 结果文件写 `round`（轮次）和 `series`（曲线名：`v0`、`v1` 是版本，`r2-spf` 是一个实验，`ix:<条目>` 是对标）；对标的点 `role: reference`。
- 快速验证阶梯（在 baseline 标定一次，见 `workspaces/baseline/r1/calibration.json`）：L0 生效证明 → L1 microbench（不能单独作为接受依据）→ L2 机理代理（dstep 等）→ L3 AgentX fast 单点 → L4 AgentX 正式单点（工作点，接受依据）→ L5 整条曲线（封版依据，只在 close 测）。

## 实验纪律

- 先诊断再选杠杆：瓶颈可能在调度、CPU 路径、KV/内存配方、传输、路由、并行策略或 kernel，按证据决定先做哪个，不默认从 kernel 下手。配置和拓扑类先做，代码类后做。
- 一个子任务对应一个大杠杆，或一组同层、同验证方法的小杠杆；改动前写清预期机理、预期幅度和证伪条件。
- 以实际运行的配置为准（`docker inspect`、`/get_server_info`、引擎日志），不信脚本和 README；证明杠杆真的生效了再测。
- 证据标注 `[M]`（实测）或 `[I]`（推断）。
- 判定只看 AgentX 结果（`ledger agentx` 从原始输出转换的；性能以 AgentX 为准，精度以 GSM8K 为准）：固定并发下 ΔY 超过接受线（project.yaml `ledger.accept_pct`，默认 3%）和噪声底、ΔX 不劣于噪声、中位数与均值方向一致才接受；在噪声底内（`ledger.noise_pct`，默认 2%，复测离散大时更高）视为噪声；介于两者之间加跑一轮（skill infera-experiment 的 ab_judge）。机理指标、fast 模式、dstep 用来筛选和解释，不是接受条件。口径只由我改：ab_judge 只能提高接受线，验收按运行开始时的快照从原始输出复算。用 X 换 Y 的改动（提并发、改 MTP K）在等 X 下比较（台账的工作点 Δ）。预期收益低于 L4 噪声的改动，ab_judge 会判 noise：加 `--note 待集成`，留给 close 在集成后的整条曲线上判。
- 作废只有两类：失败请求 > 10%（ab_judge 从 AgentX 的请求数和错误数算）；其他原因（测量期间有别人的负载、HSA out-of-resources、被抢占……）写明原因和证据文件。作废的结果也留在 results/，verdict.status 是 void；同系列同并发的其余有效结果都要参与判定。
- 每个测量点写一个 `results/<标签>.json`（infera.result/v1，skill infera-bench）；AgentX 的 `agentx_conc*.json` 用 `"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger agentx` 转换。原始数据和命令放在 `results/<标签>/` 或 `log/`（台账只读 results/ 顶层的 *.json）。
- 读台账（`workspaces/*/*/results/`）和已完成任务的报告，不重复已经否掉的方向。
- 不动别人的容器和 GPU，不删不是自己创建的文件；长任务在远端以自包含的执行链运行，并带守护（skill infera-hygiene）；后台等待一律设上限。

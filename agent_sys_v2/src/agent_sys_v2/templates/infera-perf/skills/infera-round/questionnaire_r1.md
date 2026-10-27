# 第 1 轮需求问卷（intake/r1 的 questions.md 模板）

用法（intake/r1）：
1. 先做预调研（2–5 分钟，不占 GPU）：查权重目录、读 config.json 的量化配置、查 InferenceX 上同模型的条目和镜像、探测节点（GPU 是否空闲、有没有外来容器、RDMA）。
2. 把下表复制到工作区的 `questions.md`，"建议"列换成查到的具体值和理由；"你的意见"列留空。表单只读第一张带"你的意见"列的表，所以分组用行号前缀（A 目标与部署、B 负载与目标、C 资源、D 范围与交付），不拆成多张表。
3. 结束这一轮等我填。我只改不同意的行；没填的行按建议处理。

标 ★ 的行是必答：建议列给不出可靠的默认值，或者答错会让整轮返工（例如 IX 不许 MTP draft 用 MXFP4，整轮重测 10.5 h；一个集群的 CPU KV 层不可用，白跑一轮；中途改并发点，补测 17 h）。

| # | 问题 | 建议 | 你的意见 |
|---|---|---|---|
| A1 ★ | 模型：HF 名称和本地权重路径。权重是否直接从 NFS 读、禁止拷到节点 | 预调研列出各集群上找到的候选路径；默认直读、不拷贝 | |
| A2 ★ | 精度：权重、KV cache、MTP draft 各用什么。是否有对外规则限制，例如 IX 不允许 draft 用 MXFP4 | 按 checkpoint 的 config.json；KV 与 draft 跟 InferenceX 同配置一致 | |
| A3 | 硬件：GPU 型号和每节点卡数 | 由 C1 的节点探测得到 | |
| A4 ★ | 后端：sglang / vllm / atom，以及版本 | 选 InferenceX 上该模型、该精度、该硬件的最好 AMD 条目所用的引擎 | |
| A5 ★ | 基础镜像（tag，最好带 digest）。最终交付的镜像是否与它同一基座 | InferenceX 同条目的镜像；交付镜像与测试镜像同基座 | |
| A6 | 部署形态：agg 或 PD（xPyD）；并行方式 TP/DP/EP/PP/DCP；MTP 的 K | 抄 InferenceX 同条目的 recipe；另起一个 kit 时按 Infera 现有的同类 kit | |
| A7 | 代码基线：Infera 的分支或 commit、kit 路径；还需要哪些仓库（引擎 fork、srt-slurm、InferenceX） | main 最新；kit 由预调研给出，没有现成 kit 时写"需要新建" | |
| B1 | 负载：AgentX（数据集版本、客户端版本、强制 MTP 接受长度）还是固定 ISL/OSL | AgentX，客户端钉 IX CI 版本，强制 AL 按 IX 口径 | |
| B2 | 并发点集合 | InferenceX 同条目的点，再加预调研建议的 1–2 个点 | |
| B3 ★ | 优化目标：曲线是 Y = 输入 tok/s/GPU（含缓存命中，P+D 全部卡）对 X = 1/p90 ITL。还需要定一个工作点，用来判定 A/B | 工作点：在交互性 X ≥ ＿ tok/s/user 时让 Y 最大；没有 SLA 时取对标曲线拐点附近的并发 | |
| B4 ★ | 成功标准和停止条件：做到多少算完成；每一轮做多久或花多少 GPU 时就停下来复盘 | 本轮工作点 Y +10%，或曲线在工作点处超过对标；每轮 ≤ 4 天、≤ 100 节点·小时 | |
| B5 | 对标对象：InferenceX 哪条曲线；是否包含竞品硬件（B200/GB300）和未合入的 PR | InferenceX 同模型同精度的最好 AMD 条目，加上最好的 NVIDIA 条目 | |
| B6 | 是否要提交到 InferenceX，以及要遵守的规则（同一镜像一次 sweep、上游镜像、draft 精度……） | 不提交；对标只读 | |
| C1 ★ | 集群和节点：主机名，或 Slurm 分区和 QoS；一次最多用几台；能否独占；会不会被抢占 | 预调研列出可达的节点及其状态（GPU 空闲、外来容器）；默认单节点作业优先 | |
| C2 | 各集群的能力差异：RDMA 网卡、CPU KV 层（Mooncake Store / LMCache）是否可用；节点本地盘和日志放哪 | 预调研跑探针（RDMA、Store 读回）后填写 | |
| C3 | 遇到外来容器或坏节点怎么办 | 记录后跳过、换节点，不排查；外来容器不动，要 `docker stop` 时先问 | |
| C4 | 预算：每个任务的美元上限、每轮的 GPU 时、截止日期 | 任务 $200，plan 下的实验各 $50；GPU 时见 B4 | |
| D1 ★ | 允许改动的范围：只调配置和拓扑 / router / 引擎 patch / kernel | 全部允许；引擎改动一律走 patch，不改上游 fork | |
| D2 | 正确性门槛：GSM8K 题数和阈值；是否检查真实（不强制）的 MTP 接受率 | 按 skill infera-quality-gate（GSM8K ≥ 0.96，smoke 6/6）；每个新镜像都测真实接受率 | |
| D3 | 交付形态：只要配方和报告 / Infera kit 与 overlay patch / Infera PR / 上游 PR / 发布镜像 | 每轮收尾时再定；默认只出 kit、patch 和报告 | |
| D4 | 不可逆操作和对外文字的规则：push、开 PR、发布、停别人的容器；PR 里能不能出现机器名 | 全部先问；对外文字不写集群名、机器名、IP | |
| D5 | 汇报：语言；希望在哪些节点同步（基线出来、plan 定稿、每轮收尾） | 中文；只在这三个节点同步 | |

## intake 拿到回答后的校验（机器可查的都做成 check）

| 校验 | 方法 | 不通过时 |
|---|---|---|
| 权重路径存在，精度与 config.json 一致 | `ls`、读 config.json 的 quantization_config | 追问 A1/A2 |
| 镜像可拉取，digest 记录下来 | `docker manifest inspect` 或在节点上 `docker image inspect` | 追问 A5 |
| 节点可达，GPU 空闲，无外来容器 | ssh + `rocm-smi`/`amd-smi` + `docker ps` | 写进 C1 的备注，从 machines 里去掉 |
| kit 存在且支持所选的引擎和拓扑 | `repos/Infera/examples/<kit>` | 在 plan 前加一个 develop 任务"新建 kit"（bring-up 会变成大任务，需要你知道） |
| 对标条目存在 | InferenceX API 或数据仓库 | B5 改为"无公开对标"，baseline 只做内部基线 |
| 工作点可判定 | B3 给了 X 门槛或并发 | 追问 B3 |

## intake 的产出（写到哪里）

| 内容 | 位置 | 原因 |
|---|---|---|
| 背景、部署、负载、目标、规则（A、B、D 段） | 项目的 `rules/project.md` | 每次运行都会注入，下游不用再读 intake |
| goal、machines、ledger（主指标、工作点） | `project.yaml` | 调度器和台账要用结构化字段 |
| 集群操作手册（C 段和探针结果） | 项目的 `skills/cluster-<名>/SKILL.md` | 只按需读，不占每次注入的篇幅 |
| 一页 brief 和校验记录 | intake 工作区的 `report.md` | 供你审核（`review: true`），批准后下游才开始 |

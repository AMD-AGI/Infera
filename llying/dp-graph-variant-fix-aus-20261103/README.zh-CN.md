# AUS 上的跨机验证：decode graph variant 修法 + 后续优化（11-03 起）

接 `../HANDOFF-20261103.zh-CN.md`、`../dp-graph-variant-fix-20261103/`（修法与自包含验证包，fde1fb75）。crusoe 节点被回收后，
用户 11-03 13:20 要求在 AUS 两台节点上完成验证，再按交接的顺序做后续优化。结果与分析见 `RESULT.zh-CN.md`，时间线见 `STATUS.zh-CN.md`。

## 1. 环境

| 项 | 值 |
|---|---|
| 节点 | holder 作业 35060（`Compute-Multinode`，账户 `xiewen12`，QoS `batch`，到 11-03 22:43）：prefill `smci355-ccs-aus-n06-25`（fenic 10.235.192.59），decode `smci355-ccs-aus-n06-33`（10.235.192.60），各用 GPU 4–7；后续 holder 35103（`afterany:35060`） |
| 镜像 | 两台各自按 Infera fdbafb60 构建 `infera-sglang:v0522-fdbafb60`（基础镜像层在构建缓存里，约 7 分钟）：n06-25 `1a49bf20…`、n06-33 `2d852443…`；镜像 ID 不同，**11 个基线文件 md5 两台都与 `patches/BASE_MD5SUMS` 一致** |
| 包 | `/perf_apps/liyingli/bench_agentx/vfix-20261103` = fde1fb75 的 `llying/dp-graph-variant-fix-20261103/`，只加了 `config/site.local.env`（本目录 `config/site.local.env.aus`） |
| 日志 | `/perf_apps/liyingli/vfix-logs/`：`verify.out`、`verify/`（每次部署）、`ref/`（参照组）、`opt/`（优化实验）、`patrol/`（巡检） |
| 站点值 | `MODEL=/perf_apps/data/models/GLM-5.2-MXFP4`、`MOONCAKE_DISABLE_HIP_DMABUF=1`、`MC_GID_INDEX=1`、RDMA ionic_0–7、`WORK_DIR=/scratch/liyingli/vfix`（prefill 节点本地） |

## 2. 与 REPRODUCE 的差异（都不改变被验证的内容）

- **不另交 `sbatch -N 2`**：两台已被 holder 作业占着（`/apps/liyingli/jobs/holder.sh`：在分配内按顺序执行 `/apps/liyingli/holder/queue/` 里的脚本）。
  验证作为队列项 `scripts/holder_item.sh` 执行，它调用包里原样的 `scripts/slurm_verify.sbatch`，只跑 `verify.out` 里还没完成的部署
  （以 `<label> down` 行为准，中断的部署重跑），所以 holder 被 requeue 或换新 holder 后能接着跑。
- **节点 IP 从 fenic 网卡取**：AUS 节点上 `getent hosts <本机名>` 给 127.0.1.1，`<名>.cs-aus.dcgpu` 又解析到另一网段（n06-33 → 10.235.24.90），
  `slurm_verify.sbatch` 的 `ip_of` 会取错；队列项用 `PREFILL_NODE_IP_OVERRIDE` / `DECODE_NODE_IP_OVERRIDE` 传 fenic 地址。
- 作业内节点之间 ssh 可用（本次分配里 pam_slurm_adopt 没有拒绝），没有用 sshshim。
- 包用的是 fde1fb75 版；分支上之后的 1264aee2 只给 `run_verify.sh` 的 preflight 加了"缺镜像时从 `IMAGE_TAR` 加载、预拉 etcd 镜像"，
  本次两台都已有镜像和 etcd，不影响结果。

## 3. 本次遇到的问题与处理

| 现象 | 原因 | 处理 |
|---|---|---|
| decode 节点上两个 `build_image.sh` 同时在跑 | 第一个 ssh 循环里，后台启动构建后 ssh 迟迟不返回，我另起了一次，原循环随后也在 n06-33 上启动了一次；两者共用 `CTX` 目录 | 两个都停掉，单独重跑一次（层缓存命中，几分钟）；镜像以 md5 核对为准 |
| FP8 参照组起不来：`No such file …/GLM-5.2-FP8-fixed/model-00001-of-00141.safetensors` | 该目录的分片是软链，指向上一级 `models--zai-org--GLM-5.2-FP8/blobs`，只挂模型目录时容器内解析不到 | 改为挂载上一级目录（`scripts/ref_run.next.sh`） |
| B 生产组的 lm-eval 在 InferenceX 克隆上等了约 15 分钟 | AUS 节点访问 GitHub 约 100–150 KB/s，`utils/aiperf` 子模块约 76 MB | 等它完成；检出存一份到 `/perf_apps/liyingli/vfix-logs/cache-InferenceX-918524ff`，新节点直接拷（`scripts/opt_item.sh`） |
| 拆栈后显存释放 | AUS 上约 1 分钟（crusoe 上 10–40 分钟） | 无需处理；每组限池门禁约 17 分钟 |

## 4. 参照组：错题是模型本身还是我们的栈（11-03 16:40 起）

修法组剩下的错题集中在少数几道题（第 12、45、119、100 题等），用户问这是模型能力还是我们的优化造成的。做法：同一批题、同样的采样参数
（temperature 0、top_p 1、max_tokens 16384、并发 64）、同一探测脚本，在去掉我们全部优化的配置上跑，逐题对比错题集合。

| 组 | 节点 / GPU | 配置 |
|---|---|---|
| `ref-mxfp4-stock` | n06-33 GPU 0–3（与验证并行，验证用 4–7） | 同镜像、同 MXFP4 权重；单机聚合 TP4；无 PD、无 DP attention、无 MTP、无 KF overlay / 调优 GEMM、无 prefill 补丁、无 HiCache；`--disable-custom-all-reduce`、`ROCM_QUICK_REDUCE_QUANTIZATION=NONE`；KV bf16、DSA triton（默认） |
| `ref-fp8-stock` | n06-25 GPU 0–3 | 同上，换官方 FP8 权重 `GLM-5.2-FP8-fixed`（704 GB）：区分 MXFP4 量化的影响 |

脚本 `scripts/ref_run.sh`：私有 IPC 命名空间 + bridge 网络（不与 kit 容器的 `--ipc host` 撞 RCCL 共享内存名），1,319 题 + 16k × 128（同 seed）。
计分步骤起栈前会等所有 GPU 显存释放，所以参照组不会与计分重叠。

## 5. 后续优化（交接第 4 项的顺序：decode 侧非 kernel 项 → 2P1D → dense FP8）

队列项 `scripts/opt_item.sh`（包副本 `opt-aus-20261103/pkg` = 验证包 + kit 的 router 挂载 / 环境入口，见 `opt/kit-router-hooks.diff`），
剩余时间不够的步骤不启动、等下一个分配。**预测写于开跑前（11-03 17:00）**：

| 步 | 配置（`opt/config/`） | 内容 | 预测 |
|---|---|---|---|
| E10 | `arm-E10-prod`：E10 原样，无修法 | C104 计分：AUS 同对的基线 | crusoe 同配置 51,374；节点间差约 3%，预计 49.8–52.9k |
| scoreB / scoreC | 验证包的 `arm-B-prod` / `arm-C-prod` | 验证队列的最后两步 | B ≥ E10 + 1%（去掉每步 `.item()`、verify 只一套 graph）；C ≥ E10 + 2%（再加 E11 通信） |
| traceC | `arm-C-prod-trace`：C + `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`（graph 内 kernel 可见，11-02 已证不改输出） | AgentX C104 1,500 s，预热后 600 s 起两次 40 步 decode profile（`opt/scripts/decode_profile_chain.sh`），`ana/dgaps.py` | 修法去掉了 `select()` 的 `.item()`：每步 GPU 空闲从 10-30 trace 的 2.8–3.0 ms 降到约 1.5–2 ms，剩下的是 DP attention 每轮的 CPU 同步与零散间隙（交接 6.5 节）；为下一项非 kernel 优化定位 |
| Cdpbal | C + router `--decode-request-weight 1000000`（fdbafb60 + 565461b3 重编的 router，单元测试全过） | C104 计分 | 相对 scoreC：吞吐 +1–3%，交互 p90 上升（10-21 1P1D C88：+2.7%、p90 +43%）；风险是 decode KV 池常满时按请求数均衡加重预分配等待 |
| 2p1d | C，2 × PP4 prefill + TP8/DP8 decode（16 卡） | 先门禁（smoke、200 题、16k × 128，真实接受率），干净才 C208 计分 | 每卡 0 到 +10%（点估计 +3–5%，相对 scoreC）；风险：KF MoE kernel 在 TP8 形状上的正确性、8 rank 通信 |

## 6. 文件

| 路径 | 内容 |
|---|---|
| `scripts/holder_item.sh` | 验证队列项（续跑、fenic IP、两台镜像核对） |
| `scripts/ref_run.sh`、`scripts/ref_run.next.sh` | 参照组（后者挂上一级模型目录，FP8 用） |
| `scripts/patrol.sh` | 登录节点上的巡检：作业状态、各日志的新行、失败关键字、长时间无进展 |
| `opt/` | 优化实验：`scripts/opt_item.sh`、`items/`、`config/`、router 改动、kit 改动 |
| `results/` | `verify.out`、判定、每次部署的探测统计与被标记样本、参照组、计分 |

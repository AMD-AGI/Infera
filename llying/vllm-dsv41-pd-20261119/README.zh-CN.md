# DeepSeek-V4.1-Flash FP4：Infera + vLLM 的 P/D 分离（AUS MI355X，2026-11-19 起，进行中）

目标：用 Infera + vLLM 在 DeepSeek-V4.1-Flash（FP4 专家）上做 P/D 分离，让 InferenceX AgentX 的 pareto 曲线超过
B300 dynamo + vLLM（`configs/nvidia-master.yaml` 的 `dsv41flash-fp4-b300-dynamo-vllm-agentic-dspark-disagg`）。
baseline 参数来自 InferenceX 的 MI355X vLLM 单节点配置（`dsv41flash-fp4-mi355x-vllm-agentic-dspark`，
recipe `benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`）。

本目录是工作目录 `/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/` 的记录：脚本、配置、覆盖文件的副本，
以及各轮结果的小文件（`results/`）。引擎 / router 的完整日志和探针逐条输出留在工作目录的 `logs/<run>/`。

## 0. 当前结论

### 0.0 11-19 22:05 更新（更正 20:45 的结论）

- **更正：关 breakable CUDA graph 不能修好 decode DP4 卡死**。42（DP4、`VLLM_USE_BREAKABLE_CUDAGRAPH=0` 即 `FULL_DECODE_ONLY`，门禁通过）
  在 AgentX 预热第 6 min 照样卡死，同一形状（新请求、`max_tokens=1`、1 + 5 个占位草稿），这次卡住的是 **DP0**（之前四次都是 DP2）。
  40b 里 V1 撑过 2,208 个合成请求只是运气，20:45 写的"与 breakable 有关"不成立（4.4 节已改）。
- 新的判断依据（4.4 节）：卡住 rank 的拷贝事件在目标前向 + 采样之后，所以 GPU 卡在目标前向里；采样 kernel 没有数据相关的循环；
  每一步都是"该请求在 decode 的第一步、也是最后一步（max_tokens=1）"，其草稿是该 slot 上一个请求留下的旧数据，golden 模式按比例"接受"它们。
  真实接受率的 30（c32）没出过事。下一步要测 DP4 + block 接受（不计分，只看卡不卡）和 eager。
- 现在的安排：36738 跑 43t（decode TP4 + Engram 放主机 + 显存 0.93 + max-num-seqs 256，扩大 TP4 的 KV 池，争取第一个正式点）；
  36739 跑 44（DP4 + K=3，同时检验 K=3 会不会改变卡死）。

### 0.0b 11-19 20:45 更新（第一条已被上面更正）

- ~~decode DP4 预热卡死有了绕法（单次）~~：decode 关 breakable CUDA graph 后合成负载 2,208 个请求没卡（原配置 1,150 个就卡）——
  42 在 AgentX 里证明这不是修复。
- **decode TP4 不是好的兜底**：每个 rank 都要拉每个请求的 KV（ZMQ socket 超过 1023 上限，已用 overlay v2 修），而且 MLA 的 KV 在 TP rank 上复制，
  总容量 3,612 万 token，这个负载（平均上下文 14.6 万）250 个请求就占满；DP4 总容量约 2 亿。
- **prefill 侧**：B300 的三项设置（显存 0.93、long-prefill 8192、max-num-seqs 64）过了正确性门禁，前缀池 2,632 万 → 4,100 万 token / rank，
  冷 prefill 吞吐不变。`kv-cache-dtype fp8` 在 ROCm 上是空操作（与默认同一格式）。
- **仍然没有正式成绩**：20:34:52 两个作业又同时收到抢占预告（第二次），39（TP4 基线）在预热第 16 min 被打断。42–44 已排队，作业回来后自动跑。

### 0.1 11-19 17:50 更新（接手 session）

- **11-19 的三个正式计分点都没有成绩**：32c（c512）、33d（c768）都在 AgentX 预热中 **decode DP4 卡死**——decode DP2 卡在
  `sample_tokens`、DP0/1/3 卡在 `execute_dummy_batch`，300 s 后 RPC 超时、decode 引擎退出，预热里的根请求失败，AgentX 按规则中止
  （第 4.4 节）。两次的形状一样：卡住那一步是 DP2 上一个刚拉完 KV 的新请求（max_tokens=1，重算 1 个 token + 5 个占位草稿）。
  35（c1024）在预热第 3 min 被作业抢占打断（17:22 抢占预告，guard 清容器）。**正式点要先解决这个卡死**。
- 静态读 vLLM（V2 runner 的 DP 同步、`skip_drafts`、DSpark 草稿路径、Mooncake 接收侧）没有找到确定的分歧点；36 起每轮都带
  py-spy 看门狗和 vLLM `--watchdog-config` 栈 dump，卡住时现场抓栈。兜底：37 把 decode 换成单引擎 TP4（没有 DP dummy batch）。
- **两个作业 17:27 / 17:28 都被抢占并自动重新排队**（`--requeue`）；n10-29 现在是 yixingx 的 24 h 作业。预计 36738 约 19:00、
  36739 约 21:56 重新拿到节点（`squeue --start`），holder 会自动接队列里的 36、37。
- **B300 为什么高**（`B300-ANALYSIS.zh-CN.md`）：tok/s/GPU 里 99% 是命中缓存的输入，≈ req/s × 14.6 万 ÷ GPU 数；固定并发下 GPU 越少、
  延迟越低越好。B300 c768 的 prefill 稳态只用了约 40%（12 卡里 4 卡 prefill），decode 8 卡每 rank 约 20 个请求、每步约 15.7 ms。
  我们的差距：冷 prefill 单卡慢约 2.9 倍（预热阶段每 2 卡引擎 70.8k 对 207k prompt tok/s）；prefill 前缀池每 rank 2,632 万 token，
  B300 4,789 万（`kv-cache-dtype fp8` + 显存 0.95）；B300 的 CPU 层几乎没贡献（0.6%）。可移植项见该文件第 4 节，
  第 4 点要求的 golden 2.99 配置在 DSv4.1 上对应 DSpark K=3 / golden 2.90。

### 0.2 待你 review 的自主决定

| # | 时间 | 决定 | 理由 / 可替代方案 |
|---|---|---|---|
| D1 | 17:30 | 不处理被抢占后自动重新排队的 36738 / 36739（不取消、不另交作业） | 你要求只用已分配的作业；重新排队是 Slurm `--requeue` 的行为，拿到节点后 holder 自动接队列 |
| D2 | 17:40 | 下一轮（36）与 33d 配置完全相同（c768 正式），只加 py-spy 看门狗和 vLLM 栈 dump | 先复现并抓到栈；不卡就是第一个正式基线 |
| D3 | 17:45 | 37 兜底：decode 换单引擎 TP4（每个 TP rank 拉整份 KV，传输量 4 倍），排在 36 后 | 结构上避开 DP dummy batch；修好 DP4 前先有正式点；也是 TP4 对 DP4 的对照 |
| D4 | 19:53 | 停掉 38（只重启 decode 的二分）：V0 卡死后 prefill 还压着那批请求的 KV 和拉取（1800 s 超时），V1 的 smoke 被拖了 30 min；改成每个变体整体重新部署（40b） | 二分结果会被污染 |
| D7 | 23:48 | 停掉 43u（TP4 + K=3，预热中），36739 改跑 43w（DP4 + EP4 + golden），之后 43x（DP4 eager 诊断） | 43t 的稳态显示 TP4 decode 饱和（在跑顶到 max-num-seqs 256、decode 排队 p50 628、每卡 2.2k tok/s），K=3 改变不了量级；修好 DP4 才有意义，EP 换了 MoE 通信方式，可能直接绕开 |
| D6 | 22:58 | 43s（DP4 + block 真实接受率，诊断）撑过整个预热、进入测量 6 min 后停掉，36739 改跑 43r（去掉 `max_tokens=1` 请求的首步补位，DP4 + golden） | 诊断目的已达到（block 不卡、golden 必卡），稳态引擎指标已采到；尽快验证能计分的 DP4 绕法比跑完一个不能计分的点更有用 |
| D5 | 20:16 | 停掉 40（TP4 + prefill B300 设置），让 36738 先跑 40b（DP4 卡死二分）；41（TP4 + K=3）挪到 `holder/parked/` | TP4 decode 的 KV 在 4 个 rank 上复制，总容量 3,612 万 token，预热里 250 个请求就占满（99.8%），吞吐会被 decode KV 卡住，prefill 设置的效果看不出来；DP4 总容量约 2 亿 token，修好 DP4 才有意义。39（TP4 基线）照跑，作为第一个正式点 |

### 0.3 11-19 16:45 的结论（保留）

- **单节点 PD + DPA + MTP 跑通（30）**：prefill DP2（GPU 0-1）+ decode DP4（GPU 4-7），每 rank TP1、MoE 在 DP rank 间张量并行（不开 EP），
  Engram 表放主机内存，DSpark 5，Mooncake（每张 GPU 绑自己的网卡）+ vLLM 修复。16k 64/64、GSM8K 63/64，decode 超时 0。
- **InferenceX AgentX 计分流程接通（30）**：InferenceX 自己的 client（`benchmarks/srt_agentic.sh`，agentx-harness 1.0.14，
  cc-traces-weka-062126）打 infera router，fast 模式并发 32 跑完 1,790 个请求、0 失败、汇总和校验通过。这一点用的是真实接受率、
  分母错成 1 卡（`IS_MULTINODE=false`），**不能和 InferenceX / B300 比**，只证明流程可用（第 3 节）。
- **并发要按吞吐目标定**：InferenceX MI355X vLLM agg 跑到每卡 32–192 并发（TP2 c64、DP2 c128、UMBP c256 / c384），B300 PD 在
  c256–c3584；30 的 c32（6 卡）每卡只有约 5，两边都很空（prefill 大多 0 运行 / 0 排队、decode 2–10 个在跑、KV < 2%）。
  之后的 PD 点都用 golden 接受长度 3.51（InferenceX 计分口径）+ 高并发，并且是**正式计分**（`AIPERF_EXPERIMENTAL_FAST=0`、
  3600 s、每条回放线 10 个预热请求，`scripts/agentx_formal.sh`）：fast 模式（1 个预热请求、1200 s）测量开头是瞬态、缓存没热，
  数字和 B300 / InferenceX 的正式点不可比，只适合调试或同为 fast 的相对比较。16:43 起：32c（8 卡 2P1D，c512，n10-29）、
  33d（8 卡 2P1D，c768，n06-25，18:23 前结束）、35（c1024，排在 32c 后）。
- **单节点 agg 跑通**（Infera router + 一个 vLLM worker，MI355X baseline 参数，TP4）：GSM8K 198/200，16k 捞针 32/32、
  长前缀 GSM8K 31/32；Rust router kv-aware 也能服务（第 3 节）。
- **单节点 P/D（Mooncake，prefill TP4 GPU 0-3 + decode TP4 GPU 4-7）能配对、KV 传输正确**，但有一部分请求永久卡住：
  decode 一直等 prefill 的"就绪"信号直到 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`。**根因是 vLLM 的 bug**（第 4 节）：
  DSv4.1 的 prefill 侧会去掉 prompt 最后一个 token 再采样 1 个 token，这个 token 是停止符时请求以 `FINISHED_STOPPED`
  结束，而 MooncakeConnector 只在 `FINISHED_LENGTH_CAPPED` 时发送 KV。
- **两种修法都已验证（单次）**：
  - vLLM 修复（`request_finished` 把 `FINISHED_STOPPED` 当作 `FINISHED_LENGTH_CAPPED`，与 NIXL 一致）+ 原版 Rust router：
    16k 64/64、GSM8K 64/64，decode 超时 0 次（21d）；单测不带修复时新用例失败、带修复时整个文件 159 passed（24）。
    已在 fork 开 PR [llying-001/vllm#3](https://github.com/llying-001/vllm/pull/3)（第 4.3 节）。
  - 不改 vLLM 的绕法：router 给 prefill 腿加 `min_tokens: 1`、`ignore_eos: true`（infera 覆盖层 `base.py`，目前只有
    Python router 生效）：16k 64/64、GSM8K 64/64，prefill 侧 128 个全部 `FINISHED_LENGTH_CAPPED`（21b）。
- **其他问题**：infera 0.3.3 把所有 GPU 的 Mooncake 网卡钉在 `ionic_0`（AUS 每张网卡一个子网）；Rust router kv-aware
  的 KV 事件链锚不上（flush 返回 404，要 `VLLM_SERVER_DEV_MODE=1`）。见第 6 节。
- **与 InferenceX 对齐的更正**：InferenceX 的 MI355X vLLM 主曲线是 **TP2**（并发 1–64，Engram 表放主机内存），TP4 只跑并发
  1、2。这里的 TP4 是 bring-up 时为少一个变量（Engram 留在显存）选的，不是主曲线配置；TP2 的 agg 参照见 `23`。

## 1. 环境

| 项 | 值 |
|---|---|
| 集群 / 节点 | AUS `smci355-ccs-aus-n06-25`（作业 36738，回填 6 h 40 min，11-19 11:43–18:23）、`smci355-ccs-aus-n10-29`（作业 36739，24 h，11-19 15:42 起），都是 8 × MI355X、amdgpu 6.16.13、加载了 `ib_peer_mem` |
| 镜像 | `vllm/vllm-openai-rocm:nightly-9c1d45c21836370104d4eb51d3b46eda20719c47@sha256:694108707f33…`（11-19 06:12 UTC 构建）：vLLM `0.34.0rc1.dev897+g9c1d45c21`、torch 2.13.0+rocm10.0.0、amd-aiter 0.1.29、MoRI v1.2.3.post1、nixl-rocm 1.5.0、mooncake-transfer-engine-rocm 0.3.14、tokenizers 0.23.3、httpx 0.27.2 |
| 模型 | `/perf_apps/data/models/DeepSeek-V4.1-Flash`：`deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a`（9 月首发版，与 InferenceX 相同；vLLM recipe 的新默认 `-1017` 不是这个），`DeepseekV41ForCausalLM`，FP8 dense + FP4 专家 |
| Infera | PyPI `amd-infera==0.3.3`（InferenceX infera-* 的 router 版本）`--no-deps` 装进镜像，再覆盖 `infera-overlay/`（第 5.1 节） |
| 网卡 | 8 × ionic，每张一个 /24 子网（192.168.1–8.x），GPU k 与 ionic_k 是 PCIe 邻居（`results/00-pull-inspect/nic-gpu-map.txt`）；镜像自带的 libionic 54.0-187 可直接驱动（8 个端口 active） |
| 镜像里的 Mooncake | 支持 dma-buf（`ibv_reg_dmabuf_mr`）、`MOONCAKE_DISABLE_HIP_DMABUF`、`MC_DISABLE_HIP`、`MC_RDMA_NOTIFY_ENABLED`；没有 Mooncake#4540 / #4541（AUS 是物理机，不需要） |

坑：`~/.docker/config.json` 里 Docker Hub 的个人访问令牌过期，连公开镜像都拉不下来；`lib.sh` 的 `pull_image` 失败后改用空的
`DOCKER_CONFIG` 匿名重拉（36 s 拉完 27.8 GB）。

## 2. 部署方式

- **作业**：`holder/holder.sh`（单节点、独占、执行 `holder/queue/` 里的工作项，按名字顺序一个接一个；多个 holder 作业可共用
  一个队列，`running/<item>.job` 记录归属）。提交：`sbatch holder/holder.sh`（24 h）或 `sbatch --time=<短时长> holder/holder.sh`
  去回填空档。工作项副本在 `holder/items/`。
- **容器**：每个工作项在节点上起 `llying-dsv41-etcd`（etcd）和一个引擎容器 `llying-dsv41-eng`（所有 GPU / RDMA 设备，
  宿主机数字 gid，`nofile` 1048576），router 与各 worker 都在引擎容器里用 `docker exec` 起；guard 步骤在作业被抢占 / 结束时删容器。
- **脚本**（`scripts/`）：`env.sh`（镜像、模型、端口）、`lib.sh`（拉镜像、起容器、`pd_deploy`、`router_up`、`probe_set`、
  `rdma_mr_mode`、`build_gem`、节点状态记录）、`ctr_install_infera.sh`（容器内装 infera + 覆盖层，并干解析一条 DSv4.1 命令行）、
  `ctr_router.sh`、`ctr_worker.sh`（`python -m infera.engine.vllm`）、`gsm8k_probe.py`、`longctx_probe.py`。
- **配置**（`configs/`）：

| 文件 | 内容 |
|---|---|
| `agg_tp4.sh` | MI355X baseline 基础参数 + 其 TP4 c4–c64 档（Engram 留显存，capture 1024，chunk 16384），`--monokernel`、DSpark 5、MXFP4 indexer、block 128、Rust 前端；另加三个稳定性环境变量（keep-alive 120 s、渲染超时 600 s、faulthandler） |
| `agg_tp2.sh` | InferenceX MI355X 主曲线布局：TP2，Engram `cpu_offload: true` |
| `pd_mooncake_common.sh` | Mooncake P/D 公共部分：`MC_DISABLE_HIP=1`（同节点也走 RDMA 回环）、`MC_MAX_MR_SIZE=1 GiB`、`MC_RDMA_NOTIFY_ENABLED=0`、`VLLM_HOST_IP`、每个 worker 一个 bootstrap 端口；按 `MR_MODE` 选 peer memory 或 GEM；`MC_TOPO=pergpu` 时 GPU k 只用 ionic_k |
| `pd1n_mc_prefill_tp4.sh` / `pd1n_mc_decode_tp4.sh` | 单节点 P/D 两条腿（decode 关前缀缓存和 KV 事件） |
| `pd1n_prefill_tp4.sh` / `pd1n_decode_tp4.sh` | MoRIIO 版（未跑，停放） |
| `pd1n_dpa_prefill_dp2.sh` | DPA 的 prefill：DP2、每 rank TP1、MoE 在 rank 间张量并行（不开 EP）、Engram 主机内存、关 `--monokernel`；`GPUS`、`PORT`、`BOOT_PORT`、`DP_RPC_PORT`、`DP_MASTER_PORT`、`SCAN_PORT` 可覆盖（同一容器里多个 DP 引擎要错开端口） |
| `pd1n_dpa_decode_dp4.sh` | DPA 的 decode：DP4（GPU 4-7），关前缀缓存和 KV 事件，`INFERA_VLLM_DP_ENGINE_LB=1`（vLLM 内部 DP 均衡）；`SPEC_GOLDEN=1` 时 DSpark 用 golden 接受长度 3.51 |
| `pd1n_tp2_prefill.sh` | DP2 prefill 起不来时的退路：TP2 prefill |
| `agg_tp2_golden.sh` | `agg_tp2.sh` + golden 接受长度 3.51 |
| `argtools.sh` | `args_set` / `args_drop`：改 `VLLM_ARGS` |

- **AgentX**：`scripts/agentx_run.sh OUT_DIR`（`agentx_fast.sh` 是 30 用过的旧版，保留不动）：在单独的容器里跑 InferenceX 的
  `benchmarks/srt_agentic.sh`，指向 infera router；`FAST=1` 即 workflow 的 agentx-fast（每条回放线 1 个预热请求、测 1200 s）。
  PD 用 `MODE=pd`（`IS_MULTINODE=true`，InferenceX 才按 `PREFILL_NUM_WORKERS × PREFILL_TP + DECODE_TP` 算 tok/s/GPU），agg 用 `MODE=agg`。
  trace 数据集从共享盘 HF 缓存拷到 `hf-cache/`。计分口径：decode 的 DSpark 用 golden 接受长度 3.51（InferenceX
  `infx/srt_slurm/synthetic_acceptance.py`、`infx/golden_al_distribution/deepseek-v4.1-flash__dspark__…yaml` thinking_on 5 drafts），
  prefill 保持 block；golden 接受下输出文本没有意义，只做连通性 smoke。

- **显存注册方式**（`lib.sh` `rdma_mr_mode`，按 amdgpu 版本）：6.16 上 peer memory（`ibv_reg_mr`）只记一次账、dma-buf 记两次，
  用 peer memory（`MOONCAKE_DISABLE_HIP_DMABUF=1`）；6.19 上两条路都记两次，KV 池用 amdgpu GEM BO（`gem/hip_gem_alloc.c`，
  `infera.engine.vllm.gem_mooncake`，经 dma-buf 注册、只记一次）；6.16 节点没有 peer memory 模块时也用 GEM。

## 3. 结果（都是单次）

正确性探针：GSM8K 前 200 / 64 道（temperature 0）；16k 长上下文 64 个（32 捞针 + 32 长前缀 GSM8K，每个请求独立的随机填充）。
吞吐粗测：`vllm bench serve` 随机 8k 输入 / 1k 输出（ignore_eos），经 router，4 卡；不是 AgentX 指标。

| 轮 | 配置 | 正确性 | 吞吐粗测（总 tok/s，TTFT / TPOT ms） | 备注 |
|---|---|---|---|---|
| 10 agg TP4 | `agg_tp4.sh`，Python router round-robin | GSM8K 198/200（1 道撞 8192 上限）；16k 捞针 32/32、GSM8K 31/32；Rust kv-aware GSM8K 31/32 | c4 10,068（202 / 3.24）；c16 20,333（159 / 6.56） | 起栈 1,424 s（含 NFS 预读 275 s）；KV 91.8 GiB / 3,654 万 token；DSpark 接受长度 2.87（随机数据下偏低） |
| 20 P/D TP4+TP4 | Mooncake，peer memory，Rust router round-robin | GSM8K 195/200（2 道卡住超时）；16k 完成的 38 个全对，16 个被中止、10 个卡住 | c4 10,348（429 / 2.73）；c16 20,924（377 / 5.93） | 起栈 1,090 s；所有 worker 用 `ionic_0`；Rust kv-aware 时 GSM8K 又有 2/64 卡住 |
| 21 A | 同 20 + prefill DEBUG + 拉取超时 120 s，Rust router 并发下发 | 16k 64 个错 29；GSM8K 64/64 | — | prefill 侧 99 `LENGTH_CAPPED` / 29 `STOPPED`，29 个 `STOPPED` 即 29 个失败请求 |
| 21 B | 同上，Python router 串行下发 | 16k 64 个错 28；GSM8K 60/64（错 4） | — | 串行下发照样卡，排除并发竞态；A+B 195 `LENGTH_CAPPED` / 61 `STOPPED` |
| 21b | 同 21，Python router 并发 + prefill 腿 `min_tokens: 1`、`ignore_eos: true` | 16k 64/64 无错（捞针 32/32、GSM8K 31/32）；GSM8K 64 道无错 | — | prefill 侧 128 个全部 `LENGTH_CAPPED`，decode 超时 0 |
| 21d | 同 21，vLLM 修复挂进容器 + 原版 Rust router 并发 | 16k 64/64 无错（捞针 32/32、GSM8K 32/32）；GSM8K 64 道无错 | — | prefill 侧 104 `LENGTH_CAPPED` / 24 `STOPPED`，decode 超时 0 |
| 21c / 22b | vLLM 修复的单测 | 无效：21c 镜像缺测试依赖 `tblib`；22b 命令里的引号写错 | — | 修好后为 24 |
| 22 | 同 20，`MC_TOPO=pergpu`（GPU k 只用 ionic_k），无卡死修复 | 16k 64 个错 26（卡死，预期内）；GSM8K 62/64 | c16 20,432（525 / 5.74） | 8 张卡都在用，同节点跨子网建连 0 错误；网卡流量计数没读出来 |
| 23 agg TP2 | `agg_tp2.sh`（InferenceX 主曲线布局），n10-29 | GSM8K 196/200；16k 捞针 32/32、GSM8K 31/32；Rust kv-aware 31/32 | 2 卡：c4 8,498（267 / 3.77）；c16 13,859（362 / 9.03） | 起栈 1,483 s；KV 75.3 GiB / 2,912 万 token；接受长度 2.53 |
| 24 | vLLM 修复的单测（测试依赖按 `requirements/test/rocm.txt` 固定） | 不带修复：新用例失败，整个文件 1 failed / 158 passed；带修复：159 passed | — | |
| 30 PD+DPA+MTP | prefill DP2（GPU 0-1）+ decode DP4（GPU 4-7），`MC_TOPO=pergpu`，vLLM 修复，Rust router | 16k 64/64（捞针 32/32、GSM8K 31/32）；GSM8K 63/64；decode 超时 0 | 6 卡：c16 18,999（693 / 6.19）；c64 34,078（6,280 / 9.54） | 起栈 450 s（页缓存热）；KV prefill 2,632 万 / rank、decode 5,077 万 / rank；接受长度 2.70 |
| 30 AgentX fast c32 | 同上，Rust kv-aware + session affinity，真实接受率 | 1,790 个请求 0 失败 | 总 193,040 tok/s（含命中缓存的输入），输出 1,422 tok/s；按 6 卡约 32,200 / 237 每卡；交互度 p50 238，TTFT p50 0.32 s | 只验证流程：真实接受率、分母错成 1 卡 |

| 32c 2P1D golden c512 正式 | prefill DP2 × 2 + decode DP4，golden 3.51，n10-29 | — | 无成绩 | 预热第 13 min decode 卡死（4.4 节），AgentX 中止（`0 successful / 1108 total`） |
| 33d 2P1D golden c768 正式 | 同上，n06-25 | — | 无成绩 | 预热第 7 min decode 卡死，同一形状 |
| 35 2P1D golden c1024 正式 | 同上，n10-29 | — | 无成绩 | 预热第 3 min 作业被抢占（17:22 预告、17:28 结束），容器被 guard 清掉（agentx rc=137） |

| 36 2P1D golden c768 正式 + 抓栈 | = 33d + py-spy 看门狗 + vLLM 栈 dump，n01-29 | — | 无成绩 | 预热第 10 min decode 卡死，第三次同一形状；抓到栈（4.4 节） |
| 37 2P1D decode TP4 golden c768 正式 | decode 单引擎 TP4，n01-29 | — | 无成绩 | 预热一开始 decode 拉 KV 全部 `Too many open files`（ZMQ 每 context 1023 socket 上限，4.4 节），根请求失败 |

| 38 DP4 卡死二分（合成负载） | 一次部署、只重启 decode | — | — | V0 1,150 个请求后卡死（复现）；之后的变体被 prefill 残留状态拖住，作废 |
| 39 2P1D decode TP4 + socket 修复，golden c768 正式 | n02-21 | 门禁通过：GSM8K 63/64、16k 捞针 32/32、长前缀 31/32、0 错误，真实接受长度 3.64（P TP1 → D TP4 的 KV 传输正确） | 无成绩 | 预热第 16 min（20:34:52）作业收到抢占预告，容器被清（agentx rc=137） |
| 40 = 39 + prefill B300 设置（0.93 / long-prefill 8192 / max-num-seqs 64） | n01-29 | 门禁通过：GSM8K 63/64、捞针 32/32、长前缀 31/32，接受长度 3.73；prefill KV 4,100 万 / rank（原 2,632 万） | 无成绩 | 我在预热第 10 min 停掉（D5）：decode TP4 的 KV 总容量只有 3,612 万 token，预热中已占 99.8%；冷预热 prefill 吞吐与 36 相同（71.1k 对 73.6k / 引擎） |
| 40b DP4 卡死二分 v2（每变体重新部署） | n01-29 | — | — | V1（关 breakable cudagraph）2,208 个请求不卡；20:34:52 抢占预告中断 |

| 42 DP4 关 breakable + prefill B300 设置，golden c768 | n02-29 | 门禁通过（GSM8K 63/64、捞针 32/32、长前缀 31/32、接受长度 3.65） | 无成绩 | 预热第 6 min 卡死（DP0），关 breakable 不是修复 |
| 44 = 42 + DSpark K=3，golden 2.90 | n02-25 | 门禁通过（接受长度 3.09） | 无成绩 | 预热第 9 min 卡死 |
| 43s DP4 + block 真实接受率（诊断） | n02-25 | — | 不计分 | **整个预热 8,495 个请求不卡**；测量开头 4 min：decode 14.2k tok/s、在跑 207、每用户 68 tok/s；22:58 停掉（D6） |
| 43r DP4 golden + 去掉 `max_tokens=1` 首步补位（scheduler overlay） | n02-25 | 门禁通过（3.72） | 无成绩 | 预热第 13 min 卡死，四个 rank 都在 dummy batch 里——绕法无效 |
| **43t decode TP4（Engram 主机、显存 0.93、max-num-seqs 256）+ prefill B300 设置，golden 3.51，c768 正式** | n04-33，8 卡 | 门禁通过（GSM8K 63/64、捞针 32/32、长前缀 31/32，接受长度 3.68） | **141,900 tok/s/GPU**（输出 1,101）；交互度 p50 / p90 36.8 / 32.6；ITL p50 / p90 27.2 / 30.7 ms；TTFT p50 / p90 / 均值 70.3 / 81.0 / 63.5 s；9.19 req/s | **第一个正式成绩（单次）**。decode 饱和：在跑顶到 256、decode 排队 p50 628、KV p50 57% / p90 96%；服务端命中 97.5%（理论 96.4%） |
| 43u = 43t + K=3 | n02-25 | — | 无成绩 | 预热中停掉（D7） |
| 43w DP4 + EP4，golden c768 | n02-25 | 门禁通过（GSM8K 63/64、捞针 32/32、长前缀 32/32，3.71） | 无成绩 | 预热第 20 min 卡死，EP 也不行 |

被替换掉的：31（agg TP2 AgentX，对目标没有直接意义）、32 / 33 / 33b（c32、c192，并发太低）、32b / 33c / 34（fast 模式，不能用于对比）。
（00:40）在跑：43x（DP4 eager + golden，诊断）、43y（DP4 + block 真实接受率的完整参考运行，不计分）。
以下是 20:45 时的排队说明（已过时）：42（DP4 关 breakable + prefill B300 设置）、43（DP4 关 breakable，prefill 原设置）、
44（42 + DSpark K=3 / golden 2.90），都先过真实接受率的正确性门禁。

## 4. P/D 卡死的根因与修复

### 4.1 现象

decode 侧 `pulling kv_caches for [...] failed: Timeout waiting for P side ready.`，prefill 侧同一请求
`Timeout waiting for P side ready`，四个 TP rank 一起；同一时刻 prefill 引擎空闲（运行 0、排队 0），两边没有传输错误；
卡住比例随 prompt 而变：16k 捞针最多，普通 GSM8K 少，随机 token 的 bench 为 0。证据摘录 `results/20-pd1n-mooncake-1119T1215/hang-excerpt.txt`。

### 4.2 机制（vLLM nightly `9c1d45c2`，main 上相同）

1. DSv4.1 是 SWA bounded replay，P/D 下 prefill 侧 `_truncate_mamba_request_for_prefill` 去掉 prompt 最后一个 token、
   `max_tokens` 设为 1，只算到倒数第二个位置；最后那个 token 由 decode 重算（`_p_side_truncated`）。
2. prefill 在倒数第二个位置采样的 token 是模型对被去掉那个 token 的预测，会被丢弃；但 `check_stop`
   （`vllm/v1/core/sched/utils.py`）先查 EOS 和 `stop_token_ids`、再查 `max_tokens`，所以这个 token 是停止符时请求以
   `FINISHED_STOPPED` 结束。
3. `MooncakeConnectorScheduler.request_finished` 只在 `FINISHED_LENGTH_CAPPED` 时登记待发送；其他状态记进
   `_reqs_not_processed`，worker 侧只删掉占位，不通知 decode。decode 已经发出的拉取一直等占位的 `ready`，直到
   `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`（这里设 1800 s / 120 s）。
4. NIXL 的 pull / push scheduler 已经把 `FINISHED_STOPPED` 和 `FINISHED_LENGTH_CAPPED` 同样处理；MoRIIO 与 Mooncake 一样只认
   `FINISHED_LENGTH_CAPPED`（未验证，暂不改）。

### 4.3 修复

- **vLLM**（给上游）：`/perf_apps/liyingli/bench_agentx/upstream/vllm-main`，分支 `fix/mooncake-send-kv-on-stopped-prefill`，
  提交 `[Bugfix][KV Connector] Mooncake: send KV for prefill requests that finish STOPPED`（基于 main `6bc4849aa`，DCO 签名）：
  `request_finished` 接受 `FINISHED_STOPPED`；`tests/v1/kv_connector/unit/test_mooncake_connector.py` 的
  `test_scheduler_request_finished` 加 `STOPPED` 用例。端到端 21d 通过；单测 22b 结果出来后推到 `llying-001/vllm` 开 PR。
  验证用的挂载文件 `vllm-overlay/mooncake_connector.py`（与 nightly 只差这一处）。
- **Infera**（不改 vLLM 时的绕法）：prefill 腿请求体加 `min_tokens: 1`、`ignore_eos: true`（`infera-overlay/infera/router/disagg_protocols/base.py`
  的 `prefill_single_token`）。Python router 已验证（21b）；Rust router 的 prefill 请求体在二进制里（`annotate_vllm_prefill`），
  要改源码重编才能生效。

### 4.4 decode DP4 在 AgentX 预热中卡死（32c / 33d，待定位）

- **现象**（两次相同）：AgentX 正式模式的预热请求都是 `max_tokens=1`。decode 侧大量请求在等 KV（33d 卡前 424 个、32c 658 个
  "KV fetch in progress"，因为 prefill 排队 300+），某一步之后 decode 再也不出 step：日志每 60 s 一条 `No available shared memory
  broadcast block found`，300 s 后 DP2 报 `RPC call to sample_tokens timed out`、DP0/1/3 报 `RPC call to execute_dummy_batch timed
  out`，引擎退出，Rust 前端 `LLM became permanently unhealthy`，AgentX 的根请求失败 → `ProfileAborted`。
- **卡住那一步**（DP2 的 `dump_input`）：一个新请求，prompt 23,142（32c）/ 153,990（33d）token，`num_computed_tokens = prompt - 1`
  （P 侧截掉最后一个 token、decode 重算），调度 6 个 token（1 + 5 个 `-1` 占位草稿），`max_tokens=1`；同一时刻其他 rank 跑 dummy batch。
  两次都是 DP2（单次巧合的概率 1/4）。日志里没有 `WorkerProc hit an exception`（vLLM#41530 / #60928 那一类 rank 异常后错位）。
- **已排除**（读 nightly `9c1d45c2` 源码）：`skip_drafts` 跨 DP 投票（dummy rank 投 0，所以真实 rank 不会跳过草稿）；调度器每步固定
  5 个草稿名额（无动态 K）；DSpark 草稿 pass 在真实 / dummy 两边都做一次 DP 同步和一次 query forward；这个新请求被判成
  "一个 token 续算按 decode 跑"（`prefill_runs_as_decode`），图模式与 dummy 一致；Mooncake 接收侧全异步、不在接收线程里做 GPU 同步。
  vLLM#64392（DSv4.1-Flash + DSpark 5 同签名）里说的修复 `40941f2c`（breakable cudagraph 改由 compilation config 控制）已经在我们的镜像里。
- **36 现场栈（第三次复现，18:32:53 卡住，`logs/36-*/pyspy/1-1/`、`vllm-stackdump/`）**：
  - EngineCore_DP2 在 `step_with_batch_queue` → `_wait_for_output` 等 `sample_tokens` 的回复；
  - **Worker_DP2 主线程空闲**（`worker_busy_loop` 在等下一条 RPC），它的 `WorkerAsyncOutputCopy` 线程卡在
    `AsyncOutput.get_output`（等本步采样结果的 GPU→CPU 拷贝事件）；
  - DP0/1/3 的 worker 主线程都在**下一个** dummy batch 的 `sync_cudagraph_and_dp_padding` → `cpu_all_reduce_` 里等 DP2；
  - decode 四张卡 GPU 占用 100%、显存读写 0%（kernel 在空转等对端）。
  - 推断：拷贝事件记录在目标前向 + 采样之后、DSpark propose 之前，所以 **DP2 的 GPU 卡在本步目标前向里的某个跨 DP 集合通信**，
    对端 dummy 前向发起的 GPU 集合通信与它不一致（次数或实现不同）；CPU 侧的 DP 同步点是对齐的，只是 DP2 的引擎要等这步结果
    才发下一条命令。三次都是 DP2、都是"刚拉完 KV 的新请求、max_tokens=1、1 + 5 占位草稿"的那一步（36：prompt 111,887）。
- **37（decode TP4 兜底）也没跑成，原因不同**：预热一开始 decode 每个 TP rank 都报 `pulling kv_caches ... failed: transfer failed:
  Too many open files`（1,536 次）。TP4 下每个 rank 都要为每个请求拉 KV，MooncakeConnector 每个待回复的拉取占一个 ZMQ DEALER socket，
  815 个并发拉取超过 libzmq 每个 context 默认 1023 个 socket；DP4 每个 rank 只承担约 1/4，没碰到。修法：
  `vllm-overlay/mooncake_connector_v2.py` 在创建 context 后设 `zmq.MAX_SOCKETS`（默认 65536，`VLLM_MOONCAKE_ZMQ_MAX_SOCKETS`），
  `scripts/pd_run_v2.sh` 默认挂它。
- **二分（38 / 40b，合成负载：随机 12k–36k 输入、`max_tokens=1`、并发 640、2500 个请求，模拟 AgentX 预热）**：
  - 38 V0（原样）：负载开始 3 min、1,150 个请求后卡死，签名与 AgentX 预热完全相同（又是 DP2）——**合成负载能复现**，8.5 min 内出结果。
  - 38 只重启 decode 的做法不成立：卡死那轮的请求在 prefill 侧继续占着 KV 和 Mooncake 发送状态（1800 s 超时），V1 的 smoke 被拖了 30 min。
    40b 改成每个变体整体重新部署。
  - 40b V1（decode `VLLM_USE_BREAKABLE_CUDAGRAPH=0`，vLLM 因模型不支持 torch.compile 自动回落到 `FULL_DECODE_ONLY`，与 B300 decode 同模式）：
    **2,208 个请求没有卡**，第 2,263 个时 20:34:52 作业收到抢占预告、容器被清（这一轮"DONE"是假阳性，实际没跑完）。
    前四次卡死分别在 390–1,150 个请求之间。~~结论：卡死与 breakable CUDA graph 有关~~（**更正**：42 关掉 breakable 后在 AgentX
    预热第 6 min 照样卡，见下条）；V2 / V3 / V5 没来得及跑（`holder/parked/40b-*`）。
- **42（DP4 关 breakable，`FULL_DECODE_ONLY`）照样卡**：21:53:43 起 decode 不再出 step，Worker_DP0 空闲、DP1/2/3 在下一步的 DP 同步里等，
  DP0 的 `dump_input` 又是那一种步（prompt 83,278、`num_computed = prompt - 1`、6 个 token、5 个 `-1` 占位草稿、`max_tokens=1`、
  KV 拉取在途 137）。所以与 breakable 无关，也不是 rank 2 特有（这次是 DP0）。
- 排除过的（读源码）：采样相关 GPU kernel 没有数据相关的循环（不会在采样里死循环）；DSv4.1 的 SWA 重放子批次在 DP 上有 `_agree_across_dp`，
  dummy 路径同样调用 `prepare_attn`；重放 CUDA 图（只在 breakable 下存在）的 dispatch 与本 rank 请求数无关；Engram 没开 `embedding_across_dp`。
- **44（DP4 关 breakable、DSpark K=3、golden 2.90）也卡**（22:09，Worker_DP1 卡在 `CUDAGraph.replay`），K 无关。
- **43s（DP4 原配置 + block 真实接受率，诊断）整个预热 8,495 个请求、0 错误、不卡**，22:51:48 进入测量：这是 DP4 decode 第一次撑过 AgentX 预热。
  golden 下 5 次全卡（32c、33d、36、42、44），block 下不卡（43s，单次；30 的 c32 也是真实接受率）——**诱因是 golden 合成接受**（单次判断）。
- 查过的具体机制：这一步是调度器专门为 P/D 首步做的补位（`pad_spec_decode`：只需重算 1 个 token 的新请求被补成 1 + num_spec 行以保住
  FULL 图），worker 里新请求的草稿槽清零；但 `_gather_draft_sampled_kernel` 已把"从 prefill 内部开始的一步"的草稿位标成 -1，
  合成校验遇到 -1 会停，所以不是"golden 接受了补位 0"。拒绝后的恢复采样也没有循环。确切的错位点仍未定到行。
- **绕法验证（43r）**：调度器 overlay（`vllm-overlay/sched/scheduler.py`，`VLLM_PD_PAD_FIRST_STEP_MIN_REMAINING=2`）对只能再出 1 个
  token 的请求不做首步补位，其余请求照旧补位；只影响预热里的 `max_tokens=1` 请求。

## 5. Infera 侧改动与从 `dev/pd_opt/vllm_glm52_agentx` 移植的经验

### 5.1 覆盖层（`infera-overlay/`，装在 0.3.3 之上）

| 文件 | 来源 | 作用 |
|---|---|---|
| `infera/engine/vllm/args.py` | vllm_glm52_agentx `b4f1e9cdf` | vLLM ≥ 0.34 把 `make_arg_parser` 挪到 `entrypoints/common`，0.3.3 在 nightly 上起不来；`INFERA_VLLM_DP_ENGINE_LB` |
| `infera/engine/vllm/gem_mooncake.py` | 同上 | KV 池放 amdgpu GEM BO 的 MooncakeConnector（6.19 节点） |
| `infera/router/disagg_protocols/vllm_mooncake.py` | 同上 | `INFERA_VLLM_MOONCAKE_SERIAL=1`：Python router 串行下发 |
| `infera/router/disagg_protocols/base.py` | 0.3.3 + 本次修改 | prefill 腿 `min_tokens: 1`、`ignore_eos: true`（第 4.3 节） |

### 5.2 移植了的设置

- 镜像没有 `render` 组：`docker run` 用宿主机数字 gid；router 所在容器 `nofile` 1048576（U14）。
- 引擎 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`（Rust router 复用连接时的 502，U24）、`VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=600`（U25）、
  `PYTHONFAULTHANDLER=1`（U21）。
- kv-aware 一律用 Rust router（Python router 的 kv-aware 在事件循环里同步分词，U15）；P/D 的 kv-aware 参数与 sglang recipe 相同
  （`--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on`）。
- Mooncake：`MC_DISABLE_HIP=1`、`MC_MAX_MR_SIZE=1 GiB`、`MC_RDMA_NOTIFY_ENABLED=0`（U27）、显式 `VLLM_HOST_IP`、`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`。
- 每轮前后记录驱动、peer memory 模块、`Shmem` / `Mlocked`、KFD 驱逐告警计数和显存（U23、m12）。
- infera 默认会给 vLLM 加 `--kv-cache-dtype fp8_e4m3`；MI355X baseline 用 vLLM 默认（DSv4.1 为 `fp8_ds_mla`），所以设
  `INFERA_DEFAULT_KV_FP8=0`。

### 5.3 以后要用的（优化阶段）

- CPU KV offload：ROCm 上 torch 锁页分配器按 2 的幂取整（U26），要用精确锁页覆盖层（按 v0.34.0 写的，需对 nightly 核对）。
- prefill 侧重排（缓存感知准入）要配串行下发（U4 / U16）；Rust router 的串行路径不在 0.3.3 里。
- decode：显存目标别太高（0.95 崩过）、`--max-num-batched-tokens` 别压得太小（1024 崩过）；DP decode 试 `INFERA_VLLM_DP_ENGINE_LB=1`；
  DP decode 的 MoE AG/RS 在 FULL graph 下走不到 aiter 路径（U13）。
- renderer 线程（`--renderer-num-workers 4`）：DSv4.1 的 prompt 在 Python 里编码，长会话上可能也是瓶颈。

### 5.4 不适用

crusoe 特有的问题（spur 调度器、NVMe 卷遮住 docker 数据目录、宿主机 libionic 要 glibc 2.38；AUS 宿主机 glibc 2.35 与镜像一致）、
Mooncake#4540 / #4541（物理机不需要）、GLM 专用的 indexer kernel 移植（#47897）。

## 6. 已知问题与下一步

1. **Mooncake 网卡全挤在 `ionic_0`**：infera 0.3.3 的 `apply_mooncake_topology_default` 在网卡跨多个子网时把所有 GPU 钉到一张卡
   （为避免跨子网 QP 建连失败）。每个 worker 写自己的拓扑（GPU k → ionic_k）的验证在 22；跨节点时 rank k 对 rank k 同子网。
2. **Rust router kv-aware 退化成只看负载**：KV 事件链没有锚点，router 想 `reset_prefix_cache` 被拒（404），prefill 要
   `VLLM_SERVER_DEV_MODE=1`（21 起已加）。另外 DSv4.1 没有 chat template（vLLM 用 `deepseek_v41` tokenizer mode 在 Python 里编码），
   router 侧的哈希要与引擎一致才能命中，待查。
3. 镜像 httpx 0.27.2 低于 infera 要求的 0.28（目前 etcd 注册、router 正常）。
4. 同一容器里两个 worker 的 infera 就绪探针端口撞了（`30090`，尽力而为，无影响）。
5. 下一步（第一目标是推高吞吐；11-19 17:50 起的顺序，前一版在下面）：
   0. （20:45）42：DP4 关 breakable + prefill B300 设置的正式点；43：prefill 原设置的对照；44：DSpark K=3。之后 1P1D（6 卡计分）、
      prefill CPU KV 层、两边 renderer 线程；breakable cudagraph + DP 的根因（V2 / V3 / V5 变体）和上游修复。
   1. 解决 decode DP4 预热卡死（4.4 节）：36 抓栈，37 TP4 兜底；拿到第一个正式点。
   2. prefill 侧移植 B300 的设置（`B300-ANALYSIS.zh-CN.md` 第 4 节）：显存 0.93–0.95、`kv-cache-dtype fp8`、`long-prefill-token-threshold 8192`、
      `max-num-seqs 64`，先用真实接受率过正确性，再正式计分。
   3. DSpark K=3（golden 2.90）对 K=5（3.51）。
   4. 1P1D（6 卡计分）对 2P1D。
   5. prefill CPU KV 层（GLM5.2 的写穿 OffloadingConnector + 精确锁页）、两边 `--renderer-num-workers 4`。
   以下是 16:45 的版本：
   - 看 32b / 33c / 34（2P1D，c512 / c768 / c1024）哪边先饱和：prefill 的排队、实际计算吞吐、前缀缓存命中率，decode 的在跑数、ITL、KV；
   - 按结果调 prefill : decode 比例（B300：prefill 卡 : decode 卡 = 4 : 8 @ c256 / c768，6 : 8 @ c1280，8 : 8 @ c2048，10 : 8 @ c3584）；
   - 高并发下 prefill 前缀缓存不够时开主机内存 KV 卸载（B300 prefill 用 SimpleCPUOffloadConnector lazy；ROCm 锁页取整见 5.3）；
   - decode 往 DP8 整节点走，prefill 放另一台（跨节点 P/D，rank k 对 rank k 同 rail）；EP 先过正确性门禁；
   - Rust router 的 prefill 腿修复（若 vLLM 修复合入前就要计分）；kv-aware 的 DSv4.1 prompt 哈希一致性。

## 7. 时间线（UTC）

- 10:21 提交 holder（xiewen12，预计 22:58）；10:3x 换 emad 账户（fairshare 0.28 对 0.075）。
- 11:24 预期的节点被 guanchen 的依赖作业拿走；11:31 另交 6 h 40 min 回填作业 36738（填 zirzhang 21:56 预约前的空档）。
- 11:43 拿到 n06-25；三个工作项因 Docker Hub 令牌过期失败；11:47 改匿名拉取后重排。
- 12:13 agg TP4 就绪并通过正确性检查；12:34 单节点 P/D 就绪；13:04–14:04 长上下文探针卡住。
- 14:35 21 A 段：29 个 `FINISHED_STOPPED` 对应 29 个卡住的请求；14:39 B 段（串行下发）照样卡。
- 14:5x vLLM 修复（本地提交）；15:00 21b（router 绕法）通过；15:19 21d（vLLM 修复）通过。
- 15:30 记录推到 `dev/yihou/dsv41`；15:42 24 h 作业 36739 提前拿到 n10-29（启动时停掉了 yihou 残留的 3 个占显存的容器，只停不删，
  记录在 `/apps/liyingli/holder/foreign_containers.log`）；15:45 22（每 GPU 一张网卡）可用。
- 15:56 单测 24 通过（159 passed）；推送 `fix/mooncake-send-kv-on-stopped-prefill` 到 `llying-001/vllm`，开 PR #3。
- 16:03 30（PD + DPA + MTP）就绪并通过正确性检查；16:06–16:32 AgentX fast c32 跑通（只作流程验证）。
- 16:07 / 16:16 在 30 执行 `agentx_fast.sh` 期间改了这个文件（bash 边读边执行，收尾会读错位置），16:18 恢复原样，改进版另存为
  `agentx_run.sh`。
- 16:19–16:36 按"推高吞吐"重排：停掉 31（agg）和 32 / 33 / 33b（c32、c192），改跑 32b（c512，n10-29）、33c（c768，n06-25），34（c1024）排队。
- 16:42–16:43 fast 改正式计分：停掉 32b / 33c，改跑 32c（正式 c512，n10-29）、33d（正式 c768，n06-25），35（正式 c1024）排队。
- 16:58:57 33d decode 卡死，17:03:49 RPC 超时退出；17:04 35 在 n06-25 因剩余时间不够放回队列。
- 17:04:57 32c decode 卡死，17:09:54 退出；17:10 35 在 n10-29 开始，17:19 进入预热。
- 17:20 接手 session 开始；17:22:42 两台都收到抢占预告（容器被 guard 清掉，35 的 agentx rc=137），17:27:31 / 17:28:01 36738 / 36739
  被抢占并重新排队（`Requeue=1`）。
- 17:30–17:50 读 vLLM 源码定位卡死（4.4 节）；下载 B300 计分 run 的结果和 c768 引擎日志做对比（`B300-ANALYSIS.zh-CN.md`）；
  排 36（抓栈）、37（decode TP4 兜底）；登录节点起监控（`scripts/monitor.sh`）。
- 18:05 36738 在 n01-29 重新开始；18:22 36 开始 AgentX，18:32:53 decode 卡死，18:33:57 看门狗抓栈，18:37 结束。
- 18:45 37（TP4）开始 AgentX，18:47 起 decode 拉 KV 全部 `Too many open files`，18:54 中止；写 overlay v2（ZMQ MAX_SOCKETS）。
- 18:55 38（只重启 decode 的二分）开始；19:11 V0 卡死（复现）；19:17–19:47 V1 的 smoke 被 prefill 残留状态拖住；19:53 停掉 38（D4）。
- 19:53 36739 在 n02-21 重新开始，接 39（TP4 基线）；36738 接 40（TP4 + prefill B300 设置）。20:01 / 20:11 两者门禁通过。
- 20:16 停掉 40（D5），36738 改跑 40b（每变体重新部署的二分）；20:24–20:34 V1（关 breakable）2,208 个请求不卡。
- 20:34:52 两个作业同时收到抢占预告（容器被 guard 清掉），39 / 40b 中断；排 42–44（DP4 关 breakable）。

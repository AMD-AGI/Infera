# DeepSeek-V4.1-Flash FP4：Infera + vLLM 的 P/D 分离（AUS MI355X，2026-11-19 起，进行中）

目标：用 Infera + vLLM 在 DeepSeek-V4.1-Flash（FP4 专家）上做 P/D 分离，让 InferenceX AgentX 的 pareto 曲线超过
B300 dynamo + vLLM（`configs/nvidia-master.yaml` 的 `dsv41flash-fp4-b300-dynamo-vllm-agentic-dspark-disagg`）。
baseline 参数来自 InferenceX 的 MI355X vLLM 单节点配置（`dsv41flash-fp4-mi355x-vllm-agentic-dspark`，
recipe `benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/mi355x-fp4-mtp/agentic.yaml`）。

本目录是工作目录 `/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/` 的记录：脚本、配置、覆盖文件的副本，
以及各轮结果的小文件（`results/`）。引擎 / router 的完整日志和探针逐条输出留在工作目录的 `logs/<run>/`。

## 0. 当前结论（11-19 15:30）

- **单节点 agg 跑通**（Infera router + 一个 vLLM worker，MI355X baseline 参数，TP4）：GSM8K 198/200，16k 捞针 32/32、
  长前缀 GSM8K 31/32；Rust router kv-aware 也能服务（第 3 节）。
- **单节点 P/D（Mooncake，prefill TP4 GPU 0-3 + decode TP4 GPU 4-7）能配对、KV 传输正确**，但有一部分请求永久卡住：
  decode 一直等 prefill 的"就绪"信号直到 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`。**根因是 vLLM 的 bug**（第 4 节）：
  DSv4.1 的 prefill 侧会去掉 prompt 最后一个 token 再采样 1 个 token，这个 token 是停止符时请求以 `FINISHED_STOPPED`
  结束，而 MooncakeConnector 只在 `FINISHED_LENGTH_CAPPED` 时发送 KV。
- **两种修法都已验证（单次）**：
  - vLLM 修复（`request_finished` 把 `FINISHED_STOPPED` 当作 `FINISHED_LENGTH_CAPPED`，与 NIXL 一致）+ 原版 Rust router：
    16k 64/64、GSM8K 64/64，decode 超时 0 次（21d）。分支 `fix/mooncake-send-kv-on-stopped-prefill`（第 4.3 节）。
  - 不改 vLLM 的绕法：router 给 prefill 腿加 `min_tokens: 1`、`ignore_eos: true`（infera 覆盖层 `base.py`，目前只有
    Python router 生效）：16k 64/64、GSM8K 64/64，prefill 侧 128 个全部 `FINISHED_LENGTH_CAPPED`（21b）。
- **其他问题**：infera 0.3.3 把所有 GPU 的 Mooncake 网卡钉在 `ionic_0`（AUS 每张网卡一个子网）；Rust router kv-aware
  的 KV 事件链锚不上（flush 返回 404，要 `VLLM_SERVER_DEV_MODE=1`）。见第 6 节。
- **与 InferenceX 对齐的更正**：InferenceX 的 MI355X vLLM 主曲线是 **TP2**（并发 1–64，Engram 表放主机内存），TP4 只跑并发
  1、2。这里的 TP4 是 bring-up 时为少一个变量（Engram 留在显存）选的，不是主曲线配置；TP2 的 agg 参照见 `23`。

## 1. 环境

| 项 | 值 |
|---|---|
| 集群 / 节点 | AUS `smci355-ccs-aus-n06-25`（作业 36738，回填 6 h 40 min），8 × MI355X，amdgpu 6.16.13，加载了 `ib_peer_mem` |
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
| 21c | vLLM 修复的单测 | 无效：镜像缺测试依赖 `tblib`，conftest 加载失败 | — | 补依赖后重排为 22b |

进行中 / 排队：22（`MC_TOPO=pergpu`，同节点跨子网能否建连、流量是否分散）、22b（单测重跑）、23（agg TP2 参照）。

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
5. 下一步：TP2 agg 参照（23）→ 并行度按 2 卡单元重配 P/D（对标 B300：prefill DEP2 × N + decode DEP8）→ 跨节点 P/D →
   InferenceX AgentX 计分；Rust router 的 prefill 腿修复（若 vLLM 修复合入前就要计分）。

## 7. 时间线（UTC）

- 10:21 提交 holder（xiewen12，预计 22:58）；10:3x 换 emad 账户（fairshare 0.28 对 0.075）。
- 11:24 预期的节点被 guanchen 的依赖作业拿走；11:31 另交 6 h 40 min 回填作业 36738（填 zirzhang 21:56 预约前的空档）。
- 11:43 拿到 n06-25；三个工作项因 Docker Hub 令牌过期失败；11:47 改匿名拉取后重排。
- 12:13 agg TP4 就绪并通过正确性检查；12:34 单节点 P/D 就绪；13:04–14:04 长上下文探针卡住。
- 14:35 21 A 段：29 个 `FINISHED_STOPPED` 对应 29 个卡住的请求；14:39 B 段（串行下发）照样卡。
- 14:5x vLLM 修复（本地提交）；15:00 21b（router 绕法）通过；15:19 21d（vLLM 修复）通过。

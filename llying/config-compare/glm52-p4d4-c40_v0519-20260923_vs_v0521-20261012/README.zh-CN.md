# GLM-5.2-MXFP4 同节点 1P1D P4D4 @ AgentX C40：v0.5.19（0923，21,032/chip）vs v0.5.21（1012，22,114/chip）配置差异

对比对象（均在本仓库 `dev/pd_opt/glm_5.2_agentx` 分支，`ccd4862b`）：

- **基线 A**：`yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923`（sglang v0.5.19，2026-09-23，`crsuse2-m2m-137`）
- **新实验 B**：`yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012`（sglang v0.5.21，headline 轮次 **e2g A2**，2026-10-11，`crsuse2-m2m-102`）

所有数值均直接取自两个 pack-up 内的原始文件（server_args、AgentX JSON、dryrun、日志），来源见文末。`server_args_diff.csv`、`metrics.csv`、`harness.diff` 由 `extract_compare.py` 从两个 pack-up 自动生成，可随时重跑。

---

## 0. 结论（TL;DR）

1. **引擎参数层面，真正有意义的差异只有 3 个 + 1 个新版本默认值**：
   - prefill HiCache：`--hicache-ratio 1.5`（≈153.81 GB/rank）→ `--hicache-size 450`（450 GB/rank，配合 `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`）；
   - decode `mem_fraction_static`：0.85 → **0.95**（decode KV 2,226,368 → 2,762,176 tokens/rank，+24.1 %）；
   - `index_share_for_mtp_iteration`：**true → false**；
   - v0.5.21 新增的 gfx950 fused DSA indexer 默认开启（v0.5.19 没有这个开关）。
   其余（TP4/DP4/DPA、triton DSA 双腿、EAGLE 5/1/6、模拟接受长度 3.61、fp8 KV、page 64、max-running 64、chunk 32768、custom-all-reduce 关闭等）**完全一致**。
2. **镜像整体升级**：sglang 0.5.19.dev20260917 → 0.5.21.dev20261008，aiter 0.1.21 → 0.1.23，Mooncake 加入 PR #4340（HIP IPC 子分配重定位修复），GLM-5.2 DSA 补丁按 v0.5.21 重切；torch / ROCm / triton 版本不变。另有 3 个运行时挂载补丁。
3. **测量协议不同，这是 22k 能成立的关键前提**：基线是**新起栈的第一次** AgentX 跑分；22,114 是**同一个栈跑完一轮完整 AgentX 之后、`/flush_cache` 两腿、再跑的第二次**（"warm-stack 协议"）。在 v0.5.21 上用同样的目标配置，新栈第一次跑只有 **19,640/chip（−6.6 %）**。
4. **"新栈 vs 新栈"比较时，v0.5.21 并不比 v0.5.19 快**：用接近基线的参数（e1c：ratio 1.5、decode 0.85）是 21,049/chip（+0.1 %）。22k 的增益**不能单独归因于镜像升级或调优参数**：两边都没有在相同协议下的对照（v0.5.19 没有 warm 跑，v0.5.21 的 e1c 配置也没有 warm 跑）。

---

## 1. 两次实验一览

| 项目 | A：v0.5.19 基线（0923） | B：v0.5.21（1012，e2g A2） |
|---|---|---|
| 吞吐 | 168,260 tok/s = **21,032 tok/s/chip** | 176,909 tok/s = **22,114 tok/s/chip**（+5.1 %） |
| 节点 | `crsuse2-m2m-137`（plain ssh） | `crsuse2-m2m-102`（spur 作业 202546→207495，spur-authz） |
| 拓扑 | 同节点 1P1D P4D4：prefill GPU0-3，decode GPU4-7 | 相同 |
| 并发 / 模式 | AgentX full，CONC 40，3600 s，warmup 10/lane | 相同 |
| 栈状态 | **新栈第一次跑** | **同栈第二次跑**（第一次 A1 不计分，之后 flush 两腿） |
| 跑分时间 | 2026-09-23，profiling 08:48:34 起 | 2026-10-11，bench 07:16:30→08:35:29（profiling 07:30:53→08:31:33） |
| 结论 | `AgentX passed`，4157 ok / 4 err-dropped | `AgentX passed`，4286 ok / 1 err-dropped |

---

## 2. 镜像与软件栈

| 组件 | A：v0.5.19 | B：v0.5.21 |
|---|---|---|
| 运行镜像 | `infera-sglang:v0519-yihou-0917-nextnfix-hicache`，ID `sha256:fd7220a57b7d…` | `infera-sglang:v0521-yihou-1010-mc4340`，ID `sha256:9d755dd6a7bf…` |
| 上游基础镜像 | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`（digest `21c1cc9a…`） | `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008`（digest `6c2c0b0e…`） |
| sglang | `0.5.19.dev20260917+ga9fb1c3238` | `0.5.21.dev20261008+g11b1fb21de`（树 HEAD `9f5d8712`） |
| torch / HIP | `2.9.1+rocm7.2.0.git7e1940d4`，HIP 7.2.26015 | **相同** |
| triton | 3.7.0（`+amd.rocm7.2.0.git89002410`） | 3.7.0 |
| aiter | `0.1.21.dev48+g4ad998328` | `0.1.23.dev183+ge7d2453f2` |
| Mooncake engine.so | faae8dd4 重编，sha256 `f22b1364…` | faae8dd4 + **PR #4340 port**（HIP IPC 子分配重定位修复），sha256 `3c96489a…` |
| Infera 源码 | 仓库 HEAD `2e5a2d89`（dirty） | `28121cb4` + 4 个补丁提交（构建于 `68f06e29`） |
| 镜像内补丁 | Infera DSA 补丁集（7 个）+ GLM NextN shared-experts fusion 修复 + sglang PR #37152（HiCache ROCm copy 加宽，薄层打入） | DSA 补丁按 v0.5.21 重切（01 + dp_sync + 重切 page_table_rows + NextN fusion 修复；draft_cuda_graph_dp_vote 已上游 #32196 故移除）；prefill 非最终 chunk wait-event 补丁；PD rejection-sampling 防护；#37152 已在上游（993d1fcc）；router 冲突修复（PD DP-rank affinity 适配 `pick_targets`/sessions API） |
| 运行时挂载（`:ro` 覆盖 sglang 源文件） | 无 | 3 个（见 §4.2） |

镜像中 `SGLANG_SET_CPU_AFFINITY=1` 两边都有（基础镜像 ENV），调度进程按本地 gpu_id 绑核，两边行为相同。

---

## 3. 引擎参数（实测 server_args）

来源：A 为运行中 `/get_server_info`（`results/server-info.29001.json` / `29257.json`）；B 为 e2g 栈启动日志中的 `server_args=` 字典（`logs/e2g/*.log.gz`）。完整逐字段差异见 `server_args_diff.csv`。

### 3.1 有实际意义的差异

| 参数 | 腿 | A：v0.5.19 | B：v0.5.21 | 说明 |
|---|---|---|---|---|
| `hicache_size` | prefill | 0（按 `hicache_ratio 1.5` 定容） | **450**（GB/rank） | v0.5.21 中 size>0 覆盖 ratio |
| HiCache host pool（日志） | prefill | 3,423,488 tokens，**153.81 GB/rank** | 10,016,064 tokens，**450.00 GB/rank**（×2.93） | |
| `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` | prefill（env） | 未设置 | **512** | 让 450 GB 只做一次 `hipHostRegister`。默认 256 GiB 会拆成两段注册，而 JIT kernel 只翻译基址，访问第二段（slot ≥ 6,118,144）时 MAF（H1）。该缺陷 v0.5.19 也有，但 ratio 1.5 时池 < 256 GiB 不会触发 |
| `mem_fraction_static` | decode | 0.85 | **0.95** | |
| decode KV 容量（`max_total_num_tokens`） | decode | 2,226,368 tokens/rank | **2,762,176 tokens/rank**（+24.1 %） | 初始化后剩余显存 39.67 GB → 10.89 GB |
| prefill KV 容量 | prefill | 2,282,304 tokens/rank | 2,270,656 tokens/rank | mem 0.85 不变，差异来自版本 |
| `json_model_override_args` | 两腿 | `{"index_share_for_mtp_iteration":true}` | `{"index_share_for_mtp_iteration":false}` | B 因 v0.5.21 在 C44 下的 MAF 历史（H2）关闭；B 的 e2d 轮验证 true 在 C40 无吞吐收益 |
| gfx950 fused DSA indexer（`enable_dsa_fused_indexer`） | 两腿 | 不存在 | `null` = 自动开启（v0.5.21 默认） | B 的 e2b4（关）vs e2c（开）在 C44 差 +0.9 %，单次样本不显著 |

### 3.2 字段名/默认值变化（无实际影响）

| 参数 | A | B | 说明 |
|---|---|---|---|
| `dp_size` / `attn_dp_size` | `dp_size=4` | `dp_size=1, attn_dp_size=4` | v0.5.21 把 `--dp-size 4 --enable-dp-attention` 解析为 `attn_dp_size`（`server_args.py:404-411`），拓扑不变；harness 的校验已相应修改 |
| `cuda_graph_config` | — | 多出 `max_seq_len: null` | bs 列表完全相同 |
| `file_storage_path` | `"sglang_storage"` | `null` | 未使用 storage backend |
| v0.5.21 新增字段（52 个，多为默认值） | — | 如 `disaggregation_decode_allocation_policy="early"`、`boundary_reduction="rs+rsv"`、`dsa_triton_union=0`、`hicache_host_memory_fraction`（decode 0.8，未启用 HiCache）、`disable_aiter_allreduce_fusion_in_{prefill,decode}=false` 等 | 全部是 v0.5.21 的默认值，命令行未设置 |

### 3.3 完全相同的关键参数（两腿均一致）

`tp_size 4`、`ep_size 1`、`enable_dp_attention True`、`dsa_prefill_backend triton`、`dsa_decode_backend triton`、`dsa_topk_backend sgl-kernel`、`attention_backend dsa`、`kv_cache_dtype fp8_e4m3`、`page_size 64`、`max_running_requests 64`（每 DP 16）、`chunked_prefill_size 8192`（命令行 32768 ÷ DP4）、`max_prefill_tokens 16384`、`schedule_policy fcfs`、`enable_aiter_allreduce_fusion True`、`enable_fused_qk_norm_rope True`、`disable_shared_experts_fusion False`、`moe_runner_backend auto`、`disaggregation_transfer_backend mooncake`、`disaggregation_ib_device {"0":"ionic_0",…,"3":"ionic_3"}`、`watchdog_timeout 3600`、`reasoning_parser glm45`、`tool_call_parser glm47`；
prefill：`enable_hierarchical_cache True`、`hicache_ratio 1.5`、`write_through`、`io_backend kernel`、`mem_layout page_first`、`mem_fraction_static 0.85`、`load_balance_method follow_bootstrap_room`；
decode：`speculative_algorithm EAGLE`、`num_steps 5`、`eagle_topk 1`、`num_draft_tokens 6`、`disable_custom_all_reduce True`、`disable_radix_cache True`、`enable_hierarchical_cache False`、decode cuda-graph bs 列表相同、`load_balance_method round_robin`。
decode 模拟接受：`SGLANG_SIMULATE_ACC_LEN=3.61`、`METHOD=match-expected`、`TOKEN_MODE=real-draft-token`（两边相同，输出文本按设计是乱码，正确性不在比较范围内）。

---

## 4. 容器运行环境

### 4.1 环境变量

两边 `engine.sh` 注入的公共 env 完全一致：`SGLANG_USE_AITER=1`、`SGLANG_OPT_USE_TOPK_V2=false`、`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0`、`AITER_USE_FLYDSL_MOE_SORTING=1`、`SAFETENSORS_FAST_GPU=1`、`HIP_FORCE_DEV_KERNARG=1`、`PYTHONHASHSEED=0`、`SGLANG_DP_USE_GATHERV=1`、`HSA_NO_SCRATCH_RECLAIM`（prefill 0 / decode 1）、`NCCL_IB_DISABLE=1`、`NCCL_IGNORE_CPU_AFFINITY=1`、`MC_GID_INDEX=1`、`MC_DISABLE_HIP_TRANSPORT=1`（在当前 Mooncake 中是无效变量名）、`MC_ENABLE_DEST_DEVICE_AFFINITY=1`、`MOONCAKE_DISABLE_HIP_DMABUF=0`、`RDMAV_FORK_SAFE=1`、`SGLANG_TIMEOUT_KEEP_ALIVE=900`、`SGLANG_DISAGGREGATION_{BOOTSTRAP,WAITING}_TIMEOUT=1800`、`SGLANG_ENABLE_FAILED_SESSION_PROBE=1`、`MC_LOG_LEVEL=INFO`。

差异：

| env / 选项 | A：v0.5.19 | B：v0.5.21 |
|---|---|---|
| `MC_TE_FILTERS` | `ionic_0,ionic_1,ionic_2,ionic_3` | `yihou_no_such_rdma_device`（故意不匹配 → 0 HCA） |
| `NCCL_SOCKET_IFNAME` | 未设置 | `ens3` |
| `NCCL_DEBUG` | 未设置 | decode：`INFO` |
| `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` | 未设置 | prefill：`512` |
| `INFERA_NODEPORT_RANGE` | `30000-32767` | `43000-45767` |
| decode PID 偏移 | 无 | 启动前在容器内消耗 30000 个 PID（`DECODE_PID_OFFSET=30000`），使 decode rank PID 与 prefill 不重叠 |
| `TMPDIR` / triton / HF / XDG 缓存 | 容器私有（triton 缓存在 `/root/.cache`，每次起栈冷编译） | `TMPDIR` 为 8 GB tmpfs；`TRITON_CACHE_DIR`/`HF_HOME`/`XDG_CACHE_HOME` 放在 NFS `$WS/cache/102/<kind>/<role>`（跨起栈持久） |
| AITER JIT 缓存 | 节点本地 `/tmp/aiter-jit-yihou-sn-{prefill,decode}/<image_id>` | NFS `$WS/cache/102/aiter/{prefill,decode}/<image_id>` |
| `PYTHONDONTWRITEBYTECODE` | 未设置 | `1` |
| 额外挂载 | 模型 ro、AITER 缓存、libionic | 另加 `$WS` rw 和 3 个 sglang 源文件覆盖 |

两边都**没有**设置 `HSA_ENABLE_IPC_MODE_LEGACY`、`--pid host`、ROCm debug agent（B 的 e2g 中 `IPC_MODE_LEGACY=0`、`ENGINE_PID_HOST=0`、`DEBUG_AGENT=0`）。

### 4.2 B 的 3 个运行时挂载（e2g 实际生效）

| 挂载文件 → 目标（`sglang/srt/…`） | 作用 | 原因 |
|---|---|---|
| `kv_events_publisher.yihou-noblock.py` → `managers/scheduler_components/kv_events_publisher.py` | KV metrics ZMQ PUSH 改为 NOBLOCK，EAGAIN 时丢弃 | v0.5.21 上阻塞 send 会卡死 prefill scheduler（R001） |
| `forward_mla_rocm.yihou-noabsorbcfg.py` → `models/deepseek_common/attention_forward_methods/forward_mla_rocm.py` | `_absorb_bmm_config` 置 None | v0.5.21 自带的 M=256 配置让 triton LLVM 后端 abort（R003） |
| `conn.yihou-regbase.py` → `disaggregation/mooncake/conn.py` | 每个注册区域打印 `YIHOU-REGBASE … interior=` | 纯诊断，行为无变化 |

B 的 `patches/runtime-mounts/experimental/` 中的其它补丁（xferfix、seedguard、host-variant 等）**不在** 22k 配置里。

---

## 5. KV 传输

| | A | B |
|---|---|---|
| Mooncake 拓扑发现 | `Found 1 HCAs` + `HIP transport installed` | `Found 0 HCAs` + `HIP transport installed` |
| 实际 KV 路径 | **HIP IPC over XGMI**（同主机时 Mooncake 段为 `"rdma,hip"`，优先 hip；基线 task spec 中实测约 57 GB/s） | **HIP IPC over XGMI**（RDMA 被过滤掉） |
| Mooncake 版本 | faae8dd4 | faae8dd4 + PR #4340：修复 HIP IPC handle 只导出分配基址、内部子区域注册被错误重定位的问题（R018/R019，B 的 regbase 日志显示所有运行 `interior=0`） |

即两边 KV 实际都走 XGMI，差别在于 B 显式关闭了 RDMA 设备发现，并带上 #4340 修复。

---

## 6. 内存规划

| | A | B |
|---|---|---|
| prefill HBM KV | 2,282,304 tokens/rank | 2,270,656 tokens/rank |
| prefill HiCache host pool | 153.81 GB/rank × 4 ≈ 615 GB | 450 GB/rank × 4 = 1.8 TB |
| prefill DSA indexer host 区 | 35.25 GB/rank | 103.13 GB/rank（随 pool 等比放大） |
| prefill host 锁页总量 | ≈ 756 GB | ≈ 2.21 TB（节点 Shmem 2.2 TB） |
| decode HBM KV | 2,226,368 tokens/rank | 2,762,176 tokens/rank |
| decode 初始化后剩余显存 | 39.67 GB | 10.89 GB |
| 全部 8 rank KV 池（AgentX 报告） | 18,034,688 tokens | 20,131,328 tokens |

B 的运维副作用：停掉 HiCache-450 栈后 VRAM 回收需要 28-35 分钟。

---

## 7. 硬件与节点

| | A：`crsuse2-m2m-137` | B：`crsuse2-m2m-102` |
|---|---|---|
| GPU | 8 × MI355X（gfx950），288 GiB | 相同 |
| CPU / 内存 | 2 × EPYC 9575F，236 逻辑核，2.7 TiB | 相同 |
| OS / 内核 / amdgpu 驱动 | Ubuntu 24.04.4 / `6.8.0-107-generic` / DKMS 6.14.14 | 相同 |
| Docker | 29.6.1 | 29.7.2（spur 管理的 dockerd + spur-authz，禁止 `--pid=host`） |
| 访问方式 | plain ssh | spur exec ssh shim |
| 数据面 IP | 10.245.153.247 | 10.245.147.130 |

同型号节点，但不是同一台物理机。

---

## 8. Harness 与部署流程

基础 `bench-harness/config.sh` **逐字节相同**；`tools/topology.py` 相同。差异（完整 diff 见 `harness.diff`）：

| 文件 | B 相对 A 的改动 |
|---|---|
| 配置组织 | A：单个 overlay `config.yihou.triton.sh` → `config.sh`；B：`env.e2g-102` → `config.yihou.mc4340.sh` → `config.yihou.E2-target.sh` → `config.yihou.v0521-base.sh` → `config.sh` 链式 overlay，节点/轮次参数化 |
| `engine.sh` | 镜像 ID 强校验（`EXPECTED_IMAGE_ID`）；DRY_RUN 渲染；按节点/角色划分缓存与 tmpfs；`NCCL_SOCKET_IFNAME`；`ENGINE_PID_HOST`/`IPC_MODE_LEGACY`/`DEBUG_AGENT` 开关；`DECODE_PID_OFFSET`；按腿 cpuset / 绑核开关（默认空）；`YIHOU_MOUNTS` 源文件覆盖；按角色额外挂载 |
| `launch.sh` | 同节点启动门失败时停掉已启动的腿（不删除容器）；router 容器加 `$WS` 挂载与 tmpfs |
| `stop.sh` | 只停名字含 `yihou` 且前缀匹配的容器；保留已停止容器与日志 |
| `agentx_bench.sh` | 支持 `NODE_RW_ROOT`；客户端容器不再 `--rm`，记录退出码；客户端缓存路径重定向 |
| `tools/agentx_env.py` | 新增 hicache_size / fused indexer / chunk GB / pid_host / ipc_legacy 校验；适配 `attn_dp_size` |
| `tools/wait_healthy.py` | `docker inspect` 超时视为暂态并重试 |

起栈结果：A 中 decode 首次启动死于同节点 RCCL 竞争（`NCCL error: unhandled cuda error`），约 9 分钟后手工重启才成功。B 定位到根因（两个私有 PID 命名空间中 prefill/decode rank 的 PID 相同，ROCr `\0hsa<pid>` 抽象 socket 与 CLR 同 PID 检查冲突），用 PID 偏移修复后 decode 一次起来（`status.tsv`：prefill 05:46:08 ok，decode 05:52:45 ok）。

---

## 9. Router / etcd / 端口

Router 参数完全相同：`python3 -m infera.server --router-backend rust --discovery-backend etcd --request-transport http --kv-event-transport zmq --router-policy kv-aware --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0`，`INFERA_PD_DP_RANK_AFFINITY=true`。不同点是 router 二进制来自各自的镜像（B 含 router 冲突修复补丁 0001）。etcd 都是 `quay.io/coreos/etcd:v3.5.14`。

| 端口 | A | B |
|---|---|---|
| etcd client / peer | 22379 / 22380 | 24501 / 24502 |
| router | 28000 | 28100 |
| prefill / decode engine | 29001 / 29257（stride 256） | 29101 / 29357（stride 256） |
| bootstrap | 28998 | 28198 |
| kv-event / snapshot | 25557 / 28801 | 25657 / 28901 |

B 把所有固定端口都放在 30000 以下，避开节点的临时端口范围 32768-60999（R015：e2b 中 decode 端口被 prefill 的临时 socket 占用）。

---

## 10. 压测客户端（AgentX）

完全一致：InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`（未修改）、数据集 `semianalysisai/cc-traces-weka-062126`（393 条 trace）、aiperf `--scenario inferencex-agentx-mvp`、`--concurrency 40 --benchmark-duration 3600 --warmup-requests-per-lane 10 --random-seed 42 --failed-request-threshold 0.10`、`TOTAL_CPU_DRAM_GB=2752`、`KV_OFFLOAD_BACKEND=hicache`、`SPEC_DECODING=mtp`、`SIMULATE_ACC_LEN=3.61`。`runtime.env` 除 IP/端口/路径/镜像外逐行相同。

两次运行的负载可比：平均输入 145.3k vs 148.7k tokens/请求（+2.4 %），理论前缀命中 97.48 % vs 97.55 %。

---

## 11. 测量协议（最关键的差异）

| | A | B（e2g） |
|---|---|---|
| 流程 | 起栈 → AgentX C40 一次 → 计分 | 起栈 → **A1**：AgentX C40 完整一轮（不计分）→ `POST /flush_cache`（先 decode 后 prefill，缓存清空，首轮 `#cached-token 0`）→ **A2**：AgentX C40 → 计分 |
| 计分轮次之前 decode 已执行的 step 数 | 0 | 约 76k（A1） |

B 发现新栈第一次跑存在 decode step 的变慢：在相同负载下慢 7-12 %，kineto 定位到每 step 约 9 ms GPU 全空闲的跨 stream 依赖等待（"gap A"），机理**未解决**。`/flush_cache` 不能消除这个"warm"状态；完全重启会把它重置掉。B 所有 v0.5.21 运行：

| 运行 | 栈状态 | 配置 | tok/s/chip |
|---|---|---|---|
| e1c | 新栈 | ratio 1.5、decode 0.85、fused indexer 关、index_share false | 21,049 |
| e2d | 新栈 | 目标配置 + index_share **true**（+ seedguard + decode debug agent） | 21,383（唯一没出现新栈变慢的新栈运行，原因未知） |
| e2g A1 / e2i R1 / e2j R1 | 新栈 | 目标配置（e2i 加 decode debug agent；e2j 加 host-variant + cuda-graph bs 1..16） | 19,640 / 20,371 / 19,335 |
| e2c C40 / e2g A2 / e2i R2 / e2j R2 | warm（第二次） | 目标配置（e2i/e2j 同上） | 21,696 / **22,114** / 21,921 / 21,681（均值 21,853） |

---

## 12. 结果指标对比

| 指标 | A 基线（新栈） | B e1c（新栈，宽松配置） | B e2g A1（新栈，目标配置） | **B e2g A2（warm，22k）** |
|---|---|---|---|---|
| total tok/s | 168,260 | 168,389 | 157,121 | **176,909** |
| tok/s/chip | 21,032 | 21,049 | 19,640 | **22,114** |
| input / output tok/s | 167,124 / 1,136 | 167,176 / 1,213 | 155,974 / 1,146 | 175,652 / 1,257 |
| 成功请求 / error-dropped | 4157 / 4 | 4174 / 2 | 3993 / 4 | 4286 / 1 |
| TTFT mean / p50 / p95 (s) | 6.17 / 3.60 / 20.10 | 5.14 / 2.85 / 17.62 | 7.61 / 3.94 / 28.70 | 5.30 / 3.33 / 19.21 |
| TPOT mean / p50 / p95 (ms) | 13.7 / 13.3 / 19.0 | 13.3 / 13.2 / 18.3 | 14.3 / 13.7 / 21.1 | 12.1 / 11.6 / 17.9 |
| E2E p50 / p95 (s) | 11.12 / 63.66 | 10.27 / 62.08 | 13.30 / 67.28 | 10.15 / 56.16 |
| interactivity p50 | 75.1 | 75.8 | 73.0 | 86.4 |
| server GPU / CPU cache hit | 94.22 % / 1.47 % | 94.38 % / 1.44 % | 94.09 % / 1.66 % | 92.60 % / 2.08 % |
| decode ms/step/DP、平均 running-req | 未打包 | 45.6、3.86 | 47.6、3.81 | 40.9、3.57 |
| spec accept length | 3.5625（指标 gauge 均值） | 3.608（decode 日志） | 3.608 | 3.610 |

C40 下吞吐受 decode step 耗时限制（decode 忙碌 93-96 %），A2 的 decode step 比 A1 快约 14 %，与 TPOT、交互性的改善一致。

---

## 13. 增益归因与注意事项

| 比较 | 差值 | 能说明什么 |
|---|---|---|
| B e1c（新栈）vs A（新栈） | +0.1 % | 参数接近基线时（差别只有 index_share 和 v0.5.19 没有的 fused indexer），**单换镜像到 v0.5.21 没有带来吞吐提升** |
| B e2g A1（新栈，目标配置）vs A（新栈） | −6.6 % | 新栈第一次跑时，目标配置反而更低 |
| B warm 4 次均值 vs A | +3.9 % | warm 协议下目标配置高于基线 |
| B e2g A2（最好一次）vs A | +5.1 % | headline 数字 |

注意：

1. **协议和配置/镜像的贡献没有被分离**：A 只有新栈数据，B 的宽松配置只有新栈数据。要做到同口径，需要补一次 v0.5.21 + e1c 配置的 warm 运行，或者 v0.5.19 基线配置的 warm 运行。
2. **节点不同**（137 vs 102），同型号、同内核/驱动，但节点间差异没有单独测过。
3. **两边都是模拟接受长度 3.61**，不涉及 MTP 真实接受率，也不反映输出正确性。
4. B warm 协议的单次波动只有 4 个样本（−0.8 % … +1.2 %）。
5. A 的 index_share=true 是按用户要求开启的，B 改为 false；B 在 v0.5.21 上测过 true（e2d），C40 下无吞吐收益，但它是唯一没有新栈变慢现象的新栈运行，这一点未解释。

---

## 14. 本目录文件与数据来源

| 文件 | 内容 |
|---|---|
| `README.zh-CN.md` | 本报告 |
| `server_args_diff.csv` | 两腿 ServerArgs 逐字段差异（`changed` / `only-v0519` / `only-v0521`；已排除 host/port/seed 等身份字段） |
| `metrics.csv` | A 基线与 B 的 e1c / e2d / e2g A1 / e2c C40 / e2g A2 的 AgentX 指标 |
| `harness.diff` | 两边 vendored `bench-harness/` 的 unified diff（A → B） |
| `extract_compare.py` | 从两个 pack-up 重新生成上面三个文件：`python3 extract_compare.py` |

主要来源（相对仓库根目录）：

- A：`yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923/{README.md,environment.md,REPRODUCE.md,scripts/config.yihou.triton.sh,scripts/bench-harness/,results/server-info.*.json,results/agentx_conc40.json,results/runtime.env,logs/key-excerpts.md,analysis/config_audit.yihou.md,spec/task-spec.CLAUDE.md}`
- A 镜像版本：`llying/cross-rank-rca-20260921/provenance/pre-clear/crsuse2-m2m-137/{image-id.txt,image-runtime.txt,image-inspect.json}`（同一镜像 ID `fd7220a5…`）；`yihou/glm52-agentx-c40-realacc.packup_20260918/environment.md`
- B：`yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/{README.md,environment.md,REPRODUCE.md,notes.md,patches/README.md,patches/image-recipe/,scripts/harness/{env.e2g-102,config.yihou.mc4340,config.yihou.E2-target,config.yihou.v0521-base,config.yihou.E1-relaxed}.sh,results/e2g/{dryrun.e2g-102.yihou.txt,status.tsv,A1-first-run/,A2-scored/},results/analysis/{runs_summary.csv,decode_stats.txt},results/env/,logs/e2g/*.log.gz,analysis/working_process.md}`

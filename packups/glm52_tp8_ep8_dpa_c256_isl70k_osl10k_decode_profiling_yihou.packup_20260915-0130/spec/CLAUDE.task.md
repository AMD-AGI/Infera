# Task: GLM-5.2 内部 decode bench 的 profiling 能力（实现 + 实测 + top-10 分析）

## Task description
给 `glm52_decode_internal_yihou_20260909_1057/bench/` 这套**无服务、无 Scheduler**的 GLM-5.2 decode
benchmark 加上 profiling 能力，要求 **cuda_graph 开/关两种模式都能 profile**；然后对
**TP8 / dpa=on / EP8 / conc=256** 实测，产出**排名前 10 的耗时条目**分析报告。

五步：research → 方案 → 实现 → 实测(C=256) → top-10 分析汇报。完整任务书见 `mission.md`。

## Background
- bench 工具直接驱动 `TpModelWorker` + `EAGLEWorkerV2`，自己 `mp.spawn` 出 TP ranks；
  没有 HTTP server，因此 SGLang 常规的 `/start_profile` 端点**不可用**。
- 上一轮（`../glm52_tp8ep8_dpa_c128_yihou_20260914-0423/`，已打包为
  `../packups/glm52_tp8_dpa_ep8_vs_ep1_sweep_yihou.packup_20260914-0740/`）在本机跑完了
  TP8/dpa EP8 与 EP1 各 10 个并发点，C=256 的 EP8 结果：TPOT 26.2519 ms，output 9751.66 tok/s。
- 本机 `smci355-ccs-aus-n06-25`：8×MI355X (gfx950)、ROCm 7.2、torch 2.9.1+rocm7.2.0、triton 3.7.0。

## Context — 已确认的一手事实（勿重新推导）
| 事实 | 证据来源 |
|---|---|
| 容器 `yihou-glm52-tp8ep8-0914` 仍在运行，GPU 全空闲 | `docker ps` / `rocm-smi --showuse` |
| SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a` | 容器内 `git -C /sglang rev-parse HEAD` |
| `nvtx_utils.profile_range()` 用 `record_function`，**profiler 一 active 就自动出 span，无需 env** | 读 `nvtx_utils.py` |
| `model_runner.py:1520` 已有 `step[DECODE bs=N]` span | 读源码 |
| `SGLANG_PROFILE_V2=true` 的 `ProfileManager.manual_start/stop` **raise NotImplementedError** | 读 `profile_utils.py` |
| legacy `SchedulerProfilerManager`（V2=false，默认）才支持 manual / step-count | 读 `profiler_manager.py` |
| `--enable-profile-cuda-graph` profile 的是 **capture 过程**，并打印 top-10 表 | 读 `decode_cuda_graph_runner.py:1014,1052` |
| `DEBUG_CLR_GRAPH_PACKET_CAPTURE` 存在于本机 HIP runtime | `strings /opt/rocm/lib/libamdhip64.so` |
| 容器内有 `rocprofv3` / `rocprofv2` / `rocprof-sys-*` / `rocprof-compute` | `ls /opt/rocm/bin` |

## Key references
1. `mission.md`（任务书，每 10min 注入）
2. `../packups/glm52_tp8_dpa_ep8_vs_ep1_sweep_yihou.packup_20260914-0740/`
   （`REPRODUCE.md` 有完整复现步骤、四个 host 环境坑、`--max-running-requests` 陷阱）
3. `../glm52_decode_internal_yihou_20260909_1057/bench/profile_decode.py`（被改造对象）
4. 容器内 `/sglang/python/sglang/srt/utils/profile_utils.py`、`utils/nvtx_utils.py`、
   `managers/scheduler_components/profiler_manager.py`、
   `model_executor/runner/decode_cuda_graph_runner.py`

## Core principles
1. **profile 默认关闭**。关闭时 bench 行为必须与既有 packup 逐字一致——已发布的 TPOT 口径不许动。
2. **不得为了让结果好看而调参**。ISL/OSL/accept-length/mem-fraction/max-running-requests 全部沿用。
3. **Suspend, don't conclude**：拿不到 per-kernel 归因就如实说拿不到，并指出该做哪个实验；
   绝不用 graph-off 的结果冒充 graph-on 的结果。
4. **所有临时实验活动都在本 workspace 内**；失败迭代一律 `mv` 成 `aborted_*`，**不删**。
5. **目录外永不删除**（用户级 CLAUDE.md 硬规则）。不写目标为变量的递归删除。
6. 用 docker container 执行，不直接操作 host。工作用英文，只有给用户的报告用中文。

## Other notable details
- **`--max-running-requests C` 是必须的**，不是调参：SGLang 会把它自动设成 48，
  而 `kv_cache_configurator.py:1882` 用 `max_running_requests // attn_dp_size` = `48//8` = 6 个
  request slot，小于任何 local batch，`alloc_req_slots` 直接失败。C=256 → local batch 32。
- C 必须能被 dp_size=8 整除；`--batch-size` 是**全局**并发，per-rank 是 C/8。
- CUDA graph capture 很慢（分钟级且不打印任何东西），**不要以为卡死就 kill**；可看 build 目录变化。
- 容器已修好的四个 host 环境坑（NFS root_squash / `/tmp/aiter_configs` 属主 / LDAP-only uid /
  `/root/.cargo/bin` 不可读导致 PATH walk EACCES）全部固化在
  `../glm52_tp8ep8_dpa_c128_yihou_20260914-0423/scripts/create_container_yihou.sh`，复用即可。
- profiling 一定会拖慢 decode：**profiled run 的 TPOT 不是性能数据**，只能用于耗时占比归因。
  正式性能数字仍以 packup 中的非 profiled run 为准。
- 8 rank 全开 trace 会爆磁盘/上下文：默认只让 rank 0 出 trace，窗口取少量 iteration。

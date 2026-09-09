# profiling-demo 设计方案（rev.3，待 review）

本文用中文写给人 review。落地时 package 内的 `*.yaml` / `readme.md` / 脚本一律用英文，遵循 `temp/mission.md` 的 RULES 第 6 条。

范围限定为本轮要求的一条链路：**在一台 MI355X 上部署 GLM-5.3-Flash，用 AIPerf 回放 Mooncake trace 作为负载，在负载中截取 torch profiler 窗口，再用 Magpie 产出 kernel 级 CSV**。fix_len 测试与 Optimus-AgenticBench 不在本轮实现范围内，但图的形状为它们预留了并列位置（见第 9 节）。

rev.2 相对 rev.1 的变化：只用 `smci355-ccs-aus-n04-33` 一台机器；采纳双轮运行以同时拿到干净性能基线；gz 本体进 handoff；权重用现场 FP8；trace 回放窗口先取前 2 分钟；新增一个 AI validator。

rev.3：第 10 节第 1 步（手工打通）已于 2026-08-31 完成，全链路跑通。文中带"实测"字样的数字都来自那次运行，脚本在 `temp/manual/scripts/`，完整记录在 `temp/manual/FINDINGS.md`。有一条已确认的决策需要复议（4.3 末尾的 `torch_trace` 体量），有五条新风险进入第 8 节。

rev.4：第 10 节第 2 至 5 步全部完成，package 已实现并跑通——7 个任务全部 succeeded、7 个 handoff 全部 valid、6 个 validator 全部 PASS，用时 18 分钟。设计在落地时被框架规则改动了三处，都记在第 12 节：handoff 携带 `command` 而非 `script`、原始日志压缩入包、站点路径用 `@NAME@` 占位符。唯一未实现的是 `check_reproduces`。

---

## 1. 结论摘要

流水线由 6 个叶子任务组成，全部是 program task，产出 7 种 handoff：

```
main（非叶，无 agent）
 ├── serve_baseline   → deployment_baseline                  froms: []
 ├── run_baseline     → aiperf_baseline                      froms: [serve_baseline]
 ├── serve_profiled   → deployment_profiled                  froms: [run_baseline]
 ├── run_profiled     → aiperf_profiled, torch_trace         froms: [serve_profiled]
 ├── kernel_scan      → kernel_table                         froms: [run_profiled]
 └── packup           → profile_packup                       is_end
       froms: [serve_baseline, run_baseline, serve_profiled, run_profiled, kernel_scan]
```

全部跑在 **`smci355-ccs-aus-n04-33`（10.235.192.139）** 一台机器上：

| 组件 | 端口 | 容器 |
|---|---|---|
| etcd | 2379 | `quay.io/coreos/etcd:v3.5.14` |
| sglang MIX worker（TP8） | 30000 | `infera/engine-sglang:glm53-flash` |
| infera router | 8100 | 同上（同容器内） |
| AIPerf | — | `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`，独立容器 |

`agent-sys run` 在登录节点 `smci355-ccs-aus-n01-29` 执行，program task 的 entry.sh 通过 `srun --jobid=<id> --overlap -N1 -n1 -w smci355-ccs-aus-n04-33 bash -c '...'` 进入计算节点。这与 `sglang_1p1d_glm5.2/cluster/README.md` 里 `export SSH_CMD="<your-scheduler> exec"` 的抽象是同一个位置，只是把 ssh 换成 srun。已实测该调用方式在作业 28080 上可用。

作业 28080 同时持有 `n04-33` 与 `n05-21`，本轮只使用前者，`n05-21` 空置。

---

## 2. 现场核查结果

在写方案前对 `n04-33` 做了实测，结论直接影响设计。

| 项目 | 状态 | 依据 |
|---|---|---|
| GLM-5.3-Flash BF16 权重（`glm53flash-demo` 用的那份） | **没有** | `/data/models` 上只有 `GLM-5.2-MXFP4` |
| GLM-5.3-Flash FP8 权重 | **有**，共享盘 `/apps/qiongzhu/models/GLM-5.3-Flash-FP8`，306 GB / 62 shards | `config.json` 的 `model_type: glm5_next`，`architectures: [Glm5NextForConditionalGeneration]`，`quantization_config` 为 `fmt: e4m3 / activation_scheme: dynamic` |
| `infera/engine-sglang:glm53-flash` 镜像 | **没有** | 只有 `rocm/infera:sglang-dev` 等，均不含 glm5_next |
| 已发布 sglang 镜像是否支持 glm5_next | **不支持** | 实测 `lmsysorg/sglang:v0.5.18-rocm720-mi35x`（0.5.18）与 `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260826`（0.5.18.dev20260826）导入 `sglang.srt.configs` 后 `hasattr(c, "Glm5NextConfig")` 均为 `False` |
| AIPerf 镜像 `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0` | **没有，但可拉** | `docker manifest inspect` 成功，匿名拉取可用 |
| 计算节点外网 | **通** | github.com 返回 200，nvcr.io / registry-1.docker.io 返回 401（未认证下的正常应答） |
| 本地盘 | `/data` 60 T，可用 54 T | `df -h` |
| 共享盘 `/apps` | 52 T 已用 93%，仅剩 3.7 T | `df -h`，因此大产物不写 `/apps` |
| CPU | 该作业每节点 256 核 | `scontrol show job 28080`：`AllocTRES=cpu=512`，2 节点 |
| Magpie 仓库 | 有，`/apps/tas/yaoc/research/topic/infera-with-hyperloom/Magpie` | `run_megapie_kernel_analyze.sh` 里 `cd` 的就是它 |
| Mooncake trace | 有，`sglang_1p1d_glm5.2/aiperf_trace/conversation_trace.jsonl`（12031 条 / 3.0 MB） | 直接复用 |
| `agent-sys` CLI | **未安装** | `which agent-sys` 为空，需要 `pip install -e agent_sys` 与 `pip install -e "agent_sys[claude]"` |
| Claude 后端凭据 | 已就绪 | `claude` 在 `~/.local/bin`，`ANTHROPIC_BASE_URL` / `ANTHROPIC_CUSTOM_HEADERS` 等已在环境里 |
| bwrap | **没有** | `which bwrap` 为空；内核 6.8.0，Landlock 应可用，需要在 `--dry-run` 阶段确认 CLI 的 sandbox 存在性检查能通过 |

由此得到三件必须在跑 package 之前完成的前置准备（第 7 节展开）。

---

## 3. 拓扑：单节点 MIX，AIPerf 同机独立容器

### 3.1 为什么是 MIX 而不是 1P1D

1. `patches/Dockerfile.sglang.glm53` 里 `BUILD_MOONCAKE=0`，注释写明理由是「upstream 仍把 GLM-5.3-Flash 的 PD 标为 dummy-weights-only 预览」。要做 1P1D 就得改成 `BUILD_MOONCAKE=1` 重新构建，并且踩一条上游自己都没验证的路径。
2. `glm53flash-demo` 已经在 8×MI355X 上把 MIX 跑通并留下了证据（`results/smoke_evidence.txt`、`throughput.csv`），是本轮唯一有实测背书的配方。
3. 本轮目标是"profiling 流程能稳定跑通"，不是"验证 PD 部署"。把未验证的 PD 引进来会让失败原因难以归属到 profiling 流程本身。

一个直接的好处是：MIX 下 router 只注册一个 worker，`disagg_mode` 为 `mixed`。`infera/server/profiling.py` 的 `select_targets` 接受 `role=mixed`（`DisaggMode` 枚举含 `MIXED/PREFILL/DECODE`），所以 `/v1/admin/profile/start?role=mixed` 这条控制面路径直接可用，不需要改 infera。

代价要写清楚：MIX 下 prefill 与 decode 在同一个进程里交错，一份 trace 里两类 kernel 混在一起，无法像 1P1D 那样用 role 选择器分开。本轮接受这个代价，在 `torch_trace` 的 README 的 Watch out 一节里显式说明。

### 3.2 AIPerf 同机运行的影响

`sglang_1p1d_glm5.2/engine/trace_replay.sh` 的注释指出两件事：AIPerf 必须跑在自己的容器里（引擎镜像是 Python 3.10，AIPerf 需要 ≥ 3.11），并且 `WORKERS` 这个 knob 的存在理由就是「当 AIPERF_NODE 是服务节点时才需要调」。也就是说同机运行是这套脚本预期内的用法。

同机运行会带来 CPU 争用，因为 AIPerf 在冷 cache 下要先把 trace 里每个 `hash_ids` 展开成真实 token block，这是 CPU 密集操作。缓解手段有以下几点：

1. 该作业每节点 256 核，`WORKERS=16` 只占 6%，余量充足。必要时把 `workers_max` 降到 8。
2. 采用双轮运行以后，第一轮（baseline）跑完时 `.mmap_cache/` 已经建好，第二轮（profiled）的 prompt 合成几乎不耗 CPU。也就是说真正需要担心争用的那一轮，恰好不是采 profile 的那一轮。
3. profiler 窗口的开启条件不是"启动 AIPerf 后 sleep 一段时间"，而是"轮询到真的有请求落到 worker 上，再开始 warmup 计时"。合成阶段再长也不会把 profile 窗口推到空载区间。

---

## 4. agent_sys 图设计

### 4.1 为什么是双轮，以及双轮为什么必须重启服务

CUDA graph 在 sglang 里是启动期参数（`--cuda-graph-backend-decode full|disabled`），无法在运行时切换。`glm53flash-demo` 的实测是 graph 打开后 c1 吞吐从 15.32 涨到 106.85 tok/s（约 7 倍）；而 graph 打开时 torch profiler 抓到的是 graph launch，kernel 归因会退化成聚合条目。两个诉求无法在一次部署里同时满足，所以：

- **baseline 轮**：`cuda_graph=1`，router 不加 `--enable-profiling`，rust router。产出可以对外引用的性能数字。
- **profiled 轮**：`cuda_graph=0`，router 加 `--enable-profiling` 并切 python backend。产出 trace，其性能数字只作为 profile 条件下的参照。

`serve_profiled` 的 `froms` 指向 `run_baseline` 而不是 `serve_baseline`，这条边表达的是"baseline 测完了才能把它拆掉"。`serve_profiled` 的第一步就是幂等 teardown。

### 4.2 为什么 `run_profiled` 是一个任务而不是两个

profiler 窗口必须落在负载窗口内部。如果把「发压」和「截 profile」拆成两个 `froms: [serve_profiled]` 的并列兄弟任务，agent_sys 会并发调度它们，但两者之间没有同步原语，只能靠在磁盘上放约定文件互相等待。这种耦合在图的边上表达不出来，出错时也难以定位。

因此 `run_profiled` 是单个 program task，同时产出两种 handoff（`outputs: [aiperf_profiled, torch_trace]`）。框架层面这是允许的——限制是"每个 subgraph 里每种 kind 只能有一个 producer"，不是"每个 task 只能产一种 kind"。任务内部时序：

```
1. 在同机启动 AIPerf 容器（后台，--fixed-schedule 回放 trace 窗口）
2. 轮询直到真的有请求到达（读 worker 日志里的 "Decode batch" / router /metrics），
   而不是启动后直接 sleep
3. warmup_s 之后 POST /v1/admin/profile/start?role=mixed
4. window_s 之后 POST /v1/admin/profile/stop
5. 轮询 trace 目录字节数直到稳定（torch 在 stop 返回之后才写完文件）
6. 等 AIPerf 退出，收集 artifact 目录
```

第 3～5 步直接移植 `sglang_1p1d_glm5.2/engine/capture.sh`，包括几个容易踩的细节：`with_stack:false` 必须显式给（sglang 默认 True，会让单 rank 文件从 14 MB 涨到 122 MB）、`record_shapes:true`（Magpie 的 `Input Shapes` 列依赖它）、`output_dir` 必须先 `mkdir`（sglang 不会自己建，失败发生在 profiler 回调里，那时 start 已经返回 200 了）、`TRACE_OUT` 必须是可写 bind mount（否则 docker 在容器层建目录，capture 看起来成功而宿主机什么都没有）。

### 4.3 七个 handoff kind

这里有一个框架约束必须先说明，它决定了下面的 content_type 选择。`handoff/content.py` 的 `check_items` 会**拒绝 content_type 没有声明过的顶层 item**。`structured_text` 的可选 item 只有 `{text.json, text.yaml, text.xml, schema}` 四个，所以任何需要携带产物目录（AIPerf 的 artifact 目录、8 个 trace gz、Magpie 的 CSV）的 handoff 都不能用它。`single_real_task/steps/serve.yaml` 里有一段专门讨论这件事，结论是宁可选 item 集合宽松的类型，也不要为了迁就类型去重命名产物文件。

| kind | content_type | items | 说明 |
|---|---|---|---|
| `deployment_baseline` | `reproducible` | `result`（health / `/v1/workers` / smoke 五段证据）、`env`（镜像 digest、ROCm 版本、GPU 型号、权重 sha、引擎参数全量）、`script`（本次实际执行的 mix_up / mix_worker 副本）、`logs`（worker 与 router 日志尾部）、`watchout` | graph on、无 profiling 的那次部署 |
| `deployment_profiled` | `reproducible` | 同上 | graph off、带 profiling 的那次部署 |
| `aiperf_baseline` | `reproducible` | `result`（AIPerf artifact 目录原样：`profile_export_aiperf.csv`、`profile_export.jsonl`、`profile_export_console.txt`、`server_metrics_export.csv`、`logs/`，外加一个 `summary.json`）、`env`、`command`（实际 aiperf 命令行）、`logs`、`watchout` | 可对外引用的性能数字 |
| `aiperf_profiled` | `reproducible` | 同上 | Watch out 里写明这是 profile 条件下的数字，不是基线 |
| `torch_trace` | `reproducible` | `result`（8 个 `*.trace.json.gz` + `manifest.json`：每 rank 文件名/字节数/sha256/GPU kernel 事件数、窗口起止）、`env`、`command`（profile start / stop 的请求体原文）、`watchout` | **实测 462 MB**，见下 |
| `kernel_table` | `reproducible` | `result`（`gap_analysis.csv` + `top_kernels.json`）、`env`、`command`（Magpie 命令行）、`watchout` | CSV 列为 `Name,Calls,Self CUDA total (us),Avg time (us),% Total,Input Shapes` |
| `profile_packup` | `code` | `codes`（整个 packup 目录，按 `experiment-result-packup` 的 `deliverable_layout.md` 组织） | 与 `single_real_task` 的 `runbook` 同理由：`code` 的 `codes` 内部不受约束，packup 的文件名能原样保留 |

`reproducible` 强制要求 `result` 与 `env`，且 `script`/`command` 至少有一个，README 必须含 Purpose / How to run / Result / Environment / Watch out 五节。这正好和 packup skill 的检查表重叠，也正好是本轮 mission 里"handoff 较为标准"这条目标想要的约束。

`deployment_baseline` / `deployment_profiled` 是同一套 schema 的两个 kind 名，`aiperf_baseline` / `aiperf_profiled` 同理。之所以不共用一个 kind 名，是因为框架限制每个 subgraph 里每种 kind 只能有一个 producer——`demo2` 里 `solutions_a/b/c` 是同一个原因。

各 kind 都不写 `items_schema`。`reproducible` 的 required/optional 集合已经把顶层形状定住了，`items_schema` 只能重复它；而产物目录**内部**的结构是 validator 的职责。这是两套机制存在的意义所在。

**`torch_trace` 的体量比原估大 4 倍，这一条需要复议。** 当初按 `sglang_1p1d_glm5.2/profiles/20260826_053606/prefill/` 估算，8 rank × 14.5 MB ≈ 116 MB，据此确认了「gz 本体进 handoff」。2026-08-31 实测是 **8 rank × 60.5 MB = 462 MB**（15 秒窗口，`with_stack:false`）。差 4 倍的原因是 MIX 把 prefill 与 decode 的 kernel 放进同一份 trace，且负载是饱和的——单 rank 2646817 个事件，其中 181607 个 GPU kernel。

462 MB 在 60 T 的本地盘上仍不构成压力，方案可以不变；但 `kernel_scan` 的 zone 还会 stage 一份副本，单次运行落盘接近 1 GB，已经翻越了「一个 handoff 值不值得整包搬运」的直觉门槛。备选是 handoff 只放 manifest 加 `top_kernels.json`，gz 留在节点本地由 `kernel_scan` 就地读——轻，但 handoff 不再自包含，节点回收后路径失效。缩小体量的旋钮只有 `window_s`（15 秒已接近下限）和降低负载并发。

### 4.4 六个 validator

`validator.schema.json` 是 `additionalProperties: false` 且**没有 `kind` 字段**，所以不存在"AI validator"这个 schema 概念。`single_real_task` 里那个所谓 AI validator，实际是一个把时间花在 `claude` 里的 script body，靠 `tags.logic_source` 与 `tags.cost` 与其他 validator 区分。`validator` spec §5.3 用这两个 tag 给一个 phase 排序，便宜的先跑。本设计沿用这套写法。

`validator.inputs` 是多对多绑定（schema 原文：“The binding is many-to-many”），所以 baseline 与 profiled 两条臂可以共用同一个 validator。

| validator | dimension / strength | logic_source / cost | inputs | 检查内容 |
|---|---|---|---|---|
| `check_service_live` | completeness / strong | `external_static` / `seconds` | `deployment_baseline`, `deployment_profiled` | `/v1/workers` 恰好 1 个 worker 且 `disagg_mode == "mixed"`；smoke 两问回答非空且算术题答案为 391；worker 日志中无 `memory access fault` / `HIP error` / `Traceback`；`env` 里记录的 CUDA graph 设置与该 kind 的预期一致 |
| `check_aiperf_report` | completeness / strong | `external_static` / `seconds` | `aiperf_baseline`, `aiperf_profiled` | 四个必需文件存在且非空；`profile_export.jsonl` 行数 > 0；错误请求占比低于 `max_error_rate`（默认 0.01）；`summary.json` 里 TTFT / TPOT / 吞吐 / 请求数字段齐全 |
| `check_trace_coverage` | completeness / strong | `external_static` / `minutes` | `torch_trace` | rank 数等于 `tp`（默认 8）；每个 gz 可解压且是合法 JSON；每个 rank 的 GPU kernel 事件数 > 0（判据移植自 `engine/tools/inspect_trace.py`）；窗口时长与 `window_s` 的偏差在容忍范围内 |
| `check_kernel_table` | usability / strong | `external_static` / `seconds` | `kernel_table` | CSV 表头与 Magpie 的六列一致；数据行数 ≥ `min_kernel_rows`（默认 20）；`% Total` 列求和落在 (0, 100] 内；top-1 kernel 的 `Self CUDA total (us)` > 0 |
| `check_packup_shape` | completeness / strong | `external_static` / `seconds` | `profile_packup` | 从 `single_real_task/assets/check_packup_shape.validator` 移植，按 packup skill 的 `checklist.md` 校验目录形状与内容行数下限 |
| `check_reproduces` | usability / weak | `external_dynamic` / `gpu_hours` | `profile_packup` | 一个全新的 Claude Code 会话只拿到 packup，按 `REPRODUCE.md` 实际复现一遍，本 body 在它的报告之上再核对它声称产出的文件确实存在且非空 |

`validator/report.py` 的 `blocks_the_task` 规则是：output phase 上如果一个 handoff 没有绑定任何 validator，会被判为 `unchecked` 并阻塞。上表七个 kind 全部有覆盖。

`check_reproduces` 是 `weak`，理由与 `single_real_task` 一致：判定实质上来自复现者自己的报告，body 只做旁证。它的 `cost: gpu_hours` 把它排在 `check_packup_shape` 之后，便宜的形状检查先失败就不必启动昂贵的复现。

复现的范围需要收窄，否则一次复现就是一整条流水线。`REPRODUCE.md` 里定义的"复现成功"是：拉起服务 + smoke 通过 + 一次 60 秒的 AIPerf 回放产出 CSV。不要求重跑 profile 与 Magpie。`reproduce_timeout_seconds` 默认 5400，bring-up 调试时传 `--var reproduce_timeout_seconds=120` 让它快速失败。

复现者需要 GPU，而此时 `serve_profiled` 的容器还占着显存。这一点不需要额外处理：`mix_up.sh` 的第一步本来就是幂等 teardown 加 `reset_gpus.sh` 门禁，复现者按 `REPRODUCE.md` 走就会先把现场清干净。代价是复现跑完之后原来的部署没了，所以 `check_reproduces` 一定排在所有其他任务之后。

---

## 5. 各任务实现要点

### 5.1 `serve_baseline` / `serve_profiled`

两者共用同一套脚本，靠环境变量分叉。移植 `glm53flash-demo/scripts/{mix_up,mix_worker,mix_smoke,reset_gpus}.sh`，改动如下：

1. **profiled 轮的 router 加 `--enable-profiling`**。原 `mix_up.sh` 没有这个参数，缺了它 `/v1/admin/profile/start` 返回 403。router backend 两轮都用 python：`infera/server/args.py` 里 `--router-backend` 的默认值就是 `python`，`glm53flash-demo` 从未指定过它，所以 `throughput.csv` 那批数字也是 python router 下测的。`launch_rust.py:exec_rust` 会在 `enable_profiling` 为真时直接拒绝 rust backend，因此两轮同用 python 还顺带消除了 baseline 与 profiled 之间的 backend 差异。
2. **profiled 轮把 `TRACE_OUT` 以可写 bind mount 挂进引擎容器**。
3. **`CUDA_GRAPH` 由变量控制**：baseline 轮 1，profiled 轮 0。
4. 保留 `reset_gpus.sh` 的硬门禁。`mix_up.sh` 的注释说明了原因：上一轮残留进程还占着显存时启动 worker，分布式 bootstrap 会以「memory capacity is unbalanced」这种误导性错误退出。
5. smoke 沿用 `mix_smoke.sh` 的五段证据，`check_service_live` 解析它的输出。

引擎参数沿用 AMD 验证过的配方，不加 `--speculative-*`：

```
--tp-size 8 --trust-remote-code
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--kv-cache-dtype bfloat16 --moe-runner-backend triton
--reasoning-parser glm45 --tool-call-parser glm47
--mem-fraction-static 0.85 --context-length 262144 --chunked-prefill-size 8192
```

baseline 轮额外加 `--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled --cuda-graph-bs-decode 1 2 4 8 16 24 32 48 64 96 128`；profiled 轮两个 backend 都是 `disabled`。

### 5.2 `run_baseline` / `run_profiled`

AIPerf 命令沿用 `trace_replay.sh` 的形状：

```
aiperf profile \
  --model glm5.3-flash --tokenizer <model_path> \
  --endpoint-type chat --streaming \
  --url http://10.235.192.139:8100 \
  --input-file <trace> --custom-dataset-type mooncake_trace \
  --isl-block-size 512 --workers-max 16 --request-timeout-seconds 900 \
  --no-gpu-telemetry --ui none \
  --artifact-dir <out> \
  --fixed-schedule --fixed-schedule-end-offset 120000 \
  --concurrency <max_conc> \
  --extra-inputs temperature:1.0 top_p:0.95 ignore_eos:true
```

`aiperf_compat/sitecustomize.py` 必须一起带上。AIPerf 0.12 的 `Tokenizer._resolve_local_snapshot` 不接受本地目录，那个 9 行的 patch 让它接受，否则会去 HF 拉 tokenizer。

`run_baseline` 只发压不采 profile；`run_profiled` 按 4.2 的时序在负载中截窗口。两者共用同一个 `aiperf_replay.sh`，`run_profiled` 额外调 `capture.sh`。

关于 2 分钟窗口的时间预算：`--fixed-schedule-end-offset 120000` 意味着回放 trace 时间轴的前 120 秒，墙钟时长也约 120 秒。profiled 轮里 `warmup_s + window_s` 必须小于它。默认 60 + 20 = 80 秒，余量 40 秒。首次 bring-up 建议传 `--var warmup_s=30 --var window_s=15`，把余量放大到 75 秒，确认时序正确之后再调回默认值。

### 5.3 `kernel_scan`

等价于 `run_megapie_kernel_analyze.sh`：

```
cd <magpie_root>
python3 -m Magpie benchmark --trace-dir <trace_dir> --categories kernel \
  --top-k 10000 --no-rank-csv -o <trace_dir>/megapie
```

产出 `<trace_dir>/megapie/gap_analysis/gap_analysis.csv`。任务额外把 CSV 解析成 `top_kernels.json`，让 `check_kernel_table` 和下游的 analyze 阶段不必再解析 CSV。

`magpie_root` 作为包变量传入，`permissions.grants` 里以只读方式声明该绝对路径。本轮 `AGENT_SYS_NO_PERMISSIONS=1`，grants 只是文档性声明，但写上去以后打开权限校验时不用再补。

### 5.4 `packup`

按 `experiment-result-packup` skill 的 `deliverable_layout.md` 组装：README / REPRODUCE / environment / scripts / results / logs / notes。`collect_env.sh` 直接复用 skill 里那份，通过 srun 在计算节点上执行。

`REPRODUCE.md` 需要明确写出"复现成功"的判定，因为 `check_reproduces` 的复现者只拿得到 packup，拿不到本文档。

---

## 6. 包变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `jobid` | 无 | Slurm 作业 ID。无默认，因为它每次都变 |
| `node` | 无 | 计算节点 hostname |
| `node_ip` | 无 | 该节点 IP，router 与 worker 绑定它 |
| `etcd_port` | `12379` | 不能是 2379，那是节点上 k8s 控制面 etcd 的 |
| `model_path` | 无 | 权重目录，节点上的绝对路径 |
| `image` | `infera/engine-sglang:glm53-flash` | 引擎镜像 |
| `aiperf_image` | `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0` | |
| `etcd_image` | `quay.io/coreos/etcd:v3.5.14` | |
| `trace_file` | `/apps/.../sglang_1p1d_glm5.2/aiperf_trace/conversation_trace.jsonl` | 绝对路径而非包内文件，避免每个 attempt zone 都复制一份 |
| `magpie_root` | `/apps/.../Magpie` | |
| `work_root` | `/data/agent_sys_profiling` | 节点本地工作目录，产物落在这里 |
| `served_name` | `glm5.3-flash` | |
| `tp` | `8` | |
| `warmup_s` | `60` | |
| `window_s` | `20` | |
| `trace_end_ms` | `120000` | 只回放 trace 前 2 分钟 |
| `max_conc` | `256` | |
| `workers_max` | `16` | AIPerf 进程数，同机运行时的 CPU 旋钮 |
| `max_error_rate` | `0.01` | `check_aiperf_report` 的阈值 |
| `min_kernel_rows` | `20` | `check_kernel_table` 的阈值 |
| `reproduce_timeout_seconds` | `5400` | `check_reproduces` 的上限 |

前四个无默认值是刻意的：照 `single_real_task/README.md` 的做法，缺了会在加载期报出文件、行号和变量名，而不是在跑到一半时以别的形态失败。

---

## 7. 前置准备（不属于 package）

这三件事有一次性成本，建议先手工做完并验证，再让 package 跑。

1. **构建引擎镜像**。在 n04-33 上，build context 取仓库根：

   ```
   docker build -f examples/glm53flash-demo/patches/Dockerfile.sglang.glm53 \
     -t infera/engine-sglang:glm53-flash .
   ```

   实测 9 分 25 秒（原估 30–60 分钟偏保守），产出 image `a9e6029b8750`，最后一行 `BUILD_OK Glm5NextConfig.model_type = glm5_next`。GitHub 可达已验证，其中拉 PR #36507 的 `git fetch` 单独占了约 5 分钟。构建过程中有三处断言会检查 overlay 是否真的落到解释器实际导入的路径上，任一失败都会让构建停下。

2. **拉 AIPerf 镜像**：`docker pull nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`。

权重不需要准备。最初的设计把 306 GB 从 `/apps` 拷到节点本地 NVMe 列为前置步骤，实测之后取消了：n04-33 上 `/apps`（NFSv3，rsize=1 MB，tcp）单流 `iflag=direct` 冷读稳定在 921 MB/s，306 GB 约 5.5 分钟读完；而拷贝本身要把这 306 GB 从同一个 NFS 挂载完整读一遍再写一遍本地盘，等于把第一次加载的开销提前支付并额外加一次写。该节点有 3 TB 内存（实测 `MemTotal` 3170235640 kB），第一次加载后 checkpoint 留在 page cache，第二次冷启动（graph-off 那轮）两种方案都不碰盘。容器内以 `-v /apps/qiongzhu/models:/apps/qiongzhu/models:ro` 挂载即可读，NFS 没有 root_squash 造成的读障碍（已实测）。

另外还需要 `pip install -e agent_sys` 与 `pip install -e "agent_sys[claude]"`，并用 `agent-sys run --dry-run` 确认登录节点的 sandbox 存在性检查能通过（这台机器没有 bwrap，走 Landlock 路径，内核 6.8 应当满足，但没实测过）。

---

## 8. 已知风险与取舍

**FP8 权重与验证过的配方不是同一份 —— 已实测，风险不成立。** `glm53flash-demo` 的全部实测数据来自 `zai-org/GLM-5.3-Flash`（BF16，328 GB），而现场只有 FP8 变体（306 GB，`quantization_config` 为 e4m3 dynamic）。2026-08-31 手工跑通后确认：sglang 自动选中 `quant_method=Fp8MoEMethod`，AITER mHC fused 路径生效，smoke 两问都答对，c1 解码吞吐 110.24 tok/s（BF16 那边是 106.85），CUDA graph 照常工作，无 fault。这条降级为备注。细节见 `temp/manual/FINDINGS.md`。

**双轮运行意味着两次冷启动。** 实测：首次从 NFS 加载 819 秒，重启后走 page cache 243 秒。加上 baseline 轮的 CUDA graph capture（`graph_capture_cost.csv` 记录 bs≤128 时约 33 秒 / 1.4 GB），一次完整双轮流水线（不含镜像构建与 `check_reproduces`）实测约 25–30 分钟。

**这些节点上跑着 Kubernetes 控制面。** 它的 etcd 以 TLS 占住 `127.0.0.1:2379` 与节点 IP 的 2379。`glm53flash-demo` 的 `mix_up.sh` 硬编码 2379，在这里会让 etcd 容器以 `address already in use` 退出，同时把 worker 的明文 discovery 客户端指向一个不属于我们的 TLS 端点。package 用 `etcd_port` 变量（默认 12379），并在拉起前对五个端口做占用预检。

**`reset_gpus.sh` 原样搬过来会杀掉 `slurmstepd`。** 它对 `rocm-smi --showpids` 报的每个 pid 无差别 `kill -9`，而在 Slurm GPU 节点上这个集合包含为作业步 cgroup 持有 KFD 句柄的 `slurmstepd`。实测第一次运行就杀了一个。`glm53flash-demo` 的节点是没有调度器的裸机，所以那边不会暴露。package 里的版本只杀名字像遗留推理进程的，显式保护调度器与容器运行时进程，判定关口改成 VRAM 是否回到基线。

**引擎容器以 root 运行，它创建的目录本用户写不进去。** capture 的 `output_dir` 若用 `docker exec mkdir` 创建，后续 Magpie 无法在其中建 `megapie/`。package 里由宿主机侧创建该目录（root 仍可往我们拥有的目录里写 trace），并且让分析步骤接受独立的输出目录，不假定 trace 目录可写。

**写入刚结束时 `ls -l` 报的文件大小不可信。** 实测同一脚本内 `du -sb` 报 484205285 字节而 `ls -l` 对每个文件报 118259 字节，事后 `stat` 是 60548506 字节；AIPerf 产物目录上也出现过。`check_trace_coverage` 与 `check_aiperf_report` 判定"非空"必须在读取时点用 `stat` / `du` 重新取。

**MIX 下 prefill 与 decode 的 kernel 混在一份 trace 里。** 见 3.1 末尾。

**图中途失败会留下占着显存的容器。** 两个 serve 任务开头都有幂等 teardown 加 `reset_gpus.sh` 门禁，所以重跑是安全的；但如果流水线在中间失败后不重跑，容器会一直挂着。package 提供 `assets/lib/down.sh` 供手工调用。框架本身没有 per-task 的清理钩子。

**program task 没有超时。** `agent/backends/program.py` 里只有轮询等待子进程，没有 wall-clock 上限。CLI 的 `_settle` 有 1800 秒上限，但那管的是"整张图是否静默"，不是单任务。所以每个 entry.sh 自己带 timeout。

**program task 成功时 stdout 不会被框架保留。** `_detail()` 只在退出码非 0 时把尾部 8 KB 写进结果。因此所有需要留证的输出都要显式写到 zone 的 `logs/` 或 handoff 里。

**`check_reproduces` 会把现场拆掉。** 见 4.4 末尾。它排在最后，且是 `weak`，失败不代表流水线的产物无效——但按 `blocks_the_task` 的规则，它失败仍会让 `packup` 停在 output phase。首次 bring-up 建议用 `--var reproduce_timeout_seconds=120` 让它快速失败，把注意力先放在前五个任务上。

**Slurm 作业时限。** 28080 的 9 小时上限在 2026-08-31T16:03 到期。一次完整流水线（不含 `check_reproduces`）预计 50–70 分钟，加上复现 1.5–2 小时。连续调试需要注意续期。

---

## 9. 目录布局

```
profiling-demo/
├── DESIGN.md                        本文档
├── README.md                        跑法、图的形状、预期产物
├── main.yaml                        根 closure + subgraph
├── shared.yaml                      共享的 program agent（6 个 task + 6 个 validator 共用）
├── steps/
│   ├── baseline.yaml                deployment_baseline / aiperf_baseline kind
│   │                                + check_service_live + check_aiperf_report
│   │                                + serve_baseline / run_baseline task
│   ├── profiled.yaml                deployment_profiled / aiperf_profiled / torch_trace kind
│   │                                + check_trace_coverage
│   │                                + serve_profiled / run_profiled task
│   ├── kernel_scan.yaml             kernel_table kind + check_kernel_table + task
│   └── packup.yaml                  profile_packup kind + check_packup_shape + check_reproduces + task
└── assets/
    ├── lib/
    │   ├── store.py                 从 demo 复制，validator 读 zone 内 JSON 用
    │   ├── remote.sh                srun 远程执行封装
    │   ├── common.sh                日志 / 容器 / 健康轮询 helper
    │   ├── down.sh                  手工 teardown
    │   └── aiperf_sitecustomize.py  AIPerf 0.12 本地 tokenizer 兼容
    ├── main.task/readme.md
    ├── serve_baseline.task/         readme.md + entry.sh
    ├── serve_profiled.task/         readme.md + entry.sh
    ├── run_baseline.task/           readme.md + entry.sh
    ├── run_profiled.task/           readme.md + entry.sh
    ├── kernel_scan.task/            readme.md + entry.sh + scan.py
    ├── packup.task/                 readme.md + entry.sh + packup.py + collect_env.sh
    ├── serve/                       两个 serve task 共用：mix_up.sh + mix_worker.sh
    │                                + mix_smoke.sh + reset_gpus.sh + serve.py
    ├── load/                        两个 run task 共用：aiperf_replay.sh + capture.sh + run.py
    ├── check_service_live.validator/
    ├── check_aiperf_report.validator/
    ├── check_trace_coverage.validator/
    ├── check_kernel_table.validator/
    ├── check_packup_shape.validator/
    └── check_reproduces.validator/
```

`assets/serve/` 与 `assets/load/` 不带 `.task` 后缀，因为它们不是 body 目录而是共用脚本，framework 只按 `<name>.task` / `<name>.validator` 的命名去找 body。各 task 的 `entry.sh` 从 `$AGENT_SYS_TASK_PACKAGE/assets/serve/` 取脚本。

对后续阶段的预留：`run_profiled` 之外再加 `fixlen_run`（fix_len 测试）和 `agentic_run`（Optimus-AgenticBench），都是 `froms: [serve_profiled]` 的并列兄弟，各产各的 report kind，`packup` 的 `froms` 里多列两项即可。`kernel_scan` 的 `kernel_table` 直接就是隔壁 `analyze-demo` 的输入。

### 运行方式

```
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run \
  --package agent_sys/examples/llm_e2e_performance_optimization/profiling-demo \
  --var jobid=28080 \
  --var node=smci355-ccs-aus-n04-33 --var node_ip=10.235.192.139 \
  --var model_path=/data/models/GLM-5.3-Flash-FP8
```

首次 bring-up 追加：

```
  --var warmup_s=30 --var window_s=15 --var reproduce_timeout_seconds=120
```

---

## 10. 落地顺序

分五步，每步都有可验收的中间态：

1. ~~**手工打通一遍**（不涉及 agent_sys）~~ — **已完成，2026-08-31**。构建镜像 → 拉起 MIX（graph on）→ smoke 通过 → AIPerf 回放前 2 分钟出 CSV → 重启为 graph off + profiling → capture 出 8 个 gz → Magpie 出 158 行 `gap_analysis.csv`。脚本落在 `temp/manual/scripts/`，实测数字与六处必须偏离原 demo 的地方记在 `temp/manual/FINDINGS.md`，第 8 节已吸收。
2. ~~**只做 `serve_baseline` 一个叶子**~~ — **已完成**。验证了远程执行、handoff 写入、validator 判定这条链路。这一步暴露了本设计里最贵的一个未预见约束（第 12 节）。
3. ~~**加 `run_baseline`**~~ — **已完成**。
4. ~~**加 `serve_profiled` 与 `run_profiled`**~~ — **已完成**。
5. ~~**加 `kernel_scan`、`packup` 与两个 packup validator**~~ — **已完成**，除 `check_reproduces` 外。

全图实测（2026-08-31）。跑了两次：复用已有部署 18 分钟，完全冷启动 24.1 分钟，两次结果一致：

| 任务 | 状态 | handoff | validator |
|---|---|---|---|
| `serve_baseline` | succeeded | `deployment_baseline` valid | `check_service_live` PASS |
| `run_baseline` | succeeded | `aiperf_baseline` valid | `check_aiperf_report` PASS |
| `serve_profiled` | succeeded | `deployment_profiled` valid | `check_service_live` PASS |
| `run_profiled` | succeeded | `aiperf_profiled` + `torch_trace` valid | `check_aiperf_report` / `check_trace_coverage` PASS |
| `kernel_scan` | succeeded | `kernel_table` valid | `check_kernel_table` PASS |
| `packup` | succeeded | `profile_packup` valid | `check_packup_shape` PASS |

产出数字取自冷启动那次（复用部署那次测到的是预热过的 radix cache，见下）：每轮 346 个请求；graph on 输出 631 tok/s、TTFT 均值 25.9 s，graph off 输出 380 tok/s、TTFT 均值 47.5 s——这正是为什么只有 baseline 轮的数字可以对外引用；8 个 rank 共 365 MB、1096288 个 GPU kernel 事件；159 个 kernel 排名，top 25 占 138.6 秒总自 CUDA 时间的 89.2%，最大一项是 aiter `cross_device_reduce_2stage` 22.86%，即 TP-8 的 all-reduce 占掉引擎近四分之一的 GPU 时间。

两次运行的 baseline 数字差一个量级（631 对 1004 tok/s，TTFT 25.9 s 对 484 ms），原因是复用部署那次引擎的 radix cache 已被同一份 trace 预热。Mooncake trace 带 `hash_ids`，AIPerf 会展开成真实 token block，所以前缀命中率直接决定 prefill 的工作量。结论是 `PD_REUSE_DEPLOYMENT=1` 不只省时间，它会改变测量值——完整对比记在 `temp/ARTIFACTS.md` 第 5 节。

---

## 12. 落地时被框架规则改掉的三处设计

这三处都不是实现细节，而是 rev.3 里写定的设计在接触 `handoff` 的发布检查后被推翻。详细记录在 `temp/manual/FINDINGS.md` 第二部分，两条框架缺陷记在 `temp/bugs/`。

**`deployment_*` 的 `script` item 改成了 `command`。** rev.3 的第 4.3 节写的是把拉起脚本原样复制进 handoff。`handoff.locality` 拒绝任何命名了固定允许列表之外绝对路径的内容文件，而真实的拉起脚本里有 `/tmp/glm53_mix.log` 这类路径。两条出路都比现在这条差：把路径替换掉会让发布出去的脚本不可运行，打包压缩会让它无法评审。所以 handoff 携带调用方式，脚本的出处（包路径 + commit）记在 `env/deployment.json`。`reproducible` 的 `script`/`command` 二选一正好覆盖这个选择。

**原始日志压缩后入包。** 实测一轮日志里 818 个绝对路径命中中有 817 个是误报——引擎镜像内部路径、access log 里的 HTTP 路由、etcd 的 key 前缀。这正是 `locality.py` 自己 docstring 里承认的 96% 噪声率，而本该处理它的机制（kind 的 `dependencies` 喂给 `Oracles.image_prefixes`）在类型里存在、在实现里没接线。压缩保留字节不变，seal 跳过无法按 UTF-8 解码的文件。

**站点路径用 `@NAME@` 而不是 `${NAME}`。** 这一条是替换动作自己制造的问题：`locality._CANDIDATE` 的 lookbehind 排除集 `[A-Za-z0-9._~@+-]` 不含 `}`，所以 `${TASK_PACKAGE}/assets/serve/mix_smoke.sh` 剩下的 `/assets/serve/mix_smoke.sh` 又成了新候选。`@` 在排除集里。

此外还有一条不是推翻而是加强：`agent/gate.py` 要求 `script`/`command`/`entry` 带可执行位，这把 `command` 从一段记录推成了一个可运行脚本；而写成脚本之后用 shell 变量接站点路径，反而让它天然通过了 locality 检查。rev.3 的 4.4 节里 `check_service_live` 的 rule 5 也因此从"读 handoff 自己声明的字段"改成"读引擎和 router 的实际命令行"——自我声明的字段对它要抓的那个错误毫无用处。

---

## 11. 已确认的决策

| 项 | 结论 |
|---|---|
| 机器 | 只用 `smci355-ccs-aus-n04-33`，`n05-21` 空置 |
| 拓扑 | 单节点 MIX，AIPerf 同机独立容器 |
| `torch_trace` 存储 | gz 本体进 handoff，handoff 自包含。**实测体量是 462 MB 而非确认时估的 116 MB，见 4.3 末尾，需要复议** |
| CUDA graph | 双轮：baseline（graph on，无 profile）+ profiled（graph off，带 profile） |
| 权重 | 现场 FP8（`/apps/qiongzhu/models/GLM-5.3-Flash-FP8`），先手工 smoke 验证 |
| trace | `conversation_trace.jsonl`，`trace_end_ms=120000`（前 2 分钟） |
| AI 参与 | 一个 `check_reproduces` validator（script body，时间花在 claude 里），其余全 program |

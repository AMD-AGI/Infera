# GLM-5.2-MXFP4 3P(TP4)1D(TP8)，sglang v0.5.21 + Mooncake 跨 rank 修复，AgentX 吞吐

> 结果与结论：`RESULT.zh-CN.md`；从零复现最优配置：`REPRODUCE.zh-CN.md`（`scripts/reproduce_best.sh`、`config/best.sh`）。

目标：在升级后的镜像（sglang v0.5.21）上测 3P(TP4)1D(TP8) 的 AgentX 吞吐，找出吞吐最高的并发点。
配置沿用 yaocheng 的 2P1D（`../../yaocheng/2p1d-sweep-triton-dsa-20260922/config.sh`），只改这个形态和镜像必需的部分；
router 与 yaocheng 一致（kv-aware，overlap 权重 20/2），唯一差别是关闭 PD DP-rank 亲和，靠移植的 Mooncake 补丁支持跨 rank 传输。

进度：[`operations/STATUS.zh-CN.md`](operations/STATUS.zh-CN.md)；结果：`results/<round>/c<CONC>/`；报告：`RESULT.zh-CN.md`（跑完后写）。

## 部署

| 实例 | 节点 | GPU | 网卡（`--disaggregation-ib-device` / `MC_TE_FILTERS`） | PID 偏移 |
|---|---|---|---|---|
| prefill-0（TP4/DP4） | crsuse2-m2m-135 | 0,4,5,6 | `ionic_0..6`（`ionic_7` 无 netdev） | 30000 |
| prefill-1（TP4/DP4） | crsuse2-m2m-137 | 0–3 | `ionic_0..7` | 30000 |
| prefill-2（TP4/DP4） | crsuse2-m2m-137 | 4–7 | `ionic_0..7` | 60000 |
| decode-0（TP8/DP8） | crsuse2-m2m-138 | 0–7 | `ionic_0..7` | 30000 |
| etcd / router / AgentX 客户端 | crsuse2-m2m-137 | — | — | — |

每卡吞吐按 20 张 GPU 计（AgentX 按 3×4 + 8 自动计算）。135 的 GPU1 被其他租户的 vLLM 占用，未使用。

## 镜像 `image/`

`infera-sglang:v0521-llying-1012-mc4340-destpin`，`sha256:afcc2f8abff7aee8a188e574168120826f35f26fa3d3e0db3881c1e8d91e820a`，
三台节点 ID 一致。基底 `infera-sglang:v0521-yihou-1010-mc4340`（137 上的 `e8e2838f…`，即 v0.5.21 recipe + Mooncake #4340），
只重编 Mooncake `engine.so`：faae8dd4 + `mooncake-pr4340-…port.yihou.diff` + `../cross-rank-rca-20260921/fix/` 的
`mooncake-dest-local-rail.diff`、`mooncake-ib-timeout.diff`、`mooncake-dest-pinned-hca.diff`（四个补丁互不重叠，依次干净打上），
镜像设置 `MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=18`。`build_image.sh` 在 137 构建并推送、核对各节点 ID。

## 配置 `config/`

`config.3p1d.sh`（唯一配置，launch/agentx_bench/stop 共用），`topology.3p1d.tsv`，`mounts.v0521.tsv`。

| 项 | 取值 | 来源 |
|---|---|---|
| prefill | TP4/DP4/DPA，mem 0.85，chunk 32768，max-running 256，graph bs 256，HiCache ratio 1.5（153 GB/rank） | yaocheng 2P1D |
| decode | TP8/DP8/DPA，mem 0.85，max-running 256，graph bs 256，EAGLE 5/1/6，模拟接受 3.61，custom all-reduce 开 | yaocheng 2P1D |
| 共同 | triton DSA，`index_share=false`，fused indexer 默认开，fp8 KV，`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`，`TOPK_V2=false`，aiter allreduce fusion | yaocheng 2P1D / v0.5.21 默认 |
| 挂载 | kv-events 非阻塞 PUSH、absorb-bmm 配置关闭（`patches/runtime-mounts/`，与 v0.5.21 pack-up 逐字节一致） | v0.5.21 必需 |
| KV 传输 | RDMA，每 rank 共用整组网卡，dest-device-affinity + dest-local-rail | 跨 rank 修复（B4 做法） |
| router | kv-aware，权重 20/2，`INFERA_PD_DP_RANK_AFFINITY=false` | yaocheng（亲和必须关：P DP4、D DP8） |
| 端口 | engine 26101 起步长 256，bootstrap 28198+i，kv 25657+8i，snapshot 28901+i，etcd 24501，router 28100 | 全部 < 30000 |
| AgentX | full 模式 3600 s，warmup 10/lane，InferenceX 918524ff，CONC 144/160/176 | 同 yaocheng 口径 |

## harness 改动（`scripts/bench-harness/`，以 v0.5.21 pack-up 的 bench-harness 为底）

- `tools/topology.py`：允许同一节点出现多个同角色实例（用实例名区分）。
- `launch.sh`：`GPUS_<instance>` 按实例指定 GPU，并检查同节点 GPU 不重叠；`KV_EVENT_PORT_STRIDE`（多 prefill 时须 ≥ DP）；
  router 可写目录放 `NODE_RW_ROOT`；`DRY_RUN=1` 在各节点渲染命令不启动；安全过滤改为 `llying`。
- `engine.sh`：节点键取自 hostname；`IB_DEVICES_<node>` 按节点替换网卡；缓存目录按实例区分；
  `PID_OFFSET_<instance>` 推广 PID 偏移到任意实例（同节点两个 prefill 的 rank PID 互不重叠）。
- `stop.sh`：只停名字含 `llying` 且前缀匹配的容器，保留已停止容器与日志。

## 流水线 `scripts/`

- `run_3p1d.sh all`：preflight（镜像 ID、所用 GPU 空闲、无残留容器）→ 以**真实接受率**起栈 → census →
  `kv_gate.py` 已知答案门禁 → census → `restart_decode.sh` 只重启 decode（切到模拟接受 3.61）和 router → 冒烟 →
  census（只看切换之后的日志）→ 按顺序跑 AgentX 各点，每点后 census 并把结果复制到 `results/<round>/c<CONC>/`。栈最后保持运行。
- `kv_gate.py`：每个请求一份独立的登记表文档（约 4 万 / 11.5 万 tokens），temperature 0 问一条记录的代码；P DP4→D DP8 且亲和关闭，
  绝大多数传输跨 rank。通过条件：无 HTTP 错误且正确率 ≥ 95%。
- `census.sh`：扫描 MAF、OOM、Traceback、进程退出、Mooncake 传输失败（`transfer failed`、`Session … failed.`、
  `Process failed for slice`、`retry counter exceeded`）、hipIpc/NCCL 告警。
- `seed_caches.sh`：用 137 上 yihou 的 v0.5.21 mc4340（同一 aiter/triton）JIT 缓存给 prefill-2、decode-0 做种子，只复制。

调优（2026-10-13 起，见 `PLAN-TUNING.zh-CN.md`）：

- `tune_queue.sh`：在控制节点后台按 `operations/tune_queue.tsv` 逐行执行（`round` / `run` / `stop`），运行中可追加，失败即停。
- `tune_round.sh`：一轮 = 内存快照 → 停 router → flush 3 个 prefill → 带本轮 `INFERA_*` 开关起 router 并按启动日志核对 → AgentX
  → census、`flush_check.py`、`prefill_balance.py` / `decode_balance.py` / `engine_pace.py`、router 指标 → 发布到 `results/<variant>/c<C>/`。
- `restart_router.sh`（`ROUTER_ENV`）、`restart_prefills.sh`（只重启 3 个 prefill 并安装配置覆盖，decode 保持预热，用于 2a）。
- `tune_watch.py`：人工巡检的一屏视图；`tune_report.py --write RESULT.zh-CN.md ...`：生成 RESULT §3 的对比表。
- `tune_post.sh`（一轮的收尾，AgentX rc≠0 记 FAIL）、`tune_finish.sh`（驱动意外退出时接管客户端并收尾）、`warmup_round.sh`（重启后不计分预热）。
- `sampler.py`（后台每 60 s 记录各引擎延迟直方图与各容器进程 CPU）+ `sample_window.py`（任意窗口的 TTFT 分段与进程 CPU）。
- `transfer_smoke.py`（重启后 192 并发 KV 传输冒烟）、`hang_check.py`（prefill 挂起检测）、`log_follow.sh`（server 日志断点续接）、
  `drop_prefill.sh`（从活栈拿掉一个 prefill，3P1D → 2P1D）；`restart_prefills.sh` 支持 `INSTANCES=` 只重启指定实例。

原始运行目录 `runtime/runs/<round>/`（git 忽略，含 server 日志与 aiperf 全量产物）；节点本地缓存在 `/mnt/m2m_nobackup/liyingli_3p1d/`。

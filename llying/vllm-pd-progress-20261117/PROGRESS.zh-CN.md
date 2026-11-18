# 进度与自主决定记录（2026-11-17 17:11 起，用户离开约 10 小时）

时间都是 UTC。每个主题的细节在各自目录；这里只记时间线、自主做的决定和待你确认的事。

## 待你 review 的决定

| # | 时间 | 决定 | 理由 / 可替代方案 |
|---|---|---|---|
| D1 | 17:14 | 在 294199（v3-135）里用一个 `srun --overlap` 步骤常驻队列执行器（`setsid nohup`，与登录会话脱离），所有命令经共享目录投递 | spur 每节点只允许一个进入作业的会话，且任何步骤非零退出会结束作业；执行器恒退出 0 |
| D2 | 17:25–17:47 | 实验镜像 `infera-vllm:v0340-0c5e5e89`：v0.34.0 基础 + Mooncake v0.3.14 自编（加我们的 #4540 / #4541 与一个 include 修复）+ GEM 分配器 + Infera core（不装 `[vllm]` extra） | 镜像自带的 PyPI Mooncake 0.3.14 没有目的端网卡选择与 ACK 超时；Infera 的 faae8dd4 补丁打不到 v0.3.14；`[vllm]` extra（gaie）的 grpcio 与 vLLM 冲突 |
| D3 | 17:56 | 修 Infera：`infera/engine/vllm/args.py` 兼容 v0.34 的 CLI（`make_arg_parser` 挪了位置）；新增 `infera/engine/vllm/gem_mooncake.py` | 不修 v0.34 上起不来；GEM KV 是 crusoe 驱动的绕法 |
| D4 | 18:16 | 提交第二个节点的持有作业 294784（`amd-agentx-qos`，单节点，12 h，批处理脚本内常驻执行器）给 P8D8 用 | 三个能用的 QoS 都满、排队长，提前排；开始前单节点工作照常在 v3-135 上做 |
| D5 | 19:05 | **中止 s1 计分**（prefill 只有设备缓存）并改跑 s2 | s1 在预热中就颠簸（前缀命中 5.2%、约 8 请求 / min），测出的数没有对比意义；见 `../vllm-p4d4-crusoe-20261117/` 1a 节 |
| D6 | 19:06 | s2 同时加 CPU KV 层（1 TB）和缓存感知准入（窗口 128、阈值 0） | 两者对应 sglang recipe 的 HiCache 与 SPF；为了尽快拿到可比的基线没有分开测，之后补单项 A/B |
| D7 | 19:23 | s2 的 prefill 在 66k 冷请求时 GPU 访存故障，停掉 s2，先隔离原因 | 见下文 |
| D8 | 19:41 | 绕开 GEM + offload 故障：prefill 不用 GEM（显存目标 0.6），decode 用 GEM；根因留待 GPU 空闲时查 | 先拿到可比的基线 |
| D9 | 19:58 | AgentX 的 offload 元数据用 `KV_OFFLOAD_BACKEND=vllm-native`（InferenceX 对 vLLM OffloadingConnector 的取值） | InferenceX 要求 dram 时必须给 backend |
| D10 | 20:17 | 中止 s2b（PD 死锁），s2c 去掉缓存感知准入（prefill 回 FCFS），decode 显存目标 0.95 | 死锁机制见 P4D4 README 1c；SPF 类重排在 vLLM 并发下发下不安全 |
| D12 | 21:09 | s3 改为只加 CPU 层容量（1 TB → 2 TB），串行 router + 缓存感知准入顺延到 s4 | s2c 稳态显示差距全在命中率；先单变量看容量 |
| D13 | 22:17 | 放弃加容量（本节点 `/dev/shm` 1.4 TB、2 TB 时 worker 被 SIGKILL）；s4 合并"串行 router + 缓存感知准入"与"offload 批量拷贝描述符上限 512"（针对 s2c 的静默卡住） | 时间有限，先拿到一个跑得完、命中更高的配置 |
| D11 | 20:03 | P8D8 今晚大概率做不了：第二个节点（294784）排在组上限后，最早约 03:00 | 单节点上继续 P4D4 与优化 |
| D14 | 23:27 | s5 = s4 配置 + CPU 层 1.25 TB、测量 3600 s（与 sglang 同时长），测量 30 min 时抓 decode trace | s4 有效（26,541），差距在 prefill 命中率；1.25 TB 已接近本节点 `/dev/shm` 上限 |
| D15 | 23:44 | s6 = s5 + vLLM `TieringOffloadingSpec` 的 FS 二级层（写穿，落本地 NVMe xfs 卷 28 TB），失败则退回 ARC 淘汰策略 | 有效容量是 vLLM 侧的结构性短板：CPU 层写穿使设备 KV 与 CPU 重复，sglang 写回式有效约 3,100 万 token、vLLM 约 2,000–2,400 万；FS 层是 vLLM 已有、不需改代码的扩容办法（超出 sglang recipe，内存对内存的对比另列） |
| D16 | 00:08 | 放弃 s6（FS 层起栈失败，vLLM bug U11）与其 ARC 退路（1.25 TB + ARC 与 s4 差两个变量）；改为两轮 1800 s 单变量对照 s6c（ARC）、s7（1.25 TB），日志实时落 run 目录 | s5 无有效结果；单变量才能归因 |
| D17 | 00:10 | KF 移植顺序：#47897（paged MQA logits）先做；#47278 / KF sparse MLA 不移植 | 微基准：vLLM 默认 asm sparse MLA 已比两者快 1.4–2 倍；#47897 比 aiter 快 1.9–2.3 倍 |
| D18 | 01:40 | ARC 不用（s6c 比 s4 低 10%）；decode 侧加做 MoE 通信调查（默认 / EP / EP + MoRI，排在 #47897 A/B 之后） | s6c decode trace：MoE 通信占每步 26%，远大于 KF kernel 的份量 |
| D19 | 02:05 | 路由容器加 `--ulimit nofile=1048576`；s7b 重跑 1.25 TB（带路由描述符监控），排在 #47897 A/B 之后、decode 通信调查之前 | s7 也在预热中止，路由日志显示描述符耗尽（s5 同因）；prefill 是吞吐瓶颈，容量结论优先 |
| D20 | 03:30 | A 线（decode kernel / 通信：#47897 端到端、MoE 通信、#47872 / #47276）交给新 session 在 v3-240（295840）上做；v3-135 撤掉 decomm3，继续 P4D4 计分线；B 线（GEM + offload 故障、U11）等 295841 | 你的决定；交接文档 `vllm-kf-port-20261117/HANDOFF-trackA.zh-CN.md` |
| D21 | 04:00 | s8 = s7b + P / D 各 4 个 API server | s7b：路由测的 prefill 段 22.8 s、TTFT 32.4 s，引擎内只有 4.5 s / 1.3 s，单 API server 是队列 |
| D22 | 04:20 | B 线（GEM + offload 故障 U2、TieringOffloadingSpec PP 问题 U11 + NVMe 二级层）交给新 session 在 v3-258（294784，我早先提交的第二节点作业，03:00 起）上做，用 `q2.sh` / `queue-n2/`；主 session 只用 v3-135 | 你的决定；交接 `vllm-p4d4-crusoe-20261117/HANDOFF-trackB.zh-CN.md` |
| D23 | 04:52 | s9 = s7b + 路由 round-robin（不 tokenize）；更正 D21 的判断 | s8 中 P 的 API server 空闲、路由单核 100%，py-spy：87% 在 kv-aware 策略的 tokenize |
| D24 | 05:28 | s10 = s7b + Rust 路由（kv-aware，sglang 同参数）+ 新加的 Rust vLLM 串行路径，排在 s9 之后 | Python 路由的根本问题是单线程 tokenize；Rust 路由是 sglang recipe 用的，只缺 vLLM 串行 |
| D25 | 07:07 | s11 = s10 + P / D 各 `--api-server-count 4 --renderer-num-workers 8`，排在 s10 之后 | s10 运行中按线程看 CPU：decode 单个 API server 有一个线程 91%（与 sglang 单 detokenizer 同类），Rust 路由只用 24%；s8 加 API server 时被 Python 路由瓶颈掩盖 |

## 时间线

- 17:14 节点体检通过（amdgpu 6.16.13，8 卡显存 0%，Docker 正常）；拉 v0.34.0 镜像，模型拷到本地 NVMe（82 s）。
- 17:39 / 17:53 **聚合正确性通过**：TP4 bf16 KV 无 MTP；DP4 + FP8 KV + MTP3（GSM8K 197/200，16k 32/32 + 31/32，接受长度 3.50）。
- 18:12 **P4D4 PD 门禁通过**（Infera router + vLLM PP4 prefill + DP4 decode + Mooncake RDMA 回环 + GEM KV）。
- 18:14–19:05 s1 计分：预热中颠簸，中止（D5）。
- 19:06 s2（CPU 层 + 缓存感知准入）：prefill 起栈 472 s（锁定 1 TB 主机内存），smoke 的 66k 冷请求时 prefill PP1 `HSA_STATUS_ERROR_MEMORY_FAULT`。
- 19:23–19:29 隔离：
  - `swap_blocks_batch`（offload 拷贝，ROCm 上是 `hipMemcpyBatchAsync`）在普通 / GEM 显存 × 锁页 / `hipHostRegister` 主机内存的所有组合、
    8192 个描述符分布在 64 GB 缓冲区上，都正常（`../vllm-pd-tools-20261117/scripts/swap_fault_test.py`）。
  - 聚合模式 PP4 + DSA + FP8 KV + offload 100 GB（带 / 不带 MTP）smoke 5/5、无故障。
  - 19:29 起：offload 1 TB（无 GEM）与 GEM Mooncake + offload 100 GB 分开测。
- 19:41–20:17 s2b（prefill 无 GEM + CPU 层 + 缓存感知准入）：smoke 6/6、offload 正常；预热 17 min 后 PD 死锁，中止。
- 20:19 s2c（prefill FCFS + CPU 层，decode 显存目标 0.95）。
- 20:57–21:36 s2c 测量：稳态命中 89%（sglang 96%）；21:32 prefill 静默卡住，21:36 引擎 RPC 超时退出，AIPerf 拒收。
- 22:00–22:12 s3（2 TB CPU 层）起栈失败（worker SIGKILL）。
- 22:17 s4 起。
- 22:53–23:23 **s4 测量（1800 s）= 26,541 tok/s/GPU**（sglang 53,351 的 49.7%）：req/s 2.27，TTFT p50 22.5 s，ITL p50 16 ms，prefill 命中 87.5%；
  AgentX 聚合因缺 `KV_OFFLOAD_BACKEND_METADATA` 失败，事后重跑聚合（`agentx_post.sh`）。
- 23:26 s5 起（1.25 TB CPU 层，3600 s），23:38 AgentX 开始；s6（FS 二级层）排在其后。
- 23:56 s5 预热第 18 min 一条请求返回空内容，AgentX 中止（P4D4 README 1d）；引擎日志被下一轮覆盖。
- 23:57 GPU 空档跑 kernel 微基准（KF README 3）：sparse MLA 保持 vLLM 默认 asm；#47897 快 1.9–2.3 倍。
- 23:58–00:03 s6（FS 二级层）起栈失败：vLLM `TieringOffloadingSpec` 在 PP4 下各 stage 区域大小不一致（U11）。
- 00:08 s6c（ARC）/ s7（1.25 TB）入队，各 1800 s。
- 00:25 #47897 覆盖层函数级测试通过（含 CUDA graph 重放）。
- 00:51–01:21 s6c（ARC）测量 = 23,864 tok/s/GPU（s4 26,541），保留 LRU；decode trace：MoE 通信 26%、indexer 5%（关键路径 rank 更多）。
- 01:24 s7（1.25 TB）起；之后排 #47897 聚合 A/B（门禁 + decode ITL）、decode MoE 通信对比。
- 01:54 s7（1.25 TB）预热中止：Infera 路由 `Too many open files`（容器 nofile 1024）→ prefill 建连 502 → 根请求失败（s5 同因）。修 `pd1_up.sh`。
- 01:56 #47897 聚合 A/B 起；02:05 s7b、decode 通信调查依次排队。
- 02:20–03:00 #47897 第二次聚合 A/B：门禁 198/200，但引擎里新 kernel 仍未生效，结果无效（KF README 4.0b）。
- 02:57 s7b（1.25 TB + 路由 nofile 修复）起，03:18 测量；路由描述符 30 min 内涨到约 1,190（超过原 1,024 上限），后段增长放缓。
- 03:30 A 线交接给新 session；decomm3 撤销。
- 03:18–03:48 **s7b = 30,091 tok/s/GPU**（1.25 TB，路由修复后跑完），命中率接近 sglang；TTFT 主要耗在单个 API server 上。
- 04:00 s8（4 个 API server）起。
- 03:00 第二节点作业 294784 在 v3-258 起（当时没注意到，空闲约 1 h）；04:07 投递镜像构建（提交 0c5e5e89）与模型拷贝。
- 04:20 B 线交给新 session（v3-258）。
- 04:35 s8 测量；04:44–04:51 py-spy：P 的 API server 空闲，路由单核 100%、87% 在 tokenize（P4D4 README 1i）；更正"API server 是瓶颈"。
- 04:52 s9（路由 round-robin）入队，接在 s8 之后。
- 05:00 合并 A / B 线结论：B 线定位 U2（clr `deviceId`，GEM 分配器改 VMM 导入，P4D4 GEM 0.88 验证中）；A 线 #47897 引擎内 decode 每步 −3.9%，
  U13 根因（FULL 捕获无 batch descriptor），#47872 不采用。
- 04:35–05:05 s8（4 个 API server）= 31,384 tok/s/GPU（s7b 30,091）；05:08 s9（路由 round-robin）起。B 线交接文档加路由瓶颈提醒。
- 05:13 Rust 路由加 vLLM 串行路径，编译 59 s（`bin/infera-router-vllmserial`）。
- 05:22 s9 路由起不来（漏了必填的 tokenizer 参数），05:26 手动重起路由，05:27 AgentX 开始；`pd1_up.sh` 修正并加路由退出即失败。
- 05:28 s10（Rust 路由串行）入队。
- 05:44–06:14 **s9（路由 round-robin）= 36,923 tok/s/GPU**（sglang 的 69%），TTFT p50 6.8 s；新的排队点在 decode（均值 6.3 s，round-robin 不看负载）。
- 06:29 s10（Rust 路由串行）冒烟通过、AgentX 开始。
- 06:53 会话在新登录节点（slog-007）恢复：此前在登录节点上读大 JSON 的分析可能占满内存；之后分析全部经 `q.sh` 在计算节点执行，登录节点只跑监控。
- 07:06 s10 运行中按线程采样：decode API server 最忙线程 91%、decode 4 个 EngineCore 主线程各约 99%、prefill API server 最忙线程 49%、Rust 路由 24%。
- 07:07 s11（前端扩展：4 API server + renderer 8 线程）入队。
- 06:47–07:17 s10（Rust 路由串行）= 36,265 tok/s/GPU（s9 36,923，波动内），TTFT 长尾更好；decode 排队 5.1 s 仍在（不是 round-robin 造成）。07:21 s11 起。

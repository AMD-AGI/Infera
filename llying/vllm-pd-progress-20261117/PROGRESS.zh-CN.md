# 进度与自主决定记录（2026-11-17 17:11 起；11-18 15:30 起新主线 session，用户离开约 12 小时）

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
| D26 | 07:28 | s12 = s11 + prefill GEM 0.88（B 线 VMM 导入库）+ decode #47897（A 线 v2 覆盖层 + 组大小 4） | 两项都已单独验证（+10.3% / decode 每步 −3.9%），组合配置 |
| D27 | 08:11 | s12b = s12 + `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`（P / D），排在 s12 后；s11 的结果作废 | s11 因 Rust 路由向 decode 写入已关闭连接（502）中止；reqwest 空闲 90 s 对 uvicorn 5 s |
| D28 | 08:22 | s12b 照跑；s12c = s12b + prefill 单 API server，排其后 | s12 的 prefill EngineCore 段错误一次，原因未定；用 s12c 区分 prefill 前端扩展的影响 |
| D29 | 10:35 | 组合配置暂不用 prefill GEM；s13 = s10 + decode #47897 + keep-alive；主线交接给新 session（`HANDOFF-main.zh-CN.md`） | prefill GEM 0.88 在 v3-135 上 3 次挂（s12 / s12b / s12d）；294199 即将到期 |
| D30 | 15:34 | （新主线 session）放弃 v3-106：排除该节点、取消 296460，重新申请 agentx-qos 两个节点（296780 → `queue-n3`，296781 → `queue-n4`）；hold 脚本启动检查加真实的 `docker pull` + `docker run` | v3-106 的 NVMe 卷在 dockerd 启动后才挂到 containerd 数据目录上，拉取 / 构建 / 起容器全部失败，修要重启 dockerd（root），不绕权限 |
| D31 | 15:37 | 另交一个 amd-burst-qos 作业（296788 → `queue-n5`，可被抢占）兜底，预先放好节点准备与 s13 复跑；第二个 agentx-qos 节点先并行跑 P4D4 对照，之后用于 P8D8 | agentx-qos 组额度满，我排第 4、5 位，按前面作业剩余时间估计要等 2–10 h；B 线的 burst 作业约 1 h 排到 |
| D32 | 15:40 | 之后各轮统一用 `scripts/p4d4_main.sh`（s13 参数化模板，自动取节点 IP），默认给 prefill / decode / 路由 / AgentX 开 `PYTHONFAULTHANDLER=1`、记录进程列表与内核日志 | 写死 IP 的模板换节点要改；U21 的崩溃需要每轮都能归因，faulthandler 只在致命信号时输出，不影响性能 |
| D33 | 15:40 | 计划顺序：m1 = s13 原样复跑（节点体检 + 跑间波动）→ m2 = s13 + A 线 v3comm → prefill 崩溃排查（tokenizers 压测、faulthandler 下的 prefill GEM 0.88）→ 最好配置 3600 s × 2 → P8D8 | 用户给的顺序；A / B 结论先合进共享文档 |
| D34 | 15:44 | m1 改为测 3600 s | s13 与 sglang 的 req/s 只差 2%，每请求 token 数差 22%；测量时长影响每请求 token 数，3600 s 才与 sglang 同口径，前 1800 s 的窗口可与 s13 比波动 |
| D35 | 16:00 | m2 起打开逐请求 prefill 命中日志（Rust 路由 `INFERA_LOG_PREFILL_USAGE=1` + prefill `--enable-prompt-tokens-details`） | s13 复盘：prefill 比 sglang 多算 40%，集中在冷请求的跨会话共享；要逐请求核对。日志只多一行，对性能无影响 |
| D36 | 16:32 | 在 m3（prefill GEM）之前插入 m4 = m2 + 两边 `--renderer-num-workers 4` | m1 实测 prefill 每请求渲染 1.35 s（decode 0.24 s），成本是会释放 GIL 的 Rust 分词；收益大概率为正、风险低；burst 节点随时可能被抢占，先跑把握大的 |
| D37 | 16:45 | 写好 P8D8 编排（`scripts/p8d8_run.sh`：prefill 节点上的队列项，经共享目录让另一节点起 DP8 decode；CPU 层 1.6 TB，C208），等第二个节点 | agentx-qos 两个作业仍排队（组额度满） |
| D38 | 17:22 | burst 节点被抢占后：重交两个 burst 作业（296937 → `queue-n5`，296938 → `queue-n6`），测量缩短到 1500 s；n5 跑 m5 = v3comm + 两边 renderer 4 + 逐请求日志，接着同节点 m7 = m5 + decode `--max-num-batched-tokens 1024`（KV 容量）；n6 跑 m6 = m5 + prefill GEM 0.88 | m1 30.8 min 时被抢占（burst 约 1 h）；m1 与 s13 同配置差约 10%，差在 decode（慢 8% → 排队 ×2.6），所以 decode 侧改动放在同一节点上比；prefill 侧指标跨节点稳定，GEM 可以在另一节点上比 |
| D39 | 18:31 | agentx 节点 296780（v3-006，`queue-n3`，24 h）18:05 起；第一轮用 m6 配置测 3600 s（与 sglang 同口径的成绩） | m6 = 43,555（1500 s），稳态 48.6K，三项改动的分段指标都向好，GEM 未崩 |
| D40 | 18:35 | 改为 CPU 层 1.6 TB 再跑（m8b；停掉刚部署 1.5 min 的 1.25 TB 版 m8） | 逐请求数据：长上下文会话间隔数分钟后整段丢失，占现算 26%，是容量问题；1.25 TB 时主机 `MemAvailable` 仍有 1.3 TB |
| D41 | 18:50 | m8b 被 OOM 杀后，查明 CPU 层在 ROCm 上经 torch 锁页分配器按 2 的幂取整；写精确锁页覆盖层（mmap + `host_register`），先在空闲 GPU 上做最小测试，再上 m10 = m6 配置 + 覆盖层 + CPU 层 1.9 TB（3600 s），同时挂内存守护 | 加大 CPU 层是解决长会话整段丢失的直接办法；不修取整问题，1.25 TB 以上就要付双倍内存 |
| D42 | 19:42 | decode `--max-num-batched-tokens 1024` 作废；n3 上排在 m10 之后的 m11（含该设置）改为 m12 = m10 原样重复（3600 s） | m9b 测量第 12 min decode 在 885 token 批次上 hipBLASLt / HSA 资源不足崩溃 |
| D43 | 20:36 | m10（3600 s）= 48,572（sglang 的 91%）后，n3 队列：m12（m10 重复，3600 s）→ m14（decode 用 vLLM 内部 DP 均衡，1800 s）；burst 不再用于计分 | m10 的瓶颈转到 decode：rank 间 KV 不均（一个满、一个 78%），路由按输入 token 预先登记选 rank；burst 节点 23–70 min 就被抢占 |
| D44 | 21:30 | agentx 作业 296781（v3-124，`queue-n4`）20:51 起（我 21:30 才发现）。两台并行：n4 m14（decode vLLM 内部 DP 均衡，1800 s）；n3 m12 之后 m13（路由请求权重 10 万，1500 s）；两者结束后 P8D8（n3 prefill PP8 + n4 decode DP8，C208，1800 s）；最后两台并行跑最终配置的 3600 s 与重复 | 尽量在用户回来前拿到 P8D8 和最终配置的两次 3600 s |
| D47 | 11-19 06:10 | 用户要求不再用 hold 作业占机器、不跑 benchmark 就释放：两个队列放 `STOP`，排 `finish_round.sh` 在节点上做完 f1 / f2 的全部分析，之后作业自动退出；以后计分提交"跑完就退出"的作业 | 集群规则（严格执行）；11-18 22:47–11-19 06:04 两台节点空闲约 7 h（本 session 在 22:37 后没有继续推进），正是这条规则要杜绝的情况 |
| D46 | 11-19 06:04 | 最终配置 f1（n3）/ f2（n4）同时跑 3600 s：m10 配置 + decode 内部 DP 均衡 + decode 显存目标 0.93 + `hipHostAlloc` 精确锁页 | m14（内部 DP 均衡）1800 s 49,183、p90 全面改善；m13 与 m9b 同一 decode 崩溃签名，三次故障更像 decode 显存余量不够（0.95 只留约 14 GB），0.93 同时验证这一点；两台各一次即成绩加重复 |
| D45 | 22:00 | 精确锁页覆盖层从"匿名 mmap + `hipHostRegister`"改为直接 `hipHostAlloc` 精确大小分配（不可用时回落 torch 锁页）；P8D8 和之后各轮用新版，已在跑的 m13 / m14 不动 | m12（m10 重复）测量第 12–14 min decode 卡死，变慢期间主机反复出现 amdgpu `evict_process_worker`（KFD 进程驱逐）告警，只有这一轮有；注册的用户内存是唯一新的驱动层因素，改成与 torch 锁页同一机制风险最低（单次，因果未证实，README 1r） |

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
- 07:28–07:35 decode 侧分析（P4D4 README 1k）：KV 传输不是瓶颈（0.19 s / 次）；decode KV 满（p50 0.85–0.90）；权重与理论值只差约 5 GiB；
  decode 每步慢（ITL 19 对 14 ms）是下一个主攻点。
- 07:49 s11 预热中止（Rust 路由 → decode 502，连接复用竞争）；07:50 s12 起（我等待监控时匹配条件写错，08:09 才发现，期间队列照常推进）。
- 08:11 s12b（keep-alive 120 s）入队；s12 测量期间对 decode EngineCore 做 py-spy。
- 08:17 s12（组合配置）预热中止：prefill EngineCore 段错误；08:19 s12b（keep-alive 120 s）起；08:22 s12c 入队。
- 09:12 s12b 测量第 6 min prefill worker 被杀；09:37 s12c 预热中止（prefill chat 模板渲染 30 s 超时）；10:33 s12d 测量中 prefill 又挂（473 失败）。
- 10:35 s13（s10 + decode #47897 + keep-alive，无 prefill GEM）起；主线交接文档完成，新节点作业 296460（hold_n3）排队中。
- 11:07–11:37 **s13 = 42,724 tok/s/GPU**（sglang 的 80%，单次），TTFT p50 5.62 s、ITL p50 17.3 ms、6,202 / 1。
- 13:41 296460 在 v3-106 起（runner 正常，但无人投递任务）。
- 15:10 / 15:20 A 线、B 线交付（`HANDOFF-to-main.zh-CN.md`）；15:30 新主线 session 接手。
- 15:30 v3-106 体检：模型拷贝 40 s；镜像构建失败——containerd 数据目录被晚挂的 NVMe 卷遮住，`docker pull` / `run` 也失败（D30）。15:34 释放节点、重新申请；
  15:37 另交 burst 作业（D31）。v3-106 随即被另一用户的作业拿走（他们会遇到同样问题，需要报给节点管理员）。
- 17:22–18:34 两个 burst 作业：n5（v3-075）18:05 被抢占（m5 作废）；n6（v3-162）跑完 **m6 = 43,555**（v3comm + renderer 4 + prefill GEM 0.88，1500 s，稳态 48.6K），之后 18:34 被抢占（m9 作废）。
- 11-19 06:04 f1（n3）/ f2（n4）开始（3600 s）；06:07–06:10 跨节点 Mooncake 主机内存测试（正常）；06:12 两个队列放 `STOP`，f1 / f2 结束后自动释放节点。
- 22:47–06:04 两台节点空闲（本 session 22:37 后中断）。22:38 `scripts/p4d4_run.sh` 被外部清空，06:05 从仓库副本恢复。
- 22:40 **m14 = 49,183**（1800 s，decode 内部 DP 均衡）；22:46 P8D8 smoke 失败（跨节点 Mooncake 通知 QP -302）。
- 22:33 m13 decode 崩溃（hipBLASLt → `HSA_STATUS_ERROR_OUT_OF_RESOURCES`，与 m9b 同签名），22:35 结束，无效。
- 21:11–21:18 m12（m10 重复）测量第 12–14 min decode 逐渐变慢到卡死，21:18 引擎 RPC 超时退出；同时主机出现 KFD 进程驱逐告警。21:43 AIPerf 中止（26,502，无效）。
  21:45 n3 m13 开始；21:50 精确锁页覆盖层改为直接 `hipHostAlloc`（D45）。
- 21:35 n4 m14（decode vLLM 内部 DP 均衡）开始部署，21:47 路由就绪。
- 20:31 **m10 = 48,572 tok/s/GPU（3600 s，sglang 的 91%）**，TTFT 均值 5.20 s 好于 sglang；20:32 m12（重复）开始。
- 19:37 m9b（m6 + decode mnbt 1024，n6 v3-190）decode 崩溃，作废；19:42 n3 队列改为 m10 → m12（重复）。
- 18:49 m8b（1.6 TB）部署后第 14 min 主机 OOM（torch 锁页按 2 的幂取整，合计 2.6 TB）；残留 Shmem 2.5 min 内释放；m8c 被 Shmem 检查拦下。18:58 m10（精确锁页覆盖层 + 1.9 TB，3600 s）开始，CPU 层实际 1.90 TB。
- 18:05 agentx 作业 296780 在 v3-006 起；18:35 m8b（m6 配置 + CPU 层 1.6 TB，3600 s）开始。
- 16:04–16:54 m1（s13 原样复跑，v3-069，测 3600 s）：16:54 burst 作业被抢占，测量只有前 30.8 min；按 s13 口径 38,582（s13 42,724），差在 decode（见 P4D4 README 1m）。
- 15:40 A / B 结论合并进 `CANDIDATES.zh-CN.md`（U6 / U12 / U13 替换，U1 / U2 / U4 / U8–U11 / U14–U16 更新，新增 U17–U25）、P4D4 README（第 0 节重写，1b / 1e 更正）、
  KF README（4.0b 原因确认、行宽 1M、#47872 / #47276 结论）；`trace_cat.py` 换成 A 线修好的版本（MTP 下步数按采样 kernel 计；`Cijk` 正则）。

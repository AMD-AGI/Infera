# prefill 提效（在 yihou 的 llr35 配置上），10-27 起

目标：yihou 的 4+4 C96 配置（llr35，47.8–48.0k）在 138 → 140 上复现并做 PD 正确性，作为基线；在这个工作点上找 prefill 的优化项。
每项只在 4+4 C96 上测，记录收益和适用场景（用户 10-26 定的方式）。第 6、7 步（Pareto 中段按点选 K、第二版 Pareto）暂不做（用户 10-27）。

## 10-29 06:00 第 1 项结论：长节拍来自驱逐时的同步写回，usage 只是回载量的标志；撤回"驱逐已否定"

全部离线（138 空闲时只读分析，未起栈）。脚本 `scripts/beat_vs_evict.py`、`trace_compact.py`、`trace_evict_check.py`、`evict_breakdown.py`、
`cache_capacity_sim.py`、`beat_usage_window.py`，输出在 `results/beat-vs-evict-1029/`。

**结论**
1. **有排队时 PP0 超额时间的 83% / 87% 归因于真实驱逐**（E1 两次独立运行 `c104-1027T2133` / `c104-1028T0204`，占计分窗口 23.2% / 25.4%）：
   与任一级驱逐重叠的节拍占 63% / 65%，承载超额的 94% / 95%；把驱逐序列循环平移 40 次作对照只有 76% / 77%（标准差 2%，最大 80% / 83%）。
   不与驱逐重叠的节拍中位 270 / 268 ms、p90 343 / 335 ms，基本是 GPU 满速。时间滞后扫描的峰值在 −0.5 到 −1 s，与下面"切片结束晚于驱逐"的核对一致。
   归因用反事实算：与驱逐重叠的节拍若按无驱逐节拍的平均超额（24–28 ms / 拍）计，多出的部分即为归因。门槛（≥ 30% 超额）远超。
2. **usage 没有独立作用**：每个 usage 四分位里，不与驱逐重叠的节拍中位都是 260–289 ms；usage 从低到高，与驱逐重叠的节拍占比从 35–37% 升到 87–88%，
   每秒回载从 5–7 万升到 18–19 万 token。高 usage = 大量主机命中的前缀被回载并锁住；回载吃掉空闲槽，触发驱逐。
3. **更正 10-29 03:00 的"长节拍并不主要由驱逐造成"**：`evict_vs_idle.py` 用"usage 下降 ≥ 0.03"识别驱逐，但 usage = 1 −（空闲 + 可驱逐）/ 容量
   （`pool_stats_observer.py`），驱逐只是把可驱逐挪成空闲，usage 不变。它识别的 1,032 / 1,113 次下降只有 35–37% 落在真实 PP0 驱逐 0.5 s 内，
   而这些窗口覆盖 55% 的时间，比随机还低。所以"驱逐附近不比随机更容易出长间隔"和据此撤掉 E8 的理由都不成立（E8 本身也不是对的修法，见下）。
4. **一次驱逐为什么要 0.33–0.52 s**（trace 每级 16 次）：约 80% 花在 `writing_check` 里同步等写回，PP0 每次 306 ms、约 75 次 `Event.synchronize`。
   写回量是约 53 万 token × 约 14 KB（KV + indexer）≈ 7.5 GB，走 `transfer_kv_all_layer_mla_lf_pf` 拷贝 kernel，约 25 GB/s。
   PP0 每次另有 90 ms 花在 KV events 的 `record_store`。四级树是镜像的，锁步环里四级依次各驱逐一次：每次驱逐事件环损失约 0.9 s，
   约为各级驱逐时间之和（1.6 s）的 57%。
5. **为什么每次都要全量写回**：每小时写回 5.28 亿 token ≈ 驱逐 5.29 亿 ≈ 回载 4.73 亿 + 新算 0.57 亿。`unified_tree_core.py` 的 `drive_host_eviction`
   在 write_back 下，主机满时**先回收"重复项"**，也就是回载节点留在主机上的副本，按登记顺序，最早回载的最先被回收。
   所以回载上来的前缀很快就失去主机副本，几十秒后从显存驱逐时又要整段写回。
6. **GIL / 传输线程**：不与驱逐重叠的节拍只占超额的 5–6%，GIL 争用最多也就这么多，低于 1% 门槛，不再查。

**派生指标的核对**
- aiperf `server_metrics_export.json` 的每秒驱逐切片，与 trace 里 4 级共 64 次 `evict_for_alloc` 一一对上：时长完全相同，切片结束在驱逐开始后 0.3–1.4 s。
- 日志 usage 与 metrics usage 在偏移 0 时相关 0.986–0.990。
- 日志的 `Prefill batch` 行是在 `process_batch_result_disagg_prefill` 里打印的（结果绕环回来时），比该 batch 的 step 注解晚约 1.9 s。
  这不影响按实时重叠做的统计，但日志里的 usage 是结果处理时刻的值。
- trace 的 `baseTimeNanoseconds + ts` 就是 UTC 时间，可以直接和日志对齐。

**修复方向（离线判定）**
- **不做：`SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM=1`**。这是现成开关，回载节点会保留主机副本，驱逐就不用写回；但缓存变成包含式，
  唯一容量从 32.4M 降到约 23M token。按会话 LRU 模拟（`cache-capacity-sim.txt`，用 HC250 / 300 / 500 校准：模拟 6.33 / 6.09 / 5.92%，实测 5.25 / 5.13 / 5.0%），
  现在的工作点正好在拐点上方，23M 时模拟新算 12.1%，约翻倍；按实测与模拟变化幅度之比约 0.6 换算，新算量 +30–70%，抵消并超过驱逐的收益。
- **建议下一步做 E9：后台提前写回 LRU 尾部**，保持现在的容量语义。可用槽低于约 1M 时，每轮事件循环用现有的 write-through 机制
  （`_execute_and_commit_kv_backup` 加锁、非阻塞 `writing_check` 的跨级 MIN 共识都已有）异步写回下一次驱逐会选中的叶子。
  真正驱逐时这些节点已有主机副本，`_evict_device_leaf` 直接降级，不再同步写回。去重回收的第一遍本来就跳过可驱逐的显存叶子（"imminent free demotes"）。
  当前用的是 Python TreeCore（日志："Rust TreeCore support range" 不含这个 PyTorch），所以补丁只改 Python。
  - **定量预测**：每次驱逐从 0.33–0.52 s 降到约 0.05–0.16 s，只剩 `_evict_components` 的遍历和 PP0 的 `record_store`，四级之和 −79%。
    假设环损失与驱逐时间按 0.57 成比例：每小时少 670–730 s 超额，排队节拍均值 472 → 约 350 ms，prefill 有排队时吞吐约 +30%。
    代价：异步写回和 forward 并发（每级平均约 2 GB/s，拷贝 kernel 约 8% 的时间在跑）；预先写回的重复项约 0.5M（新算约 +1%）。
    端到端按 E1 的弹性（prefill +3.6% 对应端到端 +2.8%）并考虑 decode 开始成为瓶颈，**预测 ≥ +5%**。
  - **验证顺序**（方法第 3 条）：先 20 分钟短跑 + 小 trace，应看到每级驱逐均值 < 0.15 s、写回 token 不变（约 14.5 万 / s）、
    PP0 排队节拍均值 ≤ 380 ms；然后跑限池门禁（驱逐最重）；最后 60 分钟计分。
- **次要**：PP0 的 KV events（`kv_events_config` zmq）在 trace 窗口里占调度线程 6%（含一次 0.93 s，trace 开销会放大这一项）；
  单 prefill 实例下它有没有用待查，估计收益 ≤ 1–2%，排在 E9 之后。

**06:40 补充：E9 的副作用核对，以及一个更简单的候选 E10**
- E9 不是 write_through 策略。write_through 在节点插入时就写回所有新 KV，主机变成包含式，唯一容量约等于主机的 21.7M，
  模拟新算 13.5%（现在 6.3%），所以不做。E9 仍是 write_back：只把下一次驱逐要写回的那批 LRU 尾部节点提前异步写回，写回总量不变，只是时机提前、不阻塞。
- 副作用估计：
  - 窗口 0.5–1M 占掉的唯一容量：模拟新算 6.33% → 6.37–6.38%，实测量级约 +0.6% 新算。
  - 写回后、驱逐前又被命中，造成多写：模拟上限是显存 → 主机流量的 7.1%（窗口 0.5M）/ 14.6%（1M），都在后台。
  - CU 争用：写回 kernel 是 grid [2,1,1] × 1024 线程、独立的 stream 17，不到 256 个 CU 的 1%。
  - 跨级共识：`check_hicache_events` 每步已经对写回完成数做跨级 MIN，不需要新增 collective。
  - 传输期间节点被锁：从可用 < 1M 时开始触发，提前量约一个驱逐周期（约 3.7 s），远大于约 0.3 s 的拷贝时间。
  - 未量化、需要先确认的两项：每步选 LRU 尾部的 CPU 开销（必须增量做），以及改动树状态路径带来的正确性风险（E3 的教训，需要过限池门禁）。
- **E10（更简单）**：写回调用 `transfer_kv_all_layer_mla_lf_pf` 没有传 `block_quota`，默认 2，所以只开 2 个 workgroup，约 25 GB/s；
  回载方向的 kernel 是 16 个 workgroup。驱逐是同步的，写回时本级 GPU 本来就空着，加大 grid 不会和 forward 抢资源，也不改缓存语义。
  若带宽能到约 45–50 GB/s，每次驱逐的同步部分减半，按 0.57 的环损失换算约 −300 s / 小时超额，排队节拍均值 472 → 约 420 ms。
  前提是带宽确实随 `block_quota` 上升，需要先在空闲 GPU 上做几分钟 microbench（数据采集，非性能测试）确认，再决定 E10 / E9 的先后。

## 10-29 05:05 后续计划（每一步先离线定量、预测收益 > 1% 才上机器）

1. **离线（约 2 小时）**：E1 四级 trace 按时间对上每步的 prefill token usage，核对高 usage 时 GPU 步长是否不变、空闲落在哪一级哪个阶段、KV 传输线程是否与调度线程争 CPU（GIL）。
   该机制能解释有排队超额时间的 ≥ 30%（约占窗口 9%）才设计修复并写预测；否则放弃这条线。
2. **decode kernel 分解（数据采集，非性能测试，约 1 小时，140）**：verify CUDA graph 内部从未测过（yihou 也未补上）；用 rocprof 或关 graph 短跑拿到每步约 42 ms 的构成。
   decode 变快能更早释放 KV，缓解 decode prealloc 排队（70% 的秒数）。
3. **kernel（收益可离线预测）**：先从真实运行抓相邻 query token 的 top-k 索引，量重合度、算出合并 gather 能省的访存量；值得再把上游的 union 路径扩展到每卡 64 头。
   其次是 indexer logits + top-k 融合。
4. 不做：decode mem 0.96 复测（yihou 单次 +1.2%，按机制预计在噪声内，要多次才分得清）、任何为占用机器而跑的实验。
5. 建议下一阶段开新会话，以本文件最前面的汇总作为交接，减少长会话丢细节造成的错误。

## 10-29 03:40 离线分析：PD 两侧谁等谁；长节拍与 prefill 在途 KV 用量强相关

**prefill 与 decode 的排队（`scripts/pd_queue_cause.py`，E1 复测计分窗口，按秒，decode 各 rank 沿用最近一行）**：prefill `optimistic_prefill_attempts = 0`，
即 decode 预分配 KV（bootstrap）完成后 prefill 才能算这个请求。
- prefill 有可计算队列 75% 的秒数、在算但队列空 18%、空闲 7%；decode prealloc 排队 70% 的秒数（平均 6.3 个请求）。
- **prefill 空闲且 decode 在排 prealloc 只占 5%**；两边同时排队 64%。
- decode KV 用量平均 0.86，**其中 0.23 预占给还在 prefill 队列 / 正在算 / 正在传的请求**；运行中请求用约 0.63。
- 判断：prefill 的可计算队列里都是已 bootstrap 的请求（decode 已分配），**prefill 排队不是 decode prealloc 造成的**；反过来，请求在 prefill 排队和计算的时间里
  一直占着 decode 预分配的 KV（0.23），推高 decode 用量 → decode prealloc 排队。**根因在 prefill 吞吐**，decode prealloc 主要是下游结果（被"先分配再 prefill"放大）；
  decode KV 容量是第二位的限制（C112 失败即此），decode 越快、运行中请求越早释放 KV 也能缓解。

**长节拍和什么相关（`scripts/beat_features.py`，有排队的 5,854 拍，按特征四分位看超额时间的分布，均匀则各 25%）**：缓存 token、新 token、序列数、排队数、
传输中请求数都接近均匀；**prefill token usage 单调强相关**：≤ 0.12 时中位节拍 276 ms、占超额 13%；> 0.25 时中位 569 ms、占 38%。
token usage = 1 − (空闲 + 可驱逐) / 容量，即在途请求（流水线里的、算完等传完的）锁住的 KV。trace 里 GPU 每步约 250 ms 很稳，所以高 usage 时多出的是空等而不是算得慢。
下一步：用 trace 按时间把每步对上 usage，核对高 usage 时 GPU 步长是否不变、空闲落在哪个阶段，以及传输线程是否与调度线程争 CPU（Python GIL）。

## 10-29 03:15 `GPU_MAX_HW_QUEUES=8` 没有依据，已停

- 用户问依据后补做的核对（`scripts/copy_wait_blocker.py`：每个调度线程上 > 20 ms 的阻塞拷贝，看它在 GPU 上开始前刚结束的是哪条 stream 的哪个操作）：
  - E7 PP1：118 次、GPU 侧排队共 24.5 s，**117 次排在同一条 stream（stream 5）的 kernel 后面**，跨 stream 只有 1 次 → 那些约 200 ms 的等待是普通的 stream 顺序，不是硬件队列复用。
  - E1 PP1：59 次、8.0 s；25 次同 stream 等 forward（early-send），18 次等另一条 stream 上的 NCCL kernel（3.2 s，更像"收到的输出拷回前先等接收完成"这一真实依赖），6 次等其他 kernel。
- 硬件队列复用的假设不被支持，E1 + HWQ8 预期零收益 → 03:10 在预热中停掉（计分前），`engine/down.sh` 拆栈、确认两节点无残留。
- 在找到有定量预测的实验之前不排性能测试（不用重测已测配置填空）；当前工作转为离线分析长节拍的真正原因（全量日志、带随机对照）。

## 10-29 03:00 更正与方法调整

**又一条更正：长节拍并不主要由驱逐造成**（`scripts/evict_vs_idle.py`，E1 复测 `c104-1028T0204` 的 PP0 日志，按 token usage 下降识别出 1,107 次驱逐，每分钟 18 次）：
- 有排队时的长间隔超出中位的部分占窗口 **29.5%**；其中落在驱逐后 4 秒内的占 57%，**同样数量的随机时刻后 4 秒内占 70%**——驱逐附近并不比随机时刻更容易出长间隔。
- 驱逐那一步 PP0 的间隔中位 310 ms，与整体中位 320 ms 相同。
- 所以 22:00 "长节拍 = 回载触发的驱逐沿环逐级重复"只是 trace 里的一个实例，不能代表全程；**E8（空闲时提前驱逐）撤掉**。长节拍的主因仍未找到。

**队列调整（03:00）**：停掉 night18（E1 + E7d + HWQ8 预热中；E7d 既不快、生产配置下也没修好开头坏，就算恢复速度也不会采用），用 `engine/down.sh` 拆栈；
`scripts/night19_1029.sh` 只跑 **E1 + `GPU_MAX_HW_QUEUES=8`**（一个环境变量、可直接采用），约 04:40 出结果。

**方法调整**（这一轮失败的共同原因：从单个实例或汇总指标推出因果，没先用已有数据证伪就投入 95 分钟的实验）：
1. 每个实验开跑前写下**定量预测**（"去掉 X，节拍降 Y，因为在 M 个长间隔里 X 在关键路径上 N 次"），并先用已有日志 / trace **带对照地证伪**（如上面的随机时刻对照）；预测收益 < 1%（噪声）不跑。
2. 机制先在**整段运行的统计**上成立，再看 trace 实例；派生指标（如 `prefill_pp_critical.py` 的"本级延迟"）先拿几个手工核对过的实例验证。
3. 改流 / 同步 / 树结构的补丁，先做 20 分钟短跑 + 小 trace 看机制是否按预测变化，再做 60 分钟计分。
4. 正确性结论给出基数与显著性，两份独立样本才下"根因"结论。

## 10-29 02:45 汇总（10-28 17:16 起值守的这一轮，结论先行）

**结论**：吞吐最好的仍是 **E1（驱逐留 512K 余量）**：两次 49,807 / 49,631，平均比 v3 基线 48,360 高 **2.8%**；正确性证据是限池门禁（大量驱逐下长上下文干净、
探测与不带 E1 的对照同水平）——生产配置的正确性套件不触发驱逐，测不到 E1。这一轮试的其余改动都没有超过 E1：

| 项 | 改动 | C104 tok/s/GPU | vs E1 平均 49,719 | 结论 |
|---|---|---|---|---|
| E4 | 关 early-send | 49,293 | −0.9% | 不采用（节拍不变、TTFT 略差） |
| E6 | PP 异步深度 2 | 48,291 | −2.9% | 不采用（与 yihou llr15 一致） |
| E7 | PP 接收独立 stream + `record_stream` | 43,396 | −12.7% | 不采用 |
| E7d | 接收独立 stream + 延后释放 | **44,507** | **−10.5%** | 不采用 |
| E3 | 驱逐写回批量化 | 未测性能 | — | 限池门禁乱码 12（对照 4），撤掉 |
| kernel | 稀疏 MLA > 2 GB 分桶（Triton）、`--dsa-prefill-backend aiter` | microbench | — | 分桶只快约 6%（未集成）；aiter 在 > 2 GB 池和 8K token 下结果错 / NaN |

**最重要的发现**：长节拍（p75 约 600、p90 约 840 ms，约一半的 GPU 空闲）来自**回载触发的大驱逐沿 PP 环逐级重复**——PP0 一次 `get_new_batch_prefill` 里回载 + 两次驱逐
约 1.2 s，各级树是镜像的，PP1 / PP2 / PP3 依次在下一拍做同样的事，环连续约 4 拍、每拍约 1.1 s 四级同时停；平稳段环按 GPU 满速跑（`prefill_ring_timeline.py`）。
这推翻了我 08:25 写的"E1 之后驱逐不在关键路径上"。

**更正（02:45）**：23:55 写的"开头坏的根因是 PP 接收与发送共用 comm stream"**不成立**——E7d 限池门禁 0 / 0，但生产配置完整套件**乱码 5、开头坏 6**（与 E1 的 4 / 5 同水平，
`full-site-llr35-v3-evict512k-recvstream-ka-1029T0214.out`）。目前只有 E7（独立 stream + `record_stream`）两次都是 0 / 0；E7b（只 `record_stream`）、E7d（只独立 stream）
各有一次不为 0。这个 bug 时有时无（生产约 0.4%、限池约 3%），根因未定位；它在 v3 本身就有，不是本轮改动引入的。

**现在**：night18（`scripts/night18_1029.sh`）——E1 + E7d + `GPU_MAX_HW_QUEUES=8` C104（02:40 起，约 04:15）→ E1 + `GPU_MAX_HW_QUEUES=8` C104（约 05:50）
→ E8（空闲时提前驱逐，**02:45 改为叠在 E1 上**，`config/pp-idle-evict.patch`）限池门禁 + C104（约 07:50）。

## 10-27 夜：调优实验总表（结论先行，每组跑完更新）

**10-29 01:40 更新：E7d 计分中段也偏慢约 6% → 怀疑 HIP 硬件队列复用；先测 GPU_MAX_HW_QUEUES=8**
- E7d 计分约 27 分钟时完成 5,705 个请求，E1 复测同一时刻约 6,040；最近 10 分钟速率 3.73 / 3.96 请求/秒。E7d 没有 `record_stream`，所以 E7 / E7d 变慢的主因是独立接收 stream 本身。
- 假设：HIP 把软件 stream 轮流映射到硬件队列（`GPU_MAX_HW_QUEUES` 默认 4；容器里没设），prefill 进程里 stream 很多（调度、forward、copy、PP comm、NCCL 内部、HiCache）；
  新增的接收 stream 若与调度 stream 共用一个硬件队列，它一提交就在 GPU 上自旋等上游的 NCCL 接收 kernel，会挡住调度 stream 上的小拷贝——正好对应 E7 trace 里
  `kv_to_page_indices` 的约 200 ms 等待；原代码的接收排在 comm stream 后面才开始，自旋时间短。
- **队列 `scripts/night18_1029.sh`（替换 night17）**：E7d 完整正确性 → **E1 + E7d + `GPU_MAX_HW_QUEUES=8`** C104 → **E1 + `GPU_MAX_HW_QUEUES=8`** C104（基线是否也受队列复用影响）
  → E8 限池门禁 + C104。

**10-29 00:50 更新：E3 不通过、今晚撤掉；E7d 限池门禁通过、性能在跑**
- **E1 + E3（23:35 修正后的版本）限池门禁不通过**（`full-site-llr35-v3-e1-e3-gate-1028T2347.out`）：探测 0.915、**乱码 12**、跑到上限 15、开头坏 0（对照 4 / 6 / 6）；
  长上下文干净。乱码的样子和"开头坏"不同：开头正常（"The problem asks for…"），写到很长后退化成 "0. 0. 0." / "laullaul…"，多数跑到 16K——像 prompt 中间某些页的 KV 坏了。
  最可能是"重新遍历"（同一次驱逐里第二次 `evict_device_start`）破坏了 tree core 的内部状态、释放了仍在用的页。修正前的版本正确性在对照水平但驱逐会达不到目标
  （第二遍回载慢一倍）。**E3 今晚撤掉**；正确做法要先读 Rust tree core 的遍历约束，或换成"提前后台备份"的方案。night15 按规则跳过了 E3 性能。
- **E1 + E7d 限池门禁通过**（`full-site-llr35-v3-evict-recvstream-ka-gate-1029T0011.out`）：探测 0.975、**乱码 0、开头坏 0**、跑到上限 3，长上下文干净——接收独立 stream
  修掉"开头坏"的第二个独立样本，且不用 `record_stream`。**E1 + E7d C104 00:42 起栈，约 02:12 出结果。**
- **E8：空闲时提前驱逐**（kit `patches/pp-idle-evict` = E7d 的 `scheduler_pp_mixin.py` + `config/pp-idle-evict-on-e7d.patch`，39 行）：事件循环一轮里没有任何
  microbatch（`server_is_idle`）且空闲 token 槽 < 76.8 万时，驱逐到约 105 万空闲（约设备池的 10%）。驱逐路径不变，只把时机挪到环本来就闲着的时候
  （prefill 18–37% 的时间没有排队）；"空闲"与空闲槽数在各级镜像（各级驱逐次数相同可佐证），各级做同样的决定。`rank_consensus` 只是调试检查器（默认关）。
- **队列 `scripts/night17_1029.sh`（替换 night16；night15 之后）**：E7d 完整正确性 → E8 限池门禁 → E1 + E7d + E8 C104（与 E7d 的 C104 对比）→ E7d 复测或 E1 第三次复测。

**10-28 23:55 更新：找到"开头坏"的根因所在——PP 接收与发送共用 comm stream 的执行顺序；E7d 兼顾正确与性能**
| 限池门禁（300K token / 级，200 题 + 两遍 16k × 128） | 探测正确 | 乱码 | 开头坏 | 跑到上限 | 长上下文 |
|---|---|---|---|---|---|
| 对照（v3，不带 E1） | 0.965 | 4 | 6 | 6 | 干净 |
| E1 | 0.975 | 4 | 6 | 6 | 干净 |
| E1 + E7b（只 `record_stream`） | 0.960 | 4 | 8 | 6 | 干净 |
| E1 + E7b + E3（修正前） | 0.955 | 5 | 6 | 9 | 干净 |
| **E1 + E7（接收独立 stream）**（`full-site-llr35-v3-evict-recvstream-gate-1028T2330.out`） | **0.985** | **0** | **0** | 2 | 干净 |

- 加上 E7 完整套件的 0 / 0（E1 4 / 5），**"开头坏"是 PP 的接收排在同一条 comm stream 上的顺序造成的，不是内存生命周期**（E7b 无效）。具体是哪一个依赖导致坏数据
  还没定位（候选：接收的 `work.wait()` / 事件记录与本级发送、forward 栅栏在同一 stream 上的先后），但把接收挪走就消失了，可以作为修复。
- E7 性能 −12.7%，怀疑来自 `record_stream`（分配器在显存紧时同步等事件）。**E7d = 接收独立 stream + 延后释放（不用 `record_stream`）**
  （kit `patches/pp-recv-stream-keepalive`，`config/pp-recv-stream-keepalive.patch`）。独立 stream 下仍需延后释放：张量在接收 stream 上分配、在别的 stream 上读。
- **队列 `scripts/night15_1029.sh`（替换 night14；它的 E1 + E3 限池门禁让它跑完、由 night15 判定）**：E3 完整正确性 → E1 + E3 C104（约 00:45–02:20）
  → E7d 限池门禁 → E1 + E7d C104（约 02:50–04:25）→ E3 四级 trace。

**10-28 23:35 更新：E1 + E7b + E3 限池门禁——正确，但第二遍长上下文慢一倍（警示）**
- `full-site-llr35-v3-e1-e7b-e3-gate-1028T2308.out`：smoke 6/6；探测 0.955、乱码 5、开头坏 6（对照 4 / 6，同水平）；两遍 16k × 128 捞针 64/64、长前缀 GSM8K 61/64，
  乱码 0 → **E3 在大量驱逐下不破坏正确性**。
- 但第二遍（大量回载第一遍被驱逐的前缀）用时 2 分 24 秒，E1 门禁 1 分 01 秒、E7b 门禁 1 分 11 秒；第一遍三者都约 2 分 46–55 秒。一个样本，可能是批量化改变了
  驱逐顺序（先驱逐一批叶子、父节点留到后面），第二遍要回载的更多。night14 的 E1 + E3 门禁是第二个样本，C104 性能是真正的检验。
- **找到 E3 的一个真问题并修了（23:35，`config/evict-batch-on-e1.patch` 144 行）**：原逻辑每个叶子立即 demote，父节点马上成为叶子，游标（父节点在 LRU 里
  通常排在子节点之后）稍后就能选到它；批量化要攒到 32 个才 demote，游标可能已越过父节点，一轮遍历达不到目标——回载时空间不够就放弃回载、
  改为重算整段前缀，正好对应"第二遍（回载最多）慢一倍"。修正：一轮遍历结束时若还没到目标、且本轮释放过节点，就重新遍历（每轮至少释放一个，必然终止；
  批量关闭时只走一轮，与原逻辑相同）。23:08 那次门禁用的是修正前的版本；night14 的 E3 门禁 / 完整正确性 / 性能都用修正后的版本。

**10-28 23:20 更新：E7b 限池门禁 = 对照 → 收到张量的生命周期不是"开头坏"的原因**
- E1 + E7b（只加 `record_stream`）限池门禁（`full-site-llr35-v3-evict-recvrecord-gate-1028T2244.out`）：探测 0.960、**乱码 4、开头坏 8**（对照 4 / 6），
  长上下文两遍干净。→ 我"内存被提前复用"的假设不成立；E7c（延后释放，同一思路）撤掉。E7 完整套件的 0 / 0 要么来自"接收挪到独立 stream"的时序改动，
  要么是运气（不限池时这个 bug 本来就时有时无：v3 参考 0、E1 5）——night14 先跑 **E7 限池门禁**（对照稳定在 2–4%）来区分。
- night12 停掉时它已开始第二项 E1 + E7b + E3 的限池门禁（23:08），让它跑完：正好检验 E3 在大量驱逐下的正确性。
- **队列 `scripts/night14_1028.sh`（替换 night13）**：E7 限池门禁 → E1 + E3 限池门禁 → E1 + E3 完整正确性 → E1 + E3 C104 → E1 + E3 四级 trace。

**10-28 23:10 更新：E7 慢在哪；正确性修复改为 E7c（延后释放）；E3 单独测**
- E7 四级 trace（`results/trace-llr37-v3-recvstream-4r-c104/`）：GPU 每步中位 249.6 / 242.7 / 244.9 ms（E1 254 / 243 / 246），**forward 没变慢**，"自旋抢 CU"不是原因。
  新出现的是调度线程上的长阻塞：`process_prefill_chunk → send_kv_chunk → kv_to_page_indices` 的 `hipMemcpyWithStream` PP0 61 次平均 222 ms、PP1 80 次平均 198 ms
  （E1 同处 18 次共 15 ms），其他 `send_kv_chunk` 也变长；PP1 阻塞调用占 48%（E1 27%）。这条路径没有显式等 forward stream（v3 只是记事件给传输线程）。
  推测：`record_stream` 让每个 microbatch 约 100 MB 的 hidden states 在 forward 结束前不能复用，prefill 显存紧，缓存分配器拿不到块时同步等这些事件
  （即等 forward 结束），于是调度线程在随机的分配 / 拷贝处被卡住。**E7b 同样用 `record_stream`，可能同样变慢**。
- **E7c（`config/pp-recv-keepalive.patch`）**：不用 `record_stream`，收到的张量保留最近 2 × (pp_loop_size + 2) = 14 次接收（约 7 轮）再释放——每轮都会同步等
  第 j−4 个 microbatch 的输出，说明本级更早的 forward 都已结束，保留更久就一定安全；没有事件，也不进分配器的同步路径，多占约 1 GB 显存。
- E3 改为**叠在 E1 上单独测**（`config/site-llr35-v3-e1-e3.env`），不再叠 E7b，避免被它混淆。
- **队列 `scripts/night13_1028.sh`（替换 night12；night12 第一步 E7b 限池门禁让它跑完）**：E7c 限池门禁 → E3 限池门禁 → E3 完整正确性 → E1 + E3 C104（约 00:30–02:05）
  → E7c 完整正确性 → E1 + E7c C104（约 02:30–04:05）。

**10-28 22:00 更新：长节拍 = 回载触发的驱逐沿环逐级重复；E3（批量写回驱逐）已排队**
- `scripts/prefill_ring_timeline.py`（E1 四级 trace，把四级的 GPU 区间和调度线程阶段按时间排开，`results/trace-llr37-v3-evict512k-4r-c104/ring-timeline-{16,31}.txt`）：
  - **平稳段（第 31–37 步）环按 GPU 满速跑**：PP0 GPU 每步约 255 ms、前后间隔 < 3 ms，PP1 间隔 9–14 ms；early-send 的同步只是等上一个 forward，等完 3 ms 内下一个就上 GPU。
  - **停顿波（第 18–22 步）**：PP0 第 18 步后 `get_new_batch_prefill` 用了 1,214 ms——`init_load_back`（631 ms）里为腾空间驱逐一次（560 ms），回载后分配时再驱逐一次（560 ms）；
    树在各级镜像，**PP1 下一拍做同样的事（903 ms），PP2、PP3 依次**，四级每拍都在同一时刻开始、周期约 1.1 s，持续约 4 拍。这就是 p75 / p90 的长节拍和约一半的空闲。
  - **更正 08:25 第 1 条"E1 之后驱逐基本不在关键路径上"——错**。`prefill_pp_critical.py` 把源头级的停顿归成了别的类别；逐段时间线不依赖这种推断。
- **E3：驱逐的写回批量化**（kit `patches/evict-batch/unified_radix_cache.py` = E1 的文件 + `config/evict-batch-on-e1.patch`，`SGLANG_EVICT_BATCH_NODES=32`，0 = 原行为）：
  `_evict_components` 原来每个牺牲节点"提交 D2H → 写栅栏 → 同步等 → demote"（一次驱逐约 116 个节点串行）；改为每 32 个（或待释放量已够目标）flush 一次：
  一次 `writing_check`（`start_writing` 把队列合并成一次传输）后逐个 demote。遍历是 LRU 游标、未备份的叶子在 demote 前仍是叶子、父节点 demote 后才可选，
  已在批里的节点跳过；各级跑同一套确定性逻辑，镜像不变。预期一次驱逐从约 560 ms 降到约 250–300 ms（拷贝本身与遍历重叠），停顿波减半。
- **队列 `scripts/night12_1028.sh`（替换 night11c）**：E7 trace 之后 → E7b 限池门禁 → E3 限池门禁 → E1 + E7b + E3 完整正确性 → 其 C104 性能（约 02:10–03:45）→ E1 + E7b 单独 C104 性能（分离 E3）。

**10-28 21:45 更新：kernel 线——`--dsa-prefill-backend aiter` 不可用；NCCL 自旋占 CU 很少**
- 调度层 E4 / E6 / E7 都没进展，按用户要求转 kernel。镜像里的 DSA prefill 后端：`triton`（生产）、`triton_sparse_mla`（有前缀时先把 KV 反量化成 bf16；
  `--dsa-triton-union` 要求"头数 × G"在 [16, 32]，TP1 每卡 64 头不满足）、`aiter`（`_forward_aiter_extend`：每个 prefill token 当一个 decode query 调汇编
  `mla_decode_fwd`，`mla_a16w8_qh16_..._ps.co`）、`tilelang`（CUDA）。
- aiter microbench（`scripts/bench_aiter_sparse_prefill.py`、`debug_aiter_sparse_prefill.py`，`results/*aiter_sparse_prefill*-1028T21*.txt`，140 GPU 0）：
  64 个 token、池 ≤ 370 万行（≤ 2 GB）时与 torch 参考 max|diff| 0.0011（Triton 0.0028），**池 1,070 万行（生产 6.2 GB）时 0.0407（错）**——汇编 kernel
  同样是 32 位偏移；**8,192 个 token 一次调用则输出全 NaN、0.14 ms（没算）**，1.9 GB 池也一样（持久化调度在大 token 数下失效）。→ 不改汇编不可用。
- NCCL kernel 形状（`scripts/nccl_kernel_shape.py`，E1 PP0）：PP 收发 kernel 是 8 个 block（256 线程），与 forward 重叠 6.2 s / forward 32.7 s，
  平均占 CU 约 0.6%——限制 RCCL p2p channel 数收益很小，不排；也说明 E7 的 −12.7% 不能用"接收自旋抢 CU"解释（最多约 3%），等 E7 trace。

**10-28 21:35 更新：E7 性能 −12.7%，不采用；只保留它的正确性部分（E7b）**
- E7（`results/run-llr37-v3-evict512k-recvstream-c104.txt`）= **43,396**（E1 平均 49,719）：prefill 节拍中位 / p90 368 / 1,145 ms（E1 323–328 / 820–836），
  有排队的区间 93%（E1 78%），完成 10,710（E1 11,852），1 个请求错误。**我对 NCCL 链的判断方向错了**：接收挪到独立 stream 后环更慢。
  最可能的原因：接收 kernel 一提交就在 GPU 上自旋等发送方，与本级正在跑的 forward 抢 CU。E7 四级 trace（21:25 起）会给出 GPU 每步时间来确认。
- F1（建在 E7 的接收 stream 上）撤掉。night11c（`scripts/night11c_1028.sh`，替换 night11b）改测 **E7b = 只给收到的 PP 张量 `record_stream`**
  （`config/pp-recv-record.patch`，7 行，通信时序不变）：限池门禁 → 完整正确性 → C104 性能。

**10-28 20:00 更新：E7 通过完整正确性，而且很可能修掉了栈里一直存在的"开头坏"**
| 完整套件（不限池，1,319 题探测） | 正确 | 乱码 | 跑到上限 | 开头坏（真） | 16k × 128 | 100k × 32 |
|---|---|---|---|---|---|---|
| E1（`full-site-llr35-v3-evict512k-1028T1708`） | 0.963 | 4 | 22 | 5 | 64/64、62/64 | 16/16、16/16 |
| **E1 + E7**（`full-site-llr35-v3-evict512k-recvstream-1028T1922`） | **0.973** | **0** | **9** | **0**（检测报 1 条是 "There are…" 的误报） | 64/64、62/64 | 16/16、15/16 |

乱码 + 跑到上限 26 → 9，不像随机波动。推测机制：原代码在 `pp_comm_stream` 上收的张量没有 `record_stream`，被 forward / copy stream 读的同时，
内存块一释放就可能被同一 stream 上下一次接收（例如第 8 步收"输出"）复用——上一个 microbatch 的 forward 还在读它的 hidden states，
该级这几个 token 的 KV 就算错了；错在 prompt 末尾时，decode 头几个 token 乱、之后又回到题目上（正是"开头坏"的样子）。原代码里"接收排在
本级 forward 完成事件之后"的顺序只保护了一部分情况。E7 把接收挪到独立 stream 并对收到的张量 `record_stream(forward, copy)`。
- **验证**：night11b（`scripts/night11b_1028.sh`，替换了 night11 的 KV 校验诊断——这种机制下 prefill 算出的 KV 本身就错，校验值也是对着错数据算的，
  校验查不出来）先跑 E7 的限池门禁（同条件下对照与 E1 都是 4 乱码 / 6 开头坏 / 200），之后跑 F1（E7 + 输出到达即转发）的完整正确性与性能。
- 不管 E7 性能如何，**它都应该作为正确性修复保留**（待限池门禁确认）。E7 性能 19:50 起跑，约 21:50 出结果。

**10-28 19:20 更新**
1. **E6（E1 + PP 异步深度 2）= 48,291，比 E1 两次平均 49,719 低 2.9%，不采用**（`results/run-llr37-v3-evict512k-depth2-c104.txt`）：PP0 节拍中位 342 ms
   （E1 322–329），完成请求 11,643（E1 11,852）。与 yihou llr15 的 −1.5% 一致。**E4（去 early-send）和 E6（多一个在途 microbatch）都不改变节拍**
   → 中位节拍不是"环的在途数"限制的，而是每级每轮有一次 CPU 阻塞要等本级上一个 forward 跑完。
2. 机制修正：ROCm 上写后读屏障默认**关**（`scheduler.py` `_war_barrier_enabled = is_cuda() or SGLANG_ENABLE_WAR_BARRIER`），所以不是它；
   `_pp_launch_batch` 在 `forward_stream_ctx` 里调 `run_batch`，`run_batch` 里任何 `.cpu()` / `.item()` / `.tolist()` 都排在 forward stream 上、
   等上一个 forward 跑完——early-send 的 89–140 ms 就是这个。**但定位结果（`scripts/prefill_fwd_waits.py`，`results/trace-llr37-v3-evict512k-4r-c104/fwd-waits-pp*.txt`）
   推翻了"每轮都等本级 forward"**：PP1 / PP2 约 95 步里只有 3–5 步有这种等待（都是驱逐 / 回载）；PP0 / PP3 的 early-send 等待（25–27 步）结束后约 2 ms
   下一个 forward 就上了 GPU，不造成空闲（与 E4 无收益一致）。→ PP1 / PP2 的空闲是等上游数据 / 发送被下游卡住，即 GPU 侧通信链——E7 的目标
   （按 NCCL 时序估算平均每圈约 290 ms、节拍约 −10%）。
3. 18:17:20–18:17:35 我在 138 上起了 4 个 trace 分析进程，发现 E6 正在计分后立刻停掉（15 秒，nice 10）；之后的分析都放在非计分时段。

**10-28 17:50 更新（用户 17:16 起要求全程值守、以最大吞吐为目标）**
1. **E1 生产配置完整正确性（不限池，`gate/full-site-llr35-v3-evict512k-1028T1708.out`）**：smoke 6/6；16k × 128 捞针 64/64、长前缀 GSM8K 62/64；
   100k × 32 16/16、16/16，乱码 0；**但 1,319 题探测 0.963、乱码 4、开头坏 5**（"The!!!!!!"、"The了多少"、"The77getParent"……；v3 参考 10-27 是 0.972 / 0 / 0）。
   **与 E1 无关的证据**：整次运行 prefill 池用量最高 0.02，`evict_for_alloc` 只在容量缺口时调用（`allocator/base.py:166`、`unified_radix_cache.py:1857`），
   E1 改的就是这个函数，所以整次运行 E1 的代码一次都没执行。→ **开头坏是当前栈本来就有的低频 bug（约 0.4%），10-27 参考的 0 / 1,319 是运气**；
   这也说明生产配置的正确性套件根本测不到驱逐路径，E1 的正确性证据只能来自限池门禁（长上下文在大量驱逐下干净、探测与对照同水平）。
   开头坏的特征：第一个 token（prefill 算的）对，第二个起乱，之后有的又回到题目上（"Thezek's1. The problem: Uriah needs…"）——
   像 decode 第一步用错了"上一个 token"或最后一页 KV，而不是整段上下文坏了；待查 PD 的元数据 / 辅助 buffer（首 token、MTP 草稿输入）传递。
2. **prefill 空转的结构（`scripts/prefill_pp_hops.py`，E1 四级 trace，102 步）**：三段前向跳中位只有 83 / 43 / 53 ms（下游通常在上游算完前约 100 ms
   就在等）；**环的闭合段（PP3 算完第 i 个 → PP0 开始第 i+5 个）中位 617 ms、均值 800 ms**。按 5 × 节拍 ≈ 4 级 GPU（约 1,000 ms）+ 三段跳（约 180 ms）
   + 闭合开销，实测节拍 345 ms，反推闭合开销约 545 ms；压到约 100 ms 节拍就到约 256 ms（GPU 满）。
   读代码（`scheduler_pp_mixin.py` `event_loop_pp_disagg_prefill`）：每级在第 j 轮要先收到上一级转发的"第 j−4 个 microbatch 的输出"才能发射，
   而上一级只在**下一轮（j+1）**的第 8 步转发（`_pp_send_output_to_next_stage` 发的是上一轮收到的 `pp_outputs`），
   在那之前它要做完自己下一轮的调度（`get_new_batch_prefill`，含驱逐 / 回载）——所以每级的发射都卡在上游下一轮的调度上，环是锁步的。
   E6（异步深度 2，在途 microbatch 5 → 6）正好放松这条链一轮，19:05 左右出结果；若有效，下一步是改转发时机（收到即转发）。
   **NCCL 层面的定位（`scripts/prefill_pp_nccl.py`）**：上游的 proxy 发送提交得很早（比上游 GPU 算完早 120–160 ms），发送 kernel 正好在上游 forward
   结束时开始；**但发送 kernel 本身要跑 PP0→PP1 中位 73 ms（p90 328）、PP1→PP2 19 ms、PP2→PP3 4.5 ms**（4.5 ms 才是 100 MB hidden states 的真实传输）。
   原因（读 `parallel_state.py` / `scheduler_pp_mixin.py`）：isend / irecv 都在同一条 `pp_comm_stream` 上下文里入队，PyTorch 让每对 rank 的 NCCL 内部 stream
   先等 `pp_comm_stream` 的尾部，而尾部有本级上一次发送的 `work.wait()`（`_pp_commit_comm_work`）和等本级 forward 的事件——**本级的接收排在本级的发送后面**，
   发送又要等下一级开始接收，于是越靠前的级等得越久；PP0 每次发射前对"PP3 发来的输出"做的 d2h 同步也排在这条链后面（闭合段长的原因）。
   **E7：接收单独一条 stream**（kit `patches/pp-recv-stream/scheduler_pp_mixin.py`，`config/pp-recv-stream.patch`，`SGLANG_PP_RECV_STREAM=0` 可关；
   收到的张量 `record_stream` 到 forward / copy stream，比原来更安全）。每对 rank 的消息顺序不变，不引入新的等待环。
   队列 `scripts/night10_1028.sh`：night9 之后先跑 E7 完整正确性（`scripts/gate_pass2.py`，按题数比例判长上下文、探测乱码上限 6 以容纳既有 bug），
   通过才跑 C104 性能，再采 E7 四级 trace 核对机制。
   **kernel 线（稀疏 MLA > 2 GB 池，17:45–17:52，140 GPU 0，E6 计分前）**：`scripts/sparse_mla_bucketed.py`、`bench_sparse_mla.py --bucketed`、
   `smla_asm_check.py`，结果 `results/bench_sparse_mla-{bucketed,seg,locality}-1028T17*.txt`。每行 top-k 按 2 GB 段分桶、每桶补齐到 BLOCK_N，
   每个 tile 只属于一段：(a) tile 读段号 + 标量分支选视图——汇编确认是 buffer load（29 条 buffer / 1 条 global、无溢出），但 num_stages 1 / 2 一样
   （分支里的 load 不进软件流水），6.2 GB 打散池上 4.83 ms；(b) 每段一个静态展开的内层循环（无分支）4.80–5.46 ms。生产 5.12 ms，只快约 6%；
   torch 写的分桶本身 1.48 ms，净亏。对照：生产 kernel 在 1.9 GB **完全打散**的池上仍是 3.73 ms → 差距不是局部性，是这两种写法自身的结构开销。
   结论：Triton 内的修法上限约省稀疏 MLA 的 5–15%（prefill GPU 的 2–6%，而 GPU 只忙约 50%、瓶颈在环上，端到端 < 1%）——**暂停**，优先做环（E7）。
   **两个镜像里现成的开关**：(1) `--disaggregation-decode-allocation-policy prefill_complete`（KV 留在 prefill、算完才在 decode 分配，正好能省掉
   decode 上约 22% 的预占）——`arg_groups/pd_disaggregation_hook.py:_validate_prefill_complete` 不支持 `pp_size != 1`，我们的 prefill 是 PP4，**用不了**；
   (2) `--disaggregation-enable-kv-checksum`（prefill 对每个请求的 KV 页算 Adler-32、decode 校验，两侧都要开）——用来诊断"开头坏"是不是 KV 传坏了，
   排在 E7 之后跑一次正确性（不测性能）。
3. decode 侧（`scripts/decode_dp_balance.py`，E1 复测）：4 个 DP rank 的 KV 用量中位都是 0.93、运行中请求 11.4–12.7，均衡；某 rank 排 prealloc 时
   另一 rank 低于 80% 的只占 3% → **换负载均衡方法没有用**。decode KV 里约 22% 是给"还在 prefill"的请求预占的（平均 20 个请求等传输），
   prefill 越快这部分越少——两头的瓶颈都落在 prefill 吞吐上。

**10-28 17:10 更新**
1. **E4（E1 + 关 early-send）= 49,293，比 E1 两次平均 49,719 低 0.9%，不采用**（`results/run-llr37-v3-evict512k-noearly-c104.txt`）。
   prefill 节拍中位 / p90 322 / 837 ms，和 E1（328 / 820、323 / 836）一样；TTFT p50 / p90 5.72 / 14.4 s，比 E1（5.32–5.69 / 12.5–13.9）略差；
   decode 等传输 20.0（E1 19.6–20.2）。**去掉 early-send 的 `.cpu()` 阻塞后节拍没有变**，所以 08:25 第 2 条把 `run_batch` 里 120 ms/步的
   发射延迟归给 early-send 是错的（阻塞去掉后，等待落到了下一个同步点）；关掉它只让前缀晚发，TTFT 略差（用户 09:02 的担心方向对，幅度在噪声内）。
   在它之上的 E5（early-send 延后发）同理不会有收益，撤掉。
2. **E4 门禁（同样限池）**：探测乱码 1、开头坏 3（E1 和对照都是 4 / 6），长上下文干净 → early-send 不是开头坏的唯一原因。
   关于"开头坏"的历史：10-27 不限池的完整 llr35 + v3 1,319 题是 0 条（检测报的 1 条是人名 "Thea" 的误报），当时推断"v3 也修好了"，
   **但那只是推断**；这次限池下 v3 本身就有 6 / 200 条真正的开头坏，推断不成立，问题仍在、在内存紧时出现，待单独查。
3. **10:40 队列跑完后 138 空了约 6.5 小时**：30 分钟心跳和巡检告警都按时输出了（心跳 16 次），但没有让我在两次对话之间恢复工作——
   这个会话里我只能在用户发消息时行动。改为**一次排够长、带判断的无人值守队列**。
4. **17:08 起的队列（`scripts/night9_1028.sh`）**：
   - E1 的生产配置完整正确性（不限池，与 v3 参考同一套：1,319 题探测、16k × 128、100k × 32），约 18:10；
     限池门禁连对照都过不了，不能证明 E1；这一套能和 v3 的参考（0.972 / 乱码 0 / 开头坏 0，64/64 / 63/64，16/16 / 16/16）直接比。
   - E6 = E1 + PP 异步深度 1 → 2（`config/site-llr35-v3-evict512k-depth2.env`）C104 性能，约 19:50；之后跑同一套完整正确性，约 20:55。
     依据：节拍中位约 320 ms、GPU 一步约 250 ms，中位节拍约等于"环上一圈的时间 / 在途 microbatch 数（5）"，多一个在途 microbatch 可能把空转摊薄；
     yihou llr15 测过深度 2 是 −1.5%，但那是在 v3 / E1 之前，驱逐停顿还很大的时候。

**10-28 08:25 更新**
1. **四级 trace（E1，`results/trace-llr37-v3-evict512k-4r-c104/`，87 步对齐）**：每级每步 GPU 忙 248–263 ms、步前空闲 220–246 ms（约 50%）。
   **四级"本级造成的发射延迟"加起来 241 ms/步，约等于每级的空闲**——锁步环里任何一级的延迟最后都变成每一级的空闲。
   延迟构成：`run_batch` 里发射前的阻塞 120 ms/步（PP0 14 / PP1 52 / PP2 25 / PP3 29）、发送被下游堵住 110 ms/步（PP0 41 / PP1 40 / PP2 29）。
   **E1 之后驱逐基本不在关键路径上**（驱逐时本来就在等上游；PP0 单级 trace 的"驱逐占空闲 35%"是空闲归因，不是关键路径），
   所以 E3（批量备份）降级。
2. **`run_batch` 阻塞的来源**：PD prefill 在发射 forward 前对每个请求调 `maybe_send_cached_prefix_chunk`（上游默认开的 early-send，
   提前发 GPU 上已缓存的前缀 KV），其中 `kv_to_page_indices` 的 `.cpu()` 跑在 `schedule_stream` 上，而它已经等过上一个 forward，
   所以这次 `.cpu()` 要等上一个 forward 跑完（每次 89–140 ms，每级 120 步里 49–70 次）——下一个 forward 不能提前排进 GPU 队列。
   **E4 = E1 + 关 early-send（`SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=0`）已排队**（`scripts/night8_1028.sh`：性能 → 门禁）。
   "发送被堵"要按下游当时在做什么重新归因，脚本待改。
3. **E1 门禁（08:02 起）**：smoke 6/6；两遍 16k × 128 长上下文都干净（捞针 63 / 64、长前缀 GSM8K 62 / 63，乱码 0）；
   **但 200 题探测有 4 条乱码**（正确率 0.975 本身达标）：都是第一个词 "The" 之后从第二个 token 起就坏（"Thezek's"、"Theeworld"……），
   一直生成到 16K 上限。第一个 token 来自 prefill，后面是 decode 用传过去的 KV 生成的，所以像 KV 传输 / decode 侧的问题。
   按判定（探测乱码必须为 0）E1 门禁判"不通过"，于是自动跑了同样限池、不带 E1 的对照（`site-llr35-v3-cap300k-gate-1028T0823`）：
   **对照同样 4 条乱码、6 条坏开头（正确率 0.965），长上下文同样干净**；两次出乱码的题号不同（E1：5 / 51 / 52 / 63；对照：6 / 28 / 48 / 61），
   是随机的竞争类问题。**结论：E1 没有引入新的错误，和对照一致**；乱码是 v3 在 30 万 token 限池下本来就有的问题
   （不限池的 v3 参考 1,319 题乱码 0）。这是一个独立的正确性 bug，要单独查：E4 的门禁（关 early-send、同样限池）
   能顺带说明 early-send 是否有关——它在 forward 前读已缓存前缀的页，池子紧时这些页有可能在传完之前被复用。
4. 巡检 05:58 队列跑完时自己退出了，08:22 才发现并换成不退出的版本（`scripts/watch_night7.sh`），中间约 2.5 小时没有巡检。
5. E4 性能（`llr37-v3-evict512k-noearly`）08:49 起栈，约 10:20 出结果，之后自动跑 E4 门禁（约 11:05 出结果）。

**10-28 早上更新（06:55）**
0. **E1 还没有做正确性验证**（07:15 用户问时查实）：之前写"不改语义"只是读代码的判断，性能组的 0 错误只说明请求没失败，不说明输出对。
   门禁已排在四级 trace 之后（`scripts/night7_1028.sh`，约 08:05 起、约 40 分钟）：prefill 设备池限 30 万 token / 级、
   余量 64K（`config/site-llr35-v3-evict-gate.env`：大于单次分配 8K，所以每次驱逐都像生产一样被放大，又小于池子），
   smoke + 200 题探测 + 两遍相同的 16k × 128 长上下文（第二遍从主机回载第一遍被驱逐的前缀），判定 `scripts/gate_pass.py`；
   不过 → 同样限池、不带 E1 的对照门禁。**门禁通过之前 E1 不进提交配置、不提 PR。** HC300 / HC500 只改主机层大小，不单独做门禁。
1. **E1 复现了**：复测 49,631（+2.6%），两次平均 49,719（**+2.8%**），驱逐次数（0.11 次/步）和节拍中位（323 / 328 ms）也和第一次一致。
2. **HC300 + E1 = 49,583（+2.5%），比 250 GB 下的 E1 没有提高**（差 −0.3%，在噪声内）。300 GB 也用满（26.0 / 26.0M token），
   新算 token 占比（新算 / 全部输入）只从 5.25% 降到 5.12%（HC500 降到 4.98–5.02%）。**提交配置保持 250 GB**（DRAM 用得少，结果一样）。
3. **E1 之后 PP0 的 GPU 忙 56%（基线 54%）**，步间空闲的来源（`results/trace-llr37-v3-evict512k-c104/`，PP0，107 步）：
   驱逐仍占 35%（12 次，每次约 610 ms），等环上其他级（PP 收、发、提交发送）占 59%，其余 6%。
   等其他级的那部分要四级对齐才能分，**这次 PP1–3 的 trace 没保存下来**：120 步的 trace 每级要写约 6 分钟，四级是串行写的，
   AgentX 900 s 结束时只有 PP0 写完，拆栈把另外三级删了。四级重采已在跑（AgentX 2400 s，07:00 起，约 07:55 结束）。
4. **一次驱逐为什么要约 525 ms**（PP0，`evict_for_alloc` 12 次共 6,300 ms）：`_evict_components` 每个叶子节点单独"提交 D2H 拷贝 → 同步等完"，
   一次驱逐约 116 个节点串行：第一次等前向结束（写栅栏）约 70 ms，之后 116 次逐个同步约 240 ms，节点的 CPU 处理约 150 ms。
   所以 **E2（去写栅栏）每次只能省约 13%，撤掉不做**；下一步是改成**批量备份**（一次提交所有牺牲节点、只同步一次），
   再往后是**后台提前备份**（空闲页低于水位时异步把 LRU 尾部写回主机，驱逐时只释放已备份的节点，不阻塞调度）。

**今晚结论（10-28 01:45）**
1. **最好的组合：HC500 + E1 = 50,574 tok/s/GPU，比 v3 基线 48,360 高 4.6%**（比 yihou 的 48,408 高 4.5%），C104、0 错误。
   两项单独各约 +3%（HC500 49,909 / E1 49,807），叠加只到 +4.6%：prefill 变快后 decode 开始吃紧（同时在跑的请求 47 → 52、TPOT 15.1 → 15.7 ms），
   prefill 有排队的区间已降到 63%。
2. **E1（驱逐留 512K 余量，`config/evict-headroom.patch`，只改 `unified_radix_cache.py` 的 `evict_for_alloc`，不改语义）+3.0%**：
   prefill 节拍中位 380 → 328 ms。复测（`llr37-v3-evict512k-r2`）在跑，约 03:30 出结果；若复测成立，建议进 Infera（环境变量默认关）。
3. **HC500（主机层每级 500 GB）+3.2%**：靠主机命中多，节拍不变；代价是每组启动 + 拆栈多约 1 小时（pin / 释放 2 TB）。
4. **prefill 慢的根因**：GPU 只忙约 54%；四级 PP 是锁步的环（PP 收发 + HiCache 跨级共识的反压），任何一级的停顿都会沿环传开。
   E1 去掉了最大的一类停顿（write_back 驱逐等前向结束）。剩下的要靠改 PP 事件循环解耦（见下文"跨级关键路径"），不是调参能解决的。
5. **HiCache 容量按 InferenceX 的 DRAM 预算核算（10-28 02:40；先不考虑拆栈释放时间）**：
   - 预算：`mi355x-amds` 上 `dram-utilization: 1.0` = **每个 4 卡服务器 1,499 GB**（InferenceX #4045 原文；#4308 的 0.74 → 1,109 GB 同比例）。
     ATOM PD 的 4 × 349 GiB 正好用满 1,499 GB（每卡约 375 GB）。
   - 我们的实际占用：`--hicache-size` 只算 KV（每 token 11,520 B = 20 层 × 576 B，与 `/metrics` 的主机 token 数一致），
     DSA indexer 在主机上另有一个池，约多 23%（以前的实测："400 GB/rank → 约 492 GB/rank"、"200 GB + 约 46 GB indexer"）。
     → 250 GB/rank 实占约 1.23 TB（预算的 82%）；**500 GB/rank 实占约 2.46 TB，是预算的 1.64 倍，不合规**；
     预算内最大约 **300 GB/rank**（4 × 300 × 1.23 ≈ 1,476 GB）。每卡 394 GB 也超（KV 本身 4 × 394 = 1,576 GB 已超 1,499）。
   - **500 GB 是怎么定的**：10-27 上午按"被驱逐前缀的未命中约占 prefill 新算量 6%"提了 250 → 500，只核对了 138 的 2.75 TB 放得下，
     没有按 InferenceX 的预算算，也没把 indexer 池算进去——这是我的疏漏。HC500 的 +3.2% 只作参考。
   - 预算内的版本：HC300 + E1 已排队（`config/site-llr35-v3-hc300-evict512k.env`）。250 → 300 只多 20% 容量，预期比 HC500 的收益小得多。
6. **待用户决定**：把 v3 + E1 告诉 yihou、Infera PR（v3 + E1）。（E2 和 E1 余量扫描已按分析撤掉，见 10-28 早上更新。）
   精度相关的改动今晚没做。

方向来自 18:43 的 prefill trace（见下文"新 trace"一节）：prefill GPU 只忙 54%，调度线程 35% 的时间阻塞在 write_back 驱逐的同步里。
全部是 llr37 + 补丁 v3、138（prefill TP1×PP4）→ 140（decode TP4/DP4），C104、3600 s 计分，每组整栈重启；对照 = 第 1 步 48,360（yihou 48,408）。
分析统一用 `scripts/analyze_run.sh LABEL`（138 上跑，输出存 `results/run-LABEL-c104.txt`）。

| 组 | 改动 | tok/s/GPU | vs 48,360 | 请求数 | TTFT p50 / 均值 (s) | prefill 节拍 p50 / p90 (ms) | 排队均值 / 有排队的区间 | 驱逐 次/步、平均耗时 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 基线 | v3 | 48,360 | — | 11,624 | 6.23 / 8.97 | 380 / 940 | 6.27 / 82% | 1.16、140 ms | — |
| HC500 | 主机层每级 250 → 500 GB | **49,909** | **+3.2%** | 12,006 | 4.78 / 7.19 | 382 / 917 | 4.04 / 65% | 1.20、115 ms | 有收益，来自缓存 |
| E1 | 驱逐余量 512K token | **49,807** | **+3.0%** | 11,858 | 5.32 / 7.56 | 328 / 820 | 4.92 / 78% | 0.11、308–456 ms | 有收益，来自 prefill 节拍；复测排队中 |
| HC500 + E1 | 两项叠加 | **50,574** | **+4.6%** | 12,100 | 4.67 / 6.70 | 324 / 898 | 3.16 / 63% | 0.12、254–392 ms | 最好；部分叠加（decode 开始吃紧） |
| E1 复测 | 同 E1 | **49,631** | **+2.6%** | 11,852 | 5.69 / 8.04 | 323 / 836 | 4.94 / 78% | 0.11、299–459 ms | 复现（两次平均 +2.8%） |
| HC300 + E1 | 预算内最大主机层 + E1 | **49,583** | **+2.5%** | 11,913 | 5.61 / 7.84 | 332 / 869 | 4.90 / 75% | 0.11、303–467 ms | 比 250 GB 没提高；提交用 250 |
| E1 trace | E1，120 步 / 级 | 只有 PP0（见上） | | | | | | | 四级重采在跑（`$L/prof-llr37-v3-evict512k-4r`） |

（E1 余量 1M 撤掉：估算最多多约 1%，在噪声内，不值一组。）

**E1 / HC500 之后的 GPU 利用率（估算，按每步 GPU 工作约 245 ms 不变）**：

| 组 | 节拍中位 / 均值 (ms) | 有排队时 GPU 利用率（245 / 均值） | 全窗口（新算吞吐 / 约 33.4K tok/s） |
|---|---|---|---|
| 基线 | 381 / 501 | 约 49% | 约 45% |
| HC500 | 382 / 498 | 约 49% | 约 44% |
| E1 | 328 / 466 | 约 53% | 约 47% |
| HC500 + E1 | 326 / 481 | 约 51% | 约 45% |

中位那一步已经约 75% 忙（245 / 328），均值被长节拍拉高（p75 约 580、p90 820–900 ms），缓存前缀越长节拍越长；
全窗口里另有 18–37% 的时间 prefill 没有排队（没活干）。把长节拍压到中位附近，有排队时的利用率能到约 70%——E1 trace 就是为此采的。
E1 trace（PP0）实测 GPU 忙 56%，与估算一致。E2（去写栅栏）按上面的驱逐拆分撤掉。

**HC500（`c104-1027T1903`，19:54–20:54 计分）：+3.2%，收益来自缓存命中，不是 prefill 变快。** 节拍不变（p50 382 ms），
主机命中从 5.20 亿升到 5.76 亿 token，每个请求的新算 token 少约 5.5%（全程新算 8,270 万 vs 8,478 万，但多服务了 3.3% 的请求），
prefill 排队从 6.3 降到 4.0，TTFT p50 6.2 → 4.8 s。500 GB 仍然用满（43.4 / 43.4M token），再大受 138 的 2.75 TB 内存限制，不再加。
代价：启动多约 25 分钟（pin 2 TB 主机内存），**拆栈后 GPU 显存还要约 35 分钟才释放**（KFD 进程清理，`kfd_process_wq` 的 kworker
处于 D 状态，21:00 → 21:32，下一组因此晚了 32 分钟），每组合计多约 1 小时。它和 E1/E2（调度阻塞）是独立的因素，
最后一组把最好的一项和 HC500 叠加（已备 `config/site-llr35-v3-hc500-{evict512k,nofence,nofence-evict512k}.env`）。

**E1（`c104-1027T2133`，21:56–22:56 计分）：49,807，+3.0%，收益来自 prefill 节拍。** 驱逐从每步 1.16 次降到 0.11 次
（每次约 53 万 token、平均 308–456 ms），PP0 驱逐总耗时 1,686 → 525 s；prefill 满块节拍中位 **380 → 328 ms（−14%）**、p90 940 → 820，
现算 15,076 → 15,624 tok/s（+3.6%），排队 6.3 → 4.9，TTFT p50 6.2 → 5.3 s，请求数 +2.0%。缓存命中不变（主机命中占比同基线）。
（22:25 我按前 16–28 分钟的累计吞吐写过"生效但不提速"，那是早期数据不够，以全窗口为准。）
剩余：每次大驱逐仍同步等 6.5 GB 的 D2H 和一次前向结束，PP0 平均约 50 ms / 步——E2（去写栅栏）可以再去掉其中"等前向"的部分。

**跨级关键路径（`scripts/prefill_pp_critical.py`，18:43 那份四级 trace 按步对齐，结果 `results/trace-llr37-v3-c104/pp-critical.txt`）**：
对每级每步算"本级 CPU 造成的发射延迟"= 发射时刻 − max（本级 GPU 空出、上游数据到达），并把这段时间按调度线程在做什么分类。
- 四级每步 GPU 忙 236–247 ms、步前空闲 188–222 ms，但**本级造成的延迟平均只有 PP0 27 / PP1 61 / PP2 22 / PP3 5 ms**，
  分类里没有驱逐同步（只有"等下游收走发送"和 PP1 的 `run_batch`）；空闲的大头是**等上游数据**（`pp recv` 60–120 ms/步）、
  驱逐同步（约 40 ms/步）和 HiCache 跨级同步（约 28 ms/步）。**注意这个指标在锁步环里会低估源头**：一级的停顿在其他级都表现为
  "在等数据"，E1 的 +3.0%（节拍 −14%）说明驱逐同步确实在环的关键路径上，只是没法按单级归因。
- **四级是锁步的环**：停顿像波一样**反向**传播（PP3 第 2 步停 766 ms → PP2 第 3 步 764 → PP1 第 4 步 767 → PP0 第 5 步 770）。
  机制（镜像代码）：每轮开头 `_pp_commit_comm_work` 等上一轮发给下一级的 gloo 消息被收走；HiCache 的跨级共识 `_pp_sync`
  （`unified_radix_cache.py`，驱逐 / 写回 / 回载的 `_all_reduce`）让 PP n 阻塞接收 PP0 传下来的数据，`_drain_async_work`
  每轮又等这些 isend 被下一级收走（注释 "applies backpressure when a downstream PP rank lags"）。数据路径上，PP n 在本轮末尾才发出
  第 k 批的激活，末级在**下一轮的准备做完之后**才把第 k 批的输出发回 PP0；`pp_loop_size = pp_size + async_depth = 5` 个槽位，环里几乎没有余量。
- → prefill 的节拍 ≈ GPU 一步 + 环上各段控制面 / 通信延迟的累加，而不是某一级的 CPU 慢。要提速得**解耦这个环**（末级算完即发输出、
  HiCache 跨级同步留一轮余量、给环更多槽位），这是改 PP 事件循环的结构性工作，今晚不做；depth 2（多一个槽位）yihou 测过 −1.5%（llr15）。

**E2 暂缓（22:26 取消 night3，当时依据的是 E1 的中途数据，判断错了）**：E1 全窗口 +3.0% 之后，E2 预期还有小幅收益
（去掉剩下约 50 ms / 步驱逐里"等前向"的部分），但要改流间顺序、要先过门禁（约 40 分钟）。今晚先跑 HC500 + E1 叠加和 E1 复测，
E2 留给用户决定（门禁脚本 `scripts/night3_1027.sh`、配置都在，基线可换成 E1：`config/site-llr35-v3-hc500-nofence-evict512k.env`）。
另：23:01 E1 结束后队列空了 54 分钟——巡检只显示各队列日志拼接后的最后一行，night2 的结束行被 night3 日志里提前写的取消行盖住，我等的匹配没触发
（之后换成 `scripts/watch_night5.sh`，跟最新写入的队列日志）。

**HC500 + E1（`c104-1027T2355`，10-28 00:24–01:24 计分）：50,574，+4.6%。** prefill 节拍中位 324 ms（同 E1），新算 14,991 tok/s
（比 E1 少，因为主机命中更多：5.99 亿 vs 5.37 亿 token），排队 3.2、有排队的区间 63%（基线 82%）；decode 同时在跑 52（基线 47）、
TPOT 15.7 ms（基线 15.1）。p90 节拍 898 ms 比 E1 的 820 高——主机层大了回载更多（每步 5.9 万 vs 5.1 万 token）。

复现（138 上，kit = `2p1d-kit/llr35`，每行一组，整栈重启，结果在 `$L/perf/LABEL/`；分析 `bash $P/scripts/analyze_run.sh LABEL`）：

```bash
cd $KIT && bash $P/../prefill-car-20261023/scripts/perf_plan.sh $KIT $L/perf PLAN   # PLAN 每行 "LABEL SITE_ENV 104"
# 基线 v3                $P/config/site-llr35-v3.env
# HC500                  $P/config/site-llr35-v3-hc500.env
# E1                     $P/config/site-llr35-v3-evict512k.env          （EVICT_HEADROOM=token 数可改余量）
# HC500 + E1             $P/config/site-llr35-v3-hc500-evict512k.env
# E2 门禁 / 性能          bash $P/scripts/night3_1027.sh（门禁配置 site-llr35-v3-nofence-gate.env，判定 scripts/gate_pass.py）
```

## llr35 在 PD 下长上下文严重乱码（10-27 04:10，`correct-1027T0333`，真实接受率，138 → 140）

| 检查 | llr35 | 以前验过的配置（async 4+8 / custom all-reduce 4+4，K5、无 KF；长上下文只有后者测过） |
|---|---|---|
| smoke | 5/6：`long_prefix_cold` 乱码（重复片段、`</think>` 当正文），同一前缀的 `long_prefix_repeat` 正确 | 6/6 |
| lm-eval GSM8K strict / flexible | 0.9697 / 0.9712 | 0.966–0.968 |
| 200 题探测（GSM8K，并发 64） | 0.945，**乱码 10**、跑到上限 10（生成几千 token 后退化成 "0 . 0011 . …" / "- - -"） | 0.980–0.985，乱码 0 |
| 长上下文 16k × 128（无共享前缀，prompt 中位 1.39 万 token） | 捞针 **13/64**（乱码 44、上限 49）；长前缀 GSM8K **1/64**（乱码 21、上限 64） | 捞针 64/64，长前缀 GSM8K 62–63/64，乱码 0 |
| 长上下文 100k × 32（prompt 中位 8.6 万 token） | 捞针 **2/16**（乱码 14）；长前缀 GSM8K **1/16**（乱码 8、上限 16） | — |

lm-eval 只抽最后的数字，看不出来；模拟接受率计分不看内容，也看不出来。短 prompt 的 GSM8K 在生成变长、decode 上下文超过 2048（DSA 进入稀疏 top-k）
之后开始坏，长上下文从一开始就坏，指向 decode 的长上下文路径；KF 的稀疏 MLA / indexer / top-k kernel 正是在这里接管（overlay 注释：只在
K5 的 TARGET_VERIFY 行 T = 6 × bs 上验证过，llr35 是 K3）。

**二分**（`scripts/quick_correct.sh`，每个变体只从 llr35 去掉一项、都关 HiCache；smoke + 200 题探测 + 长上下文 16k × 32）：ref（只关 HiCache，
须仍乱码）→ no-kf → no-car → no-fp8draft → k5 → no-ppcomm → no-idxshare（`config/site-bis-*.env`）。用户 04:25 决定二分提前：
4+4 C96 基线（还在等显存、未起栈）已停，二分 04:25 起，138 `$L/bisect/`（`waiter-*.out`、每个变体的 `.log` / `.probe.jsonl` / `.longctx-*.jsonl`）；
基线之后再跑（`config/plan-llr35-base.txt`）。

| 变体（llr35 去掉一项，都关 HiCache） | smoke 长前缀现算 | 200 题探测 正确 / 乱码 / 上限 | 长上下文 16k × 32：捞针 / 长前缀 GSM8K |
|---|---|---|---|
| ref（只关 HiCache） | 乱码 | 0.945 / 7 / 10 | 2/16 / 1/16 |
| no-kf（去掉 KF kernel 与 KF top-k） | 乱码 | 0.945 / 7 / 10 | 2/16 / 1/16 |
| no-car（decode 关 custom all-reduce） | 乱码 | 0.960 / 1 / 6 | 3/16 / 0/16 |

| no-fp8draft（draft MoE 不转 FP8） | 乱码 | 0.950 / 4 / 9 | 2/16 / 0/16 |
| k5（两侧 EAGLE 5/1/6） | 乱码；smoke 3/6，短 chat 也是垃圾（"Paris!!!!!riba!!!!!"） | 0.730 / 17 / 49 | 3/16 / 2/16 |
| no-idxshare（两侧 index_share false） | 乱码 | 0.945 / 3 / 7 | 3/16 / 0/16 |
| **no-ppcomm（prefill 关 `SGLANG_PP_COMM_OVERLAP`）** | **正确，smoke 6/6** | **0.975 / 0 / 3** | 2/16（乱码 2、上限 11）/ 3/16（乱码 1、上限 13） |

- **主因是 prefill 的 `SGLANG_PP_COMM_OVERLAP=1`**（yihou llr21，+1.7%）：关掉后 smoke 和短 prompt 探测恢复到干净配置的水平（0.975、0 乱码）。
  与"现算错、缓存对"一致：PP 级间通信与计算重叠，KV（含 indexer key）在算完之前被发给 decode；上下文超过 2048、DSA 开始稀疏选 token 时坏位置被选中。
- **长上下文还有第二个问题**：关掉 PP 通信重叠后乱码少了，但跑到上限的输出仍退化成 "1.1.1.11.1…" / "44489.\" 44489.\" …"，正确率很低。
- 第一轮里 KF、FP8 draft "去掉后无变化"是在 PP 通信重叠还开着时测的，被主因掩盖，对第二个问题不算排除；custom all-reduce 关掉后短 prompt 乱码 7 → 1。
- **第二轮**（`waiter3-*.out`，基础 = llr35 去掉 PP 通信重叠）：clean-k5（kit 默认配方，= 以前验过的 4+4，验证镜像与 kit 干净）、clean-k3（只换 K3）、
  再分别去掉 custom all-reduce / KF / GEMM 调优 CSV / index_share / async（`config/site-bis2-*.env`）。07:32 追加 `waiter4-*.out`：
  去掉 FP8 draft、去掉 decode 的 mem 0.95 / stream-interval 30 / CLIP 8（回到 kit 的 0.92）。

| 变体（第二轮） | smoke | 200 题探测 正确 / 乱码 / 上限 | 长上下文 16k × 32：捞针 / 长前缀 GSM8K |
|---|---|---|---|
| clean-k5（kit 默认，无 yihou 任何选项） | 6/6 | 0.990 / 0 | 16/16 / 16/16 |
| clean-k3（只换 K3） | 6/6 | 0.985 / 0 | 16/16 / 15/16 |
| no-car | 6/6 | 0.970 / 1 / 5 | 3/16 / 2/16 |
| no-kf | 6/6 | 0.935 / 6 / 12 | 4/16 / 1/16 |
| no-gemmcsv | 6/6 | 0.985 / 0 / 3 | 2/16 / 2/16 |
| no-idxshare | 6/6 | 0.935 / 6 / 7 | 6/16 / 2/16 |
| **no-async（prefill 去掉 `--pp-async-batch-depth 1`）** | 6/6 | 0.975 / 3 / 3 | **16/16 / 15/16** |
| no-fp8draft | 6/6 | 0.980 / 0 / 3 | 5/16 / 2/16 |
| no-decodeknobs（mem 0.92，去掉 stream-interval 30、CLIP 8） | 6/6 | 0.950 / 5 / 9 | 1/16 / 3/16 |

  - **第二个问题是 prefill 的 async（`--pp-async-batch-depth 1`，yihou llr35，+2.5%）**：只有去掉它长上下文才恢复到 clean-k3 的水平；
    其余 decode 侧选项单独去掉都无效。
  - 更正：10-24 的 async 正确性（`correct-1024T0212`，4+8 K5）只跑了 smoke、GSM8K、200 题探测，**没有跑长上下文**；
    上表"以前验过的干净配置"一栏的长上下文 64/64 来自 4+4 custom all-reduce K5（10-23 `prof-c88` 起栈后）。async 从未在长上下文上验过。
  - 确认（`waiter5-*.out`，`config/site-bis3-*.env`）：**clean-k3 + async（只加 async）长上下文就坏**：捞针 2/16（上限 12）、长前缀 GSM8K 2/16；
    短 prompt 正常（0.990，乱码 0）。**clean-k5 + async 同样坏**（捞针 2/16、长前缀 GSM8K 0/16）→ async 本身的问题，与 K 无关。
  - async 坏掉的长上下文输出：开头正常，之后换行丢失、数字粘连（"…Taylor2. Extract the access code3. …"），捞出的 code 错；
    像 decode 收到的 prompt KV 有一部分不对（PP4 每级只传自己的层），而不是整段错。smoke 的冷长前缀（几千 token）在 async 下是对的。
  - **并发 1 也坏**（clean-k5 + async，16k × 8，`LONGCTX_CONC=1`）：0/8，7 条到上限 → 每个长请求确定性复现，不是请求间竞争。
    smoke 的冷前缀（约 6.6 万 token、切块）在 async 下是对的，但它的针在重复填充里、答案短，比探测容易，不能说明切块就没事。
  - KV 校验（`--disaggregation-enable-kv-checksum`，Adler-32，prefill 算、decode 比）在 PP4 → TP4 下不能直接用：每个 prefill stage 只算自己的层，
    decode 算全部层，mooncake 也没有按 PP 跳过，会全部误报中止。要用需打补丁改成按层校验（定位 async 机制的候选手段）。
  - 长度扫描排在最后：clean-k5 + async，2k（低于 2048，不走稀疏 top-k）/ 4k / 8k，各 16 条、并发 8。
  - **候选机制：PP 级间收到的 proxy tensor 生命周期**（上游 [sglang PR #15545](https://github.com/sgl-project/sglang/pull/15545)，
    "incorrect output in pp mode when pp_async_batch_depth > 1"，未合并就关闭；上游 main 与 v0522 镜像都没有修）。收到的 hidden states
    由分配器记在接收的 stream 上，forward 在 forward_stream 上读；Python 释放后，下一次接收可以在 forward 还没读完时复用这块显存。
    async 让 CPU 领先 GPU 更多，PP 通信重叠把接收放到单独的 stream，两个主因都可能是这一类。长 prompt 的 proxy 大（1.4 万 × 6144 × bf16
    约 170 MB）、forward 久，最容易中招。并发 1 也坏这一点它解释不了，所以要实测。
  - 补丁（`config/pp-proxy-lifetime.patch`，kit `patches/pp-proxy-lifetime/`，用新加的 `PREFILL_MOUNT_EXTRA` 挂进 prefill，
    kit md5 见 `config/kit-llr35-1027T0940.md5`）：`_pp_recv_proxy_tensors` 收到后对每个 tensor `record_stream(forward_stream)`，只改这一处。
  - 重新排队（`scripts/reorder_1027.sh`，`waiter8-*.out`）：b3 跑完后停掉 waiter6 的剩余部分和 waiter7，依次跑 clean-k3 + PP 通信重叠、
    同 + 补丁、clean-k5 + async + 补丁（`config/site-bis5-*.env`，200 题、16k × 32）；再跑 1319 题的 clean-k3 + custom all-reduce、clean-k3；最后长度扫描。
    第一项因 `up.sh` 要求 `PREFILL_MOUNT_EXTRA` 已定义而起栈失败（`common.sh` 已补默认空值），10:00 用 `scripts/reorder2_1027.sh`（`waiter9-*.out`）按同样顺序重排。

**加回与补丁测试（200 题探测、16k × 32）**：

| 变体 | smoke 冷前缀 | 200 题探测 正确 / 乱码 | 捞针 / 长前缀 GSM8K |
|---|---|---|---|
| clean-k3 + PP 通信重叠（只加这一项） | 失败（5/6） | 0.970 / 0 | **2/16 / 1/16** |
| 同上 + 补丁 v1（接收端 `record_stream(forward_stream)`） | 失败（5/6） | 0.960 / 1 | **3/16 / 0/16** |
| 同上 + 补丁 v2（v1 + 发送端 `record_stream(通信 stream)`） | 失败（5/6） | 0.965 / 0 | **1/16 / 0/16** |
| clean-k5 + async + 补丁 v1 | 6/6 | 0.970 / 0 | **3/16 / 1/16** |
| clean-k5 + async + 补丁 v2 | 6/6 | 0.980 / 1 | **5/16 / 2/16** |

- **两版补丁对两个主因都无效**：显存分配器在 stream 之间复用（proxy tensor 生命周期，接收端和发送端）这一类假设被排除，至少不是主要原因。

**第三个现象复测（1319 题）**：clean-k3 + custom all-reduce 开头坏 0（0.967，长上下文 16/16、15/16）；clean-k3 0（0.970，16/16、16/16）；b3 0。
custom all-reduce 单独加上没问题；200 题里最多的一次（5 条）是 b3 + custom all-reduce（bis2-no-async）→ 排了 1319 题复测（`site-bis7-b3-car.env`）。

**新机制假设：PP 下中间块的 KV 在算完之前被 RDMA 读走**（读代码，11:30）。
- llr35 的 PP4 切块是 8192（`PP4_CHUNK=8192`，所有二分变体都继承），所以 1.39 万 token 的长上下文 prompt 是两块；smoke 和 200 题探测都不切块。
- `process_prefill_chunk`（`disaggregation/prefill.py`）在 PP prefill 每轮开头，对正在切块的请求、非 overlap 模式（prefill 是 `--disable-overlap-schedule`）
  立刻 `send_kv_chunk` 发出刚排进去的那一块；mooncake 传输线程直接读显存，不跟 CUDA stream 同步。
- overlap 路径为此专门加了等待：`process_batch_result_disagg_prefill` 里 Infera 的 `_infera_nonfinal_chunk_wait_event`（注释 "the transfer worker
  reads device memory outside the CUDA stream. Gate the read on this chunk's writes completing"）、`maybe_send_cached_prefix_chunk` 里的 event，
  都只在 `enable_overlap` 时设置。PP 也在 `forward_stream` 上 launch，却两处都没覆盖。
- 默认配置能对，靠的是 CPU 每轮被收发阻塞、领先 GPU 不多；**async 和 PP 通信重叠都去掉了这个阻塞**，CPU 在第一块 forward 写完之前就发出了它。
  与所有现象一致：只有超过切块大小的 prompt 坏、短 prompt 好、并发 1 也稳定复现、输出是"上下文部分看不清"（只有第一块坏）。
- 补丁 v3（`config/pp-chunk-wait.patch`，kit `patches/pp-chunk-wait/prefill.py`，只改这个文件、不带 v1/v2）：PP 下发中间块前在 `forward_stream` 上记 event
  挂给 `disagg_kv_sender._early_send_wait_event`；提前发送缓存前缀的条件也加上 PP。
- 队列（`waiter12-*.out`，长度扫描之后）：async + 切块 16384（16k 一块 / 30k 两块，同一部署内对照）、async + v3、PP 通信重叠 + v3、
  PP 通信重叠 + 切块 16384、b3 + custom all-reduce 1319 题。

| 变体（async） | 200 题探测 | ~1.8k | ~3.5k | ~7k | ~1.39 万 | ~2.6 万 |
|---|---|---|---|---|---|---|
| clean-k5 + async，切块 8192（长度扫描并发 8） | 0.938（64 题） | 8/8、8/8 | 6/8、8/8 | 6/8、7/8 | 2/16、0/16（之前） | — |
| clean-k5 + async，切块 16384 | 0.965 / 乱码 1 | — | — | — | 9/16、4/16 | **0/8**、5/8 |
| **clean-k5 + async + 补丁 v3（切块 8192）** | **0.980 / 乱码 0** | — | — | — | **16/16、15/16** | **8/8、8/8** |

（每格：捞针、长前缀 GSM8K。）单块长度下的错误也是同样的退化，因为并发时调度器把多个 prompt 塞进一个 8192 的批次，装不下的那个会被切开；
切块 16384 时 1.39 万的请求多数一块算完、明显变好，2.6 万的一定切块、依然坏。

- **async 的根因确认：PP 下中间块的 KV 在本 stage 算完之前被 mooncake 传输线程 RDMA 读走**；补丁 v3 修好（16k 与 30k 全对，长前缀 GSM8K
  错的 1 条无乱码、无到上限，与 clean-k3 相同水平）。
- **PP 通信重叠也由补丁 v3 修好**（clean-k3 + PP 通信重叠 + v3）：smoke 6/6（冷前缀通过，不打补丁时失败）；200 题 0.980、乱码 0；
  1.39 万 **16/16、15/16**；2.6 万 **8/8、8/8**。→ 两个主因是同一个 bug：PP 下中间块 KV 的传输时序只靠"CPU 不会领先 GPU 太多"，
  async 与 PP 通信重叠都打破了这个前提。
- **收尾结果（12:55）**：

| 配置 | smoke | 1319 题探测 正确 / 乱码 / 开头坏 | 16k × 128 捞针 / 长前缀 GSM8K | 100k × 32（p50 8.6 万 token） |
|---|---|---|---|---|
| **完整 llr35 + v3**（开 HiCache、PP 通信重叠、async、custom all-reduce、全部 decode 选项） | 6/6 | **0.972 / 0 / 0**（检测报 1 条是误报：题里的人名 "Thea"） | **64/64 / 63/64** | **16/16 / 16/16** |
| b3 + custom all-reduce（不打补丁） | 6/6 | 0.970 / 1 / 0 | 16/16 / 16/16（16k × 32） | — |
| 对照：llr35 不打补丁（`correct-1027T0333`） | 5/6 | 0.945（200 题）/ 10 | 13/64 / 1/64 | 2/16 / 1/16 |

  → **完整 llr35 + v3 正确**，与干净配置同一水平（clean-k3：0.970 / 0 / 0）。
- 第三个现象（开头坏）在 1319 题上不复现：b3 + custom all-reduce 0 条（200 题时同一配置 bis2-no-async 有 5 条）、完整配置 + v3 0 条。
  推断（未单独证明）：它也是同一个竞争，默认时序下窗口很小、只偶发——短 prompt 并发时也会被批次切块；custom all-reduce 只在 decode，不影响 prefill 时序，
  "只在开 custom all-reduce 的变体里出现"是 200 题样本下的巧合。
- 收尾队列（`waiter13-*.out`，`config/site-llr35-v3.env`）：完整 llr35（开 HiCache、PP 通信重叠、async、custom all-reduce 与全部 decode 选项）+ v3，
  1319 题探测 + 长上下文 16k × 128、100k × 32；同时看第三个现象在完整配置里是否还在（b3 + custom all-reduce 的 1319 题在它之前）。

- **PP 通信重叠单独加上就坏**（与 async 一样，两项都已从干净配置加回确认）。
- **补丁 v1 无效**（已确认挂进 prefill 容器）→ PP 通信重叠的问题不在接收端。看代码改为怀疑**发送端**：stage 输出的 hidden states 在 forward stream
  上分配，交给 NCCL 异步发送；`_pp_commit_comm_work` 里 `work.wait()` 之后 PyTorch 释放暂存的 tensor。默认路径 `wait()` 在 schedule stream 上，
  下一次 forward 先等 schedule stream，所以安全；开 PP 通信重叠后 `wait()` 在通信 stream 上，forward 不等它，只有 CUDA graph 时才加屏障
  （注释 "Eager prefill remains overlapped"）→ 下一次 eager prefill 可能在 NCCL 发完之前复用并覆盖这块显存，下一级收到坏的 hidden states。
  `send_tensor_dict` 发的是原 tensor 的 view，所以对原 tensor `record_stream` 有效。
- 补丁 v2（`config/pp-proxy-lifetime2.patch`）：发送前对要发的 tensor `record_stream(通信 stream)`，并保留 v1。不加屏障，保留重叠收益。
  `scripts/reorder3_1027.sh`（`waiter10-*.out`）：K5 + async + v1 跑完后先跑 clean-k3 + PP 通信重叠 + v2；若 v1 没修好 async，再跑 K5 + async + v2。

**b3 = llr35 去掉 PP 通信重叠、async、custom all-reduce（保留 KF、FP8 draft、index_share、GEMM CSV、mem 0.95、stream-interval 30、CLIP 8；关 HiCache）**：
smoke 6/6；**1319 题探测 0.967，乱码 0，开头坏 0**；长上下文 16k × 32 **捞针 16/16、长前缀 GSM8K 16/16**。19 条到上限（1.4%），
全是题意含糊时的连贯反复（"2/5 times more"、"twice as many more"），不是乱码；clean-k3 的 1319 题给对照。
  - 下一步（`waiter6-*.out`）：clean-k5 + async 长上下文并发 1（区分 microbatch 间交错 / 每请求确定性）；然后全量 GSM8K 1319 题探测
    （`PROBE_N=1319`，看低频的开头坏）跑 b3（llr35 去掉 PP 通信重叠、async、custom all-reduce）、clean-k3 + custom all-reduce、clean-k3、
    clean-k3 + PP 通信重叠（`config/site-bis4-*.env`）。

**第三个现象：开头坏（`scripts/probe_badstart.py`）**。去掉两个主因后，200 题探测里仍有 reasoning 在第一个 token 之后就坏的样本
（"Therenzcaff1:"、"The9每小时…"、多次运行里重复出现同一串 "Theedb7…"），最终答案常仍对，探测的 `garbled` 标记漏掉了它们。

| 是否开 decode custom all-reduce | 变体（200 题探测里开头坏的条数） |
|---|---|
| 开 | ref 6、no-kf 6、no-fp8draft 4、k5 4、no-ppcomm 4、no-idxshare 2、bis2-no-kf 2、bis2-no-async 5、bis2-no-decodeknobs 2；bis2-no-gemmcsv / no-idxshare / no-fp8draft 0 |
| 关 | no-car 0、bis2-no-car 0、clean-k5 0、clean-k3 0、clean-k3 + async 0、clean-k5 + async 0 |

  1–3% 的低频事件，200 题分辨不了；指向 decode custom all-reduce 与 yihou decode 栈的组合（10-23 4+4 K5 无 KF 开 custom all-reduce 的 200 题为 0）。
  全量 1319 题加回测试确认。若剩下的单项都不能单独修好，改从 clean-k3 逐项加回。

## 10-27 晚：llr37 + v3 复测、prefill trace、kernel 分析

**复测第一次被外部清理打断**：16:20:28 decode 的 `scheduler_1` 收到 SIGTERM（guard 停了 AgentX），16:22:46–16:24:57 140 上**所有**容器
（我们的 decode、kimi-serve、anandaku-v41-v7-cpu、anandaku-blogb-monofly、我们 10-20 留下的已退出容器）每约 15 s 一个被 kill 9 / destroy
（`docker events`）。节点层面的清理（140 跑着 fleet 的 k3s agent），不是配置或补丁的问题；prefill 侧 16:20:43 起的 KV 传输失败是连带。
16:35 重排：`scripts/night_1027.sh`（138 `$L/night-*.out`）= 复测 → prefill trace（C104，预热后 420 s，每级 30 步，带输入形状）→ HiCache 500 GB
（`config/site-llr35-v3-hc500.env`）。decode 日志另在 140 `/mnt/m2m_nobackup/liyingli_3p1d/decode-follow/` 实时保存；`scripts/watch_night.sh` 每 5 分钟巡查。

**第 1 步结果：llr37 + v3 在 138 / 140 上 C104 = 48,360 tok/s/GPU，与 yihou 在 135 / 136 的两次均值 48,408（48,713 / 48,103）持平（−0.1%）**
→ 补丁 v3 修好长上下文正确性、不损失性能，换节点也复现。运行 `c104-1027T1635`（138 `/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/`），
16:59–17:59 计分，11,624 个请求 0 错误（AIPerf 丢弃 5）。

| C104 | tok/s/GPU | 输出 tok/s/GPU | 每秒请求 | 完成 / 丢弃 | intvty p50 / p90 | ITL p50 / p90 (ms) | TTFT p50 / p90 / 均值 (s) | E2E p90 (s) | API 命中（理论） |
|---|---|---|---|---|---|---|---|---|---|
| llr37 + v3（138 → 140） | **48,360** | 401.2 | 3.206 | 11,624 / 5 | 66.2 / 57.4 | 15.1 / 17.4 | 6.23 / 14.74 / 8.97 | 45.3 | 96.10%（96.64%） |

**瓶颈在 prefill**（`analyze_pd_logs.py`、`pp_stage_times.py`，计分窗口）：prefill 82% 的 10 s 区间有排队（平均 6.3 个在等、`#queue-req` 常在 15 左右），
现算 15,030 tok/s；decode 最忙 rank 中位只有 16 个请求（每步约 42 ms），另有平均 21.6 个请求在等 KV 传输。TTFT p50 6.2 s 占 E2E 的很大一部分。
**prefill 满块的节拍很散**：PP0 背靠背满块（≥ 8000 新 token、有排队）5,693 个，p10 257 / p25 286 / **p50 380** / p75 586 / p90 939 / p99 1,633 ms，
**均值 499 ms**；四级的节拍分布一样（停顿会沿流水线传开）。按 batch 的缓存 token 数分桶（0–50k 到 > 400k），**p10 都约 255 ms、p90 都在 800–1,100 ms**，
即离散**不是上下文长度造成的**；p10 ≈ 旧 trace 每步计算中位数（约 243 ms）。> 600 ms 的节拍占 24%，比中位数多出的部分约占总时间 28%。
如果节拍能稳定在 p25 左右，prefill 吞吐约可提高 40%+——这比任何单个 kernel 的收益都大，所以 trace 的第一任务是解释节拍之间的空隙（GPU 在等什么）。

**yihou 的 GEMM 调优只覆盖 decode**：两份 CSV 来自 decode 的未调优清单（`decode_bf16_untuned_gemm.csv`），只在 `DECODE_MOUNT_EXTRA` 里挂；
prefill 一行都没用上。（更正：aiter 查表不只按精确 M——先按 M 查，再查补齐到 128 的 M，再查 M 的下一个 2 的幂，所以 M = 8192 的一行覆盖
4097–8192 的所有 prefill batch。）

**旧 trace 按模型代码重新归类**（`scripts/prefill_trace_map.py`：kernel → launch → Python 栈与算子输入形状；去掉 NCCL 与旁路 stream）。
数据是 10-23 的 4+8 C88（K5、无 async、无 KF），只作方向，新 trace 出来后替换。旧的 `prof_breakdown.py` 把 MoE 第二段 `gemm2_a4w4_*` 算进 gemm、
分母含 NCCL 空等（PP0 17%、PP1 27%），所以"MoE 只占 5%"是低估。

| 模块（主 stream 计算时间占比） | PP0（365 ms/步） | PP1（308 ms/步） | 主要 kernel / 代码位置 |
|---|---|---|---|
| 稀疏 MLA（triton） | 28.5% | 33.0% | `_sparse_mla_fwd_split_dim_kernel`，`triton_sparse_mla.py:458` |
| DSA indexer | 22.8% | 19.7% | `topk_packed_kernel`、`_gluon_fp8_mqa_logits_kernel`、`topk_transform`、`kv_to_page_indices` |
| 线性层 | 16.6% | 22.4% | o_proj 6144×16384（hipBLASLt，未调优）、q_b 16384×2048（triton）、q/v 吸收 bmm（a8w8）、q_a+kv_a 2624×6144 |
| MoE（MXFP4） | 11.4% | **15.7%** | `mfma_moe1_*` + `gemm2_a4w4_*`（aiter `fused_moe_`，8192 token × 257 专家含共享专家） |
| 调度 / 输入准备 | 9.5% | 7.2% | `resolve_forward_inputs` 的 HtoD、DtoH |
| rope + 写 KV cache | 5.8% | 1.1% | `_fused_qk_rope_cat_and_cache_mla_kernel` |
| norm | 5.5% | 0.9% | `add_rmsnorm_quant_kernel` |

发现（待新 trace 确认）：
1. **indexer top-k 的单次耗时 PP1 是 PP0 的 6 倍**（1.77 对 0.30 ms，同一批请求、同样的上下文长度）。GLM-5.2 只在 78 层里的 21 层算 top-k
   （`indexer_types`：0、1、2 层，之后每 4 层一次），每层按请求逐个调用。prefill 的 `topk_packed_kernel` 每个 query 行一个 block，上下文 > 16,384
   走单 block 的 Streaming（多轮扫整行）；decode 版有给长行的常驻 cluster 路径，prefill 版没有。耗时随层变，指向依赖分数分布的 radix 轮数。
   另外 logits kernel 先写出完整分数矩阵（每请求每层约 1 GB）再由 top-k 读回 → "logits + top-k 融合"是候选。
2. **PP0 上的小 kernel 普遍偏慢**：rope + 写 KV cache 每次 1.04 ms 对 PP1 的 0.17 ms，`add_rmsnorm_quant` 调用次数相同但慢 8 倍。PP0 旁路 stream
   有约 3.2 s 的 HiCache 拷贝 / 传输 kernel，像是与访存密集的小 kernel 抢带宽。
3. **MoE 效率偏低**：每层每次约 2.4 ms，约 5.6 TFLOP → 约 2.3 PFLOPS，约 MXFP4 峰值的 1/4。KernelForge 的 FlyDSL MXFP4 MoE 在 8192 token 上比 CK
   快 1.24 倍（Kimi-K2 形状），正好是 prefill 规模。
4. **o_proj（BF16，6144×16384，M ≈ 8k）**每次约 1.9 ms，约 0.86 PFLOPS，BF16 峰值的约 1/3，走的是未调优的 hipBLASLt / torch 路径。
5. **稀疏 MLA**每层每次约 5.3 ms，约 0.44 PFLOPS；是按 token 取 2048 个 KV 的 gather，相邻 query token 的 top-k 集合高度重叠，现在没有复用。

### 旧 trace 复查（17:00–17:45）：按中位数看稳态，停顿另算——推翻上面 5 条发现中的 3 条

模型结构、每层算子表与优化候选排序单独写在 [`COMPUTE-FLOW.zh-CN.md`](COMPUTE-FLOW.zh-CN.md)。

上表是 30 步的总和，混进了少数"停顿步"。`prefill_trace_map.py` 新增每步主 stream 计算时间、每个 kernel 的中位耗时、kernel 调用点
（算子 + 输入形状 + Python 帧）、旁路 stream 调用点、长 kernel 期间 GPU 上重叠的其他操作。

**停顿步**：PP0 每步主 stream 计算中位 243 ms、均值 330 ms；30 步里第 1、16、17、29 步各 0.8–1.3 s。PP1 中位 236、均值 286 ms，停在 1、16、17 步。
停顿里是平时几微秒的 kernel（6 个元素的 `fill_`、765 个元素的 `cumsum`、539 个元素的 `copy_`、rope、rmsnorm）一次 0.4–0.6 s，同 stream 的前后操作
时间重叠——是 GPU 侧等事件的时间记到了下一个 kernel 上，不是 kernel 慢。按绝对时间，同一停顿依次出现在 PP0 → PP1 → PP2 → PP3（间隔约 0.4 s），
即**跟着某个 batch 走**，不是节点级事件。期间 GPU 上同时在跑：stream 5 上的 HiCache 索引搬运（`move_indices` H2D，[250944] 一次 160 ms，[8192] 一次
3.6 ms）、`_prefix_kv_indices` 拷贝、NCCL、HiCache 备份 kernel；主 stream 自己 64 KB 的 H2D（`resolve_forward_inputs`）一次 150–230 ms、GPU 上
无任何重叠操作。真正的 HiCache 回载 kernel（`hicache_transfer_per_layer`，stream 17）30 步共 166 ms，并不慢。HIP 运行时调用里没有长的 malloc / free。
机制：主 stream 每层 `LayerDoneCounter.wait_until` 等 HiCache 回载事件；回载前的索引 H2D 慢得反常（2 MB 160 ms），推测是 pageable 拷贝排队。
**影响**：PP0 30 步合计比"按中位数"多约 2.6 s，约 26%。新 trace 要验证这条链（哪个请求、多少 token 命中主机缓存、索引拷贝在等谁）。

**按中位数的稳态 kernel 预算（旧 trace PP0，每步约 237 ms）**：

| kernel | 中位耗时 | 每步 | 稳态占比 | 说明 |
|---|---|---|---|---|
| 稀疏 MLA `_sparse_mla_fwd_split_dim_kernel` | 5.2 ms / 层 | 104 ms | **约 44%** | 见下"稀疏 MLA" |
| MoE `mfma_moe1` + `gemm2_a4w4` | 1.27 + 1.10 ms / 层 | 40 ms | 17% | aiter 已有 8192 token 的调优项（FlyDSL，2327 µs，2.4 PFLOPS），trace 与之一致 |
| hipBLASLt BF16 GEMM（o_proj、dense、q_a+kv_a、wq_b） | o_proj 0.97 ms | 36 ms | 15% | o_proj 1.7 PFLOPS（**更正**：不是 0.86，上面第 4 条作废） |
| indexer logits `_gluon_fp8_mqa_logits` | 0.57 ms / 次 | 16 ms | 7% | 每请求每 indexer 层一次 |
| q / v 吸收 bmm（a8w8） | 0.32 ms | 13 ms | 5% | |
| q_b（triton a16w16，16384×2048） | 0.56 ms | 11 ms | 5% | 0.98 PFLOPS；镜像里 llama405B 的同形状 M=8192 调优项 1.05 PFLOPS，差不多 |
| top-k `topk_packed_kernel` | 0.24 ms | 7 ms | 3% | **PP1 中位 0.23 ms，与 PP0 相同**：上面第 1 条"慢 6 倍"是停顿造成的（PP1 一次 top-k 记了 854 ms），作废 |
| rope + 写 KV、add_rmsnorm | 0.19、0.06 ms | 6 ms | 2.5% | 上面第 2 条"PP0 小 kernel 慢 6–8 倍"同样是停顿，作废 |

**稀疏 MLA 走的不是调过的路径**：KV cache 是 FP8，`triton_sparse_mla_fwd` 的 gfx950 FP8 优化只认每卡 8 或 16 个头（TP4 / TP8 的 GLM-5.2），
我们 TP1 prefill 每卡 64 个头，落到单趟 `split_dim` kernel：每个 query token 一个 program、64 头一起算（KV 每行只读一次），
autotune 在 FP8 下只剩 `BLOCK_N` ∈ {32, 64}、`num_warps` ∈ {1, 2, 4}、`num_stages` ∈ {1, 2}。64 头 × 512 维的 fp32 累加器在 4 个 warp 下每线程约
128 个 VGPR，再加 q / KV 片，寄存器压力大、占用率低；8 个 warp 或"每 16 头一个 program"（split-K 路径，`BLOCK_H` = 16，KV 多读 4 遍但并行度高）
都没试过。microbench `scripts/bench_sparse_mla.py`（prefill 形状：8192 token、64 头、top-k 2048、4 个请求各 13.1 万上下文，overlap / uniform
两种索引分布；对比生产路径、扩展配置的单趟 kernel、split-K 1 / 2；逐个对比输出误差）排在 trace 那一步、140 GPU 3 上跑。

**稀疏 MLA microbench 结果（18:15–18:45，140 GPU 3；`results/bench_sparse_mla-*.txt`）——配置无收益，但找到了 1.4 倍差距的根因**：
- 小 KV 池（0.3 GB）上生产路径 3.73 ms，是 Triton 配置空间里最快的：单趟 24 个配置、head 分块 fused kernel 84 个配置、split-K 1 / 2 都不比它快
  （最好的 fused 64 头 × BLOCK_K 64 × 4 warp × 3 级流水 0.98 倍；8 warp、`BLOCK_N` 128、split-K 都更慢）。overlap 0.7 与 uniform 两种索引分布结论相同。
  另：`BLOCK_N` 16 + 4 / 8 warp + 2 stages 两个配置输出是错的（max |diff| 0.024，24% 元素偏差 > 2%），生产的 FP8 剪枝本来就排除了它们。
- **KV 池一过 2 GB 就慢 38%**：同样的请求和索引，池 1.9 GB → 3.71 ms，2.3 GB → 5.12 ms，90 GB → 5.12 ms（请求的 token 连续存放）/ 5.20 ms（完全打散）/
  5.15 ms（64 个一段）/ 5.12 ms（8192 个一段）。与局部性、TLB 无关，只看 tensor 大小：Triton 的 AMD 后端只给存储 ≤ 2³¹ 字节的张量参数做 32 位指针范围特化、
  生成 buffer load，更大的张量走 64 位地址的 global load；autotune 也随之改选 `BLOCK_N` 32。大池下换别的配置也追不回（最好就是 autotune 选的）。
- 生产上每个 PP rank 的 KV 池是 1,074 万 token、每层一个 6.2 GB 的张量（2 GB 只装得下 373 万 token），**trace 里的 5.2 ms 正是这条慢路径**。
  修好它约省每层 1.4 ms、每级每步约 28 ms（prefill 计算约 11%）。可行的方向：kernel 内把 KV 按 ≤ 2 GB 分段做多次带掩码的 buffer load；或换成按行 64 位基址
  + 32 位列偏移的 HIP / FlyDSL 实现；把设备 KV 池压到 373 万 token 以下可以直接验证，但设备前缀缓存缩到 35%、主机回载（停顿的来源）会大增，估计得不偿失。
  需要改 kernel，今晚不做端到端，记为高价值后续项。
- **两种 Triton 内的修法都试了，都不行**（`scripts/sparse_mla_chunked.py`，`bench_sparse_mla.py --chunked / --tilebase`，结果 `results/bench_sparse_mla-chunked-1027T1850.txt`；
  Triton 3.7.0 的判断在 `triton/backends/amd/compiler.py:189-216`：存储或 `ptr_range()` ≤ 2³¹−1 字节才加 `tt.pointer_range=32`）：
  - 分段视图：把 6.2 GB 池切成 3 个 ≤ 2 GB 的视图（带 `ptr_range()` 的指针对象，范围如实），每个 tile 对每段做一次带掩码的 buffer load 再按位 OR。
    结果正确，但 6.2 GB 池上 5.98 ms（`BLOCK_N` 32）/ 12.3 ms（64），比生产 5.12 ms 慢——load 指令和寄存器压力都 ×3；只有 1 段时（小池）3.79 ms ≈ 原 kernel。
  - 每 tile 标量基址：先对每行 top-k 索引排序（attention 与顺序无关），tile 内用最小行号做标量基址 + 32 位相对偏移，并声明 2 GB 范围。
    结果**错误**（max |diff| 0.28）：编译器把标量基址也折进 32 位偏移，buffer 描述符仍用参数基址，偏移溢出；而且即使小池上这种指针写法也退回慢路径（5.2 ms）。
    不声明范围、只排序：4.99 ms（局部性好 3%），但排序本身 0.30 ms，净收益为 0。
  - 剩下的可行方案：按 2 GB 段把每行索引分桶（排序 + 计数，约 0.3 ms），对每段用如实范围的视图各 launch 一次带 `topk_length` 的 kernel，输出部分结果 + LSE 再合并
    （类似 split-K 的 combine）。估算每层 5.12 → 约 4.5 ms，prefill 计算省 4–5%；需要给单趟 kernel 加 LSE 输出，工作量较大，排在 HiCache 停顿之后。
    根治要换非 Triton 的实现（HIP / FlyDSL 用 buffer load 的 idxen 结构化寻址：32 位行号 × 576 B 步长，可覆盖整个池）。

**prefill GEMM 调优结果（18:14–18:28，140 GPU 0–2）——无收益，不做端到端**：调出来的仍是同一族 hipBLASLt kernel（MT256x256x64），与默认启发式的选择一致。
M = 8192：o_proj 1,045 µs（trace 里未调优的是 968 µs）、gate_up 1,587 µs（1,660）、q_a+kv_a 228 µs（250）、wq_b 112 µs（110）、weights_proj 21 µs（20）；
合计每级每步省不到 1 ms（< 0.3%），在 C104 的复测波动（约 ±0.6%）之内。产物 `gemm-tune/glm52_prefill_bf16_tuned_gemm.csv`。

**prefill GEMM 调优**：镜像合并后的 aiter 配置里，M = 8192 / 4096 没有 gfx950 项的是 o_proj 6144×16384、dense gate_up 24576×6144、
q_a+kv_a 2624×6144（只有 M = 16384）、indexer wq_b 4096×2048、weights_proj 32×6144；q_b、dense down、router、wk 已有。
`gemm-tune/tune_prefill_bf16.sh`（yihou 同一条调优命令）调这 5 个形状 × 2 个 M，排在 trace 那一步、140 GPU 0–2 上跑
（`scripts/gpu_side_jobs_1027.sh` 等 trace 部署起来后启动；HiCache 500 GB 那一步本来就要等 140 的显存全空，所以不会和性能测量重叠）。
按上面的效率，GEMM 调优的上限约为 prefill 计算的 1–2%。

**外部仓库**（已克隆到登录节点 `/tmp/lyl_ref/`）：SIKL 有 GLM 形状的 FlyDSL bf16 GEMM（`n2624_k6144`、`n6144_k16384`、`n32_k6144`、`n4096_k2048`），
但只测到 M ≤ 4096、很多 M 上比 aiter 慢；FlyDSL 的 `fp8_paged_mqa_logits`、`topk_transform_paged`（top-k 512 / 1024）、稀疏 `flash_mla_with_kvcache`
都是 DSv4 decode 形状，要扩展到 top-k 2048、不分页的 prefill 版本。KernelForge 的方法（基线 → SNR 门槛 → 实测 → PMC）、FlyDSL MXFP4 MoE、
VSA / SLA 稀疏 attention 的 CK 实现、按 M 选 GEMM 实现的 sglang 补丁可借鉴；跑它的 agent 需要 Anthropic 凭据。

### 新 trace（llr37 + v3，C104，18:43，每级 30 步）：prefill 的 GPU 只忙 54%，瓶颈是调度线程——write_back 驱逐时同步等"正在跑的前向"结束

**结论**：四级的 GPU 每步实际工作都很稳（PP0 每步 GPU 跨度中位 252 ms、PP1 243、PP2 246、PP3 EXTEND 约 210 + DRAFT 约 21），**步内没有停顿**；
但两步之间 GPU 空着，**空闲中位 155–212 ms、均值 210–230 ms，GPU 忙碌率 PP0 54%、PP1 52%**。252 + 210 ≈ 460 ms，正好是计分窗口里的节拍均值 499 ms。
空闲时调度线程（CPU）在做下一批的准备，**其中 35% 的时间阻塞在 `hipEventSynchronize`**（PP0 30 步 4.65 s / 13.3 s；PP1 5.0 s / 13.7 s），
几乎全在 HiCache write_back 驱逐的 `writing_check` 里；这些同步里 **79% 的时间是在等当前正在 GPU 上跑的前向结束**（PP0 3.1 / 3.9 s），
前向结束后的 D2H 备份本身每次只要约 4 ms。→ 调度线程无法领先 GPU，下一批只能在本批前向结束后再做完驱逐、回载、准备和 PP 通信才发出。
**这比任何单个 kernel 的收益都大**：若空闲减半，prefill 吞吐约 +25%。

**机制（代码，镜像 v0522）**：
- `scheduler.py:614-618` 开 HiCache 时把 `cache_controller.write_fence_stream` 设为 `forward_stream`；
- 驱逐（`unified_radix_cache.py` `_evict_components`）对每个要降级的节点调 `_execute_and_commit_kv_backup(write_back=True)` →
  `writing_check(write_back=True)`：`start_writing(wait_for_forward=True)` 让 D2H stream 先等 forward stream 上已提交的全部工作
  （`cache_controller.py:843-847`，注释"An evicted SWA page can belong to an unfinished checkpoint"），然后 CPU 对每个 ack `finish_event.synchronize()`；
- 驱逐只腾"刚好够用"的量（`allocator/base.py` `evict_to_free_tokens` 的 shortfall、回载路径 `_load_back_transfers` 的 `needed`），
  所以设备池满了以后几乎每步都要驱逐（PP0 30 步 41 次 `evict_for_alloc`），每次都付一次"等前向结束"。
- 各级调度阶段（PP0，30 步合计，墙钟）：`get_new_batch_prefill` 6.0 s（45%），其中 `evict_for_alloc` 5.4 s、`writing_check` 5.2 s、
  `init_load_back` 3.0 s、`prepare_for_extend` 2.8 s；PP 接收 `recv_tensor_dict` 1.9 s（14%）；发射前向 `run_batch` 1.8 s（14%）；
  `check_hicache_events` 0.9 s。PP1 同样（驱逐 4.8 s、接收 2.4 s）。

**第 1 步部署的 prefill `/metrics`（PP0，整个部署累计，`c104-1027T1635/metrics/prefill0.prom`）印证了量级**：驱逐 12,005 次、
`eviction_duration_seconds` 合计 **1,686 s（平均每次 140 ms）**，而整个部署只跑了约 5,000 s；新算 8,478 万 token（约 1.04 万步），
每步驱逐约 5.7 万 token、回载约 5.0 万 token（主机命中占缓存命中的 34%）；备份 5.13 亿 token / 6.39 TB，D2H 约 25 GB/s、每 token 12.4 KB。
→ 512K 余量约每 9 步驱逐一次（现在每步 1.15 次），"等前向"的次数降约 10 倍；D2H 总量不变（约 24 ms / 步）。

**更正旧 trace 的"停顿"结论**：上面"旧 trace 复查"和 `prefill_trace_map.py` 的 STALL 行把主 stream 上个别操作的时长（一次 64 KB 的 HtoD 200 ms、
一个 5 µs 的 kernel 450–780 ms）当成 GPU 在等。新 trace 里这些操作期间**同一 stream 的后续 kernel 照常在跑**（同一 stream 不可能真重叠），
是 ROCm profiler 给部分操作记了过晚的结束时间；按"截到同 stream 下一操作开始"重算后，步内空闲只有 PP0 106 ms / 30 步。
所以"HiCache 回载事件让前向停 0.5–1.2 s"不成立，真正的损失在步与步之间；按 kernel 时长求和的 PP0 每步 543 ms 也夸大了（实际忙碌约 245 ms）。

**实验 E1（排队中）：驱逐留余量**——`2p1d-kit/llr35/patches/evict-headroom/`（`evict-headroom.patch`，只改 `unified_radix_cache.py` 的 `evict_for_alloc`：
请求量 < `SGLANG_EVICT_HEADROOM_TOKENS` 时按这个量驱逐），只挂 prefill，`config/site-llr35-v3-evict512k.env`（512K token ≈ 设备池 5%）。
不改语义：多出的受害节点照常备份、降级到主机，只是把"等前向结束"分摊到多步。预期：驱逐次数降到几步一次，GPU 空闲下降。
排在 HiCache 500 GB 之后（`scripts/night2_1027.sh`，`config/plan-night2-1027.txt`，138 `$L/night2-*.out`）。
**实验 E2（排队中）：去掉写栅栏**——`patches/no-write-fence/`（`config/no-write-fence.patch`：`SGLANG_HICACHE_NO_WRITE_FENCE=1` 时
`start_writing` 不让 D2H stream 等 forward stream），`config/site-llr35-v3-nofence.env`。收益更大（去掉全部"等前向"），但改了流间顺序。
安全性的论证：栅栏的注释只说 SWA（GLM-5.2 是 DSA，无 SWA）；被驱逐的节点 lock_ref = 0，在跑的前向只读写本批请求加锁的前缀和新页；
请求结束解锁发生在本级该批前向完成之后（非末级要先收到末级的结果）；释放的页要等 CPU 同步完备份才会被再分配；回载仍等 forward stream（读栅栏保留）。
**先过门禁再测性能**（`scripts/night3_1027.sh`，E1 之后自动跑）：prefill 设备池限 30 万 token / 级（`--max-total-tokens 300000`，
`config/site-llr35-v3-nofence-gate.env`），smoke + 200 题探测 + **两遍相同的** 16k × 128 长上下文（第一遍约 200 万 token，一直在驱逐、备份；
第二遍从主机回载第一遍的前缀——备份若和前向抢跑，读回来的 KV 就是坏的）；判定 `scripts/gate_pass.py`（smoke 6/6、探测乱码 0 且 ≥ 0.955、
每遍乱码 0 且捞针 ≥ 62/64、长前缀 GSM8K ≥ 60/64）。通过 → C104 性能（`config/plan-night3-1027.txt`）；不通过 → 同样限池、不打补丁的对照门禁，不测性能。

复现（138 上，trace 在 `$L/prof-llr37-v3/traces/prefill-184337-p0/`，结果在 `results/trace-llr37-v3-c104/`）：

```bash
T=$L/prof-llr37-v3/traces/prefill-184337-p0/prefill-184337-p0-TP-0-PP-0.trace.json.gz
python3 $P/scripts/prefill_gpu_timeline.py $T      # 每步 GPU 跨度 / 忙碌、步间空闲及空闲时调度线程在做什么
python3 $P/scripts/prefill_sched_cpu.py $T         # 调度线程的阻塞运行时调用（按调用点）与各阶段墙钟
python3 $P/scripts/prefill_stall_detail.py $T      # 主 stream 上的长操作：前一个 wait 来自哪里、重叠的旁路 stream 工作
```

## 结论（10-27 04:00，第 0 步分析；第 1 步在跑）

1. **prefill 多算的新 token**（`scripts/prefill_miss.py`，逐请求）：1P C112 的未命中 3.64% = 对话增长 2.22% + 会话在本次运行首次出现 1.17%
   + 已发过的前缀没命中 0.24%。最后一项几乎都来自每小时约 62 个请求整段前缀被驱逐（合计 0.22%，约占 prefill 新算量的 6%）；
   64 token 页对齐每请求只丢约 30 token。→ **候选：HiCache 主机层每级 250 → 500 GB**（4+4 下 138 只跑一个 prefill，内存 2.75 TB 够）。
   2P1D 的跨实例丢失更多（C176 已发前缀没命中 0.32–0.41%，首次出现的请求 68–70% 未命中，1P 是 56%）。
2. **10-24 报告里"PP0 每步多 115 ms 输入准备"要修正**（`scripts/prof_memcpy_callers.py`）：PP0 的 1.94 s HtoD 里 1,938 ms 是 15 次异步拷贝
   （HiCache `move_indices` 和 `resolve_forward_inputs`，走侧 stream，数据很小，CPU 端每次 < 0.1 ms，不阻塞调度线程），按 kernel 时间求和夸大了。
   PP0 独有的真实 GPU 工作约 30 ms / 步：`_prefix_kv_indices`（调度时拷缓存前缀的 KV 索引，约 13 ms）和 DSA indexer 取 index K 时的 Fill（约 18 ms）。
3. **开 async 后 prefill 基本是 GPU 受限**：四级时间差拉平到约 396 ms、节拍 414 ms，各级每步 GPU 忙碌 386–393 ms。剩下的大头是 GPU 上的工作：
   DSA 稀疏 attention（triton，每级每步约 100 ms）、indexer、GEMM、级间等待。
4. **KF overlay 不覆盖 prefill**：只处理 decode 的 TARGET_VERIFY 行，prefill 尺寸的调用回到原实现（`kf_sparse_mla.py` 注释 "prefill-sized launch: keep today's behaviour"）。
5. **prefill 的 BF16 GEMM 走未调优的 torch 路径**：5 种 (N, K)（6144×16384、24576×6144、2624×6144、32×6144、4096×2048），但 M 随 batch 变化，
   约 650 个不同值；aiter 按精确 M 查表，要调需先给 aiter 加 M 分桶。这几种 GEMM 约占 prefill kernel 时间 7–9%，预计收益 1–3%。
6. **TP2×PP2 重新评估：吞吐上大概率不划算**。节拍已由每级 GPU 工作决定，减少级数省不掉每级的开销；TP2 还要加每层两次 all-reduce、
   indexer 在 TP rank 间重复计算，而且 MLA 的 latent KV 在 TP rank 间复制，GPU 和主机缓存能放的 token 数减半（会加重第 1 条的驱逐）。

## 步骤与状态

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | kit 移植（`2p1d-kit/llr35` = yihou kit 副本，KF overlay 与 GEMM CSV 冻结在 kit 里，md5 见 `config/`）；不占卡的分析 | 完成（上面的结论） |
| 1 | `scripts/correct_and_perf.sh`：真实接受率的 smoke、GSM8K、200 题探测、长上下文 16k × 128 与 100k × 32；然后 4+4 C96 K3（2.99）性能基线 | 正确性 03:34–04:22 完成（乱码，见上）；基线按用户决定推迟到二分之后 |
| 1' | 乱码二分（`scripts/quick_correct.sh`） | 第一轮完成（主因 PP 通信重叠）；第二轮完成（第二个问题是 async）；09:01 起确认 async × K |
| 2 | 原定 TP2×PP2；按第 6 条建议改为 HiCache 500 GB（等用户确认） | 待定 |

## 复现

```bash
P=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
L=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/llr
# on crsuse2-m2m-138
cd $KIT && setsid nohup bash $P/scripts/correct_and_perf.sh $KIT $P/config/site-llr35-4p4.env $L $P/config/plan-llr35-base.txt \
    > $L/driver-$(date -u +%m%dT%H%M).out 2>&1 < /dev/null &
python3 $P/scripts/prefill_miss.py $RUN/agentx/aiperf_artifacts/profile_export.jsonl LABEL
python3 $P/scripts/prof_memcpy_callers.py TRACE.json.gz HtoD          # or kernel:SUBSTR
```

# prefill 提效（在 yihou 的 llr35 配置上），10-27 起

目标：yihou 的 4+4 C96 配置（llr35，47.8–48.0k）在 138 → 140 上复现并做 PD 正确性，作为基线；在这个工作点上找 prefill 的优化项。
每项只在 4+4 C96 上测，记录收益和适用场景（用户 10-26 定的方式）。第 6、7 步（Pareto 中段按点选 K、第二版 Pareto）暂不做（用户 10-27）。

## 10-27 夜：调优实验总表（结论先行，每组跑完更新）

方向来自 18:43 的 prefill trace（见下文"新 trace"一节）：prefill GPU 只忙 54%，调度线程 35% 的时间阻塞在 write_back 驱逐的同步里。
全部是 llr37 + 补丁 v3、138（prefill TP1×PP4）→ 140（decode TP4/DP4），C104、3600 s 计分，每组整栈重启；对照 = 第 1 步 48,360（yihou 48,408）。
分析统一用 `scripts/analyze_run.sh LABEL`（138 上跑，输出存 `results/run-LABEL-c104.txt`）。

| 组 | 改动 | tok/s/GPU | vs 48,360 | 请求数 | TTFT p50 / 均值 (s) | prefill 节拍 p50 / p90 (ms) | 排队均值 / 有排队的区间 | 驱逐 次/步、平均耗时 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 基线 | v3 | 48,360 | — | 11,624 | 6.23 / 8.97 | 380 / 940 | 6.27 / 82% | 1.16、140 ms | — |
| HC500 | 主机层每级 250 → 500 GB | **49,909** | **+3.2%** | 12,006 | 4.78 / 7.19 | 382 / 917 | 4.04 / 65% | 1.20、115 ms | 有收益，来自缓存 |
| E1 | 驱逐余量 512K token | **49,807** | **+3.0%** | 11,858 | 5.32 / 7.56 | 328 / 820 | 4.92 / 78% | 0.11、308–456 ms | 有收益，来自 prefill 节拍；复测排队中 |
| HC500 + E1 | 两项叠加 | 跑着（23:55 起） | | | | | | | |
| E1 复测 | 同 E1 | 排队（HC500 + E1 之后） | | | | | | | |
| E2 | 去掉写栅栏（先过门禁） | 暂缓 | | | | | | | 留给用户决定 |

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
另：23:01 E1 结束后队列空了 54 分钟——巡检只显示各队列日志拼接后的最后一行，night2 的结束行被 night3 日志里提前写的取消行盖住，我等的匹配没触发。

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

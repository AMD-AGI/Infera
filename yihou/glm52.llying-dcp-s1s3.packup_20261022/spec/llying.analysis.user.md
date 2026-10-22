---
cursor:
  subagentId: "bc-46f74157-fa18-5d21-bd71-8790e59f7aa0"
---

# GLM-5.2 triton PD AgentX：liying 在 yihou 基线之上的优化，以及双方最优 tput/chip/s

分析时间 2026-10-22。只读：`git fetch` 后读 `origin/dev/pd_opt/glm_5.2_agentx`（HEAD `89a44296`）、`origin/feat/sglang-mix-glm5.2-tp4-v0522`（[PR #210](https://github.com/AMD-AGI/Infera/pull/210)）、`origin/feat/decode-dp-request-balance`、`origin/main`；在 Crusoe 上只读核对原始 run JSON 的 sha256 和镜像 ID。未 checkout / 修改工作区，未跑实验。

口径：tok/s/chip = AgentX `request_metrics.throughput.per_gpu.total_tput_tps`，即（输入 + 输出）token/s ÷ prefill+decode 实际 GPU 数。liying 文档写作 tok/s/GPU，口径相同（两边 JSON 中 total ÷ GPU 数与 per_gpu 完全一致）。所有分数都是 EAGLE MTP 5/1/6 + `SGLANG_SIMULATE_ACC_LEN=3.61`，不含正确性。**[推断]** 标记的是推断，其余均有证据路径；仓库路径相对 `~/dev/gits/infera`，默认在 `origin/dev/pd_opt/glm_5.2_agentx` 上。

## 结论

1. **双方 PD 最优（真实模型）**
   - **yihou：25,523.7 tok/s/chip** —— g3x1 A1，C56，8 卡跨节点 1P(TP4/DPA)@crsuse2-m2m-135 → 1D(TP4/DPA)@crsuse2-m2m-136，v0.5.21 + OffloaderV2 G3，2026-10-21；同栈 A2 为 25,376.1。
   - **liying：35,637.0 tok/s/chip** —— 2P(TP4/DPA)+1D(TP8/DPA)，C176，16 卡，crsuse2-m2m-138 → 140，v0.5.22，2026-10-20，n=1。
   - liying 的 8 卡形态最优是 **32,532.3**（1P(TP1×PP4)+1D(TP4/DPA)，C64，138 GPU4-7 → 140 GPU4-7，2026-10-22，n=1）。
   - 三个最优点的原始 JSON 已与仓库副本逐字节核对（sha256 一致）。
2. **两边不可直接比较。** yihou 最优是 8 卡 C56；liying 最优是 16 卡 C176，处在 Pareto 曲线上低交互、高吞吐的一端（p90 交互 35.7 对 50.4）。镜像版本、custom all-reduce、prefill 缓存与 offload、router、调度策略、测量协议也都不同。最接近的一对是 liying 4+4 C64 与 yihou g3x1：两者同为 8 卡跨节点 1P1D，p90 交互相近（48.6 对 50.4），liying 高 27.5%。但两者至少有 15 项配置不同，且都是 n=1，差距不能归因到任何一项。
3. **liying 的工作并不是在 yihou 0923 triton 配置上做单变量叠加。** 她继承了 yihou 的基础配方（triton DSA、MTP 5/1/6 + 模拟接受、TP4/DPA prefill、fp8 KV、镜像配方和 harness），配置主体则沿用 yaocheng 的 2P1D triton 配置。在此基础上，她改变了 PD 比例和并行形态（3P1D → 2P1D → 1P1D → PP4 → 4+4），再在 router、调度、缓存、前端和正确性上调优（见 §2、§3）。
4. **liying 实测收益最大的单项**（同条件对照）：
   - HiCache write_back：+19.3%
   - prefill shortest-prefill-first：+15.4%
   - decode `--tokenizer-worker-num 4`：+13.6%
   - P 会话亲和：3P1D 上 +12.1%
   - DSA TileLang→Triton：+5.9%
   - R1（prefill 完成即释放）：+5.4%
   - R2（decode 按 token 需求选 rank）：+3.2% ～ +9.3%
   - PP4 prefill（与 prefill MTP 同时引入）：12 卡上 +5.9% ～ +17%
   - decode 按请求数均衡：吞吐 +2.7%、p90 交互 +43%
   - top-k v2：在 mixed（聚合）上 +8.6%
   
   无收益或负收益：R4 −34%、Dynamo 式 P 评分 −1.9%、chunk 8K −4.5%、prefill_complete −4.0%、R2 decode hit credit、D radix（吞吐持平）、decode mem 0.92（同并发无收益）。
5. **两边在正确性和口径上有两处关键差异，比较时必须带上：**
   - **custom all-reduce：** yihou g3x1 两腿都开着（来自 10-20 的实验规则）。yihou 09-18 与 liying 10-20 的排查都表明 custom AR 会导致乱码；仓库中没有找到 g3x1 配置的真实接受率正确性验证。liying 的 v0.5.22 配置两腿都关闭，并通过了 GSM8K（0.97）。
   - **prefill 未开 MTP：** yihou g3x1 和 liying C176 的 prefill 都没开 MTP，decode 草稿层会读到未初始化的 KV。liying 已定位这一问题，并在 PP4 / 4+4 / AUS a5 这几个点上修复（§3 B 段第 10 项）。

## 1. yihou 的 triton 基线

| 项 | 值 | 证据 |
|---|---|---|
| packup | `yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923/` | `README.md` |
| 入库 | 分支 `dev/pd_opt/glm_5.2_agentx`，commit `28121cb4`（2026-10-09，"bak"）；运行时 Infera 源码 `2e5a2d89f839…`（dirty） | `environment.md:180` |
| 配置 | `scripts/config.yihou.triton.sh` 叠加在 vendored `scripts/bench-harness/config.sh` 上。相对同节点 Phase-A 配置只改 4 项：triton DSA 双腿、`index_share_for_mtp_iteration=true`、prefill HiCache 开、full 模式（3600 s、warmup 10/lane） | `config.yihou.triton.sh:169-208` |
| 镜像 | `infera-sglang:v0519-yihou-0917-nextnfix-hicache`，`sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`（sglang `0.5.19.dev20260917+ga9fb1c3238`，含 NextN fusion 修复和 PR #37152） | `environment.md:86`、`results/runtime.env:31` |
| 形态 / 节点 | 同节点 1P1D P4D4，crsuse2-m2m-137；P 用 GPU0-3，D 用 GPU4-7；两腿都用 ionic_0..3；`PD_DP_RANK_AFFINITY=1` | `config.yihou.triton.sh:24-111` |
| 实测 server_args | 两腿均为 TP4/DP4 DPA、mem 0.85、max_running 64、每 DP chunk 8192（命令行 32768）、max_prefill 16384、fp8_e4m3、page 64、fcfs、sgl-kernel top-k<br>prefill：HiCache ratio 1.5、write_through（约 154 GB/rank），无 MTP，custom AR 开<br>decode：EAGLE 5/1/6、`--disable-custom-all-reduce`、无 HiCache<br>每 DP 的 KV token：P 2,282,304 / D 2,226,368 | `results/server-info.29001.json`、`server-info.29257.json` |
| router | kv-aware，overlap 权重 20/2，PD DP-rank 亲和 | `llying/config-compare/.../README.zh-CN.md` §9 |
| 结果 | 2026-09-23，profiling 3,613.6 s，09:48Z 结束；C40，新栈首轮，n=1<br>**21,032.5 tok/s/chip**（168,260 tok/s）<br>ISL / OSL 均值 145,279 / 987；成功 4,157，错误 4<br>TTFT p50 3.60 s；交互 p90 58.3 | `results/agentx_conc40.json` |
| yihou 自己的拆解 | triton + index_share 两项合计约 +12.5%（未分离，对照是一次只跑了 89% 窗口的 TileLang 同节点 run） | packup `README.md` |
| v0.5.21 后继 | e2g A2 22,113.6（C40，warm 协议，节点 102）。liying 从 3P1D 起用的镜像和 harness 就来自这个 v0.5.21 packup 的配方 | `yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/` |

yihou 从 21,032 到 25,524 的路径见 `yihou/tune.p4p4.yihou.md` §2-§3，作为对照：v0.5.21 镜像 + HiCache 450 → decode max_running 96 + prefill mem 0.88/0.90 → OffloaderV2 G4 → GTT HiCache 260（PR #42553）+ index_share + idlefix + custom AR 开 → 跨节点 RDMA → OffloaderV2 G3。yihou 自己的说明是大多数步骤没有做单变量隔离。

## 2. liying 的工作怎样接到这个基线上

- **对 yihou 基线的审计（分析，不是优化）。** `llying/config-compare/glm52-p4d4-c40_v0519-20260923_vs_v0521-20261012/`（commit `5a9f4af8`，10-14）逐字段对比了 0923（21,032）和 1012 e2g（22,114）。结论是引擎参数层面只有 3 项有意义的差异，另加 v0.5.21 新默认的 fused indexer；新栈对新栈时，v0.5.21 并不比 v0.5.19 快（e1c 21,049，+0.1%）。
- **09-22 ~ 10-10，P8D8 / P8D4，v0.5.19 系镜像（16 / 12 卡）。** 起点是在 AUS 上"对齐 yihou P8D8 C80（19,513）设置"，得到 20,517（+5.14%），见 `llying/glm52.1p1d.c80.aus.packup_20260922/README.md`。当时 DSA 用的是 TileLang；09-28 的 B4 才切到 Triton，依据是 yaocheng 的 `2p1d-sweep-triton-dsa-20260922` 和 yihou 的配置证据（`llying/session-affinity-20260928/followup-analysis/NEXT-EXPERIMENT-PLAN.zh-CN.md` §4）。
- **10-12 起，v0.5.21 / v0.5.22。** 3P1D 的 README 写明"配置沿用 yaocheng 的 2P1D；镜像基底 `infera-sglang:v0521-yihou-1010-mc4340`；harness 以 v0.5.21 pack-up 的 bench-harness 为底"（`llying/3p1d-tp4-tp8-v0521-20261012/README.zh-CN.md`）。triton DSA 双腿一直保留到 PR #210 的 `examples/sglang_2p1d_glm5.2/engine/worker.sh`。
- **继承自 yihou、不计为 liying 优化的部分：**
  - triton DSA；EAGLE 5/1/6 + 模拟接受 3.61；TP4/DP4 DPA prefill；fp8 KV；chunk 32768；kv-aware 20/2
  - prefill HiCache（kernel / page_first）、`SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512`
  - kv_events 非阻塞、absorb-bmm 配置关闭这两个 runtime mount
  - decode PID 偏移；`GROUPED_TOPK=0`
  - "decode custom AR 破坏推测路径"这一根因（yihou 09-18）
- **[推断]** 所以下面的"liying 优化"是相对这套基础配方的增量。各阶段的基线点、节点、镜像都不同，不能把各项收益连乘成"相对 21,032 的总提升"。
- yihou 10-22 02:42 写了交接文档 `yihou/tune.p4p4.yihou.md`。此后 liying 的提交（4+4 C64、decode 均衡 A/B）都延续她自己的 v0.5.22 / AUS 线；liying 的配置和文档中没有引用这份交接，也没有使用 g3x1 的配方（OffloaderV2、PR #42553、idlefix）。依据是 grep `llying/` 下 md/sh/env/diff 无命中，命中只出现在 JSON 的数字里。

## 3. liying 的优化项（逐条）

表中"类型"列标注改动类别。百分比都是表中写明的那一对 run 之间的同条件对照；"跨节点 / 跨栈"的地方已注明。

### A 段：P8D8 / P8D4，v0.5.19 系镜像，C80（09-22 ~ 10-12）

| # | 优化项（类型） | 改了什么；commit / 文件 | 动机 | 实测影响（证据） |
|---|---|---|---|---|
| 1 | Mooncake 跨 rank 修复（镜像 / 传输） | 加入 dest-pinned HCA、dest-local-rail、RC ACK timeout 18。先作为 overlay 镜像 `…-mcdestpin-ibto18`：`7e13937f`、`0f2b0cf3`（`llying/cross-rank-rca-20260921/fix/`）；后合入 main：PR #187 `915265d1`（`deploy/docker/patches/mooncake_cpp/rdma_dest_local_rail.diff`、`rdma_ack_timeout.diff`） | ionic rail 隔离下，P rank ≠ D rank 的传输出现 ACK timeout / QP stall；yihou 基线靠 `PD_DP_RANK_AFFINITY=1` 规避，无法做 P DP4 → D DP8 | 使能项，没有吞吐 A/B。此后 3P1D、2P1D、1P1D 都是 `PD_DP_RANK_AFFINITY=false`（`llying/cross-rank-rca-20260921/REPORT.zh-CN.md`） |
| 2 | R1：prefill 完成即释放路由记账（router） | `--pd-prefill-guard-release completion`。09-23 先做 overlay 试验（`2c1fdc51`，`llying/p8d8-adaptive-31625-20260923/`）→ `225f9abb`（10-08）→ main PR #185 `b5afce15`（`rust/router/src/{disagg,config,policy}.rs`、`prefill_guard_tests.rs`）；launcher 参数 `d913b48c` → PR #188 | 原逻辑等 D 结束才释放 P 的预约，P 的选择失真 | P8D8 同节点同进程：20,166 → 21,264（**+5.44%**），TTFT p90 −29.9%（`llying/experiment-summary-20261008/REPORT.zh-CN.md`）。3P1D C144：19,503 → 19,308（噪声范围），TTFT −6~8% |
| 3 | P 会话亲和（router） | `--session-affinity prefill`：`ef92b4f4`、`50e4a67c`、`d7c8cae3`（09-28，`rust/router/src/session_affinity.rs` 等）；PR #188 `e54e9c62` 改为按 X-Session-ID 绑定 | 多轮会话在 P 之间漂移，前缀缓存失效 | P8D8：+1.88%（D 换了节点）。**3P1D C144：19,308 → 21,639（+12.1%）**，缓存份额 88% → 94%，prefill 新算量 −42%（3P1D `RESULT.zh-CN.md` §0、§3） |
| 4 | DSA TileLang → Triton（kernel 后端） | P/D 都改 `--dsa-*-backend triton`，其余不变；`fee3eb1b`、`9a0b4ad7`（`llying/router-capacity-campaign-20260928/{rb,b4}/`） | 参照 yaocheng 2P1D 的 Triton 对照和 yihou 的 triton 配置 | 同节点：21,481 → 22,756（**+5.94%**），TTFT p50 / p90 −24% / −32%，ITL 持平 |
| 5 | D radix（MTP 下）+ D 会话亲和（engine + router） | `3f0aa62a`（`infera/engine/sglang/args.py`）→ PR #184 `cb568f59`；`SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` | 减少重复 KV 传输 | P8D8：+0.23% / +0.55%，D 本地复用 91.8%，liying 自己判断吞吐因果证据不足；P4D4 C40（对齐 yihou t2f，TileLang）：+3.0%（`llying/decode-kv-aware-mtp-radix-20260924/analysis/PERF-AB.zh-CN.md`）。之后的配方没有采用 |
| 6 | R2：decode 按输入 token 需求选 rank（router） | `--decode-input-demand on`：`8cb2914e`（09-24）→ `f3560cb1`、`d913b48c` → PR #188 `e54e9c62`（`rust/router/src/decode_demand.rs`、`policy.rs`） | decode 不上报 KV，原先按请求个数轮转，D 各 rank 的 KV 不均 | P8D4：21,081 → 23,051（+9.3%，节点从 136/138 换成 137/135）（`llying/b4-align-crsuse2-20261008/`，`e78f9a5d`）。3P1D C144：21,639 → 22,340（**+3.2%**），KV>90% 的时间 6% → 1% |
| 7 | P8D4 调优（形态 / 参数） | decode 改 TP4（共 12 卡），graph bs 64，decode mem 0.90，只开 P 亲和（`23bd9838`） | 用更少的 decode 卡提高每卡吞吐 | 同节点：19,259 → 21,081（+9.5%） |

A 段无收益或负收益（同上报告和目录）：R4 有效工作量评分 −34.4%（13,958）；Dynamo 式首次 P 评分（`19a6c1d2`）−1.88%；chunk 8K −4.47%；C112 −9.5%；P KV 扩到 340 万 +2.66%（跨节点，看不到端到端收益）；R2 decode hit credit（`c8b0069f`）"mechanism works, no gain"（`ccd4862b`）。

### B 段：v0.5.21（yihou mc4340 镜像 + Mooncake 补丁）；3P1D 20 卡（crsuse2）→ 2P1D 16 卡（AUS）（10-12 ~ 10-21）

| # | 优化项（类型） | 改了什么；commit / 文件 | 动机 | 实测影响（证据） |
|---|---|---|---|---|
| 1 | 镜像 `v0521-llying-1012-mc4340-destpin`（镜像） | 在 yihou 的 `v0521-yihou-1010-mc4340` 上只重编 Mooncake `engine.so`（PR #4340 port + dest-local-rail + IB timeout + dest-pinned HCA），设 `MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=18`；ID `sha256:afcc2f8abff7…e820a`。`5a9f4af8`（`llying/3p1d-tp4-tp8-v0521-20261012/image/`） | 在 v0.5.21 上支持跨 rank 传输 | 使能项 |
| 2 | 3P(TP4/DPA)+1D(TP8/DPA)（形态） | 20 卡，亲和关闭。参数沿用 yaocheng 2P1D：P mem 0.85、max-running 256、graph 256、HiCache ratio 1.5；D TP8、mem 0.85、custom AR 开（`config/config.3p1d.sh`） | 提高 P:D 比例和 decode 宽度 | r2 基线扫描的最高点是 C112 20,545（3P1D RESULT §1） |
| 3 | `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0`（kernel / 正确性） | 从 yaocheng 的 1 改回 yihou v0.5.21 的 0 | r1（=1）已知答案门禁 0/48 | r2 改成 0 后 48/48（3P1D RESULT §2、`REVIEW.zh-CN.md`） |
| 4 | prefill mem 0.88 + HiCache 400/200/200 GB（缓存） | `config/best.sh`（`HICACHE_SIZE_prefill_*`） | 扩大 prefill 缓存 | C144 持平（22,238）；C160 21,161 → 22,279（+5.3%） |
| 5 | decode `--tokenizer-worker-num 4`（前端 / CPU） | `config/stage-tok4.sh`、`best.sh`（`5a9f4af8`） | decode 单进程 TokenizerManager CPU 100%，decode 侧 TTFT 30 s，而 prefill 3 s 就已完成 | **C144：22,238 → 25,269（+13.6%）**；C160 28,479；C176 30,562（同一个栈） |
| 6 | decode mem 0.92 + C192（显存） | `config/stage-mem092.sh`；`aa2bc285`、`07fcf065` | decode KV 已接近约束 | 同并发无收益：C176 30,562 → 30,291（−0.9%）。C192 = **31,829**（3P1D 最高，n=1，并发也同时变了） |
| 7 | prefill 关 QuickReduce（数值 / 稳定性） | `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`（AUS `config/stage-b1-wt-qr0.sh`，`3e3d6c04`） | prefill 挂起的根因：DP attention 下同组 rank 选了不同的 all-reduce 后端（QuickReduce 对 RCCL）而互锁，有 rocgdb 证据 | 不再挂起；代价是 prefill 满 chunk 慢约 13%。a1-wt2 C176 25,695（AUS RESULT §2.4-2.5） |
| 8 | HiCache `write_back`（缓存策略） | prefill `--hicache-write-policy write_back`，只重启 prefill | 让 GPU 池也计入缓存容量（每 rank GPU 2.5M + host 5.56M token） | **25,695 → 30,659（+19.3%）**；缓存份额 93.0% → 95.3%，新算量 −21%，排队 16.5 → 10.9 s（§2.7） |
| 9 | prefill `--schedule-policy shortest-prefill-first`（调度） | 只重启 prefill | FCFS 头阻塞：49% 的忙碌时间被单个长冷请求占住 | **30,659 → 35,377（+15.4%）**，排队 10.9 → 3.25 s；代价是 TTFT 最大值 685 s（§2.9） |
| 10 | prefill 也带 EAGLE 参数，即 prefill MTP（正确性 / 性能） | AUS `config/stage-b6-spf-pmtp.sh`（`d1e7ff1b`、`f671b0e2`）；v0.5.22 example 用 `d395c3b3`（`examples/sglang_2p1d_glm5.2/engine/worker.sh`） | prefill 没有 NextN 层，PD 传输里没有草稿 KV / DSA index-K；decode 草稿层读未初始化显存，冷态下 `k_scatter` 慢 15–170 倍，冷热之间漂移 | AUS C176：冷态 33,793 → 34,582（+2.3%），对热态 34,862 是 −0.8%。liying 因此宣布 34,582 为新基线，35,377 不再作为基线（§2.13-2.14）。v0.5.22 上真实接受长度中位 3.86 → 4.60 |

B 段无收益或负收益：prefill_complete + write_back + SPF 在 C192 为 32,339（−4.0%）；SPF 加到 C192 为 33,698（−4.7%，decode KV 被等传输的请求占满）；3P1D 去掉一个 prefill（t5b，2P1D）C176 为 28,900。

### C 段：v0.5.22（PR #210 分支），crsuse2-m2m-138 + 140（10-19 ~ 10-22）

| # | 优化项（类型） | 改了什么；commit / 文件 | 动机 | 实测影响（证据） |
|---|---|---|---|---|
| 1 | v0.5.22 镜像（镜像） | `135fba05`（`deploy/docker/Dockerfile.sglang` 换到 v0.5.22 base）、`ca21a67d`、`36eb84e6`、`6ba22577`（逐补丁裁决）、`4403eda6`（`infera/engine/sglang/args.py` 的 `--cuda-graph-max-bs` 兼容）；用 `fdbafb60` 构建出 `infera-sglang:v0522-fdbafb60`（`sha256:509b8427…04e2`） | 跟进上游 | PD 上没有单独 A/B |
| 2 | 两腿 `--disable-custom-all-reduce`（正确性） | `00da3cf3`（mixed）、`5b8a398f`（2P1D example） | aiter custom all-reduce 在 DP attention + decode CUDA graph 下乱码：v0.5.22 关掉后 GSM8K 0.975、0 条乱码，开着时 53–61 条乱码。与 yihou 09-18 的结论一致（`llying/mix-v0522-correctness-20261019/README.zh-CN.md`） | mixed C48：吞吐 +0.9%，p90 交互 −14.7%（`93c9e016`）。2P1D 的 decode TP8 每步比 AUS 慢 25–30%，liying 推断主因是这一项，但未单测 |
| 3 | `SGLANG_OPT_USE_TOPK_V2=true`（kernel） | example 的 env（`00da3cf3`） | DSA top-k 的新 kernel | mixed C48 同节点（140）：31,931 → 34,662（+8.6%）；GSM8K 0.980，长上下文 needle 64/64（`76eeb500`、`61b9864b`）。PD 上没有单独 A/B |
| 4 | `MC_MAX_MR_SIZE=1073741824`（传输） | example `worker.sh`（`22e40814`；复现脚本 `scripts/mr_align_test*.py`） | ionic 拒绝从 1 GiB 对齐地址开始的 2 GiB dma-buf MR，decode 起栈时 KV 注册失败 | 使能项 |
| 5 | 2P1D 融合配置（参数组合） | 以 AUS a1-spf 为底，叠加 mixed 的结论：<br>P：mem 0.88、HiCache 250 GB write_back、SPF、`HSA_NO_SCRATCH_RECLAIM=1`<br>D：mem 0.92、graph 256、max-running 256、tokenizer ×4<br>router：kv-aware 20/2 + 会话亲和 + R1 + R2；`index_share=false`（example 加入 `5b8a398f`；C176 参考点 `b5041eca`） | 汇总前面各单项 | **C176 = 35,637（16 卡）**，与 AUS a1-spf 35,377 持平（+0.7%）；TTFT p90 15.0 → 8.1 s，ITL p50 20.0 → 24.8 ms（`llying/2p1d-v0522-20261020/README.zh-CN.md`） |
| 6 | 1P(TP4)+1D(TP8)，12 卡（形态） | `PREFILL_WORKERS=1`（`4ad59fc2`；`config/kit-v5-1p.diff`） | 2P1D 在 C≤128 时 prefill 利用率低 | C64：23,241，比 2P1D C64 的 18,202 高 27.7%。C88：27,453，在 Pareto 上支配 2P1D 的 C96 和 C128 |
| 7 | PP4 prefill（TP1×PP4）+ chunk 8192（并行 / 调度） | kit `v6-pp4`（`config/kit-v6-pp4.diff`，不进 PR）：`--tp-size 1 --pp-size 4 --disable-overlap-schedule`、`SGLANG_PP_LAYER_PARTITION=20,20,20,18`；修复 `deepseek_v2.py` PP 中间级的 `embed_tokens` 问题（只挂载，不改镜像）。`6a1b5716`、`100cfece` | 参照 GB300 Dynamo 的 PP prefill，提高单 prefill 容量 | 均为 12 卡，对照 TP4/DPA 冷态；**prefill MTP 同时引入，两个变量叠加**：<br>C88：27,453 → 29,071（+5.9%）<br>C64：23,241 → 27,253（+17%）<br>C32：14,153 → 16,085（+14%）<br>chunk 16k → 8k：+3.1%<br>C112：29,540（单 PP4 prefill 约 3.0 req/s 到顶） |
| 8 | 4+4：1P(PP4)+1D(TP4/DPA)，8 卡（形态） | `config/site-1p-pp4-ck8k-d4.env`；`agentx.sh` 的 `DECODE_TP` 改为按卡数计算（`92a0d890`）；结果 `70cf64f5` | 12 卡里 prefill 只占 1/3，单 PP4 到顶后每卡上限约 30k | **C64 = 32,532（n=1）**，p90 交互 48.6，TTFT p50 1.9 s。低于 liying 自己估计的 38–40k：decode 每 token 变慢，qps 2.45 → 2.04 |
| 9 | decode 按请求数均衡（router） | `--decode-request-weight`，即 R2 的代价加上"权重 × 在跑请求数"。`a8cf30bf`，在分支 `feat/decode-dp-request-balance`，**未合入**（`rust/router/src/{policy,config,decode_demand,metrics}.rs`、`infera/server/{args,launch_rust}.py`） | DP attention 下每步步长由最忙 rank 的在跑请求数决定，而 R2 只均衡 token | AUS 1P1D C88，1800 s，同部署 A/B：26,462 → 27,186（+2.7%），p90 交互 55.7 → 79.6（**+43%**），TTFT 均值 +15%（`llying/1p1d-decode-balance-aus-20261021/RESULT.zh-CN.md`，`4944531e`、`bfeb79c6`） |

非性能产出：PR #210（v0.5.22 镜像 + mixed / 2P1D 两个 example，draft）；custom AR 乱码根因；config-compare 审计；InferenceX / GB300 / ATOM 对比分析（`llying/gb300-atom-pd-analysis-20261021/`、`llying/inferencex-glm52-b300-dynamo-agg-20261015/`）。

## 4. 双方最优 tput/chip/s（原始记录）

| | yihou 最优 | liying 最优（PD） | liying 8 卡最优 |
|---|---|---|---|
| tok/s/chip | **25,523.7**（同栈 A2 25,376.1） | **35,637.0** | **32,532.3** |
| run | g3x1 `agentx-c56-A1`（新栈首轮） | 2P1D `c176-1020T1645` | 4+4 `c64-1022T0334` |
| 原始 JSON | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/g3x1/agentx-c56-A1/agentx_conc56.json` | crsuse2-m2m-138 `/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/c176-1020T1645/agentx/agentx_conc176.json` | crsuse2-m2m-138 `…/runs/c64-1022T0334/agentx/agentx_conc64.json` |
| 仓库副本 | `yihou/glm52.v0521-baseline-xnode-g3g4.packup_20261022/results/rounds/g3x1/agentx-c56-A1/agentx_conc56.json` | `llying/2p1d-v0522-20261020/results/c176/agentx_conc176.json` | `llying/2p1d-v0522-20261020/results/pp4-ck8k-d4-c64/agentx_conc64.json` |
| sha256（原始 = 副本） | `76a8f595…96b0` ✓ | `0b76ede9…e195` ✓ | `5bf682d2…5585` ✓ |
| 时间（UTC） | 2026-10-21，bench 13:23:49–14:47:37（含 warmup，`status.tsv`） | 2026-10-20，计分 17:16–18:16（README），JSON 写于 18:23:18 | 2026-10-22，计分 03:54–04:54，JSON 写于 04:57:33 |
| 并发 / 窗口 | C56 / 3,627.4 s | C176 / 3,629.8 s | C64 / 3,628.1 s |
| 形态 / 卡数 | 1P(TP4/DP4 DPA) + 1D(TP4/DP4 DPA)，8 卡 | 2P(TP4/DP4 DPA) + 1D(TP8/DP8 DPA)，16 卡 | 1P(TP1×PP4) + 1D(TP4/DP4 DPA)，8 卡 |
| 节点 | P：crsuse2-m2m-135 GPU0-3 → D：crsuse2-m2m-136 GPU0-3 | P0/P1：138 GPU0-3 / GPU4-7 → D：140 GPU0-7 | P：138 GPU4-7 → D：140 GPU4-7 |
| 镜像 / ID | `infera-sglang:v0521-yihou-1010-mc4340`，两腿 `sha256:e8e2838f4a650584e92f1731cce26d3c7c500724f8befd301ecaac644763e22b`（`audit.txt`）；另有 6 个 sglang 源文件 runtime mount 和 offloader mount | `infera-sglang:v0522-fdbafb60`，`sha256:509b842716b88bbe13a38b3ff1f6a5b0d5d2ee6579f5c34761e0bc74cfba04e2`（138 和 140 上 `docker image inspect` 一致，创建于 10-20 15:28Z） | 同左，prefill 另挂 `deepseek_v2.py` |
| ISL 均值 / p50 | 130,174 / 82,880 | 129,353 / 92,585 | 126,487 / 78,911 |
| OSL 均值 / p50 | 1,011 / 446 | 972 / 409 | 991 / 446 |
| 成功 / 错误 | 5,646 / 5 | 15,881 / 11 | 7,407 / 4 |
| TTFT p50 / p90（s） | 4.28 / 21.96 | 4.28 / 8.08 | 1.90 / 4.20 |
| 交互 p50 / p90 | 61.9 / 50.4 | 40.4 / 35.7 | 52.9 / 48.6 |
| 重复次数 | G3 只有一个栈：A1 + A2 = 25,524 / 25,376。G4 跨节点两个栈共 4 次：24,362–24,976 | n=1。同配方 v0.5.21 在 AUS 跑过 3 次：33,793–35,377（镜像和集群都不同） | n=1；C80 正在跑 |

其他参考点：
- liying：AUS a1-spf 35,377.4、a5-pmtp 34,582.3（16 卡，v0.5.21）；3P1D t6-mem092 C192 31,829.1（20 卡）；1P(PP4)+1D C112 29,539.6（12 卡）。
- yihou：g4x1 / g4x2 跨节点 C56 24,362–24,976；同节点 C52 23,367。

## 5. 可比性

**相同的部分：** 指标定义；MI355X crsuse2 集群（物理节点不同）；客户端都是 InferenceX `918524ff`、aiperf `inferencex-agentx-mvp`、`semianalysisai/cc-traces-weka-062126`（393 条 trace）、3600 s、warmup 10/lane、模拟接受 3.61。证据：yihou 侧见 `yihou/glm52.v0521-baseline-xnode-g3g4.packup_20261022/scripts/harness/config-chain/config.sh:22` 和 `results/rounds/g3x1/status.tsv`；liying 侧见 `results/c176/runtime.env` 和 `benchmark_command.txt`。三个最优点的 ISL 均值都在 126k–130k。

**不同的部分**（yihou g3x1 对 liying 的两个最优点）：

| 维度 | yihou g3x1 | liying C176 | liying 4+4 C64 |
|---|---|---|---|
| 卡数 / P:D / 并发 | 8 / 4:4 / C56 | 16 / 8:8 / C176 | 8 / 4:4 / C64 |
| prefill 并行 | TP4/DP4 DPA | 2 个 TP4/DP4 DPA | TP1×PP4 |
| decode 并行 | TP4/DP4 | TP8/DP8 | TP4/DP4 |
| sglang | 0.5.21（`11b1fb21de`）+ 本地补丁 / mount | 0.5.22（`fdbafb60` 构建） | 0.5.22 + PP 修复 mount |
| prefill mem / chunk | 0.90 / 32768（每 DP 8192） | 0.88 / 32768 | 0.88 / 8192 |
| prefill 缓存 | GTT HiCache 260 GB/rank（PR #42553），write_through | HiCache 250 GB/rank，write_back | 250 GB，write_back，4 级共用 |
| OffloaderV2 | G3（K1，step 1） | 关 | 关 |
| prefill 调度 / prefill MTP | fcfs / 关 | SPF / 关 | SPF / **开** |
| decode mem / max_running / graph bs | 0.95 / 96 / 64 | 0.92 / 256 / 256 | 0.92 / 256 / 256 |
| decode tokenizer / detokenizer | 4 / 4 | 4 / 默认 | 4 / 默认 |
| custom all-reduce | **开**（两腿） | **关**（两腿，同时关 QuickReduce） | 关 |
| top-k v2 / index_share | false / true + idlefix | true / false | true / false |
| router | kv-aware 20/2 + PD DP-rank 亲和 | kv-aware 20/2 + 会话亲和 + R1 + R2 | 同 C176 |
| Mooncake | 4 rail，rail i↔i，xferfix2 `conn.py` | 8 rail，dest-local-rail，IB timeout 18，`MC_MAX_MR_SIZE` 1 GiB | 同 C176 |
| 测量协议 | 新栈 A1；同栈 A2 在两腿 `/flush_cache` 后 | 每个点整栈重启，跑 1 次 | 同 C176 |

判断：

- **C176（16 卡，35,637）和 g3x1（8 卡，25,524）不可比。** AgentX 是闭环负载，每卡吞吐随 P:D 比例和并发变化，也随交互水平变化。C176 的 p90 交互只有 35.7，yihou 是 50.4，两者处在 Pareto 曲线的不同位置。
- **4+4 C64（32,532）是最接近的一对。** 同为 8 卡、跨节点、1P1D，p90 交互相近，liying 高 27.5%，TTFT p50 只有 yihou 的 44%。但上表有 15 项以上配置不同，节点不同，两边都是 n=1。**[推断]** 27.5% 的差距远大于 yihou 跨节点 G4 四次之间约 ±1.3% 的波动，不太像纯噪声；但无法拆分到任何单项。
- **两边最优点各自的口径问题：**
  - yihou g3x1 和 liying C176 的 prefill 都没开 MTP（g3x1 `audit.txt` 显示 `speculative_algorithm=None`；C176 见 `b5041eca` 的 `worker.sh`），decode 草稿层都读未初始化的 KV。**[推断]** 两者都处在 liying 在 AUS 上测到的冷态降速（约 2–3%）里；g3x1 的 A1 25,524 和 A2 25,376 相差不大。
  - yihou g3x1 开了 custom AR。yihou 09-18 的结论（`yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/`）和 liying 10-20 的结论都表明它导致乱码。在 g3g4 和 idxshare-maf-fix 两个 packup 的文档中，没有找到对 custom AR 开启配置的正确性验证（检索仅限这两个 packup 的 .md）。liying v0.5.22 的配置关掉了它，GSM8K 0.97。因此即便吞吐数字可比，两者的正确性前提也不同。

## 6. 未计入和其他注意事项

- **liying 正在跑的点：** 4+4 C80（138 `…/runs/c80-1022T0513/`，P 用 138 GPU4-7、D 用 140 GPU4-7），05:24Z 时 warmup 才到 69/884，还没有结果。
- **NFS 上 yihou 尚未入仓的新结果（`/shared_nfs/yihou_glm52/dcp_pd_20261021/`，10-21 ~ 10-22）：**
  - 最高的两条是非 PD 的单机聚合 run：`mixrun/agentx/full1-dcp4-c40/agentx_conc40.json` = 26,277.0、`full-kvb-dcp4-c40` = 26,152.0。它们是 TP4/EP4 + DCP4、4 卡、`disagg=False`，镜像为上游 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020`。
  - 同期的 PD + DCP run（`pd/runs/pdBeperf-*`、`pdE2-*`，8 卡）最高只有 19,400。
  - 所以 yihou 的 PD 最优仍是 g3x1。
- **排除的 yihou 合成臂：** `8expert-*`（26,462 / 26,072，v0.5.19，两腿都用合成 8 专家模型）、x8p / x128p（25,905 / 25,297，截断专家）。见 `yihou/glm52.v0521-hicache-sensitivity.packup_20261014/README.md:36` 和 `glm52.v0521-CONCLUSIONS.yihou.md` §1。
- **排除的 liying 聚合（mixed，非 PD）结果：** 最高 34,662（C48，v0.5.22，140，关 custom AR + top-k v2）。
- **元数据标注问题：** liying PP4 系列的 AgentX JSON 把 prefill 记成 `prefill_tp=4, prefill_pp=1`（`agentx.sh` 写死），实际是 TP1×PP4。GPU 分母（4 + 4 或 4 + 8）是对的。
- **AUS 镜像 ID 随节点变化：** 同一个 tag `v0521-llying-destpin-aus1019`，a1-spf 上是 `f38645fb…`，a5-pmtp 的 decode 上是 `7826037b…`。这与"docker load 换节点会改变镜像 ID"的已知现象一致。

## 7. 核对方法

- **仓库侧：** 用只读脚本，通过 `git show` 读取全部 158 个 yihou 和 105 个 liying 的 `agentx_conc*.json`，按 per_gpu 排序，并用 total ÷ GPU 数复核。
- **router 代码归属：** 用 `git log -S` 定位 `decode-request-weight`、`decode-input-demand`、`pd-prefill-guard-release`、`session-affinity` 等参数首次出现的提交，确认所在分支和文件。
- **Crusoe 侧：** 登录节点 `crsuse2-slog-004` 只接受交互会话。本机 `known_hosts` 第 23 行是旧 key；我用已有第 11 行可信记录（指纹 `SHA256:HCYKYhwDi…4SI`）生成临时 pinned known_hosts，保持严格校验，用完即删，没有改 `~/.ssh`。在交互会话里只执行了 `ls`、`sha256sum`、`docker image inspect` 和只读的 Python 读 JSON，没有启动进程，也没有碰容器。


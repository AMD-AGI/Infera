# kernel 优化进展（11-09，按 KERNEL-PLAN-20261109 阶段 0 执行）

用户 07:44 同意边 review 边开始。优先级按用户决定：TP8 MoE 优先，同时做 TP4 / TP8 共用的 absorb BMM。
本文件只记录实测；计划与背景见 `KERNEL-PLAN-20261109.zh-CN.md`。

**最新状态（11-10 02:05 UTC，细节见第十四节末尾）**
- **MegaMoeTP 在真实 decode 里没有兑现收益**：单节点 4+4（TP4）同节点 A/B（n04-33）吞吐 49,016 对 49,242、交互 66.1 / 61.3 对 65.7 / 60.1，
  decode 步长 bs 16–24 完全持平、bs 5–12 慢 1–3 ms。单算子里的每层 −42 µs 没有出现。小 token 数测试显示它在小 batch 下本身不快（全局 64 token
  已约 200 µs）。原因要看 graph 内 trace：第一次的 trace 因抢占时拷贝失败丢了（已修脚本），已重排。**在原因查清之前不建议采用。**
- MegaMoeTP 的正确性没问题：TP4 和 TP8 两个 PD 门禁都通过（197/200、195/200，16k 64/64，接受长度 3.57 / 3.69）。
- TP8 PP8 的 MegaMoeTP 计分 01:58 起在 35436 上跑（之前三次被抢占 / 起栈段错误打断）。
- 起栈段错误定位到 `scheduler.py` 启动服务前读 stream 句柄时 HIP 内部崩溃，与节点强相关（n03-33、n04-25、n01-21 每次都崩，已排除），
  与我们的改动无关；根因未查到，是否报给管理员请你决定。

**之前的状态（21:10 UTC）**
- **最大的新收益：aiter 的 MegaMoeTP**（整段 all-gather + MoE + reduce-scatter 做成一个 kernel，aiter 已有、SGLang 未接入）。真实 GLM-5.2 权重上
  每个 MoE 层 TP8 快 34–50 µs（约 −2.7 ~ −3.5 ms/步）、TP4 快约 42 µs；与生产离 fp32 参考一样远（精度等价）。
- **已接入 SGLang**（overlay-kopt 的 `kopt_megatp`，`KOPT_MEGATP=1` 开启，生产 overlay 不变），**单节点 4+4 PD 门禁通过**：197/200、
  16k 64/64、接受长度 3.57。TP4 C104 计分对比和 TP8 PP8 的门禁 / 计分在排队（抢占和坏节点 n03-33、n04-25 拖慢了进度）。
- PP8：不加 KV 上限的 kopt（MoE 调优表的行 + absorb）吞吐 56,595 与原臂持平、交互 +6~10%；KV 上限才是上午 −5.7% 的主因，不用。
- 前处理（K2）现成开关只有 FlyDSL 排序有用（每层 −2 µs），EP8 布局不值得做；两者都被 MegaMoeTP 覆盖。

**更早的状态（16:05 UTC）**
- TP4 同节点 A/B（n02-21，C104）：kopt-tp4 吞吐 49,594 对 49,493（持平），交互 +6%，**接受长度 2.990 对 2.989（K5 没有接受长度代价）**；
  吞吐不涨是因为这个点由 prefill 限制（第十二节末尾）。
- 两节点 PP8：作业 35372 被抢占后换到 n01-29 + n03-33，decode（n03-33）起栈时 scheduler 段错误，连 PP8 原臂也崩，与我们的改动无关；
  在同一分配上重试的 64b（原臂）16:28 又崩了；按用户决定取消 35372，排除 n03-33 重新提交为 **35398**（排队中，64c → 64d）。
- EP8 MoE kernel 扫描（32 组）没有找到比 163.3 µs 更好的组合，EP8 布局不做；后续计划与待决项见 KERNEL-PLAN 第七节。

## 一、结论（09:55 UTC 更新）

找到三处 **只改配置、不写新 kernel** 就能拿到的收益，合计 TP8 每步约 **−3.7 ms**（步长约 46 ms 的 8%），TP4 约 −0.95 ms：

- 第 2 条（absorb BMM 配置）**已在 TP4 的 decode graph 里验证**：absorb BMM 每步 3.09 → 2.14 ms（−0.95 ms），**正确性门禁通过**（196/200、
  16k 64/64、接受长度 3.60），可以进生产配置（第八、九节）。
- 第 1、3 条只影响 TP8，需要一次两节点 PP8 运行验证；门禁 / trace / 计分三个条目已写好放在 `parked/`，等你同意（第十二节）。
- 其他结论：sparse MLA 的"无符号 32 位偏移"修法无效，近期只能用 KV 上限（第九节）；KF MoE 在真实权重上正确（第九节）；AKLR 现成的 o_proj /
  fused_qkv_a kernel 对我们的生产路径没有收益，后者越过 CI 的 Point 还会算错（第十节）；生产调优表缺 decode 的 q_b_proj、fused_qkv_a、router 等行，
  补调需要一次专门的长时间调参（第十节）。

下表是三条的单算子结果（09:55 前写的部分保持原样，以下各节按时间顺序）：

| # | 改动 | 原因 | 单算子实测 | 每步估算（TP8 C208） | 影响 TP4 |
|---|---|---|---|---|---|
| 1 | TP8 MoE：M 576–832 改用 aiter 调优表 token=512 那一行（block_m 32、atomic gemm2） | aiter 按 `nextPow2(M)` 查表，M 640–768 落到 token=1024 行（block_m 64 + 单独 reduction），这一区间并不是最优 | M 704：191.8 → 168.7 µs（−12%）；M 576–832 −6% ~ −14% | **约 −1.7 ms** | 否（TP4 M ≤ 448） |
| 2 | MLA absorb BMM 两个 Triton kernel 的 decode 配置（M ≤ 256 各档） | aiter 现有的 GLM 形状配置在 65–256 行都用 64×256 tile；我们每 rank 64 个头，W_vc 只有 128 个 workgroup，256 个 CU 用不满 | 88 行：W_vc 16.0 → 9.1 µs（−44%），W_kc+RoPE+写 cache 18.7 → 14.9 µs（−21%） | **约 −0.85 ms** | 是，约 −0.85 ms |
| 3 | TP8 decode 的 KV 池每层压到 2³¹ 字节以下（≤ 3,728,256 slot，即 `--max-total-tokens` 约 3.70M） | KF sparse MLA 在 `页数 × 576 > 2³¹−1` 时改用 64 位页偏移；TP8 每层 4,094,720 slot = 2.36 GB 正好越过 | 88 行：61.7 → 46.4 µs（−25%）；边界上差 1 个 slot 就差 33% | **约 −1.2 ms**，代价是 KV 容量 −9%（仍比 TP4 多 20%） | 否（TP4 池 1.78 GB） |

另外两条诊断：
- **交接文档里"TP8 sparse MLA 慢 1.7 ms"的原因已经查清**：上面第 3 条（约占 3/4），加上 KF 的 split 规则 `splits = floor(256 / 行数)` 在 86 行处从 3 降到 2
  （84 → 88 行 +23%）。强制 3 / 4 / 6 / 8 个 split 都不比 2 快，这个台阶要改 kernel 才能去掉。
- **TP8 MoE 剩下的空间主要在 gemm2**：换行后 M 704 时 gemm1 76 µs（5.6 TB/s，已接近 KF 在 TP4 的水平），gemm2 68 µs 只读 215 MB（3.2 TB/s）——
  它受 9 路 bf16 atomic 写回限制，不是读权重；前处理（两段排序 + 激活量化 + scale 排序）约 22 µs。这两块是下一步写 kernel 的目标。

## 二、环境

- 节点：n10-29（作业 35341，到 10:31）。为免和另一个 agent 的队列冲突，在 holder1 队列放了占位条目 `50-kopt-kernel-lease.sh`
  （07:45 起运行；`/apps/liyingli/holder1/kopt.lease` 存在就一直占着，删掉或 2.6 h 到期即结束，并删除 `llying-kopt-*` 容器）。
- 容器 `llying-kopt-bench`：生产镜像 `infera-sglang:v0522-fdbafb60`（aiter 2bb549102、FlyDSL 0.3.4.1、torch 2.9.1 / ROCm 7.2），按 decode worker 的方式挂载
  KF overlay 和两张调优 GEMM 表。**镜像里是 Python 3.10，AKLR 要求 ≥ 3.11，不能直接装进生产镜像**（KERNEL-PLAN 4.4 第 6 条的待确认项，结论是否）。
- 脚本（`results/opt/kernel-20261109/bench/`，原件在 `/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/`）：

| 脚本 | 内容 |
|---|---|
| `moe_bench.py` | aiter `fused_moe` 按 SGLang 的参数调用（MXFP4、16×16 shuffle、e8m0_shuffle、per_1x32、gate separated），一个 CUDA graph 里依次跑 4 组权重（像相邻几层，不命中缓存），每个 arm 强制查表的 token 行；同时给出逐 kernel 耗时和与第一个 arm 的 SQNR |
| `absorb_bench.py` | 两个 absorb BMM 按 `forward_mla_rocm.py` 的调用方式，48 组权重轮换，扫 36 个 Triton 配置，只保留与默认配置 SQNR ≥ 40 dB 的 |
| `sparse_bench.py` / `sparse_probe.py` / `sparse_2g.py` | KF `triton_sparse_mla_fwd`（生产 verify 用的就是它），按行数、KV 池大小、索引分布、split 数、2³¹ 边界扫 |
| `kc_2g.py` | W_kc 融合 kernel 在 2³¹ 边界两侧（结论：不受影响） |
| `kf_eager_check.py` | KF MoE 与 aiter 的 eager 对比（见第四节的未解决项） |

- **测量方法的校准**：ROCm 默认的 graph packet capture 下，profiler 每次调用只看到 7 个 kernel 里的 5.2 个，单个 kernel 的时间也不对（AKLR 知识库
  `pitfalls/profiler_misses_kernels` 说的就是这个）；`DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` 时逐 kernel 时间与生产 trace 一致（aiter MoE M 704：gemm1 93.8 /
  gemm2 59.3 / reduction 13 µs，trace 91.4 / 61.1 / 13.4）。两种模式下 event 计时的总时间相同（191.2 / 190.6 µs），所以下面的总时间都可比。

## 三、TP8 MoE（inter 256，257 专家，top-9）

### 3.1 换调优表的行（`out/moe_i256_keys.json`）

每次调用 µs（event 计时，graph 回放，4 组权重）：

| M | auto（aiter 自己的 nextPow2） | k64 | k128 | k256 | k512 | k1024 | 最好 | 比 auto |
|---|---|---|---|---|---|---|---|---|
| 384 | 141.9 | 151.1 | 141.6 | 140.9 | **140.3** | 171.2 | k512 | −1% |
| 448 | 144.4 | 161.9 | 155.3 | 144.5 | **143.9** | 174.8 | k512 | 0 |
| 512 | **149.8** | 192.9 | 173.5 | 150.7 | 149.8 | 178.6 | auto（即 k512） | 0 |
| 576 | 181.7 | 203.2 | 184.7 | 156.7 | **156.0** | 181.4 | k512 | **−14%** |
| 640 | 186.4 | 218.7 | 199.2 | **161.8** | 162.7 | 185.3 | k256 | **−13%** |
| 704 | 191.8 | 247.5 | 221.2 | 170.4 | **168.7** | 190.3 | k512 | **−12%** |
| 768 | 195.8 | 251.2 | 222.3 | 176.4 | **175.2** | 193.7 | k512 | **−11%** |
| 832 | 198.2 | 261.4 | 234.3 | 188.6 | **186.7** | 199.3 | k512 | −6% |
| 896 | 202.1 | 270.5 | 239.4 | **200.8** | 201.8 | 201.5 | k256 | −1% |
| 960 | **204.7** | 293.6 | 252.7 | 211.2 | 211.2 | 205.8 | auto | 0 |
| 1024 | **206.2** | 307.2 | 258.3 | 230.1 | 227.0 | 207.2 | auto | 0 |

- 各 arm 之间 SQNR 48–49 dB（不同 tile / atomic 顺序的正常差异）。
- C208 的 TP8 trace 里 graph bs 20–24（M 640–768），正好在收益最大的区间：每层 −20 ~ −25 µs，75 层约 −1.7 ms/步。
- 我的单算子结果与 trace 对得上：trace 里 fused_moe 部分（去掉 router GEMM、top-k、追加 shared）是 189.6 µs，单算子 auto 在 M 704 是 191.8 µs。
- **交付方式**（选一）：
  - 内部验证：挂一份改过的 `glm5_fp4_tuned_fmoe.csv`（把 (6144, 256, 257, token=1024) 那一行换成 token=512 行的 kernel），覆盖镜像里的同名文件，
    与 kit 现在挂调优 GEMM 表的方式相同。**不能**另加一个同 key 的 CSV：aiter 合并时发现重复 key 会改写源文件并抛异常（`jit/core.py` `update_config_files`）。
    代价：M 960–1024（graph bs 30–32）慢 3–13%，C208 下最忙 rank 没到过这么大。文件已做好：`results/opt/kernel-20261109/configs/glm5_fp4_tuned_fmoe.csv`
    （只改第 60 行）。验证过 aiter 自己的查表会选中它：`auto` 在 M 704 / 768 为 170.6 / 179.2 µs（原 191.8 / 195.8），M 1024 为 232 µs（原 206）。
  - 上游：aiter 的 `get_padded_M` 在 32768 以下一律取 2 的幂，加一档 768（或按表里最近的行查）是更干净的做法，需要 aiter PR。

### 3.2 换行以后还剩什么（M 704，k512）

| kernel | µs | 读权重 | TB/s |
|---|---|---|---|
| gemm1（`mfma_moe1_…_t32x64x256`，SiLU + MXFP4 量化） | 76.4 | 429.5 MB | 5.6 |
| gemm2（`gemm2_a4w4_port_…_bm32_…_atomic`） | 67.8 | 214.7 MB | 3.2 |
| opus 排序两段 + 激活量化 + scale 排序 | 约 22 | | |

gemm2 的 K 只有 256（inter 256），每个输出 tile 的计算很短，时间花在把 704 × 9 行按 bf16x2 atomic 加回 704 × 6144 的输出上（约 2,000 万次 atomic）；
reduce 版本（先写 6,336 × 6144 的中间结果再归约）也差不多（59 + 13 µs）。下一步的方向：gemm2 在 CTA 内先把同一 token 的多个路由合并再写、FP8 中间结果
（AKLR 知识库里 33 专家 kernel 用的办法，要先确认精度）、或把前处理折进 stage 1（KF 的 HIP 融合排序目前受 `M × 9 ≤ 4096` 限制）。

### 3.3 KF 在 TP4 / TP8 形状上的对照（`out/moe_i512_kf.json`、`out/moe_i256_kf.json`）

- TP4（inter 512）：KF 比 aiter 快 4–10%（M 96–416），M 448 慢 5%；与生产 trace 一致（KF gemm1 124–158 µs）。
- TP8（inter 256）：KF 在 M 128–384 快 4–6%（只改了查表键和排序，gemm1 仍是 aiter 的），M 448 慢 4%。

## 四、MLA absorb BMM（`out/absorb_sweep.json`、`out/absorb_bucket_pick.txt`）

每档选一个在该档所有行数上都稳的配置（不是逐行最优），默认 → 选定，µs：

| 行数（per-rank 4 × graph bs） | W_kc + RoPE + cat + 写 KV（`FUSED-FP8-BMM-ROPE-CAT-AND-CACHE-MLA-N=512-K=192`） | W_vc（`BATCHED_GEMM-A8W8-…-N=256-K=512`） |
|---|---|---|
| 16 | 7.8 → 5.0（BM16 BN128 w8） | 5.5 → 5.0（BM16 BN64 w4） |
| 32 | 11.3 → 6.5（BM32 BN256 w4） | 5.8 → 5.2（BM16 BN64 w4） |
| 48 / 64 | 9.9 / 12.0 → 8.7 / 10.3（BM64 BN256 w4） | 14.4 / 10.4 → 5.9 / 5.9（BM16 BN128 w4） |
| 72 / 80 / 88 / 96 | 16.7 / 17.8 / 18.7 / 19.9 → 12.9 / 13.9 / 14.9 / 16.2（BM32 BN128 w8） | 15.9 / 16.0 / 16.0 / 16.1 → 8.3 / 8.7 / 9.1 / 7.1（BM32 BN128 w8） |
| 104 / 112 / 128 | 21.0 / 22.2 / 24.0 → 19.1 / 20.4 / 21.7 | 16.1 / 16.0 / 10.6 → 9.6 / 9.5 / 7.2 |
| 144 | 30.8 → 24.8（BM32 BN128 w8） | 16.2 → 11.0（BM16 BN256 w8 s1） |

- 选定配置与默认配置的输出 SQNR 都 ≥ 40 dB（FP8 量化后的 bf16 / fp8 输出）。
- 生产负载下（TP4 C104 / TP8 C208，每 rank 72–96 行）每层约 −11 µs，78 层约 −0.85 ms/步。
- 改好的配置文件：`results/opt/kernel-20261109/configs/` 下两个 JSON（只改了 M ≤ 256 各档，prefill 用的 M ≥ 512 档不动）。挂到镜像
  `/sgl-workspace/aiter/aiter/ops/triton/configs/gfx950/triton/gemm/<family>/` 下同名文件即可生效（SGLang 不传 config，走 aiter 的默认查找）。
- 为什么现有配置不对：aiter 的查找键是 (M, N, K)，不含 batch（头数）；GLM-5.2 在非 DP attention 的 TP8 下每 rank 只有 8 个头，DP attention 下是 64 个，
  同一组 (M, N, K) 的最优 tile 完全不同。aiter 支持 `B=` 专门化的配置文件，但这两个调用点没有传 B——上游的正确改法是传 B 并加 B=64 的文件。

## 五、sparse MLA（`out/sparse_rows.log`、`out/sparse_probe_*.log`、`out/sparse_2g.log`）

| 实验 | 结果 |
|---|---|
| 行数 × KV 池（3.09M = TP4，4.09M = TP8），按请求共享索引 | 3.09M：72 行 36.1、84 行 38.2、**88 行 47.0**、96 行 48.3 µs；4.09M：72 行 46.0、84 行 49.0、**88 行 60.7**、96 行 61.2 µs |
| 强制 split 数（4.09M 池，88–128 行） | 2 个最快或持平（88 行：s2 61.3、s3 81.8、s4 65.4、s6 71.6、s8 60.3 µs） |
| 4.09M 的池、索引只落在前 N 个 slot | N 从 0.5M 到 4.09M 都约 61 µs；同样 N 个 slot 单独成池只要 46–48 µs → 与访问范围无关，与张量大小有关 |
| 池大小卡 2³¹ 字节边界（88 行） | 3,728,270 slot（2,147,483,520 B）46.4 µs；3,728,271 slot（2,147,484,096 B）**61.9 µs** |

- 原因：KF `kf_sparse_mla.py` 的 `_page_offsets_fit_i32`：`页数 × kv_dim ≤ 2³¹−1` 时用 32 位页偏移，否则走 64 位（第 1549、1602 行）。
- 修法：
  - 配置（立即可用）：TP8 decode 加 `--max-total-tokens 3700000`（每层池约 3.70M slot、2.13 GB）。2P1D / PP8 下 decode KV 不是瓶颈（预分配等待 0.2 / 4.0），
    少 9% 的容量预计没有影响，需要在计分里确认。
  - kernel（上游）：单纯把页偏移改成无符号 32 位**无效**（第九节实测）；要换寻址方式（拆池或按页组换基址），交给 yihou / #47278 评估。
- split 台阶在行数 86（3 → 2 个 split）：K3（每请求 4 行）是 graph bs 22 起，K5（每请求 6 行）是 graph bs 15 起；选 K 和 DP 均衡时可以参考。

## 六、未解决

- **KF MoE 在 inter 512 上与 aiter 的 SQNR 只有 4.1 dB（我的随机权重）**：只在 KF 自己的 gemm1 启动器（`GLM52_MOE_G1` = 1 / 2 / 3）时出现，用 aiter 的启动器
  （`GLM52_MOE_G1=0`）是 49.6 dB；inter 256（KF 专用 gemm1 不启用）是 49–54 dB。生产日志里 KF 自检是 52 dB 或完全一致，生产的端到端门禁（200 题、
  lm-eval 0.97、接受长度 3.62）也都通过，所以更可能是我的输入不满足 KF 某个假设（例如 scale 的取值范围）。查清楚要用真实权重做一次对比，留到之后；
  它不影响 TP8 的工作，也不影响上面的计时（KF 的计时与生产 trace 一致）。

## 七、08:10 之后

- **aiter 官方 MoE 调参器**（`gemm_moe_tune.py --mxfp4-flydsl --mxfp4-search-mode prune`，M 640 / 704 / 768，GPU 0–2，跑了 4 分钟、1,291 个组合后停掉）：
  最好的组合（`results/opt/kernel-20261109/bench/tune/tune_top.txt`）是 block_m 32 + `t32x128x{128,256}_atomic_persist`，用它自己的计时 169–176 µs，与第 1 条选的 token=512 行只差
  persist / XCD 交错这些小变化。**现有 kernel 空间里没有更大的收益**，再往下要写 kernel（gemm2 写回、前处理融合）。
- **graph 内验证（TP4，第 2 条）已开始**：08:12 起 holder1 跑 `51-kopt-absorb-trace-c104`（arm `arm-C-prod-ptpcfix-nohip-kopt-absorb-trace.env`：
  当前 4+4 的 C 臂 + 两个 absorb 配置 + `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`），AgentX C104 1500 s、600 s 后采两次 40 步。占位条目 50 已结束，
  `llying-kopt-bench` 容器已删。已确认 decode 容器里的两个配置文件 md5 与我们的一致。
- 对照用的旧 trace 按"本步最忙 rank 的 bs"（= graph 的 batch）分组（`ana/kbybs.py`，`ana/kbybs_tp*_baseline.txt`），每次调用 µs：

| 配置 | 最忙 bs | W_kc + RoPE | W_vc | sparse MLA | MoE gemm1 | MoE gemm2 |
|---|---|---|---|---|---|---|
| TP4 C104 | 17 / 18 / 19 | 20.9 / 20.9 / 22.2 | 16.6 / 16.6 / 16.4 | 30.8 / 30.9 / 31.0 | 133 / 133 / 138 | 69 / 69 / 71 |
| TP4 C104 | 22 / 24 / 26 | 23.7 / 25.0 / 26.4 | 16.6 / 16.8 / 16.8 | 41.5 / 41.6 / 41.8 | 142 / 149 / 151 | 73 / 77 / 79 |
| TP8 C208 | 20 / 21 / 22 / 23 | 23.0 / 24.1 / 24.2 / 25.6 | 16.5 / 16.7 / 16.7 / 16.8 | 45.1 / 57.3 / 57.5 / 57.6 | 91 / 92 / 92 / 93 | 59 / 61 / 61 / 63 |

  sparse MLA 的两个台阶都在 graph 内看得到：TP4 在 bs 19 → 22（3 → 2 个 split），TP8 在 bs 20 → 21（graph bs 20 → 22）且整体高一截（2 GiB 慢路径）。
  W_kc + RoPE 在 graph 内比单算子慢 4–6 µs，这次 trace 要看相对收益能否保持。
- **TP8 验证用的臂已写好，未入队**：`pkg/config/arm-C-1p1d-pp8-kopt.env`（PP8 臂 + 第 1–3 条：挂改过的 fmoe CSV 和两个 absorb 配置、decode 加
  `--max-total-tokens 3680000`）和 `arm-C-1p1d-pp8-kopt-trace.env`（再加 packet capture 关闭）。上限取 3.68M 而不是 3.70M：2P1D 日志里 KV 缓冲区比
  token 池多约 2.5 万个 slot（4,094,720 对 4,069,696），3.70M 离 2³¹ 只剩约 3 千个 slot。起栈后要看 KF 日志 `sparse_mla ... kv=(N, 1, 576)`，N 须 ≤ 3,728,270。

## 八、graph 内验证结果（TP4，第 2 条，08:45 那次 profile，`ana/kbybs_tp4_kopt_084511.txt`）

同一 graph batch 下，与旧 trace 对比（每次调用 µs）：

| 最忙 bs | W_kc + RoPE：旧 → 新 | W_vc：旧 → 新 |
|---|---|---|
| 25 | 26.1 → 22.4 | 16.8 → 10.1 |
| 26 | 26.4 → 22.5 | 16.8 → 10.2 |
| 27 / 28 / 29（旧 trace 没有这么大的 bs） | — → 23.6 / 23.5 / 24.6 | — → 10.0 / 9.9 / 10.4 |

- 每层约 −10.4 µs，78 层约 **−0.81 ms/步**，与单算子估计（−0.85）一致。**第 2 条在 graph 内成立。**
- 08:50 那次 profile 的负载低一些（最忙 bs 18–22），正好和旧 trace 重叠（`ana/kbybs_tp4_kopt.txt`）：bs 18 / 19 / 22 时 W_kc + RoPE
  20.9 / 22.2 / 23.7 → 16.7 / 17.8 / 19.0 µs，W_vc 16.6 / 16.4 / 16.6 → 8.3 / 8.7 / 9.3 µs，每层约 −12 µs。整步分类（`kcat.py`）：absorb BMM
  3.09 → **2.14 ms/步（−0.95 ms）**，其余各类与旧 trace 持平。
- 同一 trace 的附带发现：最忙 bs 27–28（M 432–448，仍在 KF 门控内）时 KF 的 MoE gemm1 163 µs，bs 29（M 464，超出门控走 aiter）只要 149 µs；
  与单算子一致（M 448：KF 256 / aiter 244 µs）。**TP4 的 KF MoE 门控上限 455 偏高，应降到约 416**（每层约 −10 µs，只在这么高的负载出现）。
- 队列（holder1，作业 35341 到 10:31）：51 trace 在收尾（AgentX 约 09:00 结束）；52 自动跑两项 GPU 检查（KF sparse MLA 无符号 32 位页偏移版本、
  KF MoE 真实权重对比，结果在 `bench/out/sparse_u32.log`、`kf_realw_tp{4,8}.log`，约 10 分钟）；53 是第 2 条的正确性门禁（4+4，200 题 + 16k × 64 +
  真实接受长度，约 45 分钟）。

## 九、两项 GPU 检查（条目 52，09:05）

- **sparse MLA 改用无符号 32 位页偏移没有用**（`bench/out/sparse_u32.log`）：4.09M 的池 88 行 production 59.4 / u32 59.7 µs，3.09M 的池两者相同，
  输出逐位一致。所以慢的不是 64 位乘加本身，而是池超过 2 GiB 后 kernel 用不上 32 位的 buffer 访存（推断，未看汇编）。kernel 级修法要换寻址方式
  （例如把每层 KV 池拆成两个 < 2 GiB 的张量，或按页组换基址），工作量比预想大；**近期就用 `--max-total-tokens` 上限**。第五节"无符号 32 位"的修法作废。
- **KF MoE 在真实权重上是对的**（`bench/out/kf_realw_tp4.log`，第 3 层、TP4 rank 0 的切片、真实 router top-k）：KF 对 aiter 40.2–47.4 dB，aiter 自己
  两次之间 54.9–59.4 dB。第六节的 4.1 dB 是随机权重造成的，不是生产问题（原因未查，KF 的专用 gemm1 对随机 scale / 权重分布有某个假设，可以告诉 yihou）。
  TP8 切片（inter 256）上 aiter 自己两次之间 55 dB（M 384，atomic 版）/ 完全一致（M 704，reduce 版）。
- **条目 53（第 2 条的正确性门禁）通过**（09:05–09:13，单节点 4+4，`arm-C-prod-ptpcfix-nohip-kopt-absorb-1109T0904`）：smoke 6/6；200 题 196/200
  （0.980），bad_start 0、garbled 0、撞上限 2；16k × 64 捞针 32/32、长前缀 32/32；真实接受长度 3.60（364 行）。对照：同一 C 臂 1,319 题 0.970、
  接受长度 3.62；PP8 / 2P1D 的 200 题门禁 197 / 196。**第 2 条可以进生产配置。**

## 十、AKLR 现成 kernel 对我们生产路径的实测（09:15，`bench/aklr_gemm_bench.py`，`bench/out/aklr_{oproj,qkva}.log`）

AKLR 的 kernel 源码就嵌在 schema JSON 里，FlyDSL / Triton 的可以不装 aklr 包直接加载（我们的镜像 Python 3.10 装不了 aklr，但 FlyDSL 0.3.4.1
与 AKLR CI 相同）。基线用生产路径：aiter `tgemm.mm` + 生产挂的两张调优表（SGLang 在 ROCm 上非量化线性层就是这样调的），graph 回放、轮换权重。

| kernel | 我们的行数 | 生产 / AKLR（>1 表示 AKLR 快） | 说明 |
|---|---|---|---|
| o_proj `gemm_n6144_k16384`（FlyDSL `e8a254fd`，AKLR CI 对 aiter 1.11–1.41x） | 16–128 | 0.97–1.08；72–128 行 0.97–1.04 | **持平**：生产调优表对这个形状逐 M 调过，AKLR 对"原版 aiter"的优势已经被吃掉 |
| fused_qkv_a `gemm_n2624_k6144`（Triton GEAK `b88c8d5f`，CI 只测 m 2 / 4 / 8 / 64） | 16–64 | 0.91–1.00 | 持平或略慢 |
| 同上 | 72–128 | — | **结果错误**（相对误差约 1.0）：它只在 CI 的 Point 上验证过，越过 64 行就算错。AKLR 的 dispatch 只服务精确 Point 正是为此；直接拿来用必须自己按我们的 M 重新验证 |

结论：**这两个 AKLR 现成 kernel 对我们没有收益**，印证了 KERNEL-PLAN 4.4 第 1、2 条（基线不同、Point 精确匹配）。同时发现生产调优表本身的缺口：
`yihou_glm52_dpa_bf16_tuned_gemm.csv` 里 fused_qkv_a（2624×6144）、indexer wq_b（4096×2048）、weights_proj（32×6144）只有 M 264–384 的行，
q_b_proj（16384×2048）和 router（256×6144）**根本没有行**——decode 的这些 GEMM 都在用默认启发式（fused_qkv_a 在 72–128 行只有 2.0 TB/s）。
09:20 起用 aiter 的 a16w16 调参器（`csrc/gemm_a16w16/gemm_tuner.py`，8 卡，默认不含 hipBLASLt）补调 decode 的 M（每 rank 32–128；router 用
all-gather 后的 TP4 256–416、TP8 576–832，共 63 个形状）：**30 分钟上限内连第一个形状（fused_qkv_a 的 13 个 M）都没调完**，没有产出——候选很多，
大量 JIT 编译。09:48 又跑了一个只有 3 个形状的窄版本（4 分钟跑完，`bench/tune/tuned_bf16_narrow.csv`）：fused_qkv_a M 88 / 96 最好 14.9 / 15.4 µs
（FlyDSL hgemm split-K 6 / 3，对生产默认 15.5 / 16.0 µs 只快 3–4%），router M 704 最好 12.6 µs（生产默认 12.0 µs，没有收益）。**这几个窄 GEMM 在 aiter
现有 kernel 里调不出明显收益**，瓶颈是启动 + split-K 归约，不是带宽；要么写新 kernel（例如 split-K 用 atomic 归约、或与前后 kernel 融合），要么走 dense FP8
（权重字节减半）。这与 AKLR 知识库 `bf16_glm52_wide_n` 的结论一致（AKLR 自己的 kernel 在这些形状上也只在 M ≤ 64 赢）。

顺便对比了 aiter 默认选择和直接用 `torch.mm`（hipBLASLt 默认启发式，`bench/out/gemm_default_vs_torch.log`，单位是 µs；日志里 TB/s 那一列实为 GB/s，
打印时单位写错了）：fused_qkv_a、indexer wq_b、router 上 aiter 默认更快（torch 慢 8–36%），q_b_proj 两者相当（±12%）。**换成 torch 没有捷径**，
这几个 GEMM 要么正经调参，要么写 kernel。当前每步：fused_qkv_a 约 1.15 ms（2.0–2.2 TB/s）、q_b_proj 1.47 ms（3.6 TB/s）、router TP8 0.95 ms，
按 4–5 TB/s 估算共约 1–1.4 ms/步的空间。

## 十一、TP8 MoE 前处理融合的难点（给后续 kernel 工作）

KF 的 HIP 前处理（排序 + 激活量化 + 输出清零一次启动）放宽到 M ≤ 1024 不难（`BM`、每线程对数都是编译期常数）。难的是 TP8 在 M ≈ 700 时要用 block_m 32
（block_m 16 的行每个专家要读两遍权重，第三节 k64 / k128 慢 25–45%），而 aiter 的 block_m 32 gemm1 要的是**按排序后顺序排好的激活 scale**
（`mxfp4_moe_sort_kernel` 的输出），KF 前处理产出的是按 token 顺序的。所以真正的修法是 KF 预量化版 gemm1 body（`_g1_body_preq`，现在只有 BM16、
inter 512）做一个 BM32、inter 256 的版本，再配上放宽后的前处理：每层约 −15 µs（TP8 约 −1.1 ms/步），属于多日的 kernel 工作，适合交给 yihou 或用
AKLR forge（以生产路径为基线）。

## 十二、用户决定（10:10）与后续安排

1. **跑两节点 PP8 验证**：60 / 61 / 62 三个条目已移入两节点 holder 的 `queue/`，10:11 提交两节点作业 **35372**（6 h，排队中，原因 Resources）。
   我挂了后台监视：作业开始、门禁、trace、计分各出结果时自动分析并记录到本文件。
2. **absorb 配置暂不加进单节点 4+4 的生产配置**（`arm-C-prod-ptpcfix-nohip-kopt-absorb.env` 保留，只作实验臂）。
3. **kernel 工作全部由我们做**（yihou 不再负责 GLM-5.2）。原先计划里写"交给 yihou"的各项（KF MoE 的 TP8 移植、indexer fused2 的 K3 版本、sparse MLA
   的 > 2 GiB 寻址、TP4 KF MoE 门控）都改由我们负责；KF overlay 的代码由我们维护。
4. 节点：n10-29（35341）10:31 到期；用户同意用空闲的 n02-33（35370，到约 15:12）做 kernel 开发，走 holder1 的占位条目。

### kernel 待办（按 TP8 每步收益排序；"可省"是估计）

| # | 项目 | 可省（TP8 / TP4，ms/步） | 状态 / 第一步 |
|---|---|---|---|
| K1 | TP8 MoE gemm2：M 576–832 时受 bf16 atomic 写回限制（约 2,000 万次 atomic、约 68 µs） | 1.5–2.5 / — | 先量化另一种布局：TP8 decode 的 MoE 改 EP8（每 rank 32 个完整专家 + shared，inter 2048），atomic 次数少 8 倍；AKLR 已有 GLM-5.2 33 专家（EP8 形状）的 kernel 可对照 |
| K2 | TP8 MoE 前处理 + BM32 预量化 gemm1（第十一节） | 1.1 / — | 依赖 K1 的布局结论 |
| K3 | sparse MLA：> 2 GiB 池的寻址；86 行的 split 台阶 | 1.2（不砍 KV）+ 0.5 / 0.4 | 近期用 KV 上限；kernel 改法待定 |
| K4 | 窄 dense GEMM（fused_qkv_a、q_b_proj、router、indexer） | 1.0–1.4 / 0.8–1.2 | aiter 现有 kernel 调不动，需要新 kernel 或 dense FP8 |
| K5 | indexer logits：fused2 的 K3（每请求 4 行）版本 | 0.6–0.8 / 0.2–0.4 | 改 KF `kf_indexer.py` 的 `_LPT_GROUP_ROWS` |
| K6 | indexer 前处理融合 + top-k | 1.0–1.3 / 1.0–1.3 | 先测 AKLR 的 dsa_indexer 插件（按我们的 M 验证正确性） |
| K7 | 通信：FP8 载荷 all-gather、RS + add + RMSNorm 融合 | 1.0–1.5 / 0.8–1.2 | 改 SGLang 层结构 |
| K8 | TP4 KF MoE 门控上限 455 → 416（第八节） | — / 约 0.3（高负载时） | 只改 overlay 一个常数 |

### K1 第一步：TP8 decode 的 MoE 改 EP8 布局值不值（10:25–10:40，n02-33，`bench/ep_moe_bench.py`、`bench/out/ep_moe*.log`）

每 rank 32 个完整专家（inter 2048）+ shared（复制），aiter `fused_moe` 带 expert_mask；同一批 token 和路由。aiter 表里没有 33 专家的调优行，
我用 inter 2048 那张表（257 专家 TP1）各 token 行的 kernel 组合作候选，放在 token=1024 行（M 704 查的就是它）逐个试（`tune/ep33/`）：

| M 704 | 每次调用 | gemm1 | gemm2 | 其他（排序 / 量化） |
|---|---|---|---|---|
| TP8 现状（aiter 默认，token=1024 行） | 185.8–191.8 µs | 93.8 | 59.3 + reduction 13 | 约 22 |
| TP8 + 第 1 条（token=512 行） | 166.5–168.7 µs | 76.2（5.6 TB/s） | 66.9（atomic） | 约 23 |
| EP8，aiter 默认启发式 | 180–183 µs（rank 0 / 3 / 7 一致） | | | |
| EP8，最好的候选（bm64 `t64x128x256` + `t64x256x128_atomic_nt`） | **163.3–164.5 µs** | 85.3（5.2 TB/s） | 54.6 | 约 24 |

- EP8 的 gemm2 少了约 12 µs（atomic 少 4–5 倍），但 gemm1 多了约 9 µs（shared expert 复制、每 rank 多读 8 倍；每个路由专家约 22 个 token 用 64 行的 tile）。
  净收益约 2 µs / 层，**不值得为此改并行布局**。
- TP8 gemm2 是 **atomic 写回限制**：k512 行下 gemm2 随 M 线性增长（M 512 / 704 / 768 / 1024：52.5 / 67.8 / 74.0 / 95.8 µs），斜率约 0.085 µs / token，
  截距接近 0——读 215 MB 权重本该至少 36 µs，却被写回完全掩盖。折算约 3,250 亿次 bf16x2 atomic / 秒。
- 所以 K1 要的是新的 gemm2 写回方式（例如不用 atomic、以紧凑格式写各路由的部分和再归约，或把 shared expert 先用普通写入初始化输出、只对 8 个路由专家做
  atomic），都是多日的 kernel 工作，排在 K5 / K8 这类能当天完成的之后。

### K5：indexer fused2 的 K3 版本（10:40–10:50，`bench/indexer_fused2_bench.py`、`bench/out/indexer_fused2.log`）

KF 的 fused2 把同一请求相邻两行配对处理、按上下文长度给"组"排序（最长的先算），组大小 `_LPT_GROUP_ROWS = 6`（K5 每请求 6 行）、最多 `_LPT_NG2 = 16` 组；
K3 下只有行数是 6 的倍数且 ≤ 96 才能用。改成组大小 4（K3 每请求 4 行）、最多 32 组。kernel 本身对配对的两行会检查是否共享 KV 页，所以分组只影响调度、
不影响正确性。按 K3 真实结构造输入（每请求 4 行共享一张页表，上下文 3 万–15 万），每次调用 µs：

| 行数 | 64 | 72 | 80 | 88 | 96 | 104 | 112 | 128 |
|---|---|---|---|---|---|---|---|---|
| 生产（组 6 / 16） | 63.2 | 64.0 | 80.2 | 79.4 | 72.8 | 92.1 | 105.5 | 116.4 |
| K3 版本（组 4 / 32） | **54.1** | **58.2** | **58.2** | **59.9** | **65.9** | **71.5** | **83.4** | **89.2** |
| 不融合的 preshuffle（参考） | 63.5 | 73.2 | 79.0 | 79.8 | 86.8 | 92.1 | 102.4 | 115.4 |

- 三者输出**逐位一致**（有效位置的最大相对误差为 0）。生产本来就走 fused2 的 72、96 行，组 4 也更快（组与请求对齐）。
- 每次调用省 6–27 µs，TP8 C208（80–96 行）约 −0.4~−0.6 ms/步，TP4 C104（72–104 行）约 −0.2~−0.4 ms/步。
- 实现：新建 `pkg/kit/overlay-kopt/`（生产 overlay 的副本，生产 overlay 一个字节没动），只把两个常数改成读环境变量
  `DSA_LPT_GROUP_ROWS` / `DSA_LPT_NG2`（默认仍是 6 / 16），同时把 KF MoE 门控上限改成读 `YIHOU_KF_MOE_M_HI`（默认 455，K8）。

### TP4 累计包的验证（10:28 起，n02-33）

臂 `arm-C-prod-ptpcfix-nohip-kopt-tp4(.env / -trace.env)`：absorb 配置（第 2 条）+ overlay-kopt（组 4 / 32 组）+ KF MoE 门控 416。

- **门禁（条目 56）没能完成，看起来是基础设施问题**：smoke 6/6 通过，200 题探针 0.975（garbled 0）但最后 3 个请求 `RemoteDisconnected`，
  之后 16k 长上下文 64 个请求全部 `Connection refused`。同一时刻（10:33:58）作业 35370 的 Slurm extern 步骤结束，之后 ssh 到 n02-33 被
  `pam_slurm_adopt` 拒绝（kit 里记过的已知现象）；经 ssh 启动的进程（如 router）随之被杀，服务端口消失。decode 日志因为 ssh 被拒没能收回，
  所以**还不能 100% 排除 decode 崩溃**；接下来的 trace（条目 57）走 `srun --overlap` 的 sshshim，正好在同样的 kernel 下复查，trace 结束后要再补一次门禁。
- trace（条目 57）10:35 开始，约 11:40 结束；结束后自动唤醒我分析。

### PP8 验证（作业 35372）

- 10:32 在 n04-33 + n10-29 开始，门禁（条目 60）10:33 开始。**开始时 kit 的标准步骤停掉了 n04-33 上别人的残留容器 `glmp_35367`**
  （只 stop、不删除；交接文档记过"停过别人容器的作业后来被 root 取消"，是否继续这样做还需要你决定）。
- **门禁（条目 60）通过**（10:33–10:50，`arm-C-1p1d-pp8-kopt-1109T1033`）：smoke 6/6；200 题 197/200（0.985），bad_start 0、garbled 0、撞上限 2；
  16k × 64 捞针 32/32、长前缀 30/32（runaway 2）；真实接受长度 3.59（607 行）。对照 PP8 原臂（11-05）：197/200、捞针 32/32、长前缀 31/32、
  接受长度 3.60。长前缀的 2 个 runaway 在单次 32 题里属于偶发（v2 单节点门禁也出现过 1 个），计分后再看是否需要多跑一次长上下文。
- **KV 上限生效**：decode 日志 `max_total_num_tokens=3680000`，KF 日志 `sparse_mla ... kv=(3680064, 1, 576)` = 2,119,716,864 B < 2³¹。
  decode 日志里的 8 处 Traceback 都是门禁结束时 SIGTERM 关机的正常输出。
- trace（条目 61，C208）10:49:50 开始，11:52 结束；**计分（条目 62）11:52:20 开始**（约 2 h）。
- **trace 结果**（`ana/kbybs_tp8_kopt_pp8.txt`、`ana/kdeep_tp8_kopt_pp8_*.txt`）：PP8 的负载比 2P1D 那次轻（最忙 bs 多在 12–22，2P1D 是 20–23），
  所以按相同 graph batch 比较每次调用 µs（旧 = 2P1D C208 trace，新 = PP8 + 第 1–3 条）：

  | 最忙 bs | W_kc + RoPE | W_vc | sparse MLA | MoE gemm1 + gemm2 + reduction |
  |---|---|---|---|---|
  | 20 | 23.0 → 18.0 | 16.5 → 9.0 | 45.1 → 30.4 | 163.5 → 153.1 |
  | 21 | 24.1 → 19.1 | 16.7 → 9.5 | 57.3 → 40.3 | 166.7 → 158.7 |
  | 22 | 24.2 → 19.1 | 16.7 → 9.5 | 57.5 → 40.3 | 166.6 → 158.6 |

  - 第 2 条（absorb）每层 −12 µs、第 3 条（KV 上限）每层 −15~−17 µs（sparse MLA −30%），与单算子一致，合计约 −2.3 ms/步。
  - **第 1 条（MoE 调优表的行）只兑现了约三分之一**：每层 −8 µs（单算子是 −23 µs），约 −0.6 ms/步。graph 内 block_m 32 的 gemm1 是 90 µs，
    单算子是 76 µs；最可能的原因是真实路由有偏斜（热门专家超过 32 个 token，要多跑一个 tile），而我的单算子用的是均匀路由。**以后 MoE 的单算子测试要用真实
    router 的输出**。
  - 合计：同负载下 verify graph 每步约 −2.9 ms（约 7%），预测是 −3.7 ms。

- **计分（条目 62，12:25–13:25 窗口，`score-kopt-pp8-score.txt`）：53,341 tok/s/GPU，比 PP8 原臂 56,593 低 5.7%**：

  | | PP8 原臂（11-06，n04-25 → n04-29） | PP8 + 第 1–3 条（n04-33 → n10-29） |
  |---|---|---|
  | 吞吐 / 输出 tok/s/GPU | 56,593 / 428.9 | **53,341** / 401.1 |
  | req/s | 5.036 | 4.769 |
  | 交互 p50 / p90 | 86.0 / 70.9 | **94.2 / 80.7**（+10% / +14%） |
  | ITL p50 / p90 | 11.6 / 14.1 ms | 10.6 / 12.4 ms |
  | TTFT p50 / p90 / 均值 | 7.50 / 14.47 / 10.82 s | 8.54 / 20.26 / **14.29** s |
  | decode 在跑中位 / 最忙中位 / 预分配等待 / 传输等待 | 58 / 12 / 4.0 / 85.2 | 49 / 10 / **11.9** / 90.3 |
  | decode 步长 @12 / @16 / @20 / @22（按最忙 rank 在跑数） | 33 / 37 / 43 / 46 ms | 32 / 36 / 40 / 43 ms |
  | prefill 满块中位 / 现算 tok/s | 324 ms / 20,453 | 339 ms / 19,264 |

  - decode 确实变快了（同负载步长 −1~−3 ms，交互 +10~14%），但 PP8 受 prefill 限制，吞吐没有因此上去。
  - 吞吐下降有两个可能原因，这一次分不开：（1）**KV 上限**：decode KV 少 10%，预分配等待 4.0 → 11.9（3 倍），请求在 decode 侧排队、TTFT 上升；
    （2）**节点差异**：这次 prefill 在 n04-33（原臂在 n04-25），prefill 满块 339 对 324 ms、现算 −5.8%，而 prefill 完全没改，PP8 的吞吐又正好由 prefill 决定。
  - 13:36 起在同一对节点上跑**只有第 1、2 条（不加 KV 上限）**的计分（条目 63，臂 `arm-C-1p1d-pp8-kopt-nocap.env`），约 15:20 出结果：如果回到 56K 左右，
    说明是 KV 上限的代价；如果仍是 53K 左右，说明是节点差异（那就还需要一次同节点的原臂计分）。

- **条目 63（不加 KV 上限的计分）没有结果：作业 35372 在 14:13:52 被抢占**（所有步骤收到 SIGTERM，guard 按设计删掉 agentx / router 容器，
  AgentX 退出码 137；14:20 作业变成 PENDING 等待重新开始）。已重新排队 `63b-kopt-pp8-nocap-score2`，并加 `64-pp8-base-score`（PP8 原臂）在**同一次分配、
  同一对节点**上跑，这样 62 / 63b / 64 的比较不再受节点差异影响（62 的 prefill 在 n04-33，满块比原臂慢 4.6%）。作业重新开始后两项共约 3.5 h。

- **重跑在新节点对上起栈失败（15:54）**：35372 14:39 在 n01-29（prefill）+ n03-33（decode）重新开始（新节点要先从 tar 装镜像）。63b（不加上限）和
  64（PP8 原臂，没有任何我们的改动）都在 decode 起栈时失败：图捕获完成后约 50 s，某个 DP rank 的 scheduler 段错误（63b 是 `scheduler_6`，64 是
  `scheduler_4`，exit −11），随后 tokenizer worker 反复报 "No HIP GPUs are available"（清理过程中的连带现象）。n03-33 的 8 张卡用 `gpu_check.sh` 单卡都正常。
  原臂也一样崩，所以**不是我们的改动，是 n03-33（或这对节点）的问题**；63b 那次 up 等了约 1 小时才判失败。已重排一次（64b 原臂、64c 不加上限），
  如果再在起栈时崩，就认定 n03-33 有问题、停止在这次分配上计分。
- **重试也崩了（16:28）**：64b（PP8 原臂）第三次在 n03-33 的 decode 起栈时段错误（又是 `scheduler_4`，exit −11），之后 /health 一直 503。
  三次都是原臂或配置级改动、三次都崩在 n03-33 的 decode，**认定 n03-33 有问题**（单卡小 kernel 检查正常，所以是负载下才出现的问题）。
  我手动停掉了卡住的 decode 容器（自己的），64b 16:30 结束、没有分数；64c 已移回两节点 holder 的 `parked/`，没有跑。作业 35372 仍占着
  n01-29 + n03-33（约 4 h），队列为空；是否取消后排除 n03-33 重新提交，等你决定。
- **16:33 按用户决定**：取消 35372，64c（不加上限）和 64d（原臂，`64d-pp8-base-score3.sh`）放回两节点队列，`sbatch --time=06:00:00
  --exclude=smci355-ccs-aus-n03-33 holder.sh` 提交为作业 **35398**（排队中，原因 Resources）。

### K1 续：EP8 MoE 的 kernel 扫描（16:36–16:40，n02-21，`bench/ep_sweep_one.py`、`bench/tune/ep33/sweep.sh`、`sweep_results.txt`）

上午的 EP8 只试了 6 组从别的 token 行借来的 gemm1 + gemm2 组合。这次按 block_m 16 / 32 / 64，把 GLM 调优表里同 tile 的 gemm1、gemm2 名字两两配对，
M 704、rank 0 掩码，每组一个进程，共 32 组。

- **按总时间，最好的是 174.8 µs**（bm32 `t32x128x256_w2_fp4` + `t32x128x256_atomic_bnt2`）和 175.5 µs（bm64 `t64x128x256_w2_fp4` +
  `t64x256x128_atomic_persist_nt`），**都不如上午的 163.3 µs**；`xcd4` 的 gemm1 在这个形状上慢到 300–490 µs。
- 所以现有 kernel 下 EP8 对 TP8（166.5 µs）只快约 3 µs，**EP8 布局不做**。TP8 MoE 的收益要靠写 kernel（K1 写回、K2 前处理）。
- **测量注意**：扫描时打印的分 kernel 时间不可靠。同一组合两次的 gemm1 是 72 µs 和 96 µs，总时间却一样；bm32 那组各 kernel 加起来只有 130 µs，总时间是 176 µs。
  设 `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` 后各 kernel 之和与总时间一致（bm32：gemm1 97.2、gemm2 50.9、其余 23.6，总 177.8 µs）。和 SGLang
  里看到的问题一样：graph 回放下 Kineto 的分 kernel 时间只有关掉 packet capture 才可信，单算子测试也一样，排名一律按总时间。上午 `ep_moe_profile.log`
  的分 kernel 数字是在 packet capture 打开时测的，只能作参考。

### TP4 累计包的 trace 与门禁（n02-21，11:45–12:57）

- **trace**（条目 57，`ana/kbybs_tp4_kopt_tp4.txt`）：K5 在 graph 内生效——最忙 bs 18–26 全部走 fused2（preshuffle 一次都没出现）。与只加 absorb 的那次
  trace 比（那次 bs 20–26 走 preshuffle），indexer logits 每次调用 bs 20 / 21 / 22 / 25 / 26：103.0 / 111.3 / 114.9 / 111.6 / 113.4 →
  82.7 / 84.2 / 87.5 / 89.8 / 91.1 µs（−20~−27 µs），21 层约 −0.45~−0.57 ms/步（bs 为 3 的倍数时原来就走 fused2，没有收益）。absorb 的收益与之前一致。
  K8（MoE 门控 416）这次最忙 bs 没超过 26（M ≤ 416），没有用到。
- **门禁重跑**（条目 57b，`arm-C-prod-ptpcfix-nohip-kopt-tp4-1109T1248`）：smoke 6/6；200 题 197/200（0.985），bad_start 0、garbled 0；16k 捞针 32/32，
  长前缀 30/32（**runaway 2**）；真实接受长度 3.62（生产 3.62）。
- **未结论：16k 长前缀 GSM8K 的 runaway**。两个 kopt 臂（PP8、TP4）各 2/32，都是复读到 8,192 token 上限（例："Let's just output 16.00." 重复），
  而 PP8 原臂 0/32、生产 C 臂 0/64、只加 absorb 的臂 0/32（合计 0/128）；4/64 对 0/128 的 Fisher 检验 p ≈ 0.01。门禁负载下 MoE 两项改动都没生效；
  实际生效的是 absorb（多数行数逐位一致，少数行数 55–59 dB）、KV 上限（只改寻址）、fused2 分组（单算子逐位一致）——都不该系统性改变数值，
  更可能是贪心解码轨迹被最后一位的差异改变后碰上了复读。正在用更大样本对比：同一节点先后跑生产 C 臂和 kopt-tp4 臂各 16000 × 512
  （256 题长前缀 GSM8K，条目 59a / 59b）。
- **结论（13:18）：复读与改动无关。** 同一节点先后跑，256 题长前缀 GSM8K：生产 C 臂 runaway **8/256**、正确 246；kopt-tp4 臂 runaway **6/256**、正确 247；
  kopt 的 6 个里有 5 个与生产臂是同一批题（idx 187 / 239 / 365 / 455 / 511），Fisher 双侧 p = 0.79。捞针 255/256 对 256/256。这是模型在部分 16k
  长前缀题上本来就有的约 3% 复读率，之前"基线 0/128"是样本太小（而且都是前 32 / 64 题）。
- 同一对运行的真实接受长度：生产 3.46、kopt 3.36（各约 930 行）。MoE 的 atomic 本来就让贪心轨迹逐次不同，这可能只是噪声，但接受长度直接影响吞吐，
  已排一次**反序**重复（59d kopt 先、59e 生产后）看两次之间的波动。
- **反序重复结果（13:39）**：四次运行（同一节点、交替）——生产 3.46 / 3.41（runaway 8 / 8，正确 246 / 243），kopt-tp4 3.36 / 3.36（runaway 6 / 5，
  正确 247 / 247）。正确性上 kopt 不差；**接受长度两次都低约 0.075（约 2%）**，生产自身两次相差 0.05，证据偏弱但方向一致。200 题探针负载下没有差别
  （3.62 对 3.62）。这个负载下生效的只有 absorb 配置和 fused2 分组（MoE 门控 416 没触发），所以接着单独跑：59g 只加 fused2、59h 只加 absorb、59i 再跑一次生产，
  各 16000 × 512。
- **归因结果（14:10，`ana/accept_weighted_longctx512.txt`）**：门禁打印的是 decode 日志各行的简单平均，长尾只剩一个复读请求的行与 60 个请求的行权重相同，
  所以另算了按 `#running-req` 加权的接受长度：

  | 臂 | 简单平均 | 按在跑请求加权 | GSM8K 正确 / runaway |
  |---|---|---|---|
  | 生产 ×3 | 3.46 / 3.41 / 3.41 | 3.447 / 3.425 / 3.404 | 246 / 243 / 250；8 / 8 / 5 |
  | kopt-tp4 ×2 | 3.36 / 3.36 | 3.375 / 3.376 | 247 / 247；6 / 5 |
  | 只加 fused2 分组 | 3.35 | **3.354** | 249；3 |
  | 只加 absorb | 3.37 | 3.399（在生产的范围内） | 246；5 |

  - 正确性各臂都不差。加权后差距仍在：kopt 类约 1.5% 低，**fused2 分组嫌疑最大**，absorb 基本在生产波动范围内。
  - fused2 在单算子上逐位一致；KF 的自检只覆盖预热时的一次调用（128 行、很短的上下文），各臂一样，说明不了长上下文。真实输入里可能有我的合成输入没覆盖到的情况
    （例如补齐行、跨页的兄弟行）；如果 logits 有细微差异，top-k 选到的 token 会变，接受长度略降而 GSM8K 准确率不变是说得通的。**未结论。**
  - 端到端吞吐 / 交互度已经包含了接受长度的影响，所以 14:13 起在 n02-21 上同节点先后跑生产 C 臂（59k）和 kopt-tp4（59l）的 C104 计分（该臂的 1500 s 窗口），
    约 16:00 出结果；在这之前 K5 不进任何配置。
- **TP4 同节点 A/B 计分结果（15:57，n02-21，C104，各自 1500 s 窗口，`score-score1500-base.txt` / `score-score1500-kopt.txt`，
  `ana/accept_weighted_tp4_ab_c104.txt`）**：生产 C 臂先跑（59k），kopt-tp4（absorb + K5 fused2 分组 + K8 门控 416）后跑（59l）。

  | | 生产 C 臂（59k） | kopt-tp4（59l） |
  |---|---|---|
  | 吞吐 / 输出 tok/s/GPU | 49,493 / 454.0 | 49,594 / 459.9（+0.2% / +1.3%） |
  | req/s，完成 / 丢弃 | 3.923，5,975 / 7 | 3.938，6,015 / 7 |
  | 交互 p50 / p90 | 65.3 / 58.8 | **69.4 / 62.6**（+6.3% / +6.5%） |
  | ITL p50 / p90 | 15.3 / 17.0 ms | 14.4 / 16.0 ms |
  | TTFT p50 / p90 / 均值 | 3.85 / 9.37 / 5.27 s | 4.05 / 9.74 / 5.87 s |
  | E2E p90 | 36.1 s | 35.9 s |
  | decode 步长 @12 / @16 / @20 / @24 / @26 | 35 / 41 / 45 / 48 / 50 ms | 36 / 40 / 43 / 46 / 48 ms |
  | 接受长度（简单 / 按在跑请求加权） | 2.989 / 2.989 | 2.990 / 2.990 |
  | prefill 满块中位 / 等待均值 / 有排队的时间片 | 319 ms / 3.17 / 70% | 315 ms / 4.73 / 77% |

  - **真实负载下接受长度完全没变**（2.989 对 2.990，各约 3,700–3,900 行）。长前缀 GSM8K 上 kopt 类低约 1.5% 的现象在 AgentX 负载下不出现，
    K5 对吞吐 / 交互没有接受长度的代价。
  - decode 同负载步长 −1~−2 ms（bs 16–26，约 −3~−4%），交互 +6%，与 trace 预测（absorb −0.95 ms/步 + fused2 −0.45~−0.57 ms/步）一致。
  - 吞吐不变：4+4 的 C104 在这个点由 prefill 限制（prefill 等待 3.17 → 4.73、有排队的时间片 70% → 77%，TTFT 均值 +0.6 s）。decode 快了，
    请求更早离开 decode，但 prefill 没有更快，所以吞吐不动、TTFT 略升。**这组改动在 TP4 上只换交互度，换不来吞吐**；要提吞吐，必须让 prefill 变快，
    或者提高并发，用交互度的余量换吞吐（例如 C112 / C120 看是否在交互约束内）。
  - 按 10:10 的决定，absorb 不进单节点 4+4 的生产配置；K5、K8 也一样只留在 `arm-C-prod-ptpcfix-nohip-kopt-tp4.env`，是否采用由你决定。

### TP4 门禁失败的原因：作业 35370 被抢占

sacct：35370 在 10:38:36 状态 PREEMPTED，10:33:58 起收到抢占通知（extern 步骤结束、ssh 被拒），guard 步骤按设计删掉了 router
（`guard.log`："signal: removing glm52_2p1d-router"），所以门禁的最后几个请求断连、长上下文全部被拒。**与 kernel 改动无关**。作业之后被 Slurm 重新排队，
11:39 在 **n02-21** 上重新开始，trace（条目 57）11:45 在 n02-21 重跑（约 12:50 结束）；已加 `57b-kopt-tp4-gate2.sh` 在 trace 之后补跑门禁，
占位条目 58 去掉了"只在 n02-33"的限制。

## 十三、下一步（10:10 之前的版本）

1. **TP8 验证运行（需要你同意，两节点约 4 h）**：三个条目已写好，放在两节点 holder 的 `parked/`（**未入队**）：
   `60-kopt-pp8-gate.sh`（门禁，约 50 min）→ `61-kopt-pp8-trace.sh`（C208 trace，约 70 min）→ `62-kopt-pp8-score.sh`（C208 计分，约 125 min，对照 56,593）。
   后两个只在门禁干净（smoke 6/6、garbled 0、bad_start ≤ 1、接受长度 ≥ 3.4）时才跑。用的臂是 `arm-C-1p1d-pp8-kopt(.env / -trace.env)`：第 1–3 条全加，
   decode `--max-total-tokens 3680000`。同意后 `mv` 到 `queue/` 并提交两节点作业即可。trace 那一步预期 verify graph 每步约 −3.7 ms；计分时要看
   预分配等待有没有因为 KV 少 10% 而变长。
2. 单节点 4+4（TP4）只用第 2 条：53 门禁通过后，可以排一次计分（约 1.5 h，对照 DP 均衡 K3 C104 的前沿点），但 −0.95 ms/步（约 2%）在单次计分 ±1% 的
   噪声边上，计分只能看方向。
3. TP4 的 KF MoE 门控上限 455 → 416（第八节），改动在 overlay 的 `MOE_M_LO, MOE_M_HI`，可以和第 2 条一起进下一次 4+4 运行。
4. 写 kernel：TP8 gemm2 写回（约 30 µs / 层）和 BM32 的 KF 预量化 gemm1 + 前处理（第十一节，约 15 µs / 层），需要和 yihou 分工。

## 十四、17:00 起：用户离开约 10 h，按 KERNEL-PLAN 第七节自主执行

### 用户 17:04 的决定

- **容器策略**：别人的容器不占显存就照常跑，占着显存就 stop（不删）；如果后来发现 stop 与 root 取消确有关联，改为"占着 GPU 就排除该节点、重新申请"。
  已改 `scripts/clear_foreign_containers.sh`：按 `/sys/class/kfd/kfd/proc/<pid>/vram_*` 汇总每个进程的显存，用 `/proc/<pid>/cgroup` 里的容器 ID
  对到容器，只 stop 占显存 ≥ 256 MiB 的别人的容器；每次判断（keep / stop、节点、作业号、容器、镜像、创建时间、显存）追加到
  `/apps/liyingli/holder/foreign_containers.log`。17:08 在 n02-21 上验证：自己的容器占 2 GiB 时读数正好 2147483648 字节、容器 ID 对得上。
  `opt_item5.sh` 调用时传入作业号。
- **AKLR forge 暂不用**：前处理融合和 gemm2 方案 A 自己写；全新 gemm2 kernel 的测试框架写好后，如果手写版离带宽上限还差很多，再请用户决定是否开 campaign
  （需要用户登录 Claude Code）。
- 节点：同时最多 1 个单节点 + 1 个两节点作业。

### 节点

- 两节点 **35398** 17:10 在 n01-29（prefill）+ n02-21（decode）开始（没有 n03-33），64c 的 decode 17:13:51 正常起来，没有段错误。
- 单节点 35370 17:09 到期；新单节点 **35400**（6 h，排除 n02-25、n03-33）排队中，`--start` 估计 23:10（集群所有 GPU 都已分配，只能等 35398 结束）。
- 所以在 35398 的队列里，64c（kopt 不加上限计分）和 64d（原臂计分）之间插了两个 GPU 测试条目（64c5、64c6，在 n01-29 上跑，约 40 min），
  64d 仍能在 35398 到期（23:10）前跑完。

### PP8 计分 64c：kopt 不加 KV 上限（17:10–18:53，n01-29 → n02-21，`score-kopt-pp8-nocap-score3.txt`，窗口 17:44–18:44）

| | PP8 原臂（11-06，n04-25 → n04-29） | 62：第 1–3 条（含 KV 上限，n04-33 → n10-29） | 64c：第 1、2 条（不加上限） |
|---|---|---|---|
| 吞吐 / 输出 tok/s/GPU | 56,593 / 428.9 | 53,341 / 401.1 | **56,595** / 427.7 |
| req/s，完成 / 丢弃 | 5.036 | 4.769 | 5.018，18,195 / 54 |
| 交互 p50 / p90 | 86.0 / 70.9 | 94.2 / 80.7 | **91.6 / 78.2**（+6.5% / +10%） |
| ITL p50 / p90 | 11.6 / 14.1 ms | 10.6 / 12.4 ms | 10.9 / 12.8 ms |
| TTFT p50 / p90 / 均值 | 7.50 / 14.47 / 10.82 s | 8.54 / 20.26 / 14.29 s | 7.51 / 14.53 / 11.43 s |
| decode 预分配等待 / 传输等待 | 4.0 / 85.2 | 11.9 / 90.3 | 3.3 / 89.7 |
| decode 步长 @12 / @16 / @20 | 33 / 37 / 43 ms | 32 / 36 / 40 ms | 32 / 36 / 42 ms |
| prefill 满块中位 / 现算 tok/s | 324 ms / 20,453 | 339 ms / 19,264 | 323 ms / 20,168 |

- 不加 KV 上限时**吞吐与原臂持平、交互 +6~10%、TTFT 基本不变**，预分配等待回到 3.3。62 的 −5.7% 主要来自 KV 上限（预分配等待 ×3、TTFT +3.5 s），
  另有 62 的 prefill 节点慢约 5% 的影响。KV 上限让 decode 步长再快约 2 ms（@20：42 → 40），但在 PP8 上用 KV 容量换来的这点 decode 速度不划算。
- 同节点原臂 64d 跑完后再最终确认（排在两个测试条目之后）。
- 结论（暂定）：PP8 上用第 1、2 条（MoE 调优表的行 + absorb 配置），**不加 KV 上限**；sparse MLA 的 > 2 GiB 寻址要靠改 kernel（K3），不能靠砍 KV。

### K2 第一步：生产的 TP8 MoE 前处理到底跑了什么（读 aiter 源码，`kopt-20261109/src/`）

- PP8 trace（`ana/kdeep_tp8_kopt_pp8_decode-113031.txt`）每层：Opus 多阶段排序 P0_v2 4.7 µs + P23 8.9 µs、逐 token 激活量化
  `dynamic_per_group_scaled_quant` 4.8 µs、scale 排序 `mxfp4_moe_sort` 4.5 µs，合计约 22.9 µs（另有 `_fused_append_shared_experts` 4.3 µs、router
  topk 5.4 µs）。
- 生产 decode 设了 `AITER_USE_FLYDSL_MOE_SORTING=1`，但 aiter 只在 `dispatch_policy == 0` 时才用 FlyDSL 排序，SGLang 传入的 policy 不是 0，
  所以 **FlyDSL 排序在生产上没生效**。
- 激活量化 + scale 排序：aiter 有一个单 kernel 的融合版本 `fused_dynamic_mx_quant_moe_sort_hip`，但只在 stage 1 的 M ≤ 8 × 256 / topk = 227 时用；
  TP8 的 M 576–832 走的是两个 kernel 的拆分路径。融合版每个路由各量化一次（9 倍重复读），拆分版每个 token 只读一次。
- 条目 **64c5**（`bench/prologue_bench.py`、`bench/run_prologue.sh`）测 7 个变体：生产；FlyDSL 排序；Opus policy 0 / 1；CK 排序；
  融合量化 + Opus；融合量化 + FlyDSL 排序。每个变体单独一个进程，总时间在默认 graph 设置下测，分 kernel 时间在 packet capture 关闭时测，
  输出与生产变体比 SNR。

### 新发现：aiter 的 MegaMoeTP（整段 TP MoE 层做成一个 kernel）

- 生产 TP8 每个 MoE 层的"通信 + MoE"约 250 µs：all-gather 38.4 µs、gemm1 88.5、gemm2 64.3、前处理约 23、追加 shared expert 4.3、reduce-scatter 32.2 µs；
  75 层合计约 18.8 ms，**占 verify graph 每步 37.6 ms 的一半**。all-gather 只搬约 7.9 MB 却要 38 µs，通信和计算没有重叠。
- aiter 的 `aiter/ops/flydsl/mega_moe_tp.py`（`MegaMoeTP`，gfx950）：expert 在 TP 组内复制、inter 维切分，一次持久化 kernel 完成
  all-gather（可选 FP8 传输）+ 前处理 + GEMM1 + 激活 + GEMM2 + reduce-scatter，可选融合尾部的残差加法 + RMSNorm（+ FP8 量化）。
  `comm_mode="ag_rs"` 正是 DP attention + TP MoE 的结构。aiter 的多卡测试 `op_tests/multigpu_tests/test_mega_moe_tp.py` 已内置 `glm5` 形状
  （hidden 6144、inter 2048、257 专家、top-9，与 GLM-5.2 融合 shared expert 后一致）。
- SGLang 里接入的是另一个 kernel（`MegaMoEV2`，EP + MORI 共享内存，`--moe-a2a-backend megamoe`），**MegaMoeTP 没有接进 SGLang**。
- 条目 **64c6**：8 卡跑 aiter 的测试，`--models glm5 --modes ag_rs -t 512 576 640 704 768 832 -r random hot`，看精度（与 torch 参考的 rel_l2）
  和 CUDA graph 下最慢 rank 的每次调用 µs（FP8 / bf16 传输）。对照的是上面约 250 µs。
- **K2 前处理变体结果（64c5，18:53–18:55，n01-29，`bench/out/prologue.log`）**，M 704 每次调用的 MoE 总时间：生产（Opus 多阶段）168.5 µs；
  **FlyDSL 排序（policy 0）166.1 µs**（排序 13.5 → 11.9 µs，576–832 各点 −0.5~−2.4 µs）；Opus auto / CK 168.4–168.8；Opus policy 1（单阶段）398.9
  （排序 246 µs）；融合量化 + scale 排序 178.7–182.4（融合 kernel 20.5 µs，比拆分的 9.4 µs 慢，因为每个路由各量化一次）。与生产输出的 SNR 都在
  50–54 dB（stage 2 atomic 顺序的正常差异）。**现成开关只有 FlyDSL 排序有用，每层约 −2 µs（约 −0.15 ms/步）**；前处理要省 10 µs 以上只能写新 kernel，
  而 MegaMoeTP（下面）会把前处理整个吞掉，所以 K2 的新 kernel 先不做。
- **MegaMoeTP 结果（64c6，18:53–18:55，n01-29 8 卡，`bench/out/megamoe_tp.log`）**：每次调用（整段 all-gather + MoE + reduce-scatter，CUDA graph，
  最慢 rank），随机路由：

  | 全局 token | 512 | 576 | 640 | 704 | 768 | 832 |
  |---|---|---|---|---|---|---|
  | FP8 传输，µs（rel_l2） | 197.4 | 202.1 | 203.2 | 203.8 | 206.0 | 208.9（0.0267） |
  | bf16 传输，µs（rel_l2） | 195.8 | 200.8 | 202.1 | 203.8 | 206.0 | 208.6（0.0031） |
  | 热点路由（FP8），µs | 119.1 | 129.8 | 136.9 | 149.4 | 157.7 | 181.2 |

  - 生产同规模每层约 250 µs（PP8 trace：all-gather 38.4 + MoE 约 175 + reduce-scatter 32.2），**MegaMoeTP 每层约省 45–50 µs，75 层约 −3.5 ms/步
    （verify 步长约 40 ms 的 8–9%）**，是目前单项最大的 kernel 收益。权重不会留在 MALL 里（每 rank 约 645 MB），测法与 `moe_bench` 可比。
  - **bf16 传输与 FP8 一样快，误差低 8.6 倍（rel_l2 0.0031 对 0.0267）**，接入时用 bf16 传输。
  - 功能用例全部通过：变长 m、多层共用实例、`out=`、CUDA graph、dynamic / LB 调度、masked ids、m=0、融合尾部（残差加法 + RMSNorm，rel_l2 0.0265）、
    host 校验、跨 rank 检查。
- **决定：MegaMoeTP 接入 SGLang 作为当前最高优先级**（K1、K2、K7 的 MoE 部分都被它覆盖）。接入办法（overlay 改 `DeepseekV2MoE` / 解码层前向，约 1–2 天）：router 改在本 rank 的 token 上算（还省掉 7/8 的 router 计算），
  跳过通信层的 MLP all-gather / reduce-scatter，改调 MegaMoeTP；"每个 rank 本地 token 数相同"的约束在 CUDA graph 下天然满足
  （graph batch 补齐到最忙 rank）。MTP 草稿模型的 MoE 不动。FP8 传输要另外过正确性门禁。

### MegaMoeTP 接入 SGLang（19:00–19:30 写好，待验证）

- **读源码确认的前提**：GLM-5.2 是 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`，MoE 层就是 `DeepseekV2MoE`；DP attention 下通信由解码层的
  `ffn_boundary`（`layers/layer_boundary/`，约 6000 行）负责：入口 `_reduce_update_read_dp_gather` 先做残差加法 + norm 再 `dp_gather`，出口做
  reduce-scatter（或延迟到下一层输入）。aiter MXFP4 路径的 topk 权重已含 `routed_scaling_factor`，shared expert 融合为第 257 个专家，MoE 之后
  `maybe_fuse_routed_scale_and_shared_add` 原样返回，MegaMoeTP 直接用 topk 输出即可。SGLang 加载后的权重用 `shuffle_weight(gate_up=True)` /
  `shuffle_scale`（非交错），按源码与 MegaMoeTP 测试用的 `shuffle_weight((16,16))` / `e8m0_shuffle` 布局相同（64e 会用真实权重实测确认）。
- **实现**：`pkg/kit/overlay-kopt/kopt_megatp/__init__.py`（`KOPT_MEGATP=1` 时由 overlay-kopt 的 `sitecustomize.py` 加载，生产 overlay 不变）。
  只在 CUDA graph 捕获模式、decode / target-verify、目标模型（非 MTP 草稿）的 MoE 层、本地 token ≤ 128 时生效；否则全部走原路径：
  - 解码层前向只加"预备"标志；FFN 入口照常做残差加法 + norm，但其中的 `dp_gather` 直通（返回本 rank 的行），之后检查行数等于本地 token 数，
    不等就报错（不会静默算错）；
  - MoE 前向：在本地 token 上算 router（`forward_gate`，split-K 路由）和 topk，所有层共用一个 MegaMoeTP 实例（每层 `set_weights`），
    `prepare` 只在 graph 捕获前的预热前向里调用；
  - FFN 出口换成"已完成求和"的出口：照 `FfnExit.finish` / `_complete_now` 的顺序只做残差写回和 `_record_output`，没有 reduce-scatter、
    不延迟求和。
- **测试臂**：`arm-C-1p1d-pp8-megatp.env`（= 64c 的 kopt-nocap + overlay-kopt + `KOPT_MEGATP=1 KOPT_MEGATP_COMM=bf16
  HSA_ENABLE_IPC_MODE_LEGACY=1`）和 `-trace.env`。
- **已排队（35398，64d 之后）**：64e 真实权重 A/B（第 3、40 层，`bench/megatp_realw_ab.py`：MegaMoeTP 对生产路径的 SNR + 用时）→
  64f 门禁（200 题 + 16k × 64）→ 64g C208 decode trace（只在门禁干净时跑）。计分要等下一次两节点分配。
- **35398 又被抢占（19:27–19:32）**：extern 步骤 19:27:14 结束（抢占预告），guard 按设计杀掉 AgentX（退出码 137），**64d（同节点原臂）没有分数**；
  19:32:06 作业因抢占取消，随后重新排队。这次没有停过别人的容器（日志只有一条 keep），是正常抢占。
- **64e 没跑成，是测试脚本的问题**：graph 捕获时 PyTorch 的 NCCL watchdog 线程在查询之前集合通信的事件，默认的全局捕获模式下报
  "operation not permitted when stream is capturing"。已改为 `capture_error_mode="thread_local"`。
- 单节点 **35400** 19:32 在 n01-29 开始（先结束了 35370 遗留的租约条目），A/B 改在单节点上跑（holder1 条目 60，19:34 开始）。
- 两节点队列现在是 64f 门禁 → 64g trace → **64h MegaMoeTP 计分**（门禁干净才跑）→ **64i kopt-nocap 计分**（同一分配上的对照）。

### MegaMoeTP 在真实权重上（holder1 条目 60 / 61b，19:34–19:39，n01-29，`bench/out/megatp_realw_ab.log`、`megatp_ref_check.log`）

真实 GLM-5.2 第 3、40 层，按 SGLang 的方式切 TP8、重排权重，真实 router（sigmoid + 偏置、top-8、×2.5 + shared）在每个 rank 自己的 token 上算。
生产路径用 RCCL 的 all-gather / reduce-scatter（比 SGLang 自己的通信 kernel 慢），所以下表单列"只有 MoE"：

| 全局 token | 576 | 640 | 704 | 768 | 832 |
|---|---|---|---|---|---|
| 第 3 层：生产只有 MoE，µs | 155.4 | 163.9 | 176.6 | 184.5 | 190.2 |
| 第 3 层：MegaMoeTP bf16（含通信），µs | 188.5 | 203.4 | 207.3 | 219.3 | 226.0 |
| 第 40 层：生产只有 MoE，µs | 155.0 | 161.0 | 172.5 | 181.9 | 192.8 |
| 第 40 层：MegaMoeTP bf16（含通信），µs | 184.4 | 187.5 | 192.9 | 209.0 | 220.8 |
| MegaMoeTP 与生产输出的 SNR（第 3 / 40 层） | 29.5 / 28.1 dB | 29.2 / 28.5 | 29.8 / 28.6 | 29.1 / 28.3 | 29.4 / 28.6 |
| 生产自身两次运行的 SNR（第 3 / 40 层） | 51.0 / 47.6 dB | 51.4 / 46.9 | 51.0 / 48.1 | 51.2 / 48.9 | 53.0 / 49.7 |

- **速度**：生产在 graph 内的通信是 all-gather 38.4 + reduce-scatter 32.2 ≈ 70 µs（PP8 trace），生产整段 ≈ 只有 MoE + 70 µs；
  MegaMoeTP 每层快 **34–50 µs**（704：第 3 层 247 → 207，第 40 层 243 → 193），75 层约 **−2.7 ~ −3.5 ms/步**。FP8 传输一样快，但与生产的 SNR
  再低约 1.5 dB，不用。
- **精度（61b）**：用反量化的 FP4 权重、bf16 激活、fp32 累加算参考（不量化激活），两条路径离参考**一样远**：第 3 层生产 14.9 / 15.0 / 14.9 dB、
  MegaMoeTP 14.9 / 15.0 / 14.9 dB（576 / 704 / 832）；第 40 层生产 13.1 / 13.1 / 13.2、MegaMoeTP 13.1 / 13.2 / 13.2 dB。
  13–15 dB 是输入和中间激活两次 MXFP4 量化本身的误差。**结论：权重布局正确（布局错了离参考会远得多），精度与生产等价**；两者之间 29 dB 的差别只是
  FP4 取整方式不同（KF 和 aiter 之间是 40–47 dB，它们的量化方式相同）。端到端还要看门禁。
- 19:41 起在单节点上跑接入后的独立服务冒烟测试（holder1 条目 62：生产 decode 参数去掉 PD、graph 最大 bs 64，先 `KOPT_MEGATP=1` 再 0，
  各跑 200 题 GSM8K 探针，全部同时发出）。
- **独立服务冒烟测试结果（19:41–20:03，`bench/out/megatp_srv/`）**：
  - 接入代码能跑：8 个 rank 都装上了补丁，graph 捕获时本地 token 4 到 128 的每一档都走了 MegaMoeTP（topk 9 列、fp32 权重、int32 ids），
    没有报错，watchdog 没有触发。
  - 200 题探针：**开 MegaMoeTP 197/200（0.985），garbled 0，撞上限 2**；关掉 187/200（0.935），garbled 3，撞上限 10。
  - 两轮的 MTP 接受长度都只有约 1.0–1.4（生产 PD 下 3.6），关掉 MegaMoeTP 也一样，所以是这种"不分离 PD 的独立配置"本身的问题
    （可能和漏掉的 `SGLANG_USE_ROCM700A=0` 或非 PD 下的 MTP 行为有关），**不是接入造成的**；这个配置因此只能用来证明接入代码能跑、
    输出正常，正确性和接受长度要看 PD 门禁。
- 两节点 35398 仍在等节点（部分节点 down / drained / 被预留），所以在单节点上加了：63 TP4 测速（aiter 测试 4 卡 + TP4 真实权重 A/B；
  TP4 生产每层约 213.5 µs：KF MoE 144.9 + 排序 / 填充 5.1 + 追加 shared 4.4 + all-gather 31.5 + reduce-scatter 27.6，`kdeep_tp4_kopt`）；
  64 单节点 4+4 PD 门禁（`arm-C-prod-ptpcfix-nohip-megatp.env`：生产 C 臂只加 MegaMoeTP），在真正的 PD 模式下验证接入的正确性和接受长度。
- **TP4 结果（63，20:04–20:06，`bench/out/megatp_tp4.log`）**：
  - TP4 生产每层实际约 **288.8 µs**（上面漏了 KF gemm2 75.3 µs：KF gemm1 144.9 + gemm2 75.3 + 排序 / 填充 5.1 + 追加 shared 4.4 +
    all-gather 31.5 + reduce-scatter 27.6）。
  - MegaMoeTP 在 TP4 真实权重上 238.5 / 245.8 / 252.7 µs（全局 256 / 320 / 384 token），插值到 336 约 247 µs，**每层快约 42 µs（−14%）**；
    离 fp32 参考两条路径都是 14.9–15.0 dB，精度等价。合成随机路由下是 284–287 µs（真实路由的活跃专家更集中），热点路由 97–131 µs。
  - 所以 MegaMoeTP 对单节点 4+4（TP4）和两节点 PP8（TP8）都有收益。
- **20:05 起**：35398 在 n04-25 + n04-33 恢复（第 1 次重启），20:05 开始 64f（TP8 PP8 MegaMoeTP 门禁）。单节点 35400 在 20:08:06 被抢占，
  64（TP4 门禁）刚开始就被打断，作业会重新排队。
- 64f 第一次起栈（20:05）：TP8 decode（n04-33）20:11:45 正常起来，MegaMoeTP 路径生效、无报错；但 **prefill（n04-25，没有任何改动）**
  20:17:46 子进程以 −9 退出（加载约 12 分钟后被 SIGKILL，像主机 OOM：每个 prefill rank 常驻约 265 GB，节点共享内存占 1960 GB），
  verify_suite 20:18 自动重新起栈。与 decode 侧的改动无关。
- **n04-25 也有问题（20:18–20:31）**：第二次起栈时 prefill 的 `scheduler_2` 段错误（exit −11，20:23:49），随后服务被清理（−9），64f 门禁失败，
  64g / 64h 因门禁不干净被跳过；64i（kopt-nocap，与 decode 改动无关的臂）在同一对节点上也起不来（20:30:55 没有分数）。三次失败、三个不同的臂，
  和下午 n03-33 一样是"起栈时某个 scheduler 段错误"，认定 n04-25 有问题。期间 n04-25 主机内存和 `/dev/shm` 都正常（可用 2888 GB、无残留）。
- **20:33 处理**：取消 35398，排除 n03-33、n04-25 重新提交两节点作业 **35416**（排队中）；单节点 35400 的排除列表加上 n04-25。两节点队列按优先级
  改为：65a 门禁 → 65b MegaMoeTP 计分（门禁干净才跑）→ 65c kopt-nocap 同分配对照计分 → 65d trace（作业剩余时间够才跑得完）。

### MegaMoeTP 端到端门禁通过（单节点 4+4，TP4 decode，20:33–20:41，35400 在 n04-33，`arm-C-prod-ptpcfix-nohip-megatp-1109T2033`）

生产 C 臂只加 MegaMoeTP（`arm-C-prod-ptpcfix-nohip-megatp.env`）：

| | MegaMoeTP | 生产 C 臂（53 / 57b） |
|---|---|---|
| smoke | 6/6 | 6/6 |
| 200 题探针 | **197/200（0.985）**，bad_start 0、garbled 0、撞上限 2 | 196–197/200 |
| 16k × 64 | 捞针 32/32，长前缀 GSM8K 32/32，runaway 0 | 32/32、30–32/32 |
| 真实接受长度 | **3.57**（263 行） | 3.60–3.62 |

- decode 的 4 个 rank 都装上了补丁，graph 捕获时本地 token 4–128 的每一档都走了 MegaMoeTP，没有报错、watchdog 没有触发。
- 接受长度正常，说明下午独立服务里约 1.0–1.4 的接受长度确实是那个非 PD 配置的问题。
- **接入在 PD 模式下端到端正确**。20:43 起在同一台节点上跑 TP4 C104 计分对比：65 MegaMoeTP → 66 生产 C 臂（各 1500 s 窗口）。
- **65 被抢占打断（21:04）**：35400 的 extern 步骤 21:04:36 结束（抢占预告），guard 先杀掉容器，AgentX 21:04:07 退出码 137，没有分数。
  顺序与 19:27 那次相同（先给步骤发信号、约 30 s 后 extern 结束），与改动无关。已把 65 重新排成 65b（排在 66 之前），作业重新开始后自动接着跑。
  今晚至此的抢占：35372（14:13）、35398（19:27）、35400（20:08、21:04）；有问题的节点：n03-33、n04-25（起栈时 scheduler 段错误）。
- 21:09 35400 被抢占取消，重新排队（`--start` 估 11-10 05:27）；35416 22:46:54 在 n06-25 + n06-33 开始，65a（TP8 MegaMoeTP 门禁）22:47 开始。
- **按新策略停了别人的容器**：两台节点上都有 `primus-training-35421`（rocm/primus:v26.9，22:25 创建，各占约 2267 GB 显存）。作业 35421
  （用户 jabowden）22:46:28 被其本人（uid 100170）取消，容器是残留；kit 只 stop、不删，已记入 `foreign_containers.log`。之后要留意 35416
  是否出现 `CANCELLED by 0`。

### TP8 PP8 上的 MegaMoeTP 门禁通过（65a，22:47–23:06，n06-25 → n06-33，`arm-C-1p1d-pp8-megatp-1109T2248`）

| | MegaMoeTP（kopt-nocap + MegaMoeTP） | 60：kopt（含 KV 上限） | PP8 原臂（11-05） |
|---|---|---|---|
| smoke | 6/6 | 6/6 | 6/6 |
| 200 题探针 | 195/200（0.975），bad_start 0、garbled 0、撞上限 4 | 197/200，撞上限 2 | 197/200 |
| 16k × 64 | 捞针 32/32，长前缀 GSM8K 32/32，runaway 0 | 32/32、30/32 | 32/32、31/32 |
| 真实接受长度 | **3.69**（609 行） | 3.59 | 3.60 |

- 8 个 decode rank 都创建了 MegaMoeTP 实例（TP8、每 rank inter 256、bf16 传输），graph 捕获时 20 档本地 token 都走了新路径；日志里唯一一条
  "crashed with exit code" 是门禁结束时的正常关停（−15，SIGTERM）。
- 195 对 197 在单次 200 题的波动范围内（没有 garbled、bad_start）。65b（MegaMoeTP C208 计分）23:05:48 自动开始，约 01:00 出结果；
  然后 65c（kopt-nocap 同分配对照）约 03:05，最后 65d trace（35416 到 04:46，可能来不及）。
- **35416 被抢占（23:17–23:22）**：extern 23:17:50 结束，65b 的 AgentX 被 guard 杀掉，没有分数；23:22:37 作业 "CANCELLED DUE TO PREEMPTION"，
  重新排队。**是正常抢占，不是 root 取消**（`CANCELLED by 0`），所以没有"停容器 → 被取消"的证据，策略不变。已把计分重排成 65b2（在 65c 前）。
- 35400 23:22 在 n04-33 重新开始（第 2 次重启），n04-33 上也有 `primus-training-35421` 的残留容器（约 2269 GB 显存），同样只 stop。
  TP4 计分对比开始：65b-score1500-megatp2（23:22）→ 66 生产 C 臂。

### TP4 计分：MegaMoeTP 没有带来预期的步长收益（65b，23:24–00:16，n04-33，`score-score1500-megatp2.txt`）

| | MegaMoeTP（65b，n04-33） | 生产 C 臂（59k，n02-21，不同节点） |
|---|---|---|
| 吞吐 / 输出 tok/s/GPU | 49,016 / 449.2 | 49,493 / 454.0 |
| 交互 p50 / p90 | 66.1 / 61.3 | 65.3 / 58.8 |
| ITL p50 / p90 | 15.1 / 16.3 ms | 15.3 / 17.0 ms |
| TTFT p50 / p90 / 均值 | 4.18 / 10.48 / 5.90 s | 3.85 / 9.37 / 5.27 s |
| decode 步长 @4 / @8 / @12 / @16 / @20 / @24 | 29 / 32 / 38 / 41 / 45 / 48 ms | 25 / 30 / 35 / 41 / 45 / 48 ms |

- 按单算子（真实权重 TP4 每层约 −42 µs），bs 约 20 时每步应快约 3 ms，**实际 bs 16–24 与生产持平，bs 4–12 反而慢 2–4 ms**。
- 可能原因：（1）小 batch（全局 64–200 token）MegaMoeTP 没测过，可能比生产的 KF / aiter 小 M 路径慢；（2）它是持久化 kernel，说明要求独占整张卡、
  不能与其他 stream 的 kernel 并发，decode graph 里若有别的 stream 同时跑会拖慢它；（3）单算子测的是同一层连续调用 10 次，真实场景每层只调一次、
  中间夹着别的 kernel，每次调用的启动 / 跨 rank 同步开销可能没被摊掉；（4）节点差异（66 同节点对照可排除）。
- 已排队（35400）：66 同节点生产对照 → 67 TP4 小 token 数（全局 16–256）的 MegaMoeTP 测试 → 68 开 MegaMoeTP 的 TP4 decode trace
  （`arm-C-prod-ptpcfix-nohip-megatp-trace.env`，看 graph 内每层的实际耗时、有没有与其他 kernel 重叠）。
- 在原因查清之前，MegaMoeTP 不进任何推荐配置；如果只是小 batch 慢，可以按本地 token 数设下限（例如 ≥ 48 才用）。
- **同节点对照（66，00:17–01:08，n04-33，`score-score1500-base2.txt`）确认了这个结论**：生产 C 臂 49,242 tok/s/GPU、交互 65.7 / 60.1、
  TTFT 均值 5.71 s，步长 @5 / @8 / @12 / @16 / @20 / @24 = 28 / 30 / 37 / 41 / 45 / 48 ms；MegaMoeTP 分别是 49,016、66.1 / 61.3、5.90 s、
  31 / 32 / 38 / 41 / 45 / 48 ms。**bs 16–24 完全持平，bs 5–12 慢 1–3 ms**；单算子里每层约 −42 µs 的优势在 decode graph 里没有兑现。
  接着看 67（小 token 数测试）和 68（graph 内 trace）。
- **67 小 token 数（01:09–01:11，`bench/out/megatp_tp4_small.log`，TP4 合成随机路由，µs/call，FP8 / bf16 传输）**：16：114 / 116；32：169 / 178；
  64：201 / 202；96：223 / 219；128：232 / 228；192：296 / 300；256：285 / 285。小 batch 下并不快：全局 64 token（bs 4）时已约 200 µs，
  因为 512 个路由几乎覆盖全部专家、主要在读权重；192 比 256 还慢，调度在这附近有不利的档位切换。与生产的对比要看 trace 按 bs 的数据
  （57 有生产的，68 是 MegaMoeTP 的，01:11 开始，约 02:20 结束）。
- **68 的 trace 丢了**：AgentX 01:57:12 正常结束，但 35400 的 extern 步骤 01:57:41 结束（又一次抢占预告），trace 在 decode 容器的
  `/root/.cache/<pid>` 里，`decode_profile_chain.sh` 用 `ssh <本机名> docker cp` 拷出时被 pam_slurm_adopt 拒绝，随后 down 删掉了容器。
  已改 `on()`：本机名直接本地执行；远端节点 ssh 被拒时改用 `srun --overlap` 进作业执行。trace 重新排成 holder1 的 68b。
- 两节点 **35436** 01:57:55 在 n06-33（prefill）+ n10-29（decode）开始，67a（TP8 MegaMoeTP C208 计分）在跑。
- **67a 又被抢占（02:34）**：extern 02:34:21 结束，AgentX 02:33:52 退出码 137，没有分数（TP8 MegaMoeTP 计分第 4 次被打断：35416 两次、
  n01-21 起栈段错误、这次）。已重排成 67a2（在 67b 前）。今晚这类低优先级作业平均 30–60 分钟就被抢占一次，2 小时的计分很难跑完。
- **03:11–03:22**：35400 恢复时分到 n01-21（它的排除列表是在发现 n01-21 有问题之前设的），68b（TP4 trace）在单节点 4+4 配置上两次起栈都失败，
  进一步确认 n01-21 有问题（不只是 PP8 prefill）。取消 35400，排除 n01-21、n02-25、n03-33、n04-25 重新提交单节点 **35456**，trace 重排成 68c。
- 35436 03:14 在 n04-29 + n04-33 恢复，67a2（TP8 MegaMoeTP 计分）在跑，decode 已起来。
- 另：03:16 有两个不是本工作流的作业（`llying-sgl-utcpu` 35454 / 35455，SGLang PR 48047 的 CPU 单测），不动。
- **TP8 MegaMoeTP 计分第 5 次被打断（06:20）**：35461（n05-29 + n06-25，两台都是 6.16.13）05:30 起栈正常，06:20 被抢占（extern 06:20:53），
  AgentX 退出码 137。PP8 计分要 2 小时以上（3600 s 窗口 + 预热），今晚平均 30–60 分钟被抢占一次，几乎跑不完。**改用 1500 s 窗口**
  （`arm-C-1p1d-pp8-megatp-d1500.env` / `arm-C-1p1d-pp8-kopt-nocap-d1500.env`，与 TP4 同节点 A/B 相同），整项约 1 小时。
  作业恢复后的顺序：69a5 驱动问题的测试脚本验证（约 20 分钟）→ 69a6 MegaMoeTP → 69a7 kopt-nocap 对照 → 69c trace。
- **1500 s 窗口的评估方法（06:25，回答用户）**：TP8 没有现成的 1500 s 基线，不同窗口的分数不能和之前 3600 s 的 56,593 / 56,595 直接比。
  所以在**同一次分配、同一对节点、同一窗口**上连着跑三组：69a6 MegaMoeTP（= kopt-nocap + MegaMoeTP）、69a7 kopt-nocap、69a8 PP8 原臂
  （`arm-C-1p1d-pp8-d1500.env`，新增）。69a6 对 69a7 是 MegaMoeTP 的纯效果，69a6 对 69a8 是整套改动相对生产的效果。判断依据按可靠程度：
  （1）decode 步长按最忙 rank 在跑数分档（@16 / @20 / @24），PP8 吞吐受 prefill 限制、步长最直接，单算子预测 bs≈20 时约 −3 ms；
  （2）交互度、ITL；（3）吞吐、TTFT（单次 ±1% 噪声，只看方向）；（4）接受长度。顺序：69a6 → 69a4（rocm10 测试脚本验证）→ 69a7 → 69a8 → 69c。
- **06:47 按用户意见改回 3600 s**：没有 1500 s 基线时，一次 3600 s（约 2 h）比三组 1500 s（约 3 h）快，并能直接和已有的 3600 s 基线比
  （64c kopt-nocap 56,595、11-06 原臂 56,593）。跨节点主要影响吞吐（1–5%），decode 步长和交互度受节点影响小（TP4 两台节点步长差 < 1 ms）。
  已删 69a7 / 69a8，改排 69a6-megatp-pp8-score8（3600 s）。改队列前 1500 s 的那一项已经开始（06:33），中途停掉会被当成抢占、卡住 holder，
  所以让它跑完：它的步长表与窗口长短无关，可以直接和 64c 比。结果只在模棱两可时再补同节点 3600 s 对照。
- **rocm10 镜像（69a4，06:30–06:34，n05-29，6.16）**：匿名拉取成功（约 3 min）；起服务时报 `ImportError: cannot import name '_check' from
  'sglang.kernels.ops.moe.smallm_moe_gfx950'`（上游镜像自身的问题，在 MoE 路径上），不是段错误。测试脚本给 rocm10 加了
  `SGLANG_ROCM_SMALLM_MOE=0`，重排成 69a5b（在 3600 s 计分前）。我们的镜像在 6.16 上 2 min 15 s 正常就绪（69a5），测试脚本可用。
- **1500 s 的 TP8 MegaMoeTP 计分（06:34–07:38，n05-29 → n06-25，窗口 07:06–07:31，`score-megatp-pp8-score-d1500.txt`）**：
  吞吐 64,944 tok/s/GPU、交互 81.1 / 71.2、ITL 12.3 / 14.0、TTFT 4.79 / 8.04 / 7.03 s；prefill 满块 238 ms、现算 28,186 tok/s；decode 在跑总数中位 82、
  最忙 rank 中位 14；步长 @8 / @12 / @16 / @20 / @24 = 28 / 34 / 37 / 41 / 43 ms（64c：28 / 32 / 36 / 42 / 45）。
  - **不能说明 MegaMoeTP 有效果**：吞吐高出的 15% 来自 prefill（每块快 26%，MegaMoeTP 只改 decode）——1500 s 窗口靠前、上下文更短，加上节点不同；
    交互度低是因为 prefill 更快、decode 里同时在跑的请求更多（82 对 54）。步长基本持平（bs 20–24 快 1–2 ms、bs 12–16 慢 1–2 ms），而这个窗口上下文
    更短、本应更快，所以与 TP4 的结论一致：在 decode graph 里没有看到单算子预测的约 −3 ms。结论等 3600 s 计分（rocm10 重试之后开始）。
- **3600 s 计分第 1 次（07:38）起栈时被抢占；第 2 次（09:00–10:04，n06-25 → n06-33）跑了 60 分钟后又被抢占**（第 7 次），没有分数。
  抢救了日志（`vfix-logs/opt/partial/c208-1110T0900`），在计算节点上用 `analyze_pd_logs.py` 取 09:34–10:04（与 64c 窗口相近的时段）：
  prefill 满块 266 ms、现算 24,196 tok/s；decode 在跑总数中位 64、最忙 rank 中位 12；**步长 @8 / @12 / @16 / @18 / @20 / @21 / @22 =
  28 / 33 / 36 / 39 / 40 / 42 / 42 ms，64c 为 28 / 32 / 36 / 39 / 42 / 43 / 44**。
- **TP8 结论（两段数据一致）：MegaMoeTP 只在 bs ≥ 20 时快 1–2 ms（约 3–5%），bs 8–18 持平**；单算子预测 bs≈20 时约 −3 ms，实际只兑现一部分
  （约每层 −15~−25 µs，而不是 −40 µs）。TP4 上完全没有收益（bs 16–24 持平、bs 5–12 慢 1–3 ms）。完整的 3600 s 吞吐 / 交互对比再排一次（69a6d）。
- **69a6d 第 3 次 3600 s 计分（11:10–11:40，n06-25 → n06-33）又被抢占**，AgentX 还在预热阶段（returned 924 / 2302）就被杀，没有可用数据。
  - 现象：两台节点上我们的所有容器 11:40:38 同时被 SIGKILL（docker events：`container kill … signal=9`）。不是 OOM，也不是 AgentX 的 15 分钟无进展保护。
    sacct：`35461.extern` 11:41:07 结束，guard 步骤 `35461.0/.1` "CANCELLED DUE to SIGNAL Terminated"，`PreemptTime=11:40:38`，作业带 BeginTime 重排。
    即 guard 收到抢占的 SIGTERM 后按设计删了容器（防止占着显存影响下一个作业）。
  - **谁在抢**：zirzhang 的 `flux-ainic-g1b`（QoS shared-medium，可以抢占我们的 batch QoS；每个 1 h）**固定指定 n06-[25,33]**，今天 08:00、10:10 都来过，
    下一个 35534 在排队；同一用户的 `flux-ainic-g1a` 固定 n01-33 + n02-21。35461 今天 4 次分到 n05-29/n06-25 或 n06-[25,33]，基本都是被这组作业挤掉。
    我们账户只能用 batch / debug / normal / shared-low，batch 已经是最高的，没有不被抢占的选项。
  - **对策**：（1）35461 的排除列表加上 n06-25、n06-33、n01-33、n02-21（11:47 `scontrol update`，原有的 6.19 节点排除保留），预计 11:48 在别的节点开始；
    （2）`opt_item5.sh` 的漏洞：抢占时 Slurm 只给 srun 步骤发 SIGTERM，不给批处理步骤发，所以被杀的计分被当成正常结束、条目进了 done/（69a6d 就是这样）。
    已在结尾加检查：作业的 `PreemptTime` 晚于本条目开始，就走原有的 `preempted`（删引擎、条目留在 running/，下次分配重跑）；
    （3）计分重排为 `69a6e-megatp-pp8-score11`（在 69c trace 之前）。
- **69a6e（13:22–13:52，n01-29 → n02-29）又被抢占**，AgentX 还在预热（returned 128 / 2302）。两次的规律一样：`PreemptEligibleTime` = 作业开始 + 30 分钟
  （11:10:29 → 11:40:29 被抢；13:22:15 → 13:52:28 被抢），即**新作业有 30 分钟免抢占期，过后只要有更高 QoS 的作业要节点就会被挤掉**，白天几乎每次都碰上。
  新加的检查生效：opt.out 记"SIGTERM (preemption notice)"，条目留在 running/。
  - 3600 s 计分要连续约 85 分钟（显存等待 5 + 起栈 15 + 预热约 20 + 窗口 60），只能等夜里节点空闲的时段。AgentX 的 trace 也不行：C208 预热第一段
    要先把 222 条首请求的长上下文 prefill 完（约 16 分钟），"Warmup complete" 之后再等 600 s 才采，第一份 trace 在开始后 40 多分钟；
    而且原脚本要等 AgentX 跑完才把 trace 拷出容器，中途被抢占 guard 删容器时 trace 一起丢（trace2/3/4 都是这样没拿到）。
  - **改用合成负载采 trace（14:05 写好）**：这次 trace 要回答的是"MegaMoeTP 在 graph 里每层 MoE + 通信实际多少 µs"，只与每 rank batch 有关，与上下文长短无关；
    kit 用模拟接受长度（SIM_ACCEPT_LEN），文本内容不影响 MTP。
    `pkg/scripts/synth_decode_load.py`（只用标准库，N 个流各自循环发 chat 请求：约 2048 token 随机 prompt、ignore_eos、4096 token 输出，分阶段提并发）
    + `scripts/decode_synth_trace.sh`（起栈 → 负载 96 / 160 / 192 并发各 150 s，约对应每 rank 12 / 20 / 24 → 每阶段 90 s 时采 40 步 →
    该阶段结束立即把 trace 拷出容器 → down），`opt_item5.sh` 新类型 `strace`，整项约 25 分钟，落在免抢占期内。
    新增 `arm-C-1p1d-pp8-kopt-nocap-trace.env`。队列：69a5s MegaMoeTP strace → 69a5u kopt-nocap strace（同负载对照）→ 69a6e 3600 s 计分；
    AgentX 版 trace（69c）移到 `parked/`。
- **K4（dense 投影）第一步排队（14:25，`kopt-20261109/bench/k4_splitk_sweep.py`，条目 `69a5w-k4-splitk-sweep`，两个 strace 之后、计分之前，约 15 分钟）**：
  - 依据第十节：生产调优表有 hipBLASLt 和 FlyDSL split-K hgemm 两类候选，aiter 调参器只调出 3–4%，瓶颈判断为启动 + split-K 归约。还有一个没排除的因素：
    调优表记的 o_proj 带宽 5.9 TB/s，而生产 trace 只有 4.2 TB/s——o_proj 权重 201 MB 小于 256 MB 的 MALL，调参时很可能是热缓存，挑出的配置不一定适合
    每层权重都不同的 decode。
  - 基准：三个形状 × 每 rank 行数 32 / 48 / 64 / 80 / 96 / 128，权重轮换 ≥ 1 GB（冷读），graph 回放。对比生产 `tgemm.mm`（挂生产调优表）、Triton
    `gemm_a16w16`（split-K + 归约 kernel）、Triton `gemm_a16w16_atomic`（split-K 直接 bf16 atomic 加到输出：清零 + 一次启动；另测不清零的 accumulate，
    即把 o_proj 与残差加法融合时的代价）。配置扫 BLOCK_M 16–64 × BLOCK_N 32–128 × BLOCK_K 128/256 × split-K 1–16 × warps 4/8，8 卡分片并行。
  - 记录仓库补上了 kopt 基准脚本（`opt/kopt-bench/`）和 MegaMoeTP 接入代码（`opt/overlay-kopt/`），之前漏了。
- **合成负载 trace 第一次成功（69a5s，35461 16:00 在 n04-33 → n10-29 开始，16:22 完成，三档 trace 全部拷出）**：
  - 负载正常：ignore_eos 生效（每条 4096 token），errors=0；起栈约 12 分钟。96 并发时所有请求同起同落成"波次"，第一次采样正好在两波之间；
    已给负载加 ±50% 输出长度和 0–20 s 错开（从 69a5u 起生效）。分析看每步实际 batch，不受影响。
  - 新分析脚本 `scripts/ana/moe_bybs.py`（`kdeep.py` 按 all-gather 切层，MegaMoeTP 没有 all-gather 了；这里按 kernel 类别求和、按该步最大 rank batch 分档），
    在计算节点上跑（`srun --overlap`），结果 `trace/opt-megatp-pp8-strace/moe_bybs.txt`：

    | 每 rank batch（全局 token） | MegaMoeTP 每层 | 每步 | verify graph kernel 合计 |
    |---|---|---|---|
    | 12（384） | 181 µs | 13.6 ms | 26.9 ms |
    | 20（640） | 222 µs | 16.6 ms | 33.9 ms |
    | 24（768） | 236 µs | 17.7 ms | 36.6 ms |

    MegaMoeTP 随 token 数明显增长（bs 12 → 24 多 55 µs），在 graph 里比单算子测试（704 token 约 204 µs）慢 10–15%。小 batch 用 `npp2` 变体。
    估算下限：每 rank 要流过 257 个专家的权重约 600 MB，384 token × top-8 几乎激活全部专家，按 5 TB/s 约 120 µs，所以 MoE 本身受权重带宽限制，
    MegaMoeTP 能省的只有通信 + 启动 + 前处理。
  - **kopt-nocap 对照（69a5u，16:23 开始）只拿到 96 并发那份**：它排在第二个、开始时已过 22 分钟，16:30（免抢占期末）被抢占，160 / 192 两份没拷出
    （extern 结束后 ssh 被拒，guard 已删容器）。条目留在 running/，下次分配排第一、能在 30 分钟内跑完。96 并发那份
    （`trace/opt-koptnocap-pp8-strace/traces/decode-c96-163011`）等有计算节点时分析（8 个约 9 MB 的 trace，登录节点内存不够）。
- **kopt-nocap 对照完整（69a5u 重跑，17:24–17:43，同一对节点，三档全部拷出）；MegaMoeTP 收益为什么只有 1–2 ms 有了答案**。
  `moe_bybs.py` 改为互斥分类（KF 融合 gemm1 名为 `mfma_moe1_..._fp4q_sort_...`，原来被前处理和 GEMM 各算一次）。每层 µs（`moe_bybs.txt`）：

  | 每 rank batch | 生产 AG + RS | 生产 gemm1 + gemm2 | 生产前处理 | **生产 MoE + 通信** | **MegaMoeTP** | 每层差 | 每步 kernel 合计差 |
  |---|---|---|---|---|---|---|---|
  | 12 | 20.0 + 20.7 | 82.6 + 47.4 | 5.3 | **176.0** | **180.9** | +4.9 | +0.10 ms |
  | 20 | 29.5 + 32.4 | 87.5 + 63.7 | 22.3 | **235.4** | **221.6** | −13.8 | −1.27 ms |
  | 24 | 35.0 + 38.7 | 91.3 + 73.6 | 23.0 | **261.6** | **235.7** | −25.9 | −2.17 ms |

  （前处理已扣掉两边都有的 shared expert 追加 4.2 µs。）与 3600 s 计分的"bs ≥ 20 快 1–2 ms、以下持平"完全吻合。
  - **潜力**：生产的通信 + 前处理每层 46 / 84 / 96 µs。MegaMoeTP 若把它们全省掉、GEMM 效率又与生产持平，每步可快 3.5–7 ms。
  - **实际**：只省 0 / 14 / 26 µs——它的单个 kernel 比"生产 GEMM 本身"（130 / 151 / 165 µs，bs 12 时约 4.6 TB/s，已接近权重带宽下限）还慢
    51 / 70 / 71 µs：内部 GEMM 效率不如 KF / port kernel，且通信没有与计算重叠（阶段串行）。
  - graph 内比单算子测试慢约 15%（768 token 236 µs，单测 704 token 约 204 µs）：较可能是各 rank 到达 MoE 层的时间不同，persistent kernel 的 all-gather
    要等最慢的 rank，等待算进它的时间（生产 all-gather 的时间里同样含这段等待，单测里没有）。
  - **后续两个方向**（写入 KERNEL-PLAN 7.4）：（a）MegaMoeTP 内部：读 `mega_moe_tp.py` 的配置面（tile、`nab` / `npp` / `cb` / `lb` / `mix` / `pf`），
    按我们的 M 调参，并看能否让 all-gather 与 gemm1 分块流水；（b）生产路径上单独压通信：gemm1 本来就吃 FP4 激活，先量化再 all-gather
    （MXFP4 约为 bf16 字节的 1/4），AG 每层 20–35 µs 有望降到约 8–12 µs，改动比重写 MegaMoeTP 小，RS 保持 bf16 不影响精度。
- **K4 扫参（69a5w，17:43 开始）17:55 被抢占**（35461 第 8 次；启动后 30 分钟），合并没来得及跑。部分结果另存
  `bench/out/k4_splitk_partial_1110T1755/`（8892 次测量，7808 有效、160 算错、924 编译 / 运行失败；`merge.txt`）。按每 rank 行数：

  | 形状 | 生产（冷读） | Triton atomic split-K（清零 + 一次启动） | atomic 累加（不清零，即融合残差加法） |
  |---|---|---|---|
  | fused_qkv_a，M 32–64 | 10.6–12.7 µs（2.5–3.1 TB/s） | 0.99–1.02x | 1.16–1.23x |
  | q_b_proj，M 32–64 | 17.8–17.9 µs（3.8 TB/s） | 0.99–1.16x（split-K 归约版 1.02–1.17x） | 1.12–1.36x |
  | o_proj，M 32–64 | 37.6–43.5 µs（4.6–5.4 TB/s） | 0.95–0.98x | 1.00–1.02x |
  | 三者，M 80–128 | o_proj 46.9–50.0 µs（4.0–4.3 TB/s） | 0.55–0.87x | — |

  - 生产冷读与 graph 内 trace 一致（o_proj 每层 42.5 + 5.1 µs），说明之前的 trace 数字可信，调优表里 5.9 TB/s 是热缓存的结果。
  - **M ≥ 80 慢是扫参空间的漏洞**：只给了 BLOCK_M ≤ 64，M 96 要切两块、权重读两遍。已补 BLOCK_M 128，并去掉需要 3 块以上的配置；
    `k4_splitk_sweep.py` 改为可续跑（读全部分片结果、跳过已测），条目不再清空结果目录，合并不再加载 torch。
  - 初步看：Triton atomic 单独替换收益不大；**"不清零直接累加"有 1.0–1.36x**，意味着把投影 GEMM 与后面的残差加法融合（y += x·Wᵀ）才有空间，
    等 BLOCK_M 128 的结果出来再判断 M 80–96（我们最常见的行数）。
- **18:39 35461 在 n01-29 → n02-29 恢复**，K4 扫参续跑（测量数 8892 → 10458，续跑生效）。
- **K4 扫参完成（`bench/out/k4_splitk/merge.txt`，10458 次测量，8618 有效）**：18:40 续跑一分钟内补完（kernel 已有编译缓存）；条目却卡到 19:09——
  条目脚本最后的 `wait` 不带参数，把一直睡眠的 guard 步骤也等进去了（已手动结束，rc 143；69a5x 是前台执行，没有这个问题）。
  补上 BLOCK_M 128 后 M ≥ 80 从 0.55–0.87x 回到接近持平。我们最常见的 M 80–96：

  | 形状 | 生产（冷读） | Triton atomic split-K | 不清零累加（融合残差加法） |
  |---|---|---|---|
  | fused_qkv_a | 14.9–15.7 µs（2.0–2.2 TB/s） | 0.93–0.95x | 1.04–1.06x |
  | q_b_proj | 20.1–20.2 µs（3.3 TB/s） | 0.92–0.93x | 1.00–1.05x |
  | o_proj | 46.9–47.8 µs（4.2–4.3 TB/s） | 0.89–0.93x | 0.91–0.98x |

  **结论：aiter 现成的 Triton split-K / atomic kernel 在我们的行数上做不出收益**（只有 M ≤ 64 且融合残差加法时 1.1–1.36x）。
  最大的短板仍是 fused_qkv_a（窄 N，2.0–2.8 TB/s）；要么手写 kernel（例如 N 方向更细的 split + 与 q/kv a-norm 融合），要么 dense FP8（需用户决定）。
  K4 暂时降级，优先 M2（MegaMoeTP 调参，69a5x 19:09 开始）。
- **M2 第一步结果：MegaMoeTP 的行 tile 有明显收益（69a5x，19:09–19:11，n01-29，第 3 层真实权重，`bench/out/megatp_sweep.jsonl`）**。µs / 层（单算子）：

  | (mt, npp) | 384 token | 640 | 768 |
  |---|---|---|---|
  | 默认 (3, 1) | 173.0 | 203.7 | 220.2 |
  | **(4, 2)** | **171.3** | **190.6** | **196.8** |
  | (2, 1) | 171.0 | 214.1 | 232.8 |
  | (6, 1) / (8, 1) | 177.0 / 176.8 | 194.7 / 194.9 | 203.4 / 203.9 |
  | (1, 1) | 204.1 | 262.8 | 307.1 |
  | 生产 MoE（不含通信，参照） | 132.2 | 163.6 | 183.1 |

  npp 只有 ±2 µs，SNR 都是 29.1（精度不变）。mt 4 / npp 2 在 640 / 768 token 比默认快 13 / 23 µs。按 graph 内多出的 8–16 µs 估算，bs 20 / 24 每层比生产
  省约 26 / 49 µs（原来 14 / 26），每步约 −2.0 / −3.7 ms（原来 −1.0 / −1.9）。只需三个环境变量，不改代码：新臂 `arm-C-1p1d-pp8-megatp-mt4.env`
  （`AITER_MEGAMOE_TP_LB_MT=4 AITER_MEGAMOE_TP_LB_MT_SMALL=4 AITER_MEGAMOE_TP_LB_NPP=2`）和 `-trace.env`。
- **19:15 队列调整**：停掉默认配置的 3600 s 计分（69a6e，刚起栈；先杀 pd_perf_chain 子进程，再 SIGHUP 让 opt_item5 走正常退出清理，避免 SIGTERM
  进"被抢占"路径挂住 holder；rc 129）。新顺序：69a6a 第二轮扫参（mt 4 / 5 × npp 2 × q 1 / 2 / 4 / 8）→ 69a6b mt4 合成负载 trace（与 69a5s / 69a5u 同负载）
  → 69a6f mt4 的 3600 s 计分。
- **第二轮扫参（69a6a，19:15–19:16）：q = 2 再快 9–13 µs**。(mt, npp, q) 在 384 / 640 / 768 token：(4, 2, 1) 171.3 / 190.6 / 196.8；
  **(4, 2, 2) 159.9 / 177.5 / 187.5**；(4, 2, 4) 162.7 / 185.1 / 194.2；(5, 2, 2) 160.8 / 178.4 / 188.3。相对默认 −13 / −26 / −33 µs / 层。
  `arm-C-1p1d-pp8-megatp-mt4.env` 加上 `AITER_MEGAMOE_TP_LB_Q=2`。已开始的 q1 trace（69a6b）和 q1 计分（69a6f）都停掉（各 1–2 分钟，rc 129），
  改排 **69a6c mt4q2 合成负载 trace（19:18 开始）→ 69a6g mt4q2 的 3600 s 计分**。
  （停 69a6b 时第一次用 `pgrep -f` 匹配到了自己的命令行、把自己杀了；改为按命令行开头精确匹配。）
- **M2 第一步排队（`bench/megatp_sweep.py`，条目 `69a5x-megatp-sweep`，K4 之后、计分之前）**：MegaMoeTP 的大 batch 调度有对外参数
  （`aiter/ops/flydsl/mega_moe_tp.py`）：行 tile 16 × `lb_mt`（≤ 512 token 用 `lb_mt_small`，我们的 384 落在这档，两者一起设）、`lb_npp`（gemm1 每读一遍 A
  处理的列块数）、`lb_q`（gemm2 每单元的输出列块数），默认 3 / 1 / 1；全局 token ≥ 256 走这套调度，我们的 384 / 640 / 768 都是。
  一个 8 卡进程加载一次第 3 层真实权重，逐个配置建实例、在三个 token 数上测 graph 时间和对生产的 SNR，并记生产 MoE 部分（不含通信）作参照；
  第一轮 mt 1 / 2 / 3 / 4 / 6 / 8 × npp 1 / 2 / 4（q = 1），可续跑。

### 起栈段错误的共同原因（00:20 推断）与对策

- 35416 23:58 在 n01-21 + n06-33 恢复，65b2（TP8 MegaMoeTP 计分）起栈失败：**n01-21 的 prefill `scheduler_2` 段错误（−11）**，与 n04-25 的
  情况一模一样（也是 scheduler_2），与 n03-33 decode 的段错误同类。三台节点都出这种问题，不像三台都是坏硬件。
- 共同点：kit 自己的提示是"驱动在引擎删除后 10–40 分钟才释放显存，up 之前要检查 rocm-smi"。n01-21 上别人的容器（占约 2249 GB 显存）23:58:36
  才被停掉，prefill 约 00:06 就开始起栈，显存很可能还没释放完；00:20 再看时各卡显存已经是 0，65c 的 prefill 正常开始加载。
- **对策（已改 `scripts/opt_item5.sh`）**：清理完容器、镜像就绪之后，起栈之前等两台节点的每张卡显存都降到 2% 以下，最多等 30 分钟（超时照常开始并记日志）。
- 65c（kopt-nocap 同分配对照）00:20 在同一对节点上开始；65b2 需要再排一次（见下）。
- **更正上面的推断**：65c 起栈时显存已经是 0，n01-21 的 prefill 仍然段错误（scheduler_2），65c2（scheduler_7）、65d 也一样，n01-21 上
  **连续 4 次**失败，所以"显存没释放"解释不了。显存等待保留（无害，实测在 n06-33 上等了约 1 分钟）。
- **崩溃点（faulthandler 调用栈）**：prefill scheduler 主线程在 `scheduler.py:1948 run_event_loop` 里，执行
  `self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream` 时，C 层 `c10::hip::HIPStream::stream()` 段错误。这一行只在
  PP > 1 或开启 overlap 时执行，位置在"引擎初始化（graph 捕获、预热）完成之后、开始服务之前"，与 n03-33 decode"graph 捕获后约 50 s"、
  n04-25 / n01-21 prefill"初始化后"崩溃的时间点都吻合，应是同一个问题。
- **与节点强相关**：坏的节点每次都崩（n03-33 decode 3 次、n04-25 prefill 2 次、n01-21 prefill 4 次），正常的节点一次都没有（n04-33、n10-29、
  n01-29、n02-21、n06-25、n06-33），崩的不是同一张卡（GPU 2 / 4 / 6 / 7）。最可能是这几台节点的 HIP 运行时 / 驱动状态有问题（读 HIP 流池时出错），
  与我们的改动无关（kopt-nocap 和 PP8 原臂也崩）。根因没有查到，留给你判断是否报给集群管理员。
- **00:40 处理**：取消 35416，排除 n01-21、n03-33、n04-25 重新提交 **35436**（排队中）；两节点队列：67a MegaMoeTP 计分 → 67b kopt-nocap
  同分配对照 → 67c trace。

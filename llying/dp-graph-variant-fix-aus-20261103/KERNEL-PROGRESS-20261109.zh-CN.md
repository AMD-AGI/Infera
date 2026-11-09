# kernel 优化进展（11-09，按 KERNEL-PLAN-20261109 阶段 0 执行）

用户 07:44 同意边 review 边开始。优先级按用户决定：TP8 MoE 优先，同时做 TP4 / TP8 共用的 absorb BMM。
本文件只记录实测；计划与背景见 `KERNEL-PLAN-20261109.zh-CN.md`。

## 一、结论（截至 08:20 UTC）

找到三处 **只改配置、不写新 kernel** 就能拿到的收益，都在单算子上实测过，合计 TP8 每步约 **−3.7 ms**（步长约 46 ms 的 8%）；
TP4 约 −0.9 ms。都还没有在完整的 decode graph 里验证，下一步需要一次 TP8 trace（第五节）。

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
  - kernel（上游）：页偏移改用无符号 32 位（到 4 GiB、约 7.4M slot 都成立），交给 yihou 并入 #47278。
- split 台阶在行数 86（3 → 2 个 split）：K3（每请求 4 行）是 graph bs 22 起，K5（每请求 6 行）是 graph bs 15 起；选 K 和 DP 均衡时可以参考。

## 六、未解决

- **KF MoE 在 inter 512 上与 aiter 的 SQNR 只有 4.1 dB（我的随机权重）**：只在 KF 自己的 gemm1 启动器（`GLM52_MOE_G1` = 1 / 2 / 3）时出现，用 aiter 的启动器
  （`GLM52_MOE_G1=0`）是 49.6 dB；inter 256（KF 专用 gemm1 不启用）是 49–54 dB。生产日志里 KF 自检是 52 dB 或完全一致，生产的端到端门禁（200 题、
  lm-eval 0.97、接受长度 3.62）也都通过，所以更可能是我的输入不满足 KF 某个假设（例如 scale 的取值范围）。查清楚要用真实权重做一次对比，留到之后；
  它不影响 TP8 的工作，也不影响上面的计时（KF 的计时与生产 trace 一致）。

## 七、下一步

1. **TP8 验证运行（需要你同意，两节点约 1.5 h）**：在 2P1D 或 PP8 上同时加第 1–3 条（挂两个 JSON、一份改过的 fmoe CSV、`--max-total-tokens 3700000`），
   先过 `verify_suite`，再采 TP8 trace 用 `kdeep.py` 对位，最后计分。预期 verify graph 每步约 −3.7 ms。
2. 单节点 4+4（TP4）只需要第 2 条，可以在单节点队列里做一次门禁 + 计分（约 1.5 h）。
3. 继续写 kernel：TP8 gemm2（约 30 µs / 层的空间）和 MoE 前处理（约 16 µs / 层）；在 n10-29 剩余时间里先做 gemm2 的方案对比。

# kernel 优化进展（11-09，按 KERNEL-PLAN-20261109 阶段 0 执行）

用户 07:44 同意边 review 边开始。优先级按用户决定：TP8 MoE 优先，同时做 TP4 / TP8 共用的 absorb BMM。
本文件只记录实测；计划与背景见 `KERNEL-PLAN-20261109.zh-CN.md`。

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

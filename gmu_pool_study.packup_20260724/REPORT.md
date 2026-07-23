# DSv4-Pro SGLang 1P1D 每卡 KV cache 实际内存 × gmu(mem-fraction-static) — 分析 + 实测报告

**日期**：2026-07-24 ｜ **机器**：chi2879 (8×MI355X gfx950, 288 GiB/卡) ｜
**镜像**：`infera/engine-sglang:pd-mcgate`（sglang **0.5.15.post1**, sgl_kernel 0.4.4, torch 2.9.1+rocm7.2.0）｜
**模型**：DeepSeek-V4-Pro（fp8 e4m3 attn + MXFP4 MoE, 61 层, MLA+DSA indexer）

---

## 0. 结论先行（TL;DR）

**每卡 KV cache 实际内存 = gmu 的严格线性函数**，DSv4-Pro TP8+DP-attn 下（per DP-rank，即每卡）：

```
KV_pool(GiB) ≈ 276.9 × gmu − 130.3          (5 点实测 R²=1.000)
每 +0.01 gmu  →  KV pool +2.77 GiB  →  +216,470 个 full-context token 容量
```

| gmu | **每卡 KV pool** | 占卡 VRAM | full_token/卡 | 卡 VRAM 实占 |
|-----|-----------------|----------|--------------|-------------|
| 0.80 | **91.24 GB** | 31.7% | 7,053,056 | 221 GB (77%) |
| 0.85 | **105.08 GB** | 36.5% | 8,135,424 | 233 GB (81%) |
| 0.88 | **113.39 GB** | 39.4% | 8,784,896 | 240 GB (84%) |
| 0.90 | **118.93 GB** | 41.3% | 9,217,792 | 245 GB (85%) |
| 0.92 | **124.47 GB** | 43.2% | 9,650,944 | 250 GB (87%) |

**机制**：`--mem-fraction-static`(gmu) 在**启动时**决定"weights + KV pool"占卡 VRAM 的比例；
KV pool = `gmu × VRAM_total − 模型权重`，一次性静态分配、常驻、**运行时不再变**。
剩余 `(1−gmu)` 留给 activations + CUDA graph。

**手动调节**：调 gmu 是放大/缩小 KV pool 的**唯一直接旋钮**，线性、可预测。
`avail mem`(启动后打印的余量) 太高(>10-20GB)就调高 gmu 榨更多 KV；OOM 就调低；每次 ±0.01 逼近。
P/D role 只改**默认值**（P 0.85 / D 0.90），池公式与 role 无关（实测逐字节一致）。

**运行时验证**（真跑 8k/1k conc64×128）：峰值 KV 占用仅 **1–5%**、**retract=0** —— 池启动即锁定、运行时远在池内。

---

## 1. 机制：gmu 如何变成 KV pool（sglang 0.5.15 源码链）

### 1.1 三步链条（file:line 均来自 upstream tag v0.5.15.post1）

**Step A — gmu 圈定可用字节**
`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:66-80` `_profile_available_bytes`：
```python
rest_memory = post_model_load_free − pre_model_load_free × (1 − mem_fraction_static)
# 等价于:  KV_budget = VRAM_total × gmu − 模型权重
```
- `pre_model_load_free` = 加载权重**前**的空闲显存（≈整卡容量）
- `post_model_load_free` = 加载权重**后**的空闲显存
- `× (1−gmu)` 是**扣给 activations + CUDA graph 的动态预留**，剩下 `rest_memory` 全给 KV。

**Step B — 字节预算 ÷ 每 token 成本 = token 容量**
DSv4 走专用配置器 `pool_configurator.py:513 DSV4PoolConfigurator`：
```python
full_token = int(available_bytes_for_tokens / bytes_per_full_token)   # :734
```
实测日志印证（gmu=0.90, per-rank）：
```
DSV4 memory calculation: bytes_per_full_token=13735.04, available_bytes=118.93 GB,
                         c128_state_fixed=1.01 GB, full_token=9217792
```

**Step C — full_token 按固定比例劈成多池**
`pool_configurator.py:651 _compute_dsv4_sizes`：
```python
swa  = full_token × swa_full_tokens_ratio   # ratio=0.15 →  full×0.15
c4   = full_token // 4                       # full-attention MLA latent 池
c128 = full_token // 128                     # DSA indexer 池
c4_logical = c128 × 32 = full_token // 4      # indexer 逻辑容量
```
实测印证：`DSV4 pool sizes: full=9217792, swa=1382656, c4=2304448, c128=72014, ...`
（1382656/9217792 = 0.150 ✓；9217792/4 = 2304448 ✓；9217792/128 = 72014 ✓）

### 1.2 DSv4 的多池结构（比老版 sglang 更细）

sglang 0.5.15 对 DSv4 用 `DeepSeekV4TokenToKVPool`（`deepseek_v4_memory_pool.py:449`），
**没有**单一 `max_total_num_tokens`，而是分 3 个 token 池 + state 池：

| 池 | 含义 | 每 token 字节/层 | 大小 |
|----|------|-----------------|------|
| **swa** | 滑窗注意力 KV（覆盖全部 61 层，窗口内） | 584 B | full×0.15 |
| **c4** | full-attention 压缩 MLA latent | 584 B | full/4 |
| **c128** | DSA indexer（稀疏注意力索引头） | 132 B | full/128 |
| c4_state / c128_state | FP8 量化 scale/state 环 | — | 小（state_fixed=1.01GB） |

- **MLA cell = 584 B/token/层**：`qk_nope_head_dim(448, FP8) + qk_rope_head_dim(64)×2(BF16) + scales`
  （`deepseek_v4_memory_pool.py:113` 的 assert 固定）。**注意不是** `(kv_lora_rank+qk_rope)×dtype` 的老 MLA 式——
  DSv4 DSA 布局把压缩 latent 存成 qk_nope 宽的 FP8。
- **indexer cell = 132 B/token/层**（FP8）：`index_head_dim(128)+4`（`:283`）。
- **综合 `bytes_per_full_token=13735.04`** = 各池每 token 字节 × 各自覆盖层数的加权和
  （`_get_bytes_per_full_token` `:609`）——这是把"1 个 full-context token"摊到所有池的总成本。

### 1.3 "static" 的含义 + DP-attn

- **static** = 模型权重 + KV pool 在**启动时一次性分配、常驻整个 server 生命周期**；
  与之相对的 activations/CUDA-graph scratch 是**每 batch 动态波动**的，由 `(1−gmu)` 覆盖。
  官方定义（`server_args.py:1457`）：`mem_fraction_static = (weights + KV pool) / VRAM_total`。
- **DP-attn**：每个 DP rank **独立** profile 自己的空闲显存、独立算自己的 pool
  （经 `get_attention_tp_size = tp//attn_dp//attn_cp`），不是显式 ÷dp_size。8 rank 各打一条
  `DSV4 pool sizes` 日志，数值在舍入内一致 → **每卡 pool = 上表数值**。
- 默认 `mem_fraction_static=None` → sglang 动态估（fallback 常数 0.88）。本研究显式传值以做干净扫描。

---

## 2. 实测：gmu 扫描（Step 1 decode 5 点 + Step 2 prefill 2 点对照）

### 2.1 decode 腿 gmu 扫描（核心曲线）

每点单机起 decode 单腿、读启动期 KV pool 分配、`rocm-smi` 抓 VRAM，kill 换下一点。**纯 launch 读数**。

| gmu | KVpool GB | full_token | swa | c4 | c128 | 启动后 avail mem | VRAM 实占 GB |
|-----|-----------|-----------|-----|-----|------|----------------|-------------|
| 0.80 | 91.24 | 7,053,056 | 1,057,792 | 1,763,264 | 55,102 | 67.92 | 221.4 |
| 0.85 | 105.08 | 8,135,424 | 1,220,096 | 2,033,856 | 63,558 | 55.95 | 233.4 |
| 0.88 | 113.39 | 8,784,896 | 1,317,632 | 2,196,224 | 68,632 | 48.84 | 240.5 |
| 0.90 | 118.93 | 9,217,792 | 1,382,656 | 2,304,448 | 72,014 | 44.02 | 245.3 |
| 0.92 | 124.47 | 9,650,944 | 1,447,424 | 2,412,736 | 75,398 | 39.31 | 250.0 |

**线性性**（每相邻两点斜率完全一致）：
- **+2.770 GB KV pool / +0.01 gmu**（4 段全为 2.768–2.770，无偏差）
- **+216,470 full_token / +0.01 gmu**
- 线性拟合 `KVpool_GB = 276.9×gmu − 130.3`。斜率 276.9 GB/unit ≈ VRAM_total(288) − weights →
  **反算模型权重 ≈ 11.1 GiB/卡**（DSv4 fp8 ~89GB 总权重 ÷ TP8 ≈ 11GB，吻合）。

### 2.2 prefill 抽 2 点对照（证明池公式与 role 无关）

| gmu | leg | KVpool GB | full_token | swa | c4 | c128 |
|-----|-----|-----------|-----------|-----|-----|------|
| 0.85 | decode | 105.08 | 8,135,424 | 1,220,096 | 2,033,856 | 63,558 |
| 0.85 | **prefill** | **105.08** | **8,135,424** | **1,220,096** | **2,033,856** | **63,558** |
| 0.90 | decode | 118.93 | 9,217,792 | 1,382,656 | 2,304,448 | 72,014 |
| 0.90 | **prefill** | **118.93** | **9,217,792** | **1,382,656** | **2,304,448** | **72,014** |

**逐字节一致** → KV pool 大小**只由 gmu 决定，与 P/D role 无关**。P/D 的真实差异仅两点：
1. **默认 gmu 不同**：prefill 默认 0.85（DP-attn prefill 高并发 activation 会 OOM），decode 默认 0.90。
2. **decode 强制 chunk-cache**：日志 `KV cache is forced as chunk cache for decode server`
   （decode 不做 radix 前缀复用，KV 是一次性从 prefill 传入的）。

### 2.3 gmu / VRAM 账（钱去哪了）

以 gmu=0.90 为例（per-rank/卡）：
```
卡 VRAM total            = 288.0 GiB
gmu × total = 0.90×288   = 259.2 GiB   ← "static"(weights+KVpool) 的目标上限
  ├─ 模型权重             ≈  11.1 GiB   (反算)
  └─ KV pool             = 118.9 GiB   ← 实测 available_bytes（本报告主角）
剩余 (1−gmu)×total        =  28.8 GiB   ← 留给 activations + CUDA graph
实测 VRAM 实占            = 245.3 GiB   (启动到 CUDA-graph capture 期采样)
启动后 avail mem headroom =  44.0 GiB   (还没被 activation/graph 完全吃满)
```
> 注：`avail mem`(44GB) > `(1−gmu)×total`(28.8GB) 是因为采样时 CUDA graph 尚未全 capture；
> 稳态下 headroom 会降到 ~5-8GB（官方建议保留量）。KV pool(118.9GB) 是启动即锁死的确定值。

---

## 3. 运行时验证（Step 3）：池启动即定、运行时不超池

单机 mix 模式 gmu=0.90 真跑：128 请求 @conc64、8192 in / 1024 out（与 P/D 同 KV 公式）。
观测 scheduler stat 全程：

| 指标 | 峰值 | 含义 |
|------|------|------|
| full-attention token usage | **1%** | c4/full 池几乎空 |
| **swa token usage** | **5%** | swa 池（较小=full×0.15）是更早触及的约束 |
| #running-req | 13 | 远低于 max_running 64/rank |
| **#retracted-req** | **0** | **全程无显存挤兑回撤** |

**证实核心命题**：gmu=0.90 启动锁定的 pool（full=9.22M, swa=1.38M/rank）在真实 8k/1k 负载下
运行时峰值只用 1–5%、retract=0。**KV pool 是静态的，运行时不增不减，gmu 是唯一决定其大小的旋钮**。
（附带印证参考运行 `pd_1p1d_dpa_8k1k` 的"decode 侧有大量余量"结论——池宽裕到运行时只碰 5%。）

---

## 4. 手动调节方法（实操指南）

### 4.1 主旋钮：`--mem-fraction-static`（gmu）
- **直接线性**：本机 DSv4-Pro **每 +0.01 → 每卡 KV pool +2.77 GB / +216K token 容量**。
  要 X GB KV pool → `gmu ≈ (X + 130.3) / 276.9`。
- **调高**（榨更多 KV，支撑更高并发/更长上下文）：看启动日志 `Memory pool end. avail mem=Y GB`，
  Y 太高（>10-20GB，稳态）就 +0.01 逐步逼近，直到 avail mem 降到 ~5-8GB 或首次 OOM 回退一格。
- **调低**（治 OOM）：见到 OOM/HSA_STATUS_ERROR/Aborted 就 −0.01。

### 4.2 role 非对称默认（P/D 分开调）
- **prefill 默认 0.85**：DP-attn prefill 高并发时 activation（chunked-prefill + prefill-delayer 批处理）
  吃掉 `(1−gmu)` 预留，0.90 会 HSA OOM → **prefill 要留更多动态余量 → 调低**。
- **decode 默认 0.90**：decode 只产 1 token/步，activation 小，可把更多显存给 KV pool → **调高**。
- **两方向诊断口诀**：`prefill 段 HSA OOM/Aborted → 调低`；`decode 段 retract/KV 不够 → 调高`。

### 4.3 与其他旋钮的相互作用
- `--max-running-requests`：并发上限。与 pool **独立**——pool 定 token 总容量，max-running 定请求槽数。
  DP-attn 下用户传的全局值 ÷dp_size 生效（512→64/rank）。pool 太小 + max-running 太大 → 运行时 retract。
- `--context-length`：单请求最大 token。不改 pool 总大小，但决定"pool 能装几个满上下文请求"
  （gmu0.90: full 9.22M ÷ ctx 9472 ≈ 973 个满上下文请求/rank）。
- `--page-size 256`：pool 按 page 对齐，token 容量向下取整到 256 的倍数。
- `--swa-full-tokens-ratio 0.15`：决定 swa 池占 full 的比例。**swa 池是运行时更早触顶的池**
  （§3 实测 swa 5% vs full 1%），长滑窗负载可适度调高此比。
- **kv-cache-dtype**：默认 `fp8_e4m3`（每 token 字节减半 vs bf16）。这是 DSv4 KV 极小的另一半原因
  （MLA 压缩 + fp8）。

### 4.4 换配置纪律
换 gmu/换轮：`pkill -9 -f sglang.launch_server` → 等两节点 VRAM≈0 → 再起下一个，否则残留 pool 叠加 OOM。

---

## 5. Environment + 复现

### 5.1 环境
| 组件 | 值 |
|------|----|
| 节点 | chi2879（8×MI355X gfx950, 288GiB/卡, EPYC 9575F, ROCm 7.2.0） |
| 镜像 | `infera/engine-sglang:pd-mcgate` |
| sglang / sgl_kernel / torch | 0.5.15.post1 / 0.4.4 / 2.9.1+rocm7.2.0 |
| 模型 | `/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro`（真权重, symlink→john, 需 `-v /mnt/vast:/mnt/vast`） |
| kv-cache-dtype | fp8_e4m3（默认） |
| 并行 | TP8 + DP-attn(dp8) + EP8, page-size 256, context-length 9472 |

### 5.2 一键复现（容器内）
```bash
# 单点读数（任意 gmu）：起 decode 单腿，抓 KV pool 分配行，kill
cd /mnt/vast/c_huggingface
ROLE=decode GMUS="0.90" OUT=/mnt/vast/c_huggingface/kv_repro bash kv_gmu_sweep.sh
# 关键行：grep "DSV4 memory calculation\|DSV4 pool sizes" kv_repro/decode_gmu0.90.log
# 期望 gmu0.90: available_bytes=118.93 GB, full_token=9217792, swa=1382656

# 全扫 + prefill 对照 + 真跑验证脚本，见 staging：
#   kv_gmu_sweep.sh        —— gmu 扫描（本报告 Step1/2）
#   run_mix_verify.sh      —— 单机 mix 真跑验证不超池（Step3）
```

### 5.3 产物位置（本 packup 内，相对本 README）
- `logs/decode_gmu0.8{0,5,8}.log.gz`, `decode_gmu0.9{0,2}.log.gz`（5 点 decode 原始 log, gzip）
- `logs/prefill_gmu0.85.log.gz`, `prefill_gmu0.90.log.gz`（2 点 prefill 对照）
- `logs/mix_verify_gmu0.90.log.gz` + `results/mix_verify_bench.log`（真跑验证）
- `results/decode_summary.csv`, `prefill_summary.csv`, `gmu_kvpool_master.csv`
- **脚本**：`scripts/kv_gmu_sweep.sh`, `scripts/run_mix_verify.sh`
- 原始 staging（远端, 只读留档）：`/mnt/vast/c_huggingface/kvcache_gmu_study_20260724_084928/`

### 5.4 依赖秘密（仅列名）
- Docker registry `infera` 登录 —— 团队 vault。
- 集群 SSH —— 跳板 `root@149.28.124.225`（chi2866），ProxyJump 到 chiXXXX。

---

## 附录 A：关键源码坐标（sglang v0.5.15.post1）
- `model_executor/model_runner_kv_cache_mixin.py:66-80` — `_profile_available_bytes`（gmu 门）
- `model_executor/pool_configurator.py:513/609/651/734` — `DSV4PoolConfigurator` / `_get_bytes_per_full_token` / `_compute_dsv4_sizes` / `calculate_pool_sizes`
- `mem_cache/deepseek_v4_memory_pool.py:449/113/283` — 池类 / 584B MLA assert / 132B indexer
- `mem_cache/kv_cache_configurator.py:1527` — `_profile_available_bytes` 顶层
- `server_args.py:1457` — mem_fraction_static 官方定义
- `docs/advanced_features/hyperparameter_tuning.md:28-52` — 官方调优指南

## 附录 B：与参考运行 pd_1p1d_dpa_8k1k(0.5.13) 的差异
- 0.5.13 打印单一 `max_total_num_tokens`（decode gmu0.90=3,895,296）；0.5.15 改为多池
  `DSV4 pool sizes`（full/swa/c4/c128），语义更细。老 `max_total` ≈ 新 `c4`×?（口径不同，勿直接比）。
- 0.5.13 `avail mem` gmu0.90≈34.6GB vs 本轮 44.0GB —— 版本/权重加载差异，佐证"必须实测、勿套旧数"。

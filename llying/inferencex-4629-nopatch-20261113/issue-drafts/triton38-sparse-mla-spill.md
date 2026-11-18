# Issue 草稿：Triton 3.8 在 gfx950 上编译 SGLang DSA sparse MLA decode kernel 时寄存器溢出

（草稿，未发布。下面两份分别给 Triton 和 SGLang；复现脚本是同目录的 `triton38-sparse-mla-repro.py`，原始数据在 AUS 的 `/apps/liyingli/holder-dsa/out/`。）

中文摘要：
- 同一份 SGLang kernel 源码，Triton 3.7（rocm720 镜像）编译的最优配置 (BLOCK_N 64, warps 4, stages 2) 每次 117 µs；Triton 3.8（rocm10 镜像）编译同一配置 191 µs。
- 所有 `num_stages=2` 的配置在 3.8 下 VGPR 用满 512 并大量溢出到 scratch（如 (64,4,2) 从溢出 8 个变成 373 个）；`num_stages=1` 的配置两版几乎相同。
- SGLang 的 autotune 在启动时用空数据、每个 rank 各自测，在 3.8 上选出的配置不稳定（262–361 µs），rank 间不一致还造成 NCCL 等待。
- 影响：GLM-5.2 MXFP4 PD decode（4 卡 DP attention + MTP）每步从约 60 ms 涨到 72–79 ms，AgentX 吞吐低约 10%。

---

## A. Triton（AMD 后端）

**Title:** [AMD][gfx950] Triton 3.8 regresses a software-pipelined (num_stages=2) attention kernel: VGPRs max out and spill (117 µs → 191 µs vs Triton 3.7)

**Summary**

SGLang's DSA sparse MLA decode kernel (`_sparse_mla_fwd_split_dim_kernel` in `python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py`) is ~1.6x slower when compiled with Triton 3.8 than with Triton 3.7 on MI355X (gfx950), for the same source and the same config. Every `num_stages=2` config now uses all 512 VGPRs and spills to scratch; `num_stages=1` configs compile to the same register footprint as before.

**Environment**

| | Old | New |
|---|---|---|
| Image | `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261113` | `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261112` |
| Triton | 3.7.0+amd.rocm7.2.0.git89002410 | 3.8.0+git4cff872c.rocm10.0.0 |
| PyTorch | 2.9.1+rocm7.2.0 | 2.11.0+rocm10.0.0 |
| GPU | MI355X (gfx950), amdgpu 6.19.14 | same node, same GPU model |

Kernel source is the same in both images for this path: the newer SGLang only adds a `KV_SCALED` constexpr branch, which is `False` here.

**Shape** (one decode step on one rank): 72 query rows, 64 heads, D_V 512 + D_TAIL 64, top-2048 indices into an fp8 (e4m3fn) KV pool `[2.2M, 1, 576]`, `USE_FP8_DOT=True`, direct-to-LDS async copy off (what SGLang sets on gfx950).

**Kernel time per launch (µs, cold cache, indices spread over the pool)**

| BLOCK_N, warps, stages | Triton 3.7 | Triton 3.8 |
|---|---|---|
| 64, 4, 2 | **117.6** | 191.4 |
| 32, 4, 2 | 147.8 | 166.4 |
| 64, 4, 1 | 201.8 | 201.7 |
| 32, 4, 1 | 336.1 | 337.9 |
| 32, 2, 2 | 257.1 | 263.2 |
| 64, 2, 2 | 538.0 | 563.1 |
| 64, 2, 1 | 308.9 | 279.6 |

The index layout (indices confined to a 16k-slot window vs spread over 2.2M slots) changes nothing, so this is not cache/TLB behaviour.

**Register usage from the compiled `.amdgcn` (vgpr / agpr / vgpr spills / scratch bytes)**

| BLOCK_N, warps, stages | Triton 3.7 | Triton 3.8 |
|---|---|---|
| 64, 4, 2 | 495 / 239 / 8 / 20 | 512 / 256 / **373** / 1460 |
| 32, 4, 2 | 390 / 134 / 0 / 0 | 512 / 256 / **176** / 708 |
| 64, 2, 2 | 512 / 256 / 427 / 1220 | 512 / 256 / **787** / 2384 |
| 32, 2, 2 | 512 / 256 / 179 / 636 | 512 / 256 / **521** / 1640 |
| 64, 4, 1 | 423 / 167 / 0 / 0 | 421 / 165 / 0 / 0 |
| 32, 4, 1 | 352 / 96 / 0 / 0 | 348 / 92 / 0 / 0 |

With async copy on, the same pattern holds (e.g. 32, 4, 2: 338/82/0/0 → 512/256/128/516).

**Repro**

`triton38-sparse-mla-repro.py` (needs an SGLang install for the kernel source; run in each image on one GPU):

```bash
python3 triton38-sparse-mla-repro.py --async-copy module   # SGLang's gfx950 setting (async copy off)
python3 triton38-sparse-mla-repro.py --async-copy on
```

It times all 12 configs the autotuner may pick and prints which one the autotuner selects for several index layouts. Set `TRITON_CACHE_DIR` to keep the `.amdgcn` for the register numbers above.

**Expected:** Triton 3.8 compiles the `num_stages=2` configs within the register budget, as 3.7 does, so that (64, 4, 2) stays near 117 µs.

---

## B. SGLang

**Title:** [AMD][gfx950] DSA sparse MLA decode is 1.6–2.4x slower on the ROCm 10 images (Triton 3.8 spills, autotune picks vary per rank)

**Summary**

On `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261112`, GLM-5.2-MXFP4 decode (4 GPUs, `--tp 4 --dp 4 --enable-dp-attention`, EAGLE 5 steps / 6 draft tokens, `--dsa-decode-backend triton`, fp8 KV) takes 72–79 ms per step against 60 ms on the ROCm 7.2 nightlies (20261101, 20261113, 20261117) at the same load: 12 requests per DP rank, ~2.1M KV tokens per rank. AgentX throughput on a 1P (TP1 PP4) + 1D (4-GPU DPA) disaggregated deployment drops by ~10%.

The difference is in `_sparse_mla_fwd_split_dim_kernel`, which serves target-verify steps once a rank has more than 64 query rows (≥ 11 requests with 6 verify tokens each, H = 64):

| | ROCm 7.2 nightly (Triton 3.7) | ROCm 10 nightly (Triton 3.8) |
|---|---|---|
| Per-launch time in the server profile | 113 µs | 267 µs (361 µs on two ranks in one run) |
| Autotuned config (all 4 ranks) | (64, 4, 2) | (32, 2, 2) / (64, 2, 1) / varies |
| Decode step at 12 req/rank, 2.1M tokens/rank | 59.7–60.8 ms | 71.6–78.6 ms |

Other kernels on the same path are unchanged (fp8 paged-MQA logits 255 vs 252 µs, top-k 56 vs 60 µs, MoE and FP4 GEMMs equal per call). The ROCm 7.2 nightlies already contain the same SGLang source (topk_v2 exact mode, the `KV_SCALED` change), so the SGLang changes are not the cause.

Two problems compound:

1. **Triton 3.8 spills on every `num_stages=2` config** (Triton issue: <link>). The best config on 3.8 is 166 µs vs 117 µs on 3.7.
2. **The autotune result depends on startup conditions.** It runs on the first call during startup, per rank, with placeholder inputs. With all-invalid indices the autotuner picks (64, 4, 1), and the servers picked (32, 2, 2) and (64, 2, 1), all 1.2–1.7x slower than the best config for real decode data. Ranks can pick different configs. In one run, two of the four DP ranks spent 228 ms in this kernel over the profiled steps against 169 ms on the other two, which then waited in NCCL.

**Suggested changes**

- Pin the split-dim config on gfx950 per Triton version, or autotune on synthetic valid indices into the KV pool rather than on placeholder data.
- Broadcast the autotune choice from rank 0 so that DP ranks run the same config.
- Until Triton is fixed, note in the ROCm 10 image docs that DSA models decode slower than on the ROCm 7.2 images.

`TRITON_HIP_USE_ASYNC_COPY=1` does not help: the autotuner then picks (64, 2, 1), about 278 µs.

**Repro**

Kernel level: see the Triton issue. Server level: launch the decode-role arguments above as a plain server on 4 GPUs, send 48 requests of 170k random tokens (`max_new_tokens` 30000, `ignore_eos`), and read the `Decode batch` log lines once prefill is over. Step time = `#running-req × accept len / gen throughput`.

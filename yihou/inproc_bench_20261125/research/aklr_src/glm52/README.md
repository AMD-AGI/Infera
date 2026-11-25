# GLM-5.2 model-level micro-bench

Times every operator on its own, sums them as `per-layer cost × layer count`, checks that sum against a full-forward measurement.

`ratio = roll-up / full forward`. Near 1 means nothing was missed — not that the split inside a layer is right.

`profile_decode.py` times one decode step. `profile_prefill.py` times every chunk of one request's TTFT.

## Usage

CI uses `lmsysorg/sglang:v0.5.22-rocm10-mi35x` with its bundled ROCm Triton 3.8.0. Keep that build: replacing it with the PyPI wheel crashes when Triton is imported after PyTorch. The benchmark publishes each worker's GPU and world rank for SGLang's process-local runtime context; older runner-owned parallel state remains supported.

The CI decode scenario uses `--full-step 128` to reduce short-window timing noise, retaining two on/off rounds and a throughput-ratio confidence lower bound of at least 1.01.

```bash
cd aklr/benchmarks/models/glm52     # required, the scripts do `import model_bench`
M=/path/to/GLM-5.2-MXFP4            # local checkout; the default is a hub id and a 408 GB fetch
```

```bash
# prefill
python3 profile_prefill.py --model $M --tp-size 8 --enable-dp-attention --dp-size 8 \
    --ISL 100000 --reps 20 [--prefix-cache-hit 0.97]

# decode
python3 profile_decode.py --model $M --tp-size 8 --enable-dp-attention --dp-size 8 \
    --ISL 100000 --CONC 64 --reps 20

# MTP -- without SGLANG_SIMULATE_ACC_LEN throughput says `not reported` and there is no ITL row
SGLANG_SIMULATE_ACC_LEN=3.61 \
python3 profile_decode.py --model $M --tp-size 8 \
    --speculative-algorithm EAGLE --speculative-num-steps 5 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
    --ISL 100000 --CONC 8 --reps 20

# EP -- mori is the only a2a backend that loads on ROCm here.
# Do NOT add --attention-backend aiter: it disables sparse attention for GLM-5.2.
SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=4096 MORI_SHMEM_MODE=ISOLATION \
python3 profile_decode.py --model $M --tp-size 8 --ep-size 8 \
    --enable-dp-attention --dp-size 8 --moe-a2a-backend mori \
    --moe-dense-tp-size 1 --enable-dp-lm-head --ISL 100000 --CONC 64 --reps 20

# without DP attention, add --mem-fraction-static 0.6: the undivided 16k chunk
# makes the indexer build a chunk x context score matrix, 6.5 GB at 100k
```

`ServerArgs` flags pass through, except that the bench forces `--disable-cuda-graph` (unless `--full-step`, since the hooks need sglang's graphs off) and rewrites `--kv-cache-dtype auto` to `fp8_e4m3`. Ours:

| flag | default | script | meaning |
| --- | --- | --- | --- |
| `--ISL` | 4096 | both | input length per request |
| `--reps` | 50 | both | graph replays per operator |
| `--repeat` | 1 | both | timing passes, fastest wins |
| `--out` | auto | both | report path |
| `--CONC` | 1 | decode | **total** concurrency, split across `dp_size`, must divide evenly |
| `--full-step N` | 0 | decode | time N whole steps under sglang's own graphs and exit |
| `--trace` | off | decode | print one step's kernels instead of timing operators |
| `--prefix-cache-hit` | 0 | prefill | fraction of `--ISL` already cached |

An MTP invocation runs the bench twice: the per-operator hooks need sglang's graphs off, and a step only looks like the server's with them on. `profile_prefill.py` refuses `--speculative-algorithm`.

## Reading a report

Rows are `kind|op`, summed as per-layer cost × layer count.

| row | contents |
| --- | --- |
| `attn.fused_qkv_a_proj_with_mqa` | q_a and kv_a are **one** GEMM, 6144 → 2624 |
| `attn.indexer` | 21 layers only; the rest reuse the previous layer's top-k |
| `attn.kv_b_proj` | prefill only, dense chunks only |
| `attn.<rest>` | qk_norm, rope, absorb bmms, kv write, attention kernel (**tilelang** by default on ROCm) |
| `mlp.experts` | MXFP4; under an a2a this row is comms + expert GEMMs |
| `mlp.<rest>` | `act_fn` on dense; shared/routed add and post-expert all-reduce on MoE |
| `layer.<rest>` | `LayerCommunicator` collectives and residual adds |
| `draft model` | MTP only, one row — the nextn layer has no per-layer multiplier to roll up |

A `<rest>` row is a **difference of two measurements, not a measurement**. Error in the parent or any child lands there at full size and reversed sign, so a small one is mostly that error and can read negative.

Two sections name what a report could not measure. **Coverage**: hooked but never fired, folded into a `<rest>` row — under plain TP decode the fused all-reduce+residual+RMSNorm path reads `layernorm.weight` directly and the norm rows vanish into `layer.<rest>`, all but layer 0's `input_layernorm`, which has no preceding all-reduce to fuse into. Prefill keeps both. **Not replayable**: fired but could not be timed alone. The table names where its cost landed -- a `<rest>` row, which is then an upper bound; the `draft model` row; or nowhere, leaving the roll-up short.

`ratio` catches missing work, not mismeasured work — a residual is a subtraction, so the roll-up balances either way. The external check is `bench_serving`'s **median** ITL at the same shape; +5% to +7% is expected. Every report carries a `bench` row in Versions, because two reports of the same shape and the same sglang can still disagree in the ranking.

`--load-format dummy` runs but its numbers cannot be acted on: the DSA top-k's cost is a function of the value distribution, and dummy weights invert the ranking. The report banners it and the filename carries a `dummy` tag.

## Model structure

| item | value |
| --- | --- |
| `architectures` | `GlmMoeDsaForCausalLM`, subclass of `DeepseekV2ForCausalLM`; the layers live in `models/deepseek_v2.py` |
| layers | 78: dense 0–2, MoE 3–77; layer 78 is the MTP draft |
| hidden / dense FFN / MoE intermediate | 6144 / 12288 / 2048 |
| experts | 256 routed, `topk 8`, 1 shared |
| MLA | `q_lora 2048`, `kv_lora 512`, `qk_nope 192`, `qk_rope 64`, `v_head 256`, 64 heads |
| indexer | `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048` |
| MXFP4 | routed and shared experts of layers 3–77 only; everything else BF16 |

| layer kind | ids | count |
| --- | --- | --- |
| dense + indexer | 0/1/2 | 3 |
| MoE + indexer | 6/10/…/74 | 18 |
| MoE (reuses topk) | the rest | 57 |

## AKLR workloads

The shapes this model runs are written by `python -m aklr export` from a run with `AKLR_EXPORT=1`; the scenarios that have a kernel are checked in under `aklr/ops/<op>/impls/<language>/` (`aklr/ops/README.md`).

## TODO

- [ ] decompose the `draft model` row — three draft ops cannot be replayed standalone (batch-size mismatch)
- [ ] a check that can catch a wrong per-operator row; `ratio` provably cannot
- [ ] split `mlp.experts` into dispatch / experts / combine under an all-to-all

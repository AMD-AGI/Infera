# DeepSeek-V4 model-level micro-bench

Times every operator on its own, sums them as `per-layer cost × layer count`, checks that sum against a full-forward measurement.

`ratio = roll-up / full forward`. Near 1 means the totals agree — not that every operator was covered or the split inside a layer is right.

`profile_decode.py` times one decode step. `profile_prefill.py` times every chunk of one request's synthetic TTFT. Both scripts support DeepSeek-V4-Flash and DeepSeek-V4-Pro; other architectures are rejected before loading weights.

## Usage

```bash
cd aklr/benchmarks/models/deepseekv4     # required, the scripts do `import model_bench`
M=/path/to/DeepSeek-V4-Flash             # or DeepSeek-V4-Pro; use local weights to avoid a download
export AITER_BF16_FP8_MOE_BOUND=16        # activation dispatch threshold; FP4 expert weights stay FP4
```

```bash
# prefill
python3 profile_prefill.py --model "$M" --tp-size 8 --enable-dp-attention --dp-size 8 \
    --ISL 100000 --reps 50
# add --prefix-cache-hit 0.97 to start with a cached prefix

# decode
python3 profile_decode.py --model "$M" --tp-size 8 --ISL 100000 --CONC 1 --reps 50 \
    --max-running-requests 1 --swa-full-tokens-ratio 0.01 --mem-fraction-static 0.6

# EP with attention DP, using mori on ROCm
SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 MORI_SHMEM_MODE=ISOLATION \
python3 profile_decode.py --model "$M" --tp-size 8 --ep-size 8 \
    --enable-dp-attention --dp-size 8 --moe-a2a-backend mori \
    --moe-dense-tp-size 1 --enable-dp-lm-head --ISL 100000 --CONC 64 --reps 50 \
    --max-running-requests 64 --swa-full-tokens-ratio 0.01 --chunked-prefill-size 256
```

The default model is `deepseek-ai/DeepSeek-V4-Flash`; pass `--model deepseek-ai/DeepSeek-V4-Pro` for Pro. The runtime target is `lmsysorg/sglang:v0.5.19-rocm724-mi35x`; these scripts use its native SGLang APIs. Each layout still needs sufficient KV capacity. `--max-running-requests` is global, before DP division; `--swa-full-tokens-ratio` controls native pool sizing. Keep the pool and activation-dispatch settings identical when comparing runs.

`ServerArgs` flags pass through, except that the bench forces `--disable-cuda-graph` (unless `--full-step`, since the hooks need SGLang's graphs off) and rewrites `--kv-cache-dtype auto` to `fp8_e4m3`. Ours:

| flag | default | script | meaning |
| --- | --- | --- | --- |
| `--ISL` | 4096 | both | positive input length per request |
| `--reps` | 50 | both | positive graph replay count per operator |
| `--repeat` | 1 | both | positive timing passes, fastest wins |
| `--out` | auto | both | report file path |
| `--CONC` | 1 | decode | positive **total** concurrencies, split across attention DP ranks; several values share one model load |
| `--full-step N` | 0 | decode | time synthetic forwards under SGLang's graphs; fastest of N+2 samples, then exit |
| `--trace` | off | decode | export one eager forward's kernel trace instead of timing operators; one `--CONC` only |
| `--export-only` | off | decode | one eager forward per CONC for `AKLR_EXPORT`; no timing, no report |
| `--prefix-cache-hit` | 0 | prefill | fraction of `--ISL` already cached |

Both scripts reject `--speculative-algorithm`; there is no MTP path. `--full-step` measures synthetic forwards, not a serving scheduler, sampling loop or end-to-end ITL.

## Reading a report

Rows are `kind|op`, summed as per-layer cost × layer count. Global embedding, final norm and logits rows run once per forward.

| row | contents |
| --- | --- |
| `attn.wqkv_a` | fused query/KV projection; `attn.wq_a` and `attn.wkv` when split |
| `attn.wq_b` / `attn.wo_b` | query and output projections |
| `attn.indexer` | C4 layers only; selects compressed tokens per query |
| `attn.<rest>` | qk_norm, rope, compressor, direct `wo_a` matmul, KV write and attention kernel |
| `mlp.experts` | expert GEMMs; under an all-to-all this row includes dispatch and combine |
| `mlp.<rest>` | shared/routed add and any post-expert collective |
| `layer.<rest>` | mHC pre/post operations and decoder collectives |

A `<rest>` row is a **difference of two measurements, not a measurement**. Error in the parent or any child lands there at full size and reversed sign, so a small one is mostly that error and can read negative.

Two sections name what a report could not measure. **Coverage**: hooked but never called — for example, direct `wo_a` weight reads or compressor helper calls bypass their modules and remain inside the attention parent. Missing globals have no containing layer residual. **Not replayable**: called but could not be timed alone; the table names the containing residual or marks the work as unattributed. The model-level mHC head has no separate timing row.

Decode reports also list Kineto kernels below each Hotspots row: calls, GPU time, percentage of that module's GPU time, and the Python/dispatcher API. These come from a separate eager replay after graph timing. Launch correlations establish ownership; hooks exclude separately measured children from parent samples. Missing correlations are reported, not guessed. The kernel columns never replace or redistribute graph timings.

KV is randomized instead of populated by a real prefix forward, and expert routing is replaced with seeded balanced choices, including hash layers. These are **synthetic performance measurements, not numerical correctness evidence**. Decode retains the full/compressed context but allocates only the live, page-aligned SWA tail through SGLang's native allocator. Both pools reserve space for the next decode token; insufficient capacity fails with `[BLOCKED_CAPACITY]`, without reducing ISL or concurrency.

Inputs are recorded before in-place mutation, outside the timing clocks; tensor snapshots may increase preparation memory. Native batch metadata stays shared and scoped MoE flags are restored per replay. Parent decode rows cycle through real layers of the same kind; small matrix weights rotate through a buffer larger than MI355X's last-level cache.

**Top kernels** ranks kernel groups by their share of the step, each grouped under its innermost AKLR-routed API; `target` marks the top 5.

`ratio` is a cross-check, not a completeness proof: a residual is a subtraction, so the roll-up can balance despite missing or mismeasured child rows. Synthetic forward throughput is not serving throughput. Every report carries source/version information so measurements from different implementations can be distinguished.

`--load-format dummy` runs but its numbers cannot be acted on: indexer and routing costs depend on values. The report banners it and the filename carries a `dummy` tag.

## Model structure

`DeepseekV4ForCausalLM`, implemented in SGLang's `models/deepseek_v4.py`. Every backbone layer is MoE. Attention uses MQA with a sliding window and compression ratios `0`, `4` or `128`; C4 layers carry the indexer. Layer counts exclude nextn entries.

| item | Flash | Pro |
| --- | --- | --- |
| layers | 43 | 61 |
| hidden size | 4096 | 7168 |
| routed experts / top-k | 256 / 6 | 384 / 6 |
| indexer top-k | 512 | 1024 |
| weights | FP8 block-scaled projections, FP4 routed experts | FP8 block-scaled projections, FP4 routed experts |

| layer kind | Flash count | Pro count |
| --- | --- | --- |
| `moe+hash+c0` | 2 | — |
| `moe+hash+c128` | — | 2 |
| `moe+hash+c4+indexer` | 1 | 1 |
| `moe+c128` | 20 | 29 |
| `moe+c4+indexer` | 20 | 29 |

The first three layers route by hash; the rest use a learned gate. The benchmark's balanced-routing override applies to both. Actual quantization and parallel layout are recorded in the report rather than inferred from the checkpoint name.

## AKLR schema export

Record every call that reaches a AKLR operator (an `@api` function, or an AITER entry point bound by `aklr_proxy`) during a run, then export the schemas:

```bash
export AKLR_EXPORT=1
export AKLR_EXPORT_DIR=/tmp/aklr_export   # default: ./aklr_export, relative to the cwd above
python -m aklr export --language flydsl   # from the repository root, after the run
```

See the workflow in `aklr/ops/README.md`. Native SGLang/AITER calls that do not go through AKLR are not recorded.

## Decode sweep

ISL 100000, decode only, for Flash or Pro. Each CONC writes its own report; `[UNSUPPORTED]` marks a CONC not divisible by the DP size, `[BLOCKED_CAPACITY]` one that does not fit.

```bash
export AITER_BF16_FP8_MOE_BOUND=16 MORI_SHMEM_MODE=ISOLATION SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
MORI="--moe-a2a-backend mori --moe-dense-tp-size 1 --enable-dp-lm-head --chunked-prefill-size 256"
for cfg in "--tp-size 8" "--tp-size 8 --enable-dp-attention --dp-size 8" \
           "--tp-size 8 --ep-size 8" "--tp-size 8 --ep-size 8 --enable-dp-attention --dp-size 8 $MORI"; do
    AKLR_EXPORT=1 python3 profile_decode.py --model "$M" --load-format dummy $cfg \
        --ISL 100000 --CONC 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 --reps 10 --repeat 2
done
```

Use `--tp-size 4` / `--dp-size 4` for the 4-GPU layouts. On Pro, aiter's `ck_moe_stage1` fails at M < 16 for `tp4`, `tp4_dp4` (CONC 8) and `ep8`, `ep4` (CONC 4, 8); on Flash, for `ep8`, `ep4` (CONC 4, 8). Skip those points.

To record Points only, replace `--reps 10 --repeat 2` with `--export-only`. It runs one eager forward per CONC; timing and profiling only replay that forward's inputs, so they add no Point.

## TODO

- [ ] give the model-level mHC head its own timing row
- [ ] add an independent check of per-operator attribution; `ratio` cannot prove it
- [ ] split `mlp.experts` into dispatch / experts / combine under an all-to-all

## Motivation

On HIP, DSA models (GLM-5.x, DeepSeek-V3.2) capture two decode graphs per batch size: dense (the indexer only writes K when every sequence fits in `index_topk`) and sparse (`DsaGraphVariants`). Each rank picks its variant from its own longest sequence, and an idle rank always picks sparse.

Under DP attention the ranks replay their graphs in lockstep, and the collectives captured inside them pair up across ranks. Some of these collectives register their graph's input buffers (aiter all-gather / reduce-scatter / custom all-reduce). When two ranks replay different variants in the same step, a rank reads its peer's buffers from the other graph: stale data. On GLM-5.2-MXFP4, MI355X, PD decode TP4 + DP4 attention with the MoE gather / combine inside the graphs, 26 of 200 GSM8K prompts came back garbled. Pinning every rank to one variant removed all of them, which confirmed the cause. It also shows up with only the custom all-reduce in the graph (first token right, then noise: `Theutzer1-case55500The $500 + 800 = 1300 ...`): on current `main` with the stock image's defaults, aggregated TP4 / DP4 attention with MTP returns 153 of 1,200 GSM8K prompts garbled (Accuracy Tests).

Two further observations from the code:
- The dense variant only changes the decode / idle indexer (`_should_skip_logits_computation()`); a target-verify capture runs the full indexer in both, so MTP deployments capture the verify graphs twice for nothing.
- On the verify path `select()` has no host lengths and syncs `seq_lens.max()` to the host before every replay.

## Modifications

- `graph_variants.py`: `create_attention_graph_variants()` takes the capture forward mode and builds the dense / sparse pair only for decode captures (`decode_cuda_graph_runner.py` passes it). Target-verify captures hold one graph.
- The MLP sync row (`dp_attn.MLPSyncBatchInfo`) gains a column: an upper bound of the rank's longest decode sequence, read on the host only (the exact `seq_lens_cpu` max when the mirror exists, else committed lengths plus two steps' width). The maximum over ranks rides `ScheduleBatch` / `ForwardBatch` as `dp_decode_graph_max_seq_len`, following the existing `dp_prefill_cuda_graph_max_prefix_len`.
- `dp_decode_max_seq_len()` returns that bound in decode / idle, and `DsaGraphVariants` and `SeqLenBucketGraphVariants` select from it.
- No new collective, no device read.
- Unit tests: `test/registered/unit/layers/test_graph_variants_dp_consistency.py` (12 tests, CPU). The bound's path to the runner is checked by calling `ScheduleBatch.copy()`, `ForwardBatch.init_new()` and the TBO `filter_batch()`.

Scope and behavior changes:
- Covered: decode and idle steps of `DsaGraphVariants` (HIP only) and `SeqLenBucketGraphVariants` (aiter backend). GLM-5.2 and DeepSeek-V3.2 use the DSA backend, so the end-to-end runs below exercise `DsaGraphVariants`; the aiter buckets follow the same rule and are covered by the unit tests.
- Not covered, follow-up: the DSV4.1 candidate variants (`Dsv41CandidateGraphVariants`, SM100+ and gfx95) keep the per-rank rule. Doing them right needs the DSpark target verify too (under DP attention it does not stage its metadata early, and an idle rank in the verify runner stays `IDLE`), so decode and verify should move together. The NVIDIA megakernel's verify / draft variants are untouched as well.
- Without the MLP sync the per-rank rule is unchanged. With the sync but a single DP rank (a gathered-buffer MoE backend), the bound is that rank's own maximum: exactly the old value when the host mirror exists, otherwise committed lengths plus two steps, which is slightly more conservative near `index_topk` and skips a `.item()`.
- NVIDIA: graph selection is unchanged. The only cross-platform change is one more int64 in the MLP-sync row under DP attention, read by the two classes above.

Why sync a bound rather than always replay the general graph under DP attention: the +1.3 % below comes mostly from dropping the `.item()` before verify, not from the dense variant. For `SeqLenBucketGraphVariants` (aiter's 16 / 128 / max-partition buckets), always taking the max-partition fallback costs more on short contexts. The bound is one more int64 per rank in the existing all-gather. The trade-off is that a short-context rank follows the longest rank's variant.

## Accuracy Tests

The same change on a v0.5.22-based build (GLM-5.2-MXFP4, MI355X), PD 1 prefill (TP1 x PP4) + 1 decode (TP4 / DP4 attention) across two nodes over RDMA, real MTP acceptance. Controls first, then the fix. The prefill device pool was capped so that evictions run continuously, the condition in which the low-rate failure appeared.

| Arm | Prompts | Bad start | Garbled |
|---|---|---|---|
| Control, image as is (2 runs) | 400 | 6 | 20 |
| Fix (3 runs) | 600 | 0 | 0 |
| Control, MoE gather / combine in-graph | 200 | 13 | 53 |
| Fix, MoE gather / combine in-graph (2 runs) | 400 | 0 | 0 |
| Control, no MTP (decode-mode graphs only) | 200 | 3 | 33 |
| Fix, no MTP (2 runs) | 800 | 0 | 0 |

- An audit build of the runner (one device read per replay, mechanism check only) over 21,504 replays: the ranks picked the same variant on every replay (the per-rank rule would have disagreed on 1,644), the bound was never below the true maximum, and dense was still used (3.7 %).
- Uncapped production config, 1,319 GSM8K prompts: 0.966 / 0.969 (the same as stock SGLang with the official FP8 checkpoint on the same prompts, 0.967), 0 garbled, 0 bad start. lm-eval GSM8K strict / flexible 0.971 / 0.973 and 0.967 / 0.970. Long context (16k, 100k needle and long-prefix GSM8K) at the control's level.

On current `main`: this branch at `959f88ab` (merged with main at `b1a02a53`) against `b1a02a53`, both run from the stock `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261117` image with `PYTHONPATH` on the checkout's `python/` (nothing else in the image changed). GLM-5.2-MXFP4 on one MI355X node, aggregated TP4 / DP4 attention, EAGLE 3/1/4, custom all-reduce on (the default). `main` and this branch run side by side on GPUs 0-3 and 4-7, swapped between runs; each server takes 400 GSM8K prompts (64 in flight) and 64 long-context prompts of about 14k tokens (32 in flight) at once. Two configs: the image's defaults (`SGLANG_USE_ROCM700A=1`, the DP gather / combine is an in-graph custom all-reduce) and `SGLANG_USE_ROCM700A=0` (in-graph aiter all-gather / reduce-scatter). Three runs each:

| Config | Arm | GSM8K (1,200) | Bad start / garbled | Long-prefix GSM8K (96), garbled | Needle (96) |
|---|---|---|---|---|---|
| Defaults | `main` | 1,025 (0.854) | 6 / 153 | 86, 7 | 96 |
| Defaults | this PR | 1,174 (0.978) | 0 / 0 | 94, 0 | 96 |
| `SGLANG_USE_ROCM700A=0` | `main` | 1,080 (0.900) | 9 / 118 | 80, 11 | 96 |
| `SGLANG_USE_ROCM700A=0` | this PR | 1,174 (0.978) | 0 / 0 | 92, 0 | 96 |

On `main` every rank's target-verify runner captures the DSA dense / sparse pair (4 `DSA dual-graph enabled` lines per server); this branch captures one verify graph. The failure on `main` needs no PD deployment or in-graph aiter collectives: the default custom all-reduce is enough.

An audit build of the runner (one device read per replay, mechanism check only) on the 11-11 merge with `main` (`c7b9aa66`), no MTP, in two runs:

| Load | Replays aligned across 4 ranks | Rank disagreements (per-rank rule would have) | Dense | Bound below device max | Dense with a row over `index_topk` | GSM8K |
|---|---|---|---|---|---|---|
| GSM8K + long context | 17,920 | 0 (468, 2.6 %) | 7 | 0 | 0 | 392/400 (0.980), 0 bad start, 0 garbled |
| GSM8K only, 600 prompts | 19,712 | 0 (448, 2.3 %) | 686 (3.5 %), the same replays on every rank | 0 | 0 | 587/600 (0.978), 0 bad start, 0 garbled |

Every replay had a bound, idle ranks included.

## Speed Tests and Profiling

Same two nodes as the v0.5.22-based accuracy runs, AgentX trace replay, concurrency 104, 1 hour, simulated acceptance 2.99:

- Unchanged otherwise: 50,797 -> 51,477 tok/s/GPU (+1.3 %, single runs, about +-1 % run to run). The GPU idle before the verify graph drops from 0.59 to 0.03 ms per step (the `.item()` is gone).
- With the MoE gather / combine inside the graphs, which this makes correct: 53,609 tok/s/GPU (+5.5 %).

Single node, the same side-by-side setup as the `main` accuracy runs (two runs each, slots swapped), `python -m sglang.benchmark.serving` from the image, random prompts cut from ShareGPT, real acceptance (3.7-4.0):

| Config | ISL / OSL, concurrency | Output tok/s, `main` -> this PR | Mean TPOT ms, `main` -> this PR |
|---|---|---|---|
| Defaults | 1k / 1k, 64 | 3,167 -> 3,232 (+2.0 %) | 19.14 -> 18.79 |
| Defaults | 8k / 1k, 32 | 956 -> 963 (+0.7 %) | 30.22 -> 29.92 |
| `SGLANG_USE_ROCM700A=0` | 1k / 1k, 64 | 3,223 -> 3,303 (+2.5 %) | 18.63 -> 18.23 |
| `SGLANG_USE_ROCM700A=0` | 8k / 1k, 32 | 962 -> 986 (+2.5 %) | 30.01 -> 29.29 |

Means of two runs; single runs vary by about +-1 %.

## Checklist

- [x] Format your code (`ruff format`, `ruff check`, `isort` pass on the changed files).
- [x] Add unit tests.
- [ ] Update documentation (no user-facing option).
- [x] Provide accuracy and speed benchmark results.

<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:white_check_mark: [Run #43080525038](https://github.com/sgl-project/sglang/actions/runs/43080525038)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #43080524836](https://github.com/sgl-project/sglang/actions/runs/43080524836)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #43080525019](https://github.com/sgl-project/sglang/actions/runs/43080525019)<!-- slot:pr-test-amd:end -->

> [!TIP]
> Base CI only covers the core cases, and its workload may not exercise your change. Use `/rerun-test` to run the tests your change touches, or copy this prompt for an agent:
>
> ```text
> Find the registered tests this PR affects and run them with /rerun-test, following https://github.com/sgl-project/sglang/blob/main/.agents/skills/rerun-test/SKILL.md
> ```
<!-- pr-states:end -->



## Motivation

On HIP, DSA models (GLM-5.x, DeepSeek-V3.2) capture two decode graphs per batch size: dense (the indexer only writes K when every sequence fits in `index_topk`) and sparse (`DsaGraphVariants`). Each rank picks its variant from its own longest sequence, and an idle rank always picks sparse.

Under DP attention the ranks replay their graphs in lockstep, and the collectives captured inside them pair up across ranks. Some of these collectives register their graph's input buffers (aiter all-gather / reduce-scatter / custom all-reduce). When two ranks replay different variants in the same step, a rank reads its peer's buffers from the other graph: stale data. On GLM-5.2-MXFP4, MI355X, PD decode TP4 + DP4 attention with the MoE gather / combine inside the graphs, 26 of 200 GSM8K prompts came back garbled. Pinning every rank to one variant removed all of them, which confirmed the cause. The low-rate version (first token right, then noise: `Theutzer1-case55500The $500 + 800 = 1300 ...`) also shows up with only the custom all-reduce in the graph.

Two further observations from the code:
- The dense variant only changes the decode / idle indexer (`_should_skip_logits_computation()`); a target-verify capture runs the full indexer in both, so MTP deployments capture the verify graphs twice for nothing.
- On the verify path `select()` has no host lengths and syncs `seq_lens.max()` to the host before every replay.

## Modifications

- `graph_variants.py`: `create_attention_graph_variants()` takes the capture forward mode and builds the dense / sparse pair only for decode captures (`decode_cuda_graph_runner.py` passes it). Target-verify captures hold one graph.
- The MLP sync row (`dp_attn.MLPSyncBatchInfo`) gains a column: an upper bound of the rank's longest decode sequence, read on the host only (the exact `seq_lens_cpu` max when the mirror exists, else committed lengths plus two steps' width). The maximum over ranks rides `ScheduleBatch` / `ForwardBatch` as `dp_decode_graph_max_seq_len`, following the existing `dp_prefill_cuda_graph_max_prefix_len`.
- `dp_decode_max_seq_len()` returns that bound in decode / idle, and `DsaGraphVariants`, `SeqLenBucketGraphVariants` and `Dsv41CandidateGraphVariants` select from it. A DSpark target verify keeps its own lengths, since its metadata is staged for that variant before the replay. Without the sync (no DP attention) the per-rank rule is unchanged.
- No new collective, no device read.
- Unit tests: `test/registered/unit/layers/test_graph_variants_dp_consistency.py` (11 tests, CPU).

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

## Speed Tests and Profiling

Same two nodes, AgentX trace replay, concurrency 104, 1 hour, simulated acceptance 2.99:

- Unchanged otherwise: 50,797 -> 51,477 tok/s/GPU (+1.3 %, single runs, about +-1 % run to run). The GPU idle before the verify graph drops from 0.59 to 0.03 ms per step (the `.item()` is gone).
- With the MoE gather / combine inside the graphs, which this makes correct: 53,609 tok/s/GPU (+5.5 %).

Limits: validated end to end on the v0.5.22-based build. On current `main` the patch applies with offsets, the 11 unit tests pass and the files format clean, but I could not run the end-to-end benchmark on `main`.

## Checklist

- [x] Format your code (`ruff format`, `ruff check`, `isort` pass on the changed files).
- [x] Add unit tests.
- [ ] Update documentation (no user-facing option).
- [x] Provide accuracy and speed benchmark results.

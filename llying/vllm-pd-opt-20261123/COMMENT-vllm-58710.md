# Draft comment for vllm-project/vllm#58710 (not posted; the user decides)

Hardware data for this change, from GLM-5.2-MXFP4 PD serving on MI355X (gfx950), with the equivalent patch applied to
v0.34.0 (where the post-recompute `synchronize()` from #47404 is still present, so our overlay also drops it, as #66400
already did on main). Single runs; numbers and scripts are in our branch.

**Setup.** One node, 1P1D: prefill TP1 x PP4 (GPUs 0-3), decode DP4 TP1 with MoE flattened to TP4 (GPUs 4-7), MTP with
`num_speculative_tokens=3` (synthetic acceptance 2.99 for the throughput runs), fp8_e4m3 KV, `ROCM_AITER_MLA_SPARSE`,
FULL_AND_PIECEWISE graphs, async scheduling; AgentX agentic replay (InferenceX), 104 concurrent sessions, contexts of
10K-950K tokens (mean ~117K), 3600 s.

**What the sync costs here.** Under MTP the drafter rebuilds this backend's metadata eagerly for every draft step (the
backend has no `update_draft_decode_metadata`), so a decode step runs `build()` four times (target + 3 drafter steps)
and pays four blocking `seq_lens.cpu()` copies. Torch-profiler traces of all 4 decode ranks:

| | without the patch | with the patch |
|---|---|---|
| decode GPU busy | 83-92% | 99.7% |
| GPU idle per step | 4-8 ms | ~0.15 ms |
| blocking `hipMemcpyWithStream` per step | 4 (~34 ms of host wait) | 0 |
| host lead over the GPU at graph launch (p50) | 0.06 ms | 76 ms |

py-spy of the decode worker: 93% of samples on the `seq_lens[:num_reqs].cpu()` line before; after, the worker waits for
the next schedule and the GPU is the bottleneck.

**Effect.** Decode FULL steps on the same node at the same busiest-rank batch: 1-5 ms shorter (e.g. 41 vs 45 ms at 12
requests per rank, 45 vs 47 ms at 16, 52 vs 54 ms at 25). End to end (same node, same config, 3600 s, one run each):
45,337 -> 49,476 tok/s/GPU (+9.1%), TTFT mean 6.56 -> 4.60 s, ITL p50 16.9 -> 16.3 ms.

**Correctness.** With real MTP acceptance: GSM8K 198/200 (no garbled / bad starts), 16K-token needle 32/32 and
long-prefix GSM8K 32/32, mean acceptance length 3.46. A CPU check of the fast path against the synced values on 20,000
random batches (contexts 1-1M, query 1 or 4, padded rows, upper bounds overshooting by 0-5) found no mismatch; the fast
path fired on 71% of them.

**One note on the slack.** For drafter builds the upper bound is `target upper bound + step`
(`spec_decode/speculator.py:500-508` on v0.34.0), so the overshoot can exceed `num_speculative_tokens` by the step index;
we used a fixed slack of 32, which costs nothing at long contexts. Happy to run the PR branch itself on MI355X if useful.

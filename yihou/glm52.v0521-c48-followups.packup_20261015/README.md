# GLM-5.2-MXFP4 v0.5.21 same-node P4D4 — C48 follow-ups: decode tokenizer workers (tw90), correctness on the best config (cx90), write_through_selective at C64 (wts64) — 2026-10-14

Completes the items listed as "in flight" in `../glm52.v0521-prefill-mem-hicache-policy.packup_20261014/README.md`.
That kit has the pf90 / wts48 baselines these rounds compare against. pf94/pf93 were never run (dryrun only, in that kit).

## Questions (user, 2026-10-14)

1. **tw90.** "decode侧如果是瓶颈的话，往往是tokenizer worker不够的情况，在我们现在性能的配置上，decode加 --tokenizer-worker-num 4, 还有其他的必要参数，测一下性能是不是有提升"
   - Setup: the pf90 config plus decode `--tokenizer-worker-num 4`, on the same node as pf90 (089).
   - Reading of the v0.5.21 code (ledger 10-14 08:0x): the flag is rejected only with `--grpc-port` or Elastic EP. `detokenizer_worker_num` was left at its default 1, to change one variable.
2. **cx90.** "插空可以在我们现在最高的吞吐的配置上做一些正确性的测试，比如smoke test，带template和top p正确配置的needle test或者长文本test"
   - Setup: the pf90 config with simulated acceptance OFF. Real EAGLE acceptance; the dryrun diff is only the three `SGLANG_SIMULATE_ACC_*` envs.
3. **wts64.** "直觉上write_through——selective应该支持更高的conc, 测一下wts64"
   - Setup: C64 on the live wts48 stack (136; HiCache write_through_selective). It was the 3rd bench on that stack, after a partial A1 and A2 C48.

## Results (AgentX, score = `request_metrics.throughput.per_gpu.total_tput_tps`; `results/results.csv`)

| run | node | score | qps | ISL | TTFT mean/p95 (s) | TPOT (ms) | GPU hit | CPU hit | err |
|---|---|---|---|---|---|---|---|---|---|
| pf90 A1 (ref, fresh) | 089 | 21,859 | 1.219 | 142,517 | 9.57/31.32 | 15.7 | 0.911 | 0.035 | 0 |
| **tw90 A1** (fresh) | 089 | **22,231** | 1.235 | 143,101 | 8.63/29.11 | 15.4 | 0.914 | 0.035 | 4 |
| pf90 A2 (ref, scored) | 089 | 23,127 | 1.276 | 143,964 | 9.30/28.76 | 13.1 | 0.860 | 0.071 | 3 |
| **tw90 A2** (scored) | 089 | **22,597** | 1.243 | 144,451 | 10.94/34.84 | 12.5 | 0.871 | 0.056 | 4 |
| wts48 A2 (ref, C48) | 136 | 21,226 | 1.169 | 144,218 | 13.63/42.03 | 12.5 | 0.868 | 0.039 | 1 |
| **wts64** (C64) | 136 | **8,681** | 0.572 | 120,479 | 107.15/307.53 | 9.2 | 0.428 | 0.185 | 3 |

- **tw90**: +1.7 % at A1 and −2.3 % at A2 vs pf90 on the same node. These are single replicates with opposite signs, so no consistent effect is resolved.
- **wts64**: AgentX warm-up alone took 5,354 s at C64 (707 requests, 07:17 → 08:46). The profiled 3600 s then ran 08:46:39 → 09:53.
- **cx90**: 22/23 PASS.
  - The FAIL is `needle_128k_d10` (sampled): `finish=length` with empty content. `reasoning_tokens` 4101 exhausted `max_tokens` 4096 while the reasoning was still re-quoting the haystack, so no answer was emitted.
  - The suite passes the other 22 tests: smoke ×6 greedy+sampled, needle 8k-256k × d10/50/90, and 5-fact 100k ×2.
  - The first 256k prefill took 470 s, on a cold cache.

## Navigate

- `REPRODUCE.md` — env files, launch, read-out.
- `environment.md` — nodes/image/versions.
- `notes.md` — gotchas: the wts64 launch confusion, flush 400, stack reuse.
- `results/` — `results.csv`; bench dirs `tw90-c48-A1/A2`, `wts48-c64-A2` (= wts64), `pf90-c48-A1/A2` (ref); `correctness/cx90/`; `rounds/{tw90,cx90,wts48}/`; `dryruns/` + `tokdiff.*`; `metrics/`.
- `logs/server/{tw90,cx90}/` — server logs (gzip). wts48/wts64 server logs are node-local on 136 and were not reachable.
- `analysis/ledger/` — `poll.md`, `working_process.md` as of 2026-10-15 ~03:10.

---
name: infera-quality-gate
description: Prove a deployment's output is correct (smoke, GSM8K, garble and long-context probes, real MTP acceptance, RDMA, PD handoff) before any performance claim
---

# infera-quality-gate

A configuration that is fast because it is wrong is the most common false win. Run the gate on every new
configuration (and after every patch) on a stack with **real** speculative acceptance: refuse to gate a run with
`SGLANG_SIMULATE_ACC_LEN` or `MTP_AL` set (forced acceptance garbles output by design).

| Gate | How | Pass |
|---|---|---|
| Smoke | the kit's `engine/smoke.sh` / `smoke.py` (P/D registered, completion, chat, stream ends with `[DONE]`, a ~66K-token needle cold and reused) | 6/6 |
| GSM8K | `engine/eval.sh` (5-shot, greedy; `EVAL_LIMIT=200`, the full 1319 for a final claim) | >= 0.96, and within 0.01 of the baseline |
| Garble | greedy probes; look for template tags (`</think>`) in the content and repetition loops | 0 |
| Long context | a 16K-token needle (top-k sparse attention only engages past 2048 tokens; GSM8K never reaches it) | found |
| MTP acceptance | SGLang `accept len` median in the decode log | 2-3; exactly 4.00 means a degenerate loop |
| PD handoff | decode-side external prefix hit ~100%, decode prompt throughput ~0 | else decode recomputes the prompt |
| KV transport | `installTransport, type=rdma` in the logs, no `MC_FORCE_TCP` | RDMA |
| Kernel / comm change | the numeric check next to the change (e.g. a reference-vs-new comparison in eager and graph mode) | within tolerance |

Write the outcome to `results/<label>.gate.json` (any shape; it is evidence, not a ledger row) and name it in the
report. A failed gate stops the performance work: bisect the cause variable by variable with temperature-0 probes
(known causes: custom all-reduce, unfused NextN, prefill without MTP, DP2xDCP2, a half-applied patch).

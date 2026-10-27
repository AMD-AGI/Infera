---
name: infera-quality-gate
description: Prove a deployment's output is correct before any performance claim — GSM8K, converted from lm_eval's per-question samples and judged by intervals, is the one gate that must pass; smoke, garble and long-context probes, MTP acceptance, RDMA and PD handoff are for finding the cause
---

# infera-quality-gate

A configuration that is fast because it is wrong is the most common false win. Accuracy is judged on GSM8K alone,
on a stack with **real** speculative acceptance: refuse to gate a run with `SGLANG_SIMULATE_ACC_LEN` or `MTP_AL` set
(forced acceptance garbles output by design; the kit's `engine/eval.sh` refuses it too).

## GSM8K: the gate

Run the kit's `engine/eval.sh` (lm_eval, 5-shot multi-turn chat, greedy, `--log_samples`: every question's outcome is
in its `samples_gsm8k_*.jsonl`), then convert the run, never copy the score by hand:

```bash
"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger gsm8k <run>/gsm8k --out "$AGENT_SYS_WORKSPACE/results" \
  --label r1-spf-200 --series r1-spf          # the stack it measured: an experiment's series, or a version (v0)
```

It counts the questions answered right under the strict-match filter and writes `results/<label>.gsm8k.json` with
the samples file's path and sha256 (no point of the ledger's curves). Keep the lm_eval output where it was converted
from, unchanged: the checks count it again.

| When | Questions | Rule (95% Newcombe interval of the difference, on the questions both runs asked) |
|---|---|---|
| an experiment, before its accept | at least 200 (`EVAL_LIMIT=200`), run on this task's stack with its output under the task's workspace | the candidate is not significantly worse than the version's GSM8K: the interval does not lie wholly below 0. Every GSM8K in the task's `results/` counts, whatever series it names (a base arm's too), counted under strict-match. ab_judge refuses to write an accept without it and judges a significantly worse candidate reject; `ledger verify` counts it again |
| the baseline (v0) and a close that seals vN | all 1319 (`EVAL_LIMIT=1319`) | the version's own; vN not worse than the version before by more than `quality.gsm8k_margin` (project.yaml, 0.02): the interval's lower bound at least -0.02 (`perfcheck close`) |

Why two sizes: at p ≈ 0.96, 200 questions give each run about ±2.8 points and the difference of two runs about ±4.2,
so they catch a gross loss only (10 of 200 questions, 5 points, worse than the version on the same questions); 1319
questions give the difference about ±1.5 points, so a 2-point margin is proved when the new version scores at most
about half a point below the old one. "Within 0.01 of the baseline" is inside the noise at either size.

## Finding the cause (not gates)

| Check | How | Expect |
|---|---|---|
| Smoke | the kit's `engine/smoke.sh` / `smoke.py` (P/D registered, completion, chat, stream ends with `[DONE]`, a ~66K-token needle cold and reused) | 6/6 |
| Garble | greedy probes; look for template tags (`</think>`) in the content and repetition loops | 0 |
| Long context | a 16K-token needle (top-k sparse attention only engages past 2048 tokens; GSM8K never reaches it) | found |
| MTP acceptance | SGLang `accept len` median in the decode log | 2-3; exactly 4.00 means a degenerate loop |
| PD handoff | decode-side external prefix hit ~100%, decode prompt throughput ~0 | else decode recomputes the prompt |
| KV transport | `installTransport, type=rdma` in the logs, no `MC_FORCE_TCP` | RDMA |
| Kernel / comm change | the numeric check next to the change (e.g. a reference-vs-new comparison in eager and graph mode) | within tolerance |

Write what they found to `results/<label>.gate.json` (any shape; it is evidence, not a ledger row) and name it in the
report. A failed GSM8K stops the performance work: bisect the cause variable by variable with these checks and
temperature-0 probes (known causes: custom all-reduce, unfused NextN, prefill without MTP, DP2xDCP2, a half-applied
patch).

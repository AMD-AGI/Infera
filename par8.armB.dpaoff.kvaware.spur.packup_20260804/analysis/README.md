# Analysis

| file | what |
|---|---|
| `sli_percentiles.md` | every latency / throughput / cache number, whole-run and per-phase, generated directly from `results/summary.json` |
| `feature_evidence.md` | the five features one by one, each with its signal — and an explicit split between "proven configured" and "proven effective" |

Read `feature_evidence.md` before quoting this run as a kv-aware result: on this
arm kv-aware is proven **loaded**, not proven **steering**, and §3 explains why
that is partly structural rather than only a capture gap.

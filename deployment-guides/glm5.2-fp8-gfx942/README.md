# GLM-5.2-FP8 on gfx942 — deployment guides

Two ways to deploy GLM-5.2-FP8 on gfx942 (MI300X / MI325X), plus the multi-turn
agentic benchmark both are measured with. Pick one and read only that guide; each
is self-contained.

| Guide | Use it when | 中文 |
|---|---|---|
| [`docker.md`](docker.md) | You have the two nodes and want the shortest path to a working, tunable deployment. Shell scripts against a long-lived container per node. | [`docker.zh.md`](docker.zh.md) |
| [`kubernetes.md`](kubernetes.md) | You are deploying through the Infera operator. Four topologies, one to four `sed` substitutions each. | [`kubernetes.zh.md`](kubernetes.zh.md) |

**The two produce the same engine.** A flag-by-flag comparison of the argv each
side really produces found zero substantive differences across prefill, decode and
router, and the same workload put decode-side latency within 1.2%. Anything you
tune on one carries to the other.

## The one result to read first

Measured on 2× MI325X, prefill/decode disaggregated, on a multi-turn agentic
workload (60 conversations, 448 requests, median input 68k tokens):

| | Batch / high concurrency | Interactive / low concurrency |
|---|---|---|
| Configuration | default (DP-attention on, chunk 8192) | DP-attention **off**, chunk 2048 |
| Aggregate throughput at concurrency 16 | **182.7 tok/s** | 139.4 tok/s |
| Per-user speed at concurrency 1, median | 62.5 tok/s/user | **133–159 tok/s/user** |
| Mean TTFT at concurrency 16 | 12.4 s | 22.2 s |
| Cache efficiency | 100.00% | 100.00% |

Same weights, same image, one flag apart. **DP-attention is a concurrency
trade-off, not an optimisation that is always on.** Decide which column you are
serving before tuning anything else — nothing else measured moved the numbers
nearly as much, and three other promising directions were tried and ruled out.

## What is validated

- **Prefill/decode disaggregation on 2× MI325X**, both deployments: 448/448
  requests, cache efficiency 100.00%, no eviction.
- **All four Kubernetes topologies** (aggregated and disaggregated, each with and
  without KV offload) on 2× MI300X: zero failures, GPU faults or Pod restarts
  across concurrency 1/8/16/32.
- **The two deployments against each other**, statically and dynamically.

## Before you start

Three things fail *silently* on this stack — they return HTTP 200 and log nothing.
Each guide covers them, but they are worth knowing up front:

1. **The RoCE GID index.** A wrong one is 4–110× slower, not broken. It is also
   not portable between clusters.
2. **The base image against the host driver.** A mismatch loads weights, captures
   graphs, and then faults under load in a place that looks like an engine bug.
3. **A broken KV hand-off.** The decode leg reads a corrupt prefix and returns
   fluent text unrelated to the question. Only checking a known answer catches it.

## Where the numbers come from

The full record — raw benchmark output, profiler traces, the deployment
comparison, and the tuning attempts that failed — is in
[`mi325x-handoff/`](../../mi325x-handoff/README.md).

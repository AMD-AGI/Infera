# Measured results — GLM-5.2 1P1D at concurrency 8

What this deployment actually did, under **two independent agentic benchmarks**, on
**two clusters with different RDMA fabrics**. Every number here comes from a
completed run against the deployment shape this kit ships; nothing is projected.

| file | what |
|---|---|
| [`infera_agenticbench_conc8.md`](infera_agenticbench_conc8.md) | infera's own agentic benchmark — closed-loop session driver, ~67-minute window, both clusters |
| [`customer_agentx_caseA_conc8.md`](customer_agentx_caseA_conc8.md) | the customer's AgentX Case-A — open-loop frozen-trace replay, 900 s window, both clusters |

## Read this before comparing any two numbers here

Three axes vary across these runs, and only one of them is the deployment.

**1. The two benchmarks apply load differently, so their concurrency numbers are
not the same operating point.**

| | infera AgenticBench | customer AgentX Case-A |
|---|---|---|
| control loop | **closed** — a session waits for its response, then sleeps think-time | **open** — N lanes kept saturated |
| what "conc 8" means | 8 *initial sessions*; in-flight is emergent and lands wherever the server allows | 8 *pinned lanes*; in-flight sits at exactly 8 |
| requests | built live by the driver | frozen in 200 session files, byte-identical every replay |
| window | ramp 400 s + sustain 3,600 s | 900 s |

In a closed loop a slower server **reduces** offered load — sessions spend longer
waiting and issue fewer turns. In an open loop with pinned lanes it gets no such
relief. So a TTFT gap between the two benches is expected and is **not** evidence
about the deployment.

The two agree where the load model does not matter: **ITL/TPOT p50 ≈ 14–16 ms** and
**prefix cache ≈ 88 %**, measured independently by both drivers on both clusters.

**2. The two clusters differ in the fabric, and in two configuration values that
follow from it.** See [Cross-cluster](#cross-cluster-why-the-single-rail-cluster-looks-slower)
below — the short version is that the cross-cluster rows are **context, not a
measurement**, because more than one variable moves.

**3. The workload shape is the same in both benches, and reproduces.** Both are
built from the same Case-A profile — ISL p50/p90/p99 ≈ 74K/155K/235K, OSL p50/p90 ≈
320/3,300, ~89 % prefix reuse — and the realised distributions land within ~7 % of
each other on every axis. Workload shape is therefore never the explanation for a
latency difference between them.

## Cross-cluster: why the single-rail cluster looks slower

Running the same workload on both clusters, the single-rail one is **~1.6× slower on
TTFT p50** while matching on TPOT, cache hit rate and success rate.

| | multi-rail cluster | single-rail cluster |
|---|---|---|
| RDMA | 8 rails, peer-mem loaded (mode A) | 1 ODP rail, no peer-mem, dma-buf (mode B) |
| aggregate KV bandwidth (preflight's own metric) | ~3,200 Gb/s | ~200 Gb/s |
| TTFT p50 (infera bench, sustain) | 1,365 ms | 2,239 ms |
| TTFT p90 | 4,903 ms | 6,389 ms |
| TPOT p50 | 14.8 ms | 16.1 ms |
| cache hit | 88.8 % | 88.7 % |

**No single-variable experiment separates these causes.** Four candidates, with the
evidence behind each stated honestly:

| # | candidate | evidence | what would settle it |
|---|---|---|---|
| 1 | **Fabric.** 1 rail vs 8, and dma-buf vs peer-mem registration. | First-hand node facts on both clusters. **No controlled experiment.** | Not separable — the fabric is the cluster. |
| 2 | **`--chunked-prefill-size` differs**: 65,536 global on the single-rail runs, 16,384 on the multi-rail ones — 4× the per-forward prefill work. | First-hand from both runs' resolved args. The two source recipes genuinely disagreed on this value and the disagreement was recorded rather than resolved. | Re-run one cluster at the other's chunk. Cheapest of the four. |
| 3 | **`--mem-fraction-static` differs**: 0.70 vs 0.80 on prefill → a smaller KV pool (−19 % measured when this was changed within one cluster). | First-hand. Forced, not chosen: 0.80 does not boot the single-rail cluster's prefill leg. | Only separable if 0.80 can be made to boot there. |
| 4 | **MTP and decode-side radix cache are mutually exclusive upstream**, so `decode_prefix_len` is always 0 and **every turn re-transfers the entire prompt KV**. A prefill-side cache hit saves *compute*, not *bytes*. | First-hand: SGLang raises on `--disaggregation-decode-enable-radix-cache` together with `--speculative-algorithm`. | This does not act alone — it **amplifies** #1, by putting a full-prompt KV transfer on every turn's critical path. |

Candidate 4 is the reason to expect #1 to matter *on this workload specifically*: at
~86K mean input tokens and ~89 % prefix reuse, the deployment re-sends the whole
prompt's KV every turn regardless of how well the cache is working. A workload with
short prompts would be far less fabric-sensitive.

**None of the four is confirmed.** The honest summary is that the single-rail cluster
is slower on TTFT by a factor that is consistent with its fabric, and that two
configuration deltas ride along with the fabric and are not controlled for.

## Provenance

Numbers are recomputed from raw per-request records where those exist, not copied
from a summary line. The customer-bench ladders come from aiperf's
`profile_export.jsonl`; the infera-bench ladders from the driver's own
`summary.json` sustain-phase block, with the ramp window excluded exactly as that
driver defines it.

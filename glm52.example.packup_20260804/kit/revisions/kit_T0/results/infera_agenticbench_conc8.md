# infera AgenticBench — Case-A request shape at concurrency 8

infera's own agentic benchmark, run against exactly the deployment shape this kit
ships. **Closed-loop**: each session issues one request, waits for the response,
sleeps its inter-turn delay, then issues the next. Offered load is therefore set by
the live-session population, not by a QPS target.

## Workload

The Case-A agentic profile at reduced load. Long inputs, a heavy output tail,
realistic think-time, and a large shared prefix:

| axis | value |
|---|---|
| input tokens | p50 74,000 · p90 155,000 · p99 235,000 (clamped at 260,000) |
| output tokens | p50 320 · p90 3,300 · p99 17,000 |
| turns per session | p50 3 · p90 20 · p99 103 |
| inter-turn delay | p50 4 s · p90 31 s · p99 240 s |
| target prefix-cache hit | 0.89 |
| initial sessions / max sessions / max in-flight | 8 / 32 / 24 |
| window | ramp 400 s (excluded) + sustain 3,600 s |

All figures below are the **sustain phase only**; the ramp is a warm-up exclusion
window, sized so the synchronised start cohort has died off and the shared prefix is
resident before measurement begins.

## Results

Three runs. All three ran the workload byte-identical; they differ in the cluster
and in the deployment shape.

| | **multi-rail cluster**<br>prefill DPA off + kv-aware | **single-rail cluster**<br>prefill DPA off + kv-aware | **single-rail cluster**<br>prefill DPA on + round-robin |
|---|---|---|---|
| requests sent / completed | 2,884 / 2,850 | 2,907 / 2,861 | 2,323 / 2,289 |
| success rate | 0.988 | 0.984 | 0.985 |
| QPS (sustain) | 0.74 | **0.75** | 0.60 |
| **TTFT p50** | **1,365 ms** | 2,239 ms | 3,504 ms |
| **TTFT p90** | **4,903 ms** | 6,389 ms | 7,079 ms |
| TTFT p99 | 9,066 ms | 10,606 ms | 16,602 ms |
| **TPOT p50** | **14.8 ms** | 16.0 ms | 16.5 ms |
| TPOT p90 | 17.7 ms | 21.1 ms | 23.3 ms |
| cache hit (actual / ideal) | 88.9 % / 89.0 % | 88.7 % / 89.0 % | 88.2 % / 89.0 % |
| cache efficiency | 100.0 % | 99.6 % | 99.1 % |
| MTP acceptance length | 2.02 (per-request) | 2.79 (engine) | 2.72 (engine) |
| engine faults | 0 | 0 | 0 |
| prefill `--mem-fraction-static` | 0.80 | 0.70 | 0.70 |
| `--chunked-prefill-size` (global) | 16,384 | 65,536 | 65,536 |

### Against the workload's own SLA block

| bar | target | multi-rail | single-rail (DPA off) | single-rail (DPA on) |
|---|---|---|---|---|
| success rate | ≥ 0.97 | 0.988 **PASS** | 0.984 **PASS** | 0.985 **PASS** |
| TTFT p90 | < 30,000 ms | 4,903 **PASS** (6.1×) | 6,389 **PASS** (4.7×) | 7,079 **PASS** (4.2×) |
| E2E p50 | < 4,500 ms | 7,400 ms **FAIL** | not recorded | not recorded |
| in-flight not pinned at cap | — | max 22 / 24 **PASS** | not pinned **PASS** | brushed 24 on 0.8 % of ticks **PASS** |

The E2E miss is not a regression. `e2e_p50_ms: 4500` is a **latency-floor** spec that
is met only at concurrency 1; at ~12 mean in-flight it is the wrong bar, and it fails
identically in every loaded run on this stack. It is reported rather than quietly
dropped.

## What these runs establish

**1. Prefill scales with input size, with no pathology.** On the multi-rail run,
TTFT p50 by input-size bucket:

| input | 0–50K | 50–100K | 100–160K | 160–220K | 220–300K |
|---|---|---|---|---|---|
| TTFT p50 | 623 ms | 1,036 ms | 1,815 ms | 3,411 ms | 5,863 ms |

Monotone, 9.4× across a 4.5× size span, **no stall bucket**. The deployment is
computing, not queueing, at this load.

**2. Decode is not the bottleneck.** TPOT p99/p50 = 1.45 on the multi-rail run — an
exceptionally tight ladder. A contended decode leg shows a fat upper tail.

**3. The prefix cache works as specified.** 88.2–88.9 % actual against an ideal of
89.0 %, i.e. 99–100 % cache efficiency and 0.2–0.9 % eviction. The workload nests
every request inside the same prefix, so this verifies cache *accounting* under load;
it does not exercise eviction pressure.

**4. MTP is healthy, not degenerate.** Acceptance length 2.02–2.79. **A steady 4.00
would be bad news**, not a better result — it means the draft model is predicting a
repetition loop perfectly.

**5. The load cap never bound**, so the workload set the offered load rather than
backpressure, and the measurement window is valid.

## The two things this data cannot tell you

**The DPA/routing comparison is two-variable.** The two single-rail arms differ in
*both* prefill DP-attention *and* router policy, because that is how they were
specified. Arm B (DPA off + kv-aware) is faster on every latency percentile and
carries 25 % more throughput, but that gap is the **combined** effect and this data
cannot split it. The 2×2 is missing its other two cells.

One prior controlled result bears on the DPA half alone: at **concurrency 1**, in a
single-variable comparison, prefill DP-attention cost 1.65–1.93× on TTFT. The
direction is consistent with the gap above; the load there is 24× lower and nothing
licenses transferring the magnitude.

**Routing policy and a memory knob are coupled, and it is not documented anywhere
else.** The DPA-on + round-robin arm **would not boot** at `--mem-fraction-static
0.80`; it aborted 60 s in with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` while token usage
read 0.05 — an empty KV pool, so activation memory, not KV exhaustion. The mechanism
is the spreading itself: under round-robin 4–5 DP ranks prefill concurrently, each
holding its own chunk's activations, where kv-aware concentrates on 1–2. At 0.70 it
ran the full 4,007 s with zero faults, at a cost of 19 % of the KV pool — which, at a
peak token usage of ~0.05, was never the binding resource.

**If you switch this kit to `ROUTER_POLICY=round-robin`, lower `GMU_PREFILL`.** That
is why the shipped default is 0.70 rather than the higher value a kv-aware-only
deployment could sustain.

## One structural finding worth knowing before you tune

**kv-aware routing performed no cache steering in these runs, on either leg.** Not a
misconfiguration — a consequence of the deployment:

- **Prefill**: with DP-attention off, `dp_size=1`, so the router has exactly one
  target to choose between.
- **Decode**: MTP forces `ChunkCache` instead of a radix tree, because SGLang raises
  on `--disaggregation-decode-enable-radix-cache` together with
  `--speculative-algorithm`. The router's KV view of that worker is therefore
  permanently empty and the cost function's overlap term cancels, leaving pure
  least-loaded routing. Verified on the wire: the decode leg's kv-event socket
  emitted 0 messages in 15 s while prefill's emitted 30.

The consequence is the one in the [cross-cluster analysis](README.md#cross-cluster-why-the-single-rail-cluster-looks-slower):
`decode_prefix_len` is always 0, so **every turn re-transfers the entire prompt KV**.
A prefill-side cache hit saves compute, not bytes.

This does not make kv-aware pointless — it steers prefill whenever DP-attention is on
there, which the round-robin arm demonstrated by contrast (round-robin spread picks
±4 % across all 8 ranks; kv-aware concentrated on 2 of 8 and left the other six at one
batch each for an entire run, because the concentration is self-reinforcing: no
traffic → empty cache view → never the cheapest candidate → no traffic).

## Running it yourself

This kit deliberately ships **no agentic bench client** — only the service self-check
(`smoke`) and a reference sweep with SGLang's own `bench_serving` (`bench`). The
agentic harness above is a separate internal tool. What this kit gives you is the
deployment those numbers were measured against.

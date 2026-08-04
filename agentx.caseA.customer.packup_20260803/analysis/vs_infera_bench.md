# The two benches side by side — what each can and cannot establish

Both benches implement the **same Case-A spec**. They produce different numbers
from the same server in the same hour. This file explains why, and states which
questions each one is actually able to answer.

---

## The specs agree; the mechanisms do not

The two workloads are drawn from the same parameter set, and the realised
distributions confirm it:

| | par8 (measured) | AgentX trace (authored) | delta |
|---|---|---|---|
| ISL p50 | 75,440 | 74,140 | 1.8 % |
| ISL p90 | 157,056 | 146,111 | 7.5 % |
| ISL p99 | 230,669 | 245,000 | 6.2 % |
| ISL mean | 86,888 | 84,213 | 3.2 % |
| OSL p50 | 316 | 306 | 3.3 % |
| OSL p90 | 2,852 | 2,980 | 4.5 % |
| cache hit | 88.9 % | 88 % (constructed) | 1.0 % |

**Within ~7 % on every axis.** The workload shape is not the explanation for any
latency difference between the two benches.

## Where they diverge: the load model

| | infera AgenticBench (par8) | customer AgentX Case-A |
|---|---|---|
| control loop | **closed** — next turn waits for the previous response, then sleeps think-time | **open** — `--concurrency` lanes kept saturated |
| concurrency | *emergent*; `max_inflight` is a ceiling that never bound (22 of 24) | **pinned**; in-flight sat at exactly 8 and exactly 16 |
| request content | constructed live by the driver | frozen in 200 session JSONs, byte-identical every replay |
| think-time | reduces offered load (session idles) | consumed **inside** a lane; the lane still holds its slot |
| duration | ramp 400 s + sustain 3,600 s | 900 s per point (scenario minimum) |
| terminator | `max_tokens` + `ignore_eos`, no `stop` | same (`require_ignore_eos` in the scenario) |
| client timeout | **240 s** — max observed 239.2 s, nearly binding | **none** — 289.9 s completed normally |
| cache metric reported | server's `cache_hits` via `--enable-cache-report` | **trace-side theoretical** (see below) |

### The consequence, stated precisely

par8's "mean in-flight 12.1" and AgentX's "concurrency 16" are **not the same
operating point**. In a closed loop, a slow server *reduces* offered load —
sessions spend longer waiting and issue fewer turns. In an open loop with pinned
lanes, a slow server does not get that relief: the lane re-issues the moment the
previous request returns. AgentX at c16 therefore applies strictly more pressure
than par8 ever did, and its TTFT p50 of 14,394 ms against par8's 1,365 ms is
consistent with that.

**But this does not explain c8.** At c8 mean in-flight is 5.13 — *less than half*
par8's 12.1 — and TTFT p50 is still 5,146 ms against par8's 1,365 ms. **That
residual 3.8× is not accounted for by this data.**

Candidates, none of which this run can discriminate between:

| # | candidate | what would settle it |
|---|---|---|
| A | the 900 s window is too short for the prefill leg's radix tree to reach par8's steady state (par8 excluded a 400 s ramp *and* ran 3,600 s) | run AgentX at c8 for 3,600 s and compare the first 900 s against the last 900 s |
| B | `cache_bust=FIRST_TURN_PREFIX` — the scenario injects a unique marker into every trajectory's first turn, deliberately defeating cross-trajectory prefix sharing that par8's driver allows | run with the scenario's cache-bust disabled (requires `--unsafe-override` semantics we have not tested) |
| C | the trace's inter-turn think-times are consumed inside the lane, so a "session" in AgentX issues turns far denser than a par8 session does | compare measured inter-arrival per conversation against the trace's `think_time` field |
| D | measurement definition — aiperf's TTFT is first *token*, our driver's is first *chunk* | inspect the first streamed chunk of a known request under both drivers |

**Do not pick one of these without running the experiment.** Candidate B is the
one most specific to the customer's method and is the cheapest to test.

## What each bench can establish that the other cannot

### Only AgentX can do this

- **Cross-topology apples-to-apples.** The trace is frozen; replaying it against
  a different deployment changes exactly one variable. par8's closed loop
  adapts to the server, so two par8 runs on different deployments are not
  strictly comparable — this is precisely the confound par8's own analysis flags
  about its TTFT headline.
- **Per-turn attribution.** aiperf tags every record with `turn_index`, which is
  how this kit could price the prefix cache at **2.5× TTFT** (first turn 8,981 ms
  vs cached 3,568 ms at matched input size). par8's metrics stream has no turn
  index.
- **A guaranteed-clean input distribution.** `max(in + out) = 258,303` by
  construction → structurally zero context overflows. par8 carries 0.52 %
  overflow errors from independent in/out sampling.
- **Per-request server cache accounting.** `usage_prompt_cache_read_tokens` is
  captured per record, giving a per-request cache% distribution
  (p50 **88.1 %**, p90 89.4 %, max 89.9 %, **zero** requests at 0 % among those
  reporting it). par8 reports one aggregate rate.

### Only par8 can do this

- **Capacity discovery.** A closed loop finds the concurrency the workload
  naturally sustains (Little's law closed at 31.3 ≈ the observed cap of 32).
  AgentX must be *told* the concurrency, so it measures the server at a point
  you chose, not the point the workload implies.
- **Long-window steady state.** 3,600 s of sustain after a 400 s warm-up
  exclusion. AgentX's scenario minimum is 900 s and its default 1,800 s.
- **Router / KV internals.** par8 captured per-DP-rank pick distributions, kvd
  counters, and kv-event socket traffic — which is how it established that
  kv-aware routing steered nothing. AgentX sees only the endpoint.
- **Realistic client-timeout pressure.** par8's 240 s client timeout is a real
  production constraint that AgentX does not model at all.

## The one methodological thing worth adopting from the customer

**Freeze the trace.** par8's own analysis says its headline TTFT is confounded
because two variables moved together and "this data cannot apportion it", and
names the missing control run. A frozen open-loop trace removes that class of
confound structurally: the offered demand is a file, not a negotiation with the
server.

This does not mean replacing our bench. The two answer different questions —
ours answers *what load can this deployment carry*, theirs answers *how do two
deployments compare under identical demand*. **Running both is the correct
posture**, and this run demonstrates they agree where they should (ITL 13.8 vs
14.8 ms; cache 88.1 vs 88.9 %) and diverge where the load model differs.

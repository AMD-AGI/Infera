# Working process — AgentX on the corrected MTP config + HiCache PR #37152

One section per step. Hypothesis / change / command / result vs target.

## Setup — 2026-09-18 ~10:35 UTC

**Research findings that changed the plan.**

1. **`SGLANG_OPT_USE_TOPK_V2=false` is already the committed default.**
   `config.sh:93` and `config.full.sh:101` both read
   `${SGLANG_OPT_USE_TOPK_V2:-false}`; `engine.sh:123` passes it into the
   container. The requested optimisation is a **no-op**. It will be asserted in
   the runtime snapshot rather than set again.

2. **"fast mode" is a real InferenceX flag and `config.sh` already equals it.**
   `AIPERF_EXPERIMENTAL_FAST=1` (`benchmarks/benchmark_lib.sh:1979`) does exactly
   two things — `duration=1200`, `warmup_requests_per_lane=1`. `config.sh:121,123`
   already set both. The variable is not plumbed through `agentx_env.py` and does
   not need to be. `config.full.sh` sets 3600 / 10 = full mode.

3. **PR #37152 needs no full rebuild.** All three hunks land on plain source
   files under `python/sglang/`; the `.cuh` is JIT-compiled at runtime against a
   content-addressed cache. A thin layer over the existing `nextnfix` image is
   sufficient and keeps the base's 7 DSA patches untouched.

4. **PR #37152 is inert in T1.** `config.sh` has `PREFILL_HICACHE=0`.

**Node state at setup** (`rocm-smi`, both nodes): GPUs 2,3,4,5 free on both.
135 GPU[1] holds the root-owned k8s vLLM pod (96.94 GB) — expected, untouched.
No running `glm52-pd-*` containers on either node.

**Artifacts created**

| path | what |
|---|---|
| `mission.md` | the task book, re-injected every 10 min |
| `topology.yihou.tsv` | 135 prefill / 138 decode (copied from the previous workspace) |
| `image/Dockerfile.yihou.hicache` | thin layer, `FROM …-nextnfix`, applies PR #37152 |
| `image/pr37152.sources.yihou.diff` | the 3 source hunks, copied verbatim from the reference packup |
| `config.yihou.base.sh` | shared: image, nodes, P4D4 on 2-5, same-rail, **`--disable-custom-all-reduce`**, selects which repo config to inherit |
| `config.yihou.fast.sh` | T1 — inherits `config.sh`, `DECODE_SIMULATE_ACC_LEN=` (real) |
| `config.yihou.full.sh` | T2 — inherits `config.full.sh`, flydsl→tilelang, aiter→sgl-kernel |

**Config self-check** — sourced both files and printed the resolved values:

| | T1 fast | T2 full |
|---|---|---|
| IMAGE | `…-nextnfix-hicache` | `…-nextnfix-hicache` |
| `DECODE_SIMULATE_ACC_LEN` | `[]` (empty — real) | `[3.61]` |
| `DECODE_EXTRA_ARGS` | `--disable-custom-all-reduce` | same |
| `SGLANG_OPT_USE_TOPK_V2` | `false` | `false` |
| `PREFILL_HICACHE` | 0 | 1 |
| duration / warmup | 1200 / 1 | 3600 / 10 |
| max-running P/D | 64 / 64 | 128 / 128 |
| DSA backends | unset (baseline) | `tilelang/tilelang/sgl-kernel` |
| `JSON_MODEL_OVERRIDE_ARGS` | `index_share…=false` | empty |

All as intended, including the single-dash `${DECODE_SIMULATE_ACC_LEN-3.61}`
subtlety that lets an empty-but-set value disable simulation.

**Team dispatched**: `image-hicache` (build + verify the layer on both nodes),
`bench-preflight` (occupancy, RDMA rails, repo preflight). 10-minute mission
injection and 20-minute teammate poll are scheduled.

## Checkpoint — 2026-09-18 ~11:00 UTC

`image-hicache`: **DONE, PASS both nodes.** Image ids `fd7220a57b7d` (135) /
`972d8fd952e9` (138). Different ids are expected and are recorded as a
provenance caveat: each is a local layer over that node's own locally-built
base, never pushed, so the tag does not identify one blob across the two nodes —
reproduction rests on the Dockerfile plus the base tag.

`bench-preflight`: still running the mooncake stage (~20 min in), containers
`infera-preflight-…-rank{0,1}` up on both nodes. All 8 per-GPU RDMA transfers so
far read `transfer_failed` with an empty `"dev"`. **Recorded, not acted on** —
first sighting; see `notes/poll-log.md` for the two candidate explanations and
what neither of them covers.

## Checkpoint — 2026-09-18 ~11:20 UTC

Preflight **run2** (`preflight-libionic/`, host `libionic` injected) is
producing per-GPU results, and the fault has moved exactly as predicted:

| GPU | run1 (no libionic) | run2 (libionic, auto-discovery) |
|---|---|---|
| 2 | `transfer_failed`, `dev` empty | **verified, 45.99 GB/s** |
| 3 | `transfer_failed`, `dev` empty | **verified, 45.70 GB/s** |
| 4 | `transfer_failed`, `dev` empty | `transfer_failed` |
| 5 | `transfer_failed`, `dev` empty | _still running_ |

So the GPUDirect provider was the whole of run1's failure. What remains on GPU4
is the **different, already-documented** fault: run2 was launched before the
leader's note about exporting `RDMA_DEVICE` / `MC_TE_FILTERS`, so Mooncake is
back on auto-discovery, and the two ends pick different HCAs from the NUMA-local
pool across physically isolated rails — the "low GPUs pass, high GPUs fail"
signature of `glm52-1p1d-samerail-c32-c40.packup_20260918` notes.md §1. A third
run with the pinned per-GPU JSON is what settles it.

Recorded, not intervened on: the teammate has the values and the reasoning.

## Preflight — stood down, 2026-09-18 11:05 UTC (leader's call, leader's error)

Ordering preflight at all was a mistake and is recorded as one. R05 ran at
~09:57 **today** on these exact nodes, the exact same GPUs 2-5, and the exact
same pinned `RDMA_DEVICE` JSON, with 0 Mooncake failures and 0 router affinity
503s. Nothing on the fabric changed since; the only delta this round is a thin
image layer over three source files. A probe re-validation could not beat a
1.5-hour-old end-to-end result from the real engine, and the two runs it did
cost ~45 min while surfacing only faults in the probe's own invocation.

What the exercise did produce, and what is kept:

- **`preflight.sh` sources no config file.** It takes only `IMAGE=` plus node
  names, and mounts the host GPUDirect provider solely when the caller exports
  `HOST_RDMA_LIB` (`preflight.sh:62-63`). `config.sh:104`'s default never reaches
  it. Verified by reading the script. A reusable gotcha about the repo's tooling.
- **Bare `preflight.sh` uses auto HCA selection only**, which on these two nodes
  reproducibly fails GPUs 4-7 in both directions while 0-3 pass — the isolated-rail
  artifact, not a fabric fault. The teammate then ran a pinned probe on today's
  image and got all four run GPUs passing both directions: gpu2 29.9/28.9,
  gpu3 27.9/30.0, gpu4 40.4/41.3, gpu5 40.3/41.2 GB/s.
- On 135, `ionic_7`'s non-zero GID in `ibv_devinfo` is **stale**; sysfs (all-zero,
  no netdev) is ground truth. Sharper than the previous pack-up's wording.

Inventory in `preflight-report.md`. Both nodes left clear of probe containers.

## T1 — fast, CONC=40, real acceptance

**Bring-up** 11:04–11:17 UTC. Prefill healthy in ~3 min; decode ~13 min (MTP
graph capture). Router ready at `http://10.245.148.209:28000`.

**Flags confirmed from the emitted command line, not from the config:** decode
carries `--disable-custom-all-reduce` and `--speculative-algorithm EAGLE` with
5 steps / topk 1 / 6 draft tokens; **no `SGLANG_SIMULATE_ACC_LEN` env is
present**, which is how real acceptance is proven — by the variable's absence;
`SGLANG_OPT_USE_TOPK_V2=false`; image `…-nextnfix-hicache`; `HiCache=0` on both
legs, so PR #37152 is inert here as expected.

One discrepancy chased down rather than waved through: the config self-check
printed empty DSA backends for T1, but the launch emitted
`--dsa-prefill-backend tilelang`. `engine.sh` supplies that default when the
config leaves it unset; it matches the C32 baseline, so it is not a deviation.

**NextN fusion fix confirmed live**: decode logs `Shared experts fusion
optimization enabled` **2×** and `Config does not support fused shared expert(s)`
**0×**. Under the old image the draft logged the disabled line.

**Correctness smoke, real acceptance, temperature 0** — both target prompts:

```
"What is 2+2? …"            -> '1.  **Analyze the Request:** … 2 + 2 = 4 …'
"Reply with exactly: …"     -> 'BASELINE_OK_YIHOU111742' (exact)
```

Coherent, and the nonce reproduces. This is the first AgentX-shaped run on this
stack that decodes correctly without simulated acceptance.

**Idle snapshot taken before load** (`t1-accept/metrics-before.txt`): all four
decode ranks read `spec_accept_rate 0.0`. Kept deliberately — this is the
idle-gauge trap, not a measurement.

AgentX C40 started 11:18 UTC into `t1-agentx-c40/`. Warmup 84 requests,
`errors=0`; profiling from ~11:30.

### T1 headline — real MTP acceptance under AgentX load, 11:48 UTC

Read while the profiling phase was live, so every rank had served tokens.

| dp_rank | `spec_accept_length` | `spec_accept_rate` |
|---|---|---|
| 0 | 2.1500 | 0.2300 |
| 1 | 2.3305 | 0.2661 |
| 2 | **3.0750** | 0.4150 |
| 3 | 2.7917 | 0.3583 |

**4/4 ranks active, min 2.1500, mean 2.5868 — PASS against the 2.0 bar.**
Independent second source, the decode server log's own line:
`accept len: 2.33 / 4.50 / 2.46 / 3.29 / 2.77`.

Baseline for contrast: before the fix this shape read `accept len 1.25 /
accept rate 0.05` **and** emitted garbled text.

**Not a regression against R05's 2.86–3.05.** R05 measured 8 short synthetic
prompts; this is a long-context multi-turn agent trace at 90–96 % prefix-cache
hit. Different workloads, not comparable point to point.

### Open item — 128+ `OSL mismatch` warnings

Requests returning far fewer tokens than the trace asked for, worst seen
-90.6 % (976 of 10,330). Must be quantified **per request from
`profile_export.jsonl`**, never from the aggregate JSON's
`request_metrics.tokens.output_expected` — the C72 pack-up recorded a correction
where exactly that mistake inflated the figure by more than 10×. Deferred to
the analysis step; flagged here so it is not forgotten.

Note this run has real acceptance and real EOS behaviour, so some shortfall may
be the model legitimately stopping earlier than the recorded trace did, rather
than truncation. **Unresolved either way — do not assume which.**

### T1 AgentX C40 result — completed 11:57 UTC

`AgentX passed`, error rate **0/1067 = 0.000 %**, profiling window 1229.3 s.

| metric | value |
|---|---|
| requests total / profiled / warmup-dropped / error-dropped | 1151 / 1067 / 84 / **0** |
| throughput input / output / total | 96,817 / 441.9 / 97,259 tok/s |
| **per chip (8 GPU)** | **12,157 tok/s/chip** (output 55.2) |
| TTFT mean / p50 / p90 / p95 | 23.97 / **8.73** / 56.20 / 101.84 s |
| E2E mean / p50 / p90 | 31.25 / **16.45** / 74.04 s |
| ITL / TPOT mean (p95) | **16.4 ms** (23.3 ms) |
| interactivity p50 | 61.3 tok/s/user |
| prefix cache (theoretical / server) | 96.2 % / 90.0 % |
| KV cache GPU usage | 0.99 % of 17,962,240 tokens |

### The OSL shortfall, computed properly — 40.95 %

Computed **per request from `profile_export.jsonl`**, using each record's own
`osl_mismatch_diff_pct`, never the aggregate `output_expected` mean. Saved to
`t1-osl-analysis.txt`.

```
profiling records                      1067
osl_mismatch_diff_pct == 0.0 exactly   884/1067  (82.8 %)
negative (short) requests              183       (17.15 %)   worst -96.2 %
positive (over-long) requests          0
sum(requested) 919,893   sum(actual) 543,225
deficit 376,668 tokens = 40.95 %
top-10 requests carry 30.0 % of it; top-50 carry 72.5 %
```

**This is an order of magnitude worse than the C72 run's 3.11 %, and it is not
tail-driven** — 17 % of requests are short and the top-50 only account for 72 %.
So `output 441.9 tok/s`, `per-GPU output 55.2` and `e2el` describe a run that
produced well under half the output tokens the trace asked for, and are **not
comparable** to a run that fulfilled them. Input throughput and TTFT are
prefill-side and unaffected.

**What the shortfall is not** — checked, not assumed:

| candidate | evidence |
|---|---|
| client cancellation | `was_cancelled: False` on all 183 |
| context overflow skip | `context_overflow_skip: False` on all 183 |
| a hard output cap | the exact-match group reaches 13,727 tokens, so long outputs *can* complete |

The short requests are the **long** ones: median actual OSL 744 vs 239 in the
exact group. So the engine stopped generating on long trajectories while
shorter ones matched the trace exactly.

**Leading hypothesis, explicitly not a conclusion:** with simulated acceptance
off, this is the first run in this series where the model emits its own EOS, and
GLM-5.2 is simply more concise than whatever produced the recorded trace. That
would make the deficit a workload-alignment artifact rather than a defect. The
competing possibility — that something in the corrected MTP path terminates long
generations early — is **not ruled out** and would need a controlled comparison
(same trace, `DECODE_MTP=0`) to separate. Not run: out of T1's scope.

### Teammates

Both `image-hicache` and `bench-preflight` are idle with their deliverables
written. No open teammate items.

## T2 — full, sweep 40/56/72/96/128, simulated acceptance_

_pending_

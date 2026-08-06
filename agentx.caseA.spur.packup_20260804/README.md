# Customer AgentX Case-A bench on infera GLM-5.2 PD — CruSoe/spur

**Ran 2026-08-04 04:00–04:53 UTC.** Three concurrency points, 900 s each,
against a two-node infera PD deployment on the **CruSoe (spur / crsuse2-m2m)**
cluster with **prefill DP-attention OFF**.

## What this run set out to do

The customer supplied their own agentic benchmark
([ROCm/MAD PR #173](https://github.com/ROCm/MAD/pull/173), `scripts/AgentX_CaseA/`).
The task (`spec/task_brief.md`) was to run **their** bench against **our**
deployment, on the cluster we actually ship to them, and analyse the result.

The customer's `replay_caseA.sh` was run **unmodified** — md5
`7cde1afc627c7e4868eac0fd13741baa`, identical to the PR blob. Everything
site-specific was passed as environment (`scripts/run_caseA.sh`).

## Headline result

| | C=2 | C=8 | C=16 |
|---|---:|---:|---:|
| requests (profiling) | 72 | 229 | 303 |
| **TTFT p50 / p90** | 2,503 / 14,458 ms | 6,698 / 23,775 ms | 19,496 / 42,484 ms |
| TTFT p99 | 24,085 ms | 33,681 ms | 83,575 ms |
| **ITL p50 / p90** | 11.8 / 15.6 ms | 13.3 / 18.5 ms | 14.7 / 23.3 ms |
| **E2E p50 / p90** | 8.6 / 28.6 s | 13.9 / 38.8 s | 27.6 / 72.3 s |
| output throughput | 98 tok/s | 258 tok/s | **402 tok/s** |
| input throughput | 6,772 tok/s | 19,899 tok/s | **26,447 tok/s** |
| ISL mean / p99 | 85,314 / 244,873 | 80,811 / 240,623 | 81,173 / 237,218 |
| OSL mean / p99 | 1,237 / 11,119 | 1,050 / 10,417 | 1,233 / 15,567 |
| server cache read / prompt | 58.6 % | 66.1 % | 66.6 % |
| **`submission_valid`** | **true** | **true** | **true** |

**Zero engine faults** on either leg across all three runs.
Corpus conformance re-verified here: **13/13 axes PASS**.

## The five features, each with a positive signal

| feature | signal | result |
|---|---|---|
| **PD** | `/v1/workers`, both disagg modes | prefill + decode, both `active` |
| **DPA** | live command lines | prefill **no** `--enable-dp-attention` (by design) / decode **8/8** `scheduler_DP*` |
| **MTP** | `accept len` in decode log, n=5,564 batches | mean **3.06**, p50 2.98 — healthy band, not degenerate 4.00 |
| **kvd** | `statctl` deltas | prefill **+168,866 sets / +121,621 evictions / 0 gets**; decode all-zero **by design** |
| **RDMA** | `MC_FORCE_TCP` / `GID is NULL` count | **0** on both legs — real mlx5 RDMA, no silent TCP fallback |

## The finding: 88 % constructed reuse measures as 66 %, and why

The corpus is built for ~88 % prefix reuse. The server's own
`usage.prompt_cache_read_tokens` reports **50.8 %** overall at C=8. That gap is
**not** a cache defect. Decomposed by turn index (profiling phase only):

| segment | C=8 | C=16 |
|---|---:|---:|
| turn 0 (session's first turn) | n=53, **0.0 %** | n=73, **0.0 %** |
| turns 1–2 | n=71, **70.1 %** | n=99, **67.1 %** |
| turns 3+ | n=105, **66.4 %** | n=131, **65.5 %** |
| overall | 50.8 % | 49.7 % |

Two independent effects:

1. **Turn 0 is forced cold by the benchmark itself.** The
   `inferencex-agentx-mvp` scenario locks `--cache-bust first_turn_prefix`,
   injecting a per-play unique marker at the head of every first user turn so a
   recycled trace cannot inherit a warm prefix. It is a deliberate
   anti-inflation rule. Turn 0 is 25 % of C=8's prompt tokens and contributes
   **exactly zero** cache read; removing it lifts the realized rate from 50.8 %
   to **67.9 %**.
2. **The residual ~66 % vs the corpus's ~88 % is measured but NOT explained.**
   It is *consistent with* block-vs-page granularity (the trace counts 64-token
   `hash_ids` blocks; the engine counts whole KV pages it can skip), but that is
   inference. See `notes.md` §1 for the measurement that would settle it.

The rate is **flat across concurrency** (66.1 % → 66.6 %), so the hot tier is
not being evicted under load at these levels.

> `Theoretical Prefix Cache Hit` (aiperf's own column, ~50.8 %) and the
> server-measured 66.1 % are **different quantities** computed from different
> sources. Do not compare them.

## What scales, and what does not

C=8 → C=16 buys **+55 %** output throughput (258 → 402 tok/s) and costs
**+191 %** TTFT p50 (6.7 → 19.5 s). ITL barely moves (13.3 → 14.7 ms p50), so
the decode loop is not the constraint — the queue in front of prefill is. With
ISL p50 ≈ 72 K and DPA off, one prefill scheduler serves the whole node and 16
in-flight long prompts serialize against it.

## ⚠ Two things this kit does NOT establish

**The TTFT numbers cannot be attributed to DPA-off.** Prefill DPA was off by
instruction, and its paired compensations (`CHUNK=65536` unchanged but now
undivided, `GMU` 0.80 → 0.70) moved with it. Separating them needs the same
config re-run with prefill DPA on. That run does not exist in this kit.

**`submission_valid=true` was not predicted, and is not fully explained.** The
customer script hardcodes `--unsafe-override` plus non-default
`--trajectory-start-*` ratios; the expectation was `false`. All three points
returned `true`. The likely reading — that `default_trajectory_start_*_ratio`
are defaults rather than locks, making `--unsafe-override` a no-op — was **not
verified against the validator source**. See `notes.md` §2.

## Arm 2 — prefill DPA ON, kv-aware weights re-derived (added 05:32–05:49 UTC)

A second arm was run at C=8 with **prefill DP-attention ON (dp8)** and the
kv-aware overlap weights re-derived from arm 1's measured miss distribution
(**prefill 20.0 → 5.0**, **decode 2.0 → 1.0**). Full treatment:
[`analysis/dpa8_arm.md`](analysis/dpa8_arm.md).

| C=8 | arm 1 (DPA off) | arm 2 (DPA on) |
|---|---:|---:|
| TTFT p50 | **6,698 ms** | 13,578 ms |
| TTFT p90 | **23,775 ms** | 32,506 ms |
| output throughput | **258.5 tok/s** | 220.2 tok/s |
| server cache read/prompt | 66.1 % | 66.2 % |

**DPA-on is worse on every axis here**, and the cache rate is flat — so this is
not a caching effect. The mechanism: sglang divides `chunked_prefill_size` by
`dp_size` only under DPA, so the same CLI 65536 resolves to **8192 per forward**
instead of 65536 — 8× smaller batches for a workload with ISL p50 ≈ 68 K.
**DPA and the resolved chunk moved together; this is not a clean ablation.**

**Per-rank load is NOT balanced.** Sampled every 15 s (87 samples, persisted to
`results/dpa8_c8/rank_samples.jsonl`): DP0 got 21.1 % of prefill batches and DP7
got 3.8 % — **max/min 5.49×, CV 0.494**, worsening monotonically through the run
(2.91× at 5 min → 5.49× at 17 min). The monotone DP0>…>DP7 gradient looks like a
dispatch/tie-break artefact rather than the cost function; three discriminating
experiments are listed in the analysis, none run.

Also recorded: this arm's **first attempt crashed** at GMU 0.80 with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` at `token usage: 0.05` — DP-attention
activation memory, not KV exhaustion. Fixed by *lowering* GMU to 0.70.

## Arm 3 — decode MTP OFF + decode radix cache ON (added 06:56–07:12 UTC)

A third arm at C=8 with **decode MTP off**, prefill and router **restored to the
arm-1 baseline**. Full treatment:
[`analysis/nomtp_radix_arm.md`](analysis/nomtp_radix_arm.md).

**MTP and the decode radix cache are one switch, not two.** SGLang rejects
`--disaggregation-decode-enable-radix-cache` under `--speculative-algorithm`, so
`infera/engine/sglang/args.py:261-278` appends it only when EAGLE is absent.
Turning MTP off is what legalises the decode radix cache; they cannot be varied
independently on this stack.

| C=8 | arm 1 (MTP + ChunkCache) | arm 3 (no MTP + RadixCache) |
|---|---:|---:|
| **TTFT p50 / p90** | 6,698 / 23,775 ms | **4,224 / 16,999 ms** |
| **ITL p50 / p90** | **13.26 / 18.49 ms** | 22.21 / 24.86 ms |
| E2E p50 / p90 | **13,874 / 38,774 ms** | 14,375 / 69,078 ms |
| **output throughput** | **258.5 tok/s** | 211.0 tok/s |
| server cache read/prompt | 50.8 % | 49.5 % |
| `submission_valid` | true | **true** |
| engine faults | 0 | **0** |

**The two axes move in opposite directions — one cause, not a trade-off.**

**⚠ The TTFT gain is not a server improvement. The bench is closed-loop and the
load got lighter.** `agentic_replay.py` issues turn N+1 from turn N's *return
callback* (:1214) and holds a lane's slot until its whole tree drains (:138), so
`--concurrency 8` is 8 serial chains in parallel and **arrival rate is an output
of server speed**. Measured from `request_start_ns`:

| | requests | span | **arrival rate** |
|---|---:|---:|---:|
| arm 1 | 231 | 897.3 s | **0.257 req/s** |
| arm 3 | 176 | 897.2 s | **0.196 req/s** (−24 %) |

```
MTP off → ITL +68 % (accept len was 3.06) → lane turnover slows
        → arrival -24 % → prefill #queue-req 2.00 → 0.50 → TTFT p50 -37 %
```

**Never read TTFT from this bench without the arrival rate beside it** — making
decode slower "improves" it. See `notes.md` Trap 11.

**The bottleneck did not move: prefill-bound in both arms.** Five-state
queue-depth analysis (`scripts/pd_bottleneck.py`, output in
`analysis/pd_bottleneck_arm1_vs_arm3.txt`): state 1 (`#queue-req`) is the only
deep queue; state 2 (`#inflight-req`) shallow ⇒ **transfer exonerated**;
`#prealloc-req` and `#retracted-req` are **0 on all 8 decode ranks**; decode
occupancy is **0.13 % / 0.27 %** of its 256-per-rank cap.

The ITL cost is the direct price of removing EAGLE: 13.26 → 22.21 ms is 1.68×,
about what a 3.06-accept draft predicts net of verify overhead. Throughput
follows ITL (−18 %) because at OSL mean 1,121 the decode loop dominates.

Also pinned this arm: `--num-reserved-decode-tokens` would have jumped 256 → 512
on its own, because `RESERVED_TOK` lives inside the leg script's MTP block.

## Navigate

| file | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | ordered, copy-pasteable reproduction + success criteria vs measured |
| [`environment.md`](environment.md) | nodes, image digests, fabric, the exact deployment, secrets needed |
| [`notes.md`](notes.md) | 5 traps and 5 open questions — the most re-read file |
| [`analysis/`](analysis/) | the turn-by-turn cache decomposition |
| [`patches/`](patches/) | the baked-in engine fixes, how they were verified, and the one env fix |
| [`spec/`](spec/) | the task brief + the customer's bench, verbatim |
| [`scripts/`](scripts/) | every script that ran |
| [`results/`](results/) | aiperf CSV/JSON per point, per-request records, kvd counters |
| [`env/`](env/) | per-node hardware/software snapshots |
| [`logs/`](logs/) | both engine logs + the driver log (gzipped) |

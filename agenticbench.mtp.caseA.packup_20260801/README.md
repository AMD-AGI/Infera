# Case A on the full-feature GLM-5.2 stack (spur) — reproduction kit

**Date:** 2026-08-01, 12:12–16:44 UTC. Measured window 15:37:08–16:43:56.
**Phase 2** of `spec/agentic.bench.kv.liying.mtp.md`.
Phase 1 (the 8-point `bench_serving` sweep) is packed separately in
`../agenticbench.mtp.sweep.packup_20260801/`.

## What this set out to prove

Run **Optimus-AgenticBench Case A** — the realistic agentic workload, 89 % shared
prefix, 74K/155K/235K input percentiles, 67-minute measured window — against the
most complete infera GLM-5.2 deployment on the **spur** cluster:
**kvaware + kvd + MTP + PD + DPA, all on at once.**

## Result at a glance

| | |
|---|---|
| duration | 4,007 s (ramp 400 + sustain 3,600 + drain) |
| requests | 2,881 sent / **2,811 completed** |
| **success rate** | **0.9757** (SLA 0.97 — **met**) |
| **TTFT p90** | **18,877 ms** sustain (SLA 30,000 — **met, 1.59× margin**) |
| **TPOT p50** | **17.9 ms** (vs 31.2 ms MTP-off reference — **1.74×**) |
| **cache hit** | **88.82 %** actual / 88.99 % ideal, efficiency 99.81 %, eviction 0.19 % |
| **MTP acceptance** | **2.736** engine-measured (healthy band; 4.00 would be the loop tell) |
| in-flight peak | **44 / 48** — never pinned |
| engine faults | **0** on both legs across the full window |

## The headline: two config defects, both already solved next door

**Case A failed twice on this cluster before it ran.** Neither failure was a new
bug, and neither was in the model, the fabric, or the workload. Both were
recorded in the vultr sibling kit (`../caseA.glm52.fullfeature.packup_20260801/`)
before the first attempt was launched:

| defect | symptom | fix | effect |
|---|---|---|---|
| prefill `--mem-fraction-static` 0.88 | `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, prefill DP0 aborts, 87-min watchdog hang | **→ 0.80** | per-rank free 33.8 GB → **284 GB** |
| DSA indexer IDLE-rank row underflow | `Expected lengths.size(0) == B` under MTP draft-extend | **`GLM52_P1V3`** (`patches/apply_p1v3.py`) | applied pre-emptively; 0 occurrences |

The `mem-fraction-static` direction is the counter-intuitive part: **prefill
activation OOM is fixed by *lowering* it** — the opposite of the decode-side
retract fix. Full mechanism, and three wrong root causes discarded along the way,
in [`notes/notes.config.md`](notes/notes.config.md).

Applicability of P1V3 was verified by digest, not assumed: this cluster's
in-image `dsa_indexer.py` md5 (`632f17acd38737459b43f830ee60ee89`) is
byte-identical to vultr's pre-patch file.

## Why the fixed-length sweep did not predict any of this

Phase 1 ran 8 rounds at **higher ISL and higher concurrency** (155K × conc 128)
and passed 8/8. It is not a superset of Case A — it is a different axis:

| | Phase 1 sweep | Case A |
|---|---|---|
| prompt shapes | homogeneous (`--random-range-ratio 1.0`) | ragged, breathing session population |
| prefix reuse | **none** — every prompt fresh, `cached_tokens` 0 | 88.8 % hit, kvd on the path |
| idle DP ranks | rare | constant |
| kvd `gets` | **0 for all 8 rounds** | on the path throughout |

The sweep proves peak throughput and correctness. Only the agentic profile
exercises ragged batches, prefix reuse, and idle ranks — which is exactly what
both defects needed.

## Feature proofs — each with the check that would go red

| feature | check | result |
|---|---|---|
| **PD + mooncake** | `MC_FORCE_TCP` count; `/v1/workers` shows both roles | **0**; PREFILL + DECODE both registered |
| **DPA** | `dp_size=8`, 8 `scheduler_DP*` per node | 8/8 both legs |
| **kvaware** | router pairs both legs, per-rank KV views | `active_workers: 2` throughout |
| **kvd** | adapter connected on prefill, absent on decode by design | **8** / **0** |
| **MTP** | `accept len` in 2.1–2.6; **4.00 is bad news** | **mean 2.736**, 4.00 only 3.0 % transient |
| **custom AR off** | `AiterCustomAllreduce` count | **0**; `NCCL` 16× both legs |

## kvd during the run — correct, but not exercised

| counter | before | after | Δ |
|---|---:|---:|---:|
| gets | 11,281 | 11,281 | **+0** |
| hits | 11,281 | 11,281 | +0 |
| **misses** | 0 | 0 | **+0** |
| sets | 349,692 | 376,791 | +27,099 |

**0 gets is the honest reading**, and it is a property of the workload, not a
defect: every Case A request nests in a prefix the in-GPU radix cache already
holds, so L3 is written but never read. kvd is proven *correct* here (0 misses,
ever), not *exercised*. The restart-and-replay proof — where `gets`/`hits` climb
with `sets` flat — is in the Phase-1 kit.

## Navigate

| file | read it for |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **the ordered, copy-pasteable reproduction** |
| [`environment.md`](environment.md) | nodes, fabric, image, SHAs, uncommitted patches, external paths, secrets |
| [`analysis/README.md`](analysis/README.md) | **headline + verdict against the Case A spec + the coherence check** |
| [`analysis/sli_percentiles.md`](analysis/sli_percentiles.md) | full percentile ladders recomputed from 2,811 raw samples; SLA verdict |
| [`analysis/yaml_vs_measured.md`](analysis/yaml_vs_measured.md) | every YAML knob vs. its measurement, with the conversion rule |
| [`analysis/mtp_comparison.md`](analysis/mtp_comparison.md) | MTP on vs off: what 2.736 accepted tokens/step bought, and what it did not |
| [`notes/notes.config.md`](notes/notes.config.md) | **the two defects: what / why / how / context** |
| [`patches/README.md`](patches/README.md) | **all three patches: what / why / how / context** |
| [`results/`](results/) | `summary.json`, `metrics.jsonl.gz`, kvd before/after |
| [`logs/`](logs/) | driver + both legs, **plus attempt 2's crash scene** |
| [`scripts/`](scripts/) | every script that ran, verbatim |
| [`env/`](env/) | `collect_env.sh` output per node — GPU/driver/CPU/RAM/kernel/fabric |

## Honest caveats

1. **No MTP-off arm was run on this image.** The 1.74× TPOT figure is a
   cross-run comparison against `agenticbench.glm52.spur.packup_20260731`, which
   also differs in image and prefill GMU (0.80 vs 0.88). The acceptance length
   (2.736) accounting arithmetically for the ratio is what makes the attribution
   credible; the clean ablation is still owed.
2. **The reference ran the SAME `--context-length 262144`**, so its input
   distribution was not truncated (its prompt p99 = 226,854 vs this run's
   225,782, max 260,013 in both). An earlier draft of this kit claimed the
   reference ran at 131,072 and clamped ~16 % of its inputs — that was wrong;
   131,072 is only the script's default and was overridden at launch. The two
   runs saw equivalent input work.
3. **TTFT is worse than the reference (p90 18,877 vs 8,652 ms), and MTP is not
   the cause.** MTP runs on decode only. The causes are ~2× the concurrency (live
   sessions 22.6 → 44.1 across sustain quarters; in-flight peak 44 vs the
   reference's 26–30) and the −10 % KV pool from GMU 0.80. Detailed in
   `analysis/mtp_comparison.md`.
4. **All 39 errors are the client's 240 s timeout**, hardcoded at
   `agent_throughput.py:929`. The prefill leg served **2,904/2,904 HTTP 200**;
   there are zero server-returned failures. The same timeout truncates
   `output_tokens.p99` to 9,688 vs the profile's 17,000 — a 17K-token generation
   needs 304 s of decode at TPOT p50 alone.
5. **`sla.e2e_p50_ms: 4500` has no measured counterpart.** `args.sla_cfg` is
   parsed and never consumed. Back-solved ≈12.0 s; stated as a gap rather than
   quoted as a result.
6. **Needle correctness was 3/5 then 4/5 across two runs, and this is sampling
   variance, not KV corruption.** The failing depths *moved* between runs (5 %/25 %
   failed run 1 and passed run 2; 75 % the reverse), and every depth returned its
   exact 7-digit needle at least once. Failures are `finish=length` repetition at
   the model's own `temperature 1.0 / top_p 0.95`. Short factual was 4/4 both times.
7. **No log was trimmed except by gzip.** Attempt 2's crash log is shipped whole.

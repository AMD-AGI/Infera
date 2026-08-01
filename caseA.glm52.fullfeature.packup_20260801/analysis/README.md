# Analysis — GLM-5.2 full-feature merged stack

Report for the mission in `mission.kv.liying.mtp.bench.md`: benchmark the most
complete infera GLM-5.2 deployment — **kvaware + kvd + MTP + PD + DPA, all on at
once** — and analyse every number.

Format follows
`../../../infera.glm5.2.experiment/agenticbench.glm52.spur.packup_20260731/analysis/`.

| file | what |
|---|---|
| [`sli_percentiles.md`](sli_percentiles.md) | Case A serving SLIs, full percentile ladders, recomputed from raw samples |
| [`yaml_vs_measured.md`](yaml_vs_measured.md) | every Case A YAML knob vs. what was measured, with the conversion rule |
| [`fixlen_analysis.md`](fixlen_analysis.md) | the 8-round fixlen sweep, every number |
| this file | headline result, the verdict, and what to do next |

---

## Headline

**The deployment works, and MTP is worth about 2× on decode latency — but the
shipped image cannot run the agentic workload without a patch we had to write.**

| | fixlen (Phase 1) | Case A (Phase 2) |
|---|---|---|
| requests | 660 | 2,988 sent / 2,952 completed |
| success | **100 %** | **98.8 %** |
| duration | 8 rounds | 4,006 s (67 min) |
| TTFT p50 / p90 | see table | 4,531 / 9,170 ms |
| TPOT p50 | 9.6 – 24.1 ms | **14.8 ms** |
| cache hit | *not measurable* (see below) | **89.2 %** (ideal 89.0, efficiency 100.3 %) |

## The five features, each with a positive signal

CLAUDE.md principle 1 requires proving each feature is genuinely on, because "a
green run that proves nothing is the default outcome here." All five verified
live:

| feature | signal | result |
|---|---|---|
| **PD** | `/v1/workers` shows both disagg modes | prefill + decode, both `active` |
| **DPA** | 8 live `sglang::scheduler_DP*` per node | 8/8 on both legs |
| **kv-aware** | per-rank pick distribution from the Rust router policy log | all 8 prefill + 8 decode ranks picked; prefill skewed (dp0 175 … dp3 35), decode near-uniform — the 20.0/2.0 weights visibly working |
| **MTP** | `accept len` in the decode log | mean **2.80** server-side / **2.04** per-request; healthy band, not the degenerate 4.00 |
| **kvd** | `statctl` gets/hits with sets flat | restart-replay: 102 gets / 102 hits / sets flat / 0 misses. Case A: **452 gets / 452 hits / 0 misses** |

## The finding that matters most

**Stock `infera/engine-sglang:merged-e` cannot complete Case A.** The decode leg
dies with

    RuntimeError: Expected lengths.size(0) == B to be true, but got false.

reproduced twice (at 125 s and 766 s into two independent full runs), in the DSA
indexer reached through the **MTP draft model**.

Root-caused live with `SGLANG_DEBUG_DSA_ROWS=1`: the image's own `GLM52_P1V2`
trim guards only `real < padded` (q_fp8 carrying DP-attention padding), but on a
DP **IDLE** rank under MTP draft-extend the inequality **inverts** —

    mode=IDLE q_fp8=(1,32,128) q_offset=2 ntnp=0 lengths=(2,) -> mqa_q=(1,...)

— so no trim runs, aiter sizes its logits from 1 row, and `fast_topk_v2` asserts
against 2 lengths entries. The patch (`GLM52_P1V3`) reconciles both sides to
`min(real, padded)`.

Full write-up: **`../notes/notes.dsa.mtp.crash.md`**. Patch:
`../patches/0004-*.txt`, applier `../scripts/apply_p1v3.py`.

**Every Case A number in this report is therefore `merged-e + P1V3`, not stock
merged-e.** This is stated in each file rather than buried.

The bug needs MTP **and** DP-attention **and** a rank going idle — which is why 8
fixlen rounds (660 requests, homogeneous batch shapes) never hit it, and an
agentic workload with a breathing session population hit it twice inside 13
minutes.

## MTP: the payoff, quantified

| | this run (MTP ON) | spur packup (MTP OFF) | ratio |
|---|---|---|---|
| TPOT p50 | **14.8 ms** | 31.3 ms | **2.11×** |
| success rate | 0.988 | 0.953 | — |
| client timeouts | 18 (0.6 %) | 96 (3.3 %) | — |

The acceptance length (mean **2.04** accepted tokens per step) is the mechanism,
and it accounts for the TPOT ratio almost exactly. The success-rate improvement is
downstream of the same effect: generations finish in half the time, so far fewer
cross the client's fixed 240 s deadline.

⚠️ **Not a controlled ablation.** The spur run differed in cluster fabric (mlx5 vs
ionic), kvd decode wiring, and `ctx`. The magnitude matching the measured
acceptance is what makes the attribution credible, not the comparison alone.

**The shipped `acc_len: 1.56` is conservative** — measured 2.04, 31 % better.

## kvd: proven correct, but Case A does not stress tiering

Case A delta: **+452 gets, +452 hits, 0 misses**, +72,844 sets, 53,576 evictions
once the 64 G L3 cap was reached.

Two honest qualifications:
1. **Phase 1 could not evaluate kvd at all.** `--dataset-name random` has no shared
   prefix; `cached_host_tok ≈ 0` in all 8 rounds, so the 0–50 % hit rates are
   GPU-radix residue from the previous round. Detailed in `fixlen_analysis.md`.
2. **Case A exercises cache *accounting*, not cache *tiering*.** Every request
   nests inside the same shared prefix, so the radix tree holds one hot path —
   hence 72,844 sets against only 452 gets. The guide says this explicitly. A
   growing-prefix workload (`code_agent_200k.yaml`) would be needed to stress
   L2/L3 promotion.

## Load calibration: the population prediction held

The 1,000 s probe measured a steady population of ~17 against a nominal 32. That
was diagnosed as **tail censoring, not a rate error** — a short window cannot
realize an inter-turn delay whose p99 is 240 s nor a 103-turn session — with the
prediction that the full 4,000 s window would let N rise at the *unchanged* rate.

It did: live sessions by quarter **22.1 → 22.3 → 27.6 → 36.0**, max session
lifetime 3,978 s vs the probe's window-bound 1,005 s.

`new_session_rate` stayed at the shipped 0.10 throughout (the guide's own Step-3
re-solve gives 0.094 — a 6 % move, inside noise). **Had the probe's N=17 been
"corrected" by doubling the rate, the full run would have pinned `max_inflight`
and backpressure would have set the load** — the exact failure the spur run hit.
In-flight peaked at 30 against a cap of 48; the cap never bound.

## Verdict against the mission

| mission requirement | status |
|---|---|
| verify each feature genuinely on before spending a measured window | **done** — 6-row feature proof, all green, before any timed run |
| set kv-aware weight to a suitable value, record data explaining it | **done** — pw=20.0 dw=2.0; per-rank pick distributions recorded, prefill skewed / decode uniform as the weights intend |
| fixlen sweep, paired percentiles, P99 dropped, one server, conc 1/32/64/128 | **done** — 8 rounds, 660/660, `fixlen_analysis.md` |
| packup after fixlen | **done** — `fixlen.glm52.fullfeature.packup_20260801/` |
| Case A agentic bench | **done** — 4,006 s, 0.988 success (after the P1V3 patch) |
| packup after Case A | **pending** |
| analysis of every value, in the reference format | **this directory** |

## Open items, stated rather than smoothed over

1. **`GLM52_P1V3` is our patch, not upstream.** It should be filed — the image
   already carries the instrumentation that proves the bug (`SGLANG_DEBUG_DSA_ROWS`)
   and its own comment admits the two bookkeeping sources "ha[ve] never been
   measured to agree ... on the MTP draft-extend path."
2. **p50 fixlen rounds ran at gmu 0.88, p90 rounds at 0.80.** Within-pair
   comparisons are clean; cross-pair carries a 13 % KV-pool confound. A ~15 min
   re-run of the p50 pair at 0.80 would close it.
3. **`sla.e2e_p50_ms: 4500` is missed 2.5×** (derived p50 11.1 s). The target is
   not gated and appears written against a much shorter generation than the
   profile produces. Flagged, not adjusted.
4. **Case A's turn-count p99 is still window-truncated** even at 3,600 s sustain.
   The guide calls this the honest minimum; it is honest but not generous.
5. **kvd tiering remains unmeasured.** Needs a growing-prefix workload.

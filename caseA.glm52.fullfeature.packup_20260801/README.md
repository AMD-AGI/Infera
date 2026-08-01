# Case A on the full-feature GLM-5.2 merged stack — reproduction kit

**Date:** 2026-08-01, 12:15–14:41 UTC.
**Phase 2** of `mission.kv.liying.mtp.bench.md` (Phase 1 = the fixlen sweep, packed
separately in `../fixlen.glm52.fullfeature.packup_20260801/`).

## What this set out to prove

Run Optimus-AgenticBench **Case A** — the realistic agentic workload, 89 % shared
prefix, 74K/155K/235K input percentiles, 67-minute measured window — against the
most complete infera GLM-5.2 deployment: **kvaware + kvd + MTP + PD + DPA, all
enabled at once.** No prior kit had measured this combination.

## Result at a glance

| | |
|---|---|
| duration | 4,006 s (ramp 400 + sustain 3600 + drain) |
| requests | 2,988 sent / 2,952 completed |
| **success rate** | **0.988** (SLA 0.97 — **met**) |
| **TTFT p90** | **9,170 ms** (SLA 30,000 — **met, 3.3× margin**) |
| **TPOT p50** | **14.8 ms** (vs 31.3 ms MTP-off reference — **2.11×**) |
| **cache hit** | **89.2 %** actual / 89.0 % ideal, efficiency 100.3 %, eviction 0.0 % |
| MTP acceptance | **2.04** per-request mean (config assumes 1.56) |
| kvd | 452 gets / **452 hits** / **0 misses** |
| in-flight peak | 30 against a cap of 48 — **never bound** |

## ⚠️ The headline finding: the shipped image cannot run this workload

**Stock `infera/engine-sglang:merged-e` crashes the decode leg under Case A.**
Reproduced twice — at 125 s and at 766 s into two independent full runs:

    RuntimeError: Expected lengths.size(0) == B to be true, but got false.

in the DSA indexer, reached through the **MTP draft model** (`deepseek_nextn`).

Root-caused live with the image's own `SGLANG_DEBUG_DSA_ROWS=1` instrumentation.
The in-image `GLM52_P1V2` trim guards only `real < padded`; on a DP-attention
**IDLE** rank under MTP draft-extend the inequality inverts, so no trim runs and
the top-k kernel gets 1 score row against 2 lengths entries.

**All results here are `merged-e` + our `GLM52_P1V3` patch** (`patches/0004`,
applier `scripts/apply_p1v3.py`). Full analysis: `notes/notes.dsa.mtp.crash.md`.

The bug needs MTP **and** DP-attention **and** an idle rank simultaneously —
which is why Phase 1's 660 fixed-shape requests never hit it, and an agentic
workload with a breathing session population hit it twice in 13 minutes.

## Navigate

| file | read it for |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **the ordered, copy-pasteable reproduction** |
| [`environment.md`](environment.md) | hardware, fabric, image, SHAs, external paths, secrets |
| [`analysis/README.md`](analysis/README.md) | headline + verdict against the mission |
| [`analysis/sli_percentiles.md`](analysis/sli_percentiles.md) | full percentile ladders, recomputed from raw samples |
| [`analysis/yaml_vs_measured.md`](analysis/yaml_vs_measured.md) | every YAML knob vs. measurement |
| [`analysis/fixlen_analysis.md`](analysis/fixlen_analysis.md) | Phase 1's 8 rounds (context for this one) |
| [`notes/notes.dsa.mtp.crash.md`](notes/notes.dsa.mtp.crash.md) | **the crash: mechanism, live evidence, fix** |
| [`notes/phase2_caseA.md`](notes/phase2_caseA.md) | the working log — probe, Step-3 re-solve, all three attempts |
| [`patches/`](patches/) | the P1V3 patch, with what/why/how/context |
| [`results/`](results/) | `summary.json`, `metrics.jsonl.gz`, kvd before/after |
| [`logs/`](logs/) | driver + both engine legs, incl. **both crash scenes** |
| [`spec/`](spec/) | the mission, `CASE_AB_GUIDE.md`, the workload YAML |

## Layout

    README.md              this file
    REPRODUCE.md           exact ordered steps
    environment.md         hw/sw/fabric/paths/secrets
    scripts/               every script that ran, verbatim
    patches/               0004 = GLM52_P1V3 (load-bearing: nothing runs without it)
    results/               summary.json, metrics.jsonl.gz, kvd snapshots
      raw/                 the 1,000 s probe run's artifacts
    analysis/              the four report files
    notes/                 crash write-up + working log
    logs/                  gzipped: driver x4, prefill x1, decode x2
    env/                   collect_env output per node
    spec/                  mission + guide + workload YAML

## Honest caveats

1. **`GLM52_P1V3` is ours, not upstream.** Worth filing — the image already ships
   the instrumentation that proves the bug and a comment admitting the two
   bookkeeping sources were never verified to agree on this path.
2. **The MTP 2.11× is not a controlled ablation.** The MTP-off reference is the
   spur packup, which also differed in fabric (mlx5 vs ionic) and kvd decode
   wiring. The acceptance length (2.04) accounting for the ratio is what makes the
   attribution credible.
3. **Case A does not stress cache tiering.** Every request nests in the *same*
   prefix (72,844 kvd sets vs 452 gets). kvd is proven *correct*, not *exercised*.
4. **Turn-count p99 is still window-truncated** at 3,600 s sustain.
5. **`sla.e2e_p50_ms: 4500` missed 2.5×** (derived 11.1 s); the target is not
   gated and looks written against a shorter generation than the profile produces.
6. **`p6_decode.log` is shipped with its `[dsa-rows]` lines stripped** — the
   diagnostic was left on for the passing run and produced 144 MB. Everything
   else in that log is intact; the diagnostic lines that matter are preserved in
   `p5_decode.dsarows_tail3000.log.gz` (crash scene #2).

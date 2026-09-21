# GLM-5.2 1P1D AgentX — T2 concurrency sweep under simulated MTP acceptance

**Ran:** 2026-09-19 to 2026-09-20 (UTC)
**Author:** yihou (yihou@amd.com)
**Status:** **PARTIAL — 4/5 concurrency points obtained (40, 56, 72, 96); CONC=128 not obtained.**

## Goal

Measure serving **timing** (throughput, TTFT, ITL, E2E latency) of a GLM-5.2-MXFP4
1P1D disaggregated deployment (P4D4 on MI355X) across an AgentX concurrency sweep
in **full mode** (3600 s profiling, warmup 10/lane) under **simulated MTP
acceptance** (`SGLANG_SIMULATE_ACC_LEN=3.61`). This is the T2 half of the task;
T1 (`glm52-agentx-c40-realacc.packup_20260918`) measured real acceptance and
correctness. **T2 measures timing, not correctness** — see "Honest limitations".

**Spec:** `spec/task-spec.CLAUDE.md` (and `spec/mission.md`).
**Success criterion (from the spec):** pack up the sweep 40/56/72/96/128; "128
may OOM — stopping there is an acceptable end."

## Result

Main arm **t2f** = the "corrected MTP config" (`--disable-custom-all-reduce` on
decode) + `index_share_for_mtp_iteration=false`, full mode, simulated 3.61, base
image, HiCache on. Numbers: `results/t2-summary.csv`.

| CONC | total tput (tok/s) | per-GPU (tok/s) | TTFT mean / p50 (s) | E2E mean (s) | ITL mean (ms) | srv cache-hit | verdict |
|---|---|---|---|---|---|---|---|
| 40 | 158,390 | 19,799 | 7.41 / 4.17 | 21.17 | 13.58 | 95.2% | ✅ obtained |
| 56 | 117,407 | 14,676 | 38.74 / 17.64 | 52.47 | 13.77 | 86.8% | ✅ obtained |
| 72 | 70,111 | 8,764 | 124.52 / 85.21 | 135.98 | 12.10 | 75.3% | ✅ obtained |
| 96 | 24,225 | 3,028 | 418.61 / 44.67 | 429.54 | 11.08 | 22.3% | ✅ obtained |
| 128 | — | — | — | — | — | — | ❌ **not obtained** (twice, non-GPU) |

**The only usable operating point is CONC=40.** Throughput *collapses* as
concurrency rises (CONC=96 is 15% of CONC=40) and TTFT explodes (7.4 s → 418.6 s
mean). This is **KV-capacity / queueing saturation, not compute** — ITL actually
*falls* (13.8 → 11.1 ms) while the server-measured GPU-cache hit rate collapses
93.9% → 11.5%. See "Honest limitations" #3.

**CONC=128 failed twice for two unrelated, non-GPU reasons** (NOT an OOM):
(1) the shared `/home` NFS volume hit 0 bytes and killed the client tree;
(2) a Mooncake PD KV-transfer wedge on the decode leg. Evidence in
`results/t2f-c128-hang/`. See "Honest limitations" #2.

### What T2 did and did NOT measure — PR #37152 was never exercised

The original task added upstream sglang **PR #37152** (HiCache ROCm copy-round
widening) to be exercised under HiCache. **It was measured in neither test:**

- In **T1**, `config.sh` sets `PREFILL_HICACHE=0`, so the patched path never
  executes — the patch is inert by construction.
- In **T2**, HiCache is ON (`config.full.sh:57` `PREFILL_HICACHE=1`; the launch
  carries `--hicache-ratio 1.5 --hicache-io-backend kernel --hicache-mem-layout
  page_first --hicache-write-policy write_through`), so the HiCache path DOES
  execute — but the **image does not contain the patch**. `config.yihou.full.sh:74`
  forces `IMAGE=infera-sglang:v0519-yihou-0917` (the plain base, no thin layer).

This is a **deliberate user instruction, not an oversight**: during the GPU-fault
investigation the user said to use the old image ("用老镜像"), and
`config.yihou.full.sh` was reverted to the C40-aligned base accordingly. The
consequence is that **T2's numbers describe HiCache-on timing on STOCK kernels** —
a valid and useful measurement, just not the one PR #37152 was added for. (Same
image revert also drops the NextN shared-experts-fusion fix; immaterial to timing
under forced acceptance.) Full detail in `environment.md`; a stale in-config
comment about this is flagged in `notes.md` §10.

## Honest limitations (read before quoting any number)

1. **Forced acceptance.** Every T2 number came from `SGLANG_SIMULATE_ACC_LEN=3.61`,
   which *forces* acceptance (`spec_utils.py:401-434`) rather than reporting it.
   The emitted text is unverified draft output, so **OSL and every
   token-count-derived figure describe the draft, not the target.** Do NOT quote
   T2's `accept_length` as a real measurement — T1 measured that (min 2.15 / mean
   2.59, real). T2 is timing only.
2. **CONC=128 not obtained.** Two attempts, ~1 h each, two *unrelated non-GPU*
   causes: (a) `/home` NFS hit 0 bytes; (b) Mooncake PD KV-transfer wedge on
   decode at 00:53:59. Neither says whether CONC=128 is runnable. **NOT an OOM** —
   the mission anticipated a memory failure; this was not one.
3. **CONC 56+ are KV-capacity-limited, not compute-limited.** ITL *falls*
   monotonically 13.8 → 11.1 ms from CONC 40→96 while throughput collapses; at
   CONC=96 server prefix/GPU-cache hit is 22.3% overall (11.5% GPU) vs 95.7%
   theoretical. The bottleneck is queueing/capacity. **CONC=40 is the only usable
   operating point.**
4. **CONC=96's TTFT p50 (44.67 s) is lower than CONC=72's (85.21 s) only because
   fewer and shorter requests completed** (808 profiled vs 1996). The honest
   number is the **mean, 418.61 s**. Both are in the table.
5. **The t2e CONC=40/56 points are valid timing data**, not garbage — the
   custom-all-reduce-**ON** arm of a single-variable A/B against t2f
   (+0.65% at CONC 40, −4.2% at CONC 56). `results/t2e/`.
6. **`index_share_for_mtp_iteration=false` is a mitigation, in force for every
   point** — correlational (0 GPU faults in ~17 h across three runs / two TP
   shapes, vs 5 faults with it on), no faulting kernel identified. Full analysis:
   `spec/dsa.topk.indexer.bug.analysis.md`.
7. **Monitoring blind spot (methodology lesson).** `t2f-launch/server-logs/*.log`
   stopped being written at 23:42 when its follower died in the NFS event; ~90 min
   of watcher `faults=0` came from that dead file before it was caught. Scan
   `docker logs <container>` directly, not an on-disk follower. See notes.md.
8. **PR #37152 / the thin image layer were not exercised** — T2 ran the base
   image with HiCache on stock kernels (see "What T2 did and did NOT measure"
   above, and `environment.md`).

## How to reproduce

See `REPRODUCE.md`. TL;DR: bring up the P4D4 stack with `config.yihou.full.dcar.sh`,
run `sweep.yihou.dcar.sh 40 56 72 96` against the live router.

## Folder map

Total ≈ **35 MB** (T1's kit was 7.1 MB; the difference is the four gzipped
per-request `profile_export.jsonl` + timeslices JSON, ~34 MB, needed for the OSL /
error analysis). **Two artifact families were EXCLUDED**, originals in the
gitignored workspace `bench/glm5p2_pd/results/yihou-agentx-hicache/`:
`server_metrics_export.json` (~11 GB, summarised into `t2-summary.csv`) and the
timeslices `.csv` (redundant with the included `.json`). See `results/README.md`.

- `REPRODUCE.md` — step-by-step reproduction
- `environment.md` — exact HW/SW/shape the numbers came from (+ the base-image caveat)
- `scripts/` — configs, sweep, gate, watcher, env + summary tools (verbatim)
- `patches/` — the `engine.sh` `DECODE_EXTRA_ARGS` escape hatch
- `results/` — `t2-summary.csv`, per-point aggregate JSONs (t2f + t2e A/B), gzipped per-request `t2f/aiperf/`, server-info, the CONC=128 hang evidence
- `notes.md` — gotchas, the two CONC=128 failures, the watcher blind spot, wrong turns
- `spec/` — task spec, mission, working narrative (`sweep-driver-log.md`), DSA fault analysis
- `env/` — per-node HW/fabric snapshots

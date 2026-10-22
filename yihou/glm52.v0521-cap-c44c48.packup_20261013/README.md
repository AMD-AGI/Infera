# GLM-5.2-MXFP4 v0.5.21 1P1D P4D4 — capacity + concurrency (C44/C48/C52) and decode in-flight KV cap — best **22,306 tok/s/chip (C48)**

**Ran:** 2026-10-12 06:47 → 17:18 UTC. **Nodes:** `crsuse2-m2m-102` (spur oai job 207495, ended 2026-10-12 20:01 TIMEOUT) and
`crsuse2-m2m-137` (plain ssh, user-approved 10:50). Burst nodes 174/029 were cancelled at 20 min (amd-burst-qos
PreemptMode=cancel); 205 had a foreign GPU pool. **Author:** yihou (with Claude).
**Status:** IN PROGRESS when packed (user is migrating the session). Goal > 23,000/chip **not reached**; best 22,306.

## Goal (spec/mission.md, user 2026-10-12)

1. Combine the HiCache-450 GB result with the decode KV-capacity report to raise prefill + decode capacity; raise AgentX
   CONC to 44 (≥ the C40 result 22,114) then 48; push throughput **> 23k tok/s/chip**.
2. Track B: a decode-side rule so running requests always keep ≥ 50 % of the decode KV pool — user formula:
   admit only if `(requested + held KV) − running KV < 50 % of the pool`; debug until C40 runs with no regression.
3. Later/last: index_share_for_mtp_iteration=true (not run yet).

## Results (first-hand, `results/analysis/runs_summary.csv`; score = AgentX full, 3600 s, tok/s/chip)

| run | node | stack | config delta vs e2g recipe (HiCache 450, decode 0.95, max_running 64) | CONC | tok/s/chip | qps | mean ISL | TTFT mean / p95 s |
|---|---|---|---|---|---|---|---|---|
| **q1 A3** | 137 | warm (3rd run) | decode max_running **96**, prefill mem **0.88** | **48** | **22,306** | 1.220 | 145.3k | 11.97 / 38.2 |
| q1 A2 | 137 | warm | same | 44 | 21,862 | 1.208 | 143.7k | 7.79 / 28.2 |
| q1 A4 | 137 | warm (4th run) | same | 52 | 21,418 | 1.256 | 135.5k | 15.82 / 52.0 |
| q1 A1 | 137 | fresh | same | 44 | 21,433 | 1.195 | 142.5k | 5.91 / 19.1 |
| X1 | 102 | e2j stack 3rd run | (+hostvariant, graph bs 1..16) | 44 | 21,773 | 1.207 | 143.3k | 9.02 / 30.6 |
| k1 A2 / A1 | 102 | warm / fresh | HiCache **360** | 44 | 21,342 / 20,729 | 1.186 | 143.0k | 8.55 / 28.9 |
| pc40c A2 / A1 | 102 | warm / fresh | runshare **v1** 0.5 (decode patch) | 40 | 21,271 / 20,361 | 1.160 | 145.7k | 6.17 / 20.9 |
| pc40d A1 | 102 | fresh | runshare **v2** 0.5 (skip + 5 s anti-starvation) | 40 | 20,680 | 1.137 | 144.5k | 5.26 / 18.3 |
| reference e2g A2 | 102 | warm | — | 40 | 22,114 | 1.181 | 148.7k | 5.30 / 19.2 |

## Key findings

1. **C48 is the sweet spot** for the q1 config (C44 21,862 → C48 22,306 → C52 21,418). Score ≈ qps × mean ISL; more lanes
   raise qps but lower mean ISL and add ~4 s TTFT per +4 lanes.
2. **The limiter is the prefill side (queue + KV send), not decode compute** (`results/analysis/q1_residence_c44_c48_c52.txt`,
   Little's law on 5-s /metrics): from C44 → C48 → C52 prefill queue residence 2.0 → 3.6 → 5.2 s, KV send (prefill
   inflight) 4.5 → 6.5 → 8.2 s, while decode running residence stays ~12-13 s and mean decode running requests stay
   ~15 (all 4 DPs). Prefill per-batch cache hit falls 0.71 → 0.63 → 0.57 and the 450 GB host pool is full after
   ~50 → ~36 → ~30 min.
3. **Decode KV also gets tight at C48/C52** (per-DP token usage max 0.99 / 0.98; first prealloc queue at C52), with 0
   retractions. decode max_running 96 is needed (C48 DP3 ran 18 > old 16/DP cap).
4. **HiCache 360 < 450** at C44 (21,342 vs 21,773/21,862) → keep 450 (or larger).
5. **Track B patch works functionally** (230k-token burst: in-flight capped at 0.50 / 0.42 of the pool, 64/64 ok, 0 faults),
   but **v1 regressed C40** (warm 21,271 vs 22,114; FIFO head-of-line behind ~0.7 M-token prompts). v2 fresh C40 20,680
   (top of the fresh band) — its scored warm A2 was **never run** (driver host lost at 17:18; node 102 job ended 20:01).
6. Decode radix cache (would cut KV send) is blocked by EAGLE in v0.5.21 (`pd_disaggregation_hook.py:97-113`).

## Folder map

- `spec/` mission (all user directives), PLAN (v1-v5), task CLAUDE.md. `analysis/` ledger `working_process.md`, `poll.md`,
  `research/` R1 (decode admission code), R2 (KV occupancy logs), R3 (levers), R5 (upstream) + adversarial verifications,
  `decode.kvcache.report.md`.
- `scripts/harness/` the full harness (verbatim; env/config/chain files of every round), `scripts/tools/` metrics sampler,
  long-prompt burst, Little's-law residence, this pack-up script.
- `patches/` runshare v1/v2 decode.py mounts (+ diffs vs the image), configs. `results/` per-run JSON/benchmark logs,
  round files, `/metrics` TSVs (gz), analysis. `logs/` gzipped server logs (all < 4 MB).
- `REPRODUCE.md`, `environment.md`, `notes.md` (gotchas, open items, next steps).

## Related

- Baseline kit (22,114 C40, warm-stack protocol, image recipe): `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`.
- `/home/yihou/dev/git/infera.glm52.view/decode.kvcache.report.md` (copied in `analysis/`).

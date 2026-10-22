# GLM-5.2 v0.5.21 same-node P4D4: decode in-flight KV cap ("runshare v2") vs no patch at AgentX C56 / C64, HiCache = original C40 capacity

**Ran:** 2026-10-13 06:36 -> 16:48 UTC, two rounds back-to-back on ONE node, `crsuse2-m2m-085` (10.245.153.145, spur
`amd-oai` job 211169). **Packed:** 2026-10-14. **Author:** yihou (with Claude). **Status: COMPLETE as a single-run A/B;
no measured benefit from the patch.**

## Goal (user directive 2026-10-13 ~05:15 UTC, verbatim in `spec/directive.20261013.md`)

> 2. 关于decode限制transfer和prealloc cache那个事情。实验验证一下直接用最初conc=40那一版的hi-cache容量压conc=56/64,
> 期望是比~/dev/git/infera.glm52.pd/yihou之前sweep的性能回退少一点。

Reading: run the decode in-flight transfer/prealloc KV cap (runshare v2, share 0.5, 5 s anti-starvation) with the
**original C40 HiCache capacity** (`--hicache-ratio 1.5`, no `--hicache-size` = ~154 GB-class per DP rank,
3,649,664 host tokens/rank here) at **CONC 56 and 64**. Expectation (user): less regression than the old sweep t2f
(v0.5.19 cross-node, same HiCache class: C40 19,799 -> C56 14,676 = -25.9 %). The user then asked for the no-patch
control on the same node ("要", ~11:55), round `tb56n`.

## Result (score = `request_metrics.throughput.per_gpu.total_tput_tps`, tok/s/chip; all [M] from `results/*/agentx_conc*.json`)

| run | CONC | stack state | tb56 (runshare v2) | tb56n (no patch) | delta patch vs none |
|---|---|---|---|---|---|
| A1 | 56 | fresh (unscored) | 17,545 | 17,791 | -1.4 % |
| **A2** | **56** | **warm, after flush (scored)** | **17,611** | **18,016** | **-2.2 %** |
| A3 | 64 | warm, 3rd run, after flush | 15,070 | 15,108 | -0.3 % |
| C56 A2 -> C64 A3 | | | -14.4 % | -16.1 % | |

Details per run (qps, ISL, TTFT p50/p95, TPOT, GPU/CPU hit) in `results/results.csv`.

Against the criteria:
1. **Patch active** — yes [M]: decode container has the runshare2 mount + `YIHOU_DECODE_RUNNING_KV_SHARE=0.5`,
   `YIHOU_RUNSHARE_MAX_WAIT_S=5`; decode log has 266,980 `YIHOU-RUNSHARE block` lines (each line = every 100th blocked
   admission check of one DP rank; 74,670 `skip`, 192,310 `stop`); decode log "pre-allocated usage" max **0.50** with
   the patch vs **0.98** without.
2. **"Less regression than the old sweep"** — **not shown.** Same-node A/B differences are -2.2 % (C56 A2) and -0.3 %
   (C64 A3), both in the patch's disfavour, single runs each. The C56->C64 drop is -14.4 % (patch) vs -16.1 % (none).
   There is **no C40 point at this exact config on 085**, so the C40->C56 regression of this config is not measured;
   the t2f comparison differs in sglang version (0.5.19), topology (cross-node), decode mem (0.85), max_running (128),
   DSA backend (tilelang) — see `analysis/ab_tb56_vs_tb56n.md`.
3. Decode running KV under load: 0 retractions in all 6 runs (both rounds), decode `token_usage` max 0.91-1.00 per DP
   in every run [M, /metrics 5 s].

What this does NOT say: whether the patch helps at other loads, other HiCache sizes, or with replicates (each cell is
n = 1). For scale: four warm C40 replicates of the e2g recipe on node 102 spanned 21,681-22,114 (2.0 %) [M, mission
ledger], so a 2.2 % single-run gap is about the size of run-to-run spread seen before; whether it is noise is not measured.

## Folder map

| path | content |
|---|---|
| `spec/` | verbatim 2026-10-13 directive (`directive.20261013.md`), full mission ledger (`mission.full.md`) |
| `REPRODUCE.md` | ordered steps: hold node -> checks -> image -> JIT seed -> env -> dryrun/tokdiff -> chain -> score |
| `environment.md`, `environment-raw/` | node 085 hardware/driver/RDMA/image digest; gaps; raw collect_env output |
| `scripts/harness/` | verbatim harness used (bench-harness core, run.v3, pipeline.v3, chain_cap2, stop_then_chain, configs, mounts, env files, audit v3, tools/pidcensus) |
| `scripts/session/` | local ssh shim, metrics sampler, tools_summ/tools_cmp, collect_env, the two dryrun renderings, hold_node.sh (spur hold job body) |
| `scripts/make_results_csv.yihou.py` | builds `results/results.csv` from the JSONs in this packup |
| `patches/` | runshare v2 decode.py + diff vs image + overlay/mount; `patches/README.md` what/why/how/context |
| `results/tb56/`, `results/tb56n/` | per-run AgentX JSON, benchmark log (gz), command, runtime.env, plots, service JSONs; round status/audit/census/chain/pipeline logs |
| `results/metrics/` | decode+prefill /metrics TSVs (5 s), gzipped, per run |
| `results/reference/` | t2f (v0.5.19 sweep) JSONs, q1 (C44/C48/C52, HiCache 450) JSONs + dryrun, pc40d A1 (runshare v2 at C40) |
| `results/results.csv`, `results/summary.v0521cap2.yihou.md` | all metrics; the session-wide table |
| `analysis/` | A/B analysis, old-sweep research (`res_sweep.md`), session ledgers (working_process, poll) |
| `notes.md` | gotchas and wrong turns |
| `logs/LOG_INDEX.md` | uncopied server logs / aiperf artifacts with absolute paths + sizes |
| `VERIFY.md` | cold-read verification by a second agent: checks run, fixes, open items |
| `SHA256SUMS` | over every file in this packup |

## Sibling packups (same parent dir)

- `../glm52.v0521-decode-transfer-kvcap.packup_20261013/` — runshare design, v1 -> v2 history, C40 rounds pc40c/pc40d,
  long-prompt burst smoke (cap held at 0.50 / 0.42). **Read it for the patch design; not duplicated here.**
- `../glm52.v0521-cap-c44c48.packup_20261013/` — q1 recipe (max_running 96, prefill mem 0.88) this round builds on;
  warm-stack protocol; image load notes.
- `../glm52-agentx-t2sweep-simacc.packup_20260920/` — the old sweep t2f/t2e (C40/C56/C72/C96).

# GLM-5.2-MXFP4 v0.5.21 1P1D P4D4 — prefill mem-fraction 0.90/0.92, HiCache `write_through_selective`, pc64, correctness round (2026-10-14)

**Ran:** 2026-10-14 01:52 → 08:00 UTC (this packup's cut-off) on spur oai nodes `crsuse2-m2m-089` (job 213048),
`crsuse2-m2m-085` (job 211169), `crsuse2-m2m-274` (job 211171) and plain-ssh node `crsuse2-m2m-136`. **Author:** yihou (with Claude).
**Packed:** 2026-10-14 ~08:00 UTC, copy-only (nothing re-run). **Status:** PARTIAL — pf90, wts48, pc64-A1 finished;
pf92 A2 / pf88 lost to node failures; wts64, cx90 (correctness) and pf94/pf93 were **in flight** at packing time.

## Goals (verbatim user directives in `spec/directives.md`)

1. "按照你的想法测0.90和0.92 prefill kv" — raise prefill `--mem-fraction-static` 0.88 → 0.90 / 0.92 on the q1 config (bigger prefill device KV pool).
2. "找我们hi-cache=450的最好的实验结果，基于此，改用selective throughput" — q1 (best HiCache-450 run, C48 22,306) with prefill HiCache `write_through_selective`.
3. "在pc48B的基础上，跑conc=64" — pc48b (prefill_complete) at C64. Later: "哦那放弃这条优化路径" (abandoned).
4. "直觉上write_through——selective应该支持更高的conc, 测一下wts64 … 增大gmu … 这些空间确实被挤压的一滴不剩了么？你调研一下 … 在0.92下不断增加 … 直到oom" — wts64; prefill memory accounting; mem-fraction probe above 0.92.
5. "插空可以在我们现在最高的吞吐的配置上做一些正确性的测试" — correctness round cx90 on the pf90 config, simulated acceptance OFF.

No numeric success target was given; the reference is q1 A3 C48 **22,306** (137, write_through, prefill mem 0.88).

## Headline (first-hand, `results/results.csv`; score = `request_metrics.throughput.per_gpu.total_tput_tps`, AgentX 3600 s)

All rows: same-node 1P1D P4D4, TP4 DP4 DPA per leg, EAGLE 5/1/6 with `SGLANG_SIMULATE_ACC_LEN=3.61` (correctness waived),
decode max_running 96, decode mem 0.95 unless noted, prefill HiCache 450 GB unless noted. A1 = fresh stack (unscored), A2 = after `/flush_cache` (scored).

| run | node | change vs q1 | prefill KV tok/rank | tok/s/chip | qps | TTFT mean / p95 s | TPOT ms | GPU / CPU hit | ok / err |
|---|---|---|---|---|---|---|---|---|---|
| q1 A3 (ref) | 137 | — (mem 0.88, write_through) | 2,433,088 | 22,306 | 1.220 | 11.97 / 38.19 | 12.1 | 0.839 / 0.088 | 4425 / 2 |
| pf90 A1 | 089 | prefill mem 0.90 | 2,541,376 | 21,859 | 1.219 | 9.57 / 31.32 | 15.7 | 0.911 / 0.035 | 4415 / 0 |
| **pf90 A2** | 089 | prefill mem 0.90 | 2,541,376 | **23,127** | 1.276 | 9.30 / 28.76 | 13.1 | 0.860 / 0.071 | 4632 / 3 |
| pf92 A1 | 085 | prefill mem 0.92 | 2,649,600 | 23,003 | 1.267 | 7.49 / 23.90 | 15.6 | 0.930 / 0.024 | 4593 / 1 |
| pf92 A2 | 085 | prefill mem 0.92 | 2,649,600 | **LOST** (job 211169 ended 05:07 NODE_FAIL at RunTime ~24 h, ~70 min into A2; 24 h QOS MaxWall as cause is [inferred]) | | | | | |
| pf88 A1 | 274 | same-session 0.88 control | 2,433,088 | **LOST** (node 274 NODE_FAIL 03:10; logs end 02:32, still in warm-up; JSON has 0 profiled / 84 errors) | | | | | |
| wts48 A1 | 136 | write_through_selective | 2,433,088 | killed at ~50 min (login-host driver terminated 04:47, cause not established), no JSON | | | | | |
| **wts48 A2** | 136 | write_through_selective | 2,433,088 | **21,226** | 1.169 | 13.63 / 42.03 | 12.5 | 0.868 / 0.039 | 4243 / 1 |
| wts64 (C64) | 136 | write_through_selective, C64 | 2,433,088 | **in flight** at packing (started 07:17, warm-up 360/707 at 07:59) | | | | | |
| pc48b A2 (ref) | 274 | prefill_complete, write_back 370, decode 0.96 | 2,433,088 | 20,002 | 1.112 | 16.57 / 52.70 | 12.2 | 0.811 / 0.101 | 4027 / 3 |
| pc64 A1 | 089 | pc48b config at C64 | 2,433,088 | 10,480 | 0.664 | 82.77 / 262.94 | 13.9 | 0.493 / 0.341 | 2409 / 2 |
| pc64 A2 | 089 | — | — | aborted by user decision (path abandoned) | | | | | |
| cx90 | 089 | pf90 config, simulated acceptance OFF | — | **in flight** (correctness tests; waiting for VRAM release at 07:59) | | | | | |
| pf94 / pf93 | 136 | prefill mem 0.94 / 0.93 | — | **armed, not started** (`after_wts64_pf94.yihou.sh` waits on wts64) | | | | | |

## Findings (measured; mechanisms not established unless stated)

1. **pf90 A2 = 23,127/chip, the best real-model (256-expert) score so far**, +3.7 % over q1 A3 22,306. Not a controlled A/B:
   different node (089 vs 137), and q1 A3 was the 3rd run of its stack vs pf90 A2 the 2nd. The same-session 0.88 control (pf88) was lost.
2. **pf92 A1 (fresh) 23,003 > pf90 A1 (fresh) 21,859 (+5.2 %)**, lower TTFT (7.49 vs 9.57 s), higher GPU hit (0.930 vs 0.911).
   Different nodes (085 vs 089), one replicate each; pf92's scored A2 was lost, so 0.92 has **no scored number**.
3. **Prefill memory accounting** (`analysis/memory_accounting.md`): per rank, 278.42 GiB free before weights, weights 119.84 GiB,
   KV 125.14 / 130.71 / 136.28 GiB at 0.88 / 0.90 / 0.92, post-pool free 33.18 / 27.70 / 22.21 GiB (≈ 278.42 × (1 − frac)).
   VRAM sampler on 089 during pc64 (prefill mem 0.88, C64, 07:29-07:51, 255 samples): prefill GPUs peaked 274.98-277.44 GiB of
   287.98 → 10.5-13.0 GiB never used at 0.88. No OOM line in the packed DP0 logs (`prefill-0.log`, `decode-0.a1.log`) of pf90, pf92, pc64, and wts48 (wts48's NFS logs stop at 03:57, so its run time is not covered).
   Whether the ~20-23 GiB runtime use above the pool is fixed or scales with fraction/concurrency is **not measured** (the probe pf93/pf94 was meant to answer it).
4. **wts48 A2 (write_through_selective) 21,226 = −4.8 % vs q1 A3**, with GPU hit 0.868 vs 0.839 and CPU hit 0.039 vs 0.088,
   TTFT 13.63 vs 11.97 s. Confounded: node 136 + image build 9872bbda (vs 137), and A2 ran after a partial (~50 min) A1 instead of a full A1.
5. **pc64 A1 C64 = 10,480 (−42.5 % vs pc48b A1 C48 18,225)**, TTFT 82.8 s mean / 262.9 s p95, GPU hit 0.493 / CPU 0.341. First smoke
   request hung 300 s (cause not established); manual retry OK; audit PASS. User abandoned the prefill_complete path; A2 aborted.

## Folder map

- `README.md`, `REPRODUCE.md`, `environment.md`, `SHA256SUMS`.
- `spec/` — `directives.md` (verbatim user directives of 10-14), `mission.md`, `PLAN.md` (session mission/plan copies).
- `scripts/harness/` — verbatim harness: `bench-harness/`, `tools/` (audit v3 etc.), `mounts-v0521/`, configs (`v0521-base` → `E2-target` →
  `mc4340n{,2,3,4}`, `pcomplete{,2}`), mounts TSVs, env files of every round today (+ `env.q1-137`, `env.pc48b-274` references),
  topology TSVs, drivers (`chain_cap2`, `chain_cap2l`, `cont_bench`, `cont_benchl`, `cont_benchl2`, `stop_then_chain`, `after_a2_then`,
  `wait_then`, `after_wts64_pf94`, `chain_cx`), `run.v3`, `pipeline.v3`, `tokdiff`, `dryrun.q1-137`.
- `scripts/session/` — `tools_summ`, `tools_hits`, `tools_cmp`, `collect_env`, `vram_peak`, `correctness_tests.yihou.py`, env templates, ssh shim, session summary.
- `scripts/tools/` — `metrics_sampler.yihou.py`, `results_csv.yihou.py`, `packup_copy.yihou.sh` (the builder of this kit).
- `patches/` — attndp validation fix used by pc64 (orig / patched / diff + README).
- `results/` — `results.csv`; per-bench dirs `<round>-c<C>-A<n>/` (JSON, plots, aiperf summaries, gz logs, server-info gz);
  `rounds/<round>/` (status.tsv, chain/pipeline/handover/cont logs, audit, census); `metrics/` (5-s `/metrics` TSV gz);
  `vram/vram.089.tsv`; `dryruns/`.
- `analysis/` — `memory_accounting.md`, `ledger/` (`poll.md`, `working_process.md` copies).
- `notes/` — `incidents.md`, `gotchas.md` (what / why / how / context).
- `logs/` — `LOG_INDEX.md`, `server/<round>/*.log.gz` (all gz < 4 MB).
- `env/` — `collect_env.{085,089,136,274}.20261014.txt`.

## Sibling packups

- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-cap-c44c48.packup_20261013` — q1 reference (C48 22,306).
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-prefill-complete.packup_20261014` — pc48b/pc48e (prefill_complete at C48), attndp fix origin.
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-prefill-kv-bottleneck.packup_20261014` — x8p/x128p (prefill KV binds, synthetic).
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-hicache-sensitivity.packup_20261014` — hc50 (HiCache 50 GB, node 136).
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-runshare-highconc.packup_20261014` — runshare (paused by user 10-14).

## Gaps (packing time ~08:00 UTC)

- **In flight, not in this kit:** wts64 result (C64 on the wts48 stack), cx90 correctness results, pf94/pf93 (armed behind wts64).
  When they finish: re-run `scripts/tools/results_csv.yihou.py` and copy `$R/wts48/agentx-c64-A2`, `$N/runs/cx90/`, `$R/cx90/`, `$R/pf94/`.
- **No scored 0.92 number** (pf92 A2 lost) and **no same-session 0.88 control** (pf88 lost; pf88b never started; user later said pf88 is not needed).
- wts48/wts64 runtime server logs are node-local on 136 (not reachable); `results/metrics/wts48.metrics.A2.tsv.gz` mixes wts48 A2 and
  wts64 samples (notes/gotchas.md §2).
- VRAM trace exists only for pc64 (prefill 0.88), not for pf90/pf92.
- Directive timestamps are [inferred] from env headers / ledger entries.

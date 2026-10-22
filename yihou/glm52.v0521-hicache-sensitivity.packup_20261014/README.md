# GLM-5.2-MXFP4 v0.5.21 1P1D P4D4 — prefill HiCache capacity sensitivity at C40 (450 GB -> 50 GB per rank): **-9.1 %**

**Ran:** round `hc50`, 2026-10-13 15:59 -> 19:11 UTC on `crsuse2-m2m-136` (10.245.154.168, plain ssh, user-approved).
Reference `e2g`: 2026-10-11 05:38 -> 08:35 UTC on `crsuse2-m2m-102` (spur oai job 207495, ended 2026-10-12).
**Packed:** 2026-10-14 ~02:00 UTC. **Author:** yihou (with Claude). **Status:** DONE for the directive (one A1 + one
scored A2); single run, node and image build differ from the reference (see "What this does and does not show").

## Goal (user, 2026-10-13 ~15:55 UTC; `spec/directives.md`)

> Run hi-cache 50 GB at conc 40 on the best v0.5.21 config (e2g), everything else unchanged, to see how HiCache
> affects performance.

Success criterion (implied): a scored AgentX C40 number at HiCache 50 GB/rank that is comparable to e2g
(HiCache 450 GB/rank, A2 = 22,114 tok/s/chip). Met: A2 = 20,112, with the comparability caveats below.

## Headline (first-hand; `results/results.csv`; score = AgentX 3600 s, `request_metrics.throughput.per_gpu.total_tput_tps`)

| run | node / image ID | prefill HiCache /rank | stack | score tok/s/chip | qps | mean ISL | TTFT mean / p95 s | TPOT ms | GPU hit | CPU hit | overall hit | records ok / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **hc50 A2** | 136 / 9872bbda | **50 GB** (1,112,896 tok) | warm (flushed), scored | **20,112** | 1.119 | 142,664 | 8.41 / 30.92 | 11.87 | 0.930 | **0.000** | 0.930 | 4048 / 3 |
| hc50 A1 | 136 / 9872bbda | 50 GB | fresh | 17,733 | 1.044 | 134,925 | 9.26 / 33.71 | 14.25 | 0.939 | (absent) | 0.939 | 3787 / 2 |
| **e2g A2 (ref)** | 102 / 9d755dd6 | 450 GB (10,016,064 tok) | warm (flushed), scored | **22,114** | 1.181 | 148,711 | 5.30 / 19.21 | 12.05 | 0.926 | 0.021 | 0.947 | 4286 / 1 |
| e2g A1 (ref) | 102 / 9d755dd6 | 450 GB | fresh | 19,640 | 1.100 | 141,791 | 7.61 / 28.70 | 14.33 | 0.941 | 0.017 | 0.958 | 3993 / 4 |

Deltas hc50 vs e2g (computed from the table): A2 **-9.05 %**, A1 -9.71 %; A2 qps -5.2 %, mean ISL -4.1 %,
TTFT mean +3.1 s (+59 %), TTFT p95 +11.7 s; TPOT about equal (11.87 vs 12.05 ms). Prefill tokens actually computed
(server `prompt_by_source.computed`) 46.53 M vs 38.26 M (+21.6 %; per profiled request 11,495 vs 8,926); host-tier hit
tokens 0 vs 29.88 M; device-hit tokens 1,228.2 M vs 1,332.3 M.

### Other HiCache data points (cited; like-for-like status stated)

| point | HiCache /rank | CONC | score | like-for-like with hc50/e2g? |
|---|---|---|---|---|
| k1 A2 (102, warm) / A1 (fresh) | 360 GB | 44 | 21,342 / 20,729 | **No** — C44 (same image 9d755dd6, same recipe otherwise). Same-CONC comparison is k1 vs X1/q1 at C44 (21,773 / 21,862 at 450 GB) in the capacity packup. |
| e1c r2 (102, fresh) | ratio 1.5 = 153.81 GB | 40 | 21,048 | **No** — E1-relaxed recipe: decode mem 0.85 (not 0.95); first run on a fresh stack (r1 died at start). |
| 8-expert ceiling sweep arm A / r001-hc274 / r001-hc450chunk512 (v0.5.19, 136/137) | 154 / 274 / 450 GB | 56 | 23,892 / 26,462 / 26,071 | **No** — v0.5.19 image, synthetic 8-expert model on both legs, `--max-total-tokens 3670016`, decode mem 0.85, C56. Internally like-for-like with each other (same model/image/CONC; separate runs). |

## What this does and does not show

- Measured: at C40 with the e2g recipe, prefill HiCache 50 GB/rank gave A2 20,112 vs 22,114 at 450 GB (-9.1 %). The
  GPU (device) prefix hit rate stayed the same (0.930 vs 0.926); the host-tier (CPU) hit went 0.021 -> 0.000.
  During hc50 A2 the host pool was full from the first sample (1,112,192 of 1,112,896 tokens used on rank 0 right
  after the flush) and `load_back_tokens_total` did not increase (`analysis/hc50_A2_metrics_summary.txt`).
- Not isolated: the run differs from e2g in **node** (136 vs 102), **image build** (136's own mc4340 build 9872bbda vs
  102's 9d755dd6; the loaded Mooncake `engine.so` sha256 is recorded as identical, other layers not compared), **cache
  location** (136 node-local `/mnt/m2m_nobackup` vs NFS on 102), date, and a **foreign container `dev_xb`**
  (lmsysorg/sglang v0.5.20, described in run.v3.yihou.sh:137 as "a foreign looping job") present on 136 throughout —
  0 KFD processes at chain start, its activity during the 3 h run not measured. There is no e2g-recipe 450 GB run on 136, so
  the node/build share of the -9.1 % is **unmeasured**. Replicates of fresh C40 on 102 spanned 19,335-20,371
  (decode-transfer packup), so a single-run delta of this size is outside that fresh-band width but n = 1.
- Mechanism (why the score dropped) is **not established**. The measurement that would separate HiCache from node:
  re-run e2g (450 GB) on 136 with image 9872bbda, or run hc50 on a 9d755dd6 node (085/274/089).

## Related in-flight work (NOT in this packup)

pf88 / pf90 / pf92 = q1 config (e2g + decode max_running 96, decode mem 0.95, HiCache 450) with **prefill mem
0.88 / 0.90 / 0.92**, chain C48 A1 -> C48 A2 -> C52 A3, started 2026-10-14 01:53-01:56 UTC on 274 (job 211171) /
089 (job 213048) / 085 (job 211169; moved from 206 after job 211763 ended). Env files copied in `scripts/harness/`
for reference only; results will land in `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/pf{88,90,92}/`.

## Folder map

- `REPRODUCE.md` ordered steps; `environment.md` HW/SW (136 captured; 102 from earlier kit); `notes.md` gotchas.
- `spec/` the directive + mission.md/PLAN.md of the session. `analysis/` metrics summary, ledgers (`ledger/`),
  research notes (`research/`).
- `scripts/harness/` verbatim chain/run/pipeline/config/env/mounts files + `bench-harness/`; `scripts/tools/`
  sampler, summary tools, `extract_results.yihou.py` (regenerates `results/results.csv`).
- `patches/` the three runtime-mount source files used by every round (`mounts.yihou.v2-regbase.tsv`).
- `results/` hc50 round + A1/A2 bench dirs, reference e2g, other reference JSONs, dryruns + token diff, `/metrics`
  TSVs (gz), `results.csv`, session summary. `logs/LOG_INDEX.md` paths/sizes of uncopied server logs and aiperf dumps.

## Sibling packups

- `../glm52.v0521-cap-c44c48.packup_20261013/` (k1, q1 C44/C48/C52; capacity), `../glm52.v0521-decode-transfer-kvcap.packup_20261013/`
  (runshare; 102 fresh band), baseline kit `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`.
- 8-expert ceiling sweep: `/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-hicache-ceiling-sweep/glm52.hicache-ceiling.packup_20260924/`.

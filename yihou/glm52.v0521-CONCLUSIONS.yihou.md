# GLM-5.2-MXFP4 v0.5.21 1P1D P4D4 (same node) — overall conclusions, v0521cap / v0521cap2 (2026-10-12 .. 10-14)

**Written:** 2026-10-14 ~08:00 UTC. **Author:** yihou (with Claude). **Status:** goal reached once (single run); work in flight.
Score = AgentX 3600 s `request_metrics.throughput.per_gpu.total_tput_tps` (tok/s/chip). A1 = fresh stack (unscored),
A2 = warm stack after `/flush_cache` (scored). All runs: EAGLE MTP 5/1/6 with `SGLANG_SIMULATE_ACC_LEN=3.61` (correctness
waived by construction), image `infera-sglang:v0521-yihou-1010-mc4340`, prefill GPU0-3 / decode GPU4-7, TP4 DP4 DPA.
Path shorthands: `R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs`, `N=/shared_nfs/yihou_glm52/v0521cap2_20261013`,
`L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/debug` (ledgers `poll.md`, `working_process.md`).
Every run is n = 1 unless stated; nodes differ between most rows. [inferred] marks anything not directly measured.

## 1. Headline — best scores vs the > 23,000 tok/s/chip goal

| run (node) | model | change vs q1 | C | prefill KV tok/DP | score | source |
|---|---|---|---|---|---|---|
| **pf90 A2** (089) | **real 256 experts** | prefill mem 0.88 -> **0.90** | 48 | 2,541,376 | **23,127** (> 23k, +3.7 % vs q1) | `$R/pf90/agentx-c48-A2/agentx_conc48.json`; KV `$R/pf90/audit/prefill-server-info.json` |
| pf92 A1 (085, fresh) | real | prefill mem **0.92** | 48 | 2,649,600 | 23,003 (fresh; A2 lost to NODE_FAIL) | `$R/pf92/agentx-c48-A1/agentx_conc48.json`; `$L/poll.md` 05:11Z |
| q1 A3 (137) previous best | real | — (e2g + decode max_running 96, prefill mem 0.88) | 48 | 2,433,088 | 22,306 | `$R/q1/agentx-c48-A3/agentx_conc48.json` |
| e2g A2 (102) baseline | real | (max_running 64, C40) | 40 | — | 22,114 | `$R/e2g/agentx-c40-A2/agentx_conc40.json` |
| x8p A2 (274) synthetic | prefill 8 experts | prefill KV uncapped | 48 | 3,670,016 | 25,905 (upper bound, not faithful) | `$R/x8p/agentx-c48-A2/agentx_conc48.json` |
| x128p A2 (085) synthetic | prefill 128 experts | truncated experts | 48 | 3,318,400 | 25,297 (180 empty-content errors per run) | `$R/x128p/agentx-c48-A2/agentx_conc48.json` |

Goal status: **> 23k reached by one faithful run (pf90 A2, 23,127)**; not yet replicated, and pf90 ran on a different node
(089) than q1 (137). The synthetic rows show ~25-26k is reachable when prefill device KV grows by 36-51 %.

## 2. What binds

| evidence | numbers | source |
|---|---|---|
| Score orders with **prefill device KV** at C48 A2 | 2.43 M: q1 22,306 / x8pc 22,489 -> 2.54 M: pf90 23,127 -> 3.32 M: x128p 25,297 -> 3.67 M: x8p 25,905 | JSONs above; KV from server-info / `prefill-kv-bottleneck` packup |
| "fewer experts" alone does nothing when KV is held | x8pc (8 experts, KV capped 2,433,088) 22,489 vs q1 (256) 22,306 | `$R/x8pc/agentx-c48-A2/agentx_conc48.json` |
| Fresh (A1) C48 series also orders with prefill KV | 2.43 M x8pc 21,591 -> 2.54 M pf90 21,859 -> 2.65 M pf92 23,003 -> 3.67 M x8p 25,347 | `$R/{x8pc,pf90,pf92,x8p}/agentx-c48-A1/agentx_conc48.json` |
| **Decode** KV: smaller effect | x8d2 (decode 8 experts, decode KV 3,670,016 = +32.9 %) A2 23,591 = +5.8 % vs q1, +4.9 % vs x8pc same node 206; TTFT 10.43 vs 11.97 s | `$R/x8d2/agentx-c48-A2/agentx_conc48.json` |
| GPU vs CPU hit move together with score | prefill GPU hit / CPU (HiCache) hit: q1 0.839/0.088, pf90 A2 0.860/0.071, x128p 0.944/0.005, x8p 0.950/0.002 | JSON `server_metrics.cache` |
| Computed (non-hit) prompt tokens fall as KV grows | q1 54.9 M, pf90 A2 51.0 M, pf92 A1 46.3 M, x128p A2 42.8 M, x8p A2 41.4 M | JSON `server_metrics.tokens.prompt_by_source.computed` |
| **C48 is the sweet spot** (q1 config) | C44 21,862 -> C48 22,306 -> C52 21,418; x8d2 C48 23,591 -> C52 21,713 | `$R/q1/agentx-c{44-A2,48-A3,52-A4}/...json`, `$R/x8d2/agentx-c52-A3/agentx_conc52.json` |
| Higher CONC regresses hard | tb56n (no patch, HiCache ratio 1.5) C56 18,016 -> C64 15,108; pc64 (prefill_complete) C64 A1 10,480, TTFT 82.8 s, GPU hit 0.493 | `$R/tb56n/...json`, `$R/pc64/agentx-c64-A1/agentx_conc64.json` |
| Little's law on q1 /metrics (C44/C48/C52) | prefill queue residence 2.0/3.6/5.2 s, KV send 4.5/6.5/8.2 s, decode running ~12-13 s flat | `glm52.v0521-cap-c44c48.packup_20261013/README.md` (`results/analysis/q1_residence_c44_c48_c52.txt`) |
| Decode runs out of retractions? no | 0 decode retractions in every run checked (q1, x8*, tb56*, pc48*) | packup READMEs (`prefill-kv-bottleneck`, `runshare-highconc`, `prefill-complete`) |

Reading [inferred, suspended]: on this recipe at C48 the prefill-side device KV (radix reuse -> fewer recomputed prompt
tokens -> shorter prefill queue + TTFT) is the binding capacity; decode KV is secondary. Not proven causally: single runs,
mixed nodes, and the synthetic arms also change MoE compute/weights.

## 3. What did not help (single runs; caveats in the right column)

| change | result | vs | caveat |
|---|---|---|---|
| runshare v2 decode in-flight KV cap 0.5 (tb56 vs tb56n, 085) | C56 A2 17,611 vs 18,016 (-2.2 %); C64 A3 15,070 vs 15,108 (-0.3 %) | same node, no patch | n = 1; C40 warm replicates on 102 spanned 2.0 % (`runshare-highconc` README) |
| runshare v1 at C40 (pc40c, 102) | A2 21,271 | e2g A2 22,114 (-3.8 %) | `decode-transfer-kvcap` README |
| `prefill_complete` decode policy (pc48b vs pc48e, 274; HiCache 370 write_back, decode 0.96) | A2 20,002 vs 19,158 (+4.4 %), A1 18,225 vs 19,313 (-5.6 %) | same node, policy `early` | sign inconsistent; both arms >10 % below q1; needs a 1-line validation fix; path abandoned by user ~07:5x 10-14 (`$L/poll.md`) |
| prefill_complete at C64 (pc64, 089) | A1 10,480 | pc48b A1 18,225 | first smoke request hung 300 s (cause not established); A2 aborted |
| HiCache 50 vs 450 GB/rank at C40 (hc50, 136) | A2 20,112 vs 22,114 (-9.1 %) | e2g (102) | node + image build differ; reducing HiCache hurt, it is not a lever upward |
| HiCache 360 vs 450 at C44 (k1, 102) | A2 21,342 | X1 21,773 / q1 A2 21,862 | different nodes for q1 |
| `write_through_selective` prefill HiCache (wts48, 136) | A2 21,226 | q1 A3 22,306 (-4.8 %) | node/image build 9872bbda differ; A1 cut at ~50 min (driver terminated, cause not established) and used as warm-up |
| decode mem 0.96 (inside pc48b/pc48e) | not isolated | — | changed together with HiCache 370 + write_back |
| decode KV +33 % (x8d2) | +5.8 % | q1 | helps, but far less than prefill KV; synthetic decode model |

## 4. Prefill memory headroom (real model, per DP rank; values as printed by sglang, labelled "GB" in the log)

Measured at startup (`$R/pf{88,90,92}/server-logs/prefill-0.log`, DP0): avail before weights 278.42, weights 119.84,
avail after weights 158.58.

| prefill mem-fraction | KV tokens | KV size | free after pool | score C48 | source |
|---|---|---|---|---|---|
| 0.88 | 2,433,088 | 125.14 | 33.18 | q1 A3 22,306 (137) | pf88 log (pf88 itself lost: node 274 NODE_FAIL 16 min into A1) |
| 0.90 | 2,541,376 | 130.71 | 27.70 | A1 21,859 / **A2 23,127** (089) | pf90 log + JSONs |
| 0.92 | 2,649,600 | 136.28 | 22.21 | A1 23,003 (085) | pf92 log + JSON |
| 0.94 (pf94, armed on 136) | ~2,757,800 [inferred, linear] | ~141.9 [inferred] | ~16.7 [inferred] | not run | `$N/dryrun.pf94-136.yihou.txt` (`--mem-fraction-static 0.94`), `$R/pf94.handover.log` |

- Free-after-pool is close to, but not exactly, 278.42 x (1 - fraction): 33.18 vs 33.41, 27.70 vs 27.84, 22.21 vs 22.27
  (measured values 0.06-0.23 below the product).
- Runtime peak above the pool: `$N/runs/pc64/vram.089.tsv` (rocm-smi, 5 s, 07:29:26-07:57:05, 328 samples; pc64 = mem 0.88,
  prefill_complete, C64): prefill GPU peak 277.44 / 276.98 / 274.98 / 276.30 GiB of 287.98 -> ~22.6 GiB used above the
  0.88 pool on the worst GPU, ~10.5 GiB never used [inferred: derived as peak - (287.98 - 33.18)]. Different config/CONC from pf90-94; allocator-cached vs required
  memory not separated. The window covers the pc64 A1 tail and A2 start; GPU0 drops to ~155 from 07:50:40
  (pc64 A2 stop, `$L/poll.md` 07:5xZ), the peaks above are window maxima.
- pf92 (22.21 GiB free after pool) completed A1 without OOM; its A2 was lost to a node failure (085 NODE_FAIL 05:07 at ~24 h RunTime; time limit as cause [inferred]), not
  an engine error (`$L/poll.md` 05:11Z).
- Ceiling estimate **~0.92-0.93 [inferred]**: 22.6 GiB runtime peak / 278.42 -> f <= ~0.92 if that peak were all required;
  pf92 running fine at 22.21 GiB says part of it is not. **To be confirmed by pf94 (and pf93 if pf94 fails); pf93 has no
  dryrun on NFS yet.**
- Even 0.94 adds only ~13 % prefill KV over 0.88 [inferred]; the synthetic arms needed +36-51 % for 25-26k. Freeing more
  prefill HBM (e.g. expert-weight offload) is the remaining lever — see section 5.

## 5. Open items / in flight (as of 2026-10-14 ~07:57 UTC)

| item | state | where |
|---|---|---|
| pf90 replicate / A2 on another node | needed — 23,127 is n = 1 on 089 | — |
| cx90 correctness round (pf90 config, simulated acceptance OFF; smoke, needle 8k-256k, 5-fact 100k) on 089 | started 07:52, waiting for VRAM drain | `$R/cx90/status.tsv`, `$R/cx90.chain.log`, `$N/correctness_tests.yihou.py` |
| wts64 (C64 on the live wts48 stack, 136) | bench running since ~07:17 (warm-up 342/707 at 07:56) | `$R/wts48/agentx-c64-A2.log` |
| pf94 (prefill mem 0.94, 136) | armed behind wts64 | `$R/pf94.handover.log`, `$N/dryrun.pf94-136.yihou.txt` |
| pf93 | not prepared | — |
| pf88 rerun (0.88 same-day control) | lost twice (274 NODE_FAIL; 085 NODE_FAIL before pf88b started) | `$L/poll.md` 03:25Z, 05:11Z |
| x128p empty-content errors (180/run) | not investigated | `prefill-kv-bottleneck` README |
| pc64 first-request 300 s hang | cause not established; path abandoned | `$L/poll.md` 05:55Z |
| prefill_complete validation bug (attn-DP) | 1-line fix, upstream PR candidate, not filed | `prefill-complete` packup `patches/` |
| **OffloaderV2 feasibility** (move MoE weights CPU<->GPU between layers to free prefill HBM for KV) | user asked for a research workflow (usability in sglang v0.5.21 + dev effort); not started at time of writing | — |

## 6. Packup index (`/home/yihou/dev/git/infera.glm52.pd/yihou/` unless noted)

| packup | one line |
|---|---|
| `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/` (also copy in this dir) | Baseline kit: v0.5.21 same-node P4D4, e2g C40 22,114, warm-stack protocol, image recipe |
| `glm52.samenode-v0521-oai102.agentx-c44.packup_20261010` | First v0.5.21 bring-up on 102; C44 target config pass (e2c) |
| `glm52.v0521-cap-c44c48.packup_20261013` | q1 recipe (max_running 96, prefill 0.88): C44/C48/C52, best 22,306 C48; Little's-law prefill-side limiter; k1 HiCache 360 |
| `glm52.v0521-decode-transfer-kvcap.packup_20261013` | runshare v1/v2 decode in-flight KV cap; v1 -3.8 % at C40; v2 warm A2 never run |
| `glm52.v0521-prefill-kv-bottleneck.packup_20261014` | x8p/x8pc/x8d2/x128p: prefill KV is the lever (25,905 / 25,297 synthetic) |
| `glm52.v0521-runshare-highconc.packup_20261014` | runshare v2 vs none at C56/C64 on 085: no benefit (-2.2 % / -0.3 %) |
| `glm52.v0521-prefill-complete.packup_20261014` | `prefill_complete` policy: needs validation fix; inconsistent A/B; >10 % below q1 |
| `glm52.v0521-hicache-sensitivity.packup_20261014` | HiCache 50 vs 450 GB at C40: -9.1 % (node/build differ) |
| `glm52.v0521-prefill-mem-hicache-policy.packup_20261014` | pf90/pf92/pf88, wts48, pc64 (10-14): pf90 A2 23,127; pf92 A1 23,003 (A2 lost); wts48 A2 21,226; pc64 A1 10,480; prefill memory accounting. wts64, cx90, pf94/pf93 in flight at packing (~08:00) |
| `glm52.v0521-c48-followups.packup_20261015` | tw90 (decode --tokenizer-worker-num 4): A1 22,231 / A2 22,597 vs pf90 21,859 / 23,127, no consistent effect; cx90 correctness on pf90 22/23 (needle_128k_d10 hit max_tokens); wts64 (selective, C64) 8,681. pf94/pf93 never run |
| `glm52.v0521-offloaderv2.packup_20261015` | OffloaderV2 on prefill (+is_shuffled patch): S0 G8K1 KV +6.9 %, correctness 23/23, C48 A2 22,825 (-1.3 % vs pf90) / A1 22,380 (+2.4 %); sweep G4/G3/G2 KV +15.6/+21.7/+32.9 %, correctness G3/G2 23/23; perf rounds in flight / G3 aborted (266 released) at packing (10-15 ~03:10) |
| `glm52.v0521-offloaderv2-highconc.packup_20261015` | after 03:10 10-15: ofg4b G4 (136) A1 22,506 / A2 23,771 / C56 21,744; ofg3b G3 (v3-020) A1 22,484 / A2 23,200; ofg2 G2 (288) A1 21,157; no like-for-like C56/C64 baseline; PD warm-up hang root cause (reused cache key 026 missing 12 aiter .so + stale lock); decode tok4/detok4 rule; C56/C64 rounds in flight at packing (~11:30) |
| `glm52.v0521-offloaderv2-burst-c56c64.packup_20261019` | 10-15 13:30 -> 10-19 00:17, decode tok4/detok4, short rounds on burst nodes: C56 A2 G2 21,780 (v3-006) / G3 21,310 (v3-010); C64 A2 G4 18,535 / G2 18,392 / G3 18,333 (all v3-006, within 1.1 %); C64 ~15 % below C56; no like-for-like no-offload C56/C64 (user decision); ofg4h4 lost to /shared_nfs EDQUOT (unofficial 17,693), rerun ofg4h5; amd-spur holds end at 20:00 (10 of 11), 26 rounds lost to preemption; G2 C56 A1 v3-006 19.7k vs v3-008 21.8k unexplained |

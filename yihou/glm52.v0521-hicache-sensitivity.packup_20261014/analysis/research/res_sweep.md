I found no v0.5.21 same-node run at C56 or C64. The v0.5.21 line stops at C52 (q1 A4). The closest earlier sweep with real experts that reaches C56 is t2f/t2e (v0.5.19, cross-node P4D4). The only same-node C56 runs use the synthetic 8-expert model. Everything below was read first-hand on the cluster (packup READMEs/CSVs, `agentx_conc*.json` via jq, `server_args` in the server logs). Remote paths: `PK=/home/yihou/dev/git/infera.glm52.pd/yihou`, `R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs`, `B=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results`. qps is `records_profiled / duration_seconds`; it matches the cap packup's qps column (1.219 vs 1.220). HiCache sizes are per DP rank, from the "Allocating kv hierarchical KV host pool" log line.

## Sweep points

| # | path | image / sglang | topology, node | HiCache /DP | prefill / decode mem | max_running (arg → /DP) | MTP / SIMACC | CONC | tok/s/chip | qps | TTFT mean / p50 / p95 s | protocol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2f | `$PK/glm52-agentx-t2sweep-simacc.packup_20260920/results/t2f/agentx_conc{40,56,72,96}.json` | `v0519-yihou-0917` (base), 0.5.19.dev20260917 | **cross-node** P4D4, P 135 GPU2-5 / D 138 GPU2-5, RDMA ionic_2-5, tilelang DSA, index_share false, `--disable-custom-all-reduce` | ratio 1.5 = 3,423,488 tok (153.81 GB) | 0.85 / 0.85 | 128 (env.md:112-113) | EAGLE on / 3.61 | 40 | **19,798** | 1.108 | 7.41 / 4.16 / 26.88 | one deployment; points run in order 40→56→72→96 with no `flush_cache` in `sweep.yihou*.sh`, so each later point started on a warm stack |
| | | | | | | | | 56 | 14,675 | 0.884 | 38.73 / 17.63 / 131.92 | |
| | | | | | | | | 72 | 8,763 | 0.549 | 124.51 / 85.21 / 390.89 | |
| | | | | | | | | 96 | 3,028 | 0.222 | 418.6 / 44.66 / 1180.99 | (C128 not obtained) |
| t2e | `…/results/t2e/agentx_conc{40,56}.json` | same | same, custom all-reduce ON | same | same | same | same | 40 / 56 | 19,670 / 15,322 | 1.107 / 0.912 | 8.35/4.19/30.15 ; 37.93/17.76/135.54 | same |
| P8D8 | `$PK/glm52.p8d8.agentx-sweep.packup_20260920/results/results.csv` | `v0519-…-nextnfix-hicache` | cross-node **P8D8** TP8/DP8, P 137 / D 136 (16 GPUs) | 4,715,584 tok (cpu_total_tokens in JSON; GB not read) | not re-read | raised to 256 (README:33-34) | EAGLE 5/6/1 / 3.61 | 80 / 112 / 144 / 192 / 256 | 19,513 / 19,013 / 12,555 / 7,409 / 5,329 | 2.612 / 2.627 (C80/C112) | C80 10.08/5.69/35.03 ; C112 24.79/11.3/85.9 | one deployment, run sequentially |
| 0923 v0.5.19 triton | `$PK/glm52.samenode-triton-idxshare.c40-full.packup_20260923/results/agentx_conc40.json` | `v0519-yihou-0917-nextnfix-hicache` | **same-node** P4D4, 137 | ratio 1.5 = 153.81 GB | 0.85 / 0.85 | 64 / 64 | EAGLE / 3.61 | 40 | 21,032 | 1.150 | 6.16 / 3.59 / 20.09 | fresh |
| **e1c** | `$R/e1c/agentx-c40-r2/agentx_conc40.json` | `v0521-yihou-1010-mc4340` | same-node P4D4, 102, triton DSA | **ratio 1.5 = 3,423,488 tok, 153.81 GB** | 0.85 / **0.85** (decode 2,226,368 tok/DP) | 64 → 16/DP | 5/1/6 / 3.61 | 40 | 21,048 | 1.151 | 5.14 / 2.85 / 17.62 | fresh |
| e2g A1 / **A2** | `$R/e2g/agentx-c40{,-A2}/` | same | same-node, 102 | **450 GB** = 10,016,064 tok (`HICACHE_HOST_REGISTER_CHUNK_GB=512`) | 0.85 / 0.95 (2,762,176 tok/DP) | 64 → 16/DP | 5/1/6 / 3.61 | 40 | 19,640 / **22,113** | 1.100 / 1.181 | 7.6/3.94/28.69 ; 5.3/3.32/19.21 | A1 fresh, A2 warm (flushed 2nd run) |
| q1 A1/A2/A3/A4 | `$R/q1/agentx-c{44,44-A2,48-A3,52-A4}/` | same | same-node, **137** | 450 GB | **0.88** / 0.95 | prefill 64 → 16/DP; decode **96 → 24/DP** | 5/1/6 / 3.61 | 44 fresh / 44 / **48** / 52 | 21,433 / 21,862 / **22,305** / 21,418 | 1.194 / 1.208 / 1.219 / 1.255 | 5.9/3.53/19.05 ; 7.78/4.29/28.21 ; 11.97/7.29/38.18 ; 15.82/9.45/51.96 | one stack, run sequentially (A1 fresh, A2-A4 warm) |
| 8-expert (synthetic, non-faithful) | `$B/yihou-hicache-ceiling-sweep/glm52.hicache-ceiling.packup_20260924/results/{armA-reference-S154,r001-hc274,r001-hc450chunk512}/` | `v0519-…-nextnfix-hicache` | same-node P4D4, 136 / 137 | 154 / 274 / 450 GB | 0.85, `--max-total-tokens 3670016` (env.md:347-348) | 64 | 5/1/6 / 3.61 | 56 | 23,892 / 26,462 / 26,071 | 1.380 / 1.579 / 1.557 | 16.07/8.02/59.29 ; 10.39/6.41/32.78 ; 11.23/6.44/32.23 | each a separate run |

Other C40/C44 v0.5.21 points (fresh/warm pairs on 102 for e2c, e2i, e2j, k1 at HiCache 360, and pc40c/pc40d with the runshare patch) are in:
- `$PK/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/results/analysis/runs_summary.csv`
- `$PK/glm52.v0521-cap-c44c48.packup_20261013/results/analysis/runs_summary.csv`

pc40c and pc40d used the e2g recipe (450 GB) plus the decode patch: pc40c A1/A2 = 20,361 / 21,271, pc40d A1 = 20,680, and pc40d's warm A2 was never run. `r003-c64` (8-expert C64) died at bring-up and has no benchmark output (ceiling README:195).

## "The original CONC=40 version" and its HiCache size

The mission (`/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/mission.md:89-90`) words it as "runshare v2 … on the original C40 config HiCache capacity, pushed to C56 / C64". Two readings are possible:

- **First C40 config of the v0.5.21 same-node line = e1c.**
  - `config.yihou.E1-relaxed.sh:11-12,24,27`: `PREFILL_HICACHE_SIZE=0`, `PREFILL_HICACHE_RATIO=1.5`, `DECODE_MEM_FRACTION=0.85`, `CONC=40`.
  - Live `server_args`: `hicache_ratio 1.5`, `hicache_size 0`, giving 153.81 GB / 3,423,488 tokens per DP.
  - Also prefill mem 0.85, max_running 64, `disaggregation_decode_allocation_policy early`.
- **The earlier v0.5.19 C40s (0923 triton and t2f) used the same capacity.** Both have `hicache-ratio 1.5` = 153.81 GB/DP, with decode mem 0.85.
- **The C40 recipe the runshare work was actually measured against = e2g** (`config.yihou.E2-target.sh:10-12`, `config.yihou.v0521-base.sh:87-111`):
  - `PREFILL_HICACHE_SIZE=450` and `HICACHE_HOST_REGISTER_CHUNK_GB=512`, i.e. 450 GB/DP.
  - Prefill mem 0.85, decode mem 0.95, max_running 64.
  - This is the "baseline e2g A2 22,114" in the decode-transfer-kvcap README.

If "original C40" means the very first C40, it is 154 GB/DP. That is also exactly the capacity the old t2f C56 point ran at, which makes t2f a matched-HiCache comparison.

## Most like-for-like baseline for a v0.5.21 C56/C64 run

- **No v0.5.21 run reaches C56.** Within v0.5.21, the closest is the q1 sweep (137, same image): peak C48 22,305, then C52 21,418 (**−4.0 %**). Its config differs from the original C40 (450 GB, decode mr96, prefill 0.88).
- **The "earlier sweep in the yihou packups" that has C56 is t2f** (and t2e). Its peak is C40 19,798, then:

  | CONC | tok/s/chip | change vs C40 |
  |---|---|---|
  | 56 | 14,675 | −25.9 % |
  | 72 | 8,763 | −55.7 % |
  | 96 | 3,028 | −84.7 % |

  t2e: 19,670 → 15,322 (**−22.1 %**). t2f matches the 154 GB HiCache, decode mem 0.85 and SIMACC 3.61. It differs in four ways:
  - cross-node over RDMA, not XGMI
  - v0.5.19 image
  - tilelang DSA, index_share false
  - max_running 128

  So it is the regression reference the user named, but it is not a same-stack A/B.
- **P8D8:** C80 → C112 = −2.6 %, C144 = −35.7 % vs C80. Per-8-GPU CONC is half the listed value [inferred], so C112 there is roughly C56 per P4D4, but the topology differs.
- **8-expert same-node C56 points (24-26k)** are not comparable to the real-expert model.

=====VERIFY=====
**Verdict:** the report's numbers, paths and capacities check out. I found one wrong citation (a file name), one point I could not confirm, and some missing facts that matter if e1c is used as the "original C40" config. Everything was read on the cluster through the remote wrapper, read-only.

**Claims checked**

1. **"No v0.5.21 run at C56 or C64; the line stops at C52 (q1 A4)" — CONFIRMED.**
   - `$WS/runs/` holds only `q1/agentx-c52-A4` in that range.
   - Searching `$PK` for `agentx_conc5*`/`agentx_conc6*` finds only `t2e/conc56`, `t2f/conc56` and `glm52.v0521-cap-c44c48.packup_20261013/results/q1-A4-C52/agentx_conc52.json`.
   - The CAP round dirs `$CAP/runs/q3` and `$CAP/runs/q4` are empty.
   - `$WS/runs/q3.chain.log` ends with "q3 chain killed by leader (pc40c A2 21,271 suggests regression; patch v2 first)".
   - `$WS/runs/q4.chain.log` only shows the q1 stack being stopped after A4. There is no benchmark output.

2. **t2f / t2e numbers — CONFIRMED.**
   - JSON `total_tput_tps`: t2f C40 19798.71, C56 14675.87, C72 8763.82, C96 3028.07. t2e C40 19670.20, C56 15322.43.
   - The topology is cross-node: README.md:10, `environment.md` lines 24-57 (rails `ionic_2..5`, `MC_TE_FILTERS=ionic_2,ionic_3,ionic_4,ionic_5`), and image `v0519-yihou-0917` at `environment.md:69`.
   - tilelang DSA is at `environment.md:125`.
   - **WRONG citation:** the report cites "env.md:112-113" for max-running 128. The file is `environment.md`, and the line is 113 ("max-running 128, CUDA-graph max BS 128"). The value itself is correct.
   - `--hicache-ratio 1.5` is at README.md:97.

3. **e1c = 154 GB/DP, decode mem 0.85, 21,048 — CONFIRMED.**
   - `total_tput_tps` is 21048.65.
   - The server logs print "Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB host memory" five times.
   - `server_args` show `hicache_ratio=1.5`, `hicache_size=0` and `mem_fraction_static=0.85`.
   - `$WS/harness/config.yihou.E1-relaxed.sh` sets `PREFILL_HICACHE_SIZE:=0`, `PREFILL_HICACHE_RATIO:=1.5`, `DECODE_MEM_FRACTION:=0.85` and `CONC:=40`.

4. **The q1 sweep — CONFIRMED.** C44 21433.29, C44-A2 21862.19, C48-A3 22305.78, C52-A4 21418.17. The q1 server logs show the 450.00 GB / 10,016,064-token pool.

5. **e2g A1/A2 = 19,640 / 22,113 at 450 GB — CONFIRMED.** The values are 19640.09 and 22113.63, and the logs show the 450 GB pool line.

6. **Mission wording at `mission.md:89-90` — CONFIRMED.** It reads "runshare v2 (decode in-flight transfer/prealloc KV cap) on the original C40 config HiCache capacity, pushed to C56 / C64; expectation: less regression than the earlier sweep in the yihou packups."

7. **8-expert same-node C56 = 23,892 / 26,462 / 26,071 — CONFIRMED.**
   - The values come from `armA-136-agentx_conc56.json`, `r001-hc274/agentx_conc56.json` and `r001-hc450chunk512/agentx_conc56.json`.
   - The `r003-c64` bring-up death is CONFIRMED by the ceiling README around lines 194-198: a harness false negative, "docker inspect timed out". That directory has no agentx JSON.

8. **"No flush_cache in `sweep.yihou*.sh`" for t2f — UNVERIFIED.** A `grep -rl flush_cache` on the t2 packup `scripts/` returned nothing, which is consistent with the claim. I did not open the sweep script itself to confirm the points ran in order on one deployment.

**Missing facts that matter**

- **e1c differs from the e2g/q1 line in more than HiCache and decode memory.** `config.yihou.E1-relaxed.sh` also sets `INDEX_SHARE_FOR_MTP:=false`, `DSA_FUSED_INDEXER:=off` and `AITER_FUSED_FP8_INDEXER_WRITER:=off` (the fused gfx950 DSA indexer is off on both legs). A runshare-v2 run at "e1c HiCache capacity" on top of the e2g/runshare config therefore changes only the HiCache pool. Running the full E1 recipe would change these knobs too. The report doesn't separate the two.
- **A 154 GB control exists in the ceiling packup, but faulted.** `r002-hc154chunk64` (`CHUNK_GB=64`) faulted (FAULT.txt, README:108, README:171) and has no benchmark output. The report doesn't list it.
- **There is another 8-expert packup:** `$PK/yihou-expert8-pd.packup_20260924`, on v0.5.19 with rails `ionic0–3`. The report doesn't cite it. The mission (around line 87) names this "yihou-expert8-pd packup" as the recipe for the 8-expert experiments. I did not check it for a C56 point.
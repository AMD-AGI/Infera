# Decode-side KV-cache capacity for GLM-5.2-MXFP4 (sglang PD, MI355X): consolidated report

Written 2026-10-12. This report only reads existing material; no new experiments were run.

Evidence tags used throughout:

- **[M]** first-hand. The number was read for this report from an engine log, result JSON or TSV, and the path is given.
- **[S]** second-hand. The number is stated in a README, ledger or notes file and was not re-read from a raw log here.
- **[I]** inference. Arithmetic or attribution derived here from [M] numbers. Not tested by an experiment.

---

## 1. Scope, timeline, and where everything lives

**Question.** How far can the **decode leg's** device KV pool (`max_total_num_tokens` per
attention-DP rank) be grown for GLM-5.2-MXFP4 in same-node 1P1D P4D4 PD disaggregation on
MI355X (287.98 GiB/card), and what does each lever cost?

**Main campaign.** The dedicated task was "reproduce full-expert P4D4 `s=450`, then enlarge
decode-side KV capacity". It ran **2026-10-08 05:17 UTC → 2026-10-09 ~03:15 UTC** on
`crsuse2-m2m-274` (job 199057), then `crsuse2-m2m-102` (job 200276). The image was
`infera-sglang:v0519-yihou-0917-nextnfix-hicache` (sglang `0.5.19.dev20260917+ga9fb1c3238`).

| what | path |
|---|---|
| workspace (spec, plan, ledger, research) | `/home/yihou/dev/git/infera.glm52.view/glm52.decode-kv-capacity_20261008/` (`mission.md`, `PLAN.md`, `debug/working_process.md`, `research/R1-R5`, `results/levers.yihou.tsv`) |
| runtime (raw engine logs per round) | `/shared_nfs/yihou_glm52/kvcap_20261008/rounds/` (`s2-*` lever rounds, `r012-1p1d-xgmi`, `r013-1p1d-validate`, …) |
| **pack-up of this campaign** | `/home/yihou/dev/git/infera.glm52.view/glm52.oai-samenode-1p1d-p4d4-xgmi.packup_20261009/`. The workspace has no pack-up under its own name; this is it (`mission.md` "STAGE 2 STATUS" names it). The pack-up's `results/stage2/levers.yihou.tsv` is byte-identical to the workspace copy (`cmp`). |

**Related work (before and after):**

| date | work | relevance | path |
|---|---|---|---|
| 2026-09-23 | synthetic 8-routed-expert model, to free weight VRAM for KV experiments | precursor: the largest decode pool ever measured, but on a non-faithful model | `/home/yihou/dev/git/infera.glm52.pd/yihou/yihou-expert8-pd.packup_20260924/` |
| 2026-09-24 | HiCache ceiling sweep on the 8-expert model, `--max-total-tokens 3670016` | the setup the main campaign set out to reproduce with real experts | `/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-hicache-ceiling-sweep/glm52.hicache-ceiling.packup_20260924/` |
| 2026-09-24 | SIKL prefill profiling (offline, no PD) | an incidental first-hand data point for the L6 index-K elision lever | `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.sikl-prefill-hotspot.packup_20260924/` |
| 2026-10-09 → 2026-10-11 | sglang **v0.5.21** same-node 1P1D, AgentX C40/C44, decode `mem_fraction_static 0.95` | the higher L1 point validated under sustained load | `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-v0521-oai102.agentx-c44.packup_20261010/`, `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`, runtime `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/` |

---

## 2. Experiments and rounds

### 2.1 Fixed geometry (needed to read the table)

All numbers below are **per attention-DP rank (one GPU)**. Each rank logs its own
`KV Cache is allocated` line, and the 114.51 GB pool sits on one 288 GB card next to
119.24 GB of weights. (The pack-up's `analysis/ANALYSIS.md` §4a says "across the 4 DP
ranks"; the per-rank arithmetic contradicts that wording.)

- **Main-pool cell = 55,224 B/token** = MLA 576×78 (44,928) + DSA index-K 132×78 (10,296).
  - [M] 114.51 GB / 2,226,368 → 55,228 B (`rounds/r012-1p1d-xgmi/server-logs/decode-0.a2.log`).
  - [M] 117.79 GB / 2,290,176 on prefill (`rounds/r000-probe/prefill-0.strings.txt`, quoted in `results/r000-probe.yihou.md`).
- **MTP draft pool = 708 B/token**, a separate pool on decode only. [M] 1.47 GB / 2,226,368.
- **Decode total = 55,932 B/token** across both pools.
- **Weights, decode leg (real 256 experts, EP1, shared-experts fusion on):** target 119.24 GB + NextN draft 6.92 GB. [M] `Load weight end` lines in every `s2-*` decode log.
- The cell is independent of expert count and `ep_size`. index-K is already fp8 with a per-128 scale.

### 2.2 Round table

Columns:

- **cap** = `max_total_num_tokens` per rank.
- **KV GB** = main pool + draft pool, as logged by sglang ("GB" in sglang's log is computed with 2^30).
- **avail** = `available_gpu_mem` after pool init.

Unless noted, every [M] number comes from `strings <log> | grep` on
`/shared_nfs/yihou_glm52/kvcap_20261008/rounds/<round>/server-logs/decode-0.a1.log`
(r012 uses `decode-0.a2.log`). The same numbers appear in
`…packup_20261009/results/stage2/levers.yihou.tsv` and `results/stage2/rounds/<round>/round.log`.

| # | date (UTC) | node | image / sglang | config delta vs baseline | cap / rank | Δ | KV GB (main + draft) | avail GB | outcome | where |
|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 2026-09-23 | 138 P / 136 D | expert8 image, v0519 era | **synthetic 8 routed experts** (weights 31.44 GB/rank), EP4, MTP **off**, mem 0.85 | **4,097,408** [S] | +84 % [I] | 210.74 [S] | not recorded | fixed-length C32 PASS 256/256 [S]. **Non-faithful model.** No long-context or full-KV soak. | `yihou-expert8-pd.packup_20260924/README.md` |
| P1 | 2026-09-24 | 136/137 | same, + HiCache | 8-expert, MTP on, `--max-total-tokens 3670016` (a cap below the derived size) | 3,670,016 [S] | — | 188.76 + 2.42 [S] | 57.42 [S] | AgentX C56 PASS, 26,072-26,462 tok/s/chip [S] | `glm52.hicache-ceiling.packup_20260924/README.md`, `environment.md:372` |
| B0 | 10-08 12:50 | 102 | v0519 `…nextnfix-hicache` | **baseline**: real 256 experts, EP1, mem 0.85, MTP 5/1/6, graph bs 64 | **2,226,368** [M] | — | 114.51 + 1.47 | 39.67 | 1P1D serves, e2e completion OK | `rounds/r012-1p1d-xgmi/`; pack-up `results/stage1-r012/` |
| C0 | 10-08/09 | 102, GPUs 0-3 | same | decode-solo control at 0.85 | 2,226,368 [M] | 0 | 114.51 + 1.47 | 39.68 | method valid (solo = 1P1D) | `rounds/s2-control-0.85-solo-gpu0123/` |
| L1 | 〃 | 〃 | 〃 | mem 0.88 | 2,390,464 [M] | +7.4 % | 122.95 + 1.58 | 31.19 | capacity only | `rounds/s2-L1-0.88/` |
| L1 | 〃 | 〃 | 〃 | mem 0.90 | 2,499,840 [M] | +12.3 % | 128.57 + 1.65 | 25.45 | capacity only | `rounds/s2-L1-0.90/` |
| L1 | 〃 | 〃 | 〃 | mem 0.92 | 2,609,280 [M] | +17.2 % | 134.20 + 1.72 | 19.75 | capacity only | `rounds/s2-L1-0.92/` |
| L1 | 〃 | 〃 | 〃 | mem 0.94 | 2,718,656 [M] | +22.1 % | 139.83 + 1.79 | 14.03 | capacity only | `rounds/s2-L1-0.94/` |
| L1 | 〃 | 〃 | 〃 | mem 0.95 | 2,773,376 [M] | +24.6 % | 142.64 + 1.83 | 11.25 | capacity only on v0519; see V2 for v0.5.21 | `rounds/s2-L1-0.95/` |
| L1 | 〃 | 〃 | 〃 | mem 0.96 | 2,828,032 [M] | +27.0 % | 145.45 + 1.86 | 8.31 | capacity only | `rounds/s2-L1-0.96/` |
| L1 | 〃 | 〃 | 〃 | mem 0.97 | 2,882,752 [M] | +29.5 % | 148.27 + 1.90 | 5.55 | capacity only | `rounds/s2-L1-0.97/` |
| L2 | 〃 | 〃 | 〃 | CUDA-graph max bs 64→16, mem 0.85 | 2,226,368 [M] | **0** | 114.51 + 1.47 | 39.67 | not a capacity lever | `rounds/s2-L2-bs16-at0.85/` |
| L1+L2 | 〃 | 〃 | 〃 | mem 0.96 + bs 16 | 2,828,032 [M] | +27.0 % (= L1-0.96) | 145.45 + 1.86 | 8.32 | L2 adds nothing | `rounds/s2-L1L2-0.96-bs16/` |
| L3 | 〃 | 〃 | 〃 | **MTP off**, mem 0.85 | 2,389,504 [M] | +7.3 % | 122.90 + — | 40.74 | capacity only; loses speculative decoding | `rounds/s2-L3-mtpoff-at0.85/` |
| L1+L3 | 〃 | 〃 | 〃 | MTP off, mem 0.94 | 2,888,128 [M] | +29.7 % | 148.54 + — | 15.14 | capacity only | `rounds/s2-L1L3-0.94-mtpoff/` |
| L1+L3 | 〃 | 〃 | 〃 | MTP off, mem 0.97 | **3,054,336** [M] | **+37.2 %** | 157.09 + — | 6.61 | capacity only (largest real-model point) | `rounds/s2-L1L3-0.97-mtpoff/` |
| V1 | 10-09 02:42-02:54 | 102, P 0-3 / D 4-7 | v0519 | **1P1D**, decode mem **0.92**, MTP on | 2,609,280 [M] | +17.2 % | 134.20 + 1.72 | 19.73 | **VALIDATED**: 64-way burst 64/64, 53.0 s, 617.7 agg tok/s, 0 OOM/fatal/traceback; intrinsic coherence 6/6 [M]. Token usage peaked at only 0.01 [S]. | `rounds/r013-1p1d-validate/`; pack-up `results/r013-validate-0.92/verdict.txt` |
| V2a | 10-10 | 102 | `infera-sglang:v0521-…` (v0.5.21) | 1P1D, decode mem **0.95**, fused DSA indexer **off** (e2b4) | 2,773,376 [M] | +24.6 % | 142.64 + 1.83 | 10.83 | **AgentX full C44 PASS**: 4138 ok / 1 err, 162,774 tok/s [S] | `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/e2b4/server-logs/decode-0.a1.log`; RESULTS: `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/analysis/RESULTS.md` |
| V2b | 10-10/11 | 102 | v0.5.21 | same, fused indexer **on** (e2c target; e2g) | 2,762,176 [M] | +24.1 % | 142.07 + 1.82 | 10.89 | e2c **C44 PASS** 4179 / 2, 164,194 tok/s [S]; e2g **C40 PASS** 176,909 tok/s = 22,114 tok/s/chip [S] | `…/runs/e2c/`, `…/runs/e2g/` decode logs; `RESULTS.md` |
| V2c | 10-10 | 102 | v0.5.21 | decode mem **0.85** (e1c, relaxed) | 2,226,368 [M] | 0 | 114.51 + 1.47 | 39.26 | AgentX C40 PASS 4174 / 2, 168,389 tok/s [S] | `…/runs/e1c/server-logs/decode-0.a1.log`; `RESULTS.md` |
| L6 | — | — | — | DSA index-K elision for the 57 `shared` indexer layers | ≈ 2,572,500 [I] | ≈ +15.5 % [I] | — | — | **unreachable under PD** (code gate + transfer invariant) | `research/R5-L6-dsa-index-elide.yihou.md` |
| L6-obs | 2026-09-24 | 256 | v0519 era | *not PD*: offline `profile_prefill.py`, TP4/DP4, mem 0.85, no MTP | 2,642,304 [M] | — | 117.38 | not recorded | cell = 117.38 GB / 2,642,304 = **47,700 B** = 44,928 + 21×132 [I]. This is exactly the elided cell. | `glm52.sikl-prefill-hotspot.packup_20260924/logs/round2-A.log:128-143` |
| L4 | — | — | — | decode D4 → D8 (TP8/EP8) | not recorded | ≈ +47 %/GPU projected [S] | — | — | not run: same-node 1P1D has no free GPUs | `research/R2-decode-kv-levers.yihou.md` lever 2 |
| L5 | — | — | — | `--cpu-offload-gb` | — | +1 token-GB per offloaded weight-GB [S] | — | — | **not run** | R2 lever 4 |

Notes on the table:

- 〃 = same as the row above.
- All `s2-*` rows are **decode-solo** (no prefill peer). `research/R4-decode-solo-validity.yihou.md` shows pools are sized before any peer object exists.
  - C0 reproduced B0 exactly, and V1 reproduced L1-0.92 exactly (2,609,280; avail 19.73 vs 19.75). So solo capacity transfers to 1P1D [M].
- The `s2-L3` row's `round.log` prints an empty capacity: the TSV row came from the asynchronously written server log (pack-up `notes.md` N7). The server log carries 2,389,504 [M].
- P0 and P1 use a synthetic 8-expert model. Their capacities are **not** achievable with the real checkpoint (+87.40 GB/rank of weights at EP4 [M], `results/r000-probe.yihou.md` §1).

---

## 3. Key conclusions

1. **`--mem-fraction-static` (L1) is the only working lever that keeps MTP, and it is exactly linear** [M].
   - From 0.85 to 0.97: +656,384 tokens, about **54,700 tokens (≈2.86 GB) per 0.01** [I].
   - The pool is a formula, so every point "comes up". Bring-up success says nothing about runtime safety. The cost is the activation/graph reserve: `available_gpu_mem` falls from 39.67 GB to 5.55 GB.
   - The post-capture resize path cannot reclaim the slack automatically: `post_capture_kv_sizing_planned` returns False for any MLA model (`overrides.py:1814`) [S, code read in the ledger].
2. **Runtime-validated ceiling so far: 0.95 with MTP on.**
   - v0519: 0.92 (+17.2 %) validated in 1P1D by a 64-way full-batch burst [M]. That burst never filled the KV (token usage 0.01) [S].
   - v0.5.21: 0.95 (+24.1 % / +24.6 %) passed **full 3600 s AgentX at C44 and C40** several times [S, `RESULTS.md`]. Decode token usage reached 0.27→0.45 (e2g A1, ledger R046) and 0.69 on one rank (c44c, ledger R004) [S].
   - **0.96-0.97 and every MTP-off point remain capacity-only.** No 1P1D or load test was run for them.
3. **MTP off (L3) buys +7.3 % (+163,136 tokens at 0.85)** [M] and composes with L1. 0.97 + MTP off gives 3,054,336 (+37.2 %), the largest real-model figure.
   - The gain is almost all **VRAM** (6.92 GB of draft weights plus draft graphs), not cell size. The main-pool cell stays 55,224 B: 122.90 GB / 2,389,504 = 55,226 [I].
   - ANALYSIS §4c's "55,932 → ~51,432 B" is an arithmetic slip. Removing the draft pool takes the cell to 55,224, not 51,432.
   - Cost: no speculative decoding. The throughput trade was **not measured**.
4. **CUDA-graph max bs (L2) moves capacity by exactly 0** [M, twice]. The KV pool is sized before graph capture. At most it changes runtime headroom, which pre-capture `available_gpu_mem` does not show.
5. **DSA index-K elision (L6) is the biggest remaining structural win, and it is blocked only by PD.**
   - The checkpoint has 21 `full` / 57 `shared` indexer layers. Eliding the shared layers' index-K takes the main cell 55,224 → 47,700 B (+15.8 % on the main pool; ≈ +15.5 % on the decode total).
   - The SIKL offline run (non-PD, so elision active) shows exactly the **47,700 B cell** [I from M]. The mechanism works in the shipped sglang.
   - Under PD, `_should_elide_dsa_index_k` requires `disaggregation_mode == "null"` (`kv_cache_configurator.py:105-113`). PD transfer also matches sender/receiver buffer lists by length (`disaggregation/utils.py:1410`, `mooncake/conn.py:948,1173-1176`). A decode-only patch would therefore corrupt transfers. It would need a symmetric prefill+decode change [S, code read in R5].
6. **Closed doors** (code-read in `research/R2`, not experiments) [S]:
   - **HiCache on decode buys 0 device tokens.** PD decode forces `disable_radix_cache` → `ChunkCache` → HiCache is a no-op. It is also a host tier, not device capacity.
   - `--enable-dsa-cache-layer-split` is rejected on decode workers.
   - **FP4 KV is unreachable**: DSA always routes to `_build_dsa_kv_pool`, so fp8_e4m3 is the floor.
   - `page_size` 64 is forced by the HIP DSA kernels.
   - `--max-total-tokens` can only cap, never raise.
   - Host-pool retraction backup is excluded on HIP.
   - Reserved decode tokens / extra slots are worth ≈ 0.4 %.
7. **Fused DSA indexer costs about 0.4 % of capacity** [M numbers, I attribution].
   - On v0.5.21 at decode 0.95: fused indexer off (e2b4) gives 2,773,376, identical to v0519's L1-0.95. Fused on (e2c/e2g, `enable_dsa_fused_indexer: None`) gives 2,762,176 (−11,200).
   - Prefill shows the same pattern: 2,282,304 off vs 2,270,656 on.
   - The v0.5.21 ledger (R001) attributes the −0.4 % to v0.5.21 vs v0519. The single-variable e2b4→e2c delta points to the fused indexer instead.
8. **The expert8 detour does not transfer.**
   - With 8 experts, decode held 4.10 M tokens [S]. With the real 256 experts at EP4, weights grow by 87.40 GB/rank [M].
   - The hard-coded `--max-total-tokens 3670016` would have needed 307.6 GB on a 287.98 GB card [M, `r000-probe` §3].
   - Expert-count-dependent knobs must be re-derived from `Load weight end` / `Memory pool end`.
9. **Infrastructure root causes found on the way** (these gated every capacity measurement):
   - Mooncake RDMA registration fails on this node class: no peer-mem, ionic without ODP. With fan-out registration the failure also kills the HIP buffer.
   - The fix is `MC_TE_FILTERS=<non-existent>` → 0 HCAs → KV over HIP IPC/XGMI [M, ledger rounds 016-022].
   - VRAM release after container exit can lag minutes to hours, whatever the stop signal. Gate every launch on idle GPUs, or capacity readings shift [M/S, ledger round 025].

---

## 4. Open questions and recommendations

1. **Is KV capacity actually binding? Open.** The S2.0 premise probe (`research/R3-premise-probe.yihou.md`) was designed but **never run on a live workload**.
   - v0.5.21 AgentX evidence points both ways [S]:
     - Admission never bound in R041 (prealloc>0 0 %, retracted 0).
     - Per-DP first-wave load (2.60-2.71 Mtok) ≈ capacity (2.76 M) (poll 13:20Z, marked "undecidable").
     - One rank reached token usage 0.69.
   - Recommendation: capture `/metrics` `token_usage`, prealloc/queue gauges and retractions across a full AgentX run at 0.85 and 0.95. Before spending more on capacity, check whether a larger pool changes the score.
2. **No controlled throughput A/B of capacity exists.** The e1c (0.85) vs e2b4/e2c (0.95) passes differ in several variables (HiCache 154 vs 450 GB, fused indexer, CONC). Recommendation: a single-variable 0.85 vs 0.95 AgentX pair on one stack, using the flush-then-second-run protocol from `RESULTS.md`.
3. **Push validation above 0.95.** Run 0.96 / 0.97 under full AgentX. Then run L1+L3 under load if losing MTP is acceptable; L3's throughput cost must be measured.
4. **L6 upstream.** Propose a symmetric elision: skip allocating and shipping index-K for `shared` layers on both legs. The sikl run already demonstrates the 47,700 B cell, so this is ≈ +15.5 % decode capacity at zero VRAM cost.
5. **Not run:** L4 (D8 decode; needs a cross-node or larger allocation) and L5 (`--cpu-offload-gb`; expected to hurt decode latency).
6. **Deterministic baseline.** Temperature-0 output diverges run-to-run on an unchanged stack [M, ANALYSIS §5]. A byte-level 对拍 correctness gate for high-L1 points is impossible until a deterministic configuration is found.

---

## 5. References

Main campaign (v0519):

- Workspace: `/home/yihou/dev/git/infera.glm52.view/glm52.decode-kv-capacity_20261008/`.
  - `mission.md`, `PLAN.md`, `debug/working_process.md` (rounds 001-026).
  - `research/R2-decode-kv-levers.yihou.md` (lever inventory, sizing chain), `R3-premise-probe`, `R4-decode-solo-validity`, `R5-L6-dsa-index-elide`.
  - `results/levers.yihou.tsv`, `results/r000-probe.yihou.md`, `results/r001-decode-solo.yihou.md`.
- Runtime: `/shared_nfs/yihou_glm52/kvcap_20261008/rounds/{s2-*, r000-probe, r001-decode-solo, r012-1p1d-xgmi, r013-1p1d-validate, r014-1p1d-control085}/`.
- Pack-up: `/home/yihou/dev/git/infera.glm52.view/glm52.oai-samenode-1p1d-p4d4-xgmi.packup_20261009/`.
  - `README.md`; `analysis/ANALYSIS.md` §4-5, §7.
  - `results/stage2/levers.yihou.tsv`, `results/stage2/rounds/*/round.log`.
  - `results/r013-validate-0.92/`, `environment.md`, `notes.md`.

v0.5.21 follow-up:

- `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/analysis/{RESULTS.md, working_process.md (R001, R004, R041, R046), poll.md}`
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-v0521-oai102.agentx-c44.packup_20261010/results/{e2b4,e2c}/census.txt`
- Raw logs: `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/{e1c,e2b4,e2c,e2g}/server-logs/decode-0.a1.log` and `/shared_nfs/yihou_glm52/agentx44_20261009/runs/c44b/server-logs/decode-0.a1.log`.
- Live ledger: `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009/debug/working_process.md`.

Precursors and incidental evidence:

- `/home/yihou/dev/git/infera.glm52.pd/yihou/yihou-expert8-pd.packup_20260924/README.md` (synthetic 8-expert decode pool).
- `/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-hicache-ceiling-sweep/glm52.hicache-ceiling.packup_20260924/` (`README.md`, `environment.md:365-380`).
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.sikl-prefill-hotspot.packup_20260924/logs/round2-A.log` (elided 47,700 B cell, non-PD).
- `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.p8d8.agentx-sweep.packup_20260920/environment.md:42`: P8 **prefill** 3,143,680 tokens/rank at 0.85 [M, `logs/prefill-0.excerpt.txt.gz`]. This is the closest data to L4, but no **decode** D8 capacity was recorded.
- Searched, no further decode-capacity experiments found: the other `*packup*` directories under `/home/yihou/dev/git/infera.glm52.view/` and `/home/yihou/dev/git/infera.glm52.pd/yihou/`. They only use mem 0.85 at default capacity.

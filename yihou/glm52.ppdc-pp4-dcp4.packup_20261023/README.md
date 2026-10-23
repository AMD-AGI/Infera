# GLM-5.2 PD cross-node 135 -> 136: PP4 prefill + DCP4 decode + llying prefill options (2026-10-23)

**Result (AgentX, 3600 s, one bench each, sim acceptance 3.61 - no correctness claim; per-GPU over 8 GPUs):**

| round | change | conc | score tok/s/chip | qps | ISL mean | TTFT p50/p90 s | TPOT ms | GPU/CPU hit | KV xfer p50/p90 s |
|---|---|---|---|---|---|---|---|---|---|
| ppdc1 | dcpx11 + PP4 prefill + SPF + write_back + no OffloaderV2 + decode empty_cache (mem 0.88) | 56 | 27,426 | 1.671 | 130.3k | 1.68 / 3.59 | 19.3 | 0.877 / 0.091 | 0.55 / 2.23 |
| ppdc2 | ppdc1 @C64 | 64 | 29,817 | 1.898 | 124.8k | 2.17 / 5.40 | 21.2 | 0.828 / 0.138 | 0.79 / 2.42 |
| **ppdc2** | ppdc1 @C72 (same stack, 2nd bench) | **72** | **31,128** (best) | 2.075 | 119.1k | 3.85 / 15.14 | 22.5 | 0.715 / 0.233 | 1.40 / 3.68 |
| ppdc4 | ppdc1 @C80 | 80 | 31,076 | 2.208 | 111.7k | 8.48 / 21.28 | 23.5 | 0.711 / 0.249 | 1.74 / 4.25 |
| ppdc3 | ppdc1 with write_through (ablation) | 56 | 27,457 | 1.675 | 130.2k | 1.60 / 3.17 | 19.4 | 0.877 / 0.090 | 0.51 / 2.08 |

References (same nodes): dpax1 (TP4/DPA prefill + DPA4 decode) 26,321 @C56; dcpx11 (TP4/DPA prefill + DCP4) 25,197 @C56;
g3x1 (v0521) 25,524 @C56. Running at pack-up time: ppda1 = ppdc4 with the DPA4 decode of dpax1 (llying's 4+4 C80 PP4 +
DPA4 = 37,310).

Reading the sweep: the prefill stopped being the limit with PP4 (prefill queue p90 0.06-0.26 s at C56-C64). The score
plateaus at C72-C80 while the mean ISL keeps falling (-4..6 % per step), TTFT p90 jumps (5.4 -> 15-21 s), the prefill
GPU prefix hit drops 0.83 -> 0.71 and TPOT rises 19.3 -> 23.5 ms. Which of these is the binding limit is not measured.
write_back vs write_through: no difference at C56 (27,426 vs 27,457), where the prefill has slack.

Open: correctness of PP prefill + our no-wait DCP pack under sim acceptance is not checked here; llying ran the same pack +
shard fix on a PP4 prefill and passed GSM8K 0.9712/0.9697, accept 4.53 (`llying/2p1d-v0522-20261020`, commit 744a1c33).

## Layout
- `opt.readme.md` - what each change is, how to turn it on, measured effect, trade-offs taken in round 1.
- `REPRODUCE.md`, `notes.md`; `patches/` (deepseek_v2 PP fix, decode empty_cache + memlog, conn/dcp_pack files, install
  scripts, env generators); `results/` (summary.tsv, xfer_stats.txt, rounds/, dryruns/); `scripts/` (envs, harness, tools).
Previous: `../glm52.dcpx-pack-fix-best.packup_20261023/` (DCP transfer fix, dpax1, dcpx11).

# DSv4.1-Flash on MI355X with vLLM — PD-disaggregation study, delta pack-up 2026-11-24

**Time covered:** 2026-11-20 05:30Z – 2026-11-24 ~04:30Z. **Owner:** yihou. **Branch:** `dev/yihou/dsv41` (infera).
**Previous pack-up (everything unchanged lives there):** `../dsv41_vllm_pd_20261119.packup_20261120/` (11-19 08:00Z – 11-20 05:30Z).
**Status at 04:30Z 11-24:** no node held; Phase 2 (reproduce + PD feasibility) still open; PD (E7) measured once (c256), below
colocated at equal GPUs; colocated R1/E4 curves done on crusoe; go/no-go (E8) not decided.

## Results in this window (official AgentX contract, 3600 s, seed 42, synthetic DSpark AL 3.51)
Image for every run: `vllm/vllm-openai-rocm:nightly-60bd6dd2…@sha256:67a3a0fb…` (see `environment.md`). tok/s/GPU = total tok/s /
all GPUs (P+D for PD). All runs shared their node with 1-3 other servers. Source: `results/ledger_openai.csv`, `results/ledger_crusoe.csv`.

### openai (n06-25, job 36733)
| run | config | conc | GPUs | tok/s/GPU | reference | note |
|---|---|---:|---:|---:|---|---|
| e3-r2-c512-11210505 | R2 = DP2 + UMBP host tier | 512 | 2 | **425,290** | CI 427,451 (-0.5%) | TTFT p90 16.4 s, 44 preemptions, KV max 1.00, p90 intvty 11.0 |
| e2-r2rep-11210325 | R2 replicate | 384 | 2 | 414,042 | CI 427,451 (-3.1%); 11-19 407,049 | p90 intvty 15.0, 0 preemptions |
| e7-pd-c256-11210342 | **PD 1P1D** Mooncake RDMA, P DP2 + D DP2, vllm-router 0.1.14, MC_STOPPED_FIX | 256 | 4 | **254,318** | colocated R1 c256 346,601 (11-19 openai, raw) = -26.6% | first official vLLM PD of DSv4.1 on AMD; TTFT p50/p90 2.45/8.64 s, ITL p50 18.3 ms, 5 errors |
| e4c-rep-11210509 | R1 + fp8_ds_mla, sparse-MLA kept | 128 | 2 | 248,079 | 11-19 243,585; mean -5.9% vs R1 c128 mean 261,352 | |
| e4b-rep-11210343 | R1 + fp8_ds_mla + sparse-MLA off (path any non-UMBP connector forces) | 128 | 2 | 227,035 | 11-19 221,199; mean -14.2% vs R1 | |
| e7-pd-c512-11210519 | PD as e7 | 512 | 4 | none | — | stopped in warmup (rc=137): 98-99x Mooncake `hipIpcOpenMemHandle` 201 -> transfer failed |
| e7h-pd-c512 / e7h-pd-c384 (11210548 / 11210653) | e7 + MC_DISABLE_HIP=1 | 512 / 384 | 4 | none | — | 0 Mooncake failures until node preempted ~08:08-08:13Z |
| e7h-pd-c384b / e7p-pd-c256 (11211204) | e7h retry / e7h + P-only mem 0.93, long-prefill 8192, max-num-seqs 64 | 384 / 256 | 4 | none | — | n02-25 job 36884 cancelled by pool daemon 12:04:47Z while loading (docker_free < 200 G) |

### crusoe colocated curves (v3-200 job 302960; v3-014 jobs 303579/304125/305057), DP2 on 2 GPUs
Mean of contract_ok=True ledger rows (n, min-max); % vs R1 mean at the same conc. Excluded as void: e4c-cru8-c320-11221401
(139,858, half-rate anomaly) and e3-r1cru-c320-11211705 (340,558, contaminated by the R2 stall).

| conc | R1 (no connector) | e4c (connector path, sparse kept) | e4b (connector path, sparse off) |
|---:|---|---|---|
| 128 | 267,106 (2, 266,568-267,644) | 245,266 (2) -8.2% | 226,397 (2) -15.2% |
| 192 | 325,754 (4, 306,857-335,588) | 304,415 (3) -6.6% | 271,104 (4) -16.8% |
| 224 | 357,339 (4, 354,795-359,383) | 310,530 (3) -13.1% | 263,915 (4) -26.1% |
| 256 | **358,296 (8, 348,490-372,545)** | 300,944 (7, 273,894-314,776) -16.0% | 272,598 (6) -23.9% |
| 288 | 349,874 (5, 336,189-356,394) | 290,706 (6) -16.9% | 261,371 (5) -25.3% |
| 320 | 337,281 (4, 321,420-349,596) | 271,977 (4) -19.4% | 236,029 (4) -30.0% |
| 384 | 273,366 (4, 248,828-290,889) | 163,455 (2) -40.2% | 157,397 (1) -42.4% |

Other crusoe scored points: e1-r1cru-11201015 R1 c128 267,644 (v3-129 job 300599; CI 271,329 -1.4%); e7r-11201015 colocated
pd_smoke reference c32 110,175 (not a perf point). Crusoe ledger: 114 rows, 93 scored, 87 contract_ok=True, 6 False, 21 no score.
R2/UMBP and PD have no score on crusoe (stalls / transport failures, see `notes.md`).

### Second-hand (not reproduced by us)
llying (liyingli), `origin/dev/pd_opt/vllm_dsv41_agentx` @ e05238e73 (2026-11-24 03:01Z),
`llying/vllm-1p1d-dsv41-crusoe-20261123/README.zh-CN.md`: 1P1D (P2D2, 4 GPUs) crusoe, P MoE a4w4, c640 **486,737 tok/s/GPU**
(his crusoe baseline c640 463,531; his AUS c640 471,698). Different stack (image nightly-9c1d45c2, Infera router). This is above
the public colocated bar 427,451 (+13.9%) and above our best (425,290).

## What changed vs 11-20
- **First PD numbers.** 1P1D Mooncake c256 = 254,318 tok/s/GPU (openai). Stage analysis from Prometheus counters
  (`research/e7_c256_stage_analysis.md`): per request P queue 8.04 s, prefill 0.92 s, D queue 0.82 s, decode 15.78 s; P prefix
  hit 94.5%; D KV avg 32%. Reading (inference): P queue is the largest stage.
- **Same-node Mooncake on ROCm fails intermittently** (`hipIpcOpenMemHandle` error 201) at c512 openai and on crusoe TCP;
  MC_DISABLE_HIP=1 with RDMA had 0 failures until preemption; TCP + MC_DISABLE_HIP moved no KV
  (`research/mooncake_samenode_transport.md`). On crusoe any PD role with the RDMA container setup dies with Triton "Code 209".
- **R2 at c512 reaches 425,290** (-0.5% vs CI); R2 replicate at c384 414,042.
- **Colocated curves on crusoe**: R1 peaks at c256-c224 (~358k); the connector-forced kernel-path cost grows with conc (e4b -15% at
  c128 -> -24% at c256 -> -42% at c384). Why it grows: not measured (`research/crusoe_colocated_curves_20261121.md`).
- **crusoe node stalls**: `shm_broadcast` 60 s / `sample_tokens` RPC TimeoutError, hitting all servers of one PCIe domain at once;
  16 episodes logged (2 on v3-200, 14 on v3-014 across 3 jobs); R2/UMBP and the KV connector are not required for it; cause unknown
  (`research/crusoe_r2_umbp_stall.md`).
- **Kit**: `MC_STOPPED_FIX=1` (KV also sent when prefill ends FINISHED_STOPPED, from llying-001/vllm#3), split RDMA docker flags,
  crusoe launch helpers with a 480 s gap between server starts (`scripts/CHANGES_since_20261120.md`).
- **Configs**: 12 new PD/smoke configs (e7h, e7p, e7th/e7tp/e7u TCP variants, e7s* Code-209 split); e7/e7s add MC_STOPPED_FIX=1.

## Navigate
- `REPRODUCE.md` — commands for this window's points (crusoe helpers, openai PD); the rest is in the previous pack-up.
- `environment.md` — nodes/jobs used in this window, image/versions, secrets (names only).
- `notes.md` — gotchas and wrong turns of this window (stalls, idle release, launch gap, login node, watchdog, kill-by-PID).
- `scripts/kit/` — ONLY changed/new kit files (+ `scripts/CHANGES_since_20261120.md`); unchanged kit files (run_point.sh, stop.sh,
  cru_env.sh, pd_smoke.*, …) are in `../dsv41_vllm_pd_20261119.packup_20261120/scripts/kit/`.
- `configs/` — new/changed configs only. `results/ledger_{openai,crusoe}.csv`, `results/runs_openai/<tag>/` (33 dirs),
  `results/runs_crusoe/<tag>/` (145 dirs, incl. `point.log`).
- `research/` — new reports of this window + `aklr_mhc.md`, `colleagues/llying_alignment_20261120.md`.
- `process/` — mission, project CLAUDE.md, team-polls.md, watchdog.md, kf_status.md (full logs incl. earlier days).
- `SKIPPED_LARGE.txt` — files over 4 MB not copied (none).

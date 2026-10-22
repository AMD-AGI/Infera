# Working process — GLM-5.2-MXFP4 + DSA DCP + SGLang on MI355X, mixed and PD (Infera), 2026-10-21..22

Chronological distillation of the leader poll ledger and the teammate ledgers. All times UTC; 10-21 unless marked 10-22.
Every round is listed, including failures and aborted runs.

Source abbreviations (all copies of the local workspace
`/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021/`):
- `poll HH:MMZ` = `poll.md` entry at that time (leader, 20-min polls).
- `P0xx` = `ledger.pdrun.md` section; `M0/Rxx` = `ledger.mixrun.md`; `Txx/Sx` = `ledger.benchrun.md`;
  `coder2` = `ledger.coder2.md`; `analyst` = `ledger.analyst.md`.
- Cluster paths: `PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd`, `MIX=.../mixrun`, `BENCH=.../bench`.

Scoring conventions used below (details in `notes.md` §11): PD = total tok/s / 8 chips, warm-stack protocol (A1 fresh,
flush, A2 scored); mixed = total / 4 GPUs (InferenceX non-multinode); "smoke" = 300-600 s with `--unsafe-override`,
not a valid score; decodebench = median decode step ms at running == N with GPU halves swapped.

## Team and machines

| Teammate | Role |
|---|---|
| leader (main) | plan, polls, interventions, machine requests via pool manager `machines-and-clusters-b1` |
| analyst | code reads (Q1/Q2, PD+DSA DCP design), code fixes on `dev/yihou/glm52_dcp_pd`, design docs, tools |
| pdrun | PD image, Infera harness copy, all PD rounds on 054 then v3-060 |
| mixrun | mixed (non-PD) DCP4 vs reference, A/B decodebenches, DP2+DCP2 bring-up, then E2 (PD with DP2+DCP2 decode) on v3-070 |
| benchrun (from 16:13Z) | all-gather microbench, split-indexer (Variant B) A/Bs, kv_b bf16 / index_share / radix-merge A/Bs on v3-060 |
| coder2 (from 19:34Z) | DP2+DCP2 bug chain, isolated worktree branch `dev/yihou/glm52_dcp_pd_dp2` |
| watchdog | 10-min service checks, `watch.md` |

| Node | Job | Kind | Use | Notes |
|---|---|---|---|---|
| crsuse2-m2m-054 | 228279 | OAI | mixed eager (12:43-13:33Z), then PD r1..r4 | amdgpu 6.14.14; containers stopped 00:31Z 10-22 (poll 00:36Z) |
| crsuse2-m2m-268 | 227779 | OAI | assigned 13:00Z | NODE_FAIL 13:05:59Z, nothing ran (P007) |
| crsuse2-m2m-v3-045 | 229517 | burst, preemptible | mixed graph + AgentX, DP2+DCP2 | amdgpu 6.16.13, needs `HSA_ENABLE_SDMA=0` (M0 R8) |
| crsuse2-m2m-v3-060 | 229524 | burst, preemptible | benchrun, then PD r5/r5b (from 00:30Z 10-22) | same driver as v3-045 |
| crsuse2-m2m-v3-070 | 230266 | burst | E2 (PD + DP2xDCP2 decode) from 00:10Z 10-22 | same driver |

## Phase 0 — research and the Q1/Q2 answer (before 12:45Z)

- Research notes `/var/tmp/yihou/research_glm52_dcp/` (copied: `spec/research/`). Key facts: sglang main has DSA DCP
  only for CUDA SM100 trtllm (#43735) and rejects DSA DCP with PD and HiCache; AMD path is PR #44322 (triton sparse-MLA
  DCP), HiCache #45205 (`spec/research/critic.md`, `spec/mission.md` §2).
- 12:44Z analyst delivered `spec/answers/dcp_q1_q2.md` (code read only): Q1 DCP is a decode-phase feature (capacity and
  per-rank sparse-KV reads); in prefill it adds collectives. Q2 MTP + decode CUDA graph can both be on under DSA DCP
  (EAGLE chain, topk 1); prefill CUDA graph off. Leader spot-checked `pd_disaggregation_hook.py:57-62` and the
  cuda_graph_hook DCP rule (poll 12:44Z).

## Phase 1 — mixed (non-PD) DCP4 on 054, eager (12:38-13:33Z)

- M0: image `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020` (digest `sha256:3a75c913...`) + overlay of branch
  `4792dc0e52` (`python/sglang` mounted ro).
- R1 12:43Z eager DCP4 (GPUs 0-3) + ref TP4 no DCP (GPUs 4-7): both crashed at the first request 12:52Z
  (`module_gemm_a16w16_asm.so` missing). Cause: one shared aiter JIT dir across containers, PID-based build lock across
  PID namespaces -> builds clobbered (M0 R1). Fix: per-GPU-slot JIT dirs.
- R2 12:54Z: ref up; DCP4 died at RCCL init `HIP failure: 'invalid argument'` (cross-container PID collision).
  Fix: `PID_OFFSET=30000` (M0 R2). Did not recur.
- Eager correctness (M0 R2): DCP4 and ref both PASS math 29 (think/no-think), passkey 8K/34K/96K. Partial eager
  GSM8K (sgl-eval 0.1.3, GLM-5.2-FP8 preset, stopped at 13:24Z): ref 194/197 = 0.9848, DCP4 193/196 = 0.9847, identical
  wrong set. KV: DCP4 8,893,952 vs ref 2,860,160 (3.1x).
- R3 13:24Z graph servers on 054: check.py PASS both; GSM8K graph started but the leader stopped mixrun's containers at
  13:33Z (node handed to pdrun; poll 13:32Z intervention after 2 unanswered handover messages). User 13:30Z: no GSM8K for
  now (mission §0.3).

## Phase 2 — PD design and image (12:38-13:35Z)

- 12:51Z design `spec/design/pd_dcp_dsa.md` approved: launch guard for ROCm DCP decode in PD, ROCm sharded-draft relayout
  in Mooncake, index-K via the generic page path (analyst T1/T2 fail-first, T3 passes). Runbook `spec/design/t4_runbook.md`.
- P001-P003: PD image `infera-sglang:v0522-1020-yihou-dcp-mc4340` built on the v0.5.22 base with Mooncake #4340.
  r1 build failed (no installed `yalantinglibsConfig.cmake` in the v0522 base); fixed by an extra `cmake --install` (P003).
  Image ID `sha256:5b6565ef...`; saved to `PD/images/*.tar` (24 GB, P007) for other nodes.
- P004-P006: overlay with recipe patches -> folded into 9 "[yihou] infera recipe" commits on the branch by analyst;
  branch HEAD `ae529c96bf` = single code base for mixed and PD (poll 13:12Z, P008). Router `PD_DP_RANK_AFFINITY=0`
  mandatory when decode has one DP rank (P006).
- 13:14Z 268 lost (NODE_FAIL). 13:25Z user decision: one prefill = the 22k e2g leg (TP4+DPA, mode N, no EAGLE),
  only the decode changes (TP4/EP4 + DCP4); perf with sim 3.61; correctness with sim off (mission §0.3, P008).

## Phase 3 — PD + DCP4 correct, first perf (054, 13:33-14:47Z)

- **pdBe-r1** (eager decode, sim off): decode cold JIT ~11 min; PASS math 29 x2, passkey 8K/34K/96K; prefill pack buffers
  `4 x 351.0 MB` per DP rank as predicted by analyst for mode N; decode `dcp_size=4`, a2a, KV 9,764,096 (P010; milestone
  poll 14:04Z).
- **pdBeperf-r1** (graph + sim 3.61): short AgentX C8 600 s, 3,989 tok/s/chip, not comparable (P011). Driver died on a
  spur exec "transport error" while the client container kept running (poll 14:24Z) — first sighting of a recurring issue.
- 14:08Z analyst `spec/design/perf_layouts.md` + `tools/decode_stats.dcp.yihou.py`.

## Phase 4 — mixed graph + AgentX on v3-045 (13:36-18:19Z)

- R4-R7 13:40-14:10Z: weight load HANG on both servers (HIP `memcpy_and_sync` futex, `kfd_wait_on_events`, GPU idle).
  Excluded: sha, cpuset, `SAFETENSORS_FAST_GPU=1`, `enable_multithread_load=false`; single-process isolation tests pass.
- R8 14:11Z `HSA_ENABLE_SDMA=0` -> loads (M0 R8). All v3 results labelled "v3, SDMA=0". Reproduced by benchrun on v3-060
  16:39Z (T2 S1/S2); reported to the pool manager.
- R9 smokes (14:33Z, 300 s, not valid): DCP4 19,213 vs ref 16,496 tok/s/GPU. Observation: DCP4 prefill ~8.0-8.2K tok/s
  vs ref ~15-17K tok/s on 32K chunks (not explained).
- **full1-dcp4-c40** 15:02Z (2nd run on the stack): **26,277 tok/s/GPU**, TTFT mean 1.30 s, TPOT 36.9 ms, GPU hit 93.3 %
  (`MIX/agentx/full1-dcp4-c40/agentx_conc40.json`). **full1-ref-c40**: 5,130, warmup 4,099 s vs 892 s, hit 26.8 %, KV 99 %.
- **full1-dcp4-c48** (fresh, max-running 96): 13,012, hit 79.1 %, KV 99 % (#44322 author c48: 13,252).

## Phase 5 — PD C40 scored, decode levers (054 + v3-060 + v3-045, 14:52-22:50Z)

- **pdBeperf-r2** (HiCache 450, C40, decode custom AR OFF as in e2g, NCCL_DEBUG=INFO): A1 fresh **16,670** (P012).
  Chain driver stopped itself at 15:42Z treating the spur exec error as "client gone" — unnoticed 70 min (poll 16:53Z
  intervention). A2 attempt 1 aborted (HiCache 48 vs 450 config check, pdrun error); **A2b 17,053** (scored) (P012).
  Intervention 15:32Z/16:13Z: the other session's shared InferenceX cache was being touched; leader created a private
  copy `.../cache/InferenceX` (918524ff).
- decode_stats r2: A1 75.3 ms/step @ 21.4 running, A2b 66.5 @ 19.4; 22k e2g 40.9 per DP @ 3.57 per DP (P013).
- Decode kineto (18:35:56Z, running 30): one verify bs 24 70.6 ms — RCCL 17.8, indexer logits 15.6 + coop_topk 9.3, MoE
  10.3, GEMM 8.0, sparse MLA 3.4 (poll 18:38Z; `pdrun/traces_r2/breakdown.TP0.v2.analyst.txt`).
- Variant B (split decode indexer, `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER`, default off; analyst a04c3f8e0c):
  - benchrun T1 all-gather microbench: 2.1-5.0 ms/step added at rows 40-400.
  - T2/T3 correctness PASS; first coverage showed the split hook only in draft extend (gfx950 fused indexer serves rows
    <= 48), then fused declines at > 48 rows (analyst 17:01Z).
  - T5 smoke split 19,542 vs ref 17,409; **T6 swapped halves 16,914 vs 16,960 (equal)**; T5 was an outlier.
  - **T7 decodebench 34K: split slower by ~6 ms/step at N 16-32** -> r3 decision reversed (poll 19:03Z).
  - T8 traces: logits 3.4x smaller but +topk, +~15 small elementwise kernels, +AG. T9 (6ffb90ae6b) 150K parity,
    34K still +5.5-6 ms -> **Variant B shelved** (poll 19:50Z).
- Custom AR: mixrun R10 correctness ON/OFF both PASS (40/40); KV OFF 9,764,352 vs ON 8,894,208 (custom-AR buffers 9.8 %).
  R11 smokes inconclusive (ON 17,102 vs OFF 20,948, noisy). **Decodebench 34K: ON −3.5..−3.9 ms/step** (poll 19:28Z).
- **r3** (decode-only restart on the live r2 stack, custom AR ON, NCCL off): phase-1 gate PASS (math, passkey
  8K/34K/96K, real accept 3.05). AgentX **aborted** 19:13Z by the client: 10.6 % "No responses with actual content"
  (132 warmup errors from the first request). Then decode **hang after client cancel** (TP0/1 in broadcast, TP2/3 in
  pop_transferred all_reduce) (P014). pd_r3 driver had first died on a spur exec error at docker stop (pd_r3b fixed).
- Hang root cause (analyst code read): rank-local deferred decode KV release; fix **c4eda0343e** (MIN-reduce, group
  decision for the empty-queue skip, 5 CPU tests, 4 fail on old code). GPU cancel repro `tools/pd_cancel_repro.yihou.py`
  never run (poll 19:39Z).
- 19:31Z 5-h checkpoint: not blocked by SDMA/transfer -> no request for nodes 135/136 (poll 19:31Z).
- **r4** fresh full stack = r2 + custom AR ON + NCCL off: A1 driver lost to spur exec; `bench_retry.yihou.sh`; **A1b
  17,473** (0 profiling errors -> r3 errors not reproduced on a fresh stack); **A2 18,321** (TPOT 19.7 ms, TTFT 3.93 s,
  61.2 ms/step @ 18.75 running) — best PD score (P015, poll 22:50Z).
- Long-context baseline (mixrun 19:50Z): 131K vs 34K +5/+10/+13 % at N 8/16/24.
- kv_b bf16 (`SGLANG_DSA_DCP_KEEP_KV_B_BF16`, 636cc68dd0): T10 **−1.6..−1.7 ms at N >= 16**, weights +0.34 GB/GPU,
  KV −0.28 % (analyst's +6.5 GB estimate was wrong); q-rep active 78 layers; correctness PASS (poll 20:30Z).
- index_share true vs false: T11 + mixrun R20-R22 equal within 0.6 ms; correctness PASS both; PD keeps infera `false`.
  Finding: checkpoint `config.json` sets true, so all mixed numbers ran true (poll 20:38Z). T11 swap pair crashed once
  with RCCL "invalid device pointer" when py-spy was attached during graph capture; no recurrence without py-spy.
- Split indexer Triton (0347a6d47e, sort merge): T12 vs ref 34K +2.0..+2.6, 150K −1.1/−2.9 ms. Radix-select merge
  a6ea78e322 failed Triton compile on GPU (constexpr reassigned in `static_range`; CPU interpreter had passed, T14);
  fixed **93034eba91**: merge 46 µs/layer (was 102), T15 150K −2.6..−4.2 ms, 34K +0.5..+1.5; **T16 200K −4.4 / −6.6 /
  −9.2 ms at N 16/24/32** (poll 23:43Z, 00:01Z 10-22).
- Context load of PD AgentX (T13, `tools/agentx_context_load.yihou.py`): r2 A2b / r4 A1b ~20.5-21.1 rows, ~180K/row,
  Σ 3.7-3.8M; at the profiled instant Σ 5.6-6.5M, mean 216K, max 697K (T13b). Analyst correction 22:21Z: the 15.6 vs
  8.36 ms indexer comparison was graph vs eager and not like-for-like.

## Phase 6 — DP2 x DCP2 bug chain (mixed, v3-045, 19:29-22:32Z)

| Round | Sha | Result | Source |
|---|---|---|---|
| R12 eager | not recorded in M R12 | crash 1st request `prepare.py:246` 49 vs 50 rows | M R12 |
| fix 31309dc4f1 | coder2 | `graph_utils.padded()` did not pad batch-major `[1,seq,H,D]` DSA triton prefill output | coder2 |
| R13 eager | 31309dc4f1 | MAF on idle DP0 ranks after DP1 38.9K prefill | M R13 |
| R14 AR off | same | MAF again under load -> not custom AR | M R14 |
| R15 RS=0 | same | MAF -> not aiter reduce-scatter | M R15 |
| R16 `AMD_SERIALIZE_KERNEL=3` | same | stack: NextN draft IDLE forward -> `_topk_transform_v2_paged` | M R16 |
| R17 debug agent | same | waves in `topk_main_kernel<..., TopKMode 1>` | M R17 |
| fix fdf73b6fb2 | coder2 | ROCm lacked the eager-IDLE indexer short circuit (gated `_is_cuda`) | coder2 |
| R18 eager | fdf73b6fb2 | 5/5 + 40/40 PASS; new MAF in 4x1500 greedy | M R18 |
| R19 debug agent | same | MAF in `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | M R19 |
| R23 bounds check | bd6a337ab7 (debug) | DRAFT_EXTEND_V2 q 42 rows vs metadata 36 | M R23 |
| fix fa810b9f69 | coder2 | `_pad_inputs_to_size` extended `extend_seq_lens_cpu` in place (shared list) | coder2 |
| R24 eager x2, R25 graph | fa810b9f69 | check 5/5, 40/40, 4x1500, no fault | M R24/R25 |
| decodebench A/B | fa810b9f69 | 34K equal; 131K N 16/24 −2.9/−3.6 ms (−6 %) both halves | M (poll 22:32Z) |

Cherry-picks onto the main branch: b17da53725, e9e1432ce6, da6e383513, 5aacce1f1c (poll 19:54Z, 20:20Z, 21:38Z).

## Phase 7 — paired mixed AgentX DCP4 vs DP2+DCP2, PD C52 (22:36Z-00:31Z 10-22)

- Mixed paired c40 on 5aacce1f1c + kvb + AR on: DP2+DCP2 arm thrashed (warmup 221/444 after 2,404 s; prefill cached 64 %
  vs ~94 %; per-DP cache_hit_rate 0.0) -> stopped 23:23Z, no score. Leading hypothesis: in-engine DP routing not
  prefix-aware (option list read; not measured). **DCP4 + kvb: 26,152** (vs 26,277 without kvb) (M, poll 00:02Z).
- **r4 C52** (warm, after flush): **18,070** < C40 18,321; TTFT mean 14.44 s, p95 56.6 s; running mean 18.31 (P015).
  054 stopped 00:31Z, deadline 00:55Z met (poll 00:36Z).
- Other prep: analyst Phase-6 merge notes `spec/design/phase6_merge.md` (idlefix #45483 5fb03011d6, OffloaderV2
  `is_shuffled` c9394c114f; cross-node DPA4 prefill -> DCP4 decode is off-rail), infera session reply on rails (poll
  19:48Z); PP pre-work 646dc699f3 / 6ee35f3617 (CPU only, `spec/design/pp_prefill.md`); `spec/design/next_node_plan.md`
  E1-E5 approved 22:14Z.

## Phase 8 — next nodes: r5 (DCP4 full knobs) and E2 (PD + DP2xDCP2 decode) (10-22 00:10Z-)

- E2 gate 1 **pdE2-r1** (v3-070, eager, sim off): PASS 5/5, 40/40, 4x1500, accept 2.55 (poll 00:59Z 10-22).
- **r5 gate pdBe-r5g** (v3-060, graph, sim off, sha 93034eba91, custom AR + kvb + split-triton radix + seq-len log):
  PASS math 29 x2, passkey 8K/34K/96K, accept 2.85; "prepared full-head Q weights for 78 MLA layers" (P016).
- E2 gate 2 **pdE2-r2** (graph, sim off): **3/200 garbled** (first tokens right, then junk) (poll 01:15Z).
  - **pdE2-r3** custom AR off: crashed at graph capture (`hipErrorCapturedEvent` in the attention_tp NCCL watchdog).
  - pdE2-r4 aborted (index_share knob is hard-set in config).
  - **pdE2-r5** index_share true: 13/240 bad.
  - **pdE2-r6** split indexer off: 27/320 bad.
  - (Rates not separable from round variance.)
- **r5** A1 died 01:42:52Z: `rocm-dbgapi: fatal error: os_queue_packet_id ... not within ...` in a prefill scheduler
  (`DEBUG_AGENT=1`); relaunched **r5b** with `DEBUG_AGENT=0`, gate skipped (P016).
- 02:05Z **user decision**: E2 garbling = recorded known risk, not pursued (mission §0). Required 3x40 sim-off check
  between r5b A1 and A2 dropped (poll 02:13Z).
- 02:08Z pd_seq double-bench bug fixed (`break if JSON present`); continuation `pd_cont.yihou.sh` (P016).
- E2 perf **pdE2-p1** (DP2+DCP2, sim 3.61, HiCache 450) started 02:19Z (prefill ok 02:25:34Z per
  `PD/runs/pdE2-p1/status.tsv`); per poll 02:35Z mixrun is told to stop it for a transfer-time measurement.
- P017 PD queue readout (r2 A2b, r4 A2, r4 C52): decode running flat ~18.5-19.4; C40 -> C52 adds load in decode
  transfer-in (4.9 -> 18.2) and prefill in-flight (1.8 -> 3.6 per rank). Transfer-bound pattern per the skill, not
  confirmed by transfer timing.
- 02:25Z user challenge ("DCP slower than TP4+DPA? misconfig?") -> `spec/design/pd_vs_mixed_decode_diff.md`: PD decode
  step +1.4..+5 ms over mixed decode-only steps at matched running; config differences TOPK_V2 (PD false / mixed true) and
  debug agent (PD on). 02:27Z analyst code read: with the split knobs TOPK_V2 is not read on decode -> TOPK_V2 A/B
  cancelled; mixrun -> MC_TE_METRIC transfer-time measurement on v3-070 (poll 02:35Z/02:27Z entries).
- Status at pack-up (02:29Z 10-22): **r5b A1 running** (client log at 02:29:06Z: srv prefix hit 95.3 %, kv_usage 63 %;
  `PD/runs/pdBeperf-r5b/agentx-c40-A1/benchmark.log`); **r5b A1/A2 scores pending**; **E2 perf pending/stopped**;
  transfer-time measurement pending.

## Code produced (branch `dev/yihou/glm52_dcp_pd`, HEAD 93034eba91)

See `spec/` design docs and RESULTS §5 (local `RESULTS.md`). Main commits: a6bdc7894d (PD DCP relayout of the ROCm
sharded draft), 228263d116 (allow ROCm DCP decode in PD), ae529c96bf (env PD MTP seed switch), c4eda0343e (decode
cancel hang), 6ee35f3617 (PP non-final chunk wait event), b17da53725 / e9e1432ce6 / da6e383513 / 5aacce1f1c (DP2),
5fb03011d6 (idlefix #45483), c9394c114f (OffloaderV2 is_shuffled), 636cc68dd0 (KEEP_KV_B_BF16), 0347a6d47e /
a6ea78e322 / 93034eba91 (split indexer Triton + radix merge), 9af2166f3f (local debug seq-len log).

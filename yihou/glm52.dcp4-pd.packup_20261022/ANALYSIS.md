# ANALYSIS — explanations behind the numbers (2026-10-22)

Companion to `README.md` / `results/RESULT.md`. Each section answers one question that came up while running the
experiment, states what was measured or read, and marks what is still open. "Code read" = read on branch
`dev/yihou/glm52_dcp_pd` @ `93034eba91`. Numbers cite files in this kit unless marked EXTERNAL.

## 1. Status: which DCP PD run completed end to end

| Run | Node | Code | Protocol | Result | Evidence |
|---|---|---|---|---|---|
| **r4** (last complete scored run) | 054, amdgpu 6.14.14 | `ae529c96bf` | A1 3600 s -> flush -> A2 3600 s | A1b 17,473; **A2 18,321 tok/s/chip**; C52 18,070 | `results/pd/pdBeperf-r4/` |
| r5b (current config) | v3-060, 6.16.13, SDMA=0 | `93034eba91` | same | **A1 19,400** (fresh, not scored); A2 lost (preemption 03:36Z) | `results/pd/pdBeperf-r5b/` |
| r5c / r5c2 / r5d | v3-033 / v3-033 / v3-027 | `93034eba91` | A1 900 s -> flush -> A2 | r5c2 A1 16,378, r5d A1 16,544 (900 s, not comparable); all A2 lost to preemption | `WS/poll.md` 04:07-06:26Z |
| r6 | OAI 046, 6.14.14 | `93034eba91` | full | lost: node NODE_FAIL (scontrol Reason=NodeDown) 07:18:39Z, ~1 min into A1 profiling (§6) | `pd/runs/pdBeperf-r6/` (EXTERNAL) |
| r6b | OAI 105, 6.14.14 | `93034eba91` | full | **running** since 07:31Z | `pd/runs/pd_seq.r6b.log` (EXTERNAL) |

Target 20k tok/s/chip: not met on any scored A2 yet. Config of r5b vs r4: `HANDOFF_r5b.md` §1.

## 2. Why the DCP4 decode loads ~17.6 GB less weight than the TP4+DPA decode

**Measured** (decode "Load weight end", per GPU):

| | 22k baseline decode (TP4 + DPA4) | r4 decode (TP4 + DCP4) | diff |
|---|---|---|---|
| target model | 119.82 GB | 102.23 GB | **17.59 GB** |
| MTP draft (NextN) | 6.92 GB | 5.46 GB | 1.46 GB |

Sources: EXTERNAL `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/e2g/server-logs/decode-0.a1.log`,
`/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/runs/pdBeperf-r4/server-logs/decode-0.a1.log`.

**Mechanism (code read + model config):** the difference is the attention TP size, not DCP itself.
- With DP attention, `attn_tp_size = 1`: every GPU holds the full attention weights. Our DCP4 decode runs without DPA,
  `attn_tp_size = 4`.
- `python/sglang/srt/models/deepseek_v2.py:2068-2160`: `fused_qkv_a_proj_with_mqa` is `ReplicatedLinear` (not split);
  `q_b_proj`, `kv_b_proj` (`ColumnParallelLinear(parallel_group="attn_tp")`) and `o_proj`
  (`RowParallelLinear(parallel_group="attn_tp")`) are split by head over attn_tp.
- GLM-5.2-MXFP4 keeps all attention projections out of FP4 (`config.json` `quantization_config.exclude`: 79 entries
  each for q_a/q_b/kv_a/kv_b/o_proj), i.e. bf16.
- Config: hidden 6144, 64 heads, q_lora 2048, kv_lora 512, qk_nope 192, v_head 256, 78 layers. Split params per
  layer: q_b 2048x(64x256) = 33.6M, kv_b 512x(64x448) = 14.7M, o_proj 16384x6144 = 100.7M -> 148.9M = 297.8 MB bf16.
  attn_tp 4 keeps 1/4 -> saves 223 MB/layer -> x78 = **16.2 GiB** of the 17.59 GB (sglang's "GB" here is GiB).
- **Unexplained: ~1.4 GiB** (target) and most of the draft's 1.46 GB (one attention layer explains ~0.21 GiB).
  Not lm_head: `enable_dp_lm_head=False` on both. Candidates, not measured: image/version (v0521 vs v0522 + overlay),
  MoE layout (baseline EP1, ours EP4). Decisive measurement: same image and sha, change one of them, read "Load weight end".
- Any attn_tp 4 layout saves this. What DCP adds is keeping KV capacity at attn_tp 4 (KV striped over 4 ranks).

## 3. TPOT vs decode step time, and where DCP4 loses to DPA4

- A decode **step** emits ~3.6 tokens per request (EAGLE, sim accept 3.61), so TPOT ≈ step / accept.
  22k baseline: step p50 ~37 ms per DP rank -> ~10.3 ms/token; measured TPOT mean 12.1 ms (p95 17.9)
  (baseline kit `runs_summary.csv`, row e2g-A2).
- Same 4 decode GPUs, decode output = running x accept / step (decode_stats means over the AgentX window):

  | | running (4 GPUs) | step | accept | output |
  |---|---|---|---|---|
  | 22k baseline (DPA4) | 14.3 (4 x 3.57) | 40.9 ms per DP, in parallel | 3.57 | **~1.25 tok/ms** |
  | r4 A2 (DCP4) | 18.75 | 61.2 ms | 3.61 | ~1.11 tok/ms |
  | r5b A1 (DCP4) | 16.7 | 52.7 ms | 3.61 | ~1.14 tok/ms |

  DCP4 produces ~9-11 % less per 4 decode GPUs. The higher running count is a consequence (fixed C40, slower decode
  -> requests stay longer), not a cause. **Why DCP4 produces less is not established.** The only DCP4 trace
  (r2, bs 24, 70.6 ms/step: RCCL 17.8, indexer logits 15.6, MoE 10.3, coop top-k 9.3, GEMM 8.0, sparse MLA 3.4 ms,
  `results/RESULT.md` §1) has no matched-load DPA4 trace to compare with. Planned measurement: §7.
- Step p50 / p90 by running (ms, TP0; `results/pd/step_quantiles.r4A2_r5bA1_e2gA2.txt`):

  | running | r4 A2 | r5b A1 |
  |---|---|---|
  | 9-16 | 50.7 / 61.2 | 47.4 / 55.2 |
  | 17-24 | 67.0 / 73.5 | 57.3 / 65.3 |
  | 25-32 | 76.4 / 81.3 | 66.6 / 76.4 |
  | all | 65.0 / 78.1 | 53.1 / 68.1 |

  22k baseline per DP rank: ~37.0 / ~48 ms (almost all steps at 1-8 running/DP).
  r4 -> r5b (~-10 ms/step at 17-40 running) changed node, driver/SDMA, debug agent, kv_b, split, sha, and A2 -> A1 at
  once; not attributable to a single knob.

## 4. Is PD throughput limited by KV transfer?

Queue-depth readout (llm-pd-bottleneck-finder method; `results/pdq/`, means per logged step):

| C40 | 22k baseline A2 (node totals) | r4 A2 | r5b A1 |
|---|---|---|---|
| prefill queue / rank | 0.8 | 0.3 | 0.4 |
| prefill in-flight (KV send) / rank | 1.4 | 1.8 | 1.8 |
| decode transfer-in | ~5.2 | 4.9 | 6.2 |
| decode running | ~14.4 | 18.9 | 16.9 |
| decode prealloc / retracted | 0 / 0 | 0 / 0 | 0 / 0 |

- At C40 no stage is backed up, and transfer-in is ~5 in both ours and the baseline: **transfer does not separate us
  from the baseline at C40.**
- C40 -> C52 (r4): decode running flat (18.9 -> 18.5) while transfer-in 4.9 -> 18.2 and TTFT mean 14.4 s; C52 scored
  lower (18,070). That is the method's transfer-bound pattern at C52, unconfirmed (no per-transfer timing recorded).
- Same bytes per token on both (78 x 576 B = 44,928 B; log "PD DCP pack buffers allocated: 4 x 351.0 MB"); ours sends
  each request to 4 decode ranks (`required_dst_info_num = 4`), the baseline to 1.
- KV transport is identical to the 22k kit: Mooncake HIP IPC over XGMI (`MC_TE_FILTERS` = nonexistent device,
  `MC_DISABLE_HIP_TRANSPORT=1`); `scripts/harness/dryrun.pdBeperf-r5b-060.yihou.txt`.

## 5. `SGLANG_OPT_USE_TOPK_V2` difference (mixed true, PD false) — does it matter?

Code read: with `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1` + `SPLIT_INDEXER_TRITON=1`, TOPK_V2 is never read on the
decode path. Target verify / draft decode with <= 48 rows use the fused gfx950 indexer (`dsa_indexer.py:2201-2215`);
> 48 rows and draft-extend take the split path (`dsa_indexer.py:1429`, gate `:1557`, return before `topk_transform`
at `:1522-1534`), whose local top-k is `sgl_kernel.fast_topk_v2` (`dsa_topk_backend.py:58-68`). TOPK_V2 is read only in
`topk_transform` (`dsa_topk_backend.py:55-56`). It matters only for a no-split config (r2's 9.3 ms coop top-k).

## 6. Infrastructure findings (what cost the most time on 10-22)

- **Burst (p2) preemption:** v3-060 (after ~14 h), v3-070, v3-033 (x2), v3-027 lost between 03:35 and 06:26Z; holds
  were 20-90 min. A full A1+A2 (~2.5-3 h with cold bring-up) does not fit; burst is for < 30-min, restartable work.
- **Cluster-wide job end at ~07:18Z:** all three of our jobs, including the non-preemptible OAI 046, ended within
  30 s (sacct COMPLETED, ExitCode -1:0); many other users' jobs ended 07:16-07:21Z and the slurm controller answering
  changed (-005 -> -006). Pool manager: not a pool cancel; cause not confirmed.
- **Containers outlive the job.** Fix: `scripts/harness/pd-tools/job_container_guard.yihou.sh` (v4). A host loop inside
  the job refreshes a heartbeat inside each `yihou-dcp-pd-<NK>-*` container via `docker exec` (no docker socket:
  burst nodes' docker-authz denies socket mounts); it stops refreshing once `squeue` (or `sacct` when squeue is empty
  for a gone job) shows a non-RUNNING state; an in-container watcher sends SIGTERM to its own PID 1 after 90 s stale.
  Event log on NFS `pd/guard/<JOB>.<NK>.log`. **Verified in the wild:** v3-075 NODE_FAIL 07:18:38Z -> both A/B servers
  logged "SIGTERM received" 07:20:13Z and "Finished server process" 07:20:18Z (~95 s). Containers without `/bin/sh`
  (distroless etcd) cannot be guarded.
- ROCm debug agent ON aborted a prefill in 2/2 perf runs on amdgpu 6.16 (`rocm-dbgapi: fatal error:
  os_queue_packet_id ...`); perf runs use `DEBUG_AGENT=0`.
- Message delivery between agents lagged up to ~85 min once (r6 assignment); unacked assignments are re-sent or
  executed by the leader.

## 7. Plan (as of 07:35Z 10-22)

1. Scored A2 of the r5b config on a non-preemptible node: r6b on OAI 105 (A2 ~10:40Z).
2. Matched-load decode A/B on one node: DPA4 (22k layout, EP1) vs DCP4 (r5b layout, EP4), ~180K context, N = 16/24,
   one kernel trace per arm at N = 24, per-category breakdown (`mixrun/ab_launch.sh`, `ab_bench.sh`, `trace_cats.py`,
   EXTERNAL). Both arms mem 0.85 (22k used 0.95). Attempted on v3-075, lost to NODE_FAIL before the bench.
3. Single-variable A/Bs chosen by (2): MoE EP1 vs EP4; split / kv_b / debug agent separately on a 6.14 node.
4. DCP4 PD sim-off 3 x 40 concurrent correctness check (only 5-request gates exist).
5. Phase 7 (PP prefill) is deprioritized pending the user: prefill computes ~18.8M new tokens/h vs 540M cached
   (`results/pd/pdBeperf-r5b/decode_stats.r5b.txt`).

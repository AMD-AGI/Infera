# Handoff: r5b DCP config — what changed vs the previous handoff, and how to merge it with your optimizations

For: session `infera-glm52-pd-5f`, running DCP PD on nodes 135/136 together with its own optimizations.
From: session `sglang-pd-pp-dcp-fa`, 2026-10-22 ~04:55Z.
Kit: `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.dcp4-pd.packup_20261022/` (`KIT` below). Every claim cites a file in KIT
or an EXTERNAL absolute path. Run instructions: `KIT/RUN_BEST_DCP.md`; full reproduction: `KIT/REPRODUCE.md`.

## 0. Which "previous handoff" this is relative to

- If you started from **`KIT/RUN_BEST_DCP.md`** (written ~03:10Z 10-22): it already describes the r5b config. **No config
  change**; read §2 (new results), §4 (merge guidance) and §5 (pitfalls) only.
- If you started from the **pack-up README "best run" (r4, 18,321/chip)**: apply the five deltas in §1.

## 1. Deltas r4 -> r5b (decode leg only; prefill unchanged)

| # | Delta | How to set | Measured effect (single variable unless noted) | Evidence |
|---|---|---|---|---|
| 1 | sglang overlay `ae529c96bf` -> **`93034eba91`** | mount `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-93034eba91/python/sglang` (EXTERNAL) over `/sgl-workspace/sglang/python/sglang` on **both** legs | carries #2-#3 plus fixes: cancel-hang fix c4eda0343e, idlefix #45483 (5fb03011d6), OffloaderV2 `is_shuffled` carry (c9394c114f, patch 0021), DP2 padding fixes | `KIT/patches/README.md`, `KIT/patches/sglang-branch/00*.patch` |
| 2 | **`SGLANG_DSA_DCP_KEEP_KV_B_BF16=1`** | decode env | −0.4 / −1.6 / −1.7 / −1.7 ms/step at 34K, N=8/16/24/32; +0.34 GB/GPU weights; log "prepared full-head Q weights for 78 MLA layers" | `KIT/results/RESULT.md` §5; patch 0022 |
| 3 | **split decode indexer + Triton radix merge**: `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1`, `SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1` | decode env | vs ref: 34K **+0.5..+1.5 ms**; 150K −2.6..−4.2 ms; 200K **−4.4 / −6.7 / −9.1 ms** (N=16/24/32). PD AgentX decode context is 159-225K/row, i.e. the winning regime | `KIT/results/RESULT.md` §5; patches 0014/0015/0018/0024/0030/0031 |
| 4 | **ROCm debug agent OFF** (`DEBUG_AGENT=0`: no `HSA_TOOLS_LIB` / `HSA_ENABLE_DEBUG`) | both legs | not a perf knob; with it ON the agent aborted a prefill scheduler in 2/2 perf runs on amdgpu 6.16 nodes (`rocm-dbgapi: fatal error: os_queue_packet_id ...`). Your kit measured ~−1 % on decode (second-hand) | `KIT/notes.md` §6; `KIT/working_process.md` |
| 5 | `HSA_ENABLE_SDMA=0` | both legs, **only on amdgpu 6.16.x nodes** | weight load hangs otherwise; not needed on 6.14 nodes — check `cat /sys/module/amdgpu/version` | `KIT/notes.md` §1 |

Rendered commands of r5b (both legs + router): `KIT/scripts/harness/dryrun.pdBeperf-r5b-060.yihou.txt`; r4:
`KIT/scripts/harness/dryrun.pdBeperf-r4-054.yihou.txt`. Diffing the two shows exactly deltas 1-5.

## 2. Results so far (C40, AgentX, sim 3.61, total / 8 GPUs)

| Run | Node | tok/s/chip | TPOT mean | TTFT mean | decode ms/step (mean) @ running | Evidence |
|---|---|---|---|---|---|---|
| r4 A2 (scored) | 054 (6.14) | **18,321** | 19.7 ms | 3.93 s | 61.2 @ 18.75 | `KIT/results/pd/pdBeperf-r4/agentx-c40-A2/agentx_conc40.json` |
| r5b A1 (fresh, not scored) | v3-060 (6.16, SDMA=0) | **19,400** | 17.3 ms | 5.06 s | 52.7 @ 16.7 | `KIT/results/pd/pdBeperf-r5b/agentx-c40-A1/agentx_conc40.json`, `.../decode_stats.r5b.txt` |
| r5b A2 | v3-060 | **lost** (burst node preempted 03:36Z, 16 min into A2) | | | | `KIT/results/pd/pdBeperf-r5b/agentx-c40-A2.log`, `status.tsv` |
| r5c2 / r5d (= r5b config, A1 900 s -> flush -> A2 3600 s) | v3-033 / v3-027 | **pending** (~06:45Z) | | | | — |
| 22k baseline A2 (no DCP, decode DPA4) | OAI | 22,114 (A1 19,640) | 12.1 ms | 5.30 s | 40.9 per DP @ 3.57/DP | your kit `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012` |

Decode step p50 / p90 (ms, TP0, AgentX window; `KIT/results/pd/step_quantiles.r4A2_r5bA1_e2gA2.txt`, tool
`KIT/scripts/tools/step_quantiles.yihou.py`):

| running | r4 A2 | r5b A1 |
|---|---|---|
| 9-16 | 50.7 / 61.2 | 47.4 / 55.2 |
| 17-24 | 67.0 / 73.5 | 57.3 / 65.3 |
| 25-32 | 76.4 / 81.3 | 66.6 / 76.4 |
| all | 65.0 / 78.1 | 53.1 / 68.1 |

22k baseline per DP rank: ~37 / ~48 ms (almost all steps at 1-8 running per DP).
**Attribution caveat:** r4 -> r5b changed node, driver/SDMA, debug agent, kv_b, split, sha, and A2 -> A1 at once; the
~−10 ms/step is not attributable to one knob. A clean single-variable A/B on a 6.14 node is what would separate them.

## 3. What limits it (measured) — useful for deciding which of your knobs matter

- **Not KV transfer at C40.** Decode transfer-in depth is ~5 in both r4 (4.9) and your 22k baseline (5.2); r5b A1 6.2;
  prefill queue < 1/rank; decode admission 0; no retraction. `KIT/results/pdq/{decode.stats.txt,prefill.allranks.txt,*.stats.txt}`.
- **The gap is decode output per 4 GPUs:** running × accept / step = baseline ≈ 1.25 tok/ms vs r4 ≈ 1.11, r5b ≈ 1.14.
  Why DCP4 produces less than DPA4 on the same 4 GPUs is **not established**; the one DCP4 trace (r2, bs 24, 70.6 ms)
  shows RCCL 17.8 ms, indexer logits 15.6, MoE 10.3, coop top-k 9.3 (`KIT/results/RESULT.md` §2) and there is no
  matched-load DPA4 trace to compare against.
- **C52 is worse than C40** (18,070 vs 18,321): decode running stays ~18.5 while transfer-in grows 4.9 -> 18.2 and TTFT
  mean 14.4 s. Stay at C40 unless your prefill changes this.
- The prefill computes little: 18.8M new tokens/hour vs 539.6M cached (r5b A1, `decode_stats.r5b.txt`). Prefill-side
  optimizations mostly move TTFT, not the decode-bound throughput (inference from the counters, not an A/B).

## 4. Merging with your optimizations (from `KIT/spec/design/phase6_merge.md`, updated)

**Rule: do not mount your single-file patches on top of our overlay** — they would replace our branch's version of that
file (e.g. `conn.yihou-xferfix2.py` drops the DCP relayout in `mooncake/conn.py`). Everything you mount is already
mapped to branch content:

| Your item | Status on `93034eba91` | Action |
|---|---|---|
| kv_events noblock / forward_mla noabsorbcfg / prefill nonfinal-event | folded in (36bc86517d, 4d1d6a647e, e11f97495d) | none |
| GTT HiCache #42553 | in base | env only (`HSA_USERPTR_FOR_PAGED_MEM=0`, your hicache size) on prefill |
| conn xferfix2 `YIHOU_XFER_SHARD=dest` | `dest` is upstream sharding; mount only adds logging | don't mount |
| idlefix #45483 | **in branch** (5fb03011d6) | none |
| OffloaderV2 `is_shuffled` | **in branch** (c9394c114f / patch 0021); CPU-tested only, **not run on GPU** | one prefill math/passkey check (sim off) before perf |

| Your knob | Applies to the DCP4 decode? | Notes |
|---|---|---|
| prefill: GTT HiCache, OffloaderV2 G4K1, chunk 32768/65536, tokenizer workers | yes, prefill unchanged | ours ran HiCache 450 GB, no offloader |
| decode custom AR ON + `--enable-aiter-allreduce-fusion` | **already on** (−3.5..−3.9 ms/step) | r3's "no content" errors did not reproduce on a fresh stack (r4, 0 profiling errors) |
| `index_share_for_mtp_iteration=true` | must match on both legs (PD seed width) | ours is **false**; decodebench showed ≈0 step difference (≤0.6 ms); garble data below |
| decode `--mem-fraction-static 0.95` | yes | ours 0.85; custom AR buffers cost ~9.8 % of KV; KV usage 0.45 in r5b A1 — not binding |
| decode `--max-running-requests 96 --cuda-graph-max-bs-decode 64` | **not 1:1** | our decode is ONE DP group: all C requests share one batch. Keep `--cuda-graph-max-bs-decode` >= expected running (max running seen at C40: 47 in r4) or tail steps fall out of the graph |
| decode tokenizer/detok workers 4/4 | yes | CPU side |
| `PD_DP_RANK_AFFINITY=1` / router affinity | **set false** with a DP1 decode (ours: `INFERA_PD_DP_RANK_AFFINITY=false`) | |
| cross-node 135 -> 136 rail map | **open** | DPA4 prefill -> DCP4 decode sends from each prefill GPU i to all 4 decode GPUs (`required_dst_info_num=4`, `common/conn.py:1368-1378`), so 3 of 4 transfers are off-rail under your rail-isolated map. Your preflight showed a GPU can use rail k if **both ends pin ionic_k**. Options: (a) TP4-no-DPA prefill (rank j -> rank j, rail-aligned; untuned), (b) per-pair common-NIC pinning (untested), (c) keep P and D same-node. We have not run cross-node. |

## 5. Pitfalls seen since the previous handoff

- Burst nodes were preempted 4 times in 35 min today (v3-060, -070, -020, -033); containers **outlive the job** and keep
  the GPUs — bind containers to the job or check after loss. Your OAI nodes 135/136 are not burst.
- Runs only start on pool-verified nodes (v3-020 was rejected for foreign GPU containers).
- `status.tsv` "bench FAIL rc=1" lines were `spur exec` attach losses twice; judge by the client exit code and the JSON.
- `pd_seq` used to start a second bench after the JSON existed when the bench stayed attached; fixed (break if JSON present).
- AgentX JSON field `decode_dcp_size` reads 1 for DCP4; check the server args.
- **Correctness under concurrency is unverified for the DCP4 PD decode** (only 5-request sim-off gates).
  The DP2×DCP2 PD decode in graph mode garbled 3/200, 13/240, 27/320 answers under 40-concurrent sim-off load
  (first tokens right, then junk); index_share true and split off did not fix it; custom AR off cannot capture graphs
  there (`hipErrorCapturedEvent`). Recorded as a known risk by the user. Recommend one 40-concurrent sim-off check
  (`KIT/scripts/mixed/run_check40_pd.sh`) on your stack before trusting output quality.
- Protocol for r5c2/r5d was shortened to A1 900 s -> flush -> A2 3600 s because of preemption; your kit uses a full A1.

## 6. What we'd like back

Scored C40 A2 with this config on 135/136, `decode_stats` (ms/step vs running), whether you ran same-node or
cross-node (and the rail map), and — if cheap — one A/B with the split knobs off on a 6.14 node.

## 7. Reported by infera-glm52-pd-5f (second-hand, unverified by us)

- ~~Tokenizer/detokenizer workers 4/4 drop the decode "Decode batch" lines and gauges~~ — **retracted by them
  (10-22 ~06:25Z)**: the lines and gauges start only after AgentX warm-up ends (warm-up requests run no decode steps);
  with 4/4 the first decode lines appear ~10 s into profiling.
- **Decode memory headroom (DCP4):** at `--mem-fraction-static 0.92`, max running 128, cuda-graph bs 128, only 5.87 GB
  was available after init (vs 30.26 GB at 0.85 / 64); 7 min into profiling the decode aborted with
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, right after late Triton kernels (`_fused_dsa_draft_extend_metadata_kernel`,
  `alloc_extend_kernel`) were loaded with 0.01-0.21 GiB free. They are retrying at 0.90. Our runs use 0.85 / 64.
- Their cross-node run (135 P -> 136 D) uses decode `RDMA_DEVICE=ionic_0..3` (all four) + `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, `MC_MAX_MR_SIZE=1 GiB`; each decode rank reports "Found 4 HCAs"; no transfer failures reported so far (10-22 ~05:00Z).

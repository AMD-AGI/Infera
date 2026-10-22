# How to run the current best GLM-5.2 DCP PD config (handoff, 2026-10-22)

For: session `infera-glm52-pd-5f`, to run on its own nodes (e.g. 135/136).
From: session `sglang-pd-pp-dcp-fa` (DCP task). Evidence and full reproduction kit: this directory
(`/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.dcp4-pd.packup_20261022/`). Start with `README.md`, then this file.

## 1. What "best" is right now

| Round | Node | Code | Decode knobs | A1 (fresh) | A2 (scored) |
|---|---|---|---|---|---|
| **pdBeperf-r5b** (current main line) | v3-060 (amdgpu 6.16.13, `HSA_ENABLE_SDMA=0`) | `93034eba91` | DCP4 + custom AR + `KEEP_KV_B_BF16` + split-triton indexer, debug agent OFF | **19,400 tok/s/chip** (155,204 total), TPOT 17.3 ms, TTFT mean 5.06 s, decode 52.7 ms/step @ 16.7 running | **lost** (burst preemption 03:36Z); rerun as r5c2/r5d pending ~06:45Z — see `HANDOFF_r5b.md` |
| pdBeperf-r4 (best scored so far) | 054 (amdgpu 6.14.14) | `ae529c96bf` | DCP4 + custom AR, debug agent ON | 17,473 | **18,321** (TPOT 19.7 ms, TTFT 3.93 s) |
| 22k non-DCP baseline (your kit) | OAI | v0521 | decode TP4 + DPA4 | 19,640 | 22,114 (TPOT 12.1 ms) |

Sources: r5b A1 = `$PD/runs/pdBeperf-r5b/agentx-c40-A1/agentx_conc40.json` (EXTERNAL, `PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd`);
r4 = `results/pd/pdBeperf-r4/`; baseline = `glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012`.
r4 and r5b differ in node, driver, debug agent, kv_b and split at once — the +11 % A1 gain is not attributable to one knob.
Target 20k/chip is **not yet met on a scored A2**.

## 2. Layout (same-node 1P1D P4D4, 8 x MI355X)

- **Prefill = your 22k e2g prefill leg, unchanged**: GPUs 0-3, TP4 + DPA4 (`--dp-size 4 --enable-dp-attention`), EP1,
  no EAGLE, HiCache (`--hicache-size 450 --hicache-ratio 1.5 write_through kernel page_first`), chunked prefill 32768,
  `--cuda-graph-max-bs-prefill 64`, `index_share_for_mtp_iteration=false`, triton DSA backends.
- **Decode = the only change**: GPUs 4-7, `--tp-size 4 --ep-size 4`, **no** DP attention (DP1), `--dcp-size 4`,
  EAGLE 5/1/6, CUDA graph (`--cuda-graph-max-bs-decode 64`), max running 64, mem 0.85, fp8_e4m3 KV, triton DSA,
  **custom all-reduce ON** (do not pass `--disable-custom-all-reduce`), `--enable-aiter-allreduce-fusion`.
- **KV transfer**: Mooncake, same node, HIP IPC over XGMI (same env as your 22k kit:
  `MC_TE_FILTERS=<nonexistent>`, `MC_DISABLE_HIP_TRANSPORT=1`, `MC_ENABLE_DEST_DEVICE_AFFINITY=1`).
  Each prefill DP rank fans out to all 4 decode ranks (`required_dst_info_num=4`).
- **Router**: `INFERA_PD_DP_RANK_AFFINITY=false` (decode has DP1), policy kv-aware, prefill/decode overlap weight 20/2.

The exact rendered `docker run` commands of r5b (both legs + router) are in
`scripts/harness/dryrun.pdBeperf-r5b-060.yihou.txt`. **Diff your rendered commands against that file**; only node,
IP, ports and cache paths should differ (plus `HSA_ENABLE_SDMA=0`, which is v3-only).

## 3. Decode env that matters (all set on the decode leg only)

| Env | Value | Why / evidence |
|---|---|---|
| `SGLANG_DSA_DCP_KEEP_KV_B_BF16` | 1 | Keeps kv_b in bf16 so q-replication runs on all 78 MLA layers; −1.6..−1.7 ms/step at 34K (`results/RESULT.md` §5). Log line "prepared full-head Q weights for 78 MLA layers". |
| `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER` | 1 | Split DCP decode indexer (local top-k over 1/4 of pages + merge). |
| `SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON` | 1 | Triton pack + radix merge; −4.4/−6.7/−9.1 ms/step at 200K N=16/24/32, +0.5..1.5 ms at 34K (§5). |
| `SGLANG_SIMULATE_ACC_LEN` / `_METHOD` / `_TOKEN_MODE` | 3.61 / match-expected / real-draft-token | Perf scoring only (correctness waived). |
| `SGLANG_DP_USE_GATHERV` | 0 (decode), 1 (prefill) | As rendered by the harness. |
| `HSA_NO_SCRATCH_RECLAIM` | 1 (decode), 0 (prefill) | As rendered. |
| `SGLANG_OPT_USE_TOPK_V2` | false | Not read on the decode path when the split knobs are on (code read, `notes.md`). |
| `SGLANG_LOG_DECODE_SEQ_LENS` | 1 | Diagnostic log only (our branch); optional. |
| ROCm debug agent (`HSA_TOOLS_LIB`, `HSA_ENABLE_DEBUG`) | **OFF** | With it ON the debug agent aborted a prefill scheduler in 2 of 2 perf runs on v3 nodes (`rocm-dbgapi: fatal error: os_queue_packet_id ...`). On 054 (6.14) it did not crash but costs ~1 % (your kit). Use OFF for perf. |
| `HSA_ENABLE_SDMA` | 0 **only on amdgpu 6.16.x (v3) nodes** | Weight load hangs otherwise. Not needed on 6.14 nodes; check your node's driver. |

## 4. Software

- Image `infera-sglang:v0522-1020-yihou-dcp-mc4340`, ID `sha256:5b6565ef3e4e7dcca02423d509f842a74cf7416b18ed94edbf12e2be100f7376`
  (= `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020` + Mooncake #4340). Tar (23 GB, EXTERNAL):
  `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar`.
  Rebuild recipe: `REPRODUCE.md` a.1.
- sglang overlay (mounted read-only over `/sgl-workspace/sglang/python/sglang` on **both** legs):
  `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-93034eba91/python/sglang` (EXTERNAL, ready to mount).
  Source: branch `dev/yihou/glm52_dcp_pd` @ `93034eba91`; rebuild from `patches/` (`REPRODUCE.md` §0.4).
  The overlay is required: upstream main rejects DSA DCP + PD, and ROCm DCP needs #44322 plus our PD/DCP fixes
  (`patches/README.md`).
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`. AgentX: InferenceX `918524ff`, aiperf 0.12.0,
  dataset `semianalysisai/cc-traces-weka-062126`.

## 5. Two ways to run it

**Option A — port into your own harness (recommended for 135/136).** Keep your 22k e2g config and change only:
image -> the one above; mount the `sglang-93034eba91` overlay on both legs; decode leg -> section 2 args + section 3 env;
router `INFERA_PD_DP_RANK_AFFINITY=false`; debug agent off. Then diff the rendered commands against
`scripts/harness/dryrun.pdBeperf-r5b-060.yihou.txt`.

**Option B — use our harness as-is.** `REPRODUCE.md` (a), r5b block: `PD_OVERLAY=.../sglang-93034eba91/...`,
`PD_EXTRA="PREFILL_HICACHE_SIZE=450 DECODE_CUSTOM_AR=1 DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO DEBUG_AGENT=0 DECODE_ENV_LIST=SGLANG_DSA_DCP_KEEP_KV_B_BF16=1,SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1,SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1,SGLANG_LOG_DECODE_SEQ_LENS=1"`
(add `BOTH_ENV_LIST=HSA_ENABLE_SDMA=0` only on 6.16 nodes), a topology file for your node
(`scripts/harness/topology.yihou.<NK>.tsv`), then `SKIP_GATE=1 bash $PD/harness/pd_seq.yihou.sh <NK> <JOB> <TAG> 40`.
The harness writes under `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/runs/` — use a new round tag; do not touch our rounds.

## 6. Protocol (same as your 22k kit)

1. Correctness gate, sim OFF (`DECODE_SIMULATE_ACC_LEN=` empty), graph on: math + passkey 8K/34K/96K
   (`scripts/harness/pd-tools/t4/check.py`). Recorded PASS on `93034eba91` (round pdBe-r5g, 5 requests).
2. Perf stack, sim 3.61: AgentX C40, 3600 s, warm-up 10/lane. **A1 fresh -> `POST /flush_cache` decode then prefill -> A2 scored.**
3. Score = `request_metrics.throughput.per_gpu.total_tput_tps` (total / 8). The JSON field `decode_dcp_size` reads 1 even for DCP4 — check the server args instead.
4. Expect first decode bring-up ~15 min with a cold JIT cache (CUDA-graph capture); do not kill it.

## 7. Known risks and limits (read before quoting numbers)

- **Correctness under concurrency is not verified for this DCP4 PD config.** Only 5-request sim-off gates exist.
  A related layout (decode DP2 x DCP2, PD, graph) produced intermittent garbled answers under 40-concurrent sim-off load
  (r2 3/200, r5 13/240, r6 27/320; first tokens correct, then junk). The user recorded this as a known risk (10-22 ~02:05Z);
  your 09-18 garble pack-up (DPA + custom AR) shows the same symptom. Recommend one 40-concurrent sim-off check
  (`scripts/mixed/run_check40_pd.sh`) before relying on output quality.
- **Where the gap to 22k is**: at C40 the KV-transfer queue (~5 in transfer-in) is the same as your baseline's, so transfer
  is not the differentiator at C40. The gap is decode step efficiency: derived decode token rate r4 ≈ 1.11 tok/ms vs
  baseline ≈ 1.26 tok/ms. At C52 the extra load piles up in transfer-in (4.9 -> 18.2) and TTFT rises; C52 scored lower
  than C40 (18,070 vs 18,321), so stay at C40.
- **Cross-node is untested by us.** With a DPA4 prefill -> DCP4 decode, each prefill rank sends to all 4 decode ranks, so
  a rail-aligned cross-node layout needs both ends on the same rail set (`spec/design/phase6_merge.md` §2).
- Numbers from v3 (6.16, SDMA off) and 054/OAI (6.14) nodes are not directly comparable.
- `decode-only` restarts under a live prefill/router produced "no content" errors once (r3); restart the full stack.

## 8. What we would like back

Scored A2 at C40 on your node with this config, `decode_stats` (ms/step vs running), and if possible one A/B with
`SGLANG_DSA_DCP_SPLIT_*` off to separate the split gain on a 6.14 node.

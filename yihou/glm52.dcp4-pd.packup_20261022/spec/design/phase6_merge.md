# Phase 6 merge prep: fold the infera-glm52-pd-5f optimizations into the DCP4 PD config (2026-10-21)

Sources, read on 2026-10-21:
- `~/dev/git/infera.glm52.pd/CLAUDE.md` (the 10-21 "new baseline").
- On spur, read-only: `H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`, the overlays `config.yihou.{g4new,idlefix,xferfix2,offload,tokw2,E2-target,xnode,mc4340n19}.sh`, `mounts.yihou.v2-xferfix2-gtt-idlefix.tsv`, and the latest cross-node env and dryrun `g4tune_20261019/{env,dryrun}.g3x1-135x136.yihou.*`.
- Local copies are in `/var/tmp/yihou/analyst_ws/phase6/`.

Their reference number is idxf1a, A2 21,814 (same node, C56). Their replicates idxf1b and idxf1c were still being scored.

## 0. The one structural rule: do not carry their file mounts over

Their stack is the v0521 image with single-file mounts, listed in `mounts.yihou.v2-xferfix2-gtt-idlefix.tsv`, plus `offloader.yihou-attrs.py` on the prefill. Ours is the v0522-mc4340 image with our whole branch overlaid at `python/sglang`.

Mounting any of their files on top of our overlay would replace our branch's version of that file. For example, their `conn.yihou-xferfix2.py` would drop the DCP relayout in `mooncake/conn.py`, and their `prefill.yihou-nonfinal-event.py` would drop the PP wait-event fix 6ee35f3617.

Instead, every mount is mapped to branch content:

| Their mount | Our branch |
|---|---|
| kv_events_publisher noblock | folded: 36bc86517d |
| forward_mla_rocm noabsorbcfg | folded: 4d1d6a647e |
| prefill nonfinal-event | folded: e11f97495d (and PP: 6ee35f3617) |
| pool_host_common #42553 (GTT) | upstream, in our base: 6966c5d027 |
| conn xferfix2 (`YIHOU_XFER_SHARD=dest`) | **nothing to do**: `dest` is upstream sharding (session-port sum, conn.yihou-xferfix2.py:3167-3172); the mount adds only YIHOU-XFER logging |
| eager_runner idlefix (#45483) | **missing**: open upstream PR; `gh pr diff 45483` applies cleanly to HEAD (2 files, +7 lines + a test) |
| offloader yihou-attrs (prefill, OffloaderV2) | **missing**: our `utils/offloader.py:429-430` `_CpuParamOffloader.create_device_tensor` drops `is_shuffled` |

## 1. Per-optimization table

| # | Optimization (their setting) | Leg | Applies to DCP4 PD? | Code on our branch | Risk |
|---|---|---|---|---|---|
| 1 | GTT HiCache #42553: `HSA_USERPTR_FOR_PAGED_MEM=0` on prefill, `--hicache-size 260` GB/rank | P | Yes, unchanged. Our prefill is their prefill. | None (#42553 in base). Env only. Their config refuses unless `260*4*1.229` GB fits in 95% of GTT (1,477 GB). | Low. We used 48 GB for debug, so the perf run must switch to 260, and HiCache drain time grows. |
| 2 | OffloaderV2 G4K1 step 1 (`--offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-mode cpu`); g3x1 tests G3 | P | Yes, unchanged (prefill only). | **Port the is_shuffled carry**: 3 lines in `utils/offloader.py` `_CpuParamOffloader.create_device_tensor`; their patch is at offloader.yihou-attrs.py:429-436. | **High if not ported.** Per their patch comment, aiter fused_moe picks its preshuffled MXFP4 kernel by `getattr(w, "is_shuffled")`. Without the carry, the offloaded layers' experts would run the wrong kernel. I have not checked what that yields; the patch exists because it mattered to them. Medium after the port: needs one prefill correctness check (math/passkey). |
| 3 | `index_share_for_mtp_iteration=true` via `--json-model-override-args` on **both** legs | P+D | Decode: yes, if wanted. Prefill: must match the decode, because it sets the PD seed width (`disaggregation/utils.py:130-139`, `get_dsa_seed_metadata_dim`). | None. The seed sentinel from the seedless (mode N) prefill is already dropped at decode (`decode.py`, all-negative -> None), and there is our `SGLANG_DSA_DCP_PD_DROP_MTP_SEED`. | **Medium, untested under DCP.** IndexTopKShareState (`layers/attention/index_topk_share.py`) reuses step 0's indexer top-k for the later draft steps. Under DCP the draft is sharded and the indexer returns slots or positions over replicated index K. I found no DCP-specific code in the share path, and no test covers DCP + index_share. Gate: a mixed DCP4 A/B, true vs false, with sim OFF (math/passkey + real accept length), then a decodebench. |
| 4 | idlefix (#45483: `_execute_idle` also inits the active forward-context backend) | D (P inert) | Probably inert on our decode: IDLE batches come from `require_mlp_sync()` = attn-DP or gathered buffer (`utils/common.py:4145-4147`), and our decode has no DPA. Their MAFs were on a DPA4 decode. | Cherry-pick #45483 (clean apply) anyway: cheap, and the draft "idle" path is not fully excluded by my read. | Low. |
| 5 | Decode custom all-reduce ON (+ `--enable-aiter-allreduce-fusion`) | D | Yes. mixrun measured -3.5..-3.9 ms/step at 34K and N=8..32, checks PASS. | None. | **Open:** r3 (custom AR ON) had about 30% "no content" errors. r4 (fresh full stack, AR ON) decides whether that comes from custom AR or from the decode-only restart. Do not adopt until r4 is clean. |
| 6 | Decode `--tokenizer-worker-num 4 --detokenizer-worker-num 4` | D | Yes, unchanged (CPU side, independent of DCP). | None. | Low. |
| 7 | `--chunked-prefill-size 32768` (65536 under test) | P | Yes, unchanged. | None. | Low. Their 65536 result is pending. |
| 8 | Decode `--mem-fraction-static 0.95` | D | Yes. Custom AR buffers cost about 9.8% of the KV (mixrun: 8.89M vs 9.76M tokens). | None. | Low. KV usage in r2 was 0.43. |
| 9 | Decode `--max-running-requests 96 --cuda-graph-max-bs-decode 64` (per DP rank on their DPA4 decode) | D | **Not 1:1.** Our decode is one DP group, so all C requests are in one batch. Set `--cuda-graph-max-bs-decode` >= the expected running count (C56 needs >= 56), or the tail falls out of the graph. | None. | Medium if forgotten (silent eager steps). |
| 10 | `SGLANG_DP_USE_GATHERV=1`, `PD_DP_RANK_AFFINITY=1` / router `INFERA_PD_DP_RANK_AFFINITY=true` | P (+router) | GATHERV: prefill only (DPA), unchanged. Affinity: the router aligns room % prefill_dp to the prefill DP rank; I checked this earlier for a DCP decode (Infera 68f06e29). | None. | Low. |
| 11 | Decode env `NCCL_DEBUG=INFO`, `HSA_NO_SCRATCH_RECLAIM=1` | D | NCCL_DEBUG: we dropped it from r3 on. Keep it identical to whatever the comparison run used. | None. | Low. |
| 12 | Cross-node placement (`config.yihou.xnode.sh`): Mooncake RDMA, `RDMA_DEVICE` GPU i -> ionic_i, `MC_TE_FILTERS=ionic_0..3`, `MC_DISABLE_HIP_TRANSPORT=1`, `MC_ENABLE_DEST_DEVICE_AFFINITY=1` | P+D | **Blocked as configured for a DPA prefill -> DCP decode (see 2).** | Depends on the option chosen in 2. | **High.** |

## 2. Cross-node: the rail problem for DPA4 -> DCP4

- **Their premise.** `config.yihou.xnode.sh` header: "the ionic rails are physically isolated (ionic_i reaches only ionic_i)". They keep transfers rail-aligned with `PD_DP_RANK_AFFINITY=1`: prefill DP rank i talks only to decode DP rank i, so GPU i -> GPU i and ionic_i -> ionic_i. This is second-hand from their packup notes; I have not verified it.
- **Our shape breaks it.** By code (`disaggregation/common/conn.py:1368-1378`), decode rank j with attn_tp 4 and a prefill with attn_tp 1 (DPA4) targets prefill tp rank 0 of the request's DP group, and `required_dst_info_num = 4`. So each prefill DP rank i (GPU i) sends to all four decode GPUs 0-3: three of the four destinations are on another rail.
- **Options:**
  - (a) **TP4-no-DPA prefill** (the mission's named fallback). With equal attn_tp, decode rank j pairs with prefill rank j (`conn.py:1363-1367`), which is rail-aligned. The DCP1 -> DCP4 relayout plan is per token slot, so it should not depend on which prefill rank sends. I have not checked that a test covers the TP4 -> TP4 pairing. Cost: their measured prefill throughput is for DPA4, and TP4 prefill is a different and untuned point.
  - (b) **Give each prefill rank all four NICs** (`--disaggregation-ib-device` with the full list per rank) and rely on Mooncake picking a local NIC on the destination's rail. Their env already sets `MC_ENABLE_DEST_DEVICE_AFFINITY=1`. I could not find that knob in the Mooncake sources I have (`/var/tmp/yihou/analyst_ws/mooncake`, partial), so whether it does this is **unknown**.
    - Decisive check: a cross-node Mooncake transfer from prefill GPU 0 to decode GPUs 1-3, with that config.
    - A GPU-i buffer leaving through ionic_j (j != i) also crosses PCIe between switches, so its bandwidth is unknown.
  - (c) Keep the PD pair same-node (current 054 setup). Then "cross-node" in Phase 6 means only placement on 135/136 as two same-node stacks. That does not meet the mission's "cross-node PD" wording.
- **Recommendation:** ask `infera-glm52-pd-5f` whether they ever ran a non-rail-aligned transfer cross-node. Otherwise run check (b) first (minutes), and fall back to (a).

## 3. What our branch needs before Phase 6 (all small, one commit each)

1. Cherry-pick sglang #45483 (eager_runner idle active backend), clean apply.
2. Port the OffloaderV2 `is_shuffled` carry in `utils/offloader.py` (3 lines + a CPU test).
3. Nothing for #42553 / noblock / noabsorbcfg / nonfinal-event / XFER_SHARD=dest; they are already in.
4. Cross-node, option (b) only: no code. It is env + `--disaggregation-ib-device` per rank.

## 4. Run order proposal (one variable at a time, from our r2/r4 config)

1. r4 result decides custom AR (row 5).
2. Prefill → their new prefill: GTT 260 GB + G4K1 + chunk 32768 + index_share json on both legs, with index_share false at the decode first if row 3 is still untested.
   - Gate: prefill-side math/passkey, sim OFF.
3. Decode tokenizer/detok 4/4 + graph bs >= C.
4. Index_share true on the decode, after the mixed A/B.
5. Cross-node per section 2.

## 5. Open questions (not answerable from code)

- Whether OffloaderV2 without the `is_shuffled` carry gives wrong output or only a slow kernel. Moot once ported.
- Whether index_share keeps the acceptance length under DCP. Measure: mixed A/B, sim off.
- Whether Mooncake can route GPU i -> ionic_j cross-node. Measure: the transfer check in section 2(b).

## Addendum (leader, 19:50Z) — answer from infera-glm52-pd-5f on off-rail (second-hand, their measurements)
- Their xnode runs (135 prefill -> 136 decode, v0.5.21) never sent off-rail: rail isolation, per-GPU NIC map
  `--disaggregation-ib-device '{"0":"ionic_0",...,"3":"ionic_3"}'`, MC_TE_FILTERS=ionic_0..3, PD_DP_RANK_AFFINITY=1.
  MC_ENABLE_DEST_DEVICE_AFFINITY=1 was set but never exercised off-rail.
- v0.5.19 135<->138 with auto NIC choice: GPUs 4-7 failed IBV_WC_RETRY_EXC_ERR (ends picked different rails).
  Packup ~/dev/git/infera.glm52.pd/yihou/glm52-1p1d-samerail-c32-c40.packup_20260918/notes.md §1.
- Preflight today (v0521, infera.tools.preflight --mooncake), BOTH ends pinned to the same ionic_k: every GPU 0-7 reached
  the peer over that rail, byte-verified; ~38.5 GB/s local-NUMA, ~27.5-29.6 GB/s cross-NUMA.
  Results /shared_nfs/yihou_glm52/g4tune_20261019/xnode/preflight/ionic_{0..3}-*/, script .../xnode/rail_probe.yihou.sh.
- Implication for DPA4 prefill -> DCP4 decode cross-node: a GPU can use a non-matching rail if BOTH ends use the same rail,
  so pinning sender and receiver to one common NIC per (prefill rank, decode rank) pair is the thing to design/test.

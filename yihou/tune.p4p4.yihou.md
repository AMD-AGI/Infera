# GLM-5.2-MXFP4 1P1D P4D4 tuning on MI355X — from 21.0k to 25.5k tok/s/chip (handoff, 2026-10-22)

**Author:** yihou (with Claude). **Audience:** a colleague who will reproduce the current best point and stack further
optimizations on it. Everything below is first-hand from the packups listed in §7; n = 1 unless stated.
**Score** = AgentX 3600 s `request_metrics.throughput.per_gpu.total_tput_tps` (tok/s/chip over 8 GPUs). **A1** = first bench
on a fresh stack; **A2** = second bench after `POST /flush_cache` on both legs (the warm-stack protocol, see
`glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012`). All runs use EAGLE MTP 5/1/6 with
`SGLANG_SIMULATE_ACC_LEN=3.61` — **correctness is waived by construction; no number here is an accuracy result.**

## 1. Current best and how to reproduce it

| | value |
|---|---|
| best point | **g3x1: cross-node 1P1D P4D4, OffloaderV2 G3, C56 — A1 25,524 / A2 25,376** (2026-10-21) |
| reference it is compared with | `glm52.samenode-triton-idxshare.c40-full.packup_20260923`: v0.5.19 same-node, C40, fresh stack — **21,032** |
| like-for-like (both fresh stacks) | 25,524 vs 21,032 = **+21.4 %** (different node, concurrency and ISL mix — see §2) |
| reproduce | `yihou/glm52.v0521-baseline-xnode-g3g4.packup_20261022/REPRODUCE.md` §0-§2 and §4 with `scripts/envs/env.g3x1-135x136.yihou.sh` |

Placement: prefill on crsuse2-m2m-135 GPU0-3, decode on crsuse2-m2m-136 GPU0-3, KV over Mooncake RDMA with ionic rail i
<-> rail i (`--disaggregation-ib-device '{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}'`,
`MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3`, `PD_DP_RANK_AFFINITY=1`, `MC_GID_INDEX=1`). Image
`infera-sglang:v0521-yihou-1010-mc4340` (sglang 0.5.21 upstream 11b1fb21de + local patches, Mooncake faae8dd4 + PR #4340).

## 2. What changed between 21.0k and 25.5k (config read from /get_server_info and the configs)

| item | triton 0923 (21,032) | g3x1 (25,524 A1 / 25,376 A2) |
|---|---|---|
| sglang / image | 0.5.19 (a9fb1c3238), `v0519-yihou-0917-nextnfix-hicache` | 0.5.21 (11b1fb21de) + local patches, `v0521-yihou-1010-mc4340` |
| placement | same node (137), P GPU0-3 / D GPU4-7 | **cross-node**, P 135 GPU0-3 / D 136 GPU0-3 |
| KV transport | ionic RDMA per config (`MC_TE_FILTERS=ionic_0..3`), same host — inferred from config, not from logs | ionic RDMA across hosts, rail i <-> rail i |
| concurrency | C40 | **C56** |
| prefill / decode mem_fraction_static | 0.85 / 0.85 | **0.90 / 0.95** |
| decode max_running_requests | 64 | **96** |
| prefill / decode KV tokens per DP | 2,282,304 / 2,226,368 | **3,092,288 (+35 %) / 2,634,048 (+18 %)** |
| prefill HiCache | ratio 1.5 (~154 GB/rank, pageable host memory) | **260 GB/rank from GTT host pools (sglang PR #42553, prefill `HSA_USERPTR_FOR_PAGED_MEM=0`)** |
| OffloaderV2 (prefill MoE weights in host memory) | off | **group 3, num-in-group 1, prefetch step 1, mode cpu**, with the `is_shuffled` fix (PR #45484) |
| decode custom all-reduce | off (`--disable-custom-all-reduce`) | **on** |
| index_share_for_mtp_iteration | true (no fix; long runs could fault) | true **+ idlefix** (PR #45483) |
| decode tokenizer / detokenizer workers | 1 / 1 | **4 / 4** |
| unchanged | triton DSA prefill+decode, EAGLE 5/1/6, sim 3.61, TP4 DP4 DP-attention, fp8 KV, chunk 32768 (8192/DP), HiCache write_through | same |
| other local patches | NextN fusion fix, PR #37152 | xferfix2 conn.py (transfer queue + YIHOU-XFER log), kv_events non-blocking publisher, forward_mla_rocm absorb-cfg off, nonfinal event |

Workload differences: ISL mean 145,279 (triton C40) vs 130,174 (g3x1 A1); not normalised.

## 3. The measured path (each row changes several things; most were never isolated)

| step | run (node) | what changed vs the row above | score | packup |
|---|---|---|---|---|
| 0 | triton 0923 (137) | reference, C40, fresh | 21,032 | glm52.samenode-triton-idxshare.c40-full.packup_20260923 |
| 1 | e2g (102) | v0.5.21 recipe: image, HiCache 450 GB, decode mem 0.95, index_share **false**, same-node KV over **HIP IPC** | A1 19,640 / **A2 22,114** (4 warm runs mean 21,853) | glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012 |
| 2 | q1 (137) | decode max_running 96, prefill mem 0.88, C48 | 22,306 | glm52.v0521-cap-c44c48.packup_20261013 |
| 3 | pf90 (089) | prefill mem 0.90 | 23,127 | glm52.v0521-prefill-mem-hicache-policy.packup_20261014 |
| 4 | ofg4b (136) | OffloaderV2 G4 (prefill KV +15.6 %) | 23,771 (C48) | glm52.v0521-offloaderv2-highconc.packup_20261015 |
| 5 | tw90 | decode tokenizer/detokenizer 4 workers | no consistent effect (kept as a rule) | glm52.v0521-c48-followups.packup_20261015 |
| 6 | idxf1a/b/c (097, 268, 054) | new rule: PR #42553 GTT HiCache 260, index_share true + idlefix, custom AR on; C56 | A2 21,814 / 21,538 / 20,516, mean **21,289** (same-node) | glm52.v0521-idxshare-maf-fix.packup_20261021, glm52.v0521-baseline-xnode-g3g4.packup_20261022 |
| 6' | g4c1 (268) | same, C52 | A1 23,048 / A2 **23,367** | glm52.v0521-baseline-xnode-g3g4.packup_20261022 |
| 7 | g4x1, g4x2 (135->136) | **cross-node RDMA**, C56 | A1 24,976 / 24,398, A2 24,362 / 24,896 | glm52.v0521-baseline-xnode-g3g4.packup_20261022 |
| 8 | g3x1 (135->136) | **OffloaderV2 G3** | **A1 25,524 / A2 25,376** | glm52.v0521-baseline-xnode-g3g4.packup_20261022 |

## 4. What the evidence supports, and what it does not

Supported by measurement:
1. **Prefill device KV capacity is the main lever** (10-14 conclusion, `glm52.v0521-CONCLUSIONS.yihou.md` §2): the score orders
   with prefill KV tokens (2.43M 22.3k -> 2.54M 23.1k -> 3.3-3.7M 25-26k in synthetic arms). Steps 2-4 and 8 move along it
   (mem fraction 0.85 -> 0.90, then OffloaderV2 G4 -> G3 moving MoE weights to host memory).
2. **KV transfer**: at the same C56 config, same-node HIP transfer_kv p50 ~7.5 s vs cross-node RDMA ~1.1 s; per Mooncake call
   2.6 vs 20 GiB/s; score +14-17 %, TTFT p50 15 -> 4.5 s. The 0923 reference also used ionic RDMA by config and had TTFT p50
   3.6 s, so the switch to same-node HIP IPC in v0.5.21 may have been a regression for same-node — not verified from 0923 logs.
3. **Concurrency peak around C52-C56**: cross-node G4 and G3 both drop at C60/C64 (G4 23,056/22,670; G3 23,905/22,860);
   same-node C52 is +9.8 % over C56. Above the peak the transfer barely changes (1.1 -> 1.3 s); prefill queue and GPU hit move.
4. **Cross-node decode TPOT 15.7 vs 12.5 ms is load, not a slower step**: at equal busiest-DP-rank running requests the per-step
   cost matches same-node; cross-node simply decodes more requests at once (`results/xnode/tpot_analysis.20261021.md`).

Not isolated or not shown to help: decode max_running 96 and decode mem 0.95 (bundled), custom AR on, tokenizer workers 4,
index_share (no measurable per-step gain; idx=false g4n3 sits on the same curve), GTT HiCache 260 (did not shorten transfer),
the v0.5.19 -> v0.5.21 upgrade itself (fresh 19,640 vs 21,032, bundled with transport and index_share changes),
OffloaderV2 prefetch step 2 (21,342 = baseline), chunk 65536 (A1 = baseline, A2 crashed with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`).

## 5. Suggested next steps for whoever continues

Controls that would settle the attribution (one variable each, A1+A2):
- same-node C56: RDMA transport (`MC_TE_FILTERS=ionic_0..3`) vs HIP IPC — tests point 2 without a second node;
- index_share true vs false at matched load (135/136);
- G3 same-node, and a second G3 cross-node bring-up (the G3 > G4 margin is 2-4 %, about the G4 run-to-run spread);
- cross-node C52 (the same-node optimum was C52).

Open performance questions:
- Same-node HIP copies run ~6.35 GiB/s median on the GPU (idle microbench 57-63 GB/s) and ~1/3 of the call wall is outside the
  copies' GPU span (`glm52.v0521-baseline-xnode-g3g4.packup_20261022/notes.md` §7): SDMA vs blit, contention with HiCache and
  OffloaderV2 H2D/D2H not measured.
- New PD requests arrive at decode without a DSA seed (prefill does not run EAGLE), forcing an eager draft step on all DP ranks;
  cost not measured.
- Above C56 the prefill side (queue, GPU hit) limits; more prefill KV (G2, larger mem fraction) is the known direction
  (G2 KV +32.9 %, `glm52.v0521-offloaderv2.packup_20261015`).

## 6. Traps worth knowing before running anything

- Ionic rails are isolated (ionic_i only reaches ionic_i): pin the per-GPU NIC map on both legs and keep PD_DP_RANK_AFFINITY=1
  (`glm52-1p1d-samerail-c32-c40.packup_20260918/notes.md` §1). Both ends pinned to one rail k work for any GPU (probe in the
  10-22 packup).
- index_share=true without idlefix faults in decode after 20-100 min (`glm52.v0521-idxshare-maf-fix.packup_20261021`).
- OffloaderV2 needs the `is_shuffled` patch or offloaded MoE layers lose the preshuffled tag (PR #45484).
- A fresh stack's first bench scores lower (warm-stack protocol); compare A1 with A1 and A2 with A2.
- docker load on a new node can re-ID the image; engine.sh keys the aiter JIT cache by the local ID — seed it under that key.
- Keep everything on NFS or check the node disk first: 136 was down for days after a full disk (10-15..10-19).
- Same-node legs: RCCL / HIP IPC pid collisions (IPC_MODE_LEGACY / DECODE_PID_OFFSET) — see the v0521 baseline packup notes.

## 7. Packup index (all under `yihou/` in this repo)

| packup | one line |
|---|---|
| glm52-1p1d-samerail-c32-c40.packup_20260918 | v0.5.19 cross-node 135->138: rail isolation, per-GPU NIC pinning, C32/C40 |
| glm52-mtp-garbled-decode-rootcause.packup_20260918 | decode custom all-reduce corrupts the speculative path with real acceptance (why sim 3.61 is used) |
| glm52-agentx-c40-realacc.packup_20260918 | 1P1D AgentX C40 on the corrected MTP config, real acceptance |
| glm52-agentx-t2sweep-simacc.packup_20260920 | 1P1D T2 concurrency sweep under simulated acceptance |
| glm52.p8d8.agentx-sweep.packup_20260920 | P8D8 concurrency sweep C80-C256; records index_share ON faulting at ~1 h 26 m |
| glm52.samenode-p4d4.agentx-c40.packup_20260922 | first same-node P4D4 (both legs on one machine), AgentX C40 fast |
| **glm52.samenode-triton-idxshare.c40-full.packup_20260923** | **the 21,032 reference (v0.5.19, triton, index_share on, C40)** |
| dsa.topk.indexer.bug.analysis.md | v0.5.19 index_share fault analysis |
| glm52.samenode-v0521-oai102.agentx-c44.packup_20261010 | first v0.5.21 same-node bring-up on 102, C44 target config pass |
| glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012 | v0.5.21 baseline 22,114 C40, warm-stack protocol, image recipe |
| glm52.v0521-cap-c44c48.packup_20261013 | q1 recipe (max_running 96, prefill 0.88): C44/C48/C52, 22,306 |
| glm52.v0521-decode-transfer-kvcap.packup_20261013 | runshare decode in-flight KV cap (no gain) |
| glm52.v0521-prefill-kv-bottleneck.packup_20261014 | prefill KV is the lever (synthetic 25-26k) |
| glm52.v0521-runshare-highconc.packup_20261014 | runshare v2 at C56/C64: no gain |
| glm52.v0521-prefill-complete.packup_20261014 | prefill_complete policy: inconclusive |
| glm52.v0521-hicache-sensitivity.packup_20261014 | HiCache 50 vs 450 GB: -9 % |
| glm52.v0521-prefill-mem-hicache-policy.packup_20261014 | pf90 23,127; prefill memory accounting |
| glm52.v0521-CONCLUSIONS.yihou.md | 10-14 overall conclusions (what binds) |
| glm52.v0521-c48-followups.packup_20261015 | tokenizer workers 4, correctness cx90 22/23 |
| glm52.v0521-offloaderv2.packup_20261015 | OffloaderV2 feasibility, KV gains G8/G4/G3/G2, is_shuffled patch |
| glm52.v0521-offloaderv2-highconc.packup_20261015 | ofg4b G4 C48 23,771 |
| glm52.v0521-offloaderv2-burst-c56c64.packup_20261019 | G2/G3/G4 at C56/C64 on burst nodes |
| glm52.v0521-g4tune-xfer.packup_20261020 | G4 at C56: KV-transfer sharding (mix vs dest), chunk size, XGMI/Mooncake transfer microbench |
| glm52.v0521-idxshare-maf-fix.packup_20261021 | index_share decode fault root cause + idlefix |
| **glm52.v0521-baseline-xnode-g3g4.packup_20261022** | **new baseline, same-node knobs, cross-node G4/G3 + conc sweep, hipxfer shim, TPOT analysis, upstream PRs** |

Upstream PRs: https://github.com/sgl-project/sglang/pull/45483 (idlefix), https://github.com/sgl-project/sglang/pull/45484
(OffloaderV2 `is_shuffled`).

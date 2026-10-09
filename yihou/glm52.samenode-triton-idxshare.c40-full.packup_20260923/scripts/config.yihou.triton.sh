#!/usr/bin/env bash
# Purpose: GLM-5.2 same-node 1P1D P4D4 on crsuse2-m2m-276 — prefill leg and decode
#   leg on ONE node. AgentX fast run at CONC=40, HiCache OFF (Phase A deliverable).
# Usage: pass to the VENDORED bench scripts as
#     CONFIG=<ws>/scripts/config.yihou.sn.p4d4.sh
#     TOPOLOGY=<ws>/scripts/topology.yihou.tsv
#   where <ws> = bench/glm5p2_pd/results/yihou-triton-idxshare.
#   A KEY=VALUE argument on the command line still wins: launch.sh:10-14 /
#   engine.sh:24-28 export it BEFORE this file is sourced, and every override
#   below uses ":=" or an explicit "unset" test.
# Artifacts: none; this file only defines shell variables.
#
# Derived from ../../glm52.p8d8.agentx-sweep.packup_20260920/scripts/config.yihou.p4d4.sh
# (relative to the repo bench dir); see analysis/harness_samenode.yihou.md for the
# same-node-specific rationale. The inherited P4D4 recipe (MTP, mem_fraction 0.85,
# DSA substitutions, IndexShare workaround) is unchanged from that file.

# --- identity ---------------------------------------------------------------
: "${IMAGE:=infera-sglang:v0519-yihou-0917-nextnfix-hicache}"   # NextN fusion fix + sglang PR #37152; must be transferred onto 276
: "${CONTROL_NODE:=crsuse2-m2m-276}"                            # router + both legs live here; matches topology.yihou.tsv
: "${BUILDER_NODE:=crsuse2-m2m-276}"                            # unused (no build; image is transferred)
: "${CONTAINER_PREFIX:=glm52-pd-yihou-sn-p4d4}"                 # same-node prefix; carries "yihou"

# --- shape: prefill GPUs 0-3, decode GPUs 4-7, both on the one node ----------
# GPUs 0-3 sit on NUMA0/PCI domain 0002 with ionic_0-3; GPUs 4-7 on NUMA1 with
# ionic_4-7. The GPU split follows that boundary; the RAIL split does NOT — see
# the rail block below, where a measurement forced both legs onto one rail set.
# These are separate PREFILL_/DECODE_ vars, so both resolve at launch.sh source
# time (role unset).
: "${PREFILL_GPU_DEVICES:=0,1,2,3}"
: "${DECODE_GPU_DEVICES:=4,5,6,7}"
: "${PREFILL_TP:=4}"
: "${PREFILL_DP:=4}"
: "${DECODE_TP:=4}"
: "${DECODE_DP:=4}"

# --- concurrency target for the fast run ------------------------------------
# Phase-A deliverable is CONC=40. agentx_bench.sh also takes CONC as a required
# argument, which supersedes this.
: "${CONC:=40}"

# --- MTP: decode leg only (inherited from config.sh:75-78) -------------------
# DECODE_MTP=1, SPEC_STEPS=5, SPEC_TOPK=1, SPEC_DRAFT_TOKENS=6, EAGLE — left unset
# so they inherit config.sh's defaults (engine.sh:200-206 consumes them).
# DECODE_SIMULATE_ACC_LEN is deliberately NOT set here: config.sh:80 defaults it
# to 3.61 via ${VAR-3.61} (single dash), and a command-line "DECODE_SIMULATE_ACC_LEN="
# (empty but set) disables it for a real-acceptance point (engine.sh:149 -n test).

# --- --disable-custom-all-reduce on the decode leg (MANDATORY here) ----------
# Without it this P4D4/TP4 shape emits garbled text at accept len 1.25 (proven
# 2026-09-18). Unlike the cross-node reference (where it was an OPEN probe), this
# mission mandates it ON, so default it ON. Delivered via DECODE_EXTRA_ARGS
# (engine.sh:63, :227-230).
: "${DECODE_NO_CUSTOM_AR:=1}"
if [[ -n "$DECODE_NO_CUSTOM_AR" ]]; then
    : "${DECODE_EXTRA_ARGS:=--disable-custom-all-reduce}"
fi

# --- same-rail KV transfer: BOTH legs share one rail set (measured) ----------
# MEASURED, not assumed (rounds/002-rdma-loopback/): on this hardware, RDMA
# between two DIFFERENT ionic devices on the SAME host silently fails — the QPs
# connect and exchange QPN/PSN/RKey/GID, then no completion ever arrives and
# ib_write_bw exits 1 with no BW row. Loopback on the SAME device runs at
# ~40 GB/s, and the cross-host control runs at 42 GB/s with the identical
# command. Reproduced on both 276 and 137. ionic_0<-ionic_1 (same NUMA0, same
# PCI domain) fails too, so the discriminator is DIFFERENT DEVICE, not different
# socket — which is why the NUMA-local rail split that this file originally
# carried was withdrawn: it could not have moved a single KV block.
#
# So both legs take the IDENTICAL rail set ionic_0-3. Every matched rank pair
# then transfers same-device -> same-device, the only pairing that works. This is
# also maximally faithful to the proven cross-node P4D4 config, where both legs
# used ionic_0-3 on their own node.
#
# ACCEPTED COST: decode's GPUs 4-7 are NUMA1 while ionic_0-3 are NUMA0, so the
# decode KV DMA crosses the socket. Accepted to get a working run; do not
# optimise it away now.
#
# FALLBACK during bring-up, if the 4-device map misbehaves: point all four keys
# at ionic_0 on BOTH legs, i.e. '{"0":"ionic_0","1":"ionic_0","2":"ionic_0",
# "3":"ionic_0"}' with MC_TE_FILTERS=ionic_0, to force same-device transfers
# unconditionally.
#
# DEFERRED, untested: interleave the GPUs instead — prefill 0,1,4,5 / decode
# 2,3,6,7 on rails ionic_0,1,4,5 — which makes each rank's shared rail NUMA-local
# to BOTH its prefill and its decode GPU, at the cost of each leg's TP group
# straddling both sockets. One variable at a time; not attempted.
#
# The $role branch MECHANISM is retained below (engine.sh:15 sets role from $1,
# engine.sh:32 sources CONFIG afterwards — VERIFIED first-hand) because we may
# need per-role values again; the two branches deliberately resolve to the same
# value today. Keys are the LOCAL visible GPU index: under HIP_VISIBLE_DEVICES=
# 4,5,6,7 (decode), local key 0 == physical GPU 4 — it is mapped to ionic_0 on
# purpose. Explicit "unset" test, never "${VAR:=...}": a bare closing brace
# inside ${VAR:=...} truncates the JSON. The guard also lets a command-line
# RDMA_DEVICE=... still win (it is exported before this file is sourced).
: "${PD_DP_RANK_AFFINITY:=1}"
if [[ -z "${RDMA_DEVICE:-}" ]]; then
    if [[ "${role:-}" == decode ]]; then
        RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}'
    else
        RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}'
    fi
fi
if [[ -z "${MC_TE_FILTERS:-}" ]]; then
    if [[ "${role:-}" == decode ]]; then
        MC_TE_FILTERS="ionic_0,ionic_1,ionic_2,ionic_3"
    else
        MC_TE_FILTERS="ionic_0,ionic_1,ionic_2,ionic_3"
    fi
fi
# MC_ENABLE_DEST_DEVICE_AFFINITY=1 + PD_DP_RANK_AFFINITY=1 assume same-NAME rails
# on both legs. With the identical rail set above that assumption now HOLDS, and
# the rank i <-> rank i <-> ionic_i pairing they produce is exactly the one the
# measurement says is the only working one. Both left at config.sh's defaults.

# --- per-role AITER JIT cache (same-node hazard) -----------------------------
# engine.sh:91 resolves ${AITER_JIT_CACHE_ROOT:-/tmp/aiter-jit-$(id -u)}/$image_key
# and bind-mounts it. That was per-MACHINE in every prior cross-node deployment;
# with both legs on one host the two containers would share one cache directory
# and can race on first compile. Give each role its own root. Both paths carry
# "yihou" so they remain cleanable under the deletion rule.
if [[ "${role:-}" == decode ]]; then
    : "${AITER_JIT_CACHE_ROOT:=/tmp/aiter-jit-yihou-sn-decode}"
else
    : "${AITER_JIT_CACHE_ROOT:=/tmp/aiter-jit-yihou-sn-prefill}"
fi

# --- DSA backend substitutions (forced by the pinned nightly's argparse) -----
# flydsl/flydsl/aiter are REJECTED by this base image. Substitute tilelang for
# both compute backends before the source; clear the top-k backend AFTER so
# engine.sh:181-182 drops --dsa-topk-backend (config.sh, unlike config.full.sh,
# defines NO DSA_TOPK_BACKEND default, so it is already unset after the source —
# the explicit clear below only guards against a command-line value).
: "${DSA_PREFILL_BACKEND:=tilelang}"
: "${DSA_DECODE_BACKEND:=tilelang}"

# --- HiCache: OFF everywhere for Phase A -------------------------------------
# config.sh defaults PREFILL_HICACHE=0 / DECODE_HICACHE=0 already; inherited, not
# re-set here. JSON_MODEL_OVERRIDE_ARGS stays at config.sh's default
# {"index_share_for_mtp_iteration":false} (the issue.md §3.3 memory-fault
# workaround) — also inherited, not touched.

# --- inherit the fast-mode target -------------------------------------------
# config.sh (NOT config.full.sh): AGENTX_DURATION=1200, warmup 1/lane == fast
# mode, HiCache off. Absolute BENCH_DIR into the VENDORED harness.
: "${BENCH_DIR:=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-triton-idxshare/scripts/bench-harness}"
source "$BENCH_DIR/config.sh"

# "aiter" is not an accepted top-k choice here; empty makes engine.sh:181-182
# drop the flag. Cleared post-source to also override any command-line value.
DSA_TOPK_BACKEND=""

# --- same-node engine port stride (MANDATORY on one host) --------------------
# SGLang derives an INTERNAL port block from --port: server_args.py:836 offsets
# by a fixed ZMQ_TCP_PORT_DELTA, then :851-861 reserve
# port_base+0..NUM_DERIVED_PORTS-1 (6, or 6+dp_size on the rust path).
# launch.sh:rows() hands out ENGINE_PORT_BASE+index, so on one host the two legs
# got 29001 and 29002 and their derived blocks OVERLAPPED. Observed first-hand
# in rounds/006-bringup-1: prefill died with
#   zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:29236')
# in DetokenizerManager.init_ipc_channels -> sigquit -> "exited with code -9".
# 256 is far wider than any NUM_DERIVED_PORTS and keeps both blocks clear of the
# harness's own bases. Consumed by the vendored launch.sh and agentx_env.py;
# defaults to 1 there, so cross-node deployments are unaffected.
: "${ENGINE_PORT_STRIDE:=256}"

# ============================================================================
# DELTAS FOR THIS RUN (triton DSA + index_share ON + HiCache + full mode)
# Everything above is inherited from the Phase-A same-node config; only the
# four things below differ, and each is stated with its evidence.
# ============================================================================

# --- DSA backends: triton, per the user's instruction ------------------------
# Modelled on yaocheng/2p1d-sweep-triton-dsa-20260922/config.sh:87-89.
# VERIFIED first-hand against THIS image's argparse (not assumed):
#   --dsa-prefill-backend {flashmla_sparse,...,tilelang,triton,aiter,trtllm}
#   --dsa-decode-backend  {same list}
# so `triton` is accepted. `flydsl` is in neither list, which confirms the
# inherited note. Plain assignment (not ":=") so these WIN over the earlier
# tilelang defaults set above in this same file.
DSA_PREFILL_BACKEND=triton
DSA_DECODE_BACKEND=triton
# --dsa-topk-backend choices are {sgl-kernel,torch,flashinfer} — no `aiter`,
# confirming the inherited claim. Empty => engine.sh:181 drops the flag and
# sglang uses its sgl-kernel default. Same as the yaocheng kit.
DSA_TOPK_BACKEND=""

# --- index_share ON, per the user's instruction ------------------------------
# The model's own config.json carries index_share_for_mtp_iteration = True, so
# "on" is the MODEL DEFAULT and could be had by clearing the override entirely.
# Set it EXPLICITLY to true instead, so the value is visible and checkable in
# /get_server_info rather than inferred from an absent flag.
# RISK, stated not hidden: glm52.p8d8.agentx-sweep.packup_20260920 records that
# this same shape with IndexShare ON faulted at 1 h 26 m, while OFF ran 13 h 33 m
# fault-free. This run carries that risk deliberately, at the user's direction.
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":true}'

# --- prefill HiCache ON (decode stays off; engine.sh:76 rejects decode+MTP) ---
# ratio/write_through/kernel/page_first all inherit config.sh's defaults, which
# already match the 20260920 reference.
PREFILL_HICACHE=1

# --- full mode ---------------------------------------------------------------
# 3600 s profiling, warmup 10/lane — matches the cross-node reference t2f, so
# the same-node vs cross-node comparison differs in co-location alone.
AGENTX_DURATION=3600
AGENTX_WARMUP_REQUESTS_PER_LANE=10

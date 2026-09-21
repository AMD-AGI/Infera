#!/usr/bin/env bash
# Purpose: 1P1D P4D4 shape (4-card prefill on crsuse2-m2m-137, 4-card decode on
#   crsuse2-m2m-136) for the P8D8-vs-P4D4 phase, AgentX fast run at CONC=40,
#   MTP simulated acceptance ON.
# Usage: pass CONFIG=<this file> TOPOLOGY=<workspace>/scripts/topology.yihou.tsv
#   to the bench scripts. A KEY=VALUE argument on the command line still wins
#   (launch.sh:10-14 / engine.sh:24-28 export it before this file is sourced,
#   and every override below uses ":=" / an "unset" test).
# Artifacts: none; this file only defines shell variables.
#
# Identical in structure to config.yihou.p8d8.sh; only the shape (4 cards,
# TP4/DP4), the concurrency target (40), the container prefix, and the size of
# the RDMA map differ. See that file and analysis/config_design.yihou.md for the
# per-setting rationale that is common to both.

# --- identity ---------------------------------------------------------------
: "${IMAGE:=infera-sglang:v0519-yihou-0917-nextnfix-hicache}"   # transferred image; NextN fusion fix + all three #37152 markers, sglang 0.5.19.dev20260917+ga9fb1c3238 (mission.md:99-103)
: "${CONTROL_NODE:=crsuse2-m2m-137}"                            # router + prefill leg; matches topology.yihou.tsv row 0
: "${BUILDER_NODE:=crsuse2-m2m-137}"                            # unused (no build; image is transferred)
: "${CONTAINER_PREFIX:=glm52-pd-yihou-p4d4}"                    # distinct from the P8D8 prefix; carries "yihou"

# --- shape: P4D4 on GPUs 0,1,2,3 of each node -------------------------------
# GPUs 0,1,2,3 is hwprep's RECOMMENDED P4D4 quad (rdma_map.yihou.md): all four
# GPUs + ionic_0-3 sit on NUMA0 / PCI domain 0002, rails 08/07/05/06, all ACTIVE
# on both nodes — one clean single-NUMA set with zero cross-socket PCIe on the KV
# path. 137/136 have all 8 GPUs free, so the reference's 2,3,4,5 (a NUMA-straddling
# compromise forced by 135's k8s pod on GPU[1] and dead ionic_7) is not inherited.
# GPUs 4,5,6,7 (NUMA1, ionic_4-7) are an equally valid mirror quad if NUMA0 is ever
# contended; switching would also require swapping the RDMA map below to ionic_4-7.
: "${PREFILL_GPU_DEVICES:=0,1,2,3}"
: "${DECODE_GPU_DEVICES:=0,1,2,3}"
: "${PREFILL_TP:=4}"
: "${PREFILL_DP:=4}"
: "${DECODE_TP:=4}"
: "${DECODE_DP:=4}"

# --- concurrency target for the fast run ------------------------------------
# Half the P8D8 point, per the mission table: 40. Overridable for the phase-E
# sweep (40/56/72/96/128). Same note as the P8D8 file: agentx_bench also takes
# CONC as a required argument, which supersedes this.
: "${CONC:=40}"

# --- MTP: decode leg only (inherited from config.full.sh:78-81) -------------
# DECODE_MTP=1, SPEC_STEPS=5, SPEC_TOPK=1, SPEC_DRAFT_TOKENS=6, EAGLE — verified
# against engine.sh:200-206; left unset so they inherit. Simulated acceptance
# stays 3.61 via config.full.sh:82's ${VAR-3.61} (no colon) for this fast run,
# and a "DECODE_SIMULATE_ACC_LEN=" command-line arg disables it for the probe
# (empty, set-but-null -> engine.sh:149 -n test false). Deliberately NOT set
# here so both paths work; do not convert to ":=".

# --- --disable-custom-all-reduce toggle (OPEN question, per shape) -----------
# UNSETTLED for this shape too — but note the P4D4/TP4 A/B is exactly where the
# flag was PROVEN necessary (glm52-mtp-garbled-decode-rootcause.packup_20260918).
# The probe still decides it here rather than assuming, so P8D8 and P4D4 are
# settled the same way. Default unset = flag OFF; export DECODE_NO_CUSTOM_AR=1 to
# turn it on. Delivered via DECODE_EXTRA_ARGS (engine.sh:63, :227-230).
: "${DECODE_NO_CUSTOM_AR:=}"
if [[ -n "$DECODE_NO_CUSTOM_AR" ]]; then
    : "${DECODE_EXTRA_ARGS:=--disable-custom-all-reduce}"
fi

# --- same-rail KV transfer --------------------------------------------------
# Same two halves as the P8D8 file: router rank affinity + a per-GPU RDMA map.
: "${PD_DP_RANK_AFFINITY:=1}"

# CONFIRMED against hwprep's rdma_map.yihou.md: 4-entry GPU->NIC map, hwprep's
# ready-to-paste P4D4 block verbatim (GPU_n<->ionic_n, NUMA-local, all four rails
# ACTIVE on both nodes). Keys are the LOCAL visible index; under
# HIP_VISIBLE_DEVICES=0,1,2,3 key i == physical GPU i. Explicit "unset" test, not
# "${VAR:=...}": a bare closing brace inside ${VAR:=...} truncates the JSON.
if [[ -z "${RDMA_DEVICE:-}" ]]; then
    RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}'
fi
# Plain comma list (hwprep's P4D4 block), not the JSON blob config.full.sh:95
# would default it to. MC_GID_INDEX=1 is the live rail, inherited unchanged.
: "${MC_TE_FILTERS:=ionic_0,ionic_1,ionic_2,ionic_3}"

# --- DSA backend substitutions (same reason as the P8D8 file) ---------------
# config.full.sh:87-89 requests flydsl/flydsl/aiter, all REJECTED by the pinned
# nightly's argparse. Substitute tilelang for both compute backends; clear the
# top-k backend AFTER the source so sglang uses sgl-kernel. Accepted-choice list
# is SECOND-HAND from the sibling image at the same nightly — see the P8D8 file
# and analysis/config_design.yihou.md.
: "${DSA_PREFILL_BACKEND:=tilelang}"
: "${DSA_DECODE_BACKEND:=tilelang}"

# --- HiCache / HSA reclaim / TOPK_V2 inherited (see the P8D8 file) -----------
# Prefill HiCache ON (config.full.sh:57), decode OFF (config.full.sh:73; +
# engine.sh:76 hard-rejects decode HiCache+MTP). HSA_NO_SCRATCH_RECLAIM role-
# scoped 0/1 (config.full.sh:56/72, engine.sh:49/62/121).
# SGLANG_OPT_USE_TOPK_V2=false is config.full.sh:101, forwarded at engine.sh:123.

# --- inherit the full target ------------------------------------------------
: "${BENCH_DIR:=/home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd}"
source "$BENCH_DIR/config.full.sh"

# "aiter" is not an accepted top-k choice on this base; empty makes engine.sh:
# 181-182 drop --dsa-topk-backend. Cleared post-source because config.full.sh:89
# uses ${DSA_TOPK_BACKEND:-aiter} and ":-" treats empty as unset.
DSA_TOPK_BACKEND=""

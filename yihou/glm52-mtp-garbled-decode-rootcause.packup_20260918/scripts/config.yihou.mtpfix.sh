#!/usr/bin/env bash
# Purpose: R03 -- the C32 same-rail baseline with exactly TWO deliberate changes:
#   (1) IMAGE carries the GLM-5.2 NextN shared-experts-fusion fix;
#   (2) simulated MTP acceptance is OFF, so acceptance is the real measured one.
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
#
# Everything else is inherited unchanged from config.yihou.base.sh (a verbatim
# copy of the previous task's config.yihou.sh), which in turn sources the repo
# baseline config.sh. That keeps the diff against the measured C32 run minimal.

# --- change 1: the fix -------------------------------------------------------
# Thin layer over infera-sglang:v0519-yihou-0917 -- same 7 DSA patches, one extra
# class attribute. Built locally on each node; see image/Dockerfile.yihou.nextnfix.
: "${IMAGE:=infera-sglang:v0519-yihou-0917-nextnfix}"

# --- change 2: real acceptance ----------------------------------------------
# config.sh assigns this with ${DECODE_SIMULATE_ACC_LEN-3.61} -- a SINGLE dash,
# so an empty-but-SET value survives and disables the simulation. With ":-" this
# would silently fall back to 3.61. engine.sh only exports SGLANG_SIMULATE_ACC_LEN
# when the value is non-empty, so empty means "measure the real thing".
DECODE_SIMULATE_ACC_LEN=""

# --- keep this run's containers distinguishable ------------------------------
: "${CONTAINER_PREFIX:=glm52-pd-yihou-mtpfix}"

# Inherit the rest: P4D4 on devices 2,3,4,5, DPA on, decode MTP on, HiCache off,
# mem_fraction 0.85, PD_DP_RANK_AFFINITY=1, the per-GPU RDMA_DEVICE JSON map and
# MC_TE_FILTERS, and index_share_for_mtp_iteration=false.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.base.sh"

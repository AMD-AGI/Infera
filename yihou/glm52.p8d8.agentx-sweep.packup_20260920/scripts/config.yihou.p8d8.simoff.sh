#!/usr/bin/env bash
# Purpose: 1P1D P8D8 shape (8-card prefill on crsuse2-m2m-137, 8-card decode on
#   crsuse2-m2m-136) for the P8D8-vs-P4D4 phase, AgentX fast run at CONC=80,
#   MTP simulated acceptance ON.
# Usage: pass CONFIG=<this file> TOPOLOGY=<workspace>/scripts/topology.yihou.tsv
#   to the bench scripts. A KEY=VALUE argument on the command line still wins
#   (launch.sh:10-14 / engine.sh:24-28 export it before this file is sourced,
#   and every override below uses ":=" / an "unset" test).
# Artifacts: none; this file only defines shell variables.
#
# House style: source config.full.sh (the full feature target — Prefill HiCache
# on, max-running/graph BS 128, JIT grouped-topk, warmup 10/lane, 3600 s
# profiling, IndexShare workaround removed) and override ONLY what differs, each
# with a WHY. The two shape-defining blocks below are restated even where they
# equal a config.full.sh default, so the shape is legible without cross-reading
# and cannot silently change if that default ever moves.

# --- identity ---------------------------------------------------------------
: "${IMAGE:=infera-sglang:v0519-yihou-0917-nextnfix-hicache}"   # transferred image; NextN fusion fix + all three #37152 markers, sglang 0.5.19.dev20260917+ga9fb1c3238 (mission.md:99-103)
: "${CONTROL_NODE:=crsuse2-m2m-137}"                            # router + prefill leg; matches topology.yihou.tsv row 0
: "${BUILDER_NODE:=crsuse2-m2m-137}"                            # unused (no build; image is transferred) — set for completeness
: "${CONTAINER_PREFIX:=glm52-pd-yihou-simoff}"                    # every container carries "yihou" and the shape, so a stray one is attributable and safe under the deletion rule

# --- shape: P8D8, all eight GPUs per node -----------------------------------
# Restated for shape legibility/drift-proofing; equals config.full.sh:29-30,48,
# 50,64,66. 137 and 136 have all 8 GPUs free once phase 1 is torn down, so the
# P4D4 file's GPU[1]/ionic_7 avoidance does not apply here.
: "${PREFILL_GPU_DEVICES:=0,1,2,3,4,5,6,7}"
: "${DECODE_GPU_DEVICES:=0,1,2,3,4,5,6,7}"
: "${PREFILL_TP:=8}"
: "${PREFILL_DP:=8}"
: "${DECODE_TP:=8}"
: "${DECODE_DP:=8}"

# --- concurrency target for the fast run ------------------------------------
# config.full.sh:128 defaults CONC to 128; the phase-1 fast point for P8D8 is 80
# (mission table). Overridable so the phase-E sweep (80/112/144/192/256) can pass
# CONC=<n> on the command line. NOTE: agentx_bench also takes CONC as a required
# argument; this in-config value documents the phase-1 intent and is superseded
# by any CONC= on the command line.
: "${CONC:=80}"

# --- concurrency ceiling, raised for the sweep -------------------------------
# config.full.sh sets these to 128. `--max-running-requests` is a GLOBAL budget
# divided across the DP ranks (pool_configurator.py:847 sizes the request pool as
# `max_running_requests // attn_dp_size`, and base_cuda_graph_runner.py:64 clamps
# the CUDA-graph capture list to that pool), so 128 means 16 per rank on P8D8 and
# a hard ceiling of 128 in-flight requests. The sweep's 144/192/256 points would
# then be measuring the scheduler's cap rather than the shape — silently, because
# past the cap the server QUEUES instead of erroring. 256 covers the top point.
#
# These live HERE, not on the launch.sh command line, deliberately: tools/
# agentx_env.py cross-checks the LIVE engine's max_running against what CONFIG
# declares and aborts on a mismatch. Overriding only at launch time passed that
# value to the engine but not to the config the bench reads, and the bench
# correctly refused to run ("live max_running=256, config expects 128"). One
# source of truth removes the whole class of drift.
: "${PREFILL_MAX_RUNNING:=256}"
: "${DECODE_MAX_RUNNING:=256}"
: "${PREFILL_GRAPH_MAX_BS:=256}"
: "${DECODE_GRAPH_MAX_BS:=256}"

# --- warmup: INHERITED from config.full.sh at 10 per lane, deliberately ------
# This file does NOT set AGENTX_WARMUP_REQUESTS_PER_LANE. It inherits
# config.full.sh:131's value of 10, which is the reference C40's setting.
#
# Measured cost, so nobody is surprised by it: at 10/lane the CONC=80 point queues
# 884 warmup requests, issued in waves of one-per-lane (84) that refill only as a
# wave drains; the first cold-cache wave took ~11-12 min. Across the five points
# that is ~8,660 warmup requests. It is a large fraction of the sweep's wall clock.
#
# A cut to 2/lane was proposed and briefly applied on 2026-09-18 14:57 UTC, then
# REVERTED at the user's direction: keeping 10/lane holds the warmup identical to
# the reference C40 and identical across all five points, so the sweep stays a
# clean single-variable series in concurrency. Wall clock was judged the cheaper
# thing to spend.

# --- MTP: decode leg only ---------------------------------------------------
# Inherited from config.full.sh:78-81 unchanged, verified against engine.sh:
#   DECODE_MTP=1              engine.sh:200 gates the speculative block
#   DECODE_SPEC_STEPS=5       -> --speculative-num-steps        (engine.sh:203)
#   DECODE_SPEC_TOPK=1        -> --speculative-eagle-topk       (engine.sh:204)
#   DECODE_SPEC_DRAFT_TOKENS=6-> --speculative-num-draft-tokens (engine.sh:205)
#   + --speculative-algorithm EAGLE                            (engine.sh:202)
# Left unset here so they inherit; re-stating them would only invite drift.
#
# Simulated acceptance: config.full.sh:82 assigns DECODE_SIMULATE_ACC_LEN with
# ${VAR-3.61} (NO colon), so it stays 3.61 for this fast run UNLESS a caller
# passes an EMPTY value. This file deliberately does NOT set it, so:
#   * unset on the command line       -> 3.61 (simulated acceptance ON)   <- fast run
#   * "DECODE_SIMULATE_ACC_LEN=" arg  -> empty, set-but-null -> engine.sh:149
#     -n test is false -> simulation OFF                                   <- probe
# That empty-string path is what the correctness probe deployment uses. Do not
# convert this to ":=" or the empty override would be clobbered.

# --- --disable-custom-all-reduce toggle (OPEN question, per shape) -----------
# Whether TP8 needs the GLM-5.2-MTP garbled-decode fix is UNSETTLED (mission
# "Open question"; the five-round A/B that established it was P4D4/TP4 only, and
# phase 1 saw coherent TP8/DP8 output at real accept 2.57). Default is unset =
# flag OFF; the 16-request temperature=0 probe (probe_mtp.yihou.sh, simulation
# off) decides it for THIS shape. Flip it by exporting DECODE_NO_CUSTOM_AR=1.
# Delivered through DECODE_EXTRA_ARGS, which engine.sh:63 reads and engine.sh:
# 227-230 word-splits onto the decode argv (patch 0001-engine-sh-extra-args-env
# is already applied in this worktree — the handling is present in engine.sh).
: "${DECODE_NO_CUSTOM_AR:=}"
if [[ -n "$DECODE_NO_CUSTOM_AR" ]]; then
    : "${DECODE_EXTRA_ARGS:=--disable-custom-all-reduce}"
fi

# --- WORKAROUND for issue.md 3.3: decode gfx950 DSA indexer memory fault -----
# REPRODUCED FIRST-HAND on this stack 2026-09-18 16:06:58, which issue.md listed
# as "needs re-confirmation on v0.5.19". A decode DP rank died mid-run with
#     Memory access fault by GPU node-3 ... Reason: Unknown.
#     Fatal Python error: Aborted
# and the prefill leg then logged `transport retry counter exceeded` two seconds
# later -- the RDMA error was the CONSEQUENCE of the peer rank dying, not a
# fabric fault. It cut the CONC=80 profiling window from 3600 s to 2640 s.
#
# issue.md 3.3's workaround is `--no-enable-dsa-fused-indexer`, which does NOT
# exist in this image's argparse (it was a flag on the older fork branch). The
# equivalent control here is the env var, read first-hand from the image:
#   environ.py:166   SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, ...)
#   dsa_indexer_kpool.py:748,941   gate the fused path on its value
# EnvBool wants 0/1, NOT true/false (contrast INFERA_PD_DP_RANK_AFFINITY, whose
# clap parser wants the opposite).
#
# Cost, stated rather than hidden: this disables an optimisation, and the
# throughput it gives up is UNMEASURED. Our only datapoint with the fused path
# ON is the truncated CONC=80 run at 18,819 tok/s/chip over 2,640 s.
: "${DECODE_EXTRA_ENV:=SGLANG_DSA_FUSE_TOPK=0}"

# --- same-rail KV transfer --------------------------------------------------
# Two independent halves, both required (see the reference packups):
#  1) PD_DP_RANK_AFFINITY=1 makes the router hand Prefill rank i to Decode rank i
#     (config.full.sh:107 already 1; restated for intent). launch.sh:128-132
#     translates 1 -> the "true" the router's clap ArgAction::Set demands.
#  2) A per-GPU RDMA_DEVICE JSON map pins each visible GPU to ONE HCA, so rank i
#     transfers over the rail rank i's GPU owns. The shared-list form leaves the
#     HCA to Mooncake auto-discovery, which lets the two ends pick different NICs
#     — and these rails are physically isolated, so a mismatch is unreachable.
: "${PD_DP_RANK_AFFINITY:=1}"

# CONFIRMED against hwprep's scripts/rdma_map.yihou.md (re-measured first-hand
# 2026-09-18 on 137+136): GPU_n<->ionic_n is NUMA-local on both nodes, both nodes
# are topologically identical, all 8 rails ACTIVE with non-zero GID, and 137's
# ionic_n carries the same rail id as 136's ionic_n — so GPU_n<->ionic_n on both
# legs gives an automatic same-rail KV path. This 8-entry map is hwprep's
# ready-to-paste P8D8 block verbatim. Keys are the LOCAL (visible) device index;
# under HIP_VISIBLE_DEVICES=0..7 key i == physical GPU i. Assigned with an
# explicit "unset" test, NOT "${VAR:=...}": a bare closing brace inside a
# ${VAR:=...} expansion terminates it early and silently truncates the JSON.
if [[ -z "${RDMA_DEVICE:-}" ]]; then
    RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3","4":"ionic_4","5":"ionic_5","6":"ionic_6","7":"ionic_7"}'
fi
# MC_TE_FILTERS defaults to $RDMA_DEVICE in config.full.sh:95, which would hand
# the transport engine a JSON blob where it expects a plain device list. Set it
# as a comma list (hwprep's P8D8 block). MC_GID_INDEX=1 (config.full.sh:94) is
# the live rail per rdma_map.yihou.md, so it is inherited unchanged.
: "${MC_TE_FILTERS:=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"

# --- DSA backend substitutions (config.full.sh asks for rejected backends) ----
# config.full.sh:87-89 requests flydsl / flydsl / aiter. The pinned nightly's
# argparse REJECTS "flydsl" (dsa_prefill/decode_backend) and "aiter"
# (dsa_topk_backend). Fall back to tilelang for both compute backends and clear
# the top-k backend so engine.sh:181-182 omits the flag and sglang uses its own
# default (sgl-kernel). The accepted-choice list is SECOND-HAND — taken from the
# sibling image infera-sglang:v0519-yihou-0917 at the same nightly
# (glm52-1p1d-samerail-c32-c40.packup_20260918/scripts/config.yihou.full.sh:29-34,
# read first-hand there); not re-verified here because this role launches no
# container. See analysis/config_design.yihou.md.
: "${DSA_PREFILL_BACKEND:=tilelang}"
: "${DSA_DECODE_BACKEND:=tilelang}"
# DSA_TOPK_BACKEND is cleared AFTER the source, not here: config.full.sh:89 uses
# ${DSA_TOPK_BACKEND:-aiter}, and ":-" treats an empty string as unset, so
# clearing it beforehand would silently yield "aiter" again.

# --- HiCache and HSA scratch reclaim are inherited, verified ----------------
# Prefill HiCache ON (config.full.sh:57), decode HiCache OFF (config.full.sh:73)
# — decode HiCache + MTP is a hard error at engine.sh:76-79, so #37152 can only
# help the prefill leg here. HSA_NO_SCRATCH_RECLAIM is role-scoped: prefill 0
# (config.full.sh:56), decode 1 (config.full.sh:72), selected at engine.sh:49/62
# and forwarded at engine.sh:121. SGLANG_OPT_USE_TOPK_V2=false is the
# config.full.sh:101 default, forwarded to the container at engine.sh:123. All
# left unchanged.

# --- inherit the full target ------------------------------------------------
# This file lives in the view workspace, OUTSIDE the bench tree, so the
# reference's relative "../../config.full.sh" idiom does not apply. BENCH_DIR
# points at the private worktree's bench dir and is overridable, in case the
# deploy step relocates config.full.sh (it must exist at this path on whichever
# node sources this config — a deploy/sync concern, flagged in the report).
: "${BENCH_DIR:=/home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd}"
source "$BENCH_DIR/config.full.sh"

# Cleared post-source (see the note above): "aiter" is not an accepted top-k
# choice on this base, and an empty value makes engine.sh:181-182 drop the flag.
DSA_TOPK_BACKEND=""

# --- PATH A: simulation OFF, DSA workaround KEPT ------------------------------
# Purpose: falsify "the SGLANG_DSA_FUSE_TOPK=0 workaround causes the garbled
# output". Everything is identical to config.yihou.p8d8.sh EXCEPT simulated
# acceptance, which is disabled here by setting the variable to the empty string
# (config.full.sh:82 uses ${VAR-3.61}, no colon, so empty survives).
#
# Predicted by the code analysis in
# glm52.dsa-indexer-gpu-fault.packup_20260919/analysis/workaround_breaks_output.yihou.md:
#   garbling PERSISTS  -> the workaround is the cause, it is not usable here
#   output COHERENT    -> the hypothesis is wrong, simulation is implicated
DECODE_SIMULATE_ACC_LEN=""

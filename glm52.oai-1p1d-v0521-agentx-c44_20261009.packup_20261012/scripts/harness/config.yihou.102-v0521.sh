#!/usr/bin/env bash
# GLM-5.2-MXFP4 same-node 1P1D P4D4 on crsuse2-m2m-102, NEW image v0521, AgentX full C44.
#
# Pass the SAME file to engine.sh, launch.sh, agentx_bench.sh and stop.sh:
#   CONFIG=/shared_nfs/yihou_glm52/agentx44_20261009/harness/config.yihou.102-v0521.sh
# agentx_bench.sh re-sources it and refuses to run if the live deployment differs.
# KEY=VALUE arguments to those scripts are exported BEFORE this file is sourced;
# every knob below uses ":=" (or "-" where empty must survive) so they still win,
# except the few post-source assignments marked HARD.
#
# Lineage: Stage-1 r012/r013 on this node
#   (kvcap_20261008/scripts/config.yihou.base.sh -> .274-1p1d.sh -> .274-1p1d-noRDMA.sh)
# with the 256-expert corrections folded in, plus the C40-full same-node reference
# (glm52.samenode-triton-idxshare.c40-full.packup_20260923) for full-mode settings.
# Not sourced from those files: they hard-code the kvcap workspace and the v0519
# image + source mounts, both wrong here.

WS=/shared_nfs/yihou_glm52/agentx44_20261009
BENCH_DIR="$WS/harness/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"

# --- image ------------------------------------------------------------------
: "${IMAGE:=infera-sglang:v0521-yihou-1009}"
# LEADER: fill with `docker image inspect --format '{{.Id}}' $IMAGE` ON 102 after
# the build/load. engine.sh refuses to start on a mismatch (DRY_RUN only warns).
: "${EXPECTED_IMAGE_ID:=sha256:3f917e9bd052717538aafab7959bc1c7d109edc7867c2adb35e90fe552aaf503}"
SGLANG_BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008

# --- model: the real 256-expert checkpoint, unmodified, no overlay ----------
SOURCE_MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
MODEL="$SOURCE_MODEL"
SERVED_MODEL=glm5.2-mxfp4

# The Stage-1 default mount list bind-mounts five v0519-era sglang sources over the
# image (deepseek_v2.py, loader.py, topk.py, kv_events_publisher.py, yihou_expert8.py).
# Over a v0521 image that would silently downgrade those files. Mount nothing.
# c44b: ONE mount — v0521 kv_events_publisher.py with the KV-metrics send made
# non-blocking. Root cause of the c44a prefill wedge (py-spy: scheduler blocked in
# emit_kv_metrics -> sock_send on a PUSH socket nothing binds; repro test
# mounts-v0521/test_noblock.yihou.py: stock BLOCKED, patched returned).
# c44c: + forward_mla_rocm.py with _absorb_bmm_config disabled (returns None = v0519 call).
# Root cause of the c44b prefill DP3 abort (triton LLVM Sequence.h:275 assertion): the
# v0521 hand-built absorb-bmm config at M=256 aborts the compile; repro
# build/repro-absorb-bmm.yihou/sweep2.yihou.log (hand config M=256 rc 134; default OK all M).
YIHOU_MOUNTS="$WS/harness/mounts.yihou.v0521-noblock.tsv"

# --- node / identity --------------------------------------------------------
: "${YIHOU_NODE:=crsuse2-m2m-102}"
: "${YIHOU_ROUND:=c44a}"
NODE_KEY="${YIHOU_NODE##*-}"
CONTROL_NODE="$YIHOU_NODE"
BUILDER_NODE="$YIHOU_NODE"
: "${CONTAINER_PREFIX:=glm52-v0521-yihou-${NODE_KEY}-${YIHOU_ROUND}}"
TOPOLOGY="$WS/harness/topology.yihou.${NODE_KEY}.tsv"
: "${SSH_OPTS:=-o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=1000}"
DATA_NIC=ens3

# --- shape: TP4/DP4/DPA both legs, EP1 (checkpoint carries ep_size 1) --------
PREFILL_GPU_DEVICES=0,1,2,3
DECODE_GPU_DEVICES=4,5,6,7
: "${PREFILL_TP:=4}"; : "${PREFILL_EP:=1}"; : "${PREFILL_DP:=4}"; : "${PREFILL_DPA:=1}"
: "${DECODE_TP:=4}";  : "${DECODE_EP:=1}";  : "${DECODE_DP:=4}";  : "${DECODE_DPA:=1}"
: "${PREFILL_MEM_FRACTION:=0.85}"
# 0.95: capacity-only in Stage-2 (2,773,376 tokens, 11.25 GB left, v0519 image);
# 0.92 is the highest value validated under load (r013). Requested by the leader.
: "${DECODE_MEM_FRACTION:=0.95}"

# --- HiCache: prefill only, 450 GB/rank host pool ----------------------------
: "${PREFILL_HICACHE:=1}"
: "${DECODE_HICACHE:=0}"          # engine.sh rejects decode HiCache + MTP
: "${PREFILL_HICACHE_SIZE:=450}"  # agentx_env.py checks it against --hicache-size
: "${PREFILL_HICACHE_WRITE_POLICY:=write_through}"
: "${PREFILL_HICACHE_IO_BACKEND:=kernel}"
: "${PREFILL_HICACHE_MEM_LAYOUT:=page_first}"

# --- decode MTP: EAGLE 5 steps / 6 draft tokens / topk 1, simulated accept ---
: "${DECODE_MTP:=1}"
: "${DECODE_SPEC_STEPS:=5}"
: "${DECODE_SPEC_DRAFT_TOKENS:=6}"
: "${DECODE_SPEC_TOPK:=1}"
# NO colon: an explicit empty value (DECODE_SIMULATE_ACC_LEN=) disables simulation.
DECODE_SIMULATE_ACC_LEN="${DECODE_SIMULATE_ACC_LEN-3.61}"

# --- kernels ----------------------------------------------------------------
: "${DSA_PREFILL_BACKEND:=triton}"
: "${DSA_DECODE_BACKEND:=triton}"
: "${KV_CACHE_DTYPE:=fp8_e4m3}"

# --- KV transport: same-node, HIP IPC over XGMI only -------------------------
# A filter that matches no device -> mooncake "Found 0 HCAs" -> no rdma transport,
# KV goes over the HIP transport. --disaggregation-ib-device keeps real names
# because sglang validates them. NEVER set MC_DISABLE_HIP.
RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}'
MC_TE_FILTERS=yihou_no_such_rdma_device
MOONCAKE_DISABLE_HIP_DMABUF=0
MC_GID_INDEX=1
: "${PD_DP_RANK_AFFINITY:=1}"     # launch.sh / RUNBOOK translate to true|false for the router

# --- per-role caches (new image => cold) -------------------------------------
# engine.sh sets $role before sourcing this file; launch.sh/agentx_bench.sh do not.
AITER_JIT_CACHE_ROOT="$WS/cache/${NODE_KEY}-v0521/aiter/${role:-tools}"
AGENTX_CACHE_DIR="$WS/cache/${NODE_KEY}-v0521/agentx"
INFERENCEX_DIR="$WS/cache/InferenceX"

# --- per-role container env --------------------------------------------------
# MC_LOG_LEVEL=INFO is what makes "Found 0 HCAs" / "HIP transport installed" visible
# for the census. NCCL_DEBUG=INFO on decode records the same-node RCCL race.
: "${PREFILL_EXTRA_ENV:=MC_LOG_LEVEL=INFO}"
: "${DECODE_EXTRA_ENV:=MC_LOG_LEVEL=INFO NCCL_DEBUG=INFO}"

# --- AgentX full mode ---------------------------------------------------------
: "${CONC:=44}"
: "${AGENTX_DURATION:=3600}"
: "${AGENTX_WARMUP_REQUESTS_PER_LANE:=10}"
# 0.10 = what the reference full runs used (config.sh default; C40-full runtime.env).
# Stage-1's base pinned 0, which aborts the window on the first failed request.
: "${AGENTX_FAILED_REQUEST_THRESHOLD:=0.10}"

# --- ports: distinct from Stage-1 (24401/38000/39001/39257) ------------------
ETCD_PORT=24501
ETCD_PEER_PORT=24502
ROUTER_PORT=38100
ENGINE_PORT_BASE=39101            # prefill 39101, decode 39357
ENGINE_PORT_STRIDE=256            # same-node: keeps sglang's derived port blocks apart
BOOTSTRAP_PORT_BASE=38198         # prefill 38198 (decode row gets 38199, unused)
KV_EVENT_PORT_BASE=35657
SNAPSHOT_PORT_BASE=38901
INFERA_NODEPORT_RANGE=43000-45767
SAME_NODE_START_GATE=health

source "$BENCH_DIR/config.sh"

# --- HARD, after the source ---------------------------------------------------
# A JSON value assigned before the source gains an extra brace from config.sh's
# ${VAR:-{...}} expansion; empty DSA_TOPK_BACKEND makes engine.sh drop the flag.
# c44d: index_share OFF. c44c decode faulted at 09:21 (6 min into profiling): "Memory access
# fault by GPU node-9" in the MTP draft-extend DSA metadata path (fused_dsa_draft_extend_
# metadata <- eagle_draft_extend_cuda_graph_runner). Same signature as
# glm52.dsa-indexer-gpu-fault.packup_20260919; P8D8 packup: idxshare ON faulted ~1h26m, OFF
# 13h33m clean. OFF is config.sh's measured default = the stable setting. See debug R004.
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
DSA_TOPK_BACKEND=''
# Shared-experts fusion ON (no --disable-shared-experts-fusion); no --max-total-tokens.
PREFILL_EXTRA_ARGS="--hicache-size $PREFILL_HICACHE_SIZE"
DECODE_EXTRA_ARGS="--disable-custom-all-reduce"   # the garbled-MTP fix (P4D4/TP4)

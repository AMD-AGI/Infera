#!/usr/bin/env bash
# AUS 2P(TP4)1D(TP8) copy of llying/3p1d-tp4-tp8-v0521-20261012/config/config.3p1d.sh (file name kept so the
# 3P1D scripts work unchanged). Deltas from the crsuse2 file: nodes (n02-25: two prefills + control, n02-29: decode),
# paths (/perf_apps run root, /scratch node-local root, model under /perf_apps/data), the image rebuilt on AUS from the
# same base digest and recipe (build/build_images.sh), and INFERA_NODEPORT_RANGE covering the AUS NodePort range.
#
# GLM-5.2-MXFP4 3P(TP4)1D(TP8) on sglang v0.5.21 with the Mooncake cross-rank fix.
#
# Engine values follow yaocheng/2p1d-sweep-triton-dsa-20260922/config.sh (the v0.5.19
# 2P1D TP8 reference); the deltas are the ones this shape and image require:
#   - image: v0.5.21 mc4340 + Mooncake dest-local-rail / pinned HCA / IB timeout
#     (llying/cross-rank-rca-20260921/fix), see ../image/;
#   - prefill TP4/DP4 (three instances), decode TP8/DP8;
#   - router PD DP-rank affinity OFF: with prefill DP4 and decode DP8 an exact-rank
#     match would leave decode ranks 4-7 idle, so KV crosses ranks and relies on the
#     Mooncake fix (every rank sees the whole rail set);
#   - v0.5.21 runtime mounts: non-blocking kv-events PUSH, absorb-bmm config off.
#
# Pass this file as CONFIG to launch.sh, agentx_bench.sh and stop.sh (agentx_bench
# re-sources it and refuses to run if the live deployment differs).
# Required env: ROUND (tag for this bring-up; container names and run dirs).

R=/perf_apps/liyingli/bench_agentx/2p1d-aus-v0521-20261019
WS="$R"
BENCH_DIR="$R/scripts/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"
: "${ROUND:?set ROUND (new tag per bring-up)}"
[[ "$ROUND" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "bad ROUND: $ROUND" >&2; return 2 2>/dev/null || exit 2; }

# --- nodes -------------------------------------------------------------------
TOPOLOGY="$R/config/topology.2p1d.aus.tsv"
CONTROL_NODE=smci355-ccs-aus-n01-29   # etcd + router + AgentX client
BUILDER_NODE=smci355-ccs-aus-n01-29
# agentx_bench.sh names the client's node-local paths after YIHOU_NODE/YIHOU_ROUND.
YIHOU_NODE="$CONTROL_NODE"
YIHOU_ROUND="$ROUND"
CONTAINER_PREFIX="glm52-llying-2p1d-$ROUND"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=1000"
# Job 34439: its extern step ended 31 min in and pam_slurm_adopt has refused ssh to its nodes since; "ssh" is
# tools/sshshim/ssh (one srun step per command). Nothing may stay in the background on a node, so the server-log
# followers run on the login host (FOLLOW_ON_NODE=0, scripts/follow_supervisor.sh), as do the queue driver, sampler,
# hang watcher and patrol.
SHIM_JOB=34439
case ":$PATH:" in *":$R/tools/sshshim:"*) ;; *) PATH="$R/tools/sshshim:$PATH" ;; esac
FOLLOW_ON_NODE=0

# Node-local writable root (container tmp, triton/hf/xdg/AITER caches, client output).
# $WS (NFS home) is mounted read-only into containers.
NODE_RW_ROOT=/scratch/liyingli_2p1d
RUNS="$R/runtime/runs/$ROUND"

# --- image -------------------------------------------------------------------
IMAGE=infera-sglang:v0521-llying-destpin-aus1019
EXPECTED_IMAGE_ID=sha256:f38645fb8d1b88a22fdff70215aafcc716b46a65bb7df864fe9a94236579dd7e
SGLANG_BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008
SOURCE_MODEL=/perf_apps/data/models/GLM-5.2-MXFP4
MODEL="$SOURCE_MODEL"
SERVED_MODEL=glm5.2-mxfp4
YIHOU_MOUNTS="$R/config/mounts.v0521.tsv"

# --- shape: 2 x prefill TP4/DP4 (one node) + 1 x decode TP8/DP8 -----------------
PREFILL_GPU_DEVICES=0,1,2,3
GPUS_prefill_0=0,1,2,3
GPUS_prefill_1=4,5,6,7
DECODE_GPU_DEVICES=0,1,2,3,4,5,6,7
PREFILL_TP=4 PREFILL_DP=4 PREFILL_DPA=1 PREFILL_EP=1
DECODE_TP=8 DECODE_DP=8 DECODE_DPA=1 DECODE_EP=1

# --- engine values (yaocheng 2P1D triton kit) ---------------------------------
PREFILL_CHUNK_SIZE=32768 PREFILL_MAX_RUNNING=256 PREFILL_GRAPH_MAX_BS=256
PREFILL_MEM_FRACTION=0.85 PREFILL_HSA_NO_SCRATCH_RECLAIM=0
PREFILL_HICACHE=1 PREFILL_HICACHE_RATIO=1.5 PREFILL_HICACHE_SIZE=0
PREFILL_HICACHE_WRITE_POLICY=write_through PREFILL_HICACHE_IO_BACKEND=kernel
PREFILL_HICACHE_MEM_LAYOUT=page_first
DECODE_CHUNK_SIZE=32768 DECODE_MAX_RUNNING=256 DECODE_GRAPH_MAX_BS=256
DECODE_MEM_FRACTION=0.85 DECODE_HSA_NO_SCRATCH_RECLAIM=1
DECODE_HICACHE=0 DECODE_HICACHE_RATIO=1.5
DECODE_HICACHE_WRITE_POLICY=write_through DECODE_HICACHE_IO_BACKEND=kernel
DECODE_HICACHE_MEM_LAYOUT=page_first
DECODE_MTP=1 DECODE_SPEC_STEPS=5 DECODE_SPEC_DRAFT_TOKENS=6 DECODE_SPEC_TOPK=1
# NO colon: an explicit empty value (DECODE_SIMULATE_ACC_LEN=) selects real acceptance.
DECODE_SIMULATE_ACC_LEN="${DECODE_SIMULATE_ACC_LEN-3.61}"
DSA_PREFILL_BACKEND=triton DSA_DECODE_BACKEND=triton
KV_CACHE_DTYPE=fp8_e4m3
SGLANG_OPT_USE_TOPK_V2=false
# yaocheng's v0.5.19 kit sets 1; on v0.5.21 r1 failed the known-answer gate (prefill
# first-token distribution degraded) with 1. 0 is config.sh's default and yihou's
# validated v0.5.21 value; bench/glm5p2_pd/issue.md 2.3 keeps it off until validated.
SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK:-0}"
AITER_ALLREDUCE_FUSION=1
KV_P2P_TRANSFER=mooncake
ENABLE_KV_AWARE=1
KV_PREFILL_OVERLAP_WEIGHT=20.0 KV_DECODE_OVERLAP_WEIGHT=2.0
PD_DP_RANK_AFFINITY=0
SGLANG_ENABLE_FAILED_SESSION_PROBE=1 SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=30
SGLANG_TIMEOUT_KEEP_ALIVE=900
READY_TIMEOUT=3600 ROUTER_READY_TIMEOUT=300 HEALTH_INTERVAL=10

# --- KV transport: cross-node RDMA over every rail, cross-rank allowed ---------
# Every rank gets the whole rail list; the image sets MC_ENABLE_DEST_LOCAL_RAIL=1 and
# MC_IB_TIMEOUT=18, and dest-local-rail needs MC_ENABLE_DEST_DEVICE_AFFINITY=1.
RDMA_DEVICE=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
MC_TE_FILTERS="$RDMA_DEVICE"
MC_GID_INDEX=1
MC_ENABLE_DEST_DEVICE_AFFINITY=1
MC_DISABLE_HIP_TRANSPORT=1
# AUS: 1, as in every earlier AUS run (b4-align README: the clusters' NIC drivers differ; crsuse2 uses 0). With 0 the
# first b1 bring-up segfaulted three prefill-0 schedulers in c10::hip::HIPStream::stream() right after memory
# registration, and a fresh process could no longer use a stream on the surviving rank's GPU.
MOONCAKE_DISABLE_HIP_DMABUF=1
RDMAV_FORK_SAFE=1
NCCL_IB_DISABLE=1
HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so
HOST_RDMA_MOUNT=/host-libionic/libionic.so

# --- same-node isolation -------------------------------------------------------
# Containers with private PID namespaces share the host netns; ROCr binds the
# abstract socket "\0hsa<pid>", so equal in-container rank PIDs on one node collide
# (RCCL init dies). Disjoint offsets per instance; the two n02-25 prefills differ.
PID_OFFSET_prefill_0=30000
PID_OFFSET_prefill_1=60000
PID_OFFSET_decode_0=30000
ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0
# Container TMPDIR is a tmpfs; 8 TP8 decode ranks compiling triton/LLVM at once overflowed
# 8g ("LLVM ERROR: IO failure on output stream: No space left on device", r1 18:33).
ENGINE_TMPFS_SIZE=32g
SAME_NODE_START_GATE=health
AITER_JIT_CACHE_ROOT="$NODE_RW_ROOT/cache/aiter/${instance:-tools}"

# --- ports (all fixed ports < 30000) --------------------------------------------
ETCD_PORT=24501 ETCD_PEER_PORT=24502
ROUTER_PORT=28100
ENGINE_PORT_BASE=26101 ENGINE_PORT_STRIDE=256   # 26101 26357 26613 26869
BOOTSTRAP_PORT_BASE=28198
KV_EVENT_PORT_BASE=25657 KV_EVENT_PORT_STRIDE=8 # prefill kv ports = base + 8*index + dp rank
SNAPSHOT_PORT_BASE=28901
# Infera's free-port picker skips this range: our fixed ports above plus the AUS NodePort range.
INFERA_NODEPORT_RANGE=24500-32767

# --- per-role container env ----------------------------------------------------
PREFILL_EXTRA_ENV="MC_LOG_LEVEL=INFO"
DECODE_EXTRA_ENV="MC_LOG_LEVEL=INFO"
PREFILL_EXTRA_ARGS=""
DECODE_EXTRA_ARGS=""

# --- per-round deltas -------------------------------------------------------------
# config/round.<ROUND>.sh, when present, holds the only differences of that round (e.g.
# r3: the decode KV allocation policy). ROUND reaches every script and the remote
# engine.sh, so a round file applies everywhere.
[[ -r "$R/config/round.$ROUND.sh" ]] && source "$R/config/round.$ROUND.sh"
# v0.5.21 --disaggregation-decode-allocation-policy {early,prefill_complete}; must match
# on both roles. early = sglang default.
: "${DECODE_ALLOC_POLICY:=early}"
case "$DECODE_ALLOC_POLICY" in early|prefill_complete) ;; *)
    echo "DECODE_ALLOC_POLICY must be early|prefill_complete" >&2; return 2 2>/dev/null || exit 2 ;; esac
if [[ "$DECODE_ALLOC_POLICY" != early ]]; then
    PREFILL_EXTRA_ARGS+="${PREFILL_EXTRA_ARGS:+ }--disaggregation-decode-allocation-policy $DECODE_ALLOC_POLICY"
    DECODE_EXTRA_ARGS+="${DECODE_EXTRA_ARGS:+ }--disaggregation-decode-allocation-policy $DECODE_ALLOC_POLICY"
fi

# --- AgentX full mode -----------------------------------------------------------
: "${CONC:=144}"
AGENTX_DURATION="${AGENTX_DURATION:-3600}"   # warmup_round.sh passes a shorter window; tune_post.sh analyses that window
AGENTX_WARMUP_REQUESTS_PER_LANE=10
AGENTX_FAILED_REQUEST_THRESHOLD=0.10
INFERENCEX_REPOSITORY=https://github.com/SemiAnalysisAI/InferenceX.git
INFERENCEX_REF=918524ff94045b3f091115f1051c22a8588edf2b
INFERENCEX_DIR=/perf_apps/liyingli/bench_agentx/glm52-c80-20260922/scripts/bench-harness/.cache/InferenceX
AGENTX_CACHE_DIR="$R/runtime/cache/agentx"

source "$BENCH_DIR/config.sh"

# --- HARD, after the source (config.sh's ${VAR:-{...}} would add a brace) --------
JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'
DSA_TOPK_BACKEND=''

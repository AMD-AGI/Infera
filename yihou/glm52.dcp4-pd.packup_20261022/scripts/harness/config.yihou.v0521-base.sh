#!/usr/bin/env bash
# GLM-5.2-MXFP4 same-node 1P1D P4D4, image v0521, NODE-AGNOSTIC base (v0521dbg, 2026-10-10).
#
# Do not pass this file as CONFIG directly; pass an overlay that sources it:
#   config.yihou.E1-relaxed.sh   config.yihou.E2-target.sh
# Pass the SAME overlay to engine.sh, agentx_bench.sh, stop.sh and run.v2.yihou.sh.
# agentx_bench.sh re-sources it and refuses to run if the live deployment differs.
#
# Required env (no defaults; the node is not known in advance):
#   YIHOU_NODE   spur node name, e.g. crsuse2-m2m-102
#   YIHOU_IP     its data IP (same IP for both legs)
#   YIHOU_JOB    spur job id holding the node (feeds SPUR_SSH_JOBS for the ssh shim);
#                set but EMPTY (YIHOU_JOB=) = plain ssh, the shim is removed from PATH
#   YIHOU_ROUND  per-bring-up tag (container names, runs/<round>)
# Optional: YIHOU_NIC (default ens3; empty = engine.sh resolves it from YIHOU_IP on the node).
#
# Every knob uses ":=" so KEY=VALUE arguments (exported before sourcing) and the
# overlays (which assign before sourcing this file) win. The few post-source
# assignments marked HARD are derived from knobs; change the knob, not the result.
#
# Lineage: phase-1 config.yihou.102-v0521.sh (agentx44_20261009/harness, c44e state),
# with the node hard-coding replaced by YIHOU_* and the experiment deltas turned into knobs.

WS=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd
BENCH_DIR="$WS/harness/bench-harness"
REMOTE_BENCH_DIR="$BENCH_DIR"

# --- node / identity (required) ----------------------------------------------
: "${YIHOU_NODE:?set YIHOU_NODE, e.g. crsuse2-m2m-102}"
: "${YIHOU_IP:?set YIHOU_IP (data IP of YIHOU_NODE)}"
# Unset = error (forces a decision); empty = plain ssh (e.g. nodes with normal sshd).
: "${YIHOU_JOB?set YIHOU_JOB (spur job id holding YIHOU_NODE; empty = plain ssh)}"
: "${YIHOU_ROUND:?set YIHOU_ROUND (new tag per bring-up)}"
[[ "$YIHOU_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "bad YIHOU_IP: $YIHOU_IP" >&2; return 2 2>/dev/null || exit 2; }
NODE_KEY="${YIHOU_NODE##*-}"
CONTROL_NODE="$YIHOU_NODE"
BUILDER_NODE="$YIHOU_NODE"
_yihou_shim=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin
if [[ -n "$YIHOU_JOB" ]]; then
    : "${SPUR_SSH_JOBS:=$YIHOU_NODE:$YIHOU_JOB}"
    # The spur ssh shim is only needed where ssh is called (the login host). engine.sh
    # runs on the node and never ssh-es, so do not touch PATH there.
    if [[ "$(hostname -s)" != "$YIHOU_NODE" && ":$PATH:" != *":$_yihou_shim:"* ]]; then
        PATH="$_yihou_shim:$PATH"
    fi
else
    # Plain ssh: make sure an inherited shim cannot intercept it.
    PATH=":$PATH:"; PATH="${PATH//:$_yihou_shim:/:}"; PATH="${PATH#:}"; PATH="${PATH%:}"
fi
unset _yihou_shim
: "${CONTAINER_PREFIX:=glm52-v0521-yihou-${NODE_KEY}-${YIHOU_ROUND}}"
: "${SSH_OPTS:=-o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=1000}"
DATA_NIC="${YIHOU_NIC-ens3}"

# Topology: two rows, prefill first, same node and IP. Generated, deterministic;
# rewritten (atomically) only when the content differs.
TOPOLOGY="$WS/harness/topology.yihou.${NODE_KEY}.tsv"
_yihou_topo="$(printf 'role\tnode\tdata_ip\nprefill\t%s\t%s\ndecode\t%s\t%s\n' \
    "$YIHOU_NODE" "$YIHOU_IP" "$YIHOU_NODE" "$YIHOU_IP")"
if [[ "$(cat "$TOPOLOGY" 2>/dev/null)" != "$_yihou_topo" ]]; then
    printf '%s\n' "$_yihou_topo" > "$TOPOLOGY.tmp.$$" && mv -f "$TOPOLOGY.tmp.$$" "$TOPOLOGY"
fi
unset _yihou_topo

# --- image ------------------------------------------------------------------
: "${IMAGE:=infera-sglang:v0521-yihou-1009}"
# Fill per node after the build: `docker image inspect --format '{{.Id}}' $IMAGE`.
# engine.sh refuses to start on a mismatch (DRY_RUN only warns). The phase-1 value
# (102) was sha256:3f917e9b...; a rebuild on a new node gets a new ID.
: "${EXPECTED_IMAGE_ID:=}"
SGLANG_BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008

# --- model: the real 256-expert checkpoint, unmodified, no overlay ----------
SOURCE_MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
MODEL="$SOURCE_MODEL"
SERVED_MODEL=glm5.2-mxfp4

# The two phase-1 fix mounts (non-blocking kv_events_publisher; forward_mla_rocm with
# the v0521 absorb-bmm config disabled). Copies in this WS, sha256-identical to phase 1.
YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2.tsv"

# --- shape: TP4/DP4/DPA both legs, EP1 ----------------------------------------
PREFILL_GPU_DEVICES=0,1,2,3
DECODE_GPU_DEVICES=4,5,6,7
: "${PREFILL_TP:=4}"; : "${PREFILL_EP:=1}"; : "${PREFILL_DP:=4}"; : "${PREFILL_DPA:=1}"
: "${DECODE_TP:=4}";  : "${DECODE_EP:=1}";  : "${DECODE_DP:=4}";  : "${DECODE_DPA:=1}"
: "${PREFILL_MEM_FRACTION:=0.85}"
: "${DECODE_MEM_FRACTION:=0.95}"

# --- HiCache: prefill only ------------------------------------------------------
: "${PREFILL_HICACHE:=1}"
: "${DECODE_HICACHE:=0}"          # engine.sh rejects decode HiCache + MTP
# engine.sh always emits --hicache-ratio $PREFILL_HICACHE_RATIO. --hicache-size is
# emitted (HARD block below) only when PREFILL_HICACHE_SIZE > 0; in v0.5.21 a size > 0
# overrides the ratio (mem_cache/pool_host/base.py:211-216). 0 = ratio sizing.
: "${PREFILL_HICACHE_RATIO:=1.5}"
: "${PREFILL_HICACHE_SIZE:=450}"  # agentx_env.py / audit compare this with the live value
: "${PREFILL_HICACHE_WRITE_POLICY:=write_through}"
: "${PREFILL_HICACHE_IO_BACKEND:=kernel}"
: "${PREFILL_HICACHE_MEM_LAYOUT:=page_first}"
# Per-call cudaHostRegister limit (srt/environ.py:770, default 256). Empty = not set
# (sglang default). Emitted into the PREFILL container env only.
: "${HICACHE_HOST_REGISTER_CHUNK_GB:=}"

# --- decode MTP: EAGLE 5 steps / 6 draft tokens / topk 1, simulated accept ---
: "${DECODE_MTP:=1}"
: "${DECODE_SPEC_STEPS:=5}"
: "${DECODE_SPEC_DRAFT_TOKENS:=6}"
: "${DECODE_SPEC_TOPK:=1}"
# NO colon: an explicit empty value (DECODE_SIMULATE_ACC_LEN=) disables simulation.
DECODE_SIMULATE_ACC_LEN="${DECODE_SIMULATE_ACC_LEN-3.61}"
# true|false -> JSON_MODEL_OVERRIDE_ARGS (HARD block). false = the stable setting.
: "${INDEX_SHARE_FOR_MTP:=false}"

# --- kernels ----------------------------------------------------------------
: "${DSA_PREFILL_BACKEND:=triton}"
: "${DSA_DECODE_BACKEND:=triton}"
: "${KV_CACHE_DTYPE:=fp8_e4m3}"
# Two DISTINCT gfx950 indexer switches (both legs), each default|off:
#  DSA_FUSED_INDEXER=off      -> engine arg --no-enable-dsa-fused-indexer: the
#     four-kernel fused decode indexer (srt/arg_groups/fields/exec_.py:340-351,
#     gate layers/attention/dsa/utils.py:77-79). default = flag absent (None = auto ON).
#  AITER_FUSED_FP8_INDEXER_WRITER=off -> env SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1:
#     the fused fp8 index-K cache WRITER (srt/environ.py:1720, dsa_indexer.py:304-310).
: "${DSA_FUSED_INDEXER:=default}"
: "${AITER_FUSED_FP8_INDEXER_WRITER:=default}"

# --- same-node HIP IPC namespace (engine containers, BOTH legs, never one leg) ---
# Root cause of the same-node "RCCL race" (repro, first-hand): ROCr default IPC binds
# "\0hsa<pid>" in the host netns with the container's namespace pid -> prefill/decode ranks
# with equal ns pids collide. ENGINE_PID_HOST=1 adds `docker run --pid host` (pids unique).
# IPC_MODE_LEGACY=1 is the alternative: HSA_ENABLE_IPC_MODE_LEGACY=1 on BOTH legs (one leg
# only fails: prefill cannot open decode's legacy handles, hipIpcOpenMemHandle 201/17 on 136).
# run.v2 retry-decode refuses if the live prefill differs from the config on either knob.
# Defaults = the end-to-end-validated fix (sanity-v0519-legacy on 136: SANITY PASS, decode first
# try with prefill up, KV transfer 0 fails). --pid host stays an untested arm.
: "${ENGINE_PID_HOST:=0}"
: "${IPC_MODE_LEGACY:=1}"
# R022: decode-only pid offset (engine.sh wraps the decode command); 0 = off.
: "${DECODE_PID_OFFSET:=0}"
# spur-authz on spur nodes denies `docker run --pid=host` (e2b3 on 102, 07:23): refuse early.
if [[ "$ENGINE_PID_HOST" == 1 && -n "$YIHOU_JOB" ]]; then
    echo "ENGINE_PID_HOST=1 is not allowed on a spur node (YIHOU_JOB=$YIHOU_JOB): spur-authz denies --pid=host" >&2
    return 2 2>/dev/null || exit 2
fi
for _k in ENGINE_PID_HOST IPC_MODE_LEGACY; do
    [[ "${!_k}" == 0 || "${!_k}" == 1 ]] || { echo "$_k must be 0|1" >&2; return 2 2>/dev/null || exit 2; }
done; unset _k

# --- KV transport: same-node, HIP IPC over XGMI only -------------------------
# A filter that matches no device -> mooncake "Found 0 HCAs" -> no rdma transport,
# KV goes over the HIP transport. --disaggregation-ib-device keeps real names
# because sglang validates them. NEVER set MC_DISABLE_HIP.
RDMA_DEVICE='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}'
MC_TE_FILTERS=yihou_no_such_rdma_device
MOONCAKE_DISABLE_HIP_DMABUF=0
MC_GID_INDEX=1
: "${PD_DP_RANK_AFFINITY:=1}"     # the driver translates to true|false for the router

# --- node-local writable root (NFS stays the read side: model, harness, mounts) --
# Nodes reached by plain ssh (136/138) mount /shared_nfs READ-ONLY, so everything the node
# writes goes under NODE_RW_ROOT: SERVER_LOG, AITER/triton/hf/xdg caches, tmpfs mount
# points, the router's caches, the AgentX client's output/caches. run.v2 copies server logs
# back for census and everything (minus caches) back at bench/stop. Empty = all on $WS (NFS rw,
# spur-job nodes). NO colon: an explicit NODE_RW_ROOT= selects the NFS behaviour.
if [[ -n "$YIHOU_JOB" ]]; then NODE_RW_ROOT="${NODE_RW_ROOT-}"
else NODE_RW_ROOT="${NODE_RW_ROOT-/mnt/m2m_nobackup/yihou_v0521dbg}"; fi
[[ -z "$NODE_RW_ROOT" || ( "$NODE_RW_ROOT" == /* && "$NODE_RW_ROOT" == *yihou* ) ]] ||
    { echo "NODE_RW_ROOT must be absolute and contain 'yihou'" >&2; return 2 2>/dev/null || exit 2; }
NODE_RW="${NODE_RW_ROOT:-$WS}"      # node-side root actually used
NODE_RD="$NODE_RW/runs/$YIHOU_ROUND"  # node-side run dir (== $WS/runs/<round> when empty)

# --- per-role caches, scoped by node key --------------------------------------
# engine.sh sets $role before sourcing this file; launch.sh/agentx_bench.sh do not.
# engine.sh itself puts triton/hf/xdg under $NODE_RW/cache/$node_key/<kind>/<role>.
AITER_JIT_CACHE_ROOT="$NODE_RW/cache/${NODE_KEY}/aiter/${role:-tools}"
AGENTX_CACHE_DIR="$WS/cache/${NODE_KEY}/agentx"   # login-side; agentx_bench maps it for the node
# $WS/cache/InferenceX is a SYMLINK into the phase-1 WS. agentx_bench keeps the logical path
# (under $REPO=$WS, so it adds no extra mount) and the client container cannot follow the link
# (e1c bench 07:52: benchmark_lib.sh "No such file"). Use the resolved path -> it gets its own :ro mount.
INFERENCEX_DIR="$(readlink -f "$WS/cache/InferenceX")"

# --- per-role container env (user part; the HARD block appends the knob part) ----
# MC_LOG_LEVEL=INFO makes "Found 0 HCAs" / "HIP transport installed" visible for the
# census. NCCL_DEBUG=INFO on decode records the RCCL race AND the eager-fallback signal.
: "${PREFILL_EXTRA_ENV_BASE:=MC_LOG_LEVEL=INFO}"
: "${DECODE_EXTRA_ENV_BASE:=MC_LOG_LEVEL=INFO NCCL_DEBUG=INFO}"

# --- ROCm debug agent, lean (both legs; engine.sh adds the env with dedicated -e) ---
# HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1
# ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals  (no --output: stderr -> server log;
# no --precise-memory / --all). census greps the server logs for the report.
: "${DEBUG_AGENT:=1}"
DEBUG_AGENT_OPTIONS=--disable-linux-signals

# --- AgentX full mode ---------------------------------------------------------
: "${CONC:=44}"
: "${AGENTX_DURATION:=3600}"
: "${AGENTX_WARMUP_REQUESTS_PER_LANE:=10}"
: "${AGENTX_FAILED_REQUEST_THRESHOLD:=0.10}"

# --- ports: every FIXED port below 30000 (R015) --------------------------------
# Node ip_local_port_range is 32768-60999 (30000-32767 is the k8s NodePort window). A fixed
# port inside the ephemeral range can be taken at random by an already-running process:
# e2b on 102, prefill's scheduler held 10.245.147.130:39595 (an OS-ephemeral socket), which was
# inside decode's sglang-derived block (port+233 .. port+243 with DP4 on the rust path), so
# decode died with "scheduler_input_port ... not available". run.v2 `up` preflights that every
# fixed port and derived block is < 30000 and not listening on the node.
: "${ETCD_PORT:=24501}"
: "${ETCD_PEER_PORT:=24502}"
: "${ROUTER_PORT:=28100}"
: "${ENGINE_PORT_BASE:=29101}"            # prefill 29101 (block 29334-29344), decode 29357 (29590-29600)
: "${ENGINE_PORT_STRIDE:=256}"            # same-node: keeps sglang's derived port blocks apart
: "${BOOTSTRAP_PORT_BASE:=28198}"         # prefill 28198 (decode row gets 28199, unused)
: "${KV_EVENT_PORT_BASE:=25657}"          # prefill publishes on 25657 + attn_dp_rank (25657-25660)
: "${SNAPSHOT_PORT_BASE:=28901}"
# INFERA_NODEPORT_RANGE is NOT a bind range: infera/common/net.py:43-64 uses it as the window its
# random port pickers must AVOID (default 30000-32767, k8s NodePort). Kept as in phase 1.
INFERA_NODEPORT_RANGE=43000-45767
SAME_NODE_START_GATE=health

source "$BENCH_DIR/config.sh"

# --- HARD, after the source: derived from the knobs above ----------------------
# A JSON value assigned before the source gains an extra brace from config.sh's
# ${VAR:-{...}} expansion; empty DSA_TOPK_BACKEND makes engine.sh drop the flag.
case "$INDEX_SHARE_FOR_MTP" in true|false) ;; *) echo "INDEX_SHARE_FOR_MTP must be true|false" >&2; return 2 2>/dev/null || exit 2;; esac
JSON_MODEL_OVERRIDE_ARGS="{\"index_share_for_mtp_iteration\":$INDEX_SHARE_FOR_MTP}"
DSA_TOPK_BACKEND=''
for _k in DSA_FUSED_INDEXER AITER_FUSED_FP8_INDEXER_WRITER; do
    case "${!_k}" in default|off) ;; *) echo "$_k must be default|off" >&2; return 2 2>/dev/null || exit 2;; esac
done; unset _k
[[ "$PREFILL_HICACHE_SIZE" =~ ^[0-9]+$ ]] || { echo "PREFILL_HICACHE_SIZE must be an integer GB (0 = ratio)" >&2; return 2 2>/dev/null || exit 2; }

# Shared-experts fusion ON (no --disable-shared-experts-fusion); no --max-total-tokens.
PREFILL_EXTRA_ARGS=""
DECODE_EXTRA_ARGS="--disable-custom-all-reduce"   # the garbled-MTP fix (P4D4/TP4)
(( PREFILL_HICACHE_SIZE > 0 )) && PREFILL_EXTRA_ARGS+=" --hicache-size $PREFILL_HICACHE_SIZE"
if [[ "$DSA_FUSED_INDEXER" == off ]]; then
    PREFILL_EXTRA_ARGS+=" --no-enable-dsa-fused-indexer"
    DECODE_EXTRA_ARGS+=" --no-enable-dsa-fused-indexer"
fi
PREFILL_EXTRA_ARGS="${PREFILL_EXTRA_ARGS# }${PREFILL_EXTRA_ARGS_ADD:+ $PREFILL_EXTRA_ARGS_ADD}"
DECODE_EXTRA_ARGS="$DECODE_EXTRA_ARGS${DECODE_EXTRA_ARGS_ADD:+ $DECODE_EXTRA_ARGS_ADD}"

PREFILL_EXTRA_ENV="$PREFILL_EXTRA_ENV_BASE"
DECODE_EXTRA_ENV="$DECODE_EXTRA_ENV_BASE"
[[ -n "$HICACHE_HOST_REGISTER_CHUNK_GB" ]] &&
    PREFILL_EXTRA_ENV+=" SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=$HICACHE_HOST_REGISTER_CHUNK_GB"
if [[ "$AITER_FUSED_FP8_INDEXER_WRITER" == off ]]; then
    PREFILL_EXTRA_ENV+=" SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1"
    DECODE_EXTRA_ENV+=" SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1"
fi

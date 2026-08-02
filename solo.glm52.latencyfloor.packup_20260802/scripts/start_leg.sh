#!/usr/bin/env bash
# Launch one PD leg at the FROZEN bench config. Runs ON the node.
#
# The config below is sized for the LARGEST fixlen pair (ISL 155,000 + OSL 3,300
# = 158,300) and is then frozen for all 8 rounds and for Case A. Small workloads
# get no server-level retuning -- the deliverable is one deployment measured
# across the load range, not eight tuned deployments.
#
# ctx 262144 also covers Case A's max_input_tokens 260000 clamp, so one server
# serves both phases.
#
# Kills the sglang.launch_server CHILD as well as the infera wrapper: the wrapper
# exits but the child keeps the DP kv-event port block bound, and the next leg
# dies with "port_base at N is not available", which reads as a port-allocation
# bug rather than as leftover state. The WAIT is the point, not the kill.
#
#   ROLE=prefill|decode  MY_IP=<rail ip>  ETCD_IP=<prefill ip>  [MTP=0|1] [TAG=p0]
set -u
ROLE="${ROLE:?}"
MY_IP="${MY_IP:?}"
ETCD_IP="${ETCD_IP:?}"
MTP="${MTP:-0}"
TAG="${TAG:-p0}"
CTR="${CTR:-bench_run}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
W=/mnt/vast/c_huggingface/bench_20260801
LOG="$W/logs/${TAG}_${ROLE}.log"

mkdir -p "$W/logs"

# Re-copy the leg script into the container on EVERY launch. reset_node.sh
# copies it once at container creation, so editing the shared-fs copy between
# rounds silently runs the OLD script -- which is how the hardcoded
# MC_GID_INDEX=1 survived its own fix for one full cold start.
docker cp "$W/scripts/glm52_leg.sh" "$CTR":/glm52_leg.sh >/dev/null

docker exec "$CTR" bash -c '
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f infera.engine.sglang 2>/dev/null
  for i in $(seq 1 20); do
    n=$(ps aux | grep -E "launch_server|infera.engine" | grep -v grep | wc -l)
    [ "$n" -eq 0 ] && exit 0
    sleep 2
  done
  echo "  WARNING: engines still present after 40s" >&2' || true

# CHUNK: glm52_leg.sh derives ISL*TP under DPA, so ISL=8192 TP=8 -> 65536,
# i.e. 8192 per rank -- the measured prefill sweet spot on this stack.
# EXTRA_SERVER_ARGS: --enable-cache-report, or the bench's cache-hit column
# reads 0 and the 89% Case A target cannot be checked at all.
# GMU: prefill 0.88 -> 0.80.
#
# At ISL 155,000 x conc 32 the prefill leg died with
#     rocdevice.cpp:3582 HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Fatal Python error: Aborted
# while `token usage` read 0.01-0.05 -- the KV pool was almost EMPTY, so this is
# not KV exhaustion. It is DP-attention runtime ACTIVATION memory: at dp8 each
# rank holds its own 8192-token chunk activations, and with the prefill-delayer
# batching a 155K prompt (19 chunks) the transient peak exceeds what
# 1 - mem_fraction_static leaves behind.
#
# The fix is therefore to LOWER mem-fraction-static (more activation room), which
# is the OPPOSITE direction from the decode-side retract fix (raise it, more KV
# room). Diagnose by phase: decode retract/NotImplemented -> raise;
# prefill HSA-OOM/Aborted -> lower.
#
# 0.88 crashed at p90 conc=32. Prior first-hand result on this stack: DSv4 DP8
# prefill OOMed at 0.90 and passed clean at 0.85. This is a longer prompt (155K
# vs 1K), so the activation peak is larger -> 0.80, one step past the known-good.
# Decode is untouched at 0.85: it did not crash, and changing both would make the
# fix a two-variable change.
GMU_P="${GMU_P:-0.80}"
GMU_D="${GMU_D:-0.85}"
[ "$ROLE" = "prefill" ] && GMU="$GMU_P" || GMU="$GMU_D"

# SGLANG_DEBUG_DSA_ROWS=1 turns on the `[dsa-rows]` line at
# dsa_indexer.py:949, which prints q_fp8 rows / q_offset / num_token_non_padded
# and whether they agree. That is exactly the bookkeeping the GLM52_P1V2 trim
# gates on, and its own comment says the two "ha[ve] never been measured to
# agree ... on the MTP draft-extend path" -- which is the crash we hit. Off by
# default: it logs once per indexer call, so it is a debug aid, not a run mode.
DSA_ROWS="${DSA_ROWS:-0}"

docker exec -d "$CTR" env \
  ROLE="$ROLE" MY_IP="$MY_IP" ETCD_IP="$ETCD_IP" MODEL="$MODEL" \
  SERVED=glm5.2-mxfp4 PORT=30000 \
  CTX=262144 ISL=8192 TP=8 DPA=1 CUDA_GRAPH_BS=128 MAX_RUNNING=2048 \
  GMU="$GMU" \
  KVAWARE=1 KVD=1 HICACHE_GB=16 KVD_SOCK=/tmp/kvd/kvd.sock \
  MTP="$MTP" \
  SGLANG_DEBUG_DSA_ROWS="$DSA_ROWS" \
  LOG="$LOG" bash /glm52_leg.sh

echo "[$TAG] $ROLE launched on $(hostname) mtp=$MTP ctx=262144 gmu=$GMU -> $LOG"

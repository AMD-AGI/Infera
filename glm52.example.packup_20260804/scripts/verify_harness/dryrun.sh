#!/usr/bin/env bash
# End-to-end dry run of the kit with docker/ssh stubbed out.
# Captures every remote command and every docker argv into one trace file, so two
# revisions of the kit can be diffed for behavioural equivalence.
#
# usage: bash dryrun.sh <kit-dir> <out-trace-dir>
set -euo pipefail
KIT="$(cd "$1" && pwd)"
OUT="$(cd "$2" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HERE/bin:$PATH"

# Site values: fixed, fake, and identical across revisions.
common_env() {
  export PREFILL_NODE=node-P DECODE_NODE=node-D
  export PREFILL_IP=10.0.0.1 DECODE_IP=10.0.0.2
  export KIT_DIR="$KIT"
  export INFERA_IMAGE=inferaimage/infera-sglang:0.2.0
  export MODEL_MOUNT=/weights MODEL=/weights/GLM-5.2-MXFP4 TOKENIZER=/weights/GLM-5.2-MXFP4
  export RDMA_IB_DEVICES=mlx5_0 MC_GID_INDEX=3
  export MC_MS_AUTO_DISC=0 MC_MS_FILTERS=mlx5_0 MOONCAKE_DISABLE_HIP_DMABUF=0
  export TP=8 CTX=262144 CHUNK=65536
  export PREFILL_DPA=0 DECODE_DPA=1 DECODE_MTP=1
  export ROUTER_POLICY=kv-aware KVAWARE=1
  export PREFILL_KVD=1 DECODE_KVD=0
  export GMU_PREFILL=0.70 GMU_DECODE=0.85
  export SSH_CMD="$HERE/bin/fake_ssh"
  export LEG_TRIES=1
}

# --- scenario 1: up, prefill MTP OFF (the shipped default) -------------------
( common_env; export PREFILL_MTP=0 TRACE="$OUT/up_mtpoff.trace"
  : > "$TRACE"; bash "$KIT/engine/up.sh" >"$OUT/up_mtpoff.stdout" 2>&1 || true )

# --- scenario 2: up, prefill MTP ON (the documented opt-in) ------------------
( common_env; export PREFILL_MTP=1 TRACE="$OUT/up_mtpon.trace"
  : > "$TRACE"; bash "$KIT/engine/up.sh" >"$OUT/up_mtpon.stdout" 2>&1 || true )

# --- scenario 3: leg.sh directly, both roles ---------------------------------
for role in prefill decode; do
  ( common_env; export TRACE="$OUT/leg_$role.trace"
    : > "$TRACE"
    export ROLE=$role MY_IP=10.0.0.1 CTR=glm52_pd ETCD_IP=10.0.0.1
    [ "$role" = decode ] && { export MY_IP=10.0.0.2 DPA=1 MTP=1 KVD=0; } || { export DPA=0 MTP=0 KVD=1; }
    bash "$KIT/engine/leg.sh" >"$OUT/leg_$role.stdout" 2>&1 || true )
done

# --- scenario 4: smoke -------------------------------------------------------
( common_env; export TRACE="$OUT/smoke.trace"
  : > "$TRACE"; bash "$KIT/engine/smoke.sh" >"$OUT/smoke.stdout" 2>&1 || true )

# --- scenario 5: bench, the exact invocation the README documents ------------
( common_env; export TRACE="$OUT/bench.trace"
  : > "$TRACE"; bash "$KIT/engine/bench.sh" 8 16 32 >"$OUT/bench.stdout" 2>&1 || true )

# --- scenario 6: down --------------------------------------------------------
( common_env; export TRACE="$OUT/down.trace"
  : > "$TRACE"; bash "$KIT/engine/down.sh" >"$OUT/down.stdout" 2>&1 || true )

# The kit dir path is baked into traces via KIT_DIR; normalise it so two
# checkouts at different paths still compare equal.
sed -i "s#$KIT#<KIT>#g" "$OUT"/*.trace "$OUT"/*.stdout "$OUT"/*.legout 2>/dev/null || true
echo "traces written to $OUT"

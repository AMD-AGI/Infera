#!/usr/bin/env bash
# Router on the prefill node. Runs ON the node.
#
# BACKEND=rust (default) execs the Rust binary -- the only backend where group
# E's bigram kv-event fix is live, which matters because MTP is ON here and MTP
# is exactly what makes SGLang emit bigram BlockStored token_ids.
#
# Two Rust-vs-python deltas, both known, neither a blocker:
#   * --kvd-socket-path is python-only (launch_rust.py builds argv explicitly),
#     so POST /v1/cache/prewarm is absent. Neither bench uses prewarm.
#   * /v1/admin/cache-view/<w>?dp_rank=N does not exist under Rust
#     (handlers.rs routes only /health,/v1/workers,/v1/models,/metrics,/v1/*completions).
#     The kv-aware discriminator therefore comes from the policy log line
#     (policy.rs:314) -- see cache_view.sh.
#
# The module is `python -m infera.server`. It is NOT `infera.router` (a package
# with no __main__, failing with a message that reads like a missing dependency).
#
# POLICY DELTA (work.agentx_rr_20260804) -- the ONLY change from par8's copy.
# --router-policy was hardcoded to kv-aware; it is now a variable whose default
# is UNCHANGED, so an unset POLICY emits a byte-identical command line.
#
# The two policies are MUTUALLY EXCLUSIVE (infera/router/policy/factory.py:56-59;
# the Rust backend's supported subset is infera/server/launch_rust.py:30). There
# is no "kv-aware with a round-robin tiebreak" -- selecting round-robin turns
# cache-locality scoring off entirely.
#
# The kv-aware-only flags (--router-tokenizer-path, --kv-*-overlap-weight) are
# deliberately LEFT on the command line under round-robin: infera.server accepts
# them for either policy, and launch_rust.py:87 gates them behind
# `if args.router_policy == "kv-aware"`, so they are never forwarded to the Rust
# binary. Keeping them makes the two invocations differ in exactly one token.
# Verify by reading the resolved argv back from `ps`, not by trusting this note.
set -u
CTR="${CTR:-bench_run}"
MY_IP="${MY_IP:-10.2.122.10}"
PORT="${PORT:-8100}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
BACKEND="${BACKEND:-rust}"
POLICY="${POLICY:-kv-aware}"   # kv-aware | round-robin
PW="${PW:-20.0}"          # --kv-prefill-overlap-weight
DW="${DW:-2.0}"           # --kv-decode-overlap-weight
KVD_SOCK=/tmp/kvd/kvd.sock

EXTRA=""
if [ "$BACKEND" = "python" ]; then
  EXTRA="--kvd-socket-path $KVD_SOCK"
fi

docker exec "$CTR" bash -c "pkill -9 -f 'infera.server' 2>/dev/null; pkill -9 -f infera-router 2>/dev/null; sleep 2" || true
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' \
  'export RUST_LOG=\${RUST_LOG:-info}' \
  'exec python3 -m infera.server --host 0.0.0.0 --port $PORT --router-backend $BACKEND --discovery-backend etcd --etcd-endpoint $MY_IP:2379 --request-transport http --kv-event-transport zmq --router-policy $POLICY --router-tokenizer-path $MODEL --kv-prefill-overlap-weight $PW --kv-decode-overlap-weight $DW $EXTRA' \
  > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$PORT/health && echo '  router healthy (backend=$BACKEND policy=$POLICY pw=$PW dw=$DW)' || { echo '  ROUTER NOT READY'; tail -30 /tmp/router.log; }"
# The health line reports the REQUESTED policy. Confirm the RESOLVED one:
docker exec "$CTR" bash -c "ps -eo args | grep '[i]nfera-router' | tr ' ' '\n' | grep -A1 -- '--router-policy' | tail -1 | sed 's/^/  resolved --router-policy: /'"

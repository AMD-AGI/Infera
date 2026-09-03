#!/bin/sh
set -eu
. "$(dirname "$0")/stub_env.sh"
mkdir -p "$E2E_KIT_WORK_ROOT"
# The extra-env seam m2 uses: a space-separated K=V list, into the worker only.
for kv in $E2E_KIT_ENGINE_EXTRA_ENV; do export "$kv"; done
STUB_ROLE=router STUB_MODEL="$KIT_MODEL" STUB_CTX="$KIT_CTX" \
  nohup python3 "$KIT_DIR/stub_router.py" "$ROUTER_PORT" $E2E_KIT_ENGINE_EXTRA_ARGS \
  > "$E2E_KIT_WORK_ROOT/router.log" 2>&1 &
echo $! > "$E2E_KIT_WORK_ROOT/router.pid"
STUB_ROLE=engine STUB_MODEL="$KIT_MODEL" STUB_CTX="$KIT_CTX" \
  nohup python3 "$KIT_DIR/stub_router.py" "$ENGINE_PORT" \
  > "$E2E_KIT_WORK_ROOT/engine.log" 2>&1 &
echo $! > "$E2E_KIT_WORK_ROOT/engine.pid"
sh "$KIT_DIR/wait_ready.sh"
cat > "$E2E_KIT_WORK_ROOT/deployment.json" <<JSON
{"endpoint": "http://$(hostname -i | awk '{print $1}'):${ROUTER_PORT}",
 "work_root_in_container": "${E2E_KIT_WORK_ROOT}",
 "containerized": false,
 "engine_endpoint": "http://$(hostname -i | awk '{print $1}'):${ENGINE_PORT}",
 "container": "stub_${E2E_KIT_RUN_TAG}",
 "run_tag": "${E2E_KIT_RUN_TAG}",
 "ports": {"router": ${ROUTER_PORT}, "engine": ${ENGINE_PORT}}}
JSON
echo "deployed ${E2E_KIT_RUN_TAG} -> ${E2E_KIT_WORK_ROOT}/deployment.json"

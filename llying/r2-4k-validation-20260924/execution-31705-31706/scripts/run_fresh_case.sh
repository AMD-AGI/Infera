#!/usr/bin/env bash
set -Eeuo pipefail
set -a; source "${CONFIG:?}"; set +a
ROOT="$TRACE_RUNTIME"
[[ ! -e "$RUN/STATUS" ]] || { echo 'Case already exists'; exit 1; }
mkdir -p "$RUN/logs" "$RUN/sampling" "$RUN/traces" "$RUN/snapshot"
trap 'code=$?; printf "%s FAILED_STARTUP line=%s\n" "$(date -u --iso-8601=seconds)" "$LINENO" > "$RUN/STATUS"; python3 "$ROOT/scripts/cleanup_owned.py" || true; exit "$code"' ERR
cp "$CONFIG" "$RUN/snapshot/config.sh"
python3 "$ROOT/scripts/require_performance_approval.py"
python3 "$ROOT/scripts/validate_allocation.py"
python3 "$ROOT/scripts/verify_guards.py"
python3 "$ROOT/scripts/wait_nodes_idle.py" "$PREFILL_NODE" "$DECODE_NODE" --timeout 600 --interval 15 > "$RUN/logs/wait-idle.log"
read -r -a opts <<< "$SSH_OPTS"
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
 [[ "$(ssh "${opts[@]}" "$node" docker image inspect "$IMAGE" --format '{{.Id}}')" == sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb ]]
done
test -x "$ROUTER_BINARY_OVERRIDE"
sha256sum "$ROUTER_BINARY_OVERRIDE" > "$RUN/snapshot/router-binary.sha256"
python3 "$ROOT/scripts/capture_model_identity.py" "$RUN/snapshot/model-identity.json"
printf '{}\n' > "$RUN/snapshot/diagnostic-cursors.json"
date +%s > "$RUN/snapshot/capture-start-epoch.txt"
docker run -d --init --name "$CONTAINER_PREFIX-collector" --network host -v "$ROOT:$ROOT" "$IMAGE" python3 "$ROOT/scripts/otlp_jsonl_collector.py" --output "$RUN/traces/spans.jsonl" --ready-file "$RUN/traces/ready.json" > "$RUN/logs/collector-id.txt"
for i in $(seq 1 60); do [[ ! -s "$RUN/traces/ready.json" ]] || break; sleep 1; done
test -s "$RUN/traces/ready.json"
printf '%s LAUNCHING\n' "$(date -u --iso-8601=seconds)" > "$RUN/STATUS"
bash "$BENCH_DIR/launch.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" "OUT_DIR=$RUN/launch" > "$RUN/logs/launch.log" 2>&1
exec bash "$ROOT/scripts/run_measurement.sh"

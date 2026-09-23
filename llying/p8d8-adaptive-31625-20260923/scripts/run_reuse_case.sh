#!/usr/bin/env bash
set -Eeuo pipefail
set -a; source "${CONFIG:?}"; set +a
ROOT="$TRACE_RUNTIME"
[[ ! -e "$RUN/STATUS" ]] || { echo 'Case already exists'; exit 1; }
mkdir -p "$RUN/logs" "$RUN/sampling" "$RUN/traces" "$RUN/snapshot"
trap 'printf "%s FAILED_REUSE_PREPARATION line=%s\n" "$(date -u --iso-8601=seconds)" "$LINENO" > "$RUN/STATUS"' ERR
cp "$CONFIG" "$RUN/snapshot/config.sh"
python3 "$ROOT/scripts/validate_allocation.py"
printf '%s PREPARING_REUSED_ENGINES\n' "$(date -u --iso-8601=seconds)" > "$RUN/STATUS"
python3 "$ROOT/scripts/reuse_prepare.py"
sha256sum "$ROUTER_BINARY_OVERRIDE" > "$RUN/snapshot/router-binary.sha256"
python3 "$ROOT/scripts/capture_model_identity.py" "$RUN/snapshot/model-identity.json"
docker run -d --init --name "$CONTAINER_PREFIX-collector" --network host -v "$ROOT:$ROOT" "$IMAGE" python3 "$ROOT/scripts/otlp_jsonl_collector.py" --output "$RUN/traces/spans.jsonl" --ready-file "$RUN/traces/ready.json" > "$RUN/logs/collector-id.txt"
for i in $(seq 1 60); do [[ ! -s "$RUN/traces/ready.json" ]] || break; sleep 1; done
test -s "$RUN/traces/ready.json"
python3 "$ROOT/scripts/start_router.py"
exec bash "$ROOT/scripts/run_measurement.sh"

#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG:?set CONFIG to the frozen case file}"
set -a; source "$CONFIG"; set +a
mkdir -p "$RUN/logs" "$RUN/snapshot" "$RUN/traces" "$RUN/sampling"
cp "$CONFIG" "$RUN/snapshot/config.sh"
sha256sum "$ROUTER_BINARY_OVERRIDE" > "$RUN/snapshot/router-binary.sha256"
ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts "$PREFILL_NODE" \
 "docker run -d --init --name $CONTAINER_PREFIX-collector --network host -v $TRACE_RUNTIME:$TRACE_RUNTIME $IMAGE python3 $TRACE_RUNTIME/scripts/otlp_jsonl_collector.py --output $RUN/traces/spans.jsonl --ready-file $RUN/traces/ready.json"
bash "$BENCH_DIR/launch.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" "OUT_DIR=$RUN/launch"

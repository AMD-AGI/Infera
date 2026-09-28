#!/usr/bin/env bash
set -euo pipefail
export CONFIG=/perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b1.sh
set -a; source "$CONFIG"; set +a
mkdir -p "$RUN/logs" "$RUN/snapshot" "$RUN/traces" "$RUN/sampling" "$RUN/launch/server-info" "$RUN/launch/server-logs"
cp "$CONFIG" "$RUN/snapshot/config.sh"
printf '{}\n' > "$RUN/live-containers.json"
date +%s > "$RUN/snapshot/capture-start-epoch.txt"
# The Decode engine is reused; P switches from the no-HiCache smoke to the fixed performance configuration.
ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts "$PREFILL_NODE" \
 'docker stop -t 60 llying-campaign-b1-smoke-prefill-0 llying-campaign-b1-smoke-router llying-campaign-b1-smoke-collector'
ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts "$PREFILL_NODE" \
 "docker run -d --init --name $CONTAINER_PREFIX-collector --network host -v $TRACE_RUNTIME:$TRACE_RUNTIME $IMAGE python3 $TRACE_RUNTIME/scripts/otlp_jsonl_collector.py --output $RUN/traces/spans.jsonl --ready-file $RUN/traces/ready.json"
ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts "$PREFILL_NODE" \
 "bash $BENCH_DIR/engine.sh prefill prefill-0 $PREFILL_IP $PREFILL_GPU_DEVICES 29001 28998 25557 28801 $CONTAINER_PREFIX-prefill-0 $PREFILL_IP:22379 CONFIG=$CONFIG SERVER_LOG=$RUN/launch/server-logs/prefill-0.log"
python3 "$BENCH_DIR/tools/wait_healthy.py" --target prefill "$PREFILL_NODE" "$CONTAINER_PREFIX-prefill-0" "http://$PREFILL_IP:29001/health" \
 --ssh-options "$SSH_OPTS" --timeout 3600 --interval 10 --probe-timeout 5 --failure-dir "$RUN/launch/failures" --summary "$RUN/launch/worker-health.json"
python3 "$TRACE_RUNTIME/scripts/start_performance_router.py"
python3 "$BENCH_DIR/tools/wait_healthy.py" --target router "$PREFILL_NODE" "$CONTAINER_PREFIX-router" "http://$PREFILL_IP:28000/health" \
 --ssh-options "$SSH_OPTS" --timeout 300 --interval 5 --probe-timeout 5 --failure-dir "$RUN/launch/failures" --summary "$RUN/launch/router-health.json"
python3 "$TRACE_RUNTIME/scripts/performance_preflight.py"
printf 'READY_FOR_C80\n' > "$RUN/STATUS"

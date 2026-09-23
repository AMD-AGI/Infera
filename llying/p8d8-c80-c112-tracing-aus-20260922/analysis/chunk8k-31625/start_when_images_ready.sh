#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/p8d8-chunk8k-31625-20260923
set -a
source "$ROOT/config/config.sh"
set +a
expected=sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb
for attempt in $(seq 1 120); do
    ready=1
    for node in "$PREFILL_NODE" "$DECODE_NODE"; do
        actual=$(ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/agentx_known_hosts "$node" docker image inspect "$IMAGE" --format "{{.Id}}" 2>/dev/null) || actual=missing
        [[ "$actual" == "$expected" ]] || ready=0
    done
    if [[ "$ready" == 1 ]]; then
        exec bash "$ROOT/scripts/run_chunk8k.sh"
    fi
    echo "$(date -u --iso-8601=seconds) WAITING_IMAGES attempt=$attempt"
    sleep 15
done
echo "IMAGE_WAIT_FAILED"
exit 1

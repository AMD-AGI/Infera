#!/usr/bin/env bash
set -Eeuo pipefail
root=/perf_apps/liyingli/bench_agentx/r2-4k-31705-31706-20260924
docker save infera-sglang:aus-0922-reqtrace | ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$root/config/known_hosts" smci355-ccs-aus-n01-21 docker load

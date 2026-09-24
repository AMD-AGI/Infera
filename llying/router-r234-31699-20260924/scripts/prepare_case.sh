#!/usr/bin/env bash
set -Eeuo pipefail
set -a; source "${CONFIG:?}"; set +a
if [[ "${SMOKE_ONLY:-0}" == 0 ]]; then python3 "$TRACE_RUNTIME/scripts/require_performance_approval.py"; fi
mkdir -p "$RUN/logs" "$RUN/snapshot"
python3 "$TRACE_RUNTIME/scripts/validate_allocation.py"
echo "$RUN" > "$TRACE_RUNTIME/active-run.txt"
read -r -a opts <<< "$SSH_OPTS"
seed=/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/artifacts/aiter-seed-n04-33-complete.tar
image_key=4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb
for role in PREFILL DECODE; do
 node_var=${role}_NODE
 node=${!node_var}
 ssh "${opts[@]}" "$node" python3 "$TRACE_RUNTIME/scripts/aiter_cache_bundle.py" install "$AITER_JIT_CACHE_ROOT/$image_key" "$seed" > "$RUN/snapshot/${role,,}-aiter-seed.json"
 ssh "${opts[@]}" "$node" 'uname -a; cat /sys/module/amdgpu/version; cat /proc/meminfo | head -3; ls /sys/class/infiniband; for dev in /sys/class/infiniband/*/device/net/*; do ethtool -i "${dev##*/}" 2>/dev/null | head -5; done' > "$RUN/snapshot/${role,,}-host.txt"
done

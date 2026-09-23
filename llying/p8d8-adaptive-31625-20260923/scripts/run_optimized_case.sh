#!/usr/bin/env bash
set -Eeuo pipefail
set -a; source "${CONFIG:?}"; set +a
ROOT="$TRACE_RUNTIME"
[[ "$GUARD_MODE" == completion ]] || { echo 'This entrypoint runs optimization-enabled cases only'; exit 1; }
read -r -a opts <<< "$SSH_OPTS"
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    ready=0
    for attempt in $(seq 1 120); do
        actual=$(timeout 20s ssh -n "${opts[@]}" "$node" docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)
        if [[ "$actual" == sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb ]]; then ready=1; break; fi
        sleep 5
    done
    [[ "$ready" == 1 ]] || { echo "Image readiness failed on $node"; exit 1; }
done
case "${1:-fresh}" in
    fresh) bash "$ROOT/scripts/run_fresh_case.sh" ;;
    reuse) bash "$ROOT/scripts/run_reuse_case.sh" ;;
    *) exit 2 ;;
esac
python3 "$ROOT/scripts/analyze_guard_lifecycle.py" "$RUN"
python3 "$ROOT/scripts/analyze_router_picks.py" "$RUN"
python3 "$ROOT/scripts/audit_case.py" "$RUN"

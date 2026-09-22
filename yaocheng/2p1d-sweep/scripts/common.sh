#!/usr/bin/env bash
# Shared argument/config loading. All entry points accept KEY=VALUE arguments.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS_DIR="$SCRIPT_DIR/bench-harness"
assignment_keys=()
for assignment in "$@"; do
    [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || {
        echo "expected KEY=VALUE, got: $assignment" >&2; exit 2;
    }
    export "$assignment"
    assignment_keys+=("${assignment%%=*}")
done
CONFIG="$(realpath "${CONFIG:-$KIT_DIR/config.sh}")"
set -a
source "$CONFIG"
# These are driver controls, not immutable engine settings. In particular a
# continuation can select only the not-yet-run points from a frozen config.
for assignment in "$@"; do
    case "${assignment%%=*}" in
        CONC|DURATION|SWEEP_POINTS) export "$assignment" ;;
    esac
done
TOPOLOGY="$(realpath "${TOPOLOGY:-$KIT_DIR/topology.tsv}")"
DURATION="${DURATION:-$AGENTX_DURATION}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_DIR="${RUN_DIR:-$SWEEP_TMP_DIR/results/$RUN_ID}"
RAW_DIR="${RAW_DIR:-$SWEEP_TMP_DIR/raw/$RUN_ID}"
INFERENCEX_DIR="${INFERENCEX_DIR:-$SWEEP_TMP_DIR/cache/InferenceX}"
set +a
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "unsafe RUN_ID" >&2; exit 2;
}
for path in "$SWEEP_TMP_DIR" "$RUN_DIR" "$RAW_DIR" "$RESULTS_DIR" "$AGENTX_CACHE_DIR" "$INFERENCEX_DIR" "$AITER_JIT_CACHE_ROOT"; do
    [[ "$path" == /* ]] || { echo "use absolute paths: $path" >&2; exit 2; }
done
python3 "$HARNESS_DIR/tools/topology.py" validate "$TOPOLOGY"
[[ "$(python3 "$HARNESS_DIR/tools/topology.py" count "$TOPOLOGY" prefill)" == 2 &&
   "$(python3 "$HARNESS_DIR/tools/topology.py" count "$TOPOLOGY" decode)" == 1 ]] || {
    echo "this campaign requires exactly 2 prefill nodes and 1 decode node" >&2; exit 2;
}
CONTROL_IP="$(python3 "$HARNESS_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
export CONTROL_IP
for value in "$DURATION" "$POINT_TIMEOUT" "$AGENTX_WARMUP_REQUESTS_PER_LANE" \
    "$PREFILL_MAX_RUNNING" "$DECODE_MAX_RUNNING" "$PREFILL_GRAPH_MAX_BS" "$DECODE_GRAPH_MAX_BS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "expected positive integer: $value" >&2; exit 2; }
done
read -r -a points <<< "$SWEEP_POINTS"
(( ${#points[@]} > 0 )) || { echo "empty SWEEP_POINTS" >&2; exit 2; }
declare -A seen_points=()
for conc in "${points[@]}"; do
    [[ "$conc" =~ ^[1-9][0-9]*$ ]] || { echo "invalid concurrency: $conc" >&2; exit 2; }
    [[ ! -v seen_points[$conc] ]] || { echo "duplicate concurrency: $conc" >&2; exit 2; }
    seen_points[$conc]=1
    (( conc <= PREFILL_MAX_RUNNING && conc <= DECODE_MAX_RUNNING &&
        conc <= PREFILL_GRAPH_MAX_BS && conc <= DECODE_GRAPH_MAX_BS )) || {
        echo "concurrency $conc exceeds configured max-running; raise caps in config before launch" >&2; exit 2;
    }
done
for key in NODE_GPU_COUNT PD_DP_RANK_AFFINITY; do
    wanted=1; [[ "$key" != NODE_GPU_COUNT ]] || wanted=8
    [[ "${!key}" == "$wanted" ]] || { echo "$key must be $wanted" >&2; exit 2; }
done
for role in PREFILL DECODE; do
    for field in TP DP; do
        key="${role}_${field}"
        [[ "${!key}" == 8 ]] || { echo "$key must be 8 for this 24-GPU campaign" >&2; exit 2; }
    done
    for field in EP DPA; do
        key="${role}_${field}"
        [[ "${!key}" == 1 ]] || { echo "$key must be 1" >&2; exit 2; }
    done
    key="${role}_GPU_DEVICES"
    [[ "${!key}" == 0,1,2,3,4,5,6,7 ]] || { echo "$key must select all 8 GPUs" >&2; exit 2; }
done
read -r -a ssh_args <<< "$SSH_OPTS"
mapfile -t nodes < <(python3 "$HARNESS_DIR/tools/topology.py" nodes "$TOPOLOGY")
topology_rows="$(python3 "$HARNESS_DIR/tools/topology.py" rows "$TOPOLOGY")"
remote_command() {
    local quoted result="" item
    for item in "$@"; do
        printf -v quoted '%q' "$item"
        result+="${result:+ }$quoted"
    done
    printf '%s\n' "$result"
}
ssh_run() {
    local node="$1"; shift
    ssh -n "${ssh_args[@]}" "$node" "$(remote_command "$@")"
}
say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
# Snapshot only declared benchmark settings, never the caller's full environment.
freeze_config() {
    local key
    while read -r key; do
        [[ -v "$key" ]] || continue
        case "$key" in
            CONC|DURATION) printf '%s=${%s:-%q}\n' "$key" "$key" "${!key}" ;;
            *) printf '%s=%q\n' "$key" "${!key}" ;;
        esac
    done < <(python3 - "$KIT_DIR/config.sh" "$CONFIG" "${assignment_keys[@]}" <<'PY'
import re, sys
from pathlib import Path
keys = {'KIT_DIR', 'DURATION', 'RUN_ID', 'RUN_DIR', 'RAW_DIR'}
for name in sys.argv[1:3]:
    keys.update(re.findall(r'^\s*([A-Z][A-Z0-9_]*)=', Path(name).read_text(), re.M))
keys.update(sys.argv[3:])
keys.difference_update({'CONFIG', 'TOPOLOGY', 'ON_CONTROL', 'DRY_RUN'})
print('\n'.join(sorted(keys)))
PY
    )
}

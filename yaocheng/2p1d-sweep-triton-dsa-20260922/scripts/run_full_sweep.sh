#!/usr/bin/env bash
# Run from any SSH-capable host. Freeze settings, then orchestrate on CONTROL_NODE.
source "$(dirname "$0")/common.sh"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '2P1D / 24 GPUs\ncontrol=%s\nconfig=%s\ntopology=%s\npoints=%s\nduration=%ss warmup=%s/lane\nrun=%s\nraw=%s\nagentx_cache=%s\ninferencex=%s\naiter_cache=%s\n' \
        "$CONTROL_NODE" "$CONFIG" "$TOPOLOGY" "$SWEEP_POINTS" "$DURATION" \
        "$AGENTX_WARMUP_REQUESTS_PER_LANE" "$RUN_DIR" "$RAW_DIR" \
        "$AGENTX_CACHE_DIR" "$INFERENCEX_DIR" "$AITER_JIT_CACHE_ROOT"
    printf 'dsa_flags=--dsa-prefill-backend %s --dsa-decode-backend %s\ncontainer_prefix=%s\n' \
        "$DSA_PREFILL_BACKEND" "$DSA_DECODE_BACKEND" "$CONTAINER_PREFIX"
    printf '%s\n' "$topology_rows"
    exit 0
fi
if [[ "${ON_CONTROL:-0}" != 1 ]]; then
    [[ ! -e "$RUN_DIR" ]] || { echo "run already exists: $RUN_DIR" >&2; exit 1; }
    mkdir -p "$RUN_DIR"
    freeze_config > "$RUN_DIR/config.resolved.sh"
    cp "$TOPOLOGY" "$RUN_DIR/topology.tsv"
    python3 - "$KIT_DIR" "$RUN_DIR/source-sha256.txt" <<'PY'
import hashlib, pathlib, sys
kit, output = map(pathlib.Path, sys.argv[1:])
files = sorted((kit / 'scripts').rglob('*.sh')) + sorted((kit / 'scripts').rglob('*.py'))
files += [kit / 'config.sh', kit / 'topology.tsv']
output.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(kit)}\n' for p in files))
PY
    printf '%s\n' "$CONTROL_NODE:$RAW_DIR" > "$RUN_DIR/REMOTE_ARTIFACTS.txt"
    printf 'connecting\n' > "$RUN_DIR/status.txt"
    rc=0
    ssh_run "$CONTROL_NODE" bash "$SCRIPT_DIR/run_full_sweep.sh" \
        "CONFIG=$RUN_DIR/config.resolved.sh" "TOPOLOGY=$RUN_DIR/topology.tsv" \
        ON_CONTROL=1 2>&1 | tee "$RUN_DIR/orchestrator.log" || rc=$?
    if (( rc != 0 )); then
        printf 'failed exit=%s; see orchestrator.log\n' "$rc" > "$RUN_DIR/status.txt"
    fi
    exit "$rc"
fi
[[ -d "$RUN_DIR" ]] || { echo "missing frozen run directory" >&2; exit 1; }
# One campaign at a time for this kit, including manual driver continuations.
mkdir -p "$SWEEP_TMP_DIR"
exec 9> "$SWEEP_TMP_DIR/.deployment.lock"
flock -n 9 || { echo "another campaign is running" >&2; exit 1; }
printf 'checking\n' > "$RUN_DIR/status.txt"
VERIFY_OUT_DIR="$RUN_DIR/patch-check" bash "$SCRIPT_DIR/check.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
    2>&1 | tee "$RUN_DIR/check.log"
[[ ! -e "$RAW_DIR" ]] || { echo "raw directory already exists: $RAW_DIR" >&2; exit 1; }
mkdir -p "$RAW_DIR" "$AGENTX_CACHE_DIR"
if [[ "$RUN_PREFLIGHT" == 1 ]]; then
    decode_node="$(awk -F '\t' '$1=="decode"{print $2}' "$TOPOLOGY")"
    while IFS=$'\t' read -r index instance role node ip; do
        [[ "$role" == prefill ]] || continue
        say "Mooncake WRITE preflight: $node -> $decode_node"
        OUT_DIR="$RUN_DIR/preflight-$instance" bash "$HARNESS_DIR/preflight.sh" "$node" "$decode_node"
    done <<< "$topology_rows"
fi
started=0
finish() {
    local rc=$? stop_rc=0
    trap - EXIT
    if (( started )) && [[ "$STOP_AFTER_SWEEP" == 1 ]]; then
        bash "$HARNESS_DIR/stop.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
            > "$RUN_DIR/stop.log" 2>&1 || stop_rc=$?
        (( rc != 0 || stop_rc == 0 )) || rc=$stop_rc
    fi
    if (( rc == 0 )); then
        printf 'complete\n' > "$RUN_DIR/status.txt"
    else
        printf 'failed exit=%s; see orchestrator.log and point logs\n' "$rc" > "$RUN_DIR/status.txt"
    fi
    python3 "$SCRIPT_DIR/publish_results.py" "$RUN_DIR" "$RESULTS_DIR" || rc=1
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
started=1
printf 'launching\n' > "$RUN_DIR/status.txt"
bash "$HARNESS_DIR/launch.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY" \
    "OUT_DIR=$RUN_DIR/launch" 2>&1 | tee "$RUN_DIR/launch.log"
python3 "$SCRIPT_DIR/audit.py" "$RUN_DIR/deployment.json"
printf 'sweeping\n' > "$RUN_DIR/status.txt"
SWEEP_LOCK_HELD=1 bash "$SCRIPT_DIR/sweep_driver.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY"
say "all points completed: $RUN_DIR/summary.md"

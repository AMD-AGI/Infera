#!/usr/bin/env bash
# Purpose: Check standalone nodes for SSH, Docker, idle GPUs, and service containers.
# Usage: ./check_nodes.sh NODE [NODE ...]
#   Comma-separated input is also accepted: ./check_nodes.sh node-a,node-b
# Artifacts: none; PASS/WARN/ERROR details are printed to the terminal.
# Artifact paths: not applicable.
set -uo pipefail

usage() {
    echo "usage: $0 NODE [NODE ...]" >&2
    echo "       $0 node-a,node-b" >&2
}

(( $# > 0 )) || { usage; exit 2; }

vram_limit="${GPU_IDLE_VRAM_PCT:-2}"
util_limit="${GPU_IDLE_UTIL_PCT:-5}"
for value in "$vram_limit" "$util_limit"; do
    [[ "$value" =~ ^[0-9]+$ ]] && (( value <= 100 )) || {
        echo "GPU idle thresholds must be integers in 0..100" >&2
        exit 2
    }
done

nodes=()
declare -A seen=()
for argument in "$@"; do
    IFS=',' read -r -a items <<<"$argument"
    for node in "${items[@]}"; do
        [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]] || {
            echo "invalid node name: $node" >&2
            exit 2
        }
        if [[ -z "${seen[$node]:-}" ]]; then
            nodes+=("$node")
            seen["$node"]=1
        fi
    done
done

read -r -a ssh_args <<<"${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
status=0
for node in "${nodes[@]}"; do
    echo "== $node =="
    if ! ssh "${ssh_args[@]}" "$node" bash -s -- \
        "$vram_limit" "$util_limit" <<'REMOTE'
set -uo pipefail
vram_limit="$1"
util_limit="$2"
failed=0

echo "PASS ssh"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR docker: command is unavailable"
    failed=1
elif ! docker info >/dev/null 2>&1; then
    echo "ERROR docker: daemon is unavailable or access is denied"
    failed=1
else
    echo "PASS docker"
    containers="$(
        docker ps -a --format '{{.Names}}|{{.Status}}|{{.Image}}' 2>/dev/null |
            awk -F'|' 'tolower($1) ~ /(prefill|decode|router|etcd)/'
    )"
    if [[ -n "$containers" ]]; then
        echo "WARNING existing prefill/decode/router/etcd containers; check before running:"
        while IFS= read -r container; do
            echo "  $container"
        done <<<"$containers"
    else
        echo "PASS containers: no prefill/decode/router/etcd names found"
    fi
fi

if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "ERROR gpu: rocm-smi is unavailable"
    failed=1
else
    report="$(rocm-smi 2>&1)"
    rc=$?
    if (( rc != 0 )); then
        echo "ERROR gpu: rocm-smi failed"
        echo "$report"
        failed=1
    else
        metrics="$(
            awk '
                $1 ~ /^[0-9]+$/ &&
                $(NF-1) ~ /^[0-9]+%$/ &&
                $NF ~ /^[0-9]+%$/ {
                    vram=$(NF-1)
                    util=$NF
                    gsub(/%/, "", vram)
                    gsub(/%/, "", util)
                    print $1, vram, util
                }
            ' <<<"$report"
        )"
        if [[ -z "$metrics" ]]; then
            echo "ERROR gpu: could not parse VRAM% and GPU% from rocm-smi"
            failed=1
        else
            while read -r gpu vram util; do
                if (( vram > vram_limit || util > util_limit )); then
                    echo "ERROR gpu[$gpu]: busy (VRAM=${vram}%, utilization=${util}%; limits ${vram_limit}%/${util_limit}%)"
                    failed=1
                else
                    echo "PASS gpu[$gpu]: idle (VRAM=${vram}%, utilization=${util}%)"
                fi
            done <<<"$metrics"
        fi
    fi
fi

exit "$failed"
REMOTE
    then
        echo "ERROR $node: one or more checks failed"
        status=1
    fi
done

(( status == 0 )) && echo "PASS: all nodes are reachable and GPUs are idle"
exit "$status"

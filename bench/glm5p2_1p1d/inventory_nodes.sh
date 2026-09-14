#!/usr/bin/env bash
# Capture the evidence used to select a stable two-node pair before deployment
# configuration is loaded.
# Usage: bash inventory_nodes.sh [NODE ...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep discovery independent from config.sh: that file represents an already
# selected pair and resolves both nodes' data-plane IPs while it is sourced.
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no}"
RESULTS_DIR="${RESULTS_DIR:-$DIR/results}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_DIR/$RUN_ID}"

NODES=("$@")
if (( ${#NODES[@]} == 0 )); then
    NODES=(crsuse2-m2m-{135..142})
fi

OUT="${INVENTORY_OUT_DIR:-$RUN_ROOT/inventory}"
mkdir -p "$OUT"
printf 'node\treachable\tmax_gpu_use_pct\tmax_vram_pct\tworkload_containers\tionic_ready\troot_use_pct\troot_available_gb\tiommu_mode\n' \
    >"$OUT/summary.tsv"

for node in "${NODES[@]}"; do
    log="$OUT/$node.log"
    if ! ssh $SSH_OPTS "$node" '
        set -euo pipefail
        command -v rocm-smi >/dev/null
        command -v docker >/dev/null
        command -v df >/dev/null
        hostname
        date -Is
        rocm_output="$(rocm-smi --showuse --showmemuse)"
        printf "%s\n" "$rocm_output"
        [[ "$(printf "%s\n" "$rocm_output" | grep -c "GPU use (%)")" == "8" ]]
        [[ "$(printf "%s\n" "$rocm_output" | grep -c "VRAM%")" == "8" ]]
        echo __DOCKER_PS__
        docker ps --format "{{.Names}}\t{{.Image}}\t{{.Status}}"
        echo __IONIC__
        shopt -s nullglob
        devices=(/sys/class/infiniband/ionic_*)
        [[ "${#devices[@]}" == "8" ]]
        for dev in "${devices[@]}"; do
            printf "%s\t%s\t%s\n" "${dev##*/}" \
                "$(cat "$dev/ports/1/state")" \
                "$(cat "$dev/ports/1/gid_attrs/types/1")"
        done
        echo __CMDLINE__
        cat /proc/cmdline
        echo __DF__
        df -P /
    ' >"$log" 2>&1; then
        printf '%s\tfalse\t\t\t\t\t\t\t\n' "$node" >>"$OUT/summary.tsv"
        continue
    fi

    max_use="$(awk -F': ' '/GPU use \(%\)/ {if ($NF + 0 > max) max=$NF + 0} END {print max + 0}' "$log")"
    max_vram="$(awk -F': ' '/VRAM%/ {if ($NF + 0 > max) max=$NF + 0} END {print max + 0}' "$log")"
    workload_containers="$(
        awk -F'\t' '
            /__DOCKER_PS__/ {in_docker=1; next}
            /__IONIC__/ {in_docker=0}
            in_docker && NF >= 2 && $2 !~ /(crusoe|vector|metrics-exporter)/ {n++}
            END {print n + 0}
        ' "$log"
    )"
    ionic_ready="$(
        awk -F'\t' '
            /__IONIC__/ {in_ionic=1; next}
            /__CMDLINE__/ {in_ionic=0}
            in_ionic && $2 ~ /ACTIVE/ && $3 == "RoCE v2" {n++}
            END {print n + 0}
        ' "$log"
    )"
    iommu_mode="$(
        awk '
            /__CMDLINE__/ {
                getline
                line = " " $0 " "
                unsafe = line ~ / (iommu=pt|iommu\.passthrough=1|iommu=off|amd_iommu=off|amd_iommu=pt) /
                if (unsafe)
                    print "passthrough-or-disabled"
                else
                    print "kernel-default"
                exit
            }
        ' "$log"
    )"
    root_use="$(awk '$6 == "/" {gsub(/%/, "", $5); print $5}' "$log" | tail -1)"
    root_available_gb="$(awk '$6 == "/" {printf "%.0f", $4 / 1048576}' "$log" | tail -1)"
    printf '%s\ttrue\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$node" "$max_use" "$max_vram" "$workload_containers" "$ionic_ready" \
        "$root_use" "$root_available_gb" "$iommu_mode" \
        >>"$OUT/summary.tsv"
done

cat "$OUT/summary.tsv"
echo "[inventory] artifacts: $OUT"

# Prefer the caller's pair when it is still healthy. Otherwise retain either
# healthy member first, then fill from the idle pool. This makes node loss a
# reselection event rather than a workflow blocker.
mapfile -t eligible_nodes < <(
    awk -F '\t' '
        NR > 1 && $2 == "true" && $3 == 0 && $4 <= 10 &&
            $5 == 0 && $6 == 8 && $7 < 90 && $8 >= 60 &&
            $9 == "kernel-default" {print $1}
    ' "$OUT/summary.tsv"
)

declare -A eligible=() added=()
for node in "${eligible_nodes[@]}"; do
    eligible["$node"]=1
done

preferred_prefill="${PREFERRED_PREFILL_NODE:-${PREFILL_NODE:-}}"
preferred_decode="${PREFERRED_DECODE_NODE:-${DECODE_NODE:-}}"
ordered_nodes=()
for node in "$preferred_prefill" "$preferred_decode" "${eligible_nodes[@]}"; do
    [[ -n "$node" && -n "${eligible[$node]:-}" && -z "${added[$node]:-}" ]] || continue
    ordered_nodes+=("$node")
    added["$node"]=1
done

if (( ${#ordered_nodes[@]} < 2 )); then
    echo "[inventory] fewer than two suitable nodes are available in the candidate pool" >&2
    exit 75
fi

selected_prefill="${ordered_nodes[0]}"
selected_decode="${ordered_nodes[1]}"
{
    printf 'PREFILL_NODE=%q\n' "$selected_prefill"
    printf 'DECODE_NODE=%q\n' "$selected_decode"
} >"$OUT/selected.env"

if [[ -n "$preferred_prefill" || -n "$preferred_decode" ]] &&
    [[ "$selected_prefill" != "$preferred_prefill" || "$selected_decode" != "$preferred_decode" ]]; then
    echo "[inventory] requested pair unavailable; selected replacement: $selected_prefill / $selected_decode"
else
    echo "[inventory] selected: $selected_prefill / $selected_decode"
fi
echo "[inventory] source $OUT/selected.env before deployment"

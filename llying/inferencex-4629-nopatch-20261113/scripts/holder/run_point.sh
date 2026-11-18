#!/usr/bin/env bash
# Run a rendered point (holder_item.sh output) with the allocation's nodes reordered so that the head
# node, where srtctl starts etcd on 2379 next to the router and the first prefill, is one with 2379 free.
# Some nodes run Kubernetes' own etcd on 2379 (n04-33). srtctl takes the order from SLURM_NODELIST.
#   run_point.sh <point script>
set -uo pipefail
free=() busy=()
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    if srun --overlap -N1 -n1 -w "$n" bash -c 'ss -ltnH | grep -qE "[:.]2379 "'; then busy+=("$n"); else free+=("$n"); fi
done
order="$(printf '%s\n' "${free[@]}" "${busy[@]}" | grep . | paste -sd,)"
echo "run_point: srtctl node order $order (2379 taken on: ${busy[*]:-none})"
export SLURM_NODELIST="$order" SLURM_JOB_NODELIST="$order"
exec bash "$1"

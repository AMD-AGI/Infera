#!/usr/bin/env bash
# Every 2 s on this node: each SGLang/Infera process's memory-map count (the VMAs MADV_DONTFORK splits on every MR
# that ibv_reg_mr registers after Mooncake's ibv_fork_init), its pinned/locked memory, the node's vm.max_map_count,
# the MRs on each ionic device, and the host's free 2 MiB-or-larger blocks per NUMA node (/proc/buddyinfo). Runs until killed or for $1 seconds (default 3600).
#   vma_sampler.sh [seconds] > out.tsv
end=$(( $(date +%s) + ${1:-3600} ))
echo "# host $(hostname -s) max_map_count $(cat /proc/sys/vm/max_map_count) amdgpu $(cat /sys/module/amdgpu/version)"
printf 'time\tpid\tname\tmaps\tVmPin_kB\tVmLck_kB\tthreads\n'
while [ "$(date +%s)" -lt "$end" ]; do
    t=$(date -u +%H:%M:%S)
    for p in $(pgrep -f 'sglang|infera.engine' 2>/dev/null); do
        [ -r /proc/$p/maps ] || continue
        name=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-60)
        case "$name" in *vma_sampler*|*pgrep*) continue ;; esac
        maps=$(wc -l < /proc/$p/maps 2>/dev/null)
        pin=$(awk '/^VmPin/{print $2}' /proc/$p/status 2>/dev/null)
        lck=$(awk '/^VmLck/{print $2}' /proc/$p/status 2>/dev/null)
        thr=$(awk '/^Threads/{print $2}' /proc/$p/status 2>/dev/null)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$p" "$name" "$maps" "$pin" "$lck" "$thr"
    done
    printf '%s\tBUDDY\t%s\n' "$t" "$(awk '$4 == "Normal" {o9 = 0; for (i = 14; i <= NF; i++) o9 += $i * 2 ^ (i - 14); printf "node%s:o9+=%d(2MiB) o0=%s ", substr($2, 1, length($2) - 1), o9, $5}' /proc/buddyinfo)"
    printf '%s\tMR\t%s\n' "$t" "$(rdma resource show 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "mr") printf "%s=%s ", $2, $(i + 1)}')"
    sleep 2
done

#!/usr/bin/env bash
# Host telemetry for GPU-stall hunting, every INTERVAL s until killed: GPU busy % per card, the kernel's compaction /
# migration / THP counters, and KFD's per-process queue-eviction time. KFD adds an eviction's whole duration to
# stats_<gpu>/evicted_ms when the queues come back (e.g. after revalidating userptr memory that an MMU notifier
# invalidated), once per GPU the process has opened, so the max over GPUs is printed (pid:ms); a jump of N ms means
# an eviction of N ms just ended. Each KFD process is named once ("kfdpid <pid> <container> nspid=<pid in container> <comm> <cmdline>").
#   host_telemetry.sh OUT [INTERVAL=2]
OUT=${1:?} INTERVAL=${2:-2}
{
    for k in kernel/numa_balancing vm/compaction_proactiveness vm/compact_unevictable_allowed vm/extfrag_threshold \
             vm/zone_reclaim_mode vm/min_free_kbytes vm/watermark_scale_factor net/core/somaxconn \
             net/ipv4/tcp_max_syn_backlog net/ipv4/tcp_abort_on_overflow net/ipv4/tcp_syncookies net/ipv4/tcp_mem; do
        echo "sysctl $k=$(cat /proc/sys/$k 2>/dev/null)"
    done
    echo "thp enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled) defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag) shmem=$(cat /sys/kernel/mm/transparent_hugepage/shmem_enabled 2>/dev/null)"
    echo "cards: $(for c in /sys/class/drm/card*/device; do [[ -e $c/gpu_busy_percent ]] && echo -n "$(basename "$(dirname "$c")")=$(basename "$(readlink -f "$c")") "; done)"
    declare -A named
    while :; do
        ev=""
        for p in /sys/class/kfd/kfd/proc/[0-9]*; do
            [[ -d $p ]] || continue
            pid=${p##*/} m=0
            for f in "$p"/stats_*/evicted_ms; do [[ -r $f ]] && { v=$(cat "$f" 2>/dev/null || echo 0); ((v > m)) && m=$v; }; done
            ev+="$pid:$m,"
            if [[ -z ${named[$pid]:-} ]]; then
                named[$pid]=1
                cid=$(grep -oE '[0-9a-f]{64}' "/proc/$pid/cgroup" 2>/dev/null | head -1)
                name=$([[ -n $cid ]] && docker inspect --format '{{.Name}}' "$cid" 2>/dev/null | tr -d /)
                echo "kfdpid $pid ${name:-host} nspid=$(awk '/^NSpid/ {print $NF}' "/proc/$pid/status" 2>/dev/null) $(cat "/proc/$pid/comm" 2>/dev/null) $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)"
            fi
        done
        echo "== $(date -u +%T) busy=$(for c in /sys/class/drm/card*/device/gpu_busy_percent; do cat "$c" 2>/dev/null; done | tr '\n' ',')" \
            "evicted_ms=$ev" \
            "$(grep -E '^(compact_(stall|migrate_scanned|isolated|success|fail|daemon_wake)|pgmigrate_(success|fail)|numa_(pte_updates|hint_faults|pages_migrated)|thp_(fault_alloc|fault_fallback|collapse_alloc|migration_success|split_page)) ' /proc/vmstat | tr '\n' ' ')"
        sleep "$INTERVAL"
    done
} > "$OUT" 2>&1

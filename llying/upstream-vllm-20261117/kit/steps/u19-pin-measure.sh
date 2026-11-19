#!/usr/bin/env bash
# U19: pinning a ~1 TB PP4-layout TieringOffloadingSpec region from 4 workers, current rule (each registers the whole
# region) vs worker slots only, in the v0.34.0 image (pin_mmap_region is the same on main). The region files live in
# the host's /dev/shm: they are removed (as root) when the step ends, whatever happened, so no Shmem is left behind.
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/checks/u19_region_pin.py
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready v034 || exit 1
GB=${U19_GB:-1000}
clean() { docker run --rm -v /dev/shm:/dev/shm --entrypoint sh busybox:1.36 -c 'rm -f /dev/shm/vllm_offload_u19probe*'; }
trap clean EXIT
clean
avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
(( avail > GB * 93 / 100 + 400 )) || { echo "MemAvailable ${avail} GiB too low for a ${GB} GB region"; exit 1; }
for mode in slots full; do
    echo "== $mode ($GB GB)"
    run_vllm "up-u19-$mode" "$IMG_V034" "$(wt v0.34.0)" v0.34.0 \
        "python3 /checks/u19_region_pin.py $GB $mode 2>&1 | grep -E '^\{|Error|error|failed' | tail -12" | tee -a "$OUT/measure.jsonl"
    clean
    grep -E '^(MemAvailable|Shmem):' /proc/meminfo
done
grep -q '"mode": "slots".*"all_pinned": true' "$OUT/measure.jsonl"

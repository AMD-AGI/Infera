#!/usr/bin/env bash
# Node state after a failed step, for offline analysis once the job has exited:
# kernel log tail, GPU / host memory, our containers (state + last log lines), top processes.
#   diag.sh OUT_DIR
d=${1:?out dir}
mkdir -p "$d"
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
date -u '+%F %T' > "$d/when.txt"
grep -E '^(MemTotal|MemAvailable|Shmem|AnonPages|Mlocked|Unevictable|HugePages_Total):' /proc/meminfo > "$d/meminfo.txt"
img=$(docker image inspect -f '{{.Id}}' "$IMG_V034" > /dev/null 2>&1 && echo "$IMG_V034" || echo busybox:1.36)
docker run --rm --cap-add SYSLOG --network none --entrypoint sh "$img" -c 'dmesg -T | tail -300' > "$d/dmesg.txt" 2>&1
for t in amd-smi rocm-smi; do command -v "$t" > /dev/null && { "$t" 2>&1 | head -80 > "$d/$t.txt"; }; done
docker run --rm $(gpu_args) --entrypoint bash "$img" -c 'amd-smi metric -m 2>/dev/null | head -120 || rocm-smi --showmeminfo vram' \
    > "$d/gpu-mem.txt" 2>&1
docker ps -a --format '{{.Names}} {{.Status}} {{.Image}}' > "$d/containers.txt"
for c in $(docker ps -a --format '{{.Names}}' | grep -E '^(up-|vpd-)'); do
    docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} err={{.State.Error}}' "$c" \
        > "$d/$c.state" 2>&1
    docker logs --tail 400 "$c" > "$d/$c.log" 2>&1
done
ps -eo pid,stat,etimes,rss,args --sort=-rss | head -25 | cut -c1-250 > "$d/ps.txt"
exit 0

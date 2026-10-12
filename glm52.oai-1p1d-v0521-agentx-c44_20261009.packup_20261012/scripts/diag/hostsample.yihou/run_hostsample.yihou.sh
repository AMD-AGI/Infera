#!/usr/bin/env bash
# Host-side sampler for a profiling window (R041/R042 follow-up). CPU-only; never execs into the
# production containers.
#  - per-thread CPU of every glm52 container's processes: one sidecar per container sharing its pid
#    namespace (spur-authz denies --pid=host); no devices, no network, --cpus 1; reads /proc only.
#  - decode/prefill: 1-s sampler of busy sglang threads' last CPU + ticks (thread_cpuplace.yihou.py).
#  - per tick, ONE shim ssh: loadavg + mpstat, docker stats (glm52), docker ps of NON-yihou containers,
#    amd-smi metric (power, gfx/mem clocks, temps, gfx/umc activity, throttle) and amd-smi static --limit
#    (power cap).
#  - deadline-based: ticks start every INT s until DUR elapses, regardless of ssh latency.
# usage: run_hostsample.yihou.sh <node> <job> [duration_s=600] [interval_s=30] [tag=e2g] [image]
set -uo pipefail
node=${1:?node}; job=${2:?job}; DUR=${3:-600}; INT=${4:-30}; TAG=${5:-e2g}
IMG=${6:-infera-sglang:v0521-yihou-1010-mc4340}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/hostsample.yihou
out=$D/out.$TAG.$(date -u +%Y%m%d-%H%M%S).yihou; mkdir -p "$out"
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"
SC=yihou-hostsample-$TAG
ssh "$node" "docker ps --no-trunc --format '{{.ID}} {{.Names}}' | grep glm52" > "$out/containers.txt"
[[ -s $out/containers.txt ]] || { echo "[hostsample] no glm52 containers on $node" >&2; exit 1; }
while read -r cid cname; do
  ssh "$node" "docker rm -f $SC-$cname >/dev/null 2>&1; docker run -d --rm --name $SC-$cname --pid container:$cname \
    --network none --cpus 1 -v $D:$D --entrypoint python3 $IMG $D/thread_cpu.yihou.py $out/$cname $DUR $INT ALL:$cname" >> "$out/sidecar.id" 2>&1 < /dev/null
  case "$cname" in *-decode-*|*-prefill-*)
    ssh "$node" "docker rm -f $SC-place-$cname >/dev/null 2>&1; docker run -d --rm --name $SC-place-$cname --pid container:$cname \
      --network none --cpus 0.5 -v $D:$D --entrypoint python3 $IMG $D/thread_cpuplace.yihou.py $out/cpuplace.$cname.tsv $DUR" >> "$out/sidecar.id" 2>&1 < /dev/null ;;
  esac
done < "$out/containers.txt"
TICK='echo "=== $(date -u +%T)"
echo "--- load"; cat /proc/loadavg; mpstat 1 1 | tail -1
echo "--- dockerstats"; docker stats --no-stream --format "{{.Name}} {{.CPUPerc}} {{.MemUsage}}" | grep glm52
echo "--- foreign"; docker ps --format "{{.Names}}|{{.Image}}|{{.Status}}" | grep -v yihou
echo "--- amdsmi"; amd-smi metric -p -c -t -u --csv 2>/dev/null
echo "--- powercap"; amd-smi static --limit --csv 2>/dev/null
echo "--- nfs"; f=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/hostsample.yihou/nfsprobe.yihou.$$; a=$(date +%s%N); head -c 4096 /dev/zero | dd of=$f bs=4096 count=1 conv=fsync status=none; b=$(date +%s%N); stat $f >/dev/null; c=$(date +%s%N); rm -f $f; echo "write_fsync_us=$(( (b-a)/1000 )) stat_us=$(( (c-b)/1000 ))"
echo "--- nfsstats"; awk '"'"'/^device .* mounted on \/shared_nfs with/{f=1;next} /^device/{f=0} f&&/^[[:space:]]+(GETATTR|LOOKUP|ACCESS|READ|WRITE|COMMIT|READDIRPLUS):/{print $1,$2,$7,$8,$9}'"'"' /proc/self/mountstats
echo "--- lograte"; for n in $(docker ps --format "{{.Names}}" | grep -E "glm52.*-(decode|prefill)-0"); do echo "$n bytes_10s=$(docker logs --since 10s $n 2>&1 | wc -c)"; done'
t0=$(date +%s); tend=$(( t0 + DUR )); k=0
while (( $(date +%s) < tend )); do
  timeout $(( INT > 20 ? INT - 2 : 18 )) ssh "$node" "$TICK" >> "$out/ticks.log" 2>&1 || echo "=== tick $k ssh failed/timeout" >> "$out/ticks.log"
  now=$(date +%s); k=$(( (now - t0) / INT + 1 )); nxt=$(( t0 + k * INT ))  # skip missed ticks
  (( nxt > now )) && sleep $(( nxt - now ))
done
sleep 5; while read -r cid cname; do ssh "$node" "docker rm -f $SC-$cname $SC-place-$cname >/dev/null 2>&1" < /dev/null; done < "$out/containers.txt"
cat "$out"/*/threads.tsv > "$out/threads.tsv" 2>/dev/null
python3 "$D/summarize_hostsample.yihou.py" "$out" | tee "$out/summary.txt"
echo "[hostsample] $out"

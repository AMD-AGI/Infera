#!/usr/bin/env bash
# A2 py-spy records (leader-approved R046): at profiling minute ~12 (peak) and ~50 (decay), 20 s each,
# --nonblocking --idle --threads -r 100, on the 4 decode scheduler PIDs, from CPU-only ptrace sidecars.
set -uo pipefail
node=crsuse2-m2m-102; job=${1:-207495}; CHAIN=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/e2g.chain.log
OUT=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/decode_cost.yihou/pyspy.e2gA2.yihou; mkdir -p "$OUT"
IMG=infera-sglang:v0521-yihou-1010-mc4340; C=glm52-v0521-yihou-102-e2g-decode-0
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"
# wait for A2 profiling start (up to 3 h)
for i in $(seq 1 1080); do
  l=$(grep -a 'e2gA2' "$CHAIN" 2>/dev/null | grep -a 'profiling started' | tail -1)
  [[ -n "$l" ]] && break; sleep 10
done
[[ -n "${l:-}" ]] || { echo "no A2 start"; exit 1; }
t0=$(date -u -d "$(echo "$l" | grep -oE '^[0-9T:-]+Z')" +%s); echo "A2 start $l"
pids=$(ssh $node "docker run --rm --pid container:$C --network none --cpus 0.5 --entrypoint bash $IMG -c 'for p in /proc/[0-9]*; do t=\$(tr \"\\0\" \" \" < \$p/cmdline 2>/dev/null); case \"\$t\" in sglang::scheduler_DP*) echo \${p#/proc/}:\${t%% *};; esac; done'" < /dev/null)
echo "$pids" | tee "$OUT/pids.txt"
iosnap() {  # $1 = file: per-thread /proc io + cpu ticks of the 4 schedulers (read-only)
  ssh $node "docker run --rm --pid container:$C --network none --cpus 0.5 --cap-add SYS_PTRACE --entrypoint bash $IMG -c 'for pt in $pids; do p=\${pt%%:*}; for t in /proc/\$p/task/*; do echo \$p \${t##*/} \$(cut -d\" \" -f14,15 \$t/stat) \$(tr \"\\n\" \" \" < \$t/io); done; done'" > "$1" 2>&1 < /dev/null
}
rec() {  # $1 = tag
  iosnap "$OUT/$1.io0.txt"
  for pt in $pids; do
    p=${pt%%:*}; n=${pt#*::}
    ssh $node "docker run --rm --name yihou-pyspy-$1-$p --pid container:$C --network none --cpus 1 --cap-add SYS_PTRACE -v $OUT:$OUT \
      --entrypoint /opt/venv/bin/py-spy $IMG record --nonblocking --idle --threads -d 20 -r 100 -f raw -o $OUT/$1.$n.txt --pid $p" \
      > "$OUT/$1.$n.log" 2>&1 < /dev/null &
  done
  wait
  iosnap "$OUT/$1.io1.txt"
}
for spec in "peak:720" "late:3000"; do
  tag=${spec%%:*}; at=$(( t0 + ${spec#*:} )); now=$(date -u +%s)
  (( at > now )) && sleep $(( at - now ))
  echo "$(date -u +%T) record $tag"; rec "$tag"
done
echo "$(date -u +%T) done"; ls -la "$OUT"

#!/usr/bin/env bash
# e2i R1: py-spy record (20 s, nonblocking, --idle --threads, 100 Hz) on the 4 decode schedulers at profiling
# minute 15 (slow-state candidate) and 50, plus per-thread /proc io snapshots. CPU-only ptrace sidecars.
set -uo pipefail
node=crsuse2-m2m-102; CHAIN=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/e2i.chain.log; TAG=${1:-e2iR1}
OUT=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/decode_cost.yihou/pyspy.$TAG.yihou; mkdir -p "$OUT"
IMG=infera-sglang:v0521-yihou-1010-mc4340
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:207495"
for i in $(seq 1 1440); do l=$(grep -a "$TAG: " "$CHAIN" 2>/dev/null | tail -1); [[ -n "$l" ]] && break; sleep 10; done
echo "$(date -u +%T) $l"; case "$l" in *"profiling started"*) ;; *) echo "no start"; exit 1;; esac
t0=$(date -u -d "$(echo "$l" | grep -oE '^[0-9T:-]+Z')" +%s)
C=$(ssh $node "docker ps --format '{{.Names}}' | grep -E 'e2i-decode-0$' | head -1" < /dev/null); echo "container=$C"
pids=$(ssh $node "docker run --rm --pid container:$C --network none --cpus 0.5 --entrypoint bash $IMG -c 'for p in /proc/[0-9]*; do t=\$(tr \"\\0\" \" \" < \$p/cmdline 2>/dev/null); case \"\$t\" in sglang::scheduler_DP*) n=\${t%% *}; echo \${p#/proc/}:DP\${n#sglang::scheduler_DP};; esac; done'" < /dev/null)
pids=$(echo $pids | sed 's/_TP[0-9]*//g'); echo "pids=$pids" | tee "$OUT/pids.txt"
plist=$(for pt in $pids; do echo -n "${pt%%:*} "; done)
iosnap() { ssh $node "docker run --rm --pid container:$C --network none --cpus 0.5 --cap-add SYS_PTRACE --entrypoint bash $IMG -c 'for p in $plist; do for t in /proc/\$p/task/*; do echo \$p \${t##*/} \$(cut -d\" \" -f14,15 \$t/stat) \$(tr \"\\n\" \" \" < \$t/io); done; done'" > "$1" 2>&1 < /dev/null; }
for spec in "m15:900" "m50:3000"; do
  name=${spec%%:*}; at=$(( t0 + ${spec#*:} )); now=$(date -u +%s); (( at > now )) && sleep $(( at - now ))
  echo "$(date -u +%T) record $name"; iosnap "$OUT/$name.io0.txt"
  for pt in $pids; do p=${pt%%:*}; n=${pt#*:}
    ssh $node "docker run --rm --name yihou-pyspy-$TAG-$name-$p --pid container:$C --network none --cpus 1 --cap-add SYS_PTRACE -v $OUT:$OUT --entrypoint /opt/venv/bin/py-spy $IMG record --nonblocking --idle --threads -d 20 -r 100 -f raw -o $OUT/$name.$n.txt --pid $p" > "$OUT/$name.$n.log" 2>&1 < /dev/null &
  done; wait; iosnap "$OUT/$name.io1.txt"
done
echo "$(date -u +%T) done"; ls -la "$OUT"

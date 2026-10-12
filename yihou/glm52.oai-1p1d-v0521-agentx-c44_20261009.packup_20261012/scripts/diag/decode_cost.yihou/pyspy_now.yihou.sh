#!/usr/bin/env bash
# one-shot: record 20 s now on the 4 decode schedulers + io snapshots; $1 = tag
set -uo pipefail
tag=$1; node=crsuse2-m2m-102; OUT=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/decode_cost.yihou/pyspy.e2gA2.yihou
IMG=infera-sglang:v0521-yihou-1010-mc4340; C=glm52-v0521-yihou-102-e2g-decode-0
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:207495"
pids="30666:DP0 30808:DP1 30960:DP2 31113:DP3"
iosnap() { ssh $node "docker run --rm --pid container:$C --network none --cpus 0.5 --cap-add SYS_PTRACE --entrypoint bash $IMG -c 'for p in 30666 30808 30960 31113; do for t in /proc/\$p/task/*; do echo \$p \${t##*/} \$(cut -d\" \" -f14,15 \$t/stat) \$(tr \"\\n\" \" \" < \$t/io); done; done'" > "$1" 2>&1 < /dev/null; }
iosnap "$OUT/$tag.io0.txt"; date -u +%T.%N > "$OUT/$tag.t0"
for pt in $pids; do p=${pt%%:*}; n=${pt#*:}
  ssh $node "docker run --rm --name yihou-pyspy-$tag-$p --pid container:$C --network none --cpus 1 --cap-add SYS_PTRACE -v $OUT:$OUT --entrypoint /opt/venv/bin/py-spy $IMG record --nonblocking --idle --threads -d 20 -r 100 -f raw -o $OUT/$tag.$n.txt --pid $p" > "$OUT/$tag.$n.log" 2>&1 < /dev/null &
done; wait
iosnap "$OUT/$tag.io1.txt"; date -u +%T.%N > "$OUT/$tag.t1"
ls -la $OUT | grep $tag

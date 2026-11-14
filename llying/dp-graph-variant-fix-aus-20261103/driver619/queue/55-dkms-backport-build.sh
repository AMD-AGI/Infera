#!/bin/bash
# DRIVER619-20261113 section 5 plan A: does the amdgpu-dkms 6.19.14.31400100 source with the backport of torvalds/linux
# 9131cc85d0de build against the node's kernel? Build only (no install, no module load), in /tmp on the 6.19 node; the
# unpatched tree is built too if the patched one fails, to tell a patch error from a build-environment one.
OUT=$K/out/dkms-build-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
SRC=$K/src/6.19.14.31400100/usr/src/amdgpu-6.19.14-2377056.22.04
PATCH=$K/backport-9131cc85d0de-amdgpu-dkms-6.19.14.31400100.patch
[ -s "$PATCH" ] || PATCH=/perf_apps/liyingli/bench_agentx/kfd619/backport-9131cc85d0de-amdgpu-dkms-6.19.14.31400100.patch
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 bash -c "
set -x
uname -r; gcc --version | head -1; ls -d /lib/modules/\$(uname -r)/build
build() {  # dir label
    rm -rf \$1 && mkdir -p \$1 && cp -a $SRC/. \$1/ || return 1
    [ \$2 = patched ] && { (cd \$1 && patch -p1 < $PATCH) || return 1; }
    (cd \$1 && timeout 3600 make KERNELVER=\$(uname -r) -j64) > $OUT/\$2.log 2>&1
    rc=\$?
    echo \"\$2 rc=\$rc\"; ls -la \$1/amd/amdgpu/amdgpu.ko 2>/dev/null
    return \$rc
}
build /tmp/llying-dkms-patched patched || build /tmp/llying-dkms-orig orig
grep -n -E ' error|Error [0-9]' $OUT/*.log | head -20
rm -rf /tmp/llying-dkms-patched /tmp/llying-dkms-orig
" > "$OUT/run.log" 2>&1
tail -30 "$OUT/run.log"

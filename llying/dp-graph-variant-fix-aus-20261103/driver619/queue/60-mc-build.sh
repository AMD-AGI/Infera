#!/bin/bash
# DRIVER619-20261113 formal delivery: build Mooncake inside the production image: faae8dd4 + the HipGemAllocator backport
# (engine + allocator, what the image runs) on the 6.19 node, upstream main + the PR (also hip_gem_allocator_test,
# rdma_hip_kfd_pin_test and TENT) on the 6.16 node; then the derived image infera-sglang:v0522-fdbafb60-hipgem on both.
OUT=$K/out/mc-build-$(date -u +%m%dT%H%M)
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-build-latest
on() { local n=$1; shift; local x=$NODE616; [ "$n" = "$NODE616" ] && x=$NODE619; srun --jobid=$JOB --overlap -N1 -n1 -w $n --exclude=$x "$@"; }
IMG=infera-sglang:v0522-fdbafb60 TAR=/perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar
C=/home/liyingli/code/bench_agentx
B="docker image inspect $IMG >/dev/null 2>&1 || docker load -i $TAR >/dev/null; docker run --rm --entrypoint bash -v /perf_apps:/perf_apps -v /home/liyingli/code:/home/liyingli/code:ro $IMG $K/mc/build_mc.sh"
on $NODE619 bash -c "$B $C/mooncake-hip-gem-faae $OUT/faae engine" > "$OUT/faae.log" 2>&1 &
on $NODE616 bash -c "$B $C/mooncake-hip-gem $OUT/main tests" > "$OUT/main.log" 2>&1 &
wait
echo "== faae"; tail -20 "$OUT/faae.log"; echo "== main"; tail -20 "$OUT/main.log"
[ -s "$OUT/faae/engine.cpython-310-x86_64-linux-gnu.so" ] && [ -s "$OUT/faae/hip_gem_allocator.so" ] || { echo "faae build failed"; exit 1; }
for n in $NODE619 $NODE616; do
    on $n bash $K/mc/node_image.sh "$OUT/faae" $C/mooncake-hip-gem-faae/mooncake-integration/allocator.py > "$OUT/image-$n.log" 2>&1 &
done
wait
cat "$OUT"/image-*.log

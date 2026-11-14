#!/bin/bash
# DRIVER619-20261113 formal delivery: build Mooncake inside the production image: faae8dd4 + the HipGemAllocator backport
# (engine + allocator, what the image runs) on the 6.19 node, upstream main + the PR (also hip_gem_allocator_test,
# rdma_hip_kfd_pin_test and TENT) on the 6.16 node; then the derived image infera-sglang:v0522-fdbafb60-hipgem on both.
# Sources are clones on /perf_apps (kfd619/mc/src-*): dockerd cannot read the root-squashed /home.
OUT=$K/out/mc-build-$(date -u +%m%dT%H%M)
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-build-latest
on() { local n=$1; shift; local x=$NODE616; [ "$n" = "$NODE616" ] && x=$NODE619; srun --jobid=$JOB --overlap -N1 -n1 -w $n --exclude=$x "$@"; }
IMG=infera-sglang:v0522-fdbafb60 TAR=/perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar
B="docker image inspect $IMG >/dev/null 2>&1 || docker load -i $TAR >/dev/null; docker run --rm --entrypoint bash -v /perf_apps:/perf_apps $IMG $K/mc/build_mc.sh"
on $NODE619 bash -c "$B $K/mc/src-faae $OUT/faae engine" > "$OUT/faae.log" 2>&1 &
on $NODE616 bash -c "$B $K/mc/src-main $OUT/main tests" > "$OUT/main.log" 2>&1 &
wait
echo "== faae"; tail -25 "$OUT/faae.log"; echo "== main"; tail -25 "$OUT/main.log"
[ -s "$OUT/faae/pkg/hip_gem_allocator.so" ] && ls "$OUT"/faae/pkg/engine.cpython-*.so >/dev/null || { echo "faae build failed"; exit 1; }
for n in $NODE619 $NODE616; do
    on $n bash $K/mc/node_image.sh $IMG infera-sglang:v0522-fdbafb60-hipgem "$OUT/faae/pkg" $TAR > "$OUT/image-$n.log" 2>&1 &
done
wait
cat "$OUT"/image-*.log

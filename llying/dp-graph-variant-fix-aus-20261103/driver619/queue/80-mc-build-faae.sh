#!/bin/bash
# DRIVER619-20261113 formal delivery (60 again): faae8dd4 + the HipGemAllocator backport (install destination fixed for
# faae8dd4) built in the production image on the 6.19 node, then infera-sglang:v0522-fdbafb60-hipgem on both nodes.
OUT=$K/out/mc-build-$(date -u +%m%dT%H%M)
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-build-latest
on() { local n=$1; shift; local x=$NODE616; [ "$n" = "$NODE616" ] && x=$NODE619; srun --jobid=$JOB --overlap -N1 -n1 -w $n --exclude=$x "$@"; }
IMG=infera-sglang:v0522-fdbafb60 TAR=/perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar
on $NODE619 bash -c "docker image inspect $IMG >/dev/null 2>&1 || docker load -i $TAR >/dev/null; docker run --rm --entrypoint bash -v /perf_apps:/perf_apps $IMG $K/mc/build_mc.sh $K/mc/src-faae $OUT/faae engine" > "$OUT/faae.log" 2>&1
tail -25 "$OUT/faae.log"
[ -s "$OUT/faae/pkg/hip_gem_allocator.so" ] && ls "$OUT"/faae/pkg/engine.cpython-*.so >/dev/null || { echo "faae build failed"; exit 1; }
for n in $NODE619 $NODE616; do
    on $n bash $K/mc/node_image.sh $IMG infera-sglang:v0522-fdbafb60-hipgem "$OUT/faae/pkg" $TAR > "$OUT/image-$n.log" 2>&1 &
done
wait
cat "$OUT"/image-*.log

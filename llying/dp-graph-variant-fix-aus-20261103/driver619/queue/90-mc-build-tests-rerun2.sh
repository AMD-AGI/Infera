#!/bin/bash
# DRIVER619-20261113, 11-17: 81 again for the two PRs (tests only) after RegistrationKeepsKfdFreeVram stopped comparing
# free VRAM across the first stream and kernel (HIP runtime state stays in VRAM; 83 failed it by 146 MiB on both PRs).
# PR 1 80a3cab0, PR 2 be8a8047. 88 itself failed: n02-25's /data (docker's root) is full, so the build tree goes to the
# host's /tmp (root file system) instead of the container's layer.
OUT=$K/out/mc-tests-$(date -u +%m%dT%H%M)-rerun2
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-tests-latest
cp $K/mc/src-main/REV "$OUT/main.rev"; cp $K/mc/src-tent/REV "$OUT/tent.rev"
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c "
docker image inspect e1ab-sglang:1112 >/dev/null 2>&1 ||
    { docker image inspect e1ab-sglang:1109 >/dev/null 2>&1 &&
        (cd $K/nightly/image && docker build -t e1ab-sglang:1112 -f Dockerfile.from1109 .); } ||
    (cd $K/nightly/image && DOCKER_CONFIG=\$(mktemp -d) timeout 5400 docker build --pull -t e1ab-sglang:1112 .) || exit 1
for v in main:tests tent:tests; do
    n=\${v%%:*} m=\${v#*:} s=\${v%%:*}; [ \$n = engine ] && s=main
    echo \"######## \$n\"
    docker run --rm --entrypoint bash -v /perf_apps:/perf_apps -v /tmp/llying-kfd619-mcb:/tmp e1ab-sglang:1112 $K/mc/build_mc.sh $K/mc/src-\$s $OUT/\$n \$m
    echo \"build \$n rc \$?\"
done
docker run --rm --entrypoint bash -v /tmp/llying-kfd619-mcb:/x e1ab-sglang:1112 -c 'rm -rf /x/mcb'
" > "$OUT/run.log" 2>&1
tail -60 "$OUT/run.log"

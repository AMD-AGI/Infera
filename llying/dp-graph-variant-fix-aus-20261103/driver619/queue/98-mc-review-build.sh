#!/bin/bash
# DRIVER619-20261113, 11-17: builds for the review round of kvcache-ai/Mooncake#5584 / #5587, in the nightly image on
# the 6.19 node (n02-25, inside the user's hold job 36271):
#   base  upstream 504f2ae4, the PRs' base: only the existing tests the PRs must not change (regress)
#   main  #5584 with the review fixes (3bfe2416): GEM tests, hip_gem_pin_test and the same existing tests
#   tent  #5587 on top (d64161e4)
# n02-25's /data (docker's root) is full, so the build tree goes to the host's /tmp, removed afterwards.
OUT=$K/out/mc-review-$(date -u +%m%dT%H%M)
mkdir -p "$OUT" && ln -sfn "$OUT" $K/out/mc-review-latest
for v in base main tent; do cp $K/mc/src-$v/REV "$OUT/$v.rev"; done
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash -c "
for v in base:regress main:tests tent:tests; do
    n=\${v%%:*} m=\${v#*:}
    echo \"######## \$n\"
    docker run --rm --entrypoint bash -v /perf_apps:/perf_apps -v /tmp/llying-kfd619-mcb:/tmp e1ab-sglang:1112 \
        $K/mc/build_mc.sh $K/mc/src-\$n $OUT/\$n \$m
    echo \"build \$n rc \$?\"
done
docker run --rm --entrypoint bash -v /tmp:/x e1ab-sglang:1112 -c 'rm -rf /x/llying-kfd619-mcb'
" > "$OUT/run.log" 2>&1
grep -E "^######## |^build |error|FAILED" "$OUT/run.log" | head -40
for v in base main tent; do echo "$v: $(ls $OUT/$v | tr '\n' ' ')"; done

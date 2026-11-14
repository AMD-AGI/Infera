#!/bin/bash
# DRIVER619-20261113 formal delivery: SGLang's new CPU unit test (and the neighbouring EFA allocator test) on the PR
# branch (a copy on /perf_apps, kfd619/sglang-pr), in the production image; if upstream main does not import in that
# environment, also against the image's SGLang with the ported utils.py (kfd619/sglang-native).
OUT=$K/out/sglang-unit-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
T=test/registered/unit/disaggregation
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE616 --exclude=$NODE619 docker run --rm --network none --entrypoint bash \
    -v /perf_apps:/perf_apps -e PYTHONDONTWRITEBYTECODE=1 infera-sglang:v0522-fdbafb60 -c "
cd $K/sglang-pr
echo '== PR branch'; PYTHONPATH=python python3 -m pytest -q -p no:cacheprovider $T/test_mooncake_amd_gem_pool.py $T/test_mooncake_efa_allocator.py 2>&1 | tail -15
echo '== image SGLang + ported utils.py'
S=\$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')
cp $K/sglang-native/utils.py \$S/srt/disaggregation/mooncake/utils.py
grep -v -E 'ci_register|register_cpu_ci' $T/test_mooncake_amd_gem_pool.py > /tmp/test_amd_gem.py
python3 -m pytest -q -p no:cacheprovider /tmp/test_amd_gem.py 2>&1 | tail -15
" > "$OUT/run.log" 2>&1
cat "$OUT/run.log"

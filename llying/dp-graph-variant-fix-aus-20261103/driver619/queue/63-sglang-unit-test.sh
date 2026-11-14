#!/bin/bash
# DRIVER619-20261113 formal delivery: SGLang's new CPU unit test (and the neighbouring EFA allocator test) against the
# PR branch; if upstream main does not import in the image's environment, against the image's SGLang with the ported
# utils.py (kfd619/sglang-native) and without the CI registration import.
OUT=$K/out/sglang-unit-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
x=$NODE619
T=test/registered/unit/disaggregation
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE616 --exclude=$x docker run --rm --network none --entrypoint bash \
    -v /home/liyingli/code:/home/liyingli/code:ro -v /perf_apps:/perf_apps -e PYTHONDONTWRITEBYTECODE=1 \
    -v $K/sglang-native/utils.py:/native/utils.py:ro infera-sglang:v0522-fdbafb60 -c "
cd /home/liyingli/code/bench_agentx/sglang-amd-gem
echo '== PR branch'; PYTHONPATH=python python3 -m pytest -q -p no:cacheprovider $T/test_mooncake_amd_gem_pool.py $T/test_mooncake_efa_allocator.py 2>&1 | tail -15
echo '== image SGLang + ported utils.py'
S=\$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')
cp /native/utils.py \$S/srt/disaggregation/mooncake/utils.py
grep -v -E 'ci_register|register_cpu_ci' $T/test_mooncake_amd_gem_pool.py > /tmp/test_amd_gem.py
python3 -m pytest -q -p no:cacheprovider /tmp/test_amd_gem.py 2>&1 | tail -15
" > "$OUT/run.log" 2>&1
cat "$OUT/run.log"

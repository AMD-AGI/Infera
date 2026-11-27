#!/usr/bin/env bash
# Which patch breaks the regression files: each file separately (an abort must not hide the others) on
#   stock (nightly base, de7cf7f8 tests) | C0 (C0 image, base/nightly-de7cf7f8-C0 tests) | C1 (C1 image, C0 + #15-17 tests)
#   test_bisect.sh [GPU]
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
gpu=${1:-4}
out=$W/results/test-bisect-$(hostname -s)
mkdir -p "$out"
bash "$(cd "$(dirname "$0")/.." && pwd)/docker/build.sh" series-C0.txt infera-vllm:nightly-de7cf7f8-C0 > "$out/build-C0.log" 2>&1 || { echo "C0 build failed"; exit 1; }
gids=(); for g in video render; do gids+=(--group-add "$(getent group $g | cut -d: -f3)"); done
U=/perf_apps/liyingli/bench_agentx/upstream
for set in "stock $NIGHTLY_BASE $U/vllm-nightly-de7cf7f8" "C0 infera-vllm:nightly-de7cf7f8-C0 $U/vllm-C0-tests" "C1 infera-vllm:nightly-de7cf7f8-C1 $U/vllm-C1-tests"; do
    read -r tag img T <<< "$set"
    for f in tests/kernels/moe/test_naive_dp_ep_dbo.py tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py \
             tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py; do
        log=$out/$tag-$(basename "$f" .py).log
        docker run --rm --name "vpd-test-$gpu" --network host --device /dev/kfd --device /dev/dri "${gids[@]}" \
            --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES="$gpu" -e VLLM_ROCM_USE_AITER=1 -v "$T:$T:ro" -w /tmp \
            --entrypoint bash "$img" -c "pip install -q pytest tblib pytest-asyncio pytest-timeout > /dev/null 2>&1; \
            python3 -m pytest -q --tb=short -rfE --import-mode=importlib -p no:cacheprovider $T/$f" > "$log" 2>&1
        echo "$(ts) $tag $(basename "$f"): $(grep -a -E '^[0-9]+ (passed|failed)|^=+ .*(passed|failed|error).* =+$|Fatal Python error' "$log" | tail -1)"
    done
done | tee "$out/summary.txt"

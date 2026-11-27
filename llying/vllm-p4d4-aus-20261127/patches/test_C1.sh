#!/usr/bin/env bash
# Unit tests of the C1 patch series inside the C1 image on one GPU of this node, and the same regression file on the
# stock nightly image with the stock test file (baseline: which failures the patches introduce).
# Test files: /perf_apps/.../upstream/vllm-C1-tests (base/nightly-de7cf7f8-C0 + #15 #16 #17) and
# /perf_apps/.../upstream/vllm-nightly-de7cf7f8 (clean de7cf7f8). --import-mode=importlib: the installed vllm is tested.
#   test_C1.sh [GPU]
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
gpu=${1:-4}
out=$W/results/test-C1-$(hostname -s)
mkdir -p "$out"
gids=(); for g in video render; do gids+=(--group-add "$(getent group $g | cut -d: -f3)"); done
pt() {   # image tests_root log files...
    local img=$1 T=$2 log=$3; shift 3
    local files=""; for f in "$@"; do files+=" $T/$f"; done
    docker run --rm --name "vpd-test-$gpu" --network host --device /dev/kfd --device /dev/dri "${gids[@]}" \
        --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES="$gpu" -e VLLM_ROCM_USE_AITER=1 -v "$T:$T:ro" -w /tmp \
        --entrypoint bash "$img" -c "pip install -q pytest tblib pytest-asyncio pytest-timeout > /dev/null 2>&1; \
        python3 -m pytest -q --tb=short -rfE --import-mode=importlib -p no:cacheprovider $files" > "$out/$log" 2>&1
    echo "$(ts) $log: $(grep -a -E '^=+ .*(passed|failed|error).* =+$' "$out/$log" | tail -1)"
}
C=/perf_apps/liyingli/bench_agentx/upstream/vllm-C1-tests S=/perf_apps/liyingli/bench_agentx/upstream/vllm-nightly-de7cf7f8
pt infera-vllm:nightly-de7cf7f8-C1 $C new-tests.log tests/kernels/moe/test_naive_dp_ep_mxfp4_dispatch.py \
    tests/kernels/moe/test_naive_dp_ep_mxfp4_dispatch_pack.py tests/kernels/attention/test_rocm_triton_indexer_topk.py
pt infera-vllm:nightly-de7cf7f8-C1 $C regression.log tests/kernels/moe/test_naive_dp_ep_dbo.py \
    tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py
pt "$NIGHTLY_BASE" $S baseline-stock.log tests/kernels/moe/test_naive_dp_ep_dbo.py \
    tests/kernels/attention/test_rocm_paged_mxfp4_indexer.py tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py

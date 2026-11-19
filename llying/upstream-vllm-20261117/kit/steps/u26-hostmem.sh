#!/usr/bin/env bash
# U26: host memory taken by 2 x 20 GB CPU offload tensors, torch's pinned allocator vs the PR's pinned_host_tensor
# (exact size, is_pinned, GPU round trip, release), in the v0.34.0 image with the PR's files.
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u26-rocm-exact-pin rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/checks/u26_host_mem.py
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready v034 || exit 1
run_vllm up-u26-hostmem "$IMG_V034" "$(wt u26-rocm-exact-pin)" v0.34.0 \
    "grep -n -E 'hipHostMalloc|hipHostAlloc|hipHostFree' \$(python3 -c 'import os; print(os.environ.get(\"ROCM_PATH\", \"/opt/rocm\"))')/include/hip/hip_runtime_api.h | grep -iE 'deprecat|^[0-9]+:hipError_t' | head -8;
     python3 /checks/u26_host_mem.py 20 2" | tee "$OUT/hostmem.jsonl"
grep -q '"pass": true' "$OUT/hostmem.jsonl"

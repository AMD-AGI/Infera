#!/bin/bash
# Local hook + analysis test on a16-19 (MI300X): Qwen2.5-0.5B, vLLM DP2 (GPUs 6,7), image 9c1d45c2, calib window of 30 steps.
set -uo pipefail
source /perf_apps/yihou/dsv41_kf_20261120/pd/scripts/env.sh
LCP=/perf_apps/yihou/dsv41_kf_20261120/calib-build-a1619
T=/perf_apps/yihou/dsv41_kf_20261120/hooktest/$(date -u +%m%dT%H%M); mkdir -p "$T/calib/eng0" && chmod 1777 "$T" "$T/calib/eng0"
docker run -d --rm --name yihou-dsv41kf-hooktest --network host --ipc host --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined -v /perf_apps:/perf_apps -e HIP_VISIBLE_DEVICES=6,7 \
  -e CALIB_ENABLE=1 -e CALIB_SDK_BUILD=$LCP/build -e CALIB_TRACE_DIR=$T/calib/eng0 -e CALIB_STEPS=30 -e CALIB_MAX_WINDOW_S=30 \
  -e ROCPROFILER_REGISTER_LIBRARY=$LCP/build/lib/librocprofiler-sdk.so \
  -e ROCP_TOOL_LIBRARIES=$LCP/build/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so \
  -e ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT -e ROCPROFILER_QUEUE_INTERPOSITION=0 \
  -e ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1 -e ROCPROFILER_TOOL_DISABLE_PERFETTO=1 -e HF_HUB_OFFLINE=1 \
  --entrypoint bash "$IMAGE" -c "export LD_LIBRARY_PATH=$LCP/build/lib:\$LD_LIBRARY_PATH PYTHONPATH=$CALIB_HOOK; \
    vllm serve /perf_apps/data/models/Qwen2.5-0.5B-Instruct --port 31999 --data-parallel-size 2 --max-model-len 4096 \
    --gpu-memory-utilization 0.3 > $T/server.log 2>&1"
echo "$T"

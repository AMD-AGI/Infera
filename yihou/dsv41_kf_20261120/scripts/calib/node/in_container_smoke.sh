#!/bin/bash
# INSIDE a 1106-image container with one visible GPU. Runs:
#   1a torch CUDA-graph probe, mode window  (server-default path)   -> validate_probe.py  (gating)
#   1b torch CUDA-graph probe, mode capture (capture attribution after stop/start)       (informative)
#   1c team native HIP probes (hipcc module + memcpy graph probes, run_native.sh)       (informative)
set -uo pipefail
: "${SDK:?}"
: "${CP:?}"
: "${SRC:?}"
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-/opt/rocm/lib}
export ROCPROFILER_REGISTER_LIBRARY=$SDK/lib/librocprofiler-sdk.so
export ROCP_TOOL_LIBRARIES=$SDK/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so
export ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT
export ROCPROFILER_QUEUE_INTERPOSITION=0
export ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1
mkdir -p "$OUT"
rc=0
for mode in window capture; do
  echo "=== probe $mode"
  ROCPROFILER_TOOL_OUTPUT_FILE=$OUT/trace-$mode.json timeout 300 python3 "$CP/smoke/probe_torch_graph.py" \
    --mode $mode --sdk-build "$SDK" --out "$OUT" > "$OUT/probe-$mode.log" 2>&1
  echo "probe rc=$?"; tail -5 "$OUT/probe-$mode.log"
  timeout 300 python3 "$CP/smoke/validate_probe.py" "$OUT" $mode > /dev/null 2>&1; v=$?
  python3 -c "import json;d=json.load(open('$OUT/validate-$mode.json'));print('$mode', d['verdict'], {k:v.get('pass',v) for k,v in d.items() if k[:1] in 'GI'})" 2>&1
  [ "$mode" = window ] && [ $v -ne 0 ] && rc=1
done
echo "=== native HIP probes (team run_native.sh; validated by the team on ROCm 10 only)"
SDK_BUILD=$SDK GRAPH_TEST_ARCH=gfx950 timeout 600 bash "$SRC/tests/graph-attribution-experiments/run_native.sh" "$OUT/native" \
  > "$OUT/native.log" 2>&1; echo "native rc=$?"; tail -8 "$OUT/native.log"
echo "=== stock comparison: image rocprofv3 on the same probe (kernel names + durations only)"
env -u ROCP_TOOL_LIBRARIES -u ROCPROFILER_REGISTER_LIBRARY LD_LIBRARY_PATH=/opt/rocm/lib timeout 300 \
  /opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv -d "$OUT/stock" -- \
  python3 -c "import torch;x=torch.randn(512,1024,device='cuda',dtype=torch.bfloat16);w=torch.randn(1024,2048,device='cuda',dtype=torch.bfloat16)
for _ in range(20): (torch.relu(x@w+1.0))
torch.cuda.synchronize()" > "$OUT/stock.log" 2>&1; echo "stock rocprofv3 rc=$?"
exit $rc

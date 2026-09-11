# Original packet-capture mode: baseline and incomplete diagnostic traces

`baseline_c32/metrics.json` is the valid, unprofiled baseline with the original ROCm packet-capture behavior.

The `profile_01` and `profile_02` traces omit the target-verify graph kernels. They are retained only as diagnostic evidence and must **not** be used for a complete operator distribution. The initial `TORCH_PROFILER_HIP_GRAPH_TRACING=1` setting did not fix this build's behavior.

Use [the complete captures](../20260910_n04_c32_complete) and [the final report](../../REPORT.md) for operator analysis. Those captures keep CUDA Graph enabled and set `DEBUG_CLR_GRAPH_PACKET_CAPTURE=false`.

# Reverse server validation

User requested reversal of the completed TP4EP4 internal sweep, not more numerical-path investigation.

1. Preserve the two sealed reference kits. Use the server kit's original three source overlays unchanged, fixed image/model, EAGLE5/6/topk1,FP8KV,FlyDSL,AITER and fused optimizations. New task-specific launch harness only.
2. Run actual streamed fake-prefill client with ISL70000/OSL10000,128 requests,16 warmups(output32),concurrency4/8/16/20/24 and DPAoff/on. Match global admission and local graph capacity using fixed source evidence. Keep overlap enabled as in the server reference.
3. Run off/C16 first. Verify readiness,successful128 requests,1280000 output tokens,per-request lengths,TP/EP/DP and parameter values before continuing. Retain original logs and failures. Stop owned successful server before next launch; no other cleanup.
4. Compare native mean/P50/P90TPOT and total output throughput with internal fixed-batch meanTPOT/throughput. Report relative differences,acceptance,backend fallback,rolling refill/tail,zero fake versus random internal prefix and node differences. No numerical correctness conclusion from timing similarity.

Resource: existing131080/node271,GPUs0-3. No requests for resources, no other nodes. Main code and all artifacts here; previous code unchanged. Initial HEAD4b2b3bc. Full sweep expected several hours because each point now generates1.28M outputs,notC*10000.

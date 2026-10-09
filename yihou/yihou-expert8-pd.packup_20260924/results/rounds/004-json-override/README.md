# Round004 — valid override and first physical load

Hypothesis: correcting inherited JSON source order permits target8 model load.
Changed: isolated config reassigns JSON_MODEL_OVERRIDE_ARGS after source; no model patch change.
Command: scripts/rdma_probe.yihou.py --out rounds/004-json-override/rdma, then scripts/launch.yihou.sh <absolute round004>.
Result so far: RDMA8/8 byte checks pass. Both model legs load successfully.75 MoE layers log routed8/topk8/EP4/local2. Each rank iterator loaded5019/skipped112391, retained44848135008 bytes (matches independent metadata accounting).
Measured post-weight allocations: prefill31.43GB/rank, decode31.44GB/rank. KV: prefill3990400 tokens/205.23GB per rank; decode4097408 tokens/210.74GB per rank. Runtime GB units are reported as-is.
CUDA graph/JIT still in progress. Do not mark full EP4 forward/smoke/benchmark passed before completion.
Evidence: launch/server-logs; weight-kv-snapshot; live monitor remains under round002/monitor, tracking same container names across retries.

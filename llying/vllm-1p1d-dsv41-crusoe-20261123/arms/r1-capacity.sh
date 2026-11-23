# Round 1: capacity settings from the B300 comparison (vs-b300-dynamo-828k/ANALYSIS.zh-CN.md, section 8 A1-A3, A5).
# The numeric path is unchanged; graph modes change, so the run keeps the accuracy gate.
#   P and D at gpu-memory-utilization 0.95 (B300 uses 0.95): ~+14 GiB of KV per rank over 0.9.
#   Decode admits KV-ready requests every 2 steps instead of every 11 (B300 decode: 2).
export PREFILL_GMU=0.95 DECODE_GMU=0.95 D_PSI=2
# Prefill emits one token per request and runs its queue in ~16k-token steps (eager): drop the FULL decode graphs,
# whose pool is reserved out of the KV budget the prefix cache lives in.
P_ARGS_X=(--compilation-config '{"decode_cudagraph_capture_request_sizes":[]}')
# Decode: FULL graphs for uniform decode steps up to 288 requests (6 tokens each: 1 + 5 drafts) and no PIECEWISE
# graphs. At c704-c768 a step carries ~190-260 requests per rank, past the 168 that 1024 tokens covered.
D_ARGS_X=(--max-cudagraph-capture-size 1728 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')

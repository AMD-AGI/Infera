# B4 vs RB: lower TTFT and matched P time, with the same nodes and KV capacities.
export DSA_PREFILL_BACKEND=triton
export DSA_DECODE_BACKEND=triton
export BASELINE_RUN=$TRACE_RUNTIME/runs/campaign-b4-triton
export BASELINE_LABEL=B4_Triton_selected_stack

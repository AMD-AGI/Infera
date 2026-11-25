# R16: r4-hipvmm + prefill steps of 32,768 tokens + no CUDA graphs on prefill (EXPERIMENT 4.11, throughput candidates).
# 32k steps amortise ~20 ms of fixed cost per step (R12 measured +2.9% prefill compute per rank) but shrink prefill's KV
# by 31%; prefill runs 97-99% of its steps without a graph yet reserves 10.3-12.6 GiB per rank for graphs, so dropping
# them (keeping torch.compile) returns ~13% of KV. Decode is unchanged.
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r4-hipvmm.sh
P_ARGS_X+=(--max-num-batched-tokens 32768 --compilation-config '{"cudagraph_mode":"NONE"}')

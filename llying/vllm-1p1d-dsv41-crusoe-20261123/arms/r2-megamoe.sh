# Round 2 candidate: round 1 plus expert parallelism with AITER MegaMoE on both legs, the ROCm counterpart of the
# B300 recipe's DEP + DeepGEMM MegaMoE. Replaces the AllGather+ReduceScatter of every MoE layer over the single xGMI
# link between a leg's two GPUs by MoRI's intra-node dispatch / combine fused with the expert GEMMs
# (vllm-project/vllm#59685: gfx950, DP + EP, TP 1, no EPLB). Numerics change: run with GATE=1.
# MoRI's symmetric heap (MORI_SHMEM_HEAP_SIZE, 8 GiB per rank by default) comes out of the KV budget.
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r1-capacity.sh
P_ARGS_X+=(--enable-expert-parallel --all2all-backend mori --moe-backend aiter_mega_moe)
D_ARGS_X+=(--enable-expert-parallel --all2all-backend mori --moe-backend aiter_mega_moe)

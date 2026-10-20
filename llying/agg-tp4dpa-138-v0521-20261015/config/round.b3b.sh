# b3b (crsuse2-m2m-138): a3 without DP attention -- plain TP4, one scheduler. Under sustained
# concurrency DP attention + decode CUDA graphs garble outputs while plain TP4 is correct
# (llying/mix-v0522-correctness-20261019); this measures what the correct layout costs at C48.
# MLA's latent KV is replicated across TP ranks, so the GPU pool holds about 30% of DP4's tokens.
# A TP4 step splits every token across the 4 GPUs, so a 32768-token chunk costs each GPU what an
# 8192-token chunk costs one DP4 rank: chunk and max-prefill both 32768. The delayer only acts
# across DP ranks.
source "$AGG/config/round.a3.sh"
MIXED_DP=1 MIXED_DPA=0 MIXED_DP_LM_HEAD=0
MIXED_CHUNK_SIZE=32768 MIXED_MAX_PREFILL_TOKENS=32768
MIXED_PREFILL_DELAYER=0

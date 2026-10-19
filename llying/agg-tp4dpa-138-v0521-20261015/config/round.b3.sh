# b3 (crsuse2-m2m-138): a3 without DP attention -- plain TP4, one scheduler. Under sustained
# concurrency DP attention + decode CUDA graphs garble outputs while plain TP4 is correct
# (llying/mix-v0522-correctness-20261019); this measures what the correct layout costs at C48.
# MLA's latent KV is replicated across TP ranks, so the GPU pool holds a quarter of DP4's tokens.
# chunk 8192 keeps a prefill step as large as one DP4 rank's; the delayer only acts across DP ranks.
source "$AGG/config/round.a3.sh"
MIXED_DP=1 MIXED_DPA=0 MIXED_DP_LM_HEAD=0
MIXED_CHUNK_SIZE=8192
MIXED_PREFILL_DELAYER=0

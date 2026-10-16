# t8a: the whole node as one TP8 + DP8-attention worker, with a4's write_back + mem 0.92.
# a4 C64 (TP4) recovered the prefix hit (95.6% of 96.2%) but stayed at the C48 level: decode step time
# per rank is the limit (~134 ms per step at ~11 running incl. prefill stalls, vs 66.7 ms for the 3P1D
# TP8/DP8 decode at 11.5 running). At TP8 the weights per GPU halve (121 -> ~61 GB): faster weight-bound
# decode steps and ~+50% KV per GPU (~3.7M tokens per rank at 0.92).
# Per-rank settings as in TP4: chunk 8192 per rank (the total is split over DP ranks), 32 running per
# rank. 8 host pools: 200 GB each (+ ~46 GB DSA indexer) = ~2 TB of the node's 2.75 TB.
GPUS_mixed_0=0,1,2,3,4,5,6,7
MIXED_TP=8 MIXED_DP=8
MIXED_CHUNK_SIZE=65536
MIXED_MAX_RUNNING=256
MIXED_HICACHE_SIZE=200
MIXED_HICACHE_WRITE_POLICY=write_back
MIXED_MEM_FRACTION=0.92

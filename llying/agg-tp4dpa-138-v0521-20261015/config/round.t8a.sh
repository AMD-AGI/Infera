# t8a: the whole node as one TP8 + DP8-attention worker. a3 was bound by GPU KV per rank; at TP8 the
# weights per GPU halve (121 -> ~61 GB), leaving ~65 GB more per GPU for KV at the same mem fraction.
# Per-rank settings as in TP4: chunk 8192 per rank (the total is split over DP ranks), 32 running per
# rank. 8 host pools: 200 GB each (+ ~46 GB DSA indexer) = ~2 TB of the node's 2.75 TB.
GPUS_mixed_0=0,1,2,3,4,5,6,7
MIXED_TP=8 MIXED_DP=8
MIXED_CHUNK_SIZE=65536
MIXED_MAX_RUNNING=256
MIXED_HICACHE_SIZE=200
MIXED_HICACHE_WRITE_POLICY=write_through

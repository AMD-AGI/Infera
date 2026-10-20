# b1 (2026-10-19): AUS 2P(TP4)1D(TP8) write_through baseline, the counterpart of 3P1D t5b-2p1d (C176 = 28,900).
# 3P1D best settings (config/best.sh + t6-mem092 decode) on this shape:
#   prefill: mem 0.88; HiCache write_through 250 GB/rank on both instances (eight ranks share one 3 TB node;
#            t5b had 400 GB/rank on 135 and 200 GB/rank on 137);
#   decode : mem 0.92, 4 tokenizer workers;
#   prefill-1 is the second prefill on its node, the 3P1D prefill-2 hang case: HSA_NO_SCRATCH_RECLAIM=1 as in the
#   3P1D A/B (prefill-0 keeps 0).
PREFILL_MEM_FRACTION=0.88
HICACHE_SIZE_prefill_0=250
HICACHE_SIZE_prefill_1=250
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_0=512
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_1=512
EXTRA_ENV_prefill_1="HSA_NO_SCRATCH_RECLAIM=1"
DECODE_MEM_FRACTION=0.92
EXTRA_ARGS_decode_0="--tokenizer-worker-num 4"

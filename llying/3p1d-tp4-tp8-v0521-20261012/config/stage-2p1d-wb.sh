# Stage t7 overlay (2026-10-15): 2P(TP4)1D with prefill HiCache write_back (PLAN-TUNING.zh-CN.md, 10-15 08:40).
#   Same shape and host sizes as t5b-2p1d (C176 = 28,900 with write_through): prefill-0 on 135 (400 GB/rank) and a 137
#   prefill on card0-3 (200 GB/rank). Here that is prefill-2, which keeps HSA_NO_SCRATCH_RECLAIM=1 (hang fix candidate;
#   prefill-0 keeps 0 as the control); prefill-1 is stopped. Only the write policy changes: write_back keeps device pages
#   off the host until eviction, so the cache holds GPU + host instead of host alone. IO backend (kernel) and layout
#   (page_first) are unchanged. Decode stays at mem 0.92 / 4 tokenizer workers (t5b ran 0.85; C176 showed decode KV is
#   not the limit there).
PREFILL_MEM_FRACTION=0.88
HICACHE_SIZE_prefill_0=400
HICACHE_SIZE_prefill_1=200
HICACHE_SIZE_prefill_2=200
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_0=512
EXTRA_ARGS_decode_0="--tokenizer-worker-num 4"
DECODE_MEM_FRACTION=0.92
GPUS_prefill_1=4,5,6,7
GPUS_prefill_2=0,1,2,3
DECODE_LOG=decode-0.after-mem092.log
EXTRA_ENV_prefill_2="HSA_NO_SCRATCH_RECLAIM=1"
PREFILL_HICACHE_WRITE_POLICY=write_back
EXCLUDE_INSTANCES="prefill-1"

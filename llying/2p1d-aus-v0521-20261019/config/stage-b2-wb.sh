# b2 (2026-10-19): b1 with prefill HiCache write_back (installed as config/round.b1.sh by restart_prefills.sh; decode
# stays up and warm). Under write_back a page is copied to host only when the GPU evicts it, so the prefix cache holds
# GPU + host instead of host alone: per rank about 2.43M GPU + ~5.5M host tokens instead of ~5.5M. IO backend (kernel)
# and layout (page_first) unchanged; on ROCm the backup uses the same transfer kernel as write_through, only the time
# of the copy moves (to eviction, possibly on the allocation path).
PREFILL_MEM_FRACTION=0.88
HICACHE_SIZE_prefill_0=250
HICACHE_SIZE_prefill_1=250
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_0=512
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_1=512
PREFILL_HSA_NO_SCRATCH_RECLAIM=1
DECODE_MEM_FRACTION=0.92
EXTRA_ARGS_decode_0="--tokenizer-worker-num 4"
PREFILL_HICACHE_WRITE_POLICY=write_back
DECODE_LOG=decode-0.log
# QuickReduce off on the prefills (see config/stage-b1-wt-qr0.sh).
PREFILL_EXTRA_ENV="MC_LOG_LEVEL=INFO ROCM_QUICK_REDUCE_QUANTIZATION=NONE"

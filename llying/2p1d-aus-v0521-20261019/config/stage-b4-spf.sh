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
# b4 spf (10-20 02:55): prefill scheduling shortest-prefill-first. In a1-wb C176 the prefill waits 10.9 s on average at ~65%
# busy: under FCFS, 49% of prefill busy time is long single-request chunk runs (cold first turns, up to 92 chunks) that the
# short continuation turns pinned to the same DP rank wait behind. SPF sorts the waiting queue by uncached work (host hits count
# as cached) and caps an in-progress long chunk to leave budget for shorter waiting requests.
PREFILL_EXTRA_ARGS="--schedule-policy shortest-prefill-first"

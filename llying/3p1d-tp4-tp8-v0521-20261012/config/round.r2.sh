# Stage t6 overlay + prefill-2 hang fix candidate (2026-10-15): config/stage-mem092.sh, and prefill-2 only runs
# with HSA_NO_SCRATCH_RECLAIM=1 (prefill-0 / prefill-1 keep 0 as controls; decode has always used 1).
#   prefill-2 hung three times (10-14 05:30, 11:31 on 137 card4-7; 10-15 02:25 on card0-3), all after 2a raised prefill
#   mem 0.85 -> 0.88 (about 9-13 GiB of HBM left at round end, the PyTorch allocator holds the rest). Signature: every
#   scheduler waits on a GPU event, GPUs at 100%, no error. With reclaim enabled, ROCr 7.2 AcquireQueueMainScratch parks a
#   queue whose scratch allocation fails while any large scratch is in use (scratch.retry, amd_gpu_agent.cpp ~1952) until
#   that scratch is released; if the holder is a collective waiting for a peer that waits for the parked queue, nothing
#   moves. HSA_NO_SCRATCH_RECLAIM=1 keeps scratch permanently on its queue and has no such wait (a failed allocation
#   errors instead). See RESULT.zh-CN.md §4.12.
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

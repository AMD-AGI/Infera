# stage-b1-wt: write_through baseline config (was round.b1n.sh); b1 rerun on a new allocation (34389 was preempted at 15:23 after the b1-wt baseline had already
# failed: prefill-0, the instance with HSA_NO_SCRATCH_RECLAIM=0, died at 13:04 on "Memory access fault by GPU node-3 ...
# address 0xfffffffff000"). Same as round.b1.sh except both prefills run HSA_NO_SCRATCH_RECLAIM=1 (decode always has;
# the 3P1D prefill with 1 ran ~5 h clean).
PREFILL_MEM_FRACTION=0.88
PREFILL_HSA_NO_SCRATCH_RECLAIM=1
HICACHE_SIZE_prefill_0=250
HICACHE_SIZE_prefill_1=250
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_0=512
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_1=512
DECODE_MEM_FRACTION=0.92
EXTRA_ARGS_decode_0="--tokenizer-worker-num 4"
DECODE_LOG=decode-0.log
# qr0 (10-19 22:35): the a1 prefill-0 hang at 22:19 had TP2 inside a QuickReduce all-reduce
# (quickreduce::allreduce_prototype_twoshot, ROCM_QUICK_REDUCE_QUANTIZATION=INT8 from the base image) while TP0/1/3 were in
# an RCCL collective (rocgdb, runtime/runs/a1/tune/hang-prefill-0-*): the backend is chosen per call from the local tensor
# size, so ranks can diverge and deadlock. QuickReduce off on the prefills; decode (fixed CUDA-graph sizes) unchanged.
PREFILL_EXTRA_ENV="MC_LOG_LEVEL=INFO ROCM_QUICK_REDUCE_QUANTIZATION=NONE"

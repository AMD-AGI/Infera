# b3 candidate (prepared 10-19 19:00, queued only if the b2 data supports it): b2 (write_back) + decode allocation policy
# prefill_complete on both roles (full restart via restart_all_engines.sh TAG=pcwb). Decode then allocates KV only after the
# prefill has finished, so requests waiting for their KV transfer (7.85/rank in warm-b1, 6.95 in t5b) stop holding decode KV,
# and the prefill no longer waits for the decode allocation before it computes (bootstrap). Needs prefill HiCache write_back
# and the v0.5.21 validator fix (config/mounts.pcwb.tsv: attn_dp_size instead of dp_size), as in 3P1D config/stage-pcwb.sh.
PREFILL_MEM_FRACTION=0.88
PREFILL_HSA_NO_SCRATCH_RECLAIM=1
HICACHE_SIZE_prefill_0=250
HICACHE_SIZE_prefill_1=250
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_0=512
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_1=512
DECODE_MEM_FRACTION=0.92
PREFILL_HICACHE_WRITE_POLICY=write_back
PREFILL_EXTRA_ARGS="--disaggregation-decode-allocation-policy prefill_complete"
EXTRA_ARGS_decode_0="--tokenizer-worker-num 4 --disaggregation-decode-allocation-policy prefill_complete"
PREFILL_MOUNTS_ADD="$R/config/mounts.pcwb.tsv"
DECODE_MOUNTS_ADD="$R/config/mounts.pcwb.tsv"
DECODE_LOG=decode-0.after-pcwb.log
# QuickReduce off on the prefills (see config/stage-b1-wt-qr0.sh).
PREFILL_EXTRA_ENV="MC_LOG_LEVEL=INFO ROCM_QUICK_REDUCE_QUANTIZATION=NONE"

# Stage t8 overlay (2026-10-15): 3P1D + prefill HiCache write_back + decode allocation policy prefill_complete
# (PLAN-TUNING.zh-CN.md, 10-15 09:40 and 10:50). At C192 with decode mem 0.92 the decode KV was full again (KV>90% 45.9%,
# pre-allocation waits 0.47/rank) and requests waiting for their KV transfer held ~19% of it; with prefill_complete decode
# allocates only after the prefill finishes, so that hold shrinks to the transfer itself. prefill_complete needs prefill
# HiCache L2 write_back (sglang _validate_prefill_complete) and is set on both prefill and decode, hence a full restart.
# Everything else as config/stage-mem092-nsr.sh (prefill-2 keeps HSA_NO_SCRATCH_RECLAIM=1, prefill-0/1 keep 0 as controls).
PREFILL_MEM_FRACTION=0.88
HICACHE_SIZE_prefill_0=400
HICACHE_SIZE_prefill_1=200
HICACHE_SIZE_prefill_2=200
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_0=512
DECODE_MEM_FRACTION=0.92
GPUS_prefill_1=4,5,6,7
GPUS_prefill_2=0,1,2,3
EXTRA_ENV_prefill_2="HSA_NO_SCRATCH_RECLAIM=1"
PREFILL_HICACHE_WRITE_POLICY=write_back
PREFILL_EXTRA_ARGS="--disaggregation-decode-allocation-policy prefill_complete"
EXTRA_ARGS_decode_0="--tokenizer-worker-num 4 --disaggregation-decode-allocation-policy prefill_complete"
DECODE_LOG=decode-0.after-pcwb2.log
# v0.5.21 _validate_prefill_complete computes the attention DP width from dp_size, which v0.5.21 itself resolves
# to 1 under DP attention (attn_dp_size=8), and rejects full DP attention as "attention TP > 1" (11:02 decode start).
# Patched copy compares against attn_dp_size; mounted on both roles.
PREFILL_MOUNTS_ADD="$R/config/mounts.pcwb.tsv"
DECODE_MOUNTS_ADD="$R/config/mounts.pcwb.tsv"

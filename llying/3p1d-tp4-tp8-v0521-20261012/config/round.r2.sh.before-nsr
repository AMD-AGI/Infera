# Stage t6 overlay (2026-10-14): 3P1D again with
#   - decode mem 0.85 -> 0.92: at C176 decode KV sat above 90% for 38% of the time (pre-allocation waits 0.52/rank);
#     KV 2.99M -> ~3.37M tokens/rank. Decode high-water left 28.3 GiB free at 0.85 (peak use beyond the pool
#     ~12.6 GiB), so 0.92 (reserve ~21.8 GiB) keeps ~9 GiB headroom;
#   - prefill-1 / prefill-2 GPU sets swapped (prefill-2 hung twice on crsuse2-m2m-137 card4-7): if a hang follows
#     card4-7 (now prefill-1) the GPUs are suspect, if it stays with prefill-2 (now card0-3) the instance path is.
# Everything else as config/stage-tok4.sh (prefill mem 0.88 + HiCache 400/200, decode 4 tokenizer workers).
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

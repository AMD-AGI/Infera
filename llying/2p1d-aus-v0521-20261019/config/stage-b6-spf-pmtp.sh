# b6 pmtp (10-21): config/stage-b4-spf.sh plus the decode's EAGLE flags on the prefills (PREFILL_MTP=1, see engine.sh).
# Until now the prefills ran without --speculative-algorithm, so they had no NextN layer and the PD transfer carried no
# draft KV / DSA index-K: the decode's draft read uninitialised slots for every prompt position. Fresh slots are zero, the
# draft indexer's scores tie at 0 and its top-k scatter kernel runs ~17x slower (a4-spf cold profile: 3.2 ms vs 0.18 ms
# per call, 4 calls per step); slots reused after decode traffic hold stale values, which is the "decode warm-up drift".
# InferenceX amd_utils/server_sglang.sh passes the MTP flags to both roles for exactly this reason.
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
PREFILL_EXTRA_ENV="MC_LOG_LEVEL=INFO ROCM_QUICK_REDUCE_QUANTIZATION=NONE"
PREFILL_EXTRA_ARGS="--schedule-policy shortest-prefill-first"
PREFILL_MTP=1

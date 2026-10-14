# Best configuration found so far (2026-10-14), as a round overlay for a fresh launch:
# scripts/reproduce_best.sh installs it as config/round.<ROUND>.sh. On top of config.3p1d.sh:
#   prefill: mem 0.88; HiCache 400 GB/rank on prefill-0 (135, one instance; 512 GB register chunk),
#            200 GB/rank on prefill-1/2 (137, two instances)       -> PLAN-TUNING.zh-CN.md 2a
#   decode : 4 HTTP/TokenizerManager workers (the single TokenizerManager saturated at 100% CPU)
# Router switches (R1 + P session affinity + R2) are passed per round (scripts/reproduce_best.sh).
PREFILL_MEM_FRACTION=0.88
HICACHE_SIZE_prefill_0=400
HICACHE_SIZE_prefill_1=200
HICACHE_SIZE_prefill_2=200
HICACHE_HOST_REGISTER_CHUNK_GB_prefill_0=512
EXTRA_ARGS_decode_0="--tokenizer-worker-num 4"

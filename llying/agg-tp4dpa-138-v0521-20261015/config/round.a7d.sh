# a7d: coordination with the a6b prefill chunk (8192 per rank, max-prefill-tokens 8192), mem 0.85,
# HSA_NO_SCRATCH_RECLAIM=0. a7c (coordination + 2048-token chunks) cut decode time per output token by
# 17% (35.9 -> 29.9 ms) but new-prefill throughput fell 15% (4,961 -> 4,219 tok/s): 4x the prefill steps,
# each with fixed costs (MoE weights, collectives, draft) plus waiting for the decode ranks' eager draft;
# queue 19.6 -> 37.4, TTFT mean 21.6 -> 46.8 s, 27,607 tok/s/GPU. Keep coordination, restore the chunk.
source "$AGG/config/round.a7c.sh"
MIXED_CHUNK_SIZE=32768
MIXED_MAX_PREFILL_TOKENS=8192

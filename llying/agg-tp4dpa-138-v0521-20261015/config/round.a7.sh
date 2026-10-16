# a7: a6b (TP4, write_back + mem 0.90, delayer 120, 4 tokenizer workers) + DP spec/prefill coordination
# with a 2048-token prefill chunk per rank (max-prefill-tokens 4096).
# Profiling of a6b under C64 (10-16 23:13-23:28): decode steps 54% of the time; the other ~46% are eager
# steps where one rank prefills (a long prompt's chunk continuations, which the delayer does not hold:
# 8 x 665 ms in a row on one rank in an 8.4 s window) while the other ranks run IDLE batches. With
# SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1 the other ranks verify in the same step instead of idling;
# a decode rank gains one step per prefill step, so the recovered share is ~ decode step / prefill step:
# ~11% at 8192-token chunks (665 ms), ~30-40% at 2048 (~170-250 ms). Both changes act together.
source "$AGG/config/round.a6b.sh"
MIXED_EXTRA_ENV="SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1"
MIXED_CHUNK_SIZE=8192
MIXED_MAX_PREFILL_TOKENS=4096

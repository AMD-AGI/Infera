# a5: a4 (TP4, write_back + mem 0.92) + DP spec/prefill coordination, prefill delayer off; see
# round.t8d.sh for the reasoning. At TP4 C48 (a3) half of the executed prefills waited the full 60
# passes and each rank had a prefill held ~62% of the time.
source "$AGG/config/round.a4.sh"
MIXED_EXTRA_ENV="SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1"
MIXED_PREFILL_DELAYER=0
MIXED_TOKENIZER_WORKERS=4

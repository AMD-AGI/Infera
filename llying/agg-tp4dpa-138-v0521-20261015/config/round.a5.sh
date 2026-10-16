# a5: a4 (TP4, write_back + mem 0.92) with the prefill delayer capped at 20 passes instead of 60, and 4
# tokenizer workers (as t8c). The delayer holds a rank's new prefill while other ranks have none, so
# that prefills share one long step (with EAGLE + DP attention any rank's prefill idles every other
# rank's decode for that step). At a3 C48 ~53% of executed prefills still waited the full 60 passes
# (~4.8 s each, the bulk of TTFT p50 4.8 s); at t8c 79% of waits ran to the cap and only ~8% lined up
# within 20 passes. 20 keeps most of the alignment that happens and cuts the waits that end in a
# timeout anyway. DP spec/prefill coordination is not used: ranks stay in lock-step per layer through
# the MoE collectives, so a decode rank gains only one step per peer prefill step.
source "$AGG/config/round.a4.sh"
MIXED_PREFILL_DELAYER_MAX_PASSES=20
MIXED_TOKENIZER_WORKERS=4

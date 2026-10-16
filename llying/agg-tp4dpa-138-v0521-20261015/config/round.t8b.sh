# t8b: t8a with half the prefill chunk. t8a C128 died 16 min in (10-16 05:57:29) with a HIP OOM in the
# MoE stage-2 output buffer (6.75 GiB requested, 8.34 GiB free, 259 GiB held by PyTorch): MoE runs on
# the tokens gathered from all DP ranks, 8 x 8192 = 65,536 at TP8 vs 32,768 at TP4, and the prefill
# delayer lines every rank's full chunk up in one step. 32768 total (4096 per rank) and
# max-prefill-tokens 4096 cap a step at the 32,768 tokens TP4 ran for an hour at mem 0.92.
source "$AGG/config/round.t8a.sh"
MIXED_CHUNK_SIZE=32768
MIXED_MAX_PREFILL_TOKENS=4096

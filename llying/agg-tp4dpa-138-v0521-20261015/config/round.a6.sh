# a6: a4 (TP4, write_back + mem 0.92) with the prefill delayer capped at 120 passes, 4 tokenizer workers.
# Third point of the delayer cap at C64: 20 (a5, 29,810: alignments mostly lost, ITL +17%, TTFT p50
# -3.3 s), 60 (a4, 31,221: ~half of the executed prefills still timed out at a3). 120 tests whether more
# alignment (better ITL) outweighs the longer waits (TTFT).
source "$AGG/config/round.a4.sh"
MIXED_PREFILL_DELAYER_MAX_PASSES=120
MIXED_TOKENIZER_WORKERS=4

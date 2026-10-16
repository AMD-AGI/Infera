# t8c: t8b with 4 tokenizer workers. t8b C128 was aborted by AgentX at 10-16 06:42:18: the warm-up's
# cache-pressure phase put ~200 requests of ~180k tokens in flight at once (06:41:46), the single
# tokenizer process (chat template + tokenization, ~0.33 CPU-s per request) went to ~1 core and the
# worker's HTTP endpoint stopped answering for ~30 s (3 /metrics scrapes timed out), the router got
# "error sending request" (502) and a root warm-up request failed. Steady-state tokenizer load is low
# (a4 C64: 0.31 cores); bursts at C128 are twice those at C64. The 3P1D decode runs 4 workers.
source "$AGG/config/round.t8b.sh"
MIXED_TOKENIZER_WORKERS=4

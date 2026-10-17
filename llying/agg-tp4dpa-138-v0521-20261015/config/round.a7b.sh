# a7b: a7 (coordination + 2048-token chunks) at mem 0.85. a7 C64 died 2 min into the warm-up
# (10-17 00:06:06): DP1's GPU queue aborted with HSA_STATUS_ERROR_OUT_OF_RESOURCES during the cold-start
# prefill burst with KV usage at 0.03-0.09, i.e. runtime allocations, not KV (same error as a6 at 0.92;
# a6b at 0.90 without coordination ran twice). Coordination runs heterogeneous steps eagerly, which needs
# more transient memory; 0.85 leaves ~14 GB more per GPU (a3 ran stably at 0.85; KV is not the limit).
# decode_check passed 32/32 on a7 with coordination on (agg-runs/a7/launch/decode_check.json).
source "$AGG/config/round.a7.sh"
MIXED_MEM_FRACTION=0.85

# b3 (crsuse2-m2m-138): a3 without DP attention -- plain TP4, one scheduler. ABORTED in warm-up
# (10-20 01:12): chunk 8192 gave TP4 a quarter of DP4's prefill capacity (a TP4 step splits each
# token across the 4 GPUs), the warm-up backlog grew to 1.7M tokens at ~11.7k tok/s. b3b reruns it
# with chunk and max-prefill 32768.
source "$AGG/config/round.a3.sh"
MIXED_DP=1 MIXED_DPA=0 MIXED_DP_LM_HEAD=0
MIXED_CHUNK_SIZE=8192
MIXED_PREFILL_DELAYER=0

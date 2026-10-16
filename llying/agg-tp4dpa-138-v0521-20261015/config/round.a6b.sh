# a6b: a6 (TP4, write_back, delayer 120 passes, 4 tokenizer workers) at mem 0.90 instead of 0.92.
# a6 C64 died 16 min in (10-16 12:27:02): DP2's GPU queue aborted with HSA_STATUS_ERROR_OUT_OF_RESOURCES
# ("the runtime failed to allocate the necessary resources") during a large warm-up prefill batch
# (5 requests, 8192 new + 638k cached tokens); the other three schedulers then waited in the
# collectives and the engine hung. rocm-smi showed 301-304 of 309 GB used at 0.92. 0.90 leaves ~5.8 GB
# more per GPU for runtime allocations at ~5% less KV (a4: C64 throughput is insensitive to KV size).
source "$AGG/config/round.a6.sh"
MIXED_MEM_FRACTION=0.90

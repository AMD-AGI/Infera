# a7c: a7b (coordination + 2048-token chunks, mem 0.85) with HSA_NO_SCRATCH_RECLAIM=0.
# a7 (mem 0.90) and a7b (mem 0.85, ~40 GB free per GPU) both died 2-3 min into the C64 warm-up with
# HSA_STATUS_ERROR_OUT_OF_RESOURCES from a queue error callback while every rank ran 2048-token prefill
# chunks at KV usage 0.04-0.09; a6 (no coordination, 0.92) hit it once too. On ROCm this callback is
# typically a dispatch whose scratch (private segment) the runtime could not allocate; with
# HSA_NO_SCRATCH_RECLAIM=1 (set since the 3P1D prefill-2 hang work) ROCr does not reclaim scratch from
# other queues and retry. 0 restores the default reclaim path (risk: the 3P1D scratch-wait hang; the
# watcher flags engine stalls).
source "$AGG/config/round.a7b.sh"
HSA_NO_SCRATCH_RECLAIM=0

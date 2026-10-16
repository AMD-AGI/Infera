# t8d: t8c + DP spec/prefill coordination, prefill delayer off.
# t8c C128 (23,366 tok/s/GPU) was bound by prefill scheduling, not compute: with EAGLE + DP attention a
# rank's prefill turns every other rank's step into an idle batch (scheduler.py, "prefill and decode
# batches will not be mixed"), so the delayer holds prefills up to 60 passes to line ranks up; at DP8
# 75-80% of executed prefills waited the full 60 passes and 71 of ~156 effective lanes sat before their
# first token. SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1 (experimental, off by default) lets ranks
# without prefill keep drafting/verifying in the same step (eager on those steps), which removes the
# reason to delay prefill.
source "$AGG/config/round.t8c.sh"
MIXED_EXTRA_ENV="SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1"
MIXED_PREFILL_DELAYER=0

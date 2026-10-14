# r3 = r2 + decode KV reserved only after prefill completes (tuning trial).
# r2 showed decode KV held by pre-allocated requests still waiting for their prefill
# transfer (C144->C160: running/rank 7.0->5.0, transfer queue 5.6->8.8), shrinking the
# effective decode batch as concurrency rose.
DECODE_ALLOC_POLICY=prefill_complete

# Notes — what / why / how / context

## 1. How decode admission works today (R1, code-read of the image, file:line in analysis/R1)
- `DecodePreallocQueue.pop_preallocated` admits FIFO; a request gets its whole prompt KV when
  free − 512·(running+transfer+waiting+k) − retracted need ≥ alloc + 512. No split between running and in-flight KV.
- Transfer-queue KV is never retracted (`retract_decode` only looks at the running batch). While anything is retracted,
  `process_decode_queue` returns early (admission and pop_transferred stop).
- `early` policy + `optimistic prefill = 0`: prefill starts a request only after decode allocated → in-flight KV is held
  through prefill queue + compute + KV send.

## 2. The patch (both versions)
- held = pool − allocator.available_size(); running = Σ running_batch.reqs `kv.kv_allocated_len` (updated each decode
  step, mem_cache/allocation.py:670,809); nonrunning = held − running (covers in-flight, arrived-not-batched, deferred
  releases of failed transfers, page slack). Admit iff nonrunning + required_alloc < (1−SHARE)·pool; nonrunning grows by
  each admission in the same call. Rebootstrap (retraction recovery) exempt; progress escape when nothing is in transfer
  or waiting.
- v1 breaks the scan on the first non-fitting request (stock FIFO semantics). v2 skips it (later, smaller requests may
  enter) until it has waited > YIHOU_RUNSHARE_MAX_WAIT_S, then breaks (anti-starvation).

## 3. Wrong turns / gotchas
- pc40c's first burst (64 × 69k tokens) never reached the cap (in-flight max 0.25-0.30) — the smoke must use ~230k-token
  prompts (64 × 200k words).
- During AgentX warmup (max_tokens=1) the cap fired ~2.6 M times in 14 min (equal, huge prompts; running = 0) — benign
  but noisy; the log is rate-limited to every 100th block.
- `sglang:pending_prealloc_token_usage` is never assigned (always 0); decode `/metrics` pool gauges read 0 in warmup.
- Burst nodes (amd-burst-qos, PreemptMode=cancel) were cancelled at exactly 20 min twice (rounds pc40 on 174, pc40b on
  029) — the validation moved to 102.
- `chain_pc40c` "patch active check: 0 env lines" is a broken log grep; the container env and mounted file were verified
  directly (docker exec).
- The 17:18 loss of the login host killed the pc40d driver right before A2.

## 4. Open / next
1. Run pc40d A2 (v2, C40 warm) — the requested no-regression check. If it regresses: try SHARE 0.4, a larger wait, or a
   size-aware rule (exempt requests ≤ X tokens), or the native `prefill_complete` policy (+ prefill write_back).
2. Apply v2 on the best capacity config (q1: max_running 96, prefill 0.88, C48 22,306) where decode KV reaches 0.98-0.99.

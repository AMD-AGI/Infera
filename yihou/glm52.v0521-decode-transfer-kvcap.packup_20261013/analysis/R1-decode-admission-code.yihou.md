# R1 — PD decode KV admission, prealloc/transfer accounting and retraction (sglang v0.5.21, image 9d755dd6)

Source root (SRC): `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang` (paths below are relative to it).
`srt/disaggregation/decode.py` md5 `bd1e5fe776e75f5dfe2bb6b27c5bf7af`. It is NOT one of the 3 runtime bind-mounts. The
mounted `mooncake/conn.py` (`conn.yihou-regbase.py`) keeps the `defer_decode_allocation` path (mount line 3128) [M].
Re-verified 2026-10-12 (2nd pass): decode.py md5, process_decode_queue 3201-3240, pop_preallocated 1262-1540,
_allocatable_token_budgets 1905-1959, _validate_prefill_complete (pd_disaggregation_hook.py:204-247), Req.prefix_len,
DecodeRequest.host_staged/is_rebootstrap, kv.holds_kv property (schedule_batch.py:973) all match [M]. Patch fix: decode.py
has no `import os`; the patch must add it.
Tags: [M] = first-hand (code/log/command), [S] = second-hand (doc/ledger), [I] = inference.

## 0. Bottom line

- The decode pool has no share split between running and in-flight (prealloc/transfer) requests. A request in the
  prealloc queue gets its whole prompt KV (page-aligned) once this holds:
  `free(+evictable) - max(512 * n_active, need_single) [- 512 * last_prebuilt_bs] - retracted_need >= alloc + 512`.
  Here `n_active = running + transfer(device) + waiting_queue + admitted-this-call`. The growth reserve is only
  `num_reserved_decode_tokens` (512) per active request [M].
- Transfer-queue KV cannot be retracted. With `early`, transfer KV is held through prefill queueing, prefill compute
  and the transfer, because prefill does not start a request until decode has allocated its KV (optimistic prefill is
  0) [M]. Today, in-flight KV can therefore take a large share of the pool. In X1 (C44 warm, e2j) the decode log
  shows `pre-allocated usage` max 0.73, above 0.5 on 373 of 8,892 decode-log lines, above 0.3 on 1,455. The /metrics
  sampler gives `(kv_used - decode_sum_seq_lens)/pool` max 0.71, above 0.5 on 148 of 2,890 active samples [M].
  Even so: no retraction, `#prealloc-req` always 0, max token usage 0.87 [M]. So admission never bound at C44, and
  the 512-token cushion was enough for this workload [I].
- Running requests can be retracted (or starved) when they grow by more than 512 tokens each while in-flight KV fills
  the rest. This is guarded only by the 512/request cushion and `need_space_for_single_req`, which is nearly always
  ≤ 0 (see §1.3) [M/I].
- No configuration expresses "running keeps ≥ 50 % of the pool". The closest knobs (`num_reserved_decode_tokens`,
  `disaggregation_decode_extra_slots`, `max_running_requests`) are per-request or count-based, not a token share
  (§3.1).
- Only two allocation policies exist: `early` (default, ours) and `prefill_complete`. `prefill_complete` allocates
  decode KV only after prefill has finished computing, which cuts in-flight hold time to roughly the transfer only.
  It is compatible with our topology, but it rejects our current prefill `--hicache-write-policy write_through`; it
  needs `write_back` (§2).
- Host receive (`host_receive_threshold > 0`) cannot be used with our setup. No built-in backend sets
  `supports_host_destination`, and the feature requires a dense MHA pool, so MLA/DSA is excluded (§2) [M].
- Recommended guarantee: a ~25-line runtime-mount patch in `DecodePreallocQueue.pop_preallocated`, gated by env
  `YIHOU_DECODE_RUNNING_KV_SHARE=0.5`. It caps device-held in-flight KV at `(1-share) * max_total_num_tokens` per DP
  rank (§3.2).

## 1. Accounting model (pseudo-code with file:line)

### 1.1 Where the decode scheduler spends a cycle

`event_loop_{normal,overlap}_disagg_decode` (decode.py:2904/2946) calls `process_decode_queue` (decode.py:3201-3240) [M]:

```
process_decode_queue():                                   # decode.py:3201
  transfer_queue.resolve_deferred_releases()              # 3210
  resumed = prealloc_queue.resume_retracted_reqs()        # 3213  (retracted reqs first)
  waiting_queue += resumed
  if prealloc_queue.retracted_queue: return               # 3215-3217  !! blocks admission AND pop_transferred
  every polling_interval (=1):
    if host_receive_threshold == 0:
      transfer_queue += prealloc_queue.pop_preallocated() # 3226-3228  admission (KV allocated here)
    waiting_queue += transfer_queue.pop_transferred()     # 3229-3236  KV arrived -> waiting
then get_next_disagg_decode_batch_to_run():               # 3022
  _get_new_prebuilt_batch: take up to min(req_to_token_pool.size, max_running_requests) - running   # 3133-3135
      from waiting_queue -> "prebuilt" extend batch, merged into running_batch
  update_running_batch(running_batch)                     # scheduler.py:4200 -> check_decode_mem / retract_decode
```

Request states and who holds device KV:

| state | container | holds device KV? | counted in `n_active` (512 reserve) |
|---|---|---|---|
| bootstrapping / waiting for slot | `prealloc_queue.queue` | no | no |
| allocated, KV in flight | `transfer_queue.queue` (`host_staged=False`) | yes, full `fill_len` page-aligned | yes (decode.py:1870) |
| KV arrived, not yet batched | `scheduler.waiting_queue` | yes | yes (decode.py:1871) |
| running | `running_batch.reqs` | yes, grows each step | yes (decode.py:1869) |
| retracted | `prealloc_queue.retracted_queue` | no (CPU backup) | its full need is subtracted (decode.py:1953-1957) |

Per DP rank (attn_tp=1 in our TP4/DP4): `max_running_requests` 64 → 16 per worker. The log line
`max_running_requests=16` and the server_args `disaggregation_decode_extra_slots: 32` confirm it (decode-0.a1.log) [M].
`extra_slots` defaults to 2 × per-worker when per-worker ≤ 32 (pd_disaggregation_hook.py:139-149) [M]. Count bounds:
- req_to_token rows `= 16 + 32 (+1 pad)` (decode.py:186; kv_cache_configurator.py:1353,1469) [M]. `alloc()` returns
  None when rows run out, and `pop_preallocated` breaks on `req_to_token_pool.available_size() <= 0` (decode.py:1390) [M].
- Metadata buffers: `req_to_token_pool.size * 2 = 32` (scheduler.py:1504-1513). `pop_preallocated` breaks at 0
  (decode.py:1364), and buffers are freed only when a transfer finishes or is released (decode.py:2770, 2801) [M].
  So transfer reqs are ≤ 32 per rank [I].
- Running ≤ 16 per rank (decode.py:3133-3135) [M].

### 1.2 Admission: prealloc queue → allocated/transfer (`pop_preallocated`, decode.py:1245-1717)

```
pop_preallocated():
  retractable = Σ_{r∈running} (len(input_r)+len(output_r))                       # 1270-1273 (NOT page-aligned)
  B = _allocatable_token_budgets(retractable, count_retracted=True)               # 1290-1292 (non-SWA path; ours)
  B -= hicache_pending_restore_tokens  (=0, decode HiCache off)                   # 1293-1294, decode_hicache_mixin.py:155-165
  FIFO over queue (optionally priority-sorted 1300-1304):
    skip if not waiting_for_input (handshake not done)                             # 1362
    break if no metadata buffer                                                    # 1364
    [host-receive branch, only if threshold>0]                                     # 1371-1388
    break if no req_to_token row                                                   # 1390
    alloc = _pre_alloc_fill_len(req) = len(input) + max(len(output)-1,0)           # 1454 / 871-884
            (page-aligned via _required_alloc_tokens only on the radix path, 1422-1424 / 2036-2046;
             the allocator still charges whole pages)
    need  = max(alloc + R, len(input) + min(max_new, CLIP) - retractable)          # 1458 -> 485-494
    if not (need <= B): unlock; BREAK   (head-of-line, FIFO)                       # 1477-1486 -> base.py:112-133
    _pre_alloc(req)  -> kv_allocated_len = fill_len, pages allocated               # 1512 / 2059-2215, 2270
    send metadata to prefill (prefill may now start prefill)                       # 1690-1697 / 1719-1753
    B = _allocatable_token_budgets(retractable, extra_reserved_reqs=n_admitted+1)  # 1525-1531 (recomputed from real pool)

_allocatable_token_budgets(retractable, extra):                                   # 1905-1959
  need_single = max_{r∈running}( min(max_new_r,CLIP) + len(input_r) - retractable )  (0 if no running)   # 1849-1865
  reserved    = R * (|running| + |transfer, device| + |waiting_queue| + extra)                           # 1867-1880
  avail       = allocator.available_size()  (+ radix evictable if decode radix; off)                    # 1936-1941
  B = avail - max(reserved, need_single)                                                                 # 1942-1944
  if last_batch is prebuilt: B -= R * len(last_batch.reqs)                                               # 1946-1951
  B -= Σ_{retracted} (ceil_page(fill_len) + R)                                                           # 1953-1957
```

Here `R = num_reserved_decode_tokens = 512` and `CLIP = SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = 4096`
(environ.py:604; decode.py:138) [M].

The admission condition, for our configuration (no SWA, no radix, no hisparse, no host receive), simplifies to [M]:

```
free_pages_tokens  -  512*(N_run + N_xfer + N_wait + k)  [- 512*N_lastprebuilt]  -  Σ_retracted(need)
      >=  max( alloc + 512 ,  len(input) + min(max_new,4096) - Σ_running seq )
```

Here k is the number admitted earlier in the same call; `need_single` is replaced by the larger reserve term whenever
it is smaller. `schedule_conservativeness` and `new_token_ratio` take no part: no reference in decode.py; the
conservativeness value is consumed only in new_token_ratio_tracker.py:24 [M].

### 1.3 Why prealloc'd requests can crowd out running ones

- The only growth reservation for a running request is the flat 512 tokens (decode.py:1875-1880). There is no term
  for `max_new_tokens`, except that `need_single` and the second argument of `_required_admission_tokens` use
  `len(input)+min(max_new,4096) - Σ_running(seq)`. With ≥ 2 running requests of non-trivial length that value is
  ≤ 0, so it never binds [M code / I magnitude].
- Once admitted, a transfer request's KV is pinned. `retract_decode` only considers `running_batch.reqs`
  (schedule_batch.py:3269-3352). The code says so itself: "Otherwise it is possible for one request running decode
  out of memory, while all other requests are in the transfer queue that cannot be retracted" (decode.py:1268-1269) [M].
- With `early`, prefill's bootstrap queue does not start a request while it is `Bootstrapping`, unless optimistic
  prefill is on; ours is 0 (prefill.py:516-525; server_args `optimistic_prefill_attempts: 0`) [M]. Decode KV is
  therefore pinned for prefill wait + chunked prefill compute + transfer [I from M]. X1 has a 20-s stretch on DP1 with
  7 transfer reqs pinning 0.71 of the pool while running held 0.03-0.06 (metrics.tsv ts 1791791096-1791791116) [M].
- Result: running requests that grow past their 512 cushion consume free pages. When `check_decode_mem` fails they
  are retracted. In-flight KV is bounded only by the token condition above and the count caps (rows 48, metadata 32,
  running 16). There is no share cap [M/I].

### 1.4 Retraction: trigger, policy, cost, fate

```
update_running_batch(batch):                              # scheduler.py:4200-4280
  if not batch.check_decode_mem():                        # schedule_batch.py:3250-3267
      # need = Σ page growth for next step; EAGLE v2: per req ceil_page(committed+2*6) - ceil_page(allocated)
      #        (allocation_sizing.py:40-70: alloc_len_per_decode = max(steps*topk, draft_tokens)=6, reserve=2x)
      # check_decode_capacity: evict cache (none here) then available >= need   (allocator/base.py:151-166)
      retracted, ratio, aborted = batch.retract_decode()  # schedule_batch.py:3269
      log "KV cache pool is full. Retract requests. #retracted_reqs: .., #new_tokens_gained: .., #new_token_ratio: a -> b"  # 4257-4269
      for r in retracted: _add_request_to_queue(r, is_retracted=True) -> prealloc_queue.retracted_queue   # 4271-4272, decode.py:762-764
retract_decode():
  order = _get_decode_retraction_order: keep long-output/short-input; pop (retract) fewest output tokens,
          then longest input first ('length' policy; 'priority' optional)                 # 3355-3389
  always retract >= 1 (first_iter), stop when check_decode_mem passes on the remainder, keep >= 1   # 3277-3289
  release_req(): backup_kv_cache(... retraction_backup) then free device KV               # 2213-2248
      backend on HIP: 'cpu_tensor' (host_pool disabled on HIP: "Large ROCm retraction restores can fault the GPU
      process", kv_cache_builder.py ~205-219) -> Req.offload_kv_cache: synchronous D2H copy of seqlen-1 tokens over
      all layers (schedule_batch.py:1982-2001; memory_pool.py:2689+, DSA override memory_pool.py:5421)
  if the backup fails ('none' backend): abort 503                                         # 3311-3326
```

Resume: `resume_retracted_reqs` (decode.py:945-1020) runs first in every `process_decode_queue`. It admits a
retracted request when `ceil_page(fill_len)+512 <= _allocatable_token_budgets(count_retracted=False)`, re-allocates
pages with `_pre_alloc`, and H2D-restores with `load_kv_cache` (schedule_batch.py:2003-2020). It then goes to the
waiting_queue as a prebuilt request; no recompute [M].

Cost [M code / I magnitude]:
1. A synchronous device sync + D2H copy on the scheduler thread for every retracted request.
2. While any request is in `retracted_queue`, `process_decode_queue` returns early (decode.py:3215-3217). That stops
   new admissions and also `pop_transferred`. Completed transfers then sit unpolled, and since prefill admission is
   gated by decode allocation under `early`, prefill stalls too.
3. An H2D restore later.
4. Lost decode progress is limited (the KV is restored), but the request's TPOT suffers. `new_token_ratio` is
   updated but unused by PD decode admission.

The draft (EAGLE) KV pool uses the same index space; the decode log shows the second "KV Cache is allocated" with
the same #tokens 2,762,176 and 1.82 GB [M]. Whether the cpu_tensor backup covers draft KV is not verified (open).

## 2. Allocation policies and related modes

`disaggregation_decode_allocation_policy: Literal["early","prefill_complete"] = "early"` (arg_groups/fields/disagg.py:59-64) [M].

| policy | what it does | gates | fit for GLM-5.2 P4D4 |
|---|---|---|---|
| `early` (ours) | Decode allocates KV as soon as the handshake reaches WaitingForInput and sends indices. Prefill starts the request only after that (prefill.py:516-525). In-flight KV is pinned for prefill queue + compute + transfer. | none | in use |
| `prefill_complete` | Prefill computes without decode bootstrap (forces `optimistic_prefill_attempts=1` on prefill, pd_disaggregation_hook.py:47-55) and retains KV on prefill. On the last chunk it calls `mark_prefill_complete()` (prefill.py:1049-1053 → common/conn.py:1847-1855), which notifies decode. Decode sets the room WaitingForInput only then (common/conn.py:788-791). Only then does `pop_preallocated` see `waiting_for_input` and allocate. The bootstrap timeout restarts at prefill completion (common/conn.py:1938-1945). Optimistic yield-to-waiting is disabled (prefill.py:1395-1398, 1722-1725). | `_validate_prefill_complete` (pd_disaggregation_hook.py:204-247): mooncake only; pp=cp=dcp=1; **attn_tp must be 1** (`tp_size == dp_size` with dp-attention); no pdmux/decoupled spec, no role switch, no encoder disagg, no HiSparse, no host receive; decode: no decode radix / decode HiCache; **prefill HiCache must be L2 `write_back` with no storage backend**; no `SGLANG_DISAGG_STAGING_BUFFER`, no Rust server. Runtime: KV state types must be DSA-only or none (common/conn.py:375-379). Peer attn TP/CP/PP must be 1, and the policy must match on both roles, else the request fails (common/conn.py:2057-2068). Prefill: `serving_hook.py:498-505` errors if the resolved config disables optimistic prefill. | Topology OK: TP4 + DP4 dp-attention → attn_tp=1 [M arithmetic]. Mooncake OK. Mount conn.py OK (mount:3128) [M]. Decode radix/HiCache off [M server_args]. **Blocker: prefill uses `--hicache-write-policy write_through` (dryrun.e2g-102 line 19) → must change to `write_back`** [M]. EAGLE on decode: no gate found [M absence], runtime untested. DSA state type: allowed by design [M]; whether GLM-5.2's state list passes is to be checked at launch [I]. ROCm: no is_hip gate on this path [M absence]. |

Related modes:
- **Host receive** `disaggregation_decode_host_receive_threshold` (float 0..1, default 0; fields/disagg.py:179-182).
  Above the threshold, decode receives KV into the host retraction pool instead of device (decode.py:1371-1388,
  1755-1828) and restores to device when batching (decode.py:3188-3198). Its help text says it "requires dense MHA
  and a transfer backend that supports host destinations. No built-in backend currently supports this." Indeed
  `supports_host_destination` is False in base/conn.py:124 and overridden nowhere (grep) [M]. It also forces
  `retraction_backup=host_pool`, which HIP never picks by default (pd_disaggregation_hook.py:64-79;
  kv_cache_builder.py) [M]. **Not usable** for MLA/DSA + mooncake on ROCm.
- **Retraction backup** `disaggregation_decode_retraction_backup ∈ {cpu_tensor, host_pool, none}`, None → inferred
  (fields/disagg.py:95-109). On HIP it resolves to `cpu_tensor` (kv_cache_builder.py, `not is_hip()` gate) [M].
- **Decode radix cache** (`disaggregation_decode_enable_radix_cache`): rejected with any speculative algorithm except
  DSPARK (pd_disaggregation_hook.py:103-110). Not available with EAGLE. Decode HiCache requires it
  (scheduler.py:512-515), so decode HiCache is a no-op for us, consistent with the report [M].
- **Offload kvcache** (`disaggregation_decode_enable_offload_kvcache`): requires `hicache_storage_backend`
  (kv_cache_hook.py:399-405). It offloads finished requests' KV to L3 and adds no device capacity [M].
- **`SGLANG_DISAGG_STAGING_BUFFER`**: raises for MLA (decode.py:439-444) [M].

## 3. Guaranteeing "running keeps ≥ 50 % of the decode KV pool"

### 3.1 Pure configuration — none is exact

| knob | effect on the share | verdict |
|---|---|---|
| `--num-reserved-decode-tokens X` | Per active request (running + transfer + waiting) reserve of X tokens, plus X for the candidate. Also gates resume. Larger X = more growth cushion per running request, and it also throttles in-flight admission (each in-flight request adds X). Does not bound in-flight tokens. | Partial: protects growth up to X/request, not a 50 % share. |
| `--disaggregation-decode-extra-slots E` | Rows = 16 + E. In-flight + waiting + running ≤ 16+E. Metadata buffers = 2·16 = 32 regardless (scheduler.py:1504-1513). | A count cap. A token bound only via max prompt length (agentic prompts are long and variable). |
| `--max-running-requests` | Also sets the req_to_token size, metadata count and the default extra_slots. | Count only. |
| `prefill_complete` | Pinned time per in-flight request ≈ transfer only. Prefill retains KV meanwhile (prefill pool pressure instead). | Reduces in-flight share; gives no guarantee. Requires prefill write_back. |
| `schedule_conservativeness`, `SGLANG_*NEW_TOKEN_RATIO*` | Not read by PD decode admission. | No effect. |

### 3.2 Minimal runtime-mount patch (recommended)

Mount a copy of `srt/disaggregation/decode.py` (new file, e.g. `$WS/harness/mounts-v0521/src/decode.yihou-runshare.py`,
base md5 `bd1e5fe7…`) at `/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py`, decode container only.

Definitions (per DP rank, which owns its pool; with attn_tp=1 no cross-rank agreement is needed. For attn_tp>1 all
attn-TP ranks see identical queues and pool state and stay deterministic, the same as the existing budget [I]):
- `pool = self.max_total_num_tokens` (2,762,176/rank).
- `inflight` = device tokens held by admitted-but-not-arrived requests =
  `Σ_{dr in transfer_queue.queue, not dr.host_staged} ceil_page(dr.req.kv.kv_allocated_len) - dr.req.prefix_len`,
  plus the pages admitted earlier in the same `pop_preallocated` call. Those are not yet in transfer_queue, because
  the caller extends after return (decode.py:3227-3228).
- "running" side = everything else in use (running batch + waiting_queue + their growth) plus free. The invariant
  `inflight <= (1-share)*pool` therefore guarantees running/arrived requests can always reach `share*pool` without
  being retracted because of in-flight KV.
  - Arrived requests in waiting_queue count as running; in-flight ones never do.
  - The ongoing 512 × n_active reserve stays (existing condition, kept as AND).

Code (inside `DecodePreallocQueue`):

```python
# module top -- decode.py does NOT import os today (decode.py:21-33 [M]); add it
import os
_YIHOU_RUN_SHARE = float(os.environ.get("YIHOU_DECODE_RUNNING_KV_SHARE", "0"))  # 0 = off; 0.5 = running keeps >= 50%

def _yihou_inflight_tokens(self) -> int:
    page = self.token_to_kv_pool_allocator.page_size
    return sum(
        ceil_align(dr.req.kv.kv_allocated_len, page) - (dr.req.prefix_len or 0)
        for dr in self.transfer_queue.queue
        if not dr.host_staged and dr.req.kv.holds_kv
    )
```

In `pop_preallocated`:
- After line 1294 (`full_allocatable_tokens -= reserved_restore_tokens`):
  ```python
  yihou_cap = int((1.0 - _YIHOU_RUN_SHARE) * self.max_total_num_tokens) if _YIHOU_RUN_SHARE > 0 else None
  yihou_inflight = self._yihou_inflight_tokens() if yihou_cap is not None else 0
  ```
- Between line 1476 (end of the SWA block) and line 1477 (`if not self._prealloc_reservation_fits(`):
  ```python
  if yihou_cap is not None and not decode_req.is_rebootstrap:
      page = self.token_to_kv_pool_allocator.page_size
      cand = ceil_align(required_alloc_tokens, page)
      # progress escape: a lone oversize request may still enter when nothing else is in flight
      if yihou_inflight > 0 and yihou_inflight + cand > yihou_cap:
          if prefix_match is not None and prefix_match.l1_prefix_len > 0:
              self.tree_cache.unlock(decode_req.req.lock); decode_req.req.lock = None
          self.yihou_cap_block_ct = getattr(self, "yihou_cap_block_ct", 0) + 1
          break        # FIFO, same as the existing budget break
  ```
- After `_pre_alloc` (line 1512-1517): `yihou_inflight += ceil_align(required_alloc_tokens, page)` when the cap is
  on. Use the same `required_alloc_tokens` as above (equals the page-charged delta).
- Optional observability: append `yihou-inflight: {x/pool:.2f}, yihou-capblk: {ct}` to the decode log. That needs
  metrics_reporter.py:996-1003, a second mount. Alternatively, log once per change from inside decode.py with
  `logger.info` at a rate limit.

Invariants and notes:
1. Page alignment: compare page-charged tokens. `kv_allocated_len = fill_len` (decode.py ~2270); the allocator
   charges whole 64-token pages.
2. Retraction interplay: retracted requests already block all admission (decode.py:3215-3217). Do not apply the cap
   in `resume_retracted_reqs`; resumed requests are "running" side. Rebootstrap requests are exempt (they are
   recovering running work).
3. Speculative/EAGLE: the draft KV shares indices with the target allocator, so no separate accounting. Per-step
   growth (≤ 1 page/request at 2×6 tokens) stays covered by the 512 reserve.
4. Deadlock freedom: the `yihou_inflight > 0` escape guarantees progress for prompts > cap (1.38 M tokens/rank, so
   moot for AgentX) [I].
5. Back-pressure: under `early` the cap delays when prefill can start, so prefill may idle. Measure prefill busy %
   and TTFT. The X1 data shows in-flight > 0.5 only ~4-5 % of the time at C44, so the cap would bind rarely at C44
   [M data / I effect].
6. Host receive / hisparse paths untouched (the cap is placed after them; host-staged requests are excluded).

Stronger variant (if growth protection, not just the share, is wanted): additionally require
`inflight + cand <= pool - (Σ_running ceil_page(seq) + 512*N_run) - share_margin`. That is already implied by the
existing budget, so no extra code is needed.

## 4. Runtime observability for verifying the policy

| quantity | source | notes |
|---|---|---|
| in-flight share | decode log `pre-allocated usage: x` = `Σ transfer(not host_staged) extend_end / max_total_num_tokens` (metrics_reporter.py:998; decode.py:1842-1847) | Not page-aligned. Log only, every `decode_log_interval`. **Primary verifier: must stay ≤ 1-share (+ page rounding).** |
| counts | decode log `#running-req`, `#prealloc-req`, `#transfer-req`, `#retracted-req`, `#queue-req` (metrics_reporter.py:999-1003). /metrics `num_running_reqs`, `num_decode_prealloc_queue_reqs`, `num_decode_transfer_queue_reqs`, `num_queue_reqs`, `num_retracted_reqs` (1406-1418; scheduler.py:4234) | `#prealloc-req > 0` = admission blocked (budget or cap). |
| pool used | decode log `#token`, `token usage` = `full_num_used` = in-flight + running + waiting (pool_stats_observer.py:144); /metrics `kv_used_tokens`, `kv_available_tokens`, `token_usage` (pool_stats_observer.py:150-166) | |
| running KV | /metrics `decode_sum_seq_lens` = Σ running seq_lens, same tick as the pool stats (metrics_reporter.py:1061-1064) | in-flight ≈ `kv_used_tokens - decode_sum_seq_lens` (also includes waiting_queue + page slack). |
| retraction | log `KV cache pool is full. Retract requests. #retracted_reqs:` (scheduler.py:4257-4269); `retract_decode: aborted` (schedule_batch.py:3324,3348); counters via `increment_retracted_reqs` (scheduler.py:4235-4244) | |
| dead gauge | `sglang:pending_prealloc_token_usage` is declared (metrics_collector.py:146, 617) but never assigned anywhere in srt (grep) → always 0 [M]. Do not use it; X1 confirms max 0.0 [M]. | |

X1 reference numbers (C44 warm e2j, 06:47-08:10 UTC, decode only, samples with running>0) [M]:
- `kv_used` p50 0.39, p99 0.73, max 0.87
- running seq p50 0.22, max 0.78
- in-flight (`used - running_seq`) p50 0.08, p90 0.40, max 0.71
- transfer reqs/rank p50 2, max 15; running/rank max 16
- `#prealloc-req` always 0; retracted 0
- decode-log `pre-allocated usage` max 0.73 (373/8,892 lines > 0.5)

Unrelated, noted only: X1 `agentx_conc44.json` reports `num_requests_successful` 4,373 of 4,861 total [M].

## 5. SGLANG_* envs on these paths (srt/environ.py)

- `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=4096` (604): the max_new clip in admission and need_single.
- `SGLANG_RETRACT_DECODE_STEPS=20` (603): tokens evicted from the tree per remaining request on release_req
  (schedule_batch.py:2244). It is a no-op without a radix cache.
- `SGLANG_INIT_NEW_TOKEN_RATIO=0.7`, `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=0.14`,
  `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS=600` (600-602): not used by PD decode admission.
- `SGLANG_TEST_RETRACT`/`_INTERVAL` (421-422): forced retraction test.
- `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=True`, `_TIMEOUT=30.0` (729-730): transfer-queue release
  deferral (decode.py:2379-2384).
- `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` (711, default 300; ours 1800) and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`
  (715, ours 1800).
- `SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK` (723); `SGLANG_DISAGG_STAGING_BUFFER` (803, MLA-forbidden).
- In utils/mixins: `SGLANG_ENABLE_DISAGG_SAMPLING_MASK`, `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE`,
  `SGLANG_MOONCAKE_CUSTOM_MEM_POOL`, `SGLANG_TEST_DISAGG_FAILURE_PROB` (grep of utils.py, decode_*_mixin.py,
  decode_kvcache_offload_manager.py).
- None of these sets a share cap.

## 6. Open items

- Does the cpu_tensor retraction backup include draft-model KV, or is the draft KV rebuilt after restore? Not read.
- `prefill_complete` with EAGLE decode + DSA + HiCache write_back on ROCm: config gates pass on paper, but it has
  never been run. The prefill-side KV retention cost is unmeasured.
- In X1, transfer-queue KV stayed pinned at 0.71 of DP1's pool for ≥ 20 s with 7 transfers. Long prefill
  queueing/compute under `early` is the likely cause [I]; check against the prefill logs.

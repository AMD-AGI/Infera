All file:line references below are under `S=…/image-9d755dd6/sglang/srt/`, except `H/` for the harness. Anything not read directly in the code is marked [inferred].

## Two blockers for the planned run

**A. Prefill HiCache is `write_through` today, and `prefill_complete` rejects it.**
- `arg_groups/pd_disaggregation_hook.py:230-238` adds "prefill HiCache other than L2 write_back" to the unsupported list when the role is prefill, HiCache is on, and either a storage backend is set or the write policy is not `write_back`.
- The harness default is `PREFILL_HICACHE_WRITE_POLICY:=write_through` (`H/config.yihou.v0521-base.sh:98`), and `engine.sh:305` emits it.
- So the run also needs `PREFILL_HICACHE_WRITE_POLICY=write_back`. That changes a second variable besides the policy, and it should be recorded in notes.
- Startup error: `ValueError: --disaggregation-decode-allocation-policy prefill_complete does not yet support prefill HiCache other than L2 write_back` (`pd_disaggregation_hook.py:243-247`).

**B. The harness never passes the policy flag.**
- `grep allocation` over `engine.sh`, `config.sh` and every `config.yihou.*.sh` finds nothing, so both legs currently run on the defaults: `early` (`arg_groups/fields/disagg.py:59-64`) and host-receive threshold `0.0` (`disagg.py:179-182`).
- Pass it through the existing hooks: `PREFILL_EXTRA_ARGS_ADD` and `DECODE_EXTRA_ARGS_ADD` (`H/config.yihou.v0521-base.sh:243-244`), which `engine.sh:48,61,318-321` turn into engine args.
- The entrypoint is `python3 -m infera.engine.sglang` (`engine.sh:251`). I did not read that wrapper, so I have not confirmed it forwards these args unchanged.

## (1) Required flags on each leg

- **Both legs must set `--disaggregation-decode-allocation-policy prefill_complete`.** The help text says "Set the same policy on both roles" (`disagg.py:61-63`).
- **How prefill tells decode its policy:**
  - Prefill puts `decode_allocation_policy` into its bootstrap registration (`disaggregation/common/conn.py:1377-1379`).
  - The bootstrap server returns HTTP 400 "Inconsistent prefill allocation policies" if prefill ranks disagree (`common/conn.py:2554-2560`).
- **Mismatch between legs does not stop startup.** Each request fails instead, in decode `CommonKVReceiver.init` (`common/conn.py:2057-2064`): `Disaggregation decode allocation policy mismatch: prefill=<p>, decode=<d>. Set --disaggregation-decode-allocation-policy identically on both roles.` The receiver is set to `KVPoll.Failed`.
- **Prefill also needs optimistic prefill, which is set automatically:**
  - If `optimistic_prefill_attempts == 0` (the default, `disagg.py:122-124`), it is resolved to 1 (`pd_disaggregation_hook.py:47-54`).
  - If a later validation turns it back to 0, startup raises `ValueError: prefill_complete requires optimistic prefill, but the resolved model/cache configuration disabled it.` (`arg_groups/serving_hook.py:497-505`). The rules that zero it are pp>1, or a HiCache write policy outside the supported set (`serving_hook.py:466-495`).

## (2) Incompatibilities and exact error text

**Startup check.** `_validate_prefill_complete` (`pd_disaggregation_hook.py:204-247`) raises `"--disaggregation-decode-allocation-policy prefill_complete does not yet support " + ", ".join(...)` with any of these items:

| Item in the error | Condition |
|---|---|
| aggregated serving | mode is not prefill or decode |
| transfer backends other than Mooncake | backend is not mooncake |
| pipeline or attention context parallelism | pp≠1, attn_cp≠1 or dcp≠1 |
| attention tensor parallelism greater than one | `tp_size != (dp_size if enable_dp_attention else 1)` (:213-215). TP4/DP4/DPA gives 4==4, so ours passes. |
| PD multiplexing or decoupled speculation | `enable_pdmux` or `decoupled_spec_role != "null"` |
| runtime PD role switching | `enable_pd_role_switch` |
| encoder disaggregation | encoder_only / language_only |
| HiSparse | `enable_hisparse` |
| decode host KV buffering | `disaggregation_decode_host_receive_threshold > 0`. Must stay 0.0, as it is now. |
| decode radix cache or decode HiCache | decode with `disaggregation_decode_enable_radix_cache` (default False, `disagg.py:85-88`) or `enable_hierarchical_cache`. We have `DECODE_HICACHE=0` (`base.sh:92`). |
| prefill HiCache other than L2 write_back | see blocker A |
| staging buffers | `SGLANG_DISAGG_STAGING_BUFFER` (default False, `environ.py:803`) |
| the Rust bootstrap registry | `SGLANG_RUST_SERVER` (default False, `environ.py:1871`) |

**KV-manager init check.**
- `common/conn.py:375-379`: `ValueError: prefill_complete currently supports full-attention and DSA KV pools only` if any `kv_args.state_types` entry is not `StateType.DSA`.
- A DSA pool registers `DSA`, plus `DSA_TAIL` only when `kpool_use_compress` is on (`disaggregation/utils.py:1564-1574`, `mem_cache/memory_pool.py:5219`). A `DSA_TAIL` entry would trip this error.
- `index_kpool_compress` defaults to False when the key is absent (`configs/model_config.py:446-447`). The GLM-5.2-MXFP4 `config.json` has no `index_kpool*` keys. So [inferred] state_types is `[DSA]` only (plus the draft DSA buffers merged into the same entry, `utils.py:1539-1554`), and the check passes. The first launch log will confirm this.

**Runtime peer check.** `common/conn.py:2065-2068` fails the request with `prefill_complete requires peer attention TP1/CP1/PP1`. With DPA, attn_tp = tp/dp = 1, so [inferred] ours passes.

**Things that are not rejected (code read):**
- EAGLE/MTP on decode: only decoupled speculation is listed.
- DP attention: allowed, and effectively required to get attn_tp=1.
- Chunked prefill: not rejected. Under this policy a chunked request is no longer yielded to waiting requests (`disaggregation/prefill.py:1394-1404`).
- Page size: not referenced by the policy checks.

**Check our patch didn't change this.** The mounted `conn.yihou-regbase.py` differs from image `mooncake/conn.py` only by three pure insertion hunks (`82a83,133`, `382a434`, `507a560`). It keeps the deferred-allocation branch, at line 3128 in the patched file and 3075 in the original.

## (3) Request lifecycle

**early:**
1. Decode `CommonKVReceiver.init` sets `WaitingForInput` immediately after the handshake setup (`common/conn.py:2096-2098`).
2. In `_update_handshake_waiters`, `waiting_for_input` becomes True (`decode.py:1067-1069`).
3. `pop_preallocated` then allocates decode KV right away, subject to budget (`decode.py:1362`, `:1512 _pre_alloc`). It sends destination indices through `_send_kv_metadata` (`decode.py:1719-1753`), and the request moves to the transfer queue.
4. Prefill admits only bootstrapped requests, plus up to `optimistic_prefill_attempts` speculative ones (`prefill.py:517-521`).
5. A chunked optimistic request that is still pending yields to bootstrapped waiters (`prefill.py:1394-1404`). After its attempts are exhausted it goes back to the bootstrap queue (`prefill.py:1716-1727`).
6. KV chunks stream to decode's already-allocated pages as prefill computes.

**prefill_complete:**
1. **Decode, request arrives.** `init` does not set WaitingForInput. It starts `_prefill_wait_start` and sends a `BOOTSTRAP` ZMQ message (room, decode ip, rank_port) to each prefill bootstrap target (`common/conn.py:2099-2113`). The request sits in the **prealloc queue** with `waiting_for_input=False`, so `pop_preallocated` skips it (`decode.py:1362-1363`). **No decode KV, req_to_token slot or metadata buffer is taken yet.**
2. **Prefill, admission.**
   - Requests in `Bootstrapping` are admitted unconditionally. The `or policy == "prefill_complete"` at `prefill.py:519-524` bypasses the attempt cap.
   - Each admitted request takes a prefill metadata buffer (`ensure_metadata_buffer`, `prefill.py:527-528`) and enters the waiting queue as optimistic (`pending_bootstrap=True`, `prefill.py:392`).
   - Prefill's `DeferredBootstrap` table (`common/bootstrap.py:27-112`) records decode's endpoint when the BOOTSTRAP message arrives (`register`, `common/conn.py:604-608`).
3. **Prefill, forward.**
   - A chunked request keeps computing even though it is not bootstrapped (no yield, `prefill.py:1395-1404`).
   - When the last chunk finishes, `mark_prefill_complete()` runs (`prefill.py:1049-1053`). It stamps `_prefill_complete_time` and, through `deferred_bootstrap.complete`, sends decode exactly one `KVPoll.WaitingForInput` status message (`common/conn.py:1847-1855`, `:579-590`, `bootstrap.py:52-57`).
   - Because `pending_bootstrap` is still True, `send_kv_chunk` is skipped (`prefill.py:1054-1055`). The request is **parked in `disagg_prefill_inflight_queue`** holding its device KV (`prefill.py:1181-1186`, comment "Parked: prefill finished before bootstrap completed").
4. **Decode, readiness.**
   - `apply_prefill_status` maps WaitingForInput to status only (`common/conn.py:788-792`).
   - `_poll` clears `_prefill_wait_start` and restarts `init_time` (`common/conn.py:2132-2138`), and `waiting_for_input` becomes True.
   - Only now does `pop_preallocated` reserve decode KV, with the same budget and admission logic as early (`decode.py:1355-1560`). It sends metadata, and the request moves to the **transfer queue**.
5. **Prefill, unpark.** The inflight-queue poll sees WaitingForInput:
   - `handle_pending_bootstrap` runs `finalize_bootstrap` (metadata/sender init) and `checkpoint_disagg_prefill` (`prefill.py:1349-1364`).
   - Then `send_kv_chunk(req, last_chunk=True)` sends the **whole KV in one go**, after compute rather than overlapped with it (`prefill.py:1183-1185`).
6. **Success.** Prefill runs `release_kv_cache(..., checkpoint=True)` and `tree_cache.finish` (`prefill.py:1192-1197`), then frees the metadata buffer (`prefill.py:1240-1243`). With deferred allocation, decode's mooncake `poll` holds `Transferring` until worker cleanup removes the staging entry (`mooncake/conn.py:3070-3078`).

**Rebootstrap (PD retract) ordering differs:**
- `prefill_complete`: the recompute is submitted at `_init_receiver`, before allocation (`decode.py:1022-1033`).
- `early`: it is submitted after `send_metadata` (`decode.py:1744-1751`).

## (4) Memory consequences

**Decode:**
- The device KV and req slot are reserved only for requests whose prefill has finished. Requests still queued or computing on prefill hold no decode pages, so the pool is not tied up during prefill time.
- `pre-allocated usage` counts only transfer-queue requests (`decode.py:1842-1847`). [inferred] Under prefill_complete it should therefore be close to the in-transfer working set, not the whole prefill backlog.
- The admission formula is unchanged: `_required_admission_tokens` = max(alloc + reserved, prefill_len + min(max_new, CLIP) − retractable) (`decode.py:485-494`).
- Decode metadata buffers = `req_to_token_pool.size × 2` (`managers/scheduler.py:1504-1513`).

**Prefill:**
- Each finished request **keeps its device KV locked** until decode has allocated and the transfer has succeeded. It is released only on Success (`prefill.py:1192-1195`). Under write_back it is checkpointed into the tree at unpark (`prefill.py:1363`, `:594-599`).
- So parked requests consume prefill device KV and prefill metadata buffers. Prefill metadata buffers = `max_running_requests × 2` (`managers/scheduler.py:1550-1553`).
- When the pool or buffers are short, `pop_bootstrapped` stops admitting: `continue` on no buffer (`prefill.py:527-528`).
- [inferred] If decode is the bottleneck, parked requests pile up on prefill, which reduces prefill KV available for new prefills and for evictable radix/HiCache L1 pages. I have not measured the size of this effect. Prefill `#inflight-req` will show it.
- Host HiCache size (370 GB) is not referenced by the policy code. The only coupling is the write_back requirement.

## (5) Timeouts and env knobs

Defaults are in `environ.py`. Both legs currently set WAITING=1800 and BOOTSTRAP=1800 (`H/dryrun.q4-137.yihou.txt`).

| Knob | Default | Meaning under prefill_complete |
|---|---|---|
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 300 (:715) | **Decode, phase 1:** from sending BOOTSTRAP until prefill-complete readiness. This covers prefill queueing plus compute. On expiry: abort, `"Timed out waiting for prefill completion"` (`common/conn.py:2126-2131`). **Decode, phase 2:** from readiness through allocation and transfer (`init_time` reset at :2138, `_check_waiting_timeout` :2345-2368): `"Request <room> timed out after Xs in KVPoll.WaitingForInput"`. |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 300 (:711) | **Prefill:** measured from `_prefill_complete_time` instead of `init_time` (`common/conn.py:1939-1958`). Decode must allocate and send metadata within this window, otherwise `"Request <room> timed out after Xs in KVPoll.Bootstrapping"` and Failed. |
| WAITING + BOOTSTRAP | n/a | TTL of the prefill `DeferredBootstrap` table for unmatched or tombstoned rooms (`common/conn.py:380-384`, `bootstrap.py:45,109`) |
| `SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT` | 1 s (:712) | ZMQ PUSH SNDTIMEO under defer. Early uses a hard-coded 30 s (`common/conn.py:1456-1461`). On ZMQError the decode BOOTSTRAP send aborts (`:2112-2113`). |

## (6) How to tell the policy is active

- **Decode log line** (`managers/scheduler_components/metrics_reporter.py:998-1000`): `pre-allocated usage`, `#prealloc-req`, `#transfer-req`.
  - Under prefill_complete, `#prealloc-req` includes requests still being prefilled (not waiting_for_input).
  - `#transfer-req` and `pre-allocated usage` cover only post-prefill requests.
  - [inferred] Compared with early, expect higher `#prealloc-req` and lower `pre-allocated usage` at the same load.
- **Prefill log line** (`metrics_reporter.py:740-746`): `#bootstrap-req`, `#inflight-req` (includes parked requests), `#optimistic-req` (batch requests with `pending_bootstrap`).
  - [inferred] Under prefill_complete, nearly every request is optimistic, and `#bootstrap-req` should stay about 0.
- **Startup:** the prefill `optimistic_prefill_attempts` resolves to 1 (`pd_disaggregation_hook.py:49-54`). The resolved server-args dump should show this.
- **Should NOT appear:**
  - `"exhausted optimistic prefill attempts falling back to bootstrap queue"` is suppressed under this policy (`prefill.py:1716-1727`).
  - `"optimistic prefill yielded"` appears only via `should_force_retry` or the abort paths.
- **Failure signatures:** `policy mismatch` (`common/conn.py:2060-2063`), `requires peer attention TP1/CP1/PP1` (:2068), `Timed out waiting for prefill completion` (:2130), `timed out ... in KVPoll.Bootstrapping` (:1956) or `in KVPoll.WaitingForInput` (:2361), and `Duplicate decode bootstrap room` / `Duplicate or cancelled bootstrap room` (:2032-2036, :1786-1790).

**Planned-run flags:** `PREFILL_EXTRA_ARGS_ADD="--disaggregation-decode-allocation-policy prefill_complete"`, `DECODE_EXTRA_ARGS_ADD="--disaggregation-decode-allocation-policy prefill_complete"`, `PREFILL_HICACHE_WRITE_POLICY=write_back`, `PREFILL_HICACHE_SIZE=370`, `DECODE_MEM_FRACTION=0.96`, `DECODE_MAX_RUNNING=96`, `PREFILL_MEM_FRACTION=0.88`, `CONC=48`. Keep the host-receive threshold at 0. Run with `DRY_RUN=1` first and grep the emitted command for the policy flag on both legs.

=====VERIFY=====
# Adversarial verification of the prefill_complete report

I re-read every cited location on spur006 (read-only). `S` means `.../image-9d755dd6/sglang/srt/` and `H` means `.../v0521dbg_20261010/harness/`.

## Overall verdict

The report holds up on almost every decision-relevant point. One attribution is wrong, and it matters for prefill memory: the "hold Transferring until cleanup" step is on the **prefill sender**, not on decode. The other problems are a few harness line numbers that are off by one to three lines, and two missing details: readiness can also be sent from `register()`, and admission of Bootstrapping requests has extra gates. Five important facts were missing, listed at the end.

## Blockers

- **A. Prefill HiCache is `write_through`, and `prefill_complete` rejects it. CONFIRMED.**
  - The check is at `S/arg_groups/pd_disaggregation_hook.py:230-238` and the raise at `:244-247`. The error text matches.
  - The harness default is `H/config.yihou.v0521-base.sh:98`, with a second default at `H/bench-harness/config.sh:56`.
  - The q4 dryrun emits `--hicache-write-policy write_through` (`H/dryrun.q4-137.yihou.txt:19`).
  - The flag is emitted at `H/bench-harness/engine.sh:305`. CONFIRMED.
  - The run therefore needs `PREFILL_HICACHE_WRITE_POLICY=write_back`.
  - Optimistic prefill still survives: `write_back` is in the allowed set at `S/arg_groups/serving_hook.py:477-480`.
- **B. The harness never passes the policy flag. CONFIRMED, with corrected line numbers.**
  - Grep finds no `allocation` anywhere in `engine.sh`, `config.sh` or the base config.
  - The hooks `PREFILL_EXTRA_ARGS_ADD` / `DECODE_EXTRA_ARGS_ADD` are at `base.sh:243-244`. CONFIRMED.
  - In `engine.sh`, `extra_args` is assigned at **:50 and :63** (not 48/61), word-split and appended at **:321-323** (not 318-321).
  - The entrypoint `python3 -m infera.engine.sglang` is at **:252** (not 251).
  - Whether the `infera` wrapper forwards these args unchanged is **UNVERIFIED**: the wrapper source is not in the extracted tree. [inferred] Other extra args such as `--hicache-size` and `--disable-custom-all-reduce` already travel this path (`base.sh:237-238`).

## (1) Required flags on each leg

- **Both legs need the flag. CONFIRMED.** The help text is at `S/arg_groups/fields/disagg.py:59-64`.
- **Prefill registers its policy with the bootstrap server. CONFIRMED.** `S/disaggregation/common/conn.py:1377-1379`.
- **HTTP 400 when prefill ranks disagree. CONFIRMED.** `common/conn.py:2553-2558`.
- **A mismatch between legs fails each request rather than stopping startup. CONFIRMED.**
  - The mismatch check is at `common/conn.py:2056-2068`.
  - The "requires peer attention TP1/CP1/PP1" message is at `:2068`.
- **Optimistic prefill is auto-resolved to 1. CONFIRMED.**
  - `pd_disaggregation_hook.py:47-55` sets it.
  - `serving_hook.py:468-505` zeroes it on pp>1 or an unsupported HiCache policy, and then raises at `:497-505`.

## (2) Incompatibilities and exact error text

- **The startup-check table. CONFIRMED, every row**, at `pd_disaggregation_hook.py:204-247`. The defaults it relies on are also confirmed:
  - `disaggregation_decode_host_receive_threshold` is 0.0 (`disagg.py:179-182`).
  - The decode radix cache flag defaults to False (`disagg.py:85-88`).
  - `SGLANG_DISAGG_STAGING_BUFFER` defaults to False (`environ.py:803`).
  - `SGLANG_RUST_SERVER` defaults to False (`environ.py:1871`).
  - Neither env var appears in the q4 dryrun's env list.
  - The transfer backend in the q4 dryrun is `mooncake`.
  - Decode has HiCache off: `HiCache=0` in the q4 dryrun (line 25).
- **The DSA-only state-type check. CONFIRMED.**
  - The raise is at `common/conn.py:375-379`.
  - `DSA_TAIL` is appended only when tail pointers exist (`disaggregation/utils.py:1564-1574`).
  - `get_compress_tail_buf_infos` returns empty unless `kpool_use_compress` is on (`mem_cache/memory_pool.py:5273-5274`). That flag is computed at `:5219`.
  - `index_kpool_compress` defaults to False (`configs/model_config.py:446-447`).
  - The model actually used, `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4/config.json` (`base.sh:74`), has no `index_kpool*` keys. The `JSON_MODEL_OVERRIDE_ARGS` value only sets `index_share_for_mtp_iteration` (`base.sh:228`).
  - So the conclusion "state_types is `[DSA]` only" is now supported by the code read. It still needs the first launch to prove it.
- **The mounted `conn.yihou-regbase.py`. CONFIRMED compatible.**
  - The diff has exactly three hunks: `82a83,133`, `382a434` and `507a560`.
  - All three only add `_yihou_log_regbase` diagnostics, wrapped in try/except.
  - The deferred-allocation branch is preserved, at line 3128 in the patched file.

## (3) Request lifecycle

- **Overall flow. CONFIRMED.** Specifically:
  - Under `early`, `init` sets `WaitingForInput` immediately (`common/conn.py:2096-2098`). Under `prefill_complete` it sends `BOOTSTRAP` instead (`:2099-2113`).
  - `pop_preallocated` skips requests that are not `waiting_for_input`, and stops when no metadata buffer is left (`S/disaggregation/decode.py:1361-1366`).
  - Prefill sets `pending_bootstrap=True` (`prefill.py:392`).
  - `mark_prefill_complete` is called before the `pending_bootstrap` check (`prefill.py:1049-1055`).
  - The parked-request unpark path is at `prefill.py:1180-1187`. Success handling, which releases the KV, is at `:1191-1199`.
  - `handle_pending_bootstrap` is at `prefill.py:1349-1366`.
  - Under `prefill_complete`, a chunked request does not yield (`prefill.py:1394-1404`).
  - The fallback to the bootstrap queue is suppressed (`prefill.py:1719-1726`).
  - The ordering of the rebootstrap recompute (`decode.py:1022-1033` vs `:1744-1751`) is correct.
- **Missing detail: readiness can also come from `register()`.** If the decode `BOOTSTRAP` message arrives *after* prefill completes, `deferred_bootstrap.register()` sends the readiness message instead of `complete()` (`common/bootstrap.py:66-75`, `:42-53`; caller at `common/conn.py:604-608`).
- **"Bootstrapping requests are admitted unconditionally" is PARTIAL.** Admission is also gated on:
  - `not req.is_retracted`;
  - the SWA / write-through branching clause;
  - having a metadata buffer.

  See `prefill.py:516-528`.
- **WRONG: "decode's mooncake poll holds Transferring until worker cleanup".**
  - The `poll` at `S/disaggregation/mooncake/conn.py:3066-3080` belongs to **`class MooncakeKVSender` (line 3010)**, which runs on the prefill side.
  - The hold therefore delays **prefill** from seeing Success. That delays `release_kv_cache` (`prefill.py:1194`) and the metadata-buffer release.
  - The effect is to lengthen prefill KV holding, not decode.

## (4) Memory consequences

- **Decode side. CONFIRMED.**
  - `num_tokens_pre_allocated` counts only the transfer queue (`decode.py:1843-1847`).
  - The admission formula is at `decode.py:485-494`.
  - Decode metadata buffers = `req_to_token_pool.size × 2` (`managers/scheduler.py:1503-1513`).
- **Prefill side. CONFIRMED, with one correction.**
  - Prefill metadata buffers = `max_running_requests × 2` (`scheduler.py:1549-1553`).
  - Prefill max_running was **64** in q4 (`H/dryrun.q4-137.yihou.txt:19`), so prefill has **128** buffers. That is the hard cap on bootstrap-admitted plus parked plus in-transfer requests per prefill DP rank.
  - Device KV is held until the sender reports Success. That includes the sender-side cleanup hold described above.
- **Checkpoint path. CONFIRMED.** Under `write_back`, `checkpoint_disagg_prefill` uses `checkpoint_kv_cache`. The `advance_unpublished_req` path is only used under `write_through` (`prefill.py:594-599`; `_uses_write_through_cache` at `:126-131`). The checkpoint runs at every chunk (`prefill.py:1384`) and again at unpark (`:1363`).
- **The magnitude of parked-request buildup is UNVERIFIED.** It has not been measured.

## (5) Timeouts and env knobs

- **Defaults. CONFIRMED.** `environ.py:711` (bootstrap 300), `:712` (ZMQ send 1 s), `:715` (waiting 300).
- **Usage sites. CONFIRMED.**
  - Phase-1 decode timeout and the `init_time` reset: `common/conn.py:2124-2138`.
  - Prefill bootstrap timeout measured from `_prefill_complete_time`: `:1939-1960`.
  - Waiting-for-input timeout: `:2345-2358`.
  - `DeferredBootstrap` TTL = WAITING + BOOTSTRAP: `:380-384`.
  - ZMQ `SNDTIMEO`: `:1458-1462`.
- **Missing: one harness variable sets both timeouts.** `engine.sh:215-216` sets `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` from the same `${BOOTSTRAP_TIMEOUT:-1800}`. They cannot be tuned independently without an extra env entry.

## (6) How to tell the policy is active

- **Log fields. CONFIRMED.**
  - The prefill line has `#bootstrap-req`, `#inflight-req` and `#optimistic-req` (`managers/scheduler_components/metrics_reporter.py:740-746`).
  - The decode line has `pre-allocated usage`, `#prealloc-req` and `#transfer-req` (`metrics_reporter.py:997-1000`).
- **Failure strings. CONFIRMED** at:
  - `Duplicate decode bootstrap room`: `common/conn.py:2034`.
  - `Duplicate or cancelled bootstrap room`: `:1789`.
  - `timed out after Xs in KVPoll.Bootstrapping`: `:1958`.
  - `timed out after Xs in KVPoll.WaitingForInput`: `:2358`.
- **The expected behaviour of those numbers is still [inferred].** It has not been measured.

## Important missing facts

1. **q4 already used most of the planned settings.** q4 (config `H/config.yihou.mc4340n.sh`, dryrun header line 2) already ran CONC=48, decode `--mem-fraction-static 0.96`, decode `--max-running-requests 96`, prefill mem 0.88, prefill max_running 64 and chunk 32768. Relative to q4, the planned run changes only three things: the allocation policy, `write_through` to `write_back`, and `--hicache-size` 450 to 370. That makes q4 the natural baseline to compare against.
2. **The prefill-side hold in `MooncakeKVSender.poll`** (`mooncake/conn.py:3066-3080`) extends prefill KV holding under this policy. This is the correction to section (3).
3. **The prefill metadata-buffer cap is 128** (64 × 2). When it is used up, `pop_bootstrapped` stops admitting (`prefill.py:527-528`).
4. **One timeout knob drives both env vars** (`engine.sh:215-216`).
5. **The `infera.engine.sglang` wrapper's argument forwarding is UNVERIFIED.** Confirm it from the live server-args dump at startup: look for `disaggregation_decode_allocation_policy='prefill_complete'` on both legs, and `optimistic_prefill_attempts=1` on prefill.
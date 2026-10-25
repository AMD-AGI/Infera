# B (verified)

# TASK B verification: decode radix cache on llr7

I re-checked every claim against the code and logs. Sources:
- **sglang:** the image `infera-sglang:v0522-fdbafb60` (`S=/sgl-workspace/sglang/python/sglang`), read through `docker run --rm` on 135. The script is at `/shared_nfs/yihou_glm52/llrepro_20261023/coderead.yihou/vrfy_rdx.sh`. I also used the local copy in `/tmp/yihou/coderead_rdx/`.
- **Router:** Infera `origin/main` (the copy in `/tmp/yihou/coderead_rdx/router/`). The image's router commit is UNKNOWN.
- **Logs:** the llr7 logs.

Nothing was launched, modified or deleted. No `yihou-llr-*` container was touched.

## Verdict: CONFIRMED, with corrections
The flag can be enabled on llr7 without code changes, but only together with `--session-affinity both`. Under the current router flags the decode DP rank is chosen without looking at prefix overlap, so decode-side hits would be rare.

## Claim-by-claim

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| 1 | EAGLE is allowed with decode radix | CONFIRMED | `speculative_hook.py:163-172` rejects only algorithms other than None, DSPARK, EAGLE and EAGLE3. The kit uses EAGLE (`worker.sh:77`). |
| 2 | DP-attention only triggers a warning; hisparse and the fake backend are rejected | CONFIRMED | `pd_disaggregation_hook.py:99-125`, which sets `disable_radix_cache=False`. |
| 3 | The SWA/DSv4/SWA-compress rejections don't apply | CONFIRMED | `kv_cache_builder.py:249-290`: `validate_decode_radix_cache` returns early when the model is not hybrid SWA. |
| 4 | `prefill_complete` rejects decode radix; llr7 uses `early` | CONFIRMED | `pd_disaggregation_hook.py:220-222`; decode log shows `allocation_policy': 'early'`. |
| 5 | The flag definition | CONFIRMED | `fields/disagg.py:85-88`. |
| 6 | An explicit flag skips Infera's auto-append | CONFIRMED | `/opt/infera/infera/engine/sglang/args.py:338-342`, via `_DECODE_RADIX_CACHE_FLAG not in remaining`. |
| 7 | `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC` only gates Infera's auto-append | CONFIRMED (refined) | `args.py:24,344-346`; grep finds nothing in sglang. The auto path is off anyway, because the kit decode passes `--no-enable-kv-events --kv-events off` (`worker.sh` decode args). |
| 8 | llr7 decode log state | CONFIRMED | Log shows "Radix cache is disabled for decode server", `disable_radix_cache': True` and `page_size': 64`. |
| 9 | Decode match/lock path; rebootstrap requests skip it | CONFIRMED | `decode.py:837-856` and `decode.py:1469-1515`; the skip is at 1470-1473. |
| 10 | Admission counts evictable pages and evicts on demand | CONFIRMED | `decode.py:2050-2059` and `decode.py:2222-2242`, with the "Eviction insufficient" warning. |
| 11 | Prefix-length protocol | CONFIRMED | `decode.py:1769-1777` → `mooncake/conn.py:2952-2960` → `prefill.py:415-428`. Prefill's send loop starts at `start_send_idx` (`prefill.py:1490`). The early cached-prefix send also respects it (`prefill.py:1437-1443`). |
| 12 | Prefill still computes the full prompt; only the transfer shrinks | UNVERIFIED (INFERENCE) | I found no compute-skip use of `decode_prefix_len` in `prefill.py`, but I did not read the PP path to the end. |
| 13 | DSA index-K is suffix-only on both sides | CONFIRMED | Prefill: `prefill.py:1573-1583`. Decode: `decode.py:1664-1674`. |
| 14 | EAGLE draft KV shares the transfer buffers | CONFIRMED | `decode.py:623-639`. The comment there says draft KV shares target virtual ids. That draft KV is reused correctly for a cached prefix remains INFERENCE, so a smoke run with real acceptance is still required. |
| 15 | "No reference to `index_share_for_mtp_iteration` in disaggregation code" | **WRONG** | `disaggregation/utils.py:130-138` (`get_dsa_seed_metadata_dim`): when the flag is set on a DSA model, PD carries DSA MTP top-k "seed" metadata. That is per-request aux metadata, not prefix KV pages. Whether it is still valid for a decode-radix-matched prefix is UNKNOWN and needs a read of the seed-metadata producer and consumer. |
| 16 | One radix tree per DP rank | CONFIRMED (from logs) | "Init Unified Radix Cache" appears 4 times in the decode log. |
| 17 | `enable_decode_hicache` needs both the flag and HiCache | CONFIRMED | `scheduler.py:515-518`. llr7 decode has no HiCache, so prefixes stay on device (L1) only. |
| 18 | "Page-granular match, see `decode.py:572-585`" | **Citation WRONG**; claim UNVERIFIED | Those lines are prealloc `ceil_align` accounting, not matching. `l1_prefix_len = len(prefix_device_indices)` (`decode_hicache_mixin.py:64-75`). Page-aligned matching is plausible: prefill's comment at `prefill.py:1574-1575` says "decode_prefix_len, page-aligned". I did not read the radix match itself. |
| 19 | Retraction backup is always `cpu_tensor` on HIP | CONFIRMED | `kv_cache_builder.py:210-223`. |
| 20 | Decode registers `dp_size` with `dp_rank` None and the router expands it to `#dp0..3` | CONFIRMED | `worker.py:161-164`, `pool.rs:129-156`. |
| 21 | The rank travels in the `X-Data-Parallel-Rank` header and sglang honours it | CONFIRMED | `dp.rs:15,27`, `disagg.rs:416`, `serving_base.py:266-290`, `data_parallel_controller.py:839-841`. |
| 22 | The bootstrap room is aligned only to the prefill rank | CONFIRMED | `dp.rs:54-57`, `disagg.rs:151`. |
| 23 | With `decode-input-demand on`, the decode pick is the least-demand target and ignores overlap | CONFIRMED (one word wrong) | `policy.rs:776-792`. The report says "a random decode rank"; it is the least-demand rank, not random. The effect on decode hits is the same. |
| 24 | Affinity mode `prefill` pins prefill picks only | CONFIRMED | `session_affinity.rs:137-139`. |
| 25 | `--session-affinity both` pins the decode target including its DP rank | CONFIRMED | `session_affinity.rs:187-203`, which matches on `route_key` and the worker. Demand is still booked, because the pick goes through `choose` with one target. Live config is `session_affinity: Prefill`, `decode_input_demand: On`, TTL 3600. |
| 26 | "Sticky sessions bypass demand balancing for an hour" | **Imprecise** | The TTL counts *idle* time (`idle_since`, `session_affinity.rs:181`). The binding lasts until the session has been idle for 3600 s, not for one hour in total. |
| 27 | The later `--session-affinity` value wins | CONFIRMED on `origin/main` | Python argparse `store` keeps the last value (`infera/server/args.py:115-121`), and `launch_rust.py:167-172` passes a single value to Rust. The image commit is unknown, so still check the router Config log line. |
| 28 | Finished requests' KV is checkpointed into the tree and unlocked (evictable) | CONFIRMED | `common.py:395-428`, `batch_result_processor.py:117-130` (`checkpoint=True`). |
| 29 | Reported token usage excludes evictable KV | CONFIRMED | `pool_stats_observer.py:236-239`. |
| 30 | Decode-step allocation evicts from the tree first | CONFIRMED | `allocation.py:152-158` → `common.py:282-288`. |
| 31 | Prefill prefix hit 0.950 | CONFIRMED | Prefill0 log: 5.198e9 cached / (2.751e8 new + 5.198e9 cached) = 0.9497. |
| 32 | About 2.97 M tokens per rank | **Corrected** | The log has `max_total_num_tokens=2858496`, about 2.86 M. That this figure is per rank is INFERENCE. |
| 33 | "Decode `cached_tokens` does not show decode hits" | **Partly WRONG** | `decode.py:2630-2637` seeds `already_computed` with prefill's count. `prepare_for_prebuilt` then adds decode reuse only beyond what prefill counted. With prefill at about 95 %, decode hits are mostly hidden, so a small overlay log of `l1_prefix_len` / `decode_prefix_len` in `pop_preallocated` is still the right measurement. |

## Corrected recommendations

1. **Run llrX = llr7 + 2 changes:**
   - `DECODE_EXTRA_ARGS` = llr7's current value plus `--disaggregation-decode-enable-radix-cache`.
   - `ROUTER_EXTRA_ARGS='--session-affinity both'`. Confirm that the router log shows `session_affinity: Both`.
2. **Correctness smoke before measuring.** Leave `SIM_ACCEPT_LEN` unset, run multi-turn traffic, and compare real acceptance with radix off vs on. The new risk is item 15: `index_share` seed metadata on reused prefixes. Draft KV and index-K reuse are the other things this catches.
3. **Risks:**
   - The decode radix path is marked EXPERIMENTAL with DP-attention.
   - The DSA + EAGLE + index_share + Mooncake-PP4 combination has not been exercised.
   - The fully-matched-prompt case (0 suffix pages) is UNKNOWN.
   - Sessions stay sticky until they have been idle for 3600 s, so DP ranks can become imbalanced.
   - Evict and match add CPU work to the decode scheduler loop.
   - The KF overlay has not been validated with radix enabled.
4. **Metrics to watch:**
   - **Pinning:** router decode session hits vs selections.
   - **Decode hit:** the overlay log from item 33; otherwise infer it from transfer_kv p50/p90 and transfer bytes.
   - **Eviction:** "Eviction insufficient" warnings.
   - **Rank balance:** `#running-req` and token usage per DP rank.
   - **Outcome:** TTFT p50/p90, TPOT, qps and the ISL mix, not the score alone.

# C (verified)

## Task C verification: prefill PP overlap (image fdbafb60, sglang `e9b22a0818`)

I re-checked every claim against the image source, the kit and the llr7 log. Nothing was run, nothing was deleted, and no `yihou-llr-*` container was touched. The scripts and outputs I wrote are `/shared_nfs/yihou_glm52/llrepro_20261023/coderead.yihou/verifyC{,2,3}.{sh,out}`. Source paths below are relative to `python/sglang/srt/` in the image.

The report's main conclusions hold. Four claims are wrong or incomplete, and several file paths and line numbers were off; corrected versions are given below.

### What was wrong
- **PP spec relay and PD:** the report said only that `SGLANG_ENABLE_PP_SPEC` rejects DP attention. In fact it is rejected for any disaggregation mode (`arg_groups/validation_hook.py:138-146`, assert `disaggregation_mode == "null"`). So in PD, PP + EAGLE on the prefill can only run through the non-relay branch (`:160-163`).
- **`mem_cache/allocator/paged.py:163`:** this `torch.cuda.synchronize()` is a one-time warm-up in `__init__` (`paged.py:155-166`). It is not a candidate for the hot path.
- **Loop line numbers:** commit-proxy-send is at 305, not 315. The release-consensus blocking recv is at 330, before `d2h_event.synchronize()` at 334, not at 337-342.
- **E1 risk wording:** the report said `PREFILL_EXTRA_ARGS` "would carry a second `--json-model-override-args`". Adding `--pp-async-batch-depth 1` adds a different flag. The real risk is keeping the existing escaped JSON intact (`engine-config.env:16`).

### Item 1: which loop runs, overlap, synchronize calls
- **Loop dispatch: CONFIRMED.** PREFILL with pp>1 always runs `event_loop_pp_disagg_prefill` (`managers/scheduler.py:6101-6103`). The overlap prefill loop is only reached when pp_size == 1 (`:6104-6105`).
- **Overlap is forced off: CONFIRMED.** `arg_groups/overrides.py:1625-1629` sets it whenever pp>1. The llr7 prefill log has the warning at 02:50:36.
- **Microbatch ring size: CONFIRMED.** `pp_loop_size = pp_size + pp_async_batch_depth` (`managers/scheduler_pp_mixin.py:558-561`).
- **Loop order: CONFIRMED.** Line numbers are corrected here (all `scheduler_pp_mixin.py`):

| Step | Line |
|---|---|
| `_process_hicache_events` | 284 |
| Blocking proxy recv | 296 |
| Output exchange, depth>0 variant | 298-304 |
| Commit proxy send | 305 |
| Launch | ~306-313 |
| Output exchange, depth==0 variant | ~314-320 |
| Release-consensus send | ~321-325 |
| Blocking pyobj recv | 330 (gloo `cpu_group`, `:765-785`) |
| Event sync | 334 |
| Async sends to the next stage | ~346-351 |

- **Synchronize calls: CONFIRMED as event syncs, not device-wide.**
  - `d2h_event` at mixin 189, 334 and 514.
  - `writing_check(write_back=True)` uses `ack.finish_event.synchronize()` after `start_writing(wait_for_forward=True)` (`mem_cache/unified_radix_cache.py:3418-3430`; `managers/cache_controller.py:835-847`).
  - Non-blocking poll: `urc:3445`.
  - `disaggregation/prefill.py:882` (`copy_done.synchronize()`).
- **Who the event sync waits on: CONFIRMED as code, the cross-stage serialization remains INFERENCE.**
  - The recv path does `copy_stream.wait_stream(schedule_stream)` (`mixin:1489-1491`).
  - Without `SGLANG_PP_COMM_OVERLAP`, `pp_comm_stream_ctx` is a `nullcontext` (`:588`). It is used for commit, send and recv (`:694`, `:863`, `:904`), and the loop runs under `StreamContext(schedule_stream)` (`scheduler.py:1955`).
  - So the d2h sync also covers the queued NCCL sends and recv. That this produces the serial cross-stage handshake is still an inference.
- **`writing_check(write_back=True)` call sites: CONFIRMED.** `urc:806`, `urc:1001` (once per evicted leaf), and `urc:1618` and `:1624`.
- **HiCache gloo chain: CONFIRMED.** I added the actual call path, which the report did not give.
  - Mixin 284 calls `scheduler.py:3810-3821` `_process_hicache_events`, which calls `tree_cache.check_hicache_events` (`urc:3589`).
  - That runs `_drain_async_work()` (`urc:3609`, defined at `:354-364`), then `_sync_hicache_ready_counts()` (`:3619`), then `_all_reduce`, then `_pp_sync` (`:366-398`).
  - `_pp_sync` does a blocking `torch.distributed.recv` from pp_rank-1 on `pp_cache_group`, which is the gloo `cpu_group` (`urc:291`; `mem_cache/kv_cache_builder.py:387`). This happens once per microbatch iteration.
- **Device-wide `synchronize()` in `index_key_cache.py:115-146`, `dsa_cache_layer_split.py:160-191, 558-599`, and `memory_pool.py` `get_cpu_copy` (defs at 1222, 2727, 4054, 4967, 5496): the lines exist (CONFIRMED). Whether any is on the llr7 hot path is UNKNOWN.**

### Item 2: async depth and micro-batch options
- **`pp_async_batch_depth`: CONFIRMED.** Field is at `arg_groups/fields/parallel.py:69-72`, default 0. The CLI flag `--pp-async-batch-depth` comes from the generic `_` to `-` conversion at `arg_groups/arg_utils.py:290` (the report had the path wrong).
  - llr7 does not set it: the prefill command line has no such flag, and `engine-config.env:16` `PREFILL_EXTRA_ARGS` holds only the json override.
- **`pp_max_micro_batch_size`: CONFIRMED.** Field at `parallel.py:66-68`. It defaults to max(256 // 4, 1) = 64 (`scheduler.py:1217-1223`) and is used as a budget at `scheduler.py:3971`. Whether it ever binds at an 8k chunk is UNKNOWN.
- **The two env switches: CONFIRMED.** Both default False (`environ.py:673, 676`).
  - `SGLANG_PP_COMM_OVERLAP` is wired at `mixin:584-599`. The llr7 log has 0 "communication overlap enabled" lines, so it was off.
  - `SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM`: gate at `mixin:47-56`, skip result at `:943-963`.
  - The skip check is at `managers/scheduler_components/batch_result_processor.py:612-630`, not `managers/batch_result_processor.py`.
- **llying's result: CONFIRMED.**
  - Commit e8227d61, as described: 32,235 vs 30,071, prefill capacity +8.5 %, step time 481 to 414 ms, image `v0522-ex-3ad7527c` (`llying/prefill-car-20261023/README.zh-CN.md:4, 151, 159, 161`).
  - The handoff is on origin only, at commit 7a9ba815: `llying/HANDOFF-20261024.zh-CN.md:13` (correctness GSM8K 0.9682 / 0.9704, not in PR #210) and `:97` (depth 2 untested).
  - New in 7a9ba815: llying now limits the async test to 2P1D C176 (`:17-22`).

### Item 3: why the kit disables overlap
- **CONFIRMED: the reason is not stated in the kit.**
  - `worker.sh:86-91` comments only on the layer partition.
  - The flag came in with `llying/2p1d-v0522-20261020/config/kit-v6-pp4.diff:55`.
  - llying's note "PP 要求 `--disable-overlap-schedule`" is at `llying/2p1d-v0522-20261020/README.zh-CN.md:~417`.
- **CONFIRMED: the flag is redundant, because of the override at `overrides.py:1625`.**

### Correction to the brief
- **Chunk size: CONFIRMED with detail.** The llr7 command line contains both `--chunked-prefill-size 32768` (`worker.sh:69`) and a later 8192 (`worker.sh:89` with `PP4_CHUNK=8192`). The server args show `chunked_prefill_size=8192` and `max_prefill_tokens=8192`. Policy is `shortest-prefill-first`.
- **Eager prefill: CONFIRMED.** The log has 37,904 "cuda graph: False" lines.

### Item 4: experiments, one variable each against llr7
- **E1: `PREFILL_EXTRA_ARGS += --pp-async-batch-depth 1`.**
  - Ring goes from 4 to 5 microbatches.
  - The depth>0 path moves the output exchange before the launch (`mixin:298-304`).
  - Risks: one more microbatch of request slots and KV in flight; a different image from llying's; keep the escaped JSON intact; rerun the correctness gate.
- **E2: `SGLANG_PP_COMM_OVERLAP=1`, prefill only.**
  - `worker.sh:124-125` only passes `DECODE_ENV_EXTRA` to the decode, so the kit needs a prefill-env hook.
  - Rationale (code CONFIRMED): `_pp_wait_forward_dependencies` makes the forward stream wait on the whole `schedule_stream` (`mixin:711-718`).
  - Check for the log line "PP tensor communication overlap enabled".
  - Risks: the path is untested on ROCm/RCCL. The `torch.cuda.is_available()` and `str(device).startswith("cuda")` gate is presumably true on ROCm (INFERENCE). The eager prefill does not use the graph fence path (`:697-702`). Rerun the correctness gate.
- **E3: `SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1`.**
  - In the llr7 log, 14,744 of 37,904 "Prefill batch" lines are "#new-seq: 1, #new-token: 8192".
  - That is an upper bound on eligible batches: last chunks and logprob requests are excluded by the gate, so the real share is UNKNOWN.
  - After the run, grep for the "PP non-skip output comm" warning. In llr7 the count is 0, but the check returns early when the env is off, so that zero means nothing.
- **E4: `--pp-async-batch-depth 2`**, only after E1 holds.
- **Code-level candidates, not flag-level:**
  - The per-iteration gloo chain (`urc:3609-3619`, `354-398`).
  - The per-leaf write_back sync (`urc:1001`).
  - Measure both first with `observe_backup_duration` (`urc:3466`) and a profiler stack on the 88-101 ms gaps.
# R1 verification (adversarial) — decode admission / prealloc / retraction, sglang v0.5.21 image 9d755dd6

Target: `research/R1-decode-admission-code.yihou.md` (not edited). SRC = `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt`.
Tags: [M] first-hand (code line / log grep / command run here), [S] second-hand, [I] inference.

## Adversarial verification

### Verdict

No R1 claim was refuted. The code model of admission, the count caps, the retraction path, the `prefill_complete`
gates and the X1 numbers all re-derive exactly. Two gaps matter for the proposed share-cap patch (G1, G2 below), and
one statement in the "22k" summary is imprecise (R-a).

### Confirmed claims (re-derived first-hand)

1. decode.py md5 `bd1e5fe776e75f5dfe2bb6b27c5bf7af` [M `md5sum`]. No `import os` at lines 21-33. `ceil_align` is
   already imported (decode.py:125) [M].
2. `process_decode_queue` (decode.py:3201-3240): retracted requests are resumed first (3213-3214). If any are still
   retracted it returns early at 3215-3217, before both `pop_preallocated` (3227-3228) and `pop_transferred`
   (3229-3231) [M].
3. Budget: `available_size - max(512*n_active, need_single)`, minus 512 × the last prebuilt batch, minus the
   retracted `ceil_page(fill)+512` (decode.py:1936-1959, 570-582). `n_active` = running + non-host-staged transfer +
   waiting_queue + extra (1867-1873). The candidate needs
   `max(alloc+512, len_in - prefix + min(max_new,4096) - Σ running(in+out))` (485-494). The budget is recomputed
   after each admission with `num_device_preallocated+1` (1525-1531, 1698). The scan is FIFO and breaks at the first
   failure (1477-1486) [M]. `CLIP_MAX_NEW_TOKEN` = `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` = 4096 (decode.py:138,
   environ.py:604) [M].
4. Non-radix path: `required_alloc_tokens = _pre_alloc_fill_len = len(in)+max(len(out)-1,0)`, not page-aligned
   (decode.py:1454, 871-884) [M].
5. Count caps per DP rank: 48 req_to_token rows (`free_slots = range(1, 16+32+1)`, decode.py:186-199) and 32
   metadata buffers (`req_to_token_pool.size*2`, scheduler.py:1504-1513). Running is capped by
   `min(req_to_token_pool.size, max_running)` = 16 (decode.py:3135). Decode server_args in the e2j log:
   `max_running_requests 64`, `disaggregation_decode_extra_slots 32`, `num_reserved_decode_tokens 512`,
   `page_size 64`, `optimistic_prefill_attempts 0`, `allocation_policy 'early'`, `enable_hierarchical_cache False`
   [M `strings decode-0.a1.log | grep -a`]. Default `extra_slots = 2*per_worker` when per_worker ≤ 32
   (pd_disaggregation_hook.py:138-150) [M].
6. Under `early`, prefill starts a `Bootstrapping` request only if `prefill_attempt_count < optimistic_prefill_attempts`
   (0) or the policy is `prefill_complete` (prefill.py:517-524). Decode KV is therefore pinned from admission through
   prefill queue, prefill compute and transfer [M code, I timing].
7. `retract_decode` iterates only `self.reqs` of the running batch, always retracts at least one, and keeps at least
   one (schedule_batch.py:3269-3289). Retraction order is `length_key=(len(out), -len(in))`, sorted with
   `reverse=True` and popped from the end, so the fewest-output requests go first and, among ties, the longest input
   (3369-3370, 3393-3397) [M].
8. The HIP retraction backup is `cpu_tensor`: `host_pool` requires `not is_hip()` (kv_cache_builder.py:207-220).
   `offload_kv_cache` copies `seqlen-1` tokens through `allocator.get_cpu_copy` (schedule_batch.py:1982-2001). The DSA
   override adds `index_k` and calls `torch.cuda.synchronize()` (memory_pool.py:5421-5432). So the copy is synchronous
   [M].
9. Only two policies exist, `Literal["early","prefill_complete"]` (fields/disagg.py:59-64) [M].
   `_validate_prefill_complete` (pd_disaggregation_hook.py:204-247) has the gates R1 lists: Mooncake only;
   `tp_size == dp_size` under dp-attention; no decode radix cache or decode HiCache; prefill HiCache must be
   `write_back` with no storage backend; no staging buffer; no Rust server [M]. Our prefill runs
   `--hicache-write-policy write_through` (dryrun.e2g-102.yihou.txt line 19) [M], so the blocker is real. No `is_hip`
   check exists in the validator [M]. At runtime the KV manager raises if any state type is not DSA (common/conn.py:375-379),
   and a peer whose attn TP/CP/PP is not 1 or whose policy differs is failed (common/conn.py:2057-2069) [M].
   `mark_prefill_complete` is called on the last chunk (prefill.py:1049-1053) [M].
10. The mounted `conn.yihou-regbase.py` differs from the image's mooncake/conn.py only by a 53-line diagnostic and
    keeps the `defer_decode_allocation` branch at line 3128 [M `diff`, `grep`].
11. Host receive is unusable. `supports_host_destination = False` (base/conn.py:124) is never overridden in SRC or in
    the mounted conn.py [M grep]. The help text says "requires dense MHA … No built-in backend currently supports
    this" (fields/disagg.py:179-182) [M].
12. `sglang:pending_prealloc_token_usage` is declared (metrics_collector.py:146, 617) and exported (1627), but nothing
    in SRC or in the mounts assigns it [M grep]. Its X1 maximum is 0 [M].
13. X1 decode-log window 2026-10-12 06:47:47-08:16 [M, recomputed]:
    - 8,892 `Decode batch` lines.
    - `pre-allocated usage`: max 0.73; above 0.5 on 373 lines and above 0.3 on 1,455.
    - `#prealloc-req` max 0, `#retracted-req` max 0, `token usage` max 0.87, `#transfer-req` max 15.
    - 0 "KV cache pool is full" lines in the whole log.

    X1 metrics.tsv (decode, running>0) [M, recomputed]:
    - 2,890 samples; `(kv_used - decode_sum_seq_lens)/max_total` max 0.710, above 0.5 on 148 samples.
    - DP1 at ts 1791791096-1791791116 (07:44:56 UTC): 7 transfer requests, in-flight 0.71, running seq 0.03-0.06.
    - `max_total_num_tokens` = 2,762,176.
14. `pre-allocated usage` = Σ `extend_end` over transfer-queue requests that are not host-staged, divided by
    `max_total_num_tokens` (metrics_reporter.py:998, decode.py:1842-1847) [M]. `extend_end` is `kv_committed_len`, not
    page-aligned (decode.py:2208) [M].
15. Every attribute the patch uses exists:
    - `DecodeRequest.host_staged` / `is_rebootstrap` (decode.py:355-356)
    - `Req.prefix_len`, set in `_pre_alloc` (decode.py:2207)
    - `kv.holds_kv` (schedule_batch.py:972-974)
    - `self.max_total_num_tokens` (decode.py:415)
    - `prefix_match` and `required_alloc_tokens`, both in scope before line 1477

    [M]

### Refuted / imprecise

- **R-a ("22k" summary, not R1 proper):** "fresh stack first runs are only 19.3-20.4k" holds only for the *target*
  config.
  - e1c (relaxed, fresh) = 21,049 and e2d (index_share, fresh) = 21,383 (packup
    `results/analysis/runs_summary.csv`) [M].
  - So the fresh-vs-warm gap is confounded with the config: target fresh is 19.3-20.5k, other fresh runs reach
    21.0-21.4k.
  - The warm numbers check out: 21,696 / 22,114 / 21,921 / 21,681, mean 21,853 [M].

### Unverifiable here

- EAGLE growth sizing (allocation_sizing.py:40-70, "2×6 tokens per step ≤ 1 page") was not re-read.
- That `check_decode_capacity` evicts nothing in our config was not re-read.
- Whether GLM-5.2's `state_types` passes the DSA-only check, and whether `prefill_complete` works end-to-end with
  EAGLE + DSA + write_back on ROCm, can only be settled at launch.
- The impact of the patch's back-pressure on prefill busy and TTFT. The patch is unrun.

### Gaps the report did not address

- **G1 (patch accounting hole).** Failed or aborted transfers can be parked in `_deferred_releases`
  (decode.py:2779-2792, `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE`, 30 s timeout).
  - They are removed from `transfer_queue.queue` (2755-2774) but still hold device KV until resolved.
  - Neither the patch's `_yihou_inflight_tokens` nor `pre-allocated usage` counts them, so the share guarantee leaks
    by that KV.
  - In X1, `num_decode_deferred_kv_release_reqs` max = 0 [M], so the leak is benign today. The patch should still add
    them, or the share-guarantee wording should exclude them.
- **G2 (draft KV backup).** The allocator's `get_cpu_copy` delegates only to `self._kvcache`, the target pool
  (allocator/paged.py:406-411). The DSA override copies KV + index_k + compress tail only. The MTP draft pools are
  separate (`mtp_draft_device_pools`, kv_cache_builder.py:186, 284). Nothing found on the retraction path copies
  draft KV [M absence in offload_kv_cache]. R1's open question therefore leans to "draft KV is not backed up" [I]. Whether
  that is harmful (draft KV recomputed vs stale reads after restore) is unverified. Any plan that relies on
  retraction at C48 needs this checked.
- **G3 (prompt size vs cap).** The DP1 stretch is 0.71 × 2.76 M / 7 ≈ 280 k tokens per in-flight request [M
  arithmetic]. In-flight share is dominated by a few very long prompts.
  - A FIFO token-share cap at 0.5 (1.38 M/rank) admits only about 4-5 such requests per rank and head-of-line-blocks
    the rest. That idles prefill under `early`.
  - R1 rates the cap as binding "rarely" (4-5 % of samples) but does not estimate the prefill idle it causes or the
    decode-rank imbalance it creates. The cap is per rank, and the prefill DP → decode DP routing is fixed per request.
- **G4 (motivation).** X1 at C44 had 0 retractions, 0 prealloc blocking and max usage 0.87, so nothing yet shows
  that running requests are being squeezed. The report gives no failure mode the cap would fix at C44/C48, and no
  experiment to show the cap is net-positive (e.g. C48 with cap 0 vs 0.5, scored on tok/s/chip + retractions).
- **G5.** No test was run on the patch, not even a syntax check or a py_compile of the mounted copy.

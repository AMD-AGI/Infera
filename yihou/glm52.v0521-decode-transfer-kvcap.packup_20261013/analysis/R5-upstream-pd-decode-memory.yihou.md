# R5 — Upstream sglang work on PD-decode KV memory management (vs our image 9d755dd6)

Date 2026-10-12. Sources: `gh` (sgl-project/sglang PRs/issues, compare API), raw docs at the image commit, and the
extracted image source `SRC=/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang`.
Tags: [M] first-hand (gh output / code line / log line read by me), [S] second-hand (PR text, docs, ledger), [I] inference.
Companion: R1 (`R1-decode-admission-code.yihou.md`) holds the line-level admission model; this file is the upstream map.

## 0. Bottom line

1. **Our image is not the v0.5.21 tag.** `SRC/_version.py` = `0.5.21.dev20261008+g11b1fb21de` [M]; commit 11b1fb21de
   = main @ 2026-10-08T11:39Z (#38841) [M]; `compare v0.5.21...11b1fb21de` = diverged, ahead 540 / behind 2 [M]. Tag
   v0.5.21 was released 2026-10-02 [M]. So "in v0.5.21" below means two things: in the tag, and in our image (main
   2026-10-08). Several relevant PRs are in the image but not in the tag (#40703), and several landed just after the
   image (#43407, #43410, #43442, #43173, #43562) [M].
2. **Upstream acknowledges our exact problem and has no token-share cap.** The Q4 PD roadmap (#43699, opened
   2026-10-11) says: "decode reserves a request's whole destination range in HBM at bootstrap, so a slow or long prefill
   occupies decode capacity that running requests need" [S, quoted from issue body]. The upstream answers are
   (a) `prefill_complete` allocation (#40703, merged, in image), (b) decode host receive (#40238, merged, but no
   in-tree transport supports it; Mooncake support is PR #43781, opened today, open), (c) a future "pre-allocate in CPU
   memory by default" item (unchecked) [S]. **No upstream PR or issue proposes a cap on prealloc/in-flight tokens or a
   "running keeps ≥ X % of the pool" rule** (searches in §6) [M for the searches; absence is not proof]. The closest
   are #30033 (issue asking for worst-case admission over running+prealloc; auto-closed) and #25035 (allow retracting
   the whole running batch for prealloc requests; open). Our YIHOU_DECODE_RUNNING_KV_SHARE patch idea (R1 §3.2) is
   therefore novel relative to upstream [I].
3. **For our config, only `early` + device admission is usable today.** `prefill_complete` exists in the image but
   requires prefill HiCache `write_back` (we run `write_through`) [M]. Host receive needs a transport with
   `supports_host_destination` (none in tree, mooncake conn mount has 0 hits) and dense MHA (we are MLA/DSA) [M].
   Decode radix cache rejects EAGLE [M]. Decode offload needs an L3 storage backend and does not add capacity [M].
4. **Upstream tuning guidance** supports raising decode `mem-fraction-static` while `available_gpu_mem` > 5-8 GB
   (ours is 10.89 GB at 0.95) [S docs / M log] → 0.96 is consistent with upstream guidance [I]; nothing upstream on
   max-running/cuda-graph-max-bs specific to PD-decode at our shape beyond "defaults to 48 for MTP, raise it" [S].

## 1. Item table

Containment checked with `gh api repos/sgl-project/sglang/compare/<merge_sha>...<ref> --jq .status`
("ahead"/"identical" ⇒ ancestor ⇒ contained) [M]. "image" = 11b1fb21de.

| # | PR / issue | state, dates | in tag v0.5.21 | in image | key flag / code |
|---|---|---|---|---|---|
| A1 | #7214 "[PD] use int32 for kv indices & get num_reserved_decode_tokens from server_args" | merged 2025-06-15 | yes | yes | `--num-reserved-decode-tokens` (default 512) — `SRC/srt/arg_groups/fields/disagg.py:106-109` [M] |
| A2 | `disaggregation_decode_extra_slots` (origin PR not found by search; present in v0.5.19 `server_args.py:3229`) | ≤ v0.5.19 | yes | yes | default 2× per-worker running when per-worker ≤ 32 — `pd_disaggregation_hook.py:139-149` [M]; ours resolves to 32 (decode log) [M] |
| A3 | #40703 "[PD] Add opt-in prefill-complete decode KV allocation" | opened 2026-09-22, merged 2026-10-03 | **no** (tag 10-02) | **yes** | `--disaggregation-decode-allocation-policy {early,prefill_complete}` default `early` — `fields/disagg.py:59-64`; validation `pd_disaggregation_hook.py:204-247` [M] |
| A4 | #43442 "[PD] Reclaim parked optimistic prefills when request slots are exhausted" | merged 2026-10-10 | no | **no** | fixes an `early`-allocation slot deadlock with optimistic prefill (we run optimistic 0) [S] |
| B1 | #40238 "[PD] Add decode host receive for custom transfer backends" | opened 09-18, merged 2026-09-24 | yes | yes | `--disaggregation-decode-host-receive-threshold` (image: default 0.0 = off) — `fields/disagg.py:179-182`; gate `decode.py:659,695-696,1374-1379` [M] |
| B2 | #43407 "[PD] Use 1 to disable decode host receive and 0 for always-on" | merged 2026-10-09 | no | **no** | **semantics flip**: new default 1 = off, 0 = always on. Our image still treats 0 = off [M code `pd_disaggregation_hook.py:68`]. Any image rebuild after 10-09 must keep passing the *disabled* value correctly (0.0 → would become "always host receive") [I] |
| B3 | #43410 separate paged SWA host pools for host receive | merged 2026-10-10 | no | no | SWA only; not relevant to MLA [S] |
| B4 | #43781 "[PD] Enable Mooncake decode host receive" | opened 2026-10-12, open | no | no | first in-tree transport with host destinations [S] |
| B5 | #39478 "Support unified memory decode host pools" | merged 2026-09-24 | yes | yes | host pool variant [S] |
| C1 | #25372 "[PDD] Add true request retraction for PDD" | merged 2026-07-09 | yes | yes | decode retraction + rebootstrap (`decode.py:731-764,843-880`) [M] |
| C2 | #34801 "[PD] Preserve decode KV across retraction in HiCache" | merged 2026-08-17 | yes | yes | `--disaggregation-decode-retraction-backup {cpu_tensor,host_pool,none}` — `fields/disagg.py:92-105` [M] |
| C3 | #36343 "[AMD] Fall back to CPU tensor for decode retraction on ROCm" | merged 2026-08-27 | yes | yes | inferred backend is `cpu_tensor` on HIP — `kv_cache_builder.py:207-221` (comment: "Large ROCm retraction restores can fault the GPU process") [M]; our decode log shows `retraction_backup: None` (resolved at runtime) [M] |
| C4 | #41103 "[PD] Add a `none` decode retraction backup …" | merged 2026-09-24 | yes | yes | `none` → victim aborted with 503 [S] |
| C5 | #35717 deficit-aware minimum-transfer victim selection | closed unmerged 2026-09-01 | — | no | throughput flat in its own A/B [S] |
| C6 | #25035 "Allow full decode batch retraction for prealloc requests" | opened 2026-05-12, **open** | no | no | image still keeps ≥ 1 running req: `schedule_batch.py:3280-3282` "Always keep at least one request" [M] |
| C7 | #42702 "[PD] Fix retraction restores racing the in-flight decode forward" | open | no | no | correctness risk if we ever retract (we see 0 retractions) [S] |
| D1 | #19746 "[P/D disagg] support decode side radix cache" | merged 2026-05-01 | yes | yes | `--disaggregation-decode-enable-radix-cache`; rejects speculative decoding except DSPARK — `pd_disaggregation_hook.py:97-115` [M] |
| D2 | #26227 HiCache prefetch + incremental transfer on decode | merged 2026-06-02 | yes | yes | decode HiCache; needs decode radix → blocked by EAGLE [M via D1] |
| D3 | #40857 "fix decode radix ownership and allow EAGLE/EAGLE3 with decode radix cache" | open | no | no | would unblock D1/D2 for MTP [S] |
| D4 | #38292 "[PD] Add L2-Only decode-side radix cache" | open | no | no | host-resident decode prefix tree [S] |
| D5 | #17216 "Decode Offload …" | merged 2026-03-08 | yes | yes | `--disaggregation-decode-enable-offload-kvcache` requires `--hicache-storage-backend` — `kv_cache_hook.py:399-412` [M]; offloads to L3, adds no device capacity [I] |
| E1 | #35049 / #41023 deferred decode-side KV release (default ON) | merged 08-18 / 09-24 | yes | yes | `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=True`, timeout 30 s — `SRC/srt/environ.py:729-730` [M]; aborted mid-transfer KV stays held until drain ACK [S] → another (small) in-flight occupant [I] |
| E2 | #43173 "Charge decode admission the page-rounded allocation when radix cache is disabled" | merged 2026-10-08T23:48Z | no | **no** | admission undercharges by < 1 page/request (page 64) on our no-radix path [S]; negligible vs 512-token reserve [I] |
| E3 | #38065 `num_prealloc_ready_tokens` in decode load snapshot | merged 2026-09-04 | yes | yes | metric for router [S] |
| E4 | #40166 "[PD] Add opt-in HRRN decode preallocation ordering" (`SGLANG_DECODE_PREALLOC_POLICY=hrrn`) | open | no | no | ordering only, "KV demand/allocation checks are unchanged" [S] |
| E5 | #42163 "[PD] Shard Mooncake transfers by bootstrap room" | open | no | no | same idea as our e2f `YIHOU_XFER_SHARD=room` [S] |
| F1 | issue #30033 "PD + DP-attention decode: no admission control against future resident-KV growth (running + prealloc) → whole-node lockstep freeze" | 2026-07-03, auto-closed 2026-09-02 (inactivity) | — | — | reporter (GLM-5.2-FP8, decode TP8/DP8, mem 0.85) shows `--max-running-requests`, `--context-length`, mem-fraction each bound one component, never running+prealloc; asks for worst-case reservation [S]. No fix PR linked [M] |
| F2 | issue #20485 "Mooncake + PP: prefill bootstrap timeout … due to decode KV cache saturation" | closed 2026-03-16 | — | — | saturation → prefill timeouts [S, title only] |
| F3 | issue #43699 PD roadmap 2026 Q4 | open 2026-10-11 | — | — | states the slow-prefill-occupies-decode problem; lists A3/B1 as done, "pre-allocate in CPU memory by default" as TODO [S] |

## 2. Semantics that matter for our C44/C48 plan

**Prealloc admission (`early`).** In the image, admission charges
`max(alloc + num_reserved_decode_tokens, rebootstrap_len − prefix + min(max_new_tokens, CLIP) − retractable)`
(`decode.py:485-495`) against free tokens minus `512 × n_active` (`decode.py:1866-1880`), where `n_active` = running +
device-held transfer + waiting (`decode.py:1866-1872`) [M]. The comment at `decode.py:1269` names the starvation case
("one request running decode out of memory, while all other requests are in the transfer queue that cannot be
retracted") [M]. There is no token-share or count cap on transfer-queue KV beyond req_to_token rows
(`max_running/DP + extra_slots`) and metadata buffers (R1 §1.1) [M/S].

**`prefill_complete` (#40703).** Delays per-request decode reservation until prefill's final result is accepted;
"not a general throughput improvement", "removes overlap between chunked prefill and KV transfer" (docs
`pd_disaggregation.mdx:22-73` at 11b1fb21de) [S]. Must be set on both roles; prefill resolves optimistic attempts 0→1
(`pd_disaggregation_hook.py:47-55`) [M]. Rejects (image `pd_disaggregation_hook.py:204-247`) [M]: non-Mooncake,
attn-TP > 1 (ours is attn-TP 1, OK), host receive, decode radix/HiCache, staging buffer, Rust bootstrap, and **prefill
HiCache other than L2 `write_back` without storage** — our prefill runs `hicache_write_policy: write_through` (e2g
prefill server_args) [M]. Using it means switching prefill HiCache to `write_back`, which changes the HiCache-450 GB
behaviour we rely on [I]. Upstream reports no throughput numbers for MLA/DSA [S].

**Host receive (#40238 → #43407).** At device usage ≥ threshold, incoming KV lands in the decode retraction host pool
and is restored on admission [S]. Image: requires `supports_host_destination` (`base/conn.py:124,231` default False; the
mooncake conn mount has 0 occurrences) [M], a `layer_first` host pool (`hicache_hook.py:57-76`) [M] and dense MHA
(help text `fields/disagg.py:181`) [M]. Not usable for GLM-5.2 MLA/DSA on our image [M/I].

**Retraction.** Running requests can be retracted (C1), backup on HIP = `cpu_tensor` (C3) [M]. `retract_decode` keeps
≥ 1 request (C6) [M]. We observed 0 retractions in R041/X1 [S, R1/R2]. Retraction is the failure mode a share cap
would prevent; it is not the throughput limiter at C40 [S R042].

**Decode radix / HiCache / offload in PD.** Decode radix is rejected with EAGLE (D1) [M]; therefore decode HiCache
(D2) is off; decode offload (D5) needs L3 storage and only offloads finished KV [M/I]. Matches the capacity report's
"decode HiCache is a no-op in PD" [S].

## 3. Upstream guidance on mem-fraction, max-running, cuda-graph-max-bs

- `docs/docs/advanced_features/hyperparameter_tuning.mdx` (at 11b1fb21de): keep `available_gpu_mem` at 5-8 GB; if
  10-20 GB, raise `--mem-fraction-static`; or step by 0.01 until OOM; raise `--cuda-graph-max-bs-decode` for large
  batches, lowering mem-fraction if needed; on decode OOM lower `--max-running-requests` [S]. Our decode startup:
  `max_total_num_tokens=2762176 … max_running_requests=16 … available_gpu_mem=10.89 GB` at 0.95 (e2g decode-0.a1.log)
  [M] → by upstream's rule ~3-6 GB more could go to KV (≈ 0.01-0.02 on 288 GB ≈ +55-110k tokens/rank using the
  report's 54.7k/0.01) [I]. Risk not covered by the rule: MTP/DSA activation peaks at bs 64 — the report validated
  only 0.95 under load [S].
- DeepSeek-V3.2 cookbook: PD decode example `--mem-fraction-static 0.9`, TP8 attn-DP8; "`--max-running-requests`
  defaults to 48 for MTP. Increase it for larger batch sizes"; high-throughput recipes use
  `--cuda-graph-max-bs-decode 128 --max-running-requests 128` (`DeepSeek-V3_2.mdx:376,400-415,495-496`) [S].
- GLM-5.2 cookbook: start mem 0.8 on H200 and tune up; `--max-running-requests` "tracks KV capacity"; MI355X MXFP4 at
  TP4 with triton DSA; 5-step MTP validated on `v0.5.20-rocm720-mi35x-20260930` with HiCache `write_through`
  (`GLM-5.2.mdx:127-140`) [S]. Nothing on PD-decode prealloc.
- PD docs examples: decode `--mem-fraction-static 0.8 --max-running-requests 128`; explicit
  `--cuda-graph-bs-decode 1 2 4 8 10 12 14 16` in an example (`pd_disaggregation.mdx:221-238,662-668`) [S].
- Open #34290 "Derive the speculative max_running_requests default from the decode CUDA-graph ladder" [S].
  Our e2g decode already has `max_running_requests 64` (16/rank), graph ladder up to 64 incl. 40/44/48/52 (server_args
  `cuda_graph_config.decode.bs`) [M]. For C48 the per-rank bound (16) is not binding at mean ~4-5 running/rank [S R041],
  but bursts on the busiest DP rank should be checked against 16 [I].

## 4. Implications for the task

1. A ≥ 50 % running-share guarantee must be our own patch (R1 §3.2); upstream offers none [M searches / I].
   It is complementary to, not a duplicate of, #40703/#40238.
2. `prefill_complete` is the only upstream lever that shrinks in-flight KV; trying it costs a prefill HiCache
   `write_through → write_back` change (validation hard-fails otherwise) and loses chunk/transfer overlap [M/S]. Treat it
   as a separate experiment, not a free switch [I].
3. If the image is ever rebuilt on main ≥ 2026-10-09, re-check every `host_receive_threshold` setting (#43407 flips
   0 to always-on) and pick up #43173 (admission page rounding) and #43442 [M/I].
4. Decode mem 0.96 is within upstream's tuning rule given 10.89 GB free at 0.95 [I]; validate under the C44/C48 load
   (MTP + DSA peaks), as the capacity report already recommends [S].

## 5. Commands used (reproducible)

```
gh api repos/sgl-project/sglang/commits/11b1fb21de
gh release list --repo sgl-project/sglang
gh api repos/sgl-project/sglang/compare/v0.5.21...11b1fb21de
gh search prs|issues --repo sgl-project/sglang "<query>" --json number,title,state,createdAt,closedAt
gh pr view <n> --repo sgl-project/sglang --json title,state,createdAt,mergedAt,mergeCommit,body
gh api repos/sgl-project/sglang/compare/<merge_sha>...{11b1fb21de,v0.5.21}  --jq .status
curl https://raw.githubusercontent.com/sgl-project/sglang/11b1fb21de/docs/docs/advanced_features/{pd_disaggregation,hyperparameter_tuning}.mdx
strings runs/e2g/server-logs/{decode-0.a1,prefill-0}.log | grep -a -m1 -o "server_args=.*"
```

## 6. Negative searches (no hit relevant to a prealloc cap)

PR/issue queries returning nothing relevant: "max_prealloc", "prealloc tokens decode kv", "decode capacity long
prefill", "decode reserve destination slots", "prealloc decode memory", "transfer queue decode kv full",
"num_reserved_decode_tokens" (issues), "max-running-requests decode PD", "cuda-graph-max-bs decode disaggregation" [M].
GitHub search is keyword/semantic and incomplete; absence is suggestive only [I].

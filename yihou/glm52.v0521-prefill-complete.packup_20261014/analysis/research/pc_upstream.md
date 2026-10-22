The `prefill_complete` policy is merged upstream as PR #40703. Our image already contains it. One follow-up fix landed after our image. As our base config stands, the planned run will not get past argument validation: the prefill HiCache write policy is `write_through`, and `prefill_complete` only accepts `write_back`.

**1. The PR that introduced the feature: #40703**
- https://github.com/sgl-project/sglang/pull/40703, "[PD] Add opt-in prefill-complete decode KV allocation".
- Author is ekrim. Opened 2026-09-22, merged 2026-10-03T13:03Z as commit `5b5d7212`, +2125/-40.
- **Not in the v0.5.21 tag.** v0.5.21 was released 2026-10-02 (tag commit `e00930c5`). `gh api compare v0.5.21...5b5d7212` returns "diverged, ahead=255".
- **It is in our image.** `$S/_version.py` reads `0.5.21.dev20261008+g11b1fb21de` (main at 2026-10-08T11:39Z). Comparing image to `5b5d7212` returns "behind=285", so the image includes it.
- Our mounted `conn.yihou-regbase.py` is the image's mooncake `conn.py` plus the YIHOU-REGBASE diagnostic and nothing else (`diff` shows only those additions). It keeps the #40703 mooncake changes: R:2717 "Publish worker cleanup last" and R:3119-3128 `poll` checking `defer_decode_allocation`.
- **Design:** decode reserves a request's destination KV only after prefill accepts its final result. The global KV pool size does not change.
  - It reuses the optimistic-prefill inflight queue. Finished requests stay parked on prefill until decode sends destination indices.
  - Chunked prefill does not yield. The whole prompt's KV goes out in one `send_kv_chunk(last_chunk=True)`, so there is no overlap between chunked prefill and KV transfer.
  - Prefill raises `optimistic_prefill_attempts` from 0 to 1 automatically.
- **Benchmark it claims:** GLM-5.2-NVFP4, MTP3, 12 Rubin GPUs, 2 prefill replicas and 1 decode replica, each TP4/DP4/EP4, FP8 KV, Mooncake. Decode KV was deliberately capped at 163,840 tokens per rank. Mixed workload: 64× 131K-token plus 192× 4K-token inputs, OSL 256, concurrency 64. Two runs per arm (ABBA order); changes are means:

  | Metric | Change |
  |---|---|
  | Output tokens/s | +26.8% |
  | TTFT p95 | -48.9% |
  | Mean TPOT | +2.1% |
  | ITL p99 | +92.8% (worse) |

  The author says these numbers predate the final rebases. A Qwen3-8B comparison showed about 1% difference, within noise. The policy stays opt-in for that reason.
- **Open design point in the review thread:** YAMY1234 proposed sending the ready signal when the final chunk is scheduled rather than after its result. The thread moved to Slack; I found no PR for it.

**2. Docs (upstream main, not in our image)**
- `docs/docs/advanced_features/pd_disaggregation.mdx`, section "Decode allocation policy" (about lines 23-72). It says:
  - Set the flag on **both** prefill and decode.
  - "Not a general throughput improvement; compare both policies for your workload."
  - Requires Mooncake. Attention DP is allowed when each DP rank has attention TP 1.
  - Prefill HiCache is limited to L2 `write_back` with no storage backend.
  - Decode radix cache and decode HiCache are not supported.
  - `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` (prefill) and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (decode) define the wait budgets.
- I found no benchmark scripts that use the flag.
- The roadmap issue https://github.com/sgl-project/sglang/issues/43699 (line 37) lists it as done.

**3. Validation in our image: `$S/srt/arg_groups/pd_disaggregation_hook.py`, `_validate_prefill_complete` at :204-247**

| Check | Line(s) | Our planned run |
|---|---|---|
| Backend must be mooncake | :210 | OK |
| `tp_size == dp_size` when DP attention is on | :213-215 | TP4/DP4 passes, 4 == 4 [inferred] |
| Host-receive threshold must not be > 0 | :224 | Default 0.0 (`fields/disagg.py:179-182`) passes |
| No decode radix cache or decode HiCache | :226-229 | `DECODE_HICACHE=0` (`config.yihou.v0521-base.sh:92`) |
| Prefill HiCache must be L2 `write_back`, no storage backend | :231-238 | **Fails.** Base sets `PREFILL_HICACHE_WRITE_POLICY=write_through` (`config.yihou.v0521-base.sh:98`) |
| No `SGLANG_DISAGG_STAGING_BUFFER`, no Rust server | :239-242 | — |

- With `write_through`, startup raises a `ValueError`. The planned run needs `PREFILL_HICACHE_WRITE_POLICY=write_back`.
- `serving_hook.py:476-495` keeps optimistic prefill for write_through or write_back. `serving_hook.py:497-505` rejects `prefill_complete` if optimistic prefill was resolved off.
- EAGLE/MTP is not excluded; only "decoupled speculation" is (:217-218).
- At runtime, decode marks a request Failed if the peer's policy differs, or if the peer has attention TP, CP or PP other than 1 (`srt/disaggregation/common/conn.py:2057-2072`). Non-DSA state pools are rejected at `common/conn.py:375-379`.

**4. Harness**
- `engine.sh`, `config.sh` and all `config.yihou.*.sh` contain no `allocation-policy` or `host-receive-threshold` string (grep returned nothing).
- So the current runs use the defaults, `early` and `0.0`, implicitly. They are not passed explicitly.
- The new run needs `--disaggregation-decode-allocation-policy prefill_complete` added to **both** legs. I did not check how the harness passes extra engine args.

**5. Follow-ups merged after our image (2026-10-08T11:39Z)**
- **#43442**, https://github.com/sgl-project/sglang/pull/43442 (cctry, merged 2026-10-10, `20948cdc`, image is behind by 167). "Reclaim parked optimistic prefills when request slots are exhausted."
  - It fixes a prefill/decode mutual wait under the `early` policy.
  - The reclaim branch is explicitly skipped for `prefill_complete` ("Preserve parking for prefill_complete").
  - It also releases the metadata buffer when an exhausted optimistic attempt goes back to bootstrap. That part applies to both policies.
  - Our image lacks it: the "Decode may have admitted different requests" branch is not in `$S/srt/disaggregation/prefill.py`.
- **#43407**, https://github.com/sgl-project/sglang/pull/43407 (merged 2026-10-09, `f6c848c8`, not in image). Flips the encoding of `--disaggregation-decode-host-receive-threshold`: 1 now means disabled and 0 means always-on. On main the check is `< 1`.
  - This only matters if we upgrade. On a newer image, an explicit `0.0` would mean always-on, which conflicts with `prefill_complete`.
- **#43719**, https://github.com/sgl-project/sglang/pull/43719 (merged 2026-10-12, not in image). After a failed Mooncake send it withholds ("poisons") the abort ack, so decode does not free pages that may still receive writes.
  - It is general to deferred KV release, not specific to `prefill_complete`.
- Already in the image:
  - #41395 (Mooncake waits on the early-send event; present at R:3093). Its validation was on MI355X with GLM-5.2.
  - #42454 and #43505 are test-only follow-ups.

**6. Open PRs and issues**
- PR #43970 (open) adds `ascend` to the backend check. Not relevant to us.
- PR #43811 (open) adds early KV send but excludes `prefill_complete`.
- No open bug issue mentions `prefill_complete`. It appears only in roadmap issue #43699 and the PP roadmap #43885 (:29, which says PP>1 is rejected).
- I found no reported bugs about its interaction with EAGLE/MTP, DP attention or HiCache.

**Not verified:** whether #40703's benchmark gain carries over to our setup.

- Our setup differs from the PR's: same-node HIP-IPC transport, AgentX workload, and decode KV not capped.
- The PR's own caveat is that this policy removes the overlap between chunked prefill and KV transfer, and its ITL p99 got worse.
- The planned C48 run, with the write policy changed to `write_back`, is the measurement that would answer it.
- That run changes two things at once: the allocation policy and the HiCache write policy. An `early` + `write_back` control run would separate them.

=====VERIFY=====
I checked the report's main claims against the image source, the harness and `gh`. Its central conclusion holds: as configured, the planned run fails argument validation, because `prefill_complete` only accepts prefill HiCache `write_back` and the base config uses `write_through`. The report misses two things: how the harness passes the flag (two existing hooks do it), and the per-leg detail of the validation. One claim was not checked: whether GLM-5.2's KV pool is DSA, which a runtime check needs.

`S=/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang`, `H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness`, and `R=$H/mounts-v0521/src/conn.yihou-regbase.py`.

**CONFIRMED**
1. **Image contents.** `$S/_version.py:21` reads `0.5.21.dev20261008+g11b1fb21de`.
   - `gh compare 11b1fb21de...5b5d7212` returns behind=285, so #40703 (merged 2026-10-03T13:03:09Z) is in the image.
   - `compare v0.5.21...5b5d7212` returns diverged, ahead=255, behind=2.
   - #43442 (`20948cdc`) is 167 commits ahead of the image and #43407 (`f6c848c8`) is 136 ahead, so neither is in it.
2. **Validation wiring.** `pd_disaggregation_hook.py:47-48` runs `_validate_prefill_complete` only when the policy is `prefill_complete`. The checks at :204-247 match the report's table:
   - backend :210
   - `tp_size != attn_dp_size` :213-215
   - `host_receive_threshold > 0` :224
   - decode radix/HiCache :226-229
   - prefill HiCache must be `write_back` with no storage backend :231-238
   - staging buffer / Rust server :239-242
   - decoupled speculation only :217-218, so EAGLE is not excluded.
   - The error is `ValueError` at :244-247.
3. **Automatic bump.** `pd_disaggregation_hook.py:49-55` raises `optimistic_prefill_attempts` from 0 to 1 on prefill.
4. **Optimistic-prefill gate.** `serving_hook.py:476-495` keeps optimistic prefill for L2 `write_back` or `write_through`. `serving_hook.py:497-505` raises if it was resolved off.
5. **Host-receive threshold default.** `fields/disagg.py:179-182` has default 0.0, and its help text says 0 disables host receive.
6. **Base config fails as-is.** `config.yihou.v0521-base.sh:98` sets `write_through`, and `bench-harness/config.sh:56` defaults to it too. So the run raises unless `PREFILL_HICACHE_WRITE_POLICY=write_back`. `DECODE_HICACHE=0` is at :92.
7. **Runtime peer checks.**
   - Policy mismatch, and peer attention TP, CP or PP other than 1, mark the request Failed at `common/conn.py:2057-2072`.
   - Non-DSA state pools raise at `common/conn.py:375-379`.
8. **Mounted `conn.py`.** `diff` against the image's mooncake `conn.py` shows only three added hunks (`82a83,133`, `382a434`, `507a560`), all YIHOU-REGBASE logging. The #40703 code is kept:
   - R:2717 "Publish worker cleanup last"
   - R:3119-3130 `poll`, gated on `defer_decode_allocation`
   - R:3093 `_early_send_wait_event` (#41395)
9. **Harness has no policy flag.** `grep` for allocation-policy / host-receive across `engine.sh`, `config.sh` and all `config.yihou.*.sh` returned nothing (rc=1). Both legs therefore run the defaults `early` and `0.0` implicitly.
10. **#43442 is absent.** The string "Decode may have admitted" is not in `$S/srt/disaggregation/prefill.py`.
11. **Prefill parks finished requests.**
    - `prefill.py:517-530`: under `prefill_complete`, a request in Bootstrapping is always kept on the optimistic path.
    - `prefill.py:1050-1055`: it calls `mark_prefill_complete()`, then `send_kv_chunk(last_chunk=True)` only if `not req.pending_bootstrap`.
    - `prefill.py:1395-1398`: the chunked-request yield is disabled under `prefill_complete`.
    - `prefill.py:1722-1724`: requests are never sent back to bootstrap after exhausting their attempts.
    - [inferred] Prefill GPU KV stays held until decode allocates, so prefill mem 0.88 and the 370 GB hicache size bear on this.
12. **#43719** (merged 2026-10-12T07:33:09Z, `4195980f`) is not in the image: `_poison_unsettled_send` occurs 0 times in the image's common/mooncake `conn.py` and in R.

**WRONG or imprecise**
- **"#43719 not in image" needs a qualifier.** The poison machinery itself is already in the image: `common/conn.py:368-370` and `:920-933` (`poison_deferred_ack_room`), and image mooncake `conn.py:2672`. #43719 adds `_poison_unsettled_send` and extra poison call sites, the in-check for concurrently registered targets, and tests. Only that increment is missing.
- **Timeouts.** The report says prefill uses `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` and decode uses `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`. In the image, the prefill-side `DeferredBootstrap` budget is WAITING_TIMEOUT **plus** BOOTSTRAP_TIMEOUT (`common/conn.py:380-384`).
- **"Harness pass-through not checked" is a gap the report should not have left.** The knobs exist:
  - `config.yihou.v0521-base.sh:243-244` appends `PREFILL_EXTRA_ARGS_ADD` and `DECODE_EXTRA_ARGS_ADD`.
  - `engine.sh:50,63` picks up the per-role args, and `engine.sh:321-323` word-splits and appends them.
  - Set both to `--disaggregation-decode-allocation-policy prefill_complete`.
  - Write policy goes through `PREFILL_HICACHE_WRITE_POLICY` to `engine.sh:46,305`.

**UNVERIFIED**
- Whether GLM-5.2-MXFP4's KV state types are all `StateType.DSA`, so that it passes `common/conn.py:375-379`.
- Upstream PR benchmark numbers, review-thread content, docs wording, issue #43699/#43885 line refs, and #43407's semantics: not re-read.
- The report's [inferred] TP4/DP4 pass. The code is `tp_size != (dp_size if enable_dp_attention else 1)` at :213-214, which passes only if DP attention is on in both legs. The engine flag was not re-read.

**Missing facts that matter**
- **Hicache size knob.** `PREFILL_HICACHE_SIZE` defaults to 450 (`config.yihou.v0521-base.sh:97`). The 370 run must set it; it is emitted as `--hicache-size` at :238.
- **`--hicache-ratio` still emitted.** `engine.sh:304` always emits `--hicache-ratio`. Per the comment at base :93-95, size > 0 overrides it in v0.5.21; that comment was read, not tested.
- **Mem fractions differ from the plan.** Base defaults are `PREFILL_MEM_FRACTION=0.85` and `DECODE_MEM_FRACTION=0.95` (:87-88). The plan needs 0.88 and 0.96, so the run changes more than the two variables the report names.
- **Decode leg is validated too.** The full validator runs on decode as well. With `DECODE_HICACHE=0` and no `--disaggregation-decode-enable-radix-cache`, decode passes :226-229. A decode HiCache config (e.g. a runshare variant) would fail.
- **SWA exclusion.** `prefill.py:526-529` excludes write-through caches with SWA branching from the optimistic retry path; it does not apply once the policy is `write_back`.
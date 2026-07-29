I have comprehensive verified findings. Here is the report.

---

# SGLang Research Report — GLM-5.2 DSA + MTP + DP-attention on MI355 (gfx950)

## Method & verification status

**GitHub MCP tools failed with `401 Bad credentials` on every call** (`get_me`, `get_file_contents`, `get_file_content_filtered`, `search_pull_requests`). Bash network access was also sandbox-denied for `curl`. No WebFetch tool was available.

**However, `git` network access worked.** I cloned `sgl-project/sglang` to `/tmp/sgl` and fetched **all 25,626 `refs/pull/*/head` refs**. This is *stronger* verification than the API for the questions asked: I can check exact ancestry, diffs, and cherry-picks deterministically.

- **GIT-VERIFIED** = confirmed by local git ancestry/diff against the real repo. High confidence.
- **SEARCH-ONLY** = from WebSearch, not API- or git-verified.
- Limitation: git PR refs give me PR *head commits and diffs*, but **not** issue bodies, PR titles-as-shown, comments, labels, or open/closed/merged state directly. I infer state from ancestry. Issues (as opposed to PRs) have **no** git ref at all, so I could not read any issue body.

Baseline verified: `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`, `Mon Jul 13 21:11:42 2026 -0700`, subject exactly as you stated. It **is** an ancestor of `origin/release/v0.5.15` and **is not** an ancestor of `origin/main`. Your description is correct.

**Important baseline fact (GIT-VERIFIED):** `release/v0.5.15` is effectively frozen. Only **one** commit exists after your baseline on that branch:
`847d1bc3f47 | 2026-07-14 | [Cherry-pick to release/v0.5.15] [AMD] Pin cmake==4.3.4 in ROCm Dockerfile... (#30942) (#31262)`. So nothing else will arrive via that branch. `release/v0.5.16` is 576 commits ahead of v0.5.15.

---

## Task 1 — Known PR audit

| # | What it actually is | Title (from commit/head) | State | Date | Target | In our image? |
|---|---|---|---|---|---|---|
| **29798** | PR, merged | `fix: avoid DSA indexer CPU seq lens fallback` | **Merged** | 2026-07-03 | main | **IN** ✅ |
| **30992** | PR, merged | `support GLM-5.2 MTP index sharing with prefill CP` | **Merged** | — | main (`7e229e2a8`) | **IN** (via #31106) ✅ |
| **31106** | PR, merged | cherry-pick of #30992 to release/v0.5.15 | **Merged** | 2026-07-13 | release/v0.5.15 | **IS our HEAD** ✅ |
| **32209** | **PR, OPEN** | `fix: narrow seedless MTP fallback to draft graphs` (5 commits) | **Open** | 2026-07-26 | main | **NOT in** ❌ |
| **32527** | **NOT a PR — an issue** | *unreadable* | Open (search-only) | ~2026-07-27 | n/a | n/a |
| **32722** | **PR, OPEN** | `test: cover GLM-5.2 PD DP attention with MTP` | **Open** | 2026-07-29 | main | **NOT in** ❌ |
| **28815** | **No PR ref exists** — issue or invalid | — | — | — | — | **UNVERIFIED** |
| **31071** | **NOT a PR you can rely on** — see below | `[Bug] eagle: broadcast finalized verify decision across TP ranks` | **NOT merged** | 2026-07-16 | main | **NOT in** ❌ |
| **31478** | **PR, OPEN** | `test: strengthen eagle verify TP-broadcast regression test` | **Open** | 2026-07-16 | main | **NOT in** ❌ |

### Corrections to your list

1. **#32527 and #28815 are not PRs.** Neither has a `refs/pull/*/head` ref in a full fetch of all 25,626 PR refs, while #32209, #32722, #31478 all do. #32527 is an issue (search corroborates: "opened by Xavier1994 on Jul 27, 2026, Open"). **#28815 I could not confirm at all** — UNVERIFIED.

2. **#31071 is mis-attributed in an important way.** You listed #28815/#31071/#31478 as the aiter-all-reduce-deadlock → `--disable-custom-all-reduce` group. #31071 is **not** about custom all-reduce. Its commit message (GIT-VERIFIED, `61c2af4d6cc`) says the greedy EAGLE verify branch computes accepted tokens from a **per-rank local `torch.argmax` with no TP broadcast**:

   > "When per-rank next_token_logits differ from a non-deterministic all-reduce (e.g. AMD `--enable-aiter-allreduce-fusion`) a near-tie makes argmax pick a different token per rank, so ranks accept a different number of drafts, committed seq_lens/batch shapes diverge, and the next TP collective deadlocks (both ranks wedge in **resolve_seq_lens_cpu -> Event.synchronize**; /health still 200 until the watchdog fires)."

   That is **your exact failure signature**, and the all-reduce flag is the *trigger*, not the bug. `--disable-custom-all-reduce` is a workaround that hides it.

3. **#31071 is NOT merged.** Two commits carry that subject (`c198cf08e70`, `61c2af4d6cc`); **neither is an ancestor of `origin/main`**, `release/v0.5.15`, or your baseline. The only branch containing `c198cf08e70` is `pr/31683`.

4. **The #31071 bug is still live in main today.** GIT-VERIFIED in `origin/main:python/sglang/srt/speculative/eagle_utils.py`: the greedy branch is entered at
   `if sampling_info.is_all_greedy or _is_cpu or _is_npu or _is_hip or _is_xpu:` (line ~721)
   and the `tp_group.broadcast(predict/accept_index/num_correct_drafts, src=0)` block sits at **8-space indent inside the `else:` (sampling) branch** at lines 826–834. **`_is_hip` unconditionally forces the greedy branch, which never broadcasts.** Identical in your baseline and in `release/v0.5.16`.

### Other cherry-picks worth knowing (GIT-VERIFIED)

- **#30839 / #31083** `Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay` — cherry-pick `1bee78dc114` **IS in your image**.
- **#30265** `[AMD] Fix GLM-5.2 MTP Quark excludes` — merged to main 2026-07-09, **NOT in your baseline**. It sits on `origin/amd/cherry-pick-release-v0.5.15`, a 9-commit AMD-only branch off v0.5.15 that was **never merged into `release/v0.5.15`**. Worth reviewing — it also carries `[AMD] Fix AITER custom all-gather CUDA-graph capture crash under torch_memory_saver (#30557)` and `[AMD] fix dsv4 indexer dtype dispatch on gfx950 (#29479)`.
- **#26672** `[AMD] Work around HIP TPOT regression from Event.wait() in MTP seq lens resolution` — merged 2026-05-29, **IN your image**. This is the origin of the `_is_hip` → `synchronize()` branch in Task 3.

---

## Task 2 — Open PRs/issues you didn't know about

All GIT-VERIFIED as "head commit not an ancestor of main" (i.e. open or squash-merged; none of their content is in main).

**Directly on your failure mode:**

1. **PR #31683** — `[ROCm] Route forced DP gather through capture-aware all-reduce` (2026-07-20, long10024070). **The most relevant thing I found.** A 10-commit ROCm stack that *contains* the #31071 fix plus: `fix: handle idle DP ranks in EAGLE draft sampling`, `fix: handle idle DP ranks without zero-work ROCm launches`, `fix(rocm): skip empty GPU all-reduce kernels`, `fix(rocm): skip empty AITER paged MQA logits`, `[ROCm] Keep speculative DSA metadata consistent with padded batches`, `fix: avoid device cumsum for eager DP logits metadata`. Touches `communicator.py`, `dp_attention.py`, `logits_processor.py`, `eagle_worker_v2.py`, `eagle_utils.py`, `forward_batch_info.py`. This is *idle-DP-rank + ROCm + EAGLE* — precisely your topology. Large (2,479 insertions, 34 files) so it will not apply cleanly to v0.5.15, but individual commits should.

2. **PR #32209** — `fix: narrow seedless MTP fallback to draft graphs` (2026-07-26, daowu.hzy). Your description is accurate. Adds `can_draft_cuda_graph` to `MLPSyncBatchInfo`, min-reduced across DP ranks via the existing all-gather, plus `BaseSpecWorker.requires_dp_attention_eager_forward()`. Commits: `synchronize seedless MTP fallback across DP ranks`, `unpad eager MTP draft outputs under DP attention`, `trust future DSA seed state during overlap`, `handle eager DP padding in DSA draft decode`. Touches `dsa_indexer.py`, `dsa_backend.py`, `dp_attn.py`, `eagle_worker_v2.py`, `disaggregation/decode.py`. Note: `can_cuda_graph` min-reduce **already exists in your baseline** (`dp_attn.py:111`); the gap is only that the *draft/spec* bit isn't folded in — that makes this comparatively backportable.

3. **PR #32196** — `Cover DSA graph fallback in disagg draft input test` (2026-07-29, weireweire). **A competing implementation of the same idea as #32209.** Its earlier commit is literally `Synchronize EAGLE DSA graph fallback across DP ranks`. Adds `_spec_input_cuda_graph_compatible()` folded into `can_cuda_graph` — a smaller, cleaner diff than #32209. Also adds `Preserve draft DP token counts across MoE backends`. Touches `dp_attn.py`, `eagle_disaggregation.py`, `eagle_info.py`, `eagle_worker_v2.py`, `forward_batch_info.py`. **These two PRs overlap and will conflict; upstream will likely pick one.**

4. **PR #32722** — `test: cover GLM-5.2 PD DP attention with MTP` (2026-07-29, IvanShan177). Confirmed: a single new file `test/.../disaggregation/test_glm52_nvfp4_pd_dpa_mtp.py`, 125 lines, no product code. Consistent with "red by design, first CI case for PD+DPA+MTP".

5. **PR #31478** — `test: strengthen eagle verify TP-broadcast regression test` (2026-07-16, zhoaa). Carries the #31071 product fix + a 161-line CPU regression test. **Small (2 files, 178 insertions) and the most backportable single item.**

**DSA / indexer / seq-lens adjacent (open):**

6. **PR #31760** — `Fix partial-DP DSA decode padding` (2026-07-20). `kernels/ops/attention/dsa/transform_index.py`. Partial-DP + DSA padding — likely relevant to your rank divergence.
7. **PR #32490** — `Avoid DSA top-k host synchronization` (2026-07-28, Ziang Li). Removes another host sync from the DSA top-k path.
8. **PR #32738** — `[Fix] DSA Indexer: pad heads for DeepGEMM paged MQA logits on decode/target-verify` (2026-07-29).
9. **PR #32317** — `keep decode in cpu seq_lens gate; idle batches route there` (2026-07-24, hnyls2002). **Directly about idle batches and the `needs_cpu_seq_lens` gate.**
10. **PR #32374** — `Fix DSpark device-only fallback layouts` (2026-07-25). See Task 3 — this is the only open PR restructuring `publish()`/`publish_ready`.
11. **PR #32187** — `Fix PD disaggregation decode warmup crash under DP-attention (#30748)` (2026-07-23).

**AMD/ROCm adjacent (open):**

12. **PR #32570** — `[AMD] Add GLM-5.2-FP8 8-GPU MI35x nightly accuracy CI` (2026-07-27).
13. **PR #32098** — `[AMD] Fix gfx950 DSA prefill context-parallel RoPE ordering` (2026-07-22).
14. **PR #31338** — `[AMD] Fix packed DSA FP8 KV write routing on HIP` (2026-07-15).
15. **PR #31955** — `perf(dsa): prefill-only FP8 dense projection GEMM for GLM-5.2 MLA (gfx950)` (2026-07-15).
16. **PR #31324** — `fix(dense-decode): also gate eager decode k-only skip to HIP (AMD)` (2026-07-24).
17. **PR #31500** — `[AMD][DI][CI] Gate overlap-plan-stream on non-MTP` (2026-07-21). Notable: AMD CI is *disabling* the overlap plan stream when MTP is on.

**Draft-extend CUDA graph on HIP:** I found **no** open PR enabling it. What I did find (GIT-VERIFIED, `eagle_draft_extend_cuda_graph_runner.py:440` in main, `:430` in your baseline — unchanged) is a HIP *exclusion* from the topk==1 fastpath:
```
# ROCm's argmax tie-breaks differently from CUDA's softmax+max
# path on FP8 logits, which corrupts MTP draft selection on AMD.
# Keep the fastpath CUDA-only.
if self.topk == 1 and not _is_hip:
```
This is a second, independent upstream acknowledgement that **ROCm argmax tie-breaking diverges** — the same root cause as #31071.

**Issues:** I could not read any issue body (no git ref, MCP 401, no WebFetch). SEARCH-ONLY, unverified titles that appear relevant: #29347 `[Bug][AMD][ROCm/MI300X] EAGLE speculative decoding + CUDA graph: non-deterministic deadlock in greedy verification fallback`; #26454 `[Bug] Non-DP multi-node TP=8 hang in event_loop_overlap ... all TP ranks block in result.copy_done.synchronize()`; #13435 `[Bug] PD decode DP attention + MTP crash`; #7026 `[Bug] PD+MTP+DeepEP+dp attention`; #31116 `DP attention + prefill CUDA graph...`. **Treat these titles as unverified.**

---

## Task 3 — `overlap_utils.py` idle-rank `publish_ready`

**Your analysis is correct, and GIT-VERIFIED against current `origin/main`** (`/tmp/sgl`, file is 533 lines).

`FutureMap.publish()` (main, ~line 470):
```python
def publish(self, future_indices, new_seq_lens, confidence=None) -> None:
    indices = future_indices
    if indices.shape[0] == 0:
        return  # DP idle          <-- line 478, returns BEFORE the event block
    self.new_seq_lens_buf[indices] = ...
    ...
    if self.spec_algo.is_some():
        device_module = torch.get_device_module(self.device)
        if self.publish_ready is None:
            self.publish_ready = device_module.Event()   # <-- line 486-487, lazy init
        else:
            device_module.current_stream().wait_event(self.publish_ready)
        self.publish_ready.record()
```
`FutureMap.resolve_seq_lens_cpu()` (main, ~line 412):
```python
if self.publish_ready is not None:
    ...
    if _is_hip:
        # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
        self.publish_ready.synchronize()      # <-- line 429-431, blocking HOST sync
    else:
        self.publish_ready.wait()
```

**Answers:**

- **Has anyone upstream touched/fixed/reported this?** **No.** I swept the diffs of every PR ref numbered ≥30000 for additions mentioning `publish_ready` or `DP idle` in this file. Twelve PRs touch it; **none** changes the `if indices.shape[0] == 0: return` early return or makes `publish_ready` idle-rank-safe. Most are DSpark/NPU work. The closest is **PR #32374** (open), which heavily restructures `publish()` — adds `_publish_stream` tracking, a `reset()` method, a new `resolve_seq_lens_device()` GPU-only path, and enables the pinned relay on HIP (`if _is_cuda or _is_hip`) — **but it leaves `if indices.shape[0] == 0: return  # DP idle` completely untouched.** I found no upstream issue reporting it (I could not read issue bodies, so this specific point is UNVERIFIED-negative rather than proven-negative).

- **Does the HIP-only `synchronize()` branch still exist in main?** **Yes, verbatim, unchanged.** Present with the identical `# Temporary workaround: Event.wait() regresses TPOT on AMD MI355.` comment in `origin/main` (line 430), in your baseline `0b3bb0cb` (line 294), and in `release/v0.5.16` (line 430). It was introduced by **#26672** (`ace730db48c`, 2026-05-29, author "Arik"), which is in your image. **It has not been replaced.**

- **Note on divergence severity:** the `_is_hip` branch uses `Event.synchronize()` — a **blocking host-side** wait — whereas CUDA uses `Event.wait()`, a non-blocking *stream* wait. So on HIP a busy rank blocks its Python thread while an idle rank sails past, which is exactly the asymmetry that turns a benign rank divergence into a hard deadlock. **This is the same wedge point named in #31071's commit message** ("both ranks wedge in `resolve_seq_lens_cpu -> Event.synchronize`"). The two bugs converge on this line.

Also relevant (GIT-VERIFIED): `dsa_backend.py` sets `needs_cpu_seq_lens: bool = False` (main line 350, baseline line 339). With DSA, `resolve_seq_lens_cpu` returns early at the `not self.needs_cpu_seq_lens` block — **but only *after* the `publish_ready` HIP `synchronize()` has already executed.** The sync is not skipped for you.

---

## Most actionable findings for us

1. **#31071 is the highest-value finding, and it is NOT merged anywhere.** The greedy EAGLE verify path never broadcasts the accept decision across TP ranks, and `_is_hip` forces every AMD run onto that path. Its commit message describes your deadlock signature exactly, including the `resolve_seq_lens_cpu -> Event.synchronize` wedge point and "/health still 200 until the watchdog fires". **Correct your notes: #31071 is not an all-reduce-deadlock PR — `--disable-custom-all-reduce` merely suppresses the trigger.** Cheapest real fix available: cherry-pick `61c2af4d6cc` (2 files, 178 insertions) from `pr/31478`, which hoists the broadcast out of the `else:` branch to after the accept decision is finalized.

2. **PR #31683 is the closest thing to a ready-made fix for our exact topology** and neither of us knew about it. It bundles the #31071 fix with four separate idle-DP-rank ROCm fixes (`handle idle DP ranks in EAGLE draft sampling`, `handle idle DP ranks without zero-work ROCm launches`, `skip empty GPU all-reduce kernels`, `Keep speculative DSA metadata consistent with padded batches`). Too big to backport whole; mine it commit-by-commit.

3. **Task 3 bug is real, unreported, and unfixed on every branch** (baseline, main, v0.5.16). The idle-rank `publish_ready is None` asymmetry combined with the HIP-only *blocking* `synchronize()` is a genuine gap. Since no one upstream has touched it, **this is ours to fix and upstream** — and it will not conflict with #32374 as long as we only guard the idle path.

4. **Two competing open PRs (#32209 and #32196) both add a DP-voted spec-graph bit** and will conflict. #32196's `_spec_input_cuda_graph_compatible()` is the smaller, cleaner approach. Since `can_cuda_graph` min-reduce **already exists in our baseline**, either idea backports as a small patch — we don't need the whole PR.

5. **`release/v0.5.15` is dead** — one commit after our baseline, ever. Nothing will arrive by waiting. Every fix above must be manually backported, or we move to v0.5.16 (576 commits ahead) — but note **v0.5.16 still contains both bugs**, so moving up does not help by itself.

6. **Check `origin/amd/cherry-pick-release-v0.5.15`** — a 9-commit AMD branch off our exact release line that was never merged, including `[AMD] Fix GLM-5.2 MTP Quark excludes (#30265)`, `[AMD] fix dsv4 indexer dtype dispatch on gfx950 (#29479)`, and an AITER all-gather CUDA-graph capture crash fix (#30557). Trivially applicable to us.

**What I could not verify:** any issue body or comment (MCP 401, no WebFetch); PR titles as rendered on GitHub (I used head-commit subjects, which usually but not always match); official open/closed/merged labels (inferred from ancestry); and **#28815 entirely** — it has no PR ref, and search could not surface it.
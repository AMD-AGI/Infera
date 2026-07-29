# Upstream ecosystem — sglang × GLM-5.2 × AMD/ROCm × MTP

Written 2026-07-29. **This is the *upstream* view.** For *our* bugs — symptoms, py-spy
evidence, root-cause chains, fix options — read `dpa.mtp.pd.debug.md` first; this file
cross-references it and deliberately does not restate it.

Purpose: a future engineer with no memory of this work should be able to read this one file
and know what upstream has and has not fixed, what is worth cherry-picking, what is ours to
report, and what is still unexplained — without redoing the research.

---

## 0. Scope and how to read it

Primary input: a completed research pass (`/home/yihou/glm52_fix/github_research_report.md`)
that could **not** use the GitHub API — every MCP call returned `401 Bad credentials`, `curl`
was sandbox-denied, and no WebFetch existed. Instead it cloned `sgl-project/sglang` and
fetched **all 25,626 `refs/pull/*/head` refs**, then answered by git ancestry and diff.

For the questions asked, that is **stronger** than the API: ancestry and diffs are
deterministic. But it has a hard boundary.

| Label | Means | Trust |
|---|---|---|
| **GIT-VERIFIED** | Confirmed by local ancestry/diff against the real repo | High. Treat as fact. |
| **PR-DIFF-VERIFIED** | Confirmed by fetching and grepping the PR's actual diff in this session | High. Treat as fact. |
| **BASELINE-VERIFIED** | Confirmed by direct grep of *our* image/commit, not just `main` | High. Treat as fact. |
| **SEARCH-ONLY** | From web search; not API- or git-verified | Low. **Do not cite as fact.** |
| **UNVERIFIED** | Could not confirm at all | None. Listed only so nobody re-hunts it. |

**What the git method cannot see, ever:**

- **Issue bodies, titles, and comments.** Issues have no git ref. Zero issues were read.
- **PR titles as rendered on GitHub.** Head-commit subjects were used instead — usually but
  not always the same string.
- **Official open/closed/merged labels.** Inferred from ancestry ("head commit is not an
  ancestor of `main`" ⇒ open *or* squash-merged; either way its content is not in `main`).

**Do not upgrade a SEARCH-ONLY claim to fact when quoting this document.** Its whole value is
that the confidence labels are honest.

---

## 1. Our baseline, and why the version numbers matter

| | |
|---|---|
| Commit | `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` |
| Date | 2026-07-13 |
| Branch | `release/v0.5.15` |
| Version | `0.5.15.post1` |
| Image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| Hardware | AMD MI355 / gfx950, ROCm 7.2.0 |
| Topology | TP8 + DP-attention (`dp_size=8`) + EP8 |
| Model | GLM-5.2-MXFP4, `GlmMoeDsaForCausalLM`, DSA, `index_topk=2048` |
| Spec decode | EAGLE MTP spec-v2, `num_steps=3`, `draft_tokens=4` |
| Serving | PD disaggregation over mooncake RDMA |

GIT-VERIFIED: the baseline **is** an ancestor of `origin/release/v0.5.15` and **is not** an
ancestor of `origin/main`.

**Provenance note:** PyPI's `sglang` tops out at **0.5.10**. `0.5.15.post1` is published *only*
as the LMSYS ROCm docker image, never as a wheel. Line numbers quoted anywhere in this
directory come from that image and will not match a github checkout of any public tag.

### The two facts that shape every decision below

**(a) `release/v0.5.15` is frozen.** GIT-VERIFIED: exactly **one** commit exists after our
baseline on that branch, ever —

```
847d1bc3f47 | 2026-07-14 | [Cherry-pick to release/v0.5.15] [AMD] Pin cmake==4.3.4
                           in ROCm Dockerfile... (#30942) (#31262)
```

A cmake pin. **Nothing arrives by waiting.** Every fix in this document must be manually
backported or it does not reach us.

**(b) Upgrading does not help by itself.** `release/v0.5.16` is 576 commits ahead of
v0.5.15 — and GIT-VERIFIED, it still contains the greedy-broadcast bug (§2a) and the
idle-rank `publish_ready` bug (§2b/§4 Bug 4), verbatim. So "just move to v0.5.16" buys 576
commits of churn and fixes neither of the two defects that are actually blocking us.

---

## 2. The upstream defect landscape, by theme

Five themes emerged. They are not independent — (a) and (e) share a root cause, and (a), (b)
and (c) all converge on the *same wedge line* in `overlap_utils.py`.

### (a) Missing TP broadcast of the EAGLE verify decision on the greedy path — `_is_hip` forces this path

The single highest-value finding.

In `eagle_utils.py`, the greedy branch computes accepted-draft counts from a **per-rank local
`torch.argmax` with no TP broadcast**. The broadcast that would reconcile ranks lives inside
the *other* (sampling) branch.

BASELINE-VERIFIED by direct grep of **our** commit, not just `main`:

- `eagle_utils.py:620` —
  `if sampling_info.is_all_greedy or _is_npu or _is_hip or _is_xpu:`
  → **`_is_hip` unconditionally forces the greedy path on every AMD run.**
- `eagle_utils.py:726-728` — the
  `tp_group.broadcast(predict / accept_index / num_correct_drafts, src=0)` calls sit nested
  inside the `else:` (sampling) branch → **unreachable on HIP.**

GIT-VERIFIED that the same shape holds in `origin/main` (greedy gate ~line 721 with an extra
`_is_cpu`; broadcast block at 8-space indent inside `else:` at lines 826-834) and in
`release/v0.5.16`. **Still live everywhere as of the research date.**

The upstream commit that fixes this (`61c2af4d6cc`, GIT-VERIFIED commit message) describes
the failure mode:

> "When per-rank next_token_logits differ from a non-deterministic all-reduce (e.g. AMD
> `--enable-aiter-allreduce-fusion`) a near-tie makes argmax pick a different token per rank,
> so ranks accept a different number of drafts, committed seq_lens/batch shapes diverge, and
> the next TP collective deadlocks (both ranks wedge in **resolve_seq_lens_cpu ->
> Event.synchronize**; /health still 200 until the watchdog fires)."

That is our failure signature, including the `/health` 200 detail. **Note carefully:** the
all-reduce flag is the *trigger*, not the bug. `--disable-custom-all-reduce` suppresses one
trigger; it does not fix the missing broadcast. See §7 — this distinction is the crux of our
open question.

**Correction to earlier notes:** #31071 is **not** an "aiter all-reduce deadlock" PR. It is
this broadcast bug. Earlier notes (including `dpa.mtp.pd.debug.md` §7) grouped
#28815/#31071/#31478 as an all-reduce-deadlock cluster; that grouping is wrong.

### (b) Idle-DP-rank zero-size tensors and collectives on ROCm

Distinct from (a), same topology. Under DP-attention a rank with no work still must enter
every collective (see `dpa.mtp.pd.debug.md` §1 "Fact A"). On ROCm, several code paths then
launch **zero-sized** kernels or build metadata from empty tensors, and ROCm's collective
kernels reject that.

PR-DIFF-VERIFIED, from the #31683 diff, three concrete instances:

1. `_draft_extend_for_prefill` gains an early return of
   `EagleDraftInput.create_idle_input(...)` when `forward_mode.is_idle()`.
2. A HIP guard in all-reduce:
   `if is_hip() and not input_.is_cpu and input_.numel() == 0: return input_`, with the
   comment *"ROCm collective kernels reject a zero-sized launch. This can happen when every
   rank in a DP-attention subgroup is idle."*
3. `dsa_indexer.forward_cuda`'s guard widens from `seq_lens.numel() == 0` to
   `is_idle() or seq_lens.numel() == 0`.

Item 3 is adjacent to our Bug 1 (aiter indexer row mismatch) — same function, same class of
"idle batch reaches a kernel that assumes work". Ours was a row-count assert rather than a
zero-size launch, and our fix shipped independently.

### (c) DSA host-sync removal / `needs_cpu_seq_lens`

An active, ongoing upstream effort — framed by upstream as **performance**, which matters for
how we pitch our Bug 2 fix.

The v0.5.15 release notes / LMSYS blog (SEARCH-ONLY, paraphrase from a search-index summary,
**not byte-exact**) describe making `seq_lens_cpu` optional for DSA to "drop the D2H sync".

Merged, in our image (GIT-VERIFIED): **#29798** `fix: avoid DSA indexer CPU seq lens fallback`
(2026-07-03). Its scope was the eager path for batches exceeding `--cuda-graph-max-bs` — a
**global** property all ranks enter together. Its commit text explicitly declines the
`needs_cpu_seq_lens = True` workaround:

> "An earlier local workaround was to set `…needs_cpu_seq_lens=True`. That fixes the assertion
> but makes **all** spec-v2 DSA draft decode materialize a CPU sequence-length mirror,
> **including CUDA graph replay cases**…"

**Upstream never anticipated per-rank asymmetric entry into that path.** That gap is our Bug 2.
Do not propose flipping the flag; it has already been rejected on record.

Still open in this theme (GIT-VERIFIED as not-in-main): **#32490** `Avoid DSA top-k host
synchronization`; **#32317** `keep decode in cpu seq_lens gate; idle batches route there` —
directly about idle batches and the `needs_cpu_seq_lens` gate; **#31760** `Fix partial-DP DSA
decode padding`; **#32738** DSA indexer head-padding for DeepGEMM paged MQA logits.

### (d) DP-voted cuda-graph-vs-eager decisions

The theme upstream *thinks* explains this failure class. Two competing open PRs both add a
DP-min-reduced "can the spec/draft path use a graph" bit, so ranks vote instead of deciding
locally:

- **#32209** — `fix: narrow seedless MTP fallback to draft graphs`. Adds `can_draft_cuda_graph`
  to `MLPSyncBatchInfo`, min-reduced across DP ranks via the existing all-gather, plus
  `BaseSpecWorker.requires_dp_attention_eager_forward()`.
- **#32196** — `Cover DSA graph fallback in disagg draft input test`. Same idea, smaller and
  cleaner: `_spec_input_cuda_graph_compatible()` folded into the existing `can_cuda_graph`.

They overlap and will conflict; upstream will likely take one. **Neither can fix our Bug 2** —
a host-side sync is invisible to a vote about graph capture (see `dpa.mtp.pd.debug.md` §5).
And we measured `can_cuda_graph` is *uniformly* False on HIP, so there is no divergence for a
vote to resolve. Useful context, not a fix for us.

Backportability note (GIT-VERIFIED): the `can_cuda_graph` min-reduce **already exists in our
baseline** (`dp_attn.py:111`). Only the spec/draft bit is missing — so either idea is a small
patch, not a whole-PR backport.

### (e) ROCm argmax tie-break divergence

The root cause under (a), and **independently corroborated** by a second, unrelated place in
the tree. GIT-VERIFIED in `eagle_draft_extend_cuda_graph_runner.py` (main line 440, our
baseline line 430 — unchanged between them):

```python
# ROCm's argmax tie-breaks differently from CUDA's softmax+max
# path on FP8 logits, which corrupts MTP draft selection on AMD.
# Keep the fastpath CUDA-only.
if self.topk == 1 and not _is_hip:
```

So upstream has acknowledged, in two independent locations, that **ROCm argmax tie-breaks
differ from CUDA's**. This matters enormously for §7: argmax tie-break divergence is a source
of per-rank non-determinism that is **completely independent of all-reduce**, and therefore
**not suppressed by `--disable-custom-all-reduce`**.

---

## 3. PR / issue table

State column: "Open" = head commit is not an ancestor of `main` (open, or squash-merged with
content absent — either way not in main). Titles are head-commit subjects unless noted.

| # | What it actually is | Title | State | Date | Target | In our image? | Relevance to us |
|---|---|---|---|---|---|---|---|
| **29798** | PR, merged | `fix: avoid DSA indexer CPU seq lens fallback` | Merged | 2026-07-03 | main | **IN** | Defines the `needs_cpu_seq_lens` contract our Bug 2 violates; its text pre-rejects the flag-flip workaround |
| **30992** | PR, merged | `support GLM-5.2 MTP index sharing with prefill CP` | Merged | — | main | **IN** (via #31106) | IndexShare groundwork |
| **31106** | PR, merged | cherry-pick of #30992 to release/v0.5.15 | Merged | 2026-07-13 | release/v0.5.15 | **IS our HEAD** | — |
| **30839 / 31083** | PR, merged | `Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay` (`1bee78dc114`) | Merged | — | release/v0.5.15 | **IN** | IndexShare + PD + graph replay |
| **26672** | PR, merged | `[AMD] Work around HIP TPOT regression from Event.wait() in MTP seq lens resolution` | Merged | 2026-05-29 | main | **IN** | **Origin of the HIP-only blocking `synchronize()`** — see §2/§4 Bug 4 |
| **31071** | PR — **NOT merged on any branch** | `[Bug] eagle: broadcast finalized verify decision across TP ranks` | **Not merged** | 2026-07-16 | main | **NOT in** | **The greedy-broadcast fix (§2a).** Two commits carry the subject (`c198cf08e70`, `61c2af4d6cc`); neither is an ancestor of main, v0.5.15, or our baseline. `c198cf08e70` exists only on `pr/31683`. **Not an all-reduce PR.** |
| **31478** | PR, open | `test: strengthen eagle verify TP-broadcast regression test` | Open | 2026-07-16 | main | **NOT in** | Carries the #31071 product fix + a 161-line CPU regression test. **2 files, 178 insertions — the most backportable single item in this table.** |
| **31683** | PR, **open, not merged** (PR-DIFF-VERIFIED: base `main`, 2513 insertions / 136 deletions, ~34 files) | GitHub title: **`[ROCm][MI35X] Enable GLM-5.2-MXFP4 MTP speculative decoding`** (head-commit subject differs: `[ROCm] Route forced DP gather through capture-aware all-reduce`) | **Open** | 2026-07-20 | main | **NOT in** | **Closest thing to a ready-made fix for our exact topology.** See §3.1. |
| **32209** | PR, open | `fix: narrow seedless MTP fallback to draft graphs` (5 commits) | Open | 2026-07-26 | main | **NOT in** | DP-voted graph bit (§2d). Cannot fix Bug 2. |
| **32196** | PR, open | `Cover DSA graph fallback in disagg draft input test` | Open | 2026-07-29 | main | **NOT in** | Competing, smaller implementation of #32209's idea |
| **32722** | PR, open | `test: cover GLM-5.2 PD DP attention with MTP` | Open | 2026-07-29 | main | **NOT in** | GIT-VERIFIED: one new 125-line test file, **no product code**. Proves there is **no existing CI coverage** of PD+DPA+MTP. |
| **32374** | PR, open | `Fix DSpark device-only fallback layouts` | Open | 2026-07-25 | main | **NOT in** | The only open PR restructuring `publish()`/`publish_ready` — and it **leaves the idle early-return untouched**. See §4 Bug 4. |
| **32490** | PR, open | `Avoid DSA top-k host synchronization` | Open | 2026-07-28 | main | **NOT in** | Same direction as our Bug 2 fix |
| **32317** | PR, open | `keep decode in cpu seq_lens gate; idle batches route there` | Open | 2026-07-24 | main | **NOT in** | Idle batches × the `needs_cpu_seq_lens` gate |
| **31760** | PR, open | `Fix partial-DP DSA decode padding` | Open | 2026-07-20 | main | **NOT in** | Partial-DP + DSA padding |
| **32738** | PR, open | `[Fix] DSA Indexer: pad heads for DeepGEMM paged MQA logits on decode/target-verify` | Open | 2026-07-29 | main | **NOT in** | Indexer, adjacent to Bug 1 |
| **32187** | PR, open | `Fix PD disaggregation decode warmup crash under DP-attention (#30748)` | Open | 2026-07-23 | main | **NOT in** | PD + DPA |
| **30265** | PR, merged to main | `[AMD] Fix GLM-5.2 MTP Quark excludes` | Merged (main) | 2026-07-09 | main | **NOT in baseline** | On `amd/cherry-pick-release-v0.5.15` — see §6 |
| **29479** | PR, merged | `[AMD] fix dsv4 indexer dtype dispatch on gfx950` | Merged | — | main | **NOT in baseline** | On the same AMD branch — see §6 |
| **30557** | PR, merged | `[AMD] Fix AITER custom all-gather CUDA-graph capture crash under torch_memory_saver` | Merged | — | main | **NOT in baseline** | Same AMD branch — see §6 |
| **32570** | PR, open | `[AMD] Add GLM-5.2-FP8 8-GPU MI35x nightly accuracy CI` | Open | 2026-07-27 | main | **NOT in** | AMD CI direction |
| **32098** | PR, open | `[AMD] Fix gfx950 DSA prefill context-parallel RoPE ordering` | Open | 2026-07-22 | main | **NOT in** | gfx950 DSA |
| **31338** | PR, open | `[AMD] Fix packed DSA FP8 KV write routing on HIP` | Open | 2026-07-15 | main | **NOT in** | HIP DSA |
| **31955** | PR, open | `perf(dsa): prefill-only FP8 dense projection GEMM for GLM-5.2 MLA (gfx950)` | Open | 2026-07-15 | main | **NOT in** | Perf only |
| **31324** | PR, open | `fix(dense-decode): also gate eager decode k-only skip to HIP (AMD)` | Open | 2026-07-24 | main | **NOT in** | HIP eager gating |
| **31500** | PR, open | `[AMD][DI][CI] Gate overlap-plan-stream on non-MTP` | Open | 2026-07-21 | main | **NOT in** | Notable: **AMD CI is disabling the overlap plan stream when MTP is on** |
| **32527** | **NOT a PR — an issue** | unreadable | Open (SEARCH-ONLY: "opened by Xavier1994 on Jul 27, 2026") | ~2026-07-27 | n/a | n/a | Reportedly same topology and precondition as ours; **body never read** |
| **28815** | **No PR ref exists** | — | — | — | — | — | **UNVERIFIED entirely.** No `refs/pull/28815/head` in a full 25,626-ref fetch; search could not surface it. Stop citing it. |

### Corrections this research made — carry these forward

1. **#32527 and #28815 are not PRs.** Neither has a `refs/pull/*/head` ref in a full fetch of
   all 25,626 refs, while #32209, #32722 and #31478 all do. #32527 is an issue. **#28815 could
   not be confirmed to exist at all.**
2. **#31071 is not an all-reduce-deadlock PR.** It is the greedy-path missing-broadcast fix
   (§2a). The all-reduce flag appears in its commit message only as an example *trigger*.
3. **#31071 is not merged anywhere** — not main, not v0.5.15, not our baseline.
4. **Issue titles are SEARCH-ONLY and must stay that way.** These appeared relevant but no
   body was ever read: #29347 `[Bug][AMD][ROCm/MI300X] EAGLE speculative decoding + CUDA
   graph: non-deterministic deadlock in greedy verification fallback`; #26454 `[Bug] Non-DP
   multi-node TP=8 hang in event_loop_overlap ... all TP ranks block in
   result.copy_done.synchronize()`; #13435 `[Bug] PD decode DP attention + MTP crash`; #7026
   `[Bug] PD+MTP+DeepEP+dp attention`; #31116 `DP attention + prefill CUDA graph...`.
   **Treat every one of these titles as unverified.**

### 3.1 PR #31683 in detail — the one to mine

PR-DIFF-VERIFIED this session: **open, not merged**, base `main`, **2513 insertions / 136
deletions across ~34 files**, GitHub title `[ROCm][MI35X] Enable GLM-5.2-MXFP4 MTP speculative
decoding`. A 10-commit ROCm stack by `long10024070`, 2026-07-20.

It **contains the #31071 fix**: its `eagle_utils.py` hunk removes the broadcast from inside the
`else:` branch and re-adds it **outside both branches**, with a new comment explaining that
*both* the greedy (per-rank argmax) and the sampling path can diverge. That is exactly the
right shape of fix.

Plus the idle-DP-rank work quoted in §2b, plus (from head-commit subjects, GIT-VERIFIED):
`fix: handle idle DP ranks in EAGLE draft sampling`, `fix: handle idle DP ranks without
zero-work ROCm launches`, `fix(rocm): skip empty GPU all-reduce kernels`, `fix(rocm): skip
empty AITER paged MQA logits`, `[ROCm] Keep speculative DSA metadata consistent with padded
batches`, `fix: avoid device cumsum for eager DP logits metadata`. Touches `communicator.py`,
`dp_attention.py`, `logits_processor.py`, `eagle_worker_v2.py`, `eagle_utils.py`,
`forward_batch_info.py`.

**Idle-DP-rank + ROCm + EAGLE + GLM-5.2-MXFP4 — precisely our topology, and neither we nor the
prior notes knew it existed.**

**But it does not cover everything.** PR-DIFF-VERIFIED by grepping the diff:

- It does **not** touch `overlap_utils.py` at all — **zero matches**. So the idle-rank
  `publish_ready` bug (Bug 4) is untouched.
- It does **not** touch the `max_seqlen_k` / `seq_lens.max()` D2H sync in `dsa_backend.py` —
  **zero matches**. So our Bug 2 is untouched.

Too large to backport whole onto frozen v0.5.15. **Mine it commit by commit.**

---

## 4. Coverage matrix — our four bug layers vs upstream

| | Bug | Ours / upstream | Our status | Upstream coverage | Evidence |
|---|---|---|---|---|---|
| **Bug 1** | aiter indexer row mismatch (`lengths.size(0) == B`) on idle DPA batches | Ours | **Fixed, shipped** (`infera` branch `worktree-dsa-hip-dp-rows-fix`, `9bcec49`) | **Adjacent only.** #31683 widens the `dsa_indexer.forward_cuda` guard to `is_idle() or seq_lens.numel() == 0` — same function, related class of defect, but a different symptom. | PR-DIFF-VERIFIED |
| **Bug 2** | `dsa_backend` rank-divergent D2H sync — busy rank blocks on `seq_lens.max().item()` while idle peers race into the next collective | Ours | **Root-caused, fix designed & measured effective; see `dpa.mtp.pd.debug.md` §4/§6** | **NOT COVERED.** #31683 does not touch `dsa_backend.py`'s `max_seqlen_k` path (zero grep matches). #32209/#32196 vote on graph capture, which is invisible to a host sync. #29798 fixed the *global* eager path, never per-rank asymmetric entry. | PR-DIFF-VERIFIED + GIT-VERIFIED |
| **Bug 3** | Greedy-path missing TP broadcast of the verify decision; `_is_hip` forces the greedy path | Upstream's | **Not yet applied, not yet verified by us** | **COVERED.** #31071 (the fix commits) and #31683 (which contains them) both fix it. #31478 carries the same product fix in a 2-file / 178-insertion diff. **None of them is merged.** | BASELINE-VERIFIED (bug) + PR-DIFF-VERIFIED (fix) |
| **Bug 4** | `publish_ready` is `None` on idle ranks, because `FutureMap.publish()` early-returns at `if indices.shape[0] == 0: return  # DP idle` **before** lazily constructing the Event | Ours | Identified, unfixed | **NOT COVERED.** A sweep of every PR ref ≥ 30000 for additions mentioning `publish_ready` or `DP idle` in `overlap_utils.py` found **12 PRs touching the file and none changing the early return**. The closest, **#32374**, heavily restructures `publish()` (adds `_publish_stream`, a `reset()`, a GPU-only `resolve_seq_lens_device()`, enables the pinned relay on HIP via `if _is_cuda or _is_hip`) yet leaves `if indices.shape[0] == 0: return  # DP idle` **completely untouched**. #31683 does not touch the file at all. | GIT-VERIFIED + PR-DIFF-VERIFIED |

### Why Bug 4 is worse on HIP than on CUDA

GIT-VERIFIED in `overlap_utils.py`, present **verbatim and unchanged** in our baseline
(line 294), `origin/main` (line 430) and `release/v0.5.16` (line 430), introduced by #26672
(`ace730db48c`, 2026-05-29):

```python
if _is_hip:
    # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
    self.publish_ready.synchronize()   # BLOCKING HOST wait
else:
    self.publish_ready.wait()          # non-blocking STREAM wait
```

`Event.synchronize()` blocks the **Python thread**; `Event.wait()` only orders a **stream**.
So on HIP a busy rank blocks while an idle rank (whose `publish_ready` is still `None`, so it
skips the whole block) sails past — the exact asymmetry that turns a benign rank divergence
into a hard deadlock.

**All roads lead here.** This is also the wedge point named in #31071's commit message ("both
ranks wedge in `resolve_seq_lens_cpu -> Event.synchronize`"). Bug 3 and Bug 4 converge on this
one line. It has never been replaced.

One more GIT-VERIFIED subtlety: `dsa_backend.py` sets `needs_cpu_seq_lens: bool = False` (main
line 350, baseline line 339), so `resolve_seq_lens_cpu` returns early — **but only *after* the
HIP `synchronize()` has already run.** The sync is *not* skipped for us.

---

## 5. What is ours to upstream

**Bug 2** and **Bug 4** appear unreported. Both are genuine gaps, not local misconfiguration.

- **Bug 2** — pitch it as an extension of upstream's own D2H-sync-removal effort (§2c), which
  they frame as performance; our contribution reframes it as **also a correctness fix**
  (per-rank asymmetric entry → DP deadlock). #32722 shows they *want* PD+DPA+MTP coverage and
  have none. Do **not** propose `needs_cpu_seq_lens = True` — #29798 already rejected it on
  record.
- **Bug 4** — nobody upstream has touched the idle early-return in 12 PRs. **This will not
  conflict with the open #32374 as long as our fix only guards the idle path** and leaves the
  `publish()` restructuring to them.

Caveat on the negative claims: no issue body was ever readable, so "unreported" is
**UNVERIFIED-negative** (not proven absent) rather than proven. The *code* claims — that no
merged or open PR changes these lines — are GIT-VERIFIED and solid.

---

## 6. Leads worth following

**`origin/amd/cherry-pick-release-v0.5.15`** — GIT-VERIFIED: a **9-commit AMD-only branch off
our exact release line**, never merged into `release/v0.5.15`. Because it is based on our
release line, these should apply near-trivially. It carries at least:

- `[AMD] Fix GLM-5.2 MTP Quark excludes (#30265)` — merged to main 2026-07-09, **not in our
  baseline**. MXFP4 quantization excludes for GLM-5.2 MTP; directly on our model+feature.
- `[AMD] fix dsv4 indexer dtype dispatch on gfx950 (#29479)` — gfx950 indexer dtype dispatch;
  same file family as Bug 1.
- `[AMD] Fix AITER custom all-gather CUDA-graph capture crash under torch_memory_saver
  (#30557)`.

**Review this branch before writing any new AMD patch.** Someone at AMD already staged fixes
against our exact baseline and they never landed.

Other leads, in rough priority order:

1. **Cherry-pick `61c2af4d6cc` from `pr/31478`** (2 files, 178 insertions) — the cheapest real
   fix for Bug 3, and it comes with a CPU regression test.
2. **Mine #31683 commit by commit** for the idle-DP-rank ROCm fixes (§2b).
3. **#32196's `_spec_input_cuda_graph_compatible()`** — since `can_cuda_graph` min-reduce
   already exists in our baseline, this is a small patch, not a PR backport. Low expected value
   for us (we measured `can_cuda_graph` uniformly False), but cheap.
4. **No open PR enables draft-extend CUDA graph on HIP.** A full search found none. If we want
   Fix B from `dpa.mtp.pd.debug.md` §6, we are on our own.

---

## 7. Open questions — UNRESOLVED

### ⚠️ The `--disable-custom-all-reduce` contradiction — read this before debugging further

**We run with `--disable-custom-all-reduce`.** Per #31071's commit message, that should
suppress the non-deterministic-all-reduce trigger it describes. **Yet we still deadlock.**

Exactly one of these must be true, and we do not know which:

1. **There is another non-determinism source.** The strongest candidate is the **ROCm argmax
   tie-break** (§2e) — upstream's own comment in
   `eagle_draft_extend_cuda_graph_runner.py` says ROCm's argmax tie-breaks differently from
   CUDA's on FP8 logits. **That divergence is independent of all-reduce and is therefore not
   suppressed by `--disable-custom-all-reduce` at all.** If ranks disagree on a near-tie
   argmax, the missing greedy broadcast (Bug 3) still lets accept-counts diverge, and we still
   wedge in `Event.synchronize`. Under this hypothesis the flag was never going to help and
   Bug 3 is fully live for us.
2. **More than one bug is live.** Bug 2 (our measured `.item()` stall) and Bug 3 (missing
   broadcast) can both produce a wedge at the same place. Bug 2 is measured and reproduced with
   py-spy (`dpa.mtp.pd.debug.md` §3); Bug 3 is verified in source but **we have never observed
   it firing**. Fixing only one may leave the hang.

**Do not assume applying the #31071 broadcast fix will resolve the hang, and do not assume
fixing Bug 2 will either.** Instrument for which one fires before choosing. The two are
distinguishable: Bug 2 stalls a rank at `dsa_backend.py`'s `seq_lens.max().item()` with the
others in a collective; Bug 3 would show ranks with *different accept counts* / divergent
`seq_lens` before anyone blocks.

### Lower-priority open questions

- **Is CUDA permanently immune?** `can_run_dp_cuda_graph` is an all-gathered *min*; if it goes
  False under partial occupancy, CUDA falls eager and hits Bug 2 too. Unresolved statically.
  Strengthens the case for a platform-independent fix. (See `dpa.mtp.pd.debug.md` §5.)
- **Does the GLM-5.2 cookbook disable MTP on AMD outright?** SEARCH-ONLY, resting on a
  search-index summary of a JS-rendered page: *"MTP on gfx950 still depends on the spec-decode
  draft kernel, which isn't yet validated on this hardware."* If true, we are the first to
  exercise this path on gfx950 and should expect more bugs behind these. **Unverified — do not
  cite.**
- **Why is AMD CI gating the overlap plan stream off when MTP is on (#31500)?** Possibly a
  known interaction we have not characterized.
- **Issue bodies were never read.** #32527 (reportedly our exact topology, opened one day before
  our investigation), #29347, #26454, #13435, #7026, #31116 are all unread. Anyone with a
  working GitHub token should read these first — it is the cheapest remaining information.

---

## 8. Cross-references

| File (same dir) | Covers |
|---|---|
| `dpa.mtp.pd.debug.md` | **Read first.** Our bugs, measured evidence, root-cause chains, fix options, live environment, traps |
| `CODE_VERIFICATION_bug2.md` | Source reading behind Bug 2 |
| `RESEARCH_bug2.md` | Bug 2 upstream + CUDA-vs-HIP comparison + docs |
| `HANDOFF_pd_mtp_hang.md` | Chronological trail, including refuted hypotheses |
| `NOTES_rootcause_and_fix.md` | Bug 1 |
| `/home/yihou/glm52_fix/github_research_report.md` | Raw research report this document distils |

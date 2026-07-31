# sglang DSA patches

Patches against the sglang source tree bundled in the ROCm engine images
(`lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` and derivatives, where sglang is an
editable checkout at `/sgl-workspace/sglang`).

> **`Dockerfile.sglang` applies these at build time by default**
> (`APPLY_SGLANG_DSA_PATCHES=1`), via
> `deploy/docker/scripts/apply_sglang_dsa_patches.sh`, which also verifies each patch
> reached the **bytecode** (a stale `__pycache__` entry silently reverts a patch and has
> already invalidated a full experiment here). Set `APPLY_SGLANG_DSA_PATCHES=0` to build
> a stock engine for A/B.
>
> The base image tag is pinned deliberately: these target one sglang commit and are
> applied at `--fuzz=0`, so a base bump fails the build at the patch step rather than
> mis-applying silently.
>
> **Prerequisite, applied earlier in the same Dockerfile:**
> `deploy/docker/patches/sglang/patch_glm52_nextn_quark_exclude.py` fixes the GLM-5.2
> nextn `eh_proj` quark-exclude check (backport of sglang #30265). GLM-5.2 MTP cannot
> load its draft weights without it — a `3072 vs 6144` shape mismatch. It is not part of
> this set, but `apply_sglang_dsa_patches.sh` **asserts** it, because that script is
> idempotent and would otherwise "skip" silently.

## `dsa_indexer_hip_dp_padded_rows.diff`

**Enables DP-attention and EAGLE MTP to run together for GLM-5.2 DSA on gfx950.**
Without it, the combination crashes on the first batch:

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```

The DSA indexer's aiter/HIP paged-MQA path sizes its `logits` output from the
**DP-padded** row count, while the CUDA path slices to the real row count (`q_offset`) —
which is what `lengths` is sized to. The patch makes HIP follow the CUDA contract: run
top-k over the real rows, then restore the padding. Two guards keep the non-DP path
bit-identical. Full rationale is in the patch header.

Applies to `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, exact context, no
fuzz:

```bash
cd /sgl-workspace/sglang
patch -p1 --fuzz=0 < dsa_indexer_hip_dp_padded_rows.diff
```

Or bind-mount the patched file over the container's copy.

### Verified

8× MI355X (gfx950), ROCm 7.2.0, sglang 0.5.15.post1, GLM-5.2-MXFP4, single node
`--dp-size 8 --enable-dp-attention --ep-size 8` + `--speculative-algorithm EAGLE
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`:

| Check | Result |
|---|---|
| Correctness probe | 4/4 |
| Spec-dec accept length | median 3.86 / 4 (n=251) |
| conc=64, ISL/OSL 1k/1k, 256 prompts | 256/256, 0 failed, median TPOT 17.2 ms |
| Regression: DP-attention only | 4/4, unchanged |
| Regression: MTP only | 4/4, unchanged |

Full reproduction kit (environment, commands, raw logs, evidence, root-cause notes) lives
in the `infera.yihou.glm5.2.mxfp4` workspace under
`glm52.mxfp4.spur.mooncake.packup_20260728/dpa_mtp_fix/`.

> **Updated 2026-07-29.** This diff was revised: its slice guard was
> `if 0 < q_offset < q_fp8.shape[0]`, and the `0 <` lower bound made the slice a
> no-op on DP-**idle** ranks (where `q_offset == 0` because the rank holds no
> requests), reproducing the very same assert on those ranks. It only surfaces
> once some ranks are idle while others are busy, i.e. under concurrency, which
> the original single-node validation never exercised. The bound is now removed.
> `q_fp8[:0]` is a legal empty slice and is what an idle rank should pass; the
> CUDA path has no such bound either.

## `dsa_backend_dp_sync_and_page_table_rows.diff`

Two fixes to `python/sglang/srt/layers/attention/dsa_backend.py`, both required
for **PD disaggregation with MTP on the decode leg**.

**1 — rank-divergent GPU→CPU syncs (fixes the PD+MTP deadlock).**
`max_seqlen_k = int(forward_batch.seq_lens.max().item())` is a host sync sitting on a
branch that only *some* DP ranks take; idle peers take the cheap arm, so the DP
collectives desynchronize and the group deadlocks. Replaced with the sync-free
`self.req_to_token.shape[1]`, the same idiom already used by
`_graph_page_table_width()` and the graph-capture paths. Over-allocating columns is
safe: the page table is only indexed through top-k, which masks per row by
`cache_seqlens`. Two further unconditional `.cpu()` syncs are removed.

*How this was established.* Not by reverting the fix, but by observing the divergence
directly: a py-spy dump caught a **busy** rank blocked inside `dsa_backend` on
`.max().item()` while **idle** peers had already advanced into the next collective. After
the fix, no rank appears in `dsa_backend` in a dump again and PD warmup passes on all 8
ranks. The second half (removing the two unconditional `.cpu()` syncs) was **forced by
experiment, not by reading**: with the `max_seqlen_k` change applied and those syncs left
in, the hang persists — they sit on the same rank-divergent branch. Safety of the wider
page table was verified by walking every consumer, and the sync-free idiom is established
in-tree three times (`triton_backend.py:704-708`, `trtllm_mha_backend.py:555-559`, and
DSA's own graph path at `dsa_backend.py:689-695`).

Upstream does not cover this. Per-diff greps: #31683 does not touch this file's
`max_seqlen_k` path at all; #32209/#32196 vote on *graph capture*, which a host-side sync
is invisible to; merged #29798 repaired the **global** eager path (batches exceeding
`--cuda-graph-max-bs`, entered by all ranks together) and never anticipated per-rank
asymmetric entry.

**2 — page-table rows vs top-k rows (fixes a crash under concurrency).**
`metadata.page_table_1` is `req_to_token[req_pool_indices]`, one row per **request**,
while `topk_indices` has just been padded to `q.shape[0]`, one row per **token**. Those
are equal for plain decode but not under MTP, where the draft model runs several tokens
per request — so `transform_index_page_table_decode_fast` trips
`assert page_table.shape[0] == topk_indices.shape[0]` and every rank dies:

```
deepseek_nextn.py:271 -> dsa_backend.py:2154 forward_decode
  -> dsa/transform_index.py:138  assert page_table.shape[0] == topk_indices.shape[0]
```

A helper expands the page table to token rows via `repeat_interleave`. Rows added by
`_pad_topk_indices` hold all `-1` and are masked out by the triton kernel
(`valid_topk_mask = mask & (loaded_topk_indices >= 0)`), so only the row *count* has to
match. Applied at **both** decode call sites — the traceback named only the first, but
the second pairs the same unpadded page table with a padded `topk_indices`. The prefill
sibling already solves this with an explicit `output_num_tokens` argument; the decode
entry point had no equivalent.

## nextn `eh_proj` — not here, applied earlier

Earlier revisions of this directory carried a fourth diff,
`deepseek_nextn_glm52_mtp_bf16.diff`: one line in
`python/sglang/srt/models/deepseek_nextn.py` making the quark-exclude check test
`model.layers.{N}.eh_proj` instead of the bare layer prefix. **Required for GLM-5.2
MTP**, whose nextn layer is bf16 while the rest of the model is quantized — without it
the weight load dies with a `3072 vs 6144` shape mismatch.

It was removed as a duplicate: `deploy/docker/patches/sglang/patch_glm52_nextn_quark_exclude.py`
(main, commit `0d8d0ff`) already makes the identical edit — same file, same line, same
resulting value — as an idempotent Python patch that runs earlier in `Dockerfile.sglang`.
Keeping both would have been worse than redundant: main's script runs first, so our
context diff would then fail at `--fuzz=0` on an already-edited anchor.

`apply_sglang_dsa_patches.sh` asserts the edit is present before proceeding.

## `draft_cuda_graph_dp_vote.diff`

**Lets the EAGLE draft CUDA graph stay enabled on the PD decode leg.** Without it the
first routed request hard-deadlocks.

`draft()` decides per rank whether to replay the draft graph or run the multi-step draft
eagerly. The two paths do not issue the same host-side collective sequence, and two of
the guard's four terms are rank-dependent by construction:

* `not forward_batch.forward_mode.is_idle()` — occupancy differs per rank every step;
* `draft_input.dsa_topk_indices is None` — on the **PD decode leg** this field is built
  from RDMA-shipped per-request payloads (`eagle_disaggregation.py:54-59`), so it is a
  function of which requests a rank happens to hold.

So one rank replays a graph while another runs eager inside the same collective step, and
the group desynchronizes permanently.

Measured rather than argued: instrumenting all four terms per rank per iteration, the
*decision* diverged on **exactly** the iteration that froze — one busy rank eager, seven
idle ranks graph — and that rank is exactly the one py-spy found blocked in the eager
`init_forward_metadata`. Reproduced twice with different victim ranks (dp2, dp3), which
is the race behaving as a race. This also explains why single-node **mix** never hangs:
with no disaggregation the top-k seed is always produced locally by identical code on
every rank, so the fourth term never flips asymmetrically.

### Where the vote lives, and why it moved

The fix makes the graph/eager choice a **group** decision instead of a per-rank one. Two
placements were implemented and both were measured:

| | placement | collectives added | draft-graph usage |
|---|---|---|---|
| v1 (superseded) | a 1-element gloo all-reduce inside `draft()` | **one per draft call** | 98.4 % |
| **v2 (this diff)** | one more int64 slot in the MLP-sync all-gather the scheduler **already** performs | **none** | **97.1 %** |

v2 is the shape of upstream PR [#32209](https://github.com/sgl-project/sglang/pull/32209)
(`Fix PD decode hang with DP attention and GLM-5.2 MTP`, open). Adopting it means this
tree converges with upstream rather than diverging from it, and it drops a collective
whose cost we never measured. It also separates the *draft*-graph vote from the generic
`can_cuda_graph` vote, so target-verify and draft-extend graphs are not collateral damage
— #32209 reports that folding them together cost 9.7 % of decode batches their generic
graph replay.

The rank-local predicate is `requires_dp_attention_eager_forward()`; the scheduler calls
it just before `maybe_prepare_mlp_sync_batch`, and the gathered value is `min()`-reduced
so **any** rank needing eager takes the whole group eager. Idle/inactive ranks contribute
`1` (permissive) and so can never drag the group into eager on their own.

Two details that would otherwise make the patch silently inert:

* under overlap scheduling the draft input is resolved **after** the scheduler-side vote,
  so `dsa_topk_indices` is stale at that point and must not be read directly. The
  predicate consults `future_dsa_topk_indices_available` instead — which is exactly
  "will term 4 be satisfied once resolved". An earlier revision wrongly assumed that
  attribute was absent and fell back to requiring eager whenever `future_indices` was
  set; since overlap scheduling sets it on **every** decode iteration, that refused the
  draft graph 100 % of the time (measured: 0.0 % usage, 200/200 calls refused) and
  silently degraded the fix into the "just disable the draft graph" workaround while
  still passing every functional test.
* the v1 form had to use `get_tp_group()`, **not** `get_attention_tp_group()` — under
  DPA8 on tp8 the attention TP group is `attn_tp_size = tp/dp = 1` rank wide, so voting
  on it is a no-op. v2 rides the scheduler's existing all-gather and so does not face
  this choice.

An earlier attempt that removed only the `not is_idle()` term does **not** work, and the
measurements say why: it fixes the busy-rank iteration and breaks the idle-rank one, which
matches the observed symptom of the failure moving earlier, into warmup.

### Verified

2× 8×MI355X, GLM-5.2-MXFP4, PD over mooncake/mlx5, DPA8 + EAGLE MTP(3,1,4), **draft graph
enabled**.

v1's evidence, which established the root cause and carries the differential control:

| Check | Result |
|---|---|
| PD warmup, 4×24-token, 1×512-token | pass |
| conc 1/2/4/8/16/64/128/256 × 512 tok | **1384/1384**, 0 failures, 0 KVTransferError |
| Single-node mix regression (shared code path) | **132/132** |
| Durability, 8 × conc=128 × 512 back to back | **1024/1024**, accept length flat at ~3.0 |
| **Draft graph actually used** | **98.4 %** of iterations |
| Same-node control, only this diff reverted | **0/4 — deadlock, 120 s timeout on request 1** |

Cumulative **2540/2540** with the fix, **0/4** without.

The mechanism was verified, not just the outcome. Over 2992 iterations with all 8 ranks
logging: the *local* decision still diverges 38 times (the defect is latent and real, not
masked by timing) while the *acted-on* decision diverges **0** times, and the vote changed
some rank's mind 190 times — each one an averted deadlock.

v2 (this diff) was then measured on its own:

| Check | Result |
|---|---|
| 4-prompt probe | 4/4, accept length 2.18–3.00 |
| conc=32 × 512, ×2 | **32/32**, **32/32** |
| conc=64 × 512 | **64/64** |
| **Draft graph actually used** | **97.1 %** (777/800 calls, identical on all 8 ranks) |

The graph-usage number is the point of that run. Forcing the draft path eager passes every
functional test while disabling the feature under test, so a green stress result alone
cannot distinguish a fix from that workaround — the usage counter can.

> The v1 diff (`eagle_worker_v2_uniform_draft_graph.diff`) has been **removed** from this
> directory in favour of v2. It is preserved, with its full evidence, in
> `glm52.mxfp4.spur.mooncake.packup_20260729_bug2b_draft_graph/` in the
> `infera.yihou.glm5.2.mxfp4` workspace.

## A configuration-only alternative to part of this set

Turning GLM-5.2's MTP **IndexShare** off avoids the same deadlock without patch 04 or the
page-table half of patch 02:

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

It works because IndexShare is the *source* of the rank divergence: the guard term
`draft_input.dsa_topk_indices is None` is seeded on the PD decode leg from RDMA-shipped
per-request payloads (`eagle_disaggregation.py:54-59`), so it is a function of which
requests each rank happens to hold. Remove the seed and the term stops diverging — no vote
needed. Each diff's header records where it stands relative to this.

| patch | substituted by IndexShare-off? |
|---|---|
| 01 `dsa_indexer_hip_dp_padded_rows` | **No** — independent bug, present regardless |
| 02a `dsa_backend` DP host-sync | **No** — a host sync is invisible to the graph/eager decision either mechanism changes |
| 02b `dsa_backend` page-table rows | **Yes**, in effect — the arm ran without it and passed |
| nextn `eh_proj` (the prerequisite) | **No** — weight-load bug, unrelated |
| 04 `draft_cuda_graph_dp_vote` | **Yes** — this is what it targets |

Measured on 2 × 8 MI355X with 02b and 04 asserted **absent from the bytecode**: 4/4 probe,
32/32 twice, 64/64, zero tracebacks, accept length 2.98–3.01 — no measurable cost. Two
conditions are easy to miss: MTP must be on the **prefill** leg too (otherwise the seed
never reaches decode and the setting is untested rather than tested), and that arm was
taken to conc=64, not 128.

**Why the patches remain the default here.** The override is nearly free only because
IndexShare's consumer is currently disabled under PD by `should_use_dsa_fused_topk`.
Upstream PR [#31477](https://github.com/sgl-project/sglang/pull/31477) exists to remove
that limitation; once it lands the override starts costing (~3 % TPOT, reported by AMD's
llying — second-hand, not measured by us). Checked with `gh` on 2026-07-31: #31477 is
**open**, `reviewDecision = REVIEW_REQUIRED`, unmerged. So it is a good answer today if
IndexShare is not wanted, and a dated one if it is.

Full arm with anti-marker verification:
`glm52.mxfp4.spur.mooncake.packup_20260730_exp2_indexshare_off` in the
`infera.yihou.glm5.2.mxfp4` workspace.

## Applying

All three apply with exact context, no fuzz, against sglang commit
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (v0.5.15.post1):

```bash
cd /sgl-workspace/sglang
for d in dsa_indexer_hip_dp_padded_rows.diff \
         dsa_backend_dp_sync_and_page_table_rows.diff \
         draft_cuda_graph_dp_vote.diff; do
  patch -p1 --fuzz=0 < "$d"
done
```

`deploy/docker/scripts/apply_sglang_dsa_patches.sh` does exactly this and then verifies
each patch in the **bytecode**; prefer it. `Dockerfile.sglang` runs it at build time.

Verified by applying for real into a scratch tree and byte-compiling the result — note
that `patch --dry-run` and `git apply --check` **fuzz by default**, and a hand-written
diff in this series once silently dropped a hunk while still "passing".

## PD + MTP status

> **Provenance of the numbers below (2026-07-29).** They were measured with the **v1**
> draft-graph vote and the **v1** indexer patch. Both have since been reshaped to follow
> upstream (#32209 and #32762 respectively), and each reshape was separately re-validated
> — but the two reshaped patches were validated in *different* runs, never together.
> They were also measured with the nextn fix supplied as a fourth diff in this directory
> rather than by the earlier patch loop; the two make the identical edit. The end-to-end
> run of the current arrangement is recorded below under "Final validation".

With the full set applied, PD disaggregation with DP-attention + EAGLE MTP and the
**draft CUDA graph enabled** is working on 2× 8×MI355X, GLM-5.2-MXFP4,
`--dp-size 8 --enable-dp-attention --ep-size 8`, mooncake RDMA over mlx5 + dma-buf:

| Check | Result |
|---|---|
| PD warmup (8 concurrent, one per DP rank) | passes |
| conc 1/2/4/8/16/64/128/256 × 512 tokens | **1384/1384** |
| Durability, 8 × conc=128 × 512 back to back | **1024/1024** |
| Scheduler exceptions / KVTransferError | **0 / 0** |
| Spec-dec | active, mean accept length ~2.9, all 8 `dp_rank`s serving |

### Both earlier caveats are now closed

1. ~~"the draft CUDA graph must be disabled; a proper fix is not written yet"~~ —
   **written and verified**, see `draft_cuda_graph_dp_vote.diff` above. The
   earlier 640/640 was measured with that graph off; it is now on and used 98.4% of the
   time.
2. ~~"about 2 % of responses under concurrency are degenerate"~~ — **falsified; this was
   a test-harness error, not an engine bug.** The harness posted raw prompts to
   `/generate`, skipping GLM-5.2's chat template (so the model was doing base-LM
   completion), and forced `temperature=0` over the model's own recommended
   `temperature 1.0 / top_p 0.95`. Degenerate repetition under greedy decoding is expected
   behaviour (Holtzman et al. 2019), not a defect. With the template applied and the
   model's own sampling, **0/128 degenerate at conc=128 on both MXFP4 and FP8** — and the
   official FP8 build reproduced the "failure" under the *wrong* config exactly as MXFP4
   did, so quantization was never implicated either.

   The related note about `eagle_utils.py:620` forcing `argmax` on HIP stands as a code
   observation, but it is confined to the spec-decode verify path and does not discard
   user sampling params for the request as a whole — that over-generalization is
   withdrawn.

### Final validation (2026-07-31) — this directory's exact contents

Re-run after the three overlaps with main were dropped, so that what was measured is what
this branch ships: an image built from `Dockerfile.sglang` on the PR branch, patches
applied **at build time**, nothing patched in the running container. 2 × 8×MI355X
(gfx950), GLM-5.2-MXFP4, PD + `--dp-size 8 --enable-dp-attention --ep-size 8` + EAGLE MTP
with the draft CUDA graph **enabled**, mooncake RDMA over mlx5 + dma-buf
(`MOONCAKE_DISABLE_HIP_DMABUF=0`).

| Check | Target | Result |
|---|---|---|
| Build-time bytecode verification, both nodes | all markers | **8/8 + prereq + patch2a** |
| 4-prompt correctness probe | 4/4, `acc_len` > 1 | **4/4**, 2.00–3.43 |
| conc=32 × 512 tok | 32/32 | **32/32** |
| conc=128 × 512 tok | 128/128 | **128/128** |
| conc=128 × 512, repeat | — | **128/128** |
| `Traceback` / `KVTransferError`, either leg | 0 | **0 / 0** |
| DP ranks serving | 8 | **8**, every run, 0 retries |

The nextn prerequisite was confirmed to come from the earlier patch loop and not from
this directory: the edited line in the running image carries **no** trailing comment,
which our removed diff would have added. `apply_sglang_dsa_patches.sh` logged
`PREREQ nextn eh_proj -> src=1` after `[glm52-nextn] patched ...`, in that order.

**Not established by this run:** the draft-graph replay count was not re-measured (it
needs an added probe, i.e. a different image — measured at 92.0 %, identical on all 8
ranks, on the immediately preceding build of the same patch set). No differential control
was re-run either; each patch's necessity is established in the kits below.

Measurements: `glm52.mxfp4.spur.mooncake.packup_20260729_bug2b_draft_graph/` (the draft
graph fix, with its differential control) and
`glm52.mxfp4.spur.mooncake.packup_20260729_degenerate_output/` (the falsification, with
`RETRACTIONS.md`), both in the `infera.yihou.glm5.2.mxfp4` workspace.

### Known limits of the validation

* Performance was **not** measured against the DPA-only baseline. The vote adds one
  1-element host collective per `draft()` call; the cost is expected to be negligible but
  is unquantified.
* All runs used `--disable-custom-all-reduce` (required on gfx942/gfx950 for EAGLE), so
  the custom all-reduce path remains unexercised.
* Context 32768, short prompts, 512-token outputs, one image, one hardware
  configuration. Long-context and 400k-context configs are untested.
* A seventh occurrence of the padded-vs-real-rows crash family was seen **once** in 500+
  requests during an earlier session, on a machine already carrying the `dsa_indexer` fix.
  It did not recur in 2540 requests here, but at that rate absence is not evidence.

### Upstream

The `dsa_indexer` fix is ROCm-specific; the CUDA path was always correct. It is **not**
the same issue as sglang PR #30378 / #30427 ("clamp padded-row seq_lens to >= 0" in
`triton_ops/pad.py::seqlens_expand_kernel`), which fixes padded row *values* and is
already present in this image. This one fixes the HIP-side row *count*.

The draft-graph vote is **not** ROCm-specific in principle. The guard it repairs is
platform-neutral, and its terms are rank-dependent on any backend running DP-attention
with PD disaggregation; CUDA is protected today only because the draft-extend graph is
captured there, which keeps the ranks on the same path.

State of the relevant upstream work, queried with `gh` on 2026-07-31 (all **open**, none
merged — so none of it reaches a `release/v0.5.15` baseline without a manual backport):

| PR / issue | what it is | relation |
|---|---|---|
| [#32209](https://github.com/sgl-project/sglang/pull/32209) | `Fix PD decode hang with DP attention and GLM-5.2 MTP` | **same defect, same strategy as our vote.** `draft_cuda_graph_dp_vote.diff` adopts its placement. |
| [#32527](https://github.com/sgl-project/sglang/issues/32527) | issue, same deadlock on 8× Blackwell / GLM-5.2-FP8 | independent report, 2026-07-27, same analysis. Proposes a third strategy: a dummy all-zeros seed so the guard's 4th term is false. |
| [#32762](https://github.com/sgl-project/sglang/pull/32762) | `[NPU] Fix DSA eager padding mismatch in PD MTP warm-up` | same bug class as `dsa_indexer_hip_dp_padded_rows`; that diff is written in its shape. |
| [#31683](https://github.com/sgl-project/sglang/pull/31683) | `[ROCm][MI35X] Enable GLM-5.2-MXFP4 MTP` | carries an independent, differently-placed implementation of the same indexer fix. |
| [#30839](https://github.com/sgl-project/sglang/pull/30839) / [#31083](https://github.com/sgl-project/sglang/pull/31083) | introduced the guard; cherry-picked to `release/v0.5.15` | **merged 2026-07-14** — so this deadlock is a *regression* in the baseline, not a legacy wart. |
| [#32722](https://github.com/sgl-project/sglang/pull/32722) | a test covering PD + DP-attention + MTP | proves **no CI covers this topology today**, which is why the family survives. |

Three corrections to earlier revisions of this file, all of which asserted upstream state
from notes instead of from `gh`:

1. ~~"this mechanism appears unreported"~~ — **false**, #32527 reported it independently
   two days before we fixed it.
2. ~~"#32209 all-gathers `can_cuda_graph`, and we measured that to be uniform, so its vote
   would not have fixed it"~~ — **false**. #32209 adds a *separate* `can_draft_cuda_graph`
   field; it is the same idea as ours, better placed. We now use its placement.
3. ~~"#32209 is CI red"~~ — not checked at the time; its review state is
   `REVIEW_REQUIRED`.

The `dsa_backend` fixes have **no upstream counterpart we have found** — per-diff greps of
#31683, #32175 and #32209 match neither site. That is weaker than it sounds: `gh search`
matches titles and bodies, not diff content, so an upstream PR could touch either site
without naming it.

One negative result worth recording, because it looks like an obvious convergence and is
not: #32209's *other* half reconciles the decode row counts by **trimming q/top-k** where
we **expand the page table**. Porting that half to this HIP/tilelang path fails
reproducibly at conc=32 (0/32 across seven runs, three node pairs, a rebuilt image).
Seventeen candidate causes were instrumented and eliminated; the root cause is **not**
identified. Details and the reproducer are in
`glm52.mxfp4.spur.mooncake.packup_20260731_exp3a_32209_patch2b_unresolved/`. Until that is
understood, `dsa_backend_dp_sync_and_page_table_rows.diff` keeps our expand-the-page-table
form.

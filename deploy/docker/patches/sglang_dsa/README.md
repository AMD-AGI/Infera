# sglang DSA patches

Patches that make **PD disaggregation + DP-attention** work for GLM-5.2, and,
with the fourth, **EAGLE MTP** on top. Without them the combination crashes on
the first batch or deadlocks the whole DP group under concurrency.

They apply to the sglang tree bundled in the ROCm engine images (an editable
checkout at `/sgl-workspace/sglang`). The two engine bases do **not** carry the
same arm of the set — see [Applying](#applying):

| image | base | arm |
|---|---|---|
| `Dockerfile.sglang` (gfx950 / MI355X) | `lmsysorg/sglang:v0.5.17-rocm720-mi35x` | 01 + dp_sync + page_table_rows + draft_dp_vote |
| `Dockerfile.sglang.gfx942` (gfx942 / MI325X) | `lmsysorg/sglang:v0.5.16-rocm720-mi30x` | **01 only**, plus a mandatory runtime flag |

| # | patch | fixes |
|---|---|---|
| 01 | `patch_dsa_indexer_hip_dp_padded_rows.py` | the HIP/aiter paged-MQA row count and `lengths` disagree — **in both directions** → `Expected lengths.size(0) == B` |
| 02a | `dsa_dp_sync.diff` | a host sync on a branch only *some* DP ranks take → collectives desync → deadlock. This file is upstream PR sglang#33973 verbatim |
| 02b | `dsa_page_table_rows.diff` | page table has one row per **request**, top-k one per **token** under MTP → `assert page_table.shape[0] == topk_indices.shape[0]` |
| 04 | `draft_cuda_graph_dp_vote.diff` | the draft graph/eager choice is made **per rank** from rank-dependent inputs and diverges on the PD decode leg → deadlock |

Patch 01 is a **script** and the rest are **context diffs**, and that is the
whole reason the two images can differ: the diffs are `--fuzz=0` against one
release, while 01 anchors on source text. Its `GLM52_P1V2` edit sites are
byte-identical on both releases; its later `GLM52_P1V3` anchor (the bare
`topk_transform` call) has been re-read against v0.5.15.post1 only — see the
anchor note in the script's header.

The script replaced a `dsa_indexer_hip_dp_padded_rows.diff` that carried the
same `GLM52_P1V2` edits; the diff is gone rather than kept as a second source of
truth. It is **no longer equivalent to that diff**: `GLM52_P1V3` was added after
the replacement, and only the script has it.

**Each patch's own header is the record**: what it fixes, why, how it was
established, the upstream issue / third-party PR / our own PR, how it differs
from our own upstream PR, and whether the IndexShare workaround substitutes for
it. Read the `.diff` before changing it. The one exception is `dsa_dp_sync.diff`,
kept byte-identical to sglang#33973 so it can be dropped by deletion the day that
merges; its record is the table above and the upstream status ledger.

Upstream linkage for these and every other patch in the repo is indexed in
[`deploy/docker/patch.upstream.status.md`](../../patch.upstream.status.md).

## Applying

Both engine images apply their arm at build time by default
(`APPLY_SGLANG_DSA_PATCHES=1`) via `deploy/docker/scripts/apply_sglang_dsa_patches.sh`,
which takes `DSA_PATCH_SET`:

| arm | used by | applies | verification |
|---|---|---|---|
| `full` (default) | `Dockerfile.sglang` | 01 + 02a + 02b | 4 bytecode markers |
| `indexer` | `Dockerfile.sglang.gfx942` | 01 | the `_p1v2_trim` bytecode marker |

Set `APPLY_SGLANG_DSA_PATCHES=0` for a stock engine to A/B against.

Prefer the script over patching by hand: it also verifies each patch reached the
**bytecode**, not just the source. A stale `__pycache__` entry silently reverts a
patch and has already invalidated a full experiment here — the source showed the
fix, the runtime did not have it.

By hand, against the pinned base:

```bash
cd /sgl-workspace/sglang
python3 patch_dsa_indexer_hip_dp_padded_rows.py
for d in dsa_dp_sync.diff dsa_page_table_rows.diff; do
  patch -p1 --fuzz=0 < "$d"
done
```

On the v0.5.16 base run only the first line; the two diffs will not apply.

`--fuzz=0` is deliberate: those two target one pinned commit, and a fuzzy apply
that "succeeds" against a different base is worse than a clean failure. The
mi35x base image tag is pinned for the same reason — bumping it fails the build
here rather than mis-applying silently. Note that `patch --dry-run` and `git
apply --check` **fuzz by default**, and a hand-written diff in this series once
silently dropped a hunk while still "passing". Patch 01 needs none of that care
for a different reason: an absent or ambiguous anchor makes it write nothing and
exit 1.

### gfx942 / v0.5.16 — the runtime half is not optional

That image carries patch 01 only, so **every leg must launch with**

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

This is not a tuning knob there. It is what stands in for 02b and 04 (see
[the alternative](#a-configuration-only-alternative-to-part-of-this-set), whose
substitution table applies unchanged) — leave IndexShare on and the decode leg
deadlocks on the first request, exactly as it does on gfx950 without patch 04.
The 1P1D bring-up behind this image was validated with the flag set.

Patch **02a** is neither carried nor substituted on that base. Its diff was not
re-cut for v0.5.16, and the failure it prevents (a host sync on a branch only
some DP ranks take) was not observed there — FP8, `dp8`, concurrency up to 128.
Absence of the symptom is not a fix; if a gfx942 run deadlocks with IndexShare
already off, 02a is the first thing to port.

### Patch 01: the row count diverges in BOTH directions (`GLM52_P1V3`)

Patch 01 originally guarded one direction only — `real < padded`, i.e. it
assumed DP padding always makes `q_fp8` **longer** than the real row count. On a
DP-attention **IDLE** rank under MTP draft-extend the inequality inverts.
Captured live with the patch's own `SGLANG_DEBUG_DSA_ROWS=1`:

    mode=IDLE q_fp8=(1,32,128) q_offset=2 ntnp=0 agree=False lengths=(2,) -> mqa_q=(1,32,128)

`q_offset` (= `sum(dsa_extend_len_cpu)`) is **2** while only **1** row is
materialized in `q_fp8`. `2 < 1` is false, so no trim runs; aiter sizes its
`logits` from `q_fp8.shape[0] = 1`, and `fast_topk_v2` gets 1 score row against
2 lengths entries — killing the scheduler rank (`deepseek_nextn.py` →
`dsa_backend.py` → `top_k.py`) and dropping the router to `active_workers: 1`.

**A trim cannot fix this direction** — there are *fewer* query rows than lengths
entries, so there is nothing to cut. Both sides reconcile down to
`min(real, padded)`:

| | `real < padded` | `real > padded` |
|---|---|---|
| cause | DP padding (the #32762 case) | IDLE rank, MTP draft-extend |
| `_p1v2_trim` | **True** — slice `q`/`weights` | False |
| `_p1v2_clip` | False | **True** — clip the lengths |
| restore | re-pad by `padded - rows` | nothing to restore |

`_p1v2_rows = min(_p1v2_real, _p1v2_padded)` is the single source of truth and
both flags derive from it, which is what keeps them from disagreeing. The
lengths side is clipped by passing `ke_offset=metadata.get_seqlens_expanded()
[:rows]` — an **existing** parameter of `DSAIndexerMetadata.topk_transform` whose
sole effect is to override `seq_lens_topk`, so the fix rides an intended
extension point rather than mutating shared (possibly graph-captured) metadata.

Two details worth keeping in view when reading the script:

* `_p1v2_clip` and `_p1v2_rows` are bound **before** the `is_aiter()` branch.
  `_p1v2_clip` is read unconditionally at the `topk_transform` call, so a
  branch-local binding is a `NameError` on any non-aiter backend. (aiter is
  unconditional on ROCm, so this never fired in practice — but a build-time patch
  should not depend on that.)
* the restore assert keys off `_p1v2_rows`, not `_p1v2_real`: under the clip case
  the two differ, and it is `_p1v2_rows` the kernel actually ran over.

**This half is what makes the patch survive an *agentic* workload.** The bug
needs MTP **and** DP-attention **and** an idle rank simultaneously. An 8-round
fixed-shape sweep ran MTP + DPA for 660 requests without hitting it, because
`--random-range-ratio 1.0` keeps every request in a round the same length —
batch shapes stay homogeneous and ranks rarely go idle mid-flight. An agentic
workload's breathing session population produces ragged batches constantly, and
the one-directional revision crashed the decode leg roughly 13 minutes into an
agentic benchmark. Over the last 400 indexer calls before one crash, 2 had the
fatal shape.

**Validated.** Reproduced twice under that workload (125 s and 766 s into two
independent runs on the same stock image). With the fix: **0** occurrences of
`Expected lengths.size` across a full ~4,000 s window, 0 scheduler exceptions, 0
retractions, `active_workers: 2` throughout — and independently over a second
~4,000 s window on a different cluster.

> **Provenance.** Second-hand for the reader: this half was first validated as a
> runtime script applied inside the decode container, *not* as part of patch 01.
> Folding it in here is a re-shaping of an already-measured fix, and the fold was
> checked for equivalence: the two forms differ only in the two pre-branch
> bindings noted above. The runtime script, both crash logs and both clean-window
> logs are in the internal reproduction kit — ask the patch author.

**Upstream status.** Not filed, and it is not in our own PR #33059 either, which
carries the `real < padded` half only. It should be — this repository already
ships the instrumentation that proves the bug (`SGLANG_DEBUG_DSA_ROWS`) and, before
this revision, a comment conceding the two bookkeeping sources had never been
measured to agree on the MTP draft-extend path. They do not; that is the bug.

Re-searched 2026-08-03 for the P1V3 half specifically ("DSA idle rank MTP draft
extend", "q_offset dsa_extend_len", "fast_topk_v2 lengths"): no issue, no PR.
But searching for the *function* rather than the symptom surfaces two OPEN PRs
that rewrite what patch 01 anchors on — neither fixes this defect:
[#32738](https://github.com/sgl-project/sglang/pull/32738) pads heads for
DeepGEMM at the same two aiter call sites (`q_fp8` → `q_fp8_padded`), and
[#31480](https://github.com/sgl-project/sglang/pull/31480) (updated the same day)
extracts the paged-MQA backend, restructuring the `is_aiter()` dispatch this
patch hangs off. Either landing in a future base drifts the anchors, which fails
the build rather than mis-applying — but re-cut patch 01 when bumping past them.

### Prerequisite

There is no longer one. GLM-5.2 MTP used to need a backport of sglang #30265 to
load its draft weights, carried here as `patches/sglang/`; v0.5.17 resolves the
MTP quark excludes inside `GlmMoeDsaForCausalLMNextN`, so that patch is retired
and the script no longer asserts it.

An earlier revision of this directory carried the same edit as a fourth diff.
Keeping both would have been worse than redundant — main's loop runs first, so
our context diff would then fail at `--fuzz=0` against an already-edited anchor.

## Validation

Everything below is the **gfx950 / v0.5.15.post1 / all-three arm**. The gfx942
arm is validated separately and much more narrowly — see
[gfx942](#gfx942--v0516--the-runtime-half-is-not-optional) and the caveats after
this section.

2 × 8×MI355X (gfx950), ROCm 7.2.0, sglang 0.5.15.post1, GLM-5.2-MXFP4, PD over
mooncake/mlx5 + dma-buf, `--dp-size 8 --enable-dp-attention --ep-size 8` + EAGLE
MTP(3,1,4), **draft CUDA graph enabled**.

Final run (2026-07-31) on the **v0.5.15.post1** base, when the set was 01 + 02 +
04 and the nextn backport was still a prerequisite. Built by `Dockerfile.sglang`
with patches applied **at build time** — nothing patched in the running container.
Not re-measured since the base moved to v0.5.17:

| Check | Target | Result |
|---|---|---|
| Build-time bytecode verification, both nodes | all markers | **8/8 + prereq + patch2a** |
| 4-prompt correctness probe | 4/4, `acc_len` > 1 | **4/4**, 2.00–3.43 |
| conc=32 × 512 tok | 32/32 | **32/32** |
| conc=128 × 512 tok, ×2 | 128/128 | **128/128**, **128/128** |
| `Traceback` / `KVTransferError`, either leg | 0 | **0 / 0** |
| DP ranks serving | 8 | **8**, every run, 0 retries |

Cumulative across the earlier arms with the fixes: **2540/2540**, versus **0/4**
with patch 04 reverted (same nodes, same image, deadlock at 120 s on request 1).
Draft-graph replay was measured at **97.1 %** (777/800 calls, identical on all 8
ranks) — that counter is the point, because forcing the draft path eager passes
every functional test while disabling the feature under test.

Raw measurements are held in internal reproduction kits, one per arm: the run
above, the draft-graph fix with its differential control, the IndexShare-off
workaround arm, and the failed #32209 port. Ask the patch author for access.

### v0.5.17, patch 04 re-cut (2026-08-11)

Only patch 04 was re-measured after the base moved; the rest of the set rests on
the run above. 2 × 8×MI355X, GLM-5.2-**FP8** 1P1D, both legs `tp8 --dp-size 8
--enable-dp-attention` + EAGLE MTP(3,1,4), IndexShare **on**, tilelang DSA
backends. Patch applied to the built image and verified in bytecode, all four of
its markers present:

| Check | Result |
|---|---|
| First routed request, patch reverted | **deadlock**, 120 s timeout |
| First routed request, patched | **2.3 s** |
| conc=64, ISL 1024 / OSL 512 | **256/256**, `acc_len` 2.31–4.00 of a ceiling of 4 |

Draft-graph replay share was **not** re-measured here — it needs an added probe.
What stands in for it is that the deadlock is gone at all: were the vote never
flipping, the group decision would stay permissive and the hang would remain.

### What this validation does NOT establish

* **Nothing about gfx942.** That arm was exercised on 2 × 8×MI325X, ROCm 7.2.0,
  sglang v0.5.16, GLM-5.2-**FP8** 1P1D, `tp8 dp8 --enable-dp-attention` + EAGLE
  MTP(3,1,4), IndexShare **off**, `attention-backend dsa` with the tilelang
  prefill/decode backends and `dsa-paged-mqa-logits-backend auto` — a different
  quantization, a different base, and a different backend selection. Patch 01 is
  carried there because the row mismatch it fixes shows up at concurrency > 1;
  the 02a exposure noted above is untested rather than ruled out.
* **The image built from this branch after the rebase was not re-run.** `main`
  has since added a `libionic` layer (`eb7da57`) that the measured image did not
  carry. It is orthogonal to these patches — RDMA ABI matching, not DSA — but it
  is a difference between what was measured and what this branch now builds.
* Draft-graph replay was not re-measured on the final image (it needs an added
  probe, i.e. a different image); 97.1 % is from the immediately preceding build
  of the same patch set. No differential control was re-run either — each patch's
  necessity is established in the kits above.
* Performance was not measured against the DPA-only baseline.
* All runs used `--disable-custom-all-reduce` (required on gfx942/gfx950 for
  EAGLE), so the custom all-reduce path is unexercised.
* Context 32768, short prompts, 512-token outputs, one hardware configuration.
  Long-context and 400k-context configs are untested.
* A seventh occurrence of the padded-vs-real-rows crash family was seen **once**
  in 500+ requests during an earlier session, on a machine already carrying the
  `dsa_indexer` fix. It did not recur in 2540 requests here, but at that rate
  absence is not evidence.

### One earlier caveat, withdrawn

> ~~"about 2 % of responses under concurrency are degenerate"~~ — **falsified;
> this was a test-harness error, not an engine bug.** The harness posted raw
> prompts to `/generate`, skipping GLM-5.2's chat template (so the model was
> doing base-LM completion), and forced `temperature=0` over the model's own
> recommended `1.0 / 0.95`. Degenerate repetition under greedy decoding is
> expected (Holtzman et al. 2019). With the template applied and the model's own
> sampling: **0/128 degenerate at conc=128 on both MXFP4 and FP8** — and the
> official FP8 build reproduced the "failure" under the wrong config exactly as
> MXFP4 did, so quantization was never implicated either.
>
> The related note about `eagle_utils.py:620` forcing `argmax` on HIP stands as a
> code observation, but it is confined to the spec-decode verify path and does
> not discard user sampling params for the request as a whole — that
> over-generalization is withdrawn.

## A configuration-only alternative to part of this set

On gfx950 this is an alternative. On **gfx942 it is the mechanism** — that image
ships neither 04 nor 02b, so the flag below is required, not optional.

Turning GLM-5.2's MTP **IndexShare** off avoids the same deadlock without 04 or
02b:

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

It works because IndexShare is the *source* of the rank divergence — remove the
seed and the guard term stops diverging, so no vote is needed.

| patch | substituted by IndexShare-off? |
|---|---|
| 01 `dsa_indexer_hip_dp_padded_rows` | **No** — independent bug, present regardless |
| 02a `dsa_dp_sync` | **No** — a host sync is invisible to the graph/eager decision either mechanism changes |
| 02b `dsa_page_table_rows` | **Yes**, in effect — the arm ran without it and passed |
| 04 `draft_cuda_graph_dp_vote` | **Yes** — this is what it targets |

Measured with 02b and 04 asserted **absent from the bytecode**: 4/4, 32/32 twice,
64/64, zero tracebacks, accept length 2.98–3.01 — no measurable cost. Two
conditions are easy to miss: MTP must be on the **prefill** leg too (otherwise the
seed never reaches decode and the setting is untested rather than tested), and
that arm was taken to conc=64, not 128.

**Why the patches remain the default where they can be.** The override is nearly
free only while IndexShare's consumer stays disabled under PD by
`should_use_dsa_fused_topk`. Upstream PR #31477 removes that limitation and
**merged 2026-08-05**; on a base that carries it the override starts costing
(~3 % TPOT, reported internally — second-hand, not measured by us). Neither of
our bases carries it yet: v0.5.17's `dsa/utils.py` still returns
`... and not pd_index_share_seed` under its TODO. So the override is still a good
answer on both bases today, and a dated one on whichever base comes next.

That dating is the standing cost of the gfx942 arm, which has no other option
until 02b and 04 are re-cut against v0.5.16 — 04's v0.5.17 cut is the template.

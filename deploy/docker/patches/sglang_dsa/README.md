# sglang DSA patches

Three patches that together make **PD disaggregation + DP-attention + EAGLE MTP**
work for GLM-5.2 on gfx950. Without them the combination crashes on the first
batch or deadlocks the whole DP group under concurrency.

They apply to the sglang tree bundled in the ROCm engine images
(`lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`, an editable checkout at
`/sgl-workspace/sglang`).

| # | patch | fixes |
|---|---|---|
| 01 | `dsa_indexer_hip_dp_padded_rows.diff` | HIP/aiter paged-MQA sizes its output from **DP-padded** rows while `lengths` is sized to **real** rows → `Expected lengths.size(0) == B` |
| 02 | `dsa_backend_dp_sync_and_page_table_rows.diff` | (a) a host sync on a branch only *some* DP ranks take → collectives desync → deadlock; (b) page table has one row per **request**, top-k one per **token** under MTP → assert |
| 04 | `draft_cuda_graph_dp_vote.diff` | the draft graph/eager choice is made **per rank** from rank-dependent inputs and diverges on the PD decode leg → deadlock |

**Each patch's own header is the record**: what it fixes, why, how it was
established, the upstream issue / third-party PR / our own PR, how it differs
from our own upstream PR, and whether the IndexShare workaround substitutes for
it. Read the `.diff` before changing it.

Upstream linkage for these and every other patch in the repo is indexed in
[`deploy/docker/patch.upstream.status.md`](../../patch.upstream.status.md).

## Applying

`Dockerfile.sglang` applies them at build time by default
(`APPLY_SGLANG_DSA_PATCHES=1`) via `deploy/docker/scripts/apply_sglang_dsa_patches.sh`.
Set `APPLY_SGLANG_DSA_PATCHES=0` for a stock engine to A/B against.

Prefer the script over patching by hand: it also verifies each patch reached the
**bytecode**, not just the source. A stale `__pycache__` entry silently reverts a
patch and has already invalidated a full experiment here — the source showed the
fix, the runtime did not have it.

By hand, against sglang `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (v0.5.15.post1):

```bash
cd /sgl-workspace/sglang
for d in dsa_indexer_hip_dp_padded_rows.diff \
         dsa_backend_dp_sync_and_page_table_rows.diff \
         draft_cuda_graph_dp_vote.diff; do
  patch -p1 --fuzz=0 < "$d"
done
```

`--fuzz=0` is deliberate: these target one pinned commit, and a fuzzy apply that
"succeeds" against a different base is worse than a clean failure. The base image
tag is pinned for the same reason — bumping it fails the build here rather than
mis-applying silently. Note that `patch --dry-run` and `git apply --check` **fuzz
by default**, and a hand-written diff in this series once silently dropped a hunk
while still "passing".

### Prerequisite

`deploy/docker/patches/sglang/patch_glm52_nextn_quark_exclude.py` (a backport of
sglang #30265) must run **first** — GLM-5.2 MTP cannot load its draft weights
without it, dying with a `3072 vs 6144` shape mismatch. `Dockerfile.sglang` runs
that loop before this one, and `apply_sglang_dsa_patches.sh` **asserts** the edit
rather than assuming it: that script is idempotent, so a missing anchor would
print `skipped`, exit 0, and surface only at runtime.

An earlier revision of this directory carried the same edit as a fourth diff.
Keeping both would have been worse than redundant — main's loop runs first, so
our context diff would then fail at `--fuzz=0` against an already-edited anchor.

## Validation

2 × 8×MI355X (gfx950), ROCm 7.2.0, sglang 0.5.15.post1, GLM-5.2-MXFP4, PD over
mooncake/mlx5 + dma-buf, `--dp-size 8 --enable-dp-attention --ep-size 8` + EAGLE
MTP(3,1,4), **draft CUDA graph enabled**.

Final run (2026-07-31), from an image built by `Dockerfile.sglang` on this branch
with patches applied **at build time** — nothing patched in the running container:

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

Measurements live in the `yihou.dev.glm5.2.mxfp4.experiment` branch:
`glm52.mxfp4.spur.mooncake.packup_20260731_main_converged` (the run above),
`..._20260729_bug2b_draft_graph` (the draft-graph fix and its differential
control), `..._20260730_exp2_indexshare_off` (the workaround arm),
`..._20260731_exp3a_32209_patch2b_unresolved` (the failed #32209 port).

### What this validation does NOT establish

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

Turning GLM-5.2's MTP **IndexShare** off avoids the same deadlock without patch
04 or the page-table half of patch 02:

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

It works because IndexShare is the *source* of the rank divergence — remove the
seed and the guard term stops diverging, so no vote is needed.

| patch | substituted by IndexShare-off? |
|---|---|
| 01 `dsa_indexer_hip_dp_padded_rows` | **No** — independent bug, present regardless |
| 02a `dsa_backend` DP host-sync | **No** — a host sync is invisible to the graph/eager decision either mechanism changes |
| 02b `dsa_backend` page-table rows | **Yes**, in effect — the arm ran without it and passed |
| 04 `draft_cuda_graph_dp_vote` | **Yes** — this is what it targets |
| nextn `eh_proj` (the prerequisite) | **No** — weight-load bug, unrelated |

Measured with 02b and 04 asserted **absent from the bytecode**: 4/4, 32/32 twice,
64/64, zero tracebacks, accept length 2.98–3.01 — no measurable cost. Two
conditions are easy to miss: MTP must be on the **prefill** leg too (otherwise the
seed never reaches decode and the setting is untested rather than tested), and
that arm was taken to conc=64, not 128.

**Why the patches remain the default.** The override is nearly free only because
IndexShare's consumer is currently disabled under PD by
`should_use_dsa_fused_topk`. Upstream PR #31477 exists to remove that limitation;
once it lands the override starts costing (~3 % TPOT, reported by AMD's llying —
second-hand, not measured by us). Checked with `gh` on 2026-08-01: #31477 is
**open**, `REVIEW_REQUIRED`, unmerged. A good answer today if IndexShare is not
wanted, a dated one if it is.

# Final deliverable — GLM-5.2-MXFP4 PD + DP-attention + EAGLE MTP on gfx950

**Ran:** 2026-07-31, AMD spur cluster `crsuse2-m2m`, 2 × MI355X nodes.
**Author:** yihou
**Status:** **PASS** — every acceptance criterion met, from an image built by a
committed Dockerfile with **no in-container patching**.

## What makes this run different from the eight kits before it

Every previous result on this bug was obtained by patching a *running* container
by hand. This one is built from `deploy/docker/Dockerfile.sglang.dmabuf` on the
PR branch, which applies the patch set at **image build time** and fails the
build if any patch does not reach the bytecode. The container is started from
that image and nothing is patched afterwards.

It is also the first run of the **final** patch set. Two of the four patches
were reshaped to follow upstream, and each reshape was validated separately —
but never together:

| patch | shape | validated in | with |
|---|---|---|---|
| 1 `dsa_indexer` | upstream **#32762** | exp1 (2026-07-30) | **our** patch 4 |
| 4 `draft_cuda_graph_dp_vote` | upstream **#32209** | exp3b (2026-07-30) | **our** patch 1 |

So "1v2 + 4v2 together" had never been run until now. That is the gap this kit
closes.

## Result

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| Image builds from the committed Dockerfile | — | yes, patches verified in bytecode at build time | ✅ |
| 4-prompt correctness probe | 4/4 | **4/4**, `acc_len` 2.18–3.00 | ✅ |
| conc=32 × 512 tok | 32/32 | **32/32** | ✅ |
| conc=128 × 512 tok | 128/128 | **128/128** | ✅ |
| conc=128 × 512, repeat | — | **128/128** | ✅ |
| Traceback / `KVTransferError` in either leg | 0 | **0 / 0** | ✅ |
| **Draft CUDA graph provably in use** | > 0 | **92.0 %**, identical on all 8 ranks | ✅ |

Raw per-request data, from `results/`:

| run | ok | full 512 tok | acc_len mean | min | max | DP ranks | retries |
|---|---|---|---|---|---|---|---|
| `stress_c32.jsonl` | **32/32** | 32 | 2.81 | 2.26 | 3.39 | 8 | 0 |
| `stress_c128.jsonl` | **128/128** | 122 | 2.82 | 2.13 | 3.94 | 8 | 0 |
| `stress_c128_r2.jsonl` | **128/128** | 118 | 2.82 | 1.52 | 3.94 | 8 | 0 |
| `stress_c128_guse.jsonl` | **128/128** | 121 | 2.81 | 1.96 | 3.94 | 8 | 0 |

All eight DP ranks served traffic in every run, so a pass is not an artifact of
one rank doing all the work.

> "full 512 tok" below the request count is **not** a failure. `max_new_tokens=512`
> is a cap and greedy decoding hits EOS earlier on some prompts. Every request
> returned HTTP 200 with a complete response; the `ok` column is the criterion.

### The measurement that matters

Forcing the draft path eager passes every functional test above while disabling
the very feature under test — so a green stress result **cannot on its own**
distinguish a fix from that workaround. The only thing that can is counting
replays:

```
GLM52_GUSE periodic rank=0 calls=200 graph=184 (92.0%) refused_bs=0 refused_dp=0 refused_draftvote=16
GLM52_GUSE periodic rank=7 calls=200 graph=184 (92.0%) refused_bs=0 refused_dp=0 refused_draftvote=16
```

Identical on all 8 ranks — which is the fix working, since a *uniform* decision
is exactly what the patch exists to produce. The 8 % refusals are the group
correctly going eager together when a rank's DSA top-k seed has not arrived:

```
GLM52_GUSE_WHY rank=0 total=200 future_seed_missing=8 (4.0%) future_seed_ok=192 (96.0%) seed_none=0 (0.0%)
```

`seed_none=0` confirms the predicate is reading `future_dsa_topk_indices_available`
and not the stale direct field — an earlier revision got this wrong and refused
the graph 100 % of the time while still passing every functional test.

**Caveat on this number, stated plainly.** The graph-usage counters come from an
*added probe* (`instr_graph_usage.py`), so the measured run is the shipped image
**plus** that probe. The 128/128 in `stress_c128_guse.jsonl` was collected in
that configuration; the other three runs are the unmodified image. The probe only
increments counters and logs, but it is a difference and it is not hidden here.

## The patch set

| # | file | what it fixes | shape |
|---|---|---|---|
| 1 | `01_dsa_indexer_hip_dp_padded_rows.diff` | HIP/aiter paged-MQA sizes its output from **DP-padded** rows while `lengths` is sized to **real** rows → `Expected lengths.size(0) == B` | upstream **#32762** (NPU, same bug class): one boolean gates both trim and restore, and the post-kernel row count is **asserted** before padding is restored |
| 2 | `02_dsa_backend_dp_sync_and_page_table_rows.diff` | (a) `seq_lens.max().item()` is a host sync on a branch only *some* DP ranks take → collectives desynchronize; (b) page table has one row per **request**, top-k one per **token** under MTP → assert | ours; no upstream counterpart found |
| 3 | `03_deepseek_nextn_glm52_mtp_bf16.diff` | GLM-5.2 MXFP4 exports the nextn layer bf16 while the model is quantized → `3072 vs 6144` at load | ours; #32175 carries the same fix upstream |
| 4 | `04_draft_cuda_graph_dp_vote.diff` | the draft graph/eager choice is made **per rank** from rank-dependent inputs and diverges on the PD decode leg → deadlock | upstream **#32209**: the vote rides the MLP-sync all-gather the scheduler already performs — **zero** extra collectives |

All four apply at `--fuzz=0` against sglang
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (v0.5.15.post1). Verified by applying
into a pristine tree in this run, not by `--dry-run` (which fuzzes by default).

## What this kit does NOT establish

- **No differential control was run here.** The evidence that each patch is
  *necessary* lives in the earlier kits — patch 4's same-node revert control
  (0/4, deadlock) is in `..._20260729_bug2b_draft_graph`. This run demonstrates
  the assembled set works from a built image; it does not re-derive necessity.
- **Patch 2a still has no differential control**, in this kit or any other. It
  has never been run with only 2a reverted. This is the oldest open gap in the
  series and is stated here rather than left implied.
- **Performance was not measured** against a DPA-only baseline. Patch 4 v2 adds
  no collective (that was the point of adopting #32209's placement), but no
  throughput comparison was run.
- **One configuration only**: context 32768, short prompts, 512-token outputs,
  `--disable-custom-all-reduce`, MTP on the decode leg only. Long-context and
  prefill-MTP configurations are untested here.
- The draft-graph number was measured with a probe present — see the caveat
  above.

## Related kits

- `..._20260729_bug2b_draft_graph` — root cause of the deadlock, patch 4 **v1**,
  and the differential control (0/4 with the fix reverted)
- `..._20260730_exp1_patch1_v2` — patch 1 reshaped to #32762, validated
- `..._20260730_exp3b_patch4_32209` — patch 4 reshaped to #32209, validated at
  97.1 % graph usage
- `..._20260731_exp3a_32209_patch2b_unresolved` — **negative result**: porting
  #32209's *other* half (trim q/top-k instead of expanding the page table) fails
  0/32 across seven runs; 17 causes eliminated, root cause open. This is why
  patch 2b keeps our form.

## Folder map

- `REPRODUCE.md` — cold-start reproduction, from image build to stress
- `environment.md` — hardware, image, commit, node pair
- `notes.md` — what went wrong during the run and how it was diagnosed
- `patches/` — the four final diffs + the Dockerfile that bakes them in
- `scripts/` — build, boot, router, probe, stress, and the build-time apply script
- `results/` — raw per-request jsonl, 4 runs
- `logs/` — both legs, router, and the image build log

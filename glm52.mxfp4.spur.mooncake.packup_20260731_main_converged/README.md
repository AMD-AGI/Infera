# Converged validation — GLM-5.2-MXFP4 PD + DP-attention + EAGLE MTP, rebased onto main

**Ran:** 2026-07-31, 10:29–10:53 UTC. AMD spur cluster `crsuse2-m2m`, 2 × MI355X.
**Author:** yihou
**Status:** **PASS** — every criterion met, from an image built by `Dockerfile.sglang`
on the PR branch after it was rebased onto `origin/main`, with **no in-container
patching**.

## Why this run exists

The previous kit (`..._20260731_final_deliverable`) validated the patch set on a branch
based on `cf85272`. `main` has since moved 29 commits ahead and **absorbed three of the
things that branch carried**. Rebasing applied cleanly — which was misleading: the branch
only *added* files, so git saw no conflict with changes that overlap semantically.

Dropping those three moved one fix onto a different code path, and "the two make the
identical edit" was a static argument, not a measurement. This run turns it into one.

| dropped from our branch | now supplied by main | overlap |
|---|---|---|
| `deepseek_nextn_glm52_mtp_bf16.diff` | `patches/sglang/patch_glm52_nextn_quark_exclude.py` (`0d8d0ff`) | same file, same line, same resulting value |
| `build_mooncake_dmabuf.sh` | `scripts/build_mooncake_sglang.sh` | main's is a strict superset |
| `Dockerfile.sglang.dmabuf` | `Dockerfile.sglang` | main now compiles dma-buf in and switches at runtime |

Keeping the nextn diff would have been **actively broken**, not merely redundant: main's
patch loop runs first, so our context diff would then fail at `--fuzz=0` against an
already-edited anchor. The DSA layer therefore moved into `Dockerfile.sglang`, defaulting
**on** (`APPLY_SGLANG_DSA_PATCHES=1`), placed after that loop so its prerequisite holds.

## Result

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| Build-time bytecode verification, both nodes | all markers | **8/8 markers + prereq + patch2a** | ✅ |
| 4-prompt correctness probe | 4/4, `acc_len` > 1 | **4/4**, 2.00–3.43 | ✅ |
| conc=32 × 512 tok | 32/32 | **32/32** | ✅ |
| conc=128 × 512 tok | 128/128 | **128/128** | ✅ |
| conc=128 × 512, repeat | — | **128/128** | ✅ |
| `Traceback` / `KVTransferError`, either leg | 0 | **0 / 0** | ✅ |
| nextn fix provenance is main's, not ours | — | **confirmed** (see below) | ✅ |

Raw per-request data, from `results/`:

| run | ok | full 512 tok | acc mean | min | max | DP ranks | retries |
|---|---|---|---|---|---|---|---|
| `stress_c32.jsonl` | **32/32** | 31 | 2.93 | 2.32 | 3.94 | 8 | 0 |
| `stress_c128.jsonl` | **128/128** | 125 | 2.81 | 1.95 | 3.82 | 8 | 0 |
| `stress_c128_r2.jsonl` | **128/128** | 113 | 2.79 | 1.60 | 3.91 | 8 | 0 |

> "full 512 tok" below the request count is **not** a failure. `max_new_tokens=512` is a
> cap and decoding hits EOS earlier on some prompts. Every request returned HTTP 200; the
> `ok` column is the criterion. `acc > 1` is what proves MTP is genuinely active — a
> 128/128 with `acc == 1` would be a failure dressed as a pass.

### The dependency swap was measured, not assumed

The edited line in the running image carries **no trailing comment**, which our removed
diff would have added:

```
363:            ckpt_prefix = f"model.layers.{config.num_hidden_layers}.eh_proj"
```

and the build log shows main's script running *first*, our assert reading it *after*:

```
[glm52-nextn] patched /sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py
  PREREQ nextn eh_proj      -> src=1 (want 1)
```

That ordering is the whole point of the `PREREQ` assert: main's script is idempotent and
would otherwise "skip" silently, and the failure would surface only at runtime, as
GLM-5.2 dying at draft weight-load with `3072 vs 6144`.

## The patch set (three diffs, down from four)

| # | file | what it fixes | shape |
|---|---|---|---|
| 1 | `dsa_indexer_hip_dp_padded_rows.diff` | HIP/aiter paged-MQA sizes its output from **DP-padded** rows while `lengths` is sized to **real** rows → `Expected lengths.size(0) == B` | upstream **#32762** (NPU, same bug class) |
| 2 | `dsa_backend_dp_sync_and_page_table_rows.diff` | (a) `seq_lens.max().item()` is a host sync on a branch only *some* DP ranks take → collectives desynchronize; (b) page table has one row per **request**, top-k one per **token** under MTP → assert | ours; no upstream counterpart found |
| 4 | `draft_cuda_graph_dp_vote.diff` | the draft graph/eager choice is made **per rank** from rank-dependent inputs and diverges on the PD decode leg → deadlock | upstream **#32209**: the vote rides an all-gather the scheduler already performs — **zero** extra collectives |

Patch 3 (nextn `eh_proj`) is no longer here — see above. All three apply at `--fuzz=0`
against sglang `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`.

### There is a configuration-only alternative to patches 2 and 4

Turning GLM-5.2's MTP **IndexShare** off avoids the same deadlock without either patch:

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

Measured on this cluster (`..._20260730_exp2_indexshare_off`): 4/4, 32/32 ×2, 64/64, with
patches 2 and 4 asserted **absent from the bytecode**. It works because IndexShare is the
*source* of the divergence — see `notes.md` §2 for which patches it substitutes for, which
it does **not**, and why it has an expiry date (upstream #31477).

## What this kit does NOT establish

- **The draft-graph replay count was not re-measured.** That needs an added probe, i.e. a
  different image. Measured at **92.0 %, identical on all 8 ranks**, on the immediately
  preceding build of the same patch set (`..._20260731_final_deliverable`). A green stress
  result alone cannot distinguish a fix from forcing the path eager.
- **No differential control was run here.** Each patch's necessity lives in the earlier
  kits — patch 4's same-node revert control (0/4, deadlock) is in
  `..._20260729_bug2b_draft_graph`; patch 2's is in `..._20260728/dpa_mtp_fix/`, where the
  divergence was caught in a py-spy dump and its second half was forced by experiment
  (with only the `max_seqlen_k` change, the hang persists). See `notes.md` §4.
- **No performance comparison** against a DPA-only baseline.
- **One configuration only**: context 32768, short prompts, 512-token outputs,
  `--disable-custom-all-reduce`, MTP on the decode leg only.

## Related kits

- `..._20260731_final_deliverable` — same patch set, pre-rebase branch, **with** the
  graph-usage measurement (92.0 %)
- `..._20260729_bug2b_draft_graph` — root cause of the deadlock and the differential control
- `..._20260730_exp2_indexshare_off` — the IndexShare-off alternative, measured
- `..._20260731_exp3a_32209_patch2b_unresolved` — negative result: porting #32209's *other*
  half fails 0/32 across seven runs, root cause open

## Folder map

- `REPRODUCE.md` — cold-start reproduction, from image build to stress
- `environment.md` — hardware, image, commits, node pair
- `notes.md` — the convergence reasoning, the IndexShare alternative, traps hit
- `patches/` — the three DSA diffs (each headed with its IndexShare-alternative status), the
  nextn prerequisite, the mooncake C++ patches, the Dockerfile, the patch-set README
- `scripts/` — DSA apply, mooncake rebuild, boot, router, probe, stress, env capture
- `results/` — raw per-request jsonl, 3 runs
- `logs/` — both legs, router, and the image build log

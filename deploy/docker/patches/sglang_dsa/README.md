# sglang DSA patches

Patches against the sglang source tree bundled in the ROCm engine images
(`lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` and derivatives, where sglang is an
editable checkout at `/sgl-workspace/sglang`).

> **These diffs are NOT applied at image build time.** `Dockerfile.sglang` no longer
> carries a patch loop (the previous `deploy/docker/patches/sglang/*.py` mechanism was
> removed along with it), so a stock build of the image does **not** contain any of them.
> Apply them into a running container — by `patch -p1 --fuzz=0`, or by bind-mounting the
> patched files over the container's copies — which is how every result quoted below was
> obtained. If you want them baked in, a build step has to be added; that is deliberate
> for now, since they target a specific pinned sglang commit and would silently no-op or
> conflict on a different base.

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

## `deepseek_nextn_glm52_mtp_bf16.diff`

One line in `python/sglang/srt/models/deepseek_nextn.py`. **Required for GLM-5.2 MTP**,
whose nextn layer is bf16 while the rest of the model is quantized — without it the
weight load dies with a `3072 vs 6144` shape mismatch.

## `eagle_worker_v2_uniform_draft_graph.diff`

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

The fix votes the *local need for eager* across the DP group (one 1-element **gloo**
all-reduce on `cpu_group` — no GPU sync, cannot serialize the compute stream) and has
every rank act on the group's answer. An idle rank contributes `False`, so it can never
drag the group to eager.

Two details that would otherwise make the patch silently inert:

* it must use `get_tp_group()`, **not** `get_attention_tp_group()` — under DPA8 on tp8 the
  attention TP group is `attn_tp_size = tp/dp = 1` rank wide, so voting on it is a no-op;
* a collective is safe at this line because instrumentation showed all 8 ranks entering
  `draft()` an equal number of times every iteration, idle ones included — the vote sits
  on a branch every rank takes, which is precisely the property the buggy code lacked.

An earlier attempt that removed only the `not is_idle()` term does **not** work, and the
measurements say why: it fixes the busy-rank iteration and breaks the idle-rank one, which
matches the observed symptom of the failure moving earlier, into warmup.

### Verified

2× 8×MI355X, GLM-5.2-MXFP4, PD over mooncake/mlx5, DPA8 + EAGLE MTP(3,1,4), **draft graph
enabled**:

| Check | Result |
|---|---|
| PD warmup, 4×24-token, 1×512-token | pass |
| conc 1/2/4/8/16/64/128/256 × 512 tok | **1384/1384**, 0 failures, 0 KVTransferError |
| Single-node mix regression (shared code path) | **132/132** |
| Durability, 8 × conc=128 × 512 back to back | **1024/1024**, accept length flat at ~3.0 |
| **Draft graph actually used** | **98.4%** of iterations |
| Same-node control, only this diff reverted | **0/4 — deadlock, 120 s timeout on request 1** |

Cumulative **2540/2540** with the fix, **0/4** without.

The mechanism was verified, not just the outcome. Over 2992 iterations with all 8 ranks
logging: the *local* decision still diverges 38 times (the defect is latent and real, not
masked by timing) while the *acted-on* decision diverges **0** times, and the vote changed
some rank's mind 190 times — each one an averted deadlock. The 98.4% graph usage is what
distinguishes this from simply disabling the draft graph.

## Applying

All four apply with exact context, no fuzz, against sglang commit
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (v0.5.15.post1):

```bash
cd /sgl-workspace/sglang
for d in dsa_indexer_hip_dp_padded_rows.diff \
         dsa_backend_dp_sync_and_page_table_rows.diff \
         deepseek_nextn_glm52_mtp_bf16.diff \
         eagle_worker_v2_uniform_draft_graph.diff; do
  patch -p1 --fuzz=0 < "$d"
done
```

Verified by applying for real into a scratch tree and byte-compiling the result — note
that `patch --dry-run` and `git apply --check` **fuzz by default**, and a hand-written
diff in this series once silently dropped a hunk while still "passing".

## PD + MTP status (updated 2026-07-29 evening)

With **all four** diffs applied, PD disaggregation with DP-attention + EAGLE MTP and the
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
   **written and verified**, see `eagle_worker_v2_uniform_draft_graph.diff` above. The
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

The `eagle_worker_v2` fix is **not** ROCm-specific in principle. The guard it repairs is
platform-neutral, and its terms are rank-dependent on any backend running DP-attention
with PD disaggregation; CUDA is protected today only because the draft-extend graph is
captured there, which keeps the ranks on the same path. Upstream has related open work
that blames `can_cuda_graph` (#32209 all-gathers that decision, CI red; #32527 same
topology; #32722 shows **no CI covers PD+DPA+MTP**), but none of them addresses this
guard — and we measured `can_cuda_graph` to be *uniform* across ranks here, so #32209's
vote would not have fixed it. This mechanism appears unreported.

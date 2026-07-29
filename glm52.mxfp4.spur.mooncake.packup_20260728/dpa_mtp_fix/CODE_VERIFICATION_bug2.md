# Bug 2 — deep code verification (task 1 of the deepsearch)

Read from a pristine container (`infera.yihou.sglang.1.0`, base
`lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`), node `crsuse2-m2m-029`, 2026-07-28 ~13:00 UTC.
Only the Bug 1 patch was applied; the Bug 2 probes/fix were **not** installed, so every line
number below is stock upstream.

> Note on provenance: PyPI's `sglang` tops out at **0.5.10**, so `0.5.15.post1` is only
> published as the LMSYS ROCm docker image, not as a wheel. Line numbers here are from that
> image and may not match a github checkout of any public tag.

---

## 1. The `needs_cpu_seq_lens` contract — CONFIRMED, and the comment explains the intent

Both DSA classes opt out (`layers/attention/dsa_backend.py`):

```python
333: class DeepseekSparseAttnBackend(
334:     DeepseekSparseAttnBackendMTPPrecomputeMixin, AttentionBackend
335: ):
336:     # Decode/verify/draft graph replay rebuilds metadata from static buffers
337:     # (page-table width) and never reads seq_lens_cpu / seq_lens_sum; opt out of
338:     # the D2H sync. The eager fallback derives lengths from GPU seq_lens.
339:     needs_cpu_seq_lens: bool = False
```

```python
2899: class DeepseekSparseAttnMultiStepBackend:
2901:     # Per-step draft decode replays from precomputed GPU metadata; opt out so
2902:     # decide_needs_cpu_seq_lens' OR over the backends stays False.
2903:     needs_cpu_seq_lens: bool = False
```

**The design assumes CUDA-graph replay.** The comment says so outright: replay "rebuilds
metadata from static buffers (page-table width) and never reads seq_lens_cpu", and the
`.item()` is explicitly the *"eager fallback"* — understood to be the rare path.

That assumption is exactly what breaks on our stack: **on HIP the draft-extend graph is
never captured** (`eagle_worker_v2.py:441-482`; `supports_cuda_draft_extend_graph` requires
`_is_cuda or _is_musa`, and `supports_hip_aiter_draft_extend_graph` requires the draft
backend to be an `AiterMultiStepDraftBackend`, which DSA/tilelang is not). So the
"fallback" is the **only** path, on every step.

The flag propagates as expected — `managers/overlap_utils.py`:

```python
 45:     return any(getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None)
...
281:     def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
300:         if not self.needs_cpu_seq_lens:
301:             # GPU gather above is kept (SB.seq_lens must advance each verify);
302:             # skip the .cpu() D2H. Downstream takes the GPU-only path.
303:             batch.seq_lens_cpu = None
304:             batch.seq_lens_sum = None
305:             return
```
Called from `managers/scheduler.py:3232`. So `seq_lens_cpu = None` is deliberate, and
`gpu_only` in `base_spec_worker.py:112` follows from it.

## 2. NEW: the hang is not just `max_seqlen_k` — `DRAFT_EXTEND_V2` re-syncs unconditionally

This changes the fix. `init_forward_metadata` spans **lines 726-1017** and contains
**eight** D2H sync sites:

| line | expression | when |
|---|---|---|
| 740 | `seq_lens_cpu.max().item()` | host-mirror arm |
| **746** | **`seq_lens.max().item()`** | **`else` arm — the py-spy stall site** |
| **809** | **`extend_prefix_lens.cpu().tolist()`** | **`is_draft_extend_v2()`, if `extend_prefix_lens_cpu is None`** |
| **812** | **`seq_lens_cpu = seq_lens.cpu()`** | **`is_draft_extend_v2()`, if `seq_lens_cpu is None`** |
| 872 | `seq_lens_cpu.tolist()` | ragged topk transform |
| 890 | `indexer_seq_lens_cpu.max().item()` | CP round-robin split |
| 921 | `indexer_seq_lens_cpu.tolist()` | ragged path |
| 935 | `page_table_1_flattened.max().item()` | prefix-sharing assert |

The `DRAFT_EXTEND_V2` branch (line 805 onward) *repairs the very mirror the backend asked
not to be published*:

```python
805:  elif forward_batch.forward_mode.is_draft_extend_v2():
806:      if forward_batch.extend_prefix_lens_cpu is None:
808:          forward_batch.extend_prefix_lens_cpu = (
809:              forward_batch.extend_prefix_lens.cpu().tolist()     # D2H
810:          )
811:      if forward_batch.seq_lens_cpu is None:
812:          forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu()   # D2H
813:          forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
```

…and then, ~26 lines later, carries this comment:

```python
836:      # DRAFT_EXTEND_V2: ... Use scalar to avoid GPU sync.
```

**Consequence:** patching only line 746 (fix option (a) as originally scoped) would **not**
fix the hang — lines 809/812 still force a blocking D2H on exactly the ranks that have
work, while idle ranks skip the whole `elif` and proceed to the next collective. Any real
fix must address the whole `DRAFT_EXTEND_V2` path, not one expression.

## 3. The in-codebase precedent for a sync-free bound EXISTS

`trtllm_mla_backend.py` — another `needs_cpu_seq_lens = False` MLA backend — uses a static
host-known width on its graph path:

```python
363:         # Capture with full width so future longer sequences are safe during replay.
364:         max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
367:         metadata.max_seq_len_k = self.max_context_len
```

So "over-size the page table to a host-known bound" is a **sanctioned pattern** in this
codebase, not an invention.

And the DSA backend already has the same value to hand:

```python
364:         self.max_context_len = model_runner.model_config.context_len
```

**Caveat measured, not assumed:** in DSA the sliced `page_table` is subsequently
`repeat_interleave`d by `speculative_num_draft_tokens` (lines 802-803, 838-840) and by
`extend_seq_lens` (line 849), so over-sizing the width is multiplied. With
`CTX=32768` and `num_draft_tokens=4` the widened table is real memory traffic, not free.
Over-sizing is *correct* but may be *costly*; it must be benchmarked, not assumed cheap.

## 4. `trtllm_mla` has the identical latent pattern

```python
518:             if getattr(forward_batch, "seq_lens_cpu", None) is not None:
519:                 max_seq = forward_batch.seq_lens_cpu.max().item()
520:             else:
521:                 max_seq = forward_batch.seq_lens.max().item()
```

Byte-for-byte the same rank-divergent shape as DSA's 740/746. So this is **a shared idiom
across sglang attention backends**, not a DSA-specific slip. It is latent everywhere the
combination "backend opts out of the CPU mirror" + "no graph capture" + "DP-attention with
partial rank occupancy" can co-occur. Worth reporting upstream as a class of bug.

## 5. A second HIP-only sync on the same path

`managers/overlap_utils.py`, inside `resolve_seq_lens_cpu`:

```python
292:         if _is_hip:
293:             # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
294:             self.publish_ready.synchronize()
295:         else:
296:             self.publish_ready.wait()
```

On HIP this is a **full device synchronize** where CUDA does a non-blocking event wait —
on the very code path that publishes (or withholds) `seq_lens_cpu`. Self-described as a
"temporary workaround". Not proven to contribute to this deadlock, but it is a second
HIP-specific blocking point in the same flow and should be considered when reasoning about
ordering.

---

## What this means for the fix

The originally-planned option (a) ("just avoid the `.item()` at 746") is **insufficient** —
finding #2 kills it. Revised options:

- **(a′) Make the whole `DRAFT_EXTEND_V2` metadata path sync-free.** Use
  `self.max_context_len` for the page-table width, and derive `extend_prefix_lens_cpu` /
  `seq_lens_cpu` from host-side values the spec worker already has
  (`extend_seq_lens_cpu = [num_draft_tokens]*bs` is set at `base_spec_worker.py:155`).
  Biggest change; removes the hazard rather than synchronising around it; needs a
  benchmark because of the repeat_interleave widening.
- **(b) Publish the host mirror for this path** — make `gpu_only` False for
  `DRAFT_EXTEND_V2`, i.e. accept the H2D that `needs_cpu_seq_lens=False` was avoiding.
  Smallest, most obviously-correct change; directly restores the invariant the backend's
  own code at 811-813 is trying to reconstruct anyway (it already pays that D2H!). Since
  the backend *already* forces `seq_lens.cpu()` on this path, publishing the mirror
  up-front should cost no more than the status quo — and possibly less, because the
  scheduler can overlap it on its private D2H stream (`fwd_prepare_d2h_stream`).
- **(c) Uniform branch for all ranks** — still the fallback if neither lands.

**(b) is now the recommended first attempt**, and for a reason stronger than "it's small":
the `DRAFT_EXTEND_V2` branch already performs the exact D2H that opting out was meant to
save, so the opt-out is buying nothing on this path while costing correctness.

---

# CORRECTION (upstream research, same day) — (b) was already rejected upstream

The recommendation immediately above is **wrong as stated**, and the upstream history says
why. Recording it rather than deleting it, because the reasoning error is instructive.

**Lines 809/812 are not a pre-existing accident — they ARE an upstream fix**, added by
merged PR **#29798** (`fix: avoid DSA indexer CPU seq lens fallback`, +8/−0 on
`dsa_backend.py`, +1/−2 on `dsa_indexer.py`, nothing else). The diff is exactly the block I
flagged. Its purpose was to repair `AssertionError: All of them must not be None` when a
decode batch exceeds `--cuda-graph-max-bs` and runs eager.

And that PR **explicitly considered and rejected option (b)**:

> *"An earlier local workaround was to set `DeepseekSparseAttnMultiStepBackend.needs_cpu_seq_lens=True`.
> That fixes the assertion but makes all spec-v2 DSA draft decode materialize a CPU
> sequence-length mirror, **including CUDA graph replay cases and FP8-style deployments that
> do not need it**. This PR instead keeps the common graph path GPU-only and populates CPU
> metadata only in the over-graph eager fallback."*

So flipping the flag is a known-and-declined trade: it taxes every graph-replay deployment
to fix an eager-only path. A downstream patch doing it would be a local workaround, not
something upstream would take.

**Where my inference went wrong:** I argued "the opt-out buys nothing *on this path*, so
publishing the mirror is free." True for this path, false globally — the flag is
per-backend, not per-path, so setting `True` also re-enables the D2H for every graph-replay
step, which is the majority case on CUDA. I over-generalised from one code path to the
flag's whole blast radius. The measurement (lines 809/812 sync unconditionally) was right;
the conclusion drawn from it was not.

**What #29798 also proves:** upstream's mental model of this eager arm is
*"rare, and entered by all ranks together"* — it is scoped to "decode batch exceeds
`--cuda-graph-max-bs`", a global property. Our failure mode is different in kind: entered by
**one rank** because of DP occupancy, on **every** step, because HIP never captures the
draft-extend graph at all. The fix must therefore target *rank-asymmetric entry*, not the
eager path's existence.

## Revised recommendation

**(a′) — make the `DRAFT_EXTEND_V2` metadata path sync-free**, i.e. supply the host-side
lengths from values the spec worker already knows, so lines 809/812 never need to fire.
`base_spec_worker.py:155` already sets `extend_seq_lens_cpu = [num_draft_tokens]*bs` on the
`gpu_only` path for precisely this reason; the missing pieces are `extend_prefix_lens_cpu`
and `seq_lens_cpu`, and `prepare_for_draft_extend` holds host-side equivalents of both
before it nulls them. This keeps the flag `False` (so no graph-replay tax — respects
#29798's constraint) while removing the rank-asymmetric stall.

Note the upstream fix in flight, **PR #32209**, takes a *different* route for a sibling
bug: all-gather the graph-vs-eager decision so ranks vote and agree. That does not help us —
a host-side `cudaStreamSynchronize` on the busy rank is invisible to such a vote — but it
does establish that upstream considers "make the ranks agree" the sanctioned shape. A
defensible alternative is therefore **(c) uniform entry**, which composes with #32209's
philosophy even though it costs idle ranks a sync.

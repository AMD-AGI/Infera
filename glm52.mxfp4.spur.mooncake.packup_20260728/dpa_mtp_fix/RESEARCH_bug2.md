# Bug 2 — research findings (tasks 2 & 3 of the deepsearch)

Three parallel investigations: upstream GitHub, CUDA-vs-HIP source comparison, and public
docs/release notes. Completed 2026-07-29. Combined with the independent code reading in
`CODE_VERIFICATION_bug2.md` (task 1), which reached the same conclusion by a different route.

> Tooling caveat, recorded because it bounds confidence: the GitHub MCP server returned
> `401 Bad credentials` throughout, so the GitHub agents used the **unauthenticated public
> REST API** and a **direct git clone** instead. GitHub *code search* was therefore
> unavailable (it requires auth) and was substituted by grepping a real checkout. The
> docs agent had **no page-fetch tool** — its quotes come from search-index summaries and
> are paraphrase/quote hybrids, not byte-exact. Items resting on a single unverified
> summary are flagged inline. I independently re-verified every load-bearing GitHub claim
> below against the API (see `git log` for the verification commit).

---

## THE ANSWER TO "how does CUDA do it?" — CUDA doesn't do it differently. CUDA never runs it.

Read at tag **`v0.5.15.post1` = commit `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`**.

The eager `max_seqlen_k` code is **byte-identical on both platforms**. There is no cleverer
CUDA implementation. The divergence is upstream of it:

```python
# eagle_worker_v2.py:459-466
if _is_cuda or _is_musa:
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    graph_supported_backend_types.append(DeepseekSparseAttnBackend)   # <-- CUDA only
...
# :472-474
supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and graph_supported_backend
```

On **CUDA**: DSA is in the draft-extend graph whitelist → `cuda_graph_runner_for_draft_extend`
is constructed → `can_run_graph()` is True for captured batch sizes → the guard at
`base_spec_worker.py:162` (`if not is_idle() and not can_cuda_graph`) **skips
`init_forward_metadata` entirely**. The `.item()` is never executed. The graph path uses
`_graph_page_table_width()` (`dsa_backend.py:689-695`), a pure shape read:

```python
if metadata.page_table_1 is not None:
    return metadata.page_table_1.shape[1]
return self.req_to_token.shape[1]
```

On **HIP**: that registration is gated out; the only alternative
(`supports_hip_aiter_draft_extend_graph`) requires an `AiterMultiStepDraftBackend`, which
DSA/tilelang is not; and `Device2ExtendCudaGraphRunner` (`eagle_worker_v2.py:435-440`) has
keys `xpu/npu/cuda/musa` — **no `hip` key at all**, so even passing the isinstance gate
would `KeyError`. The runner stays `None`, every DRAFT_EXTEND_V2 batch runs eager, and each
iteration performs a blocking D2H.

**`gpu_only` is True on CUDA too** — `decide_needs_cpu_seq_lens` has no platform branch. So
CUDA's immunity comes *entirely* from graph capture, not from the flag. Confirms and
sharpens task 1's conclusion.

### Caveat that narrows CUDA's immunity (important, unresolved)

`eagle_draft_extend_cuda_graph_runner.py:304-305` additionally requires
`forward_batch.can_run_dp_cuda_graph`, an **all-gathered min over ranks**
(`scheduler_components/dp_attn.py:110`). If that goes False under partial occupancy, CUDA
would *also* fall eager and hit the identical bug. Could not be determined statically —
it depends on runtime batch composition. **This means NVIDIA may not be permanently
immune, only usually immune.** Relevant when arguing the fix upstream.

---

## Upstream status: the failure CLASS is known and active; our MECHANISM is unreported

### Same class, actively being worked

| Ref | State | Relation |
|---|---|---|
| **#32527** | OPEN, filed **2026-07-27** (one day before our investigation) | `[BUG] EAGLE + DP Attention + PD Disaggregation: Deadlock … GLM-5.2`. Identical topology and precondition: "a request is sent to one DP rank while the other 7 DP ranks are idle". Attributes it to `can_cuda_graph` divergence. NVIDIA B30Z. Only maintainer comment treats it as a config issue. |
| **#32209** | OPEN, **CI red** | `Fix PD decode hang with DP attention and GLM-5.2 MTP`. "the rank receiving the real request can enter the eager MTP draft path while idle DP ranks replay the target verification CUDA graph." Fix approach: all-gather the graph-vs-eager decision so ranks vote. Touches `dsa_backend.py` (+67/−5) but **not** `init_forward_metadata`. |
| **#32722** | OPEN, **red by design** | `[RED regression] Test GLM-5.2 PD + DP attention + MTP` — "The first routed request is expected to hang on current main because the fix has not merged." **Direct proof upstream has no CI covering this combination.** |
| **#29144** | OPEN | EAGLE/NextN + DP attention collective deadlock, DSv4, CUDA. Its py-spy shows the active rank blocked in `resolve_seq_lens_cpu` (`cudaStreamSynchronize`) — closest published stack to ours. Root-caused to stale `EagleDraftInput`, claimed fixed in v0.5.14+. |
| #20404 | — | MI355X disagg DPA + MTP errors — our exact hardware, same interaction class. |

**Critically: none of these is our mechanism.** All attribute the divergence to
`can_cuda_graph`. We **measured** that `can_cuda_graph` is uniformly False on HIP (the graph
is never captured), so their proposed fix — including #32209's all-gather vote — **cannot
fix our hang**: a host-side `cudaStreamSynchronize` on the busy rank is invisible to a
graph-vs-eager vote. Ours is one layer deeper and survives rank agreement on graph/eager.

Exhaustive negative search: no issue or PR anywhere describes a busy DP rank stalling on a
`.max().item()` D2H while idle peers advance. No PR modifies `init_forward_metadata`'s
`max_seqlen_k`. Confirmed against 25 commits of that file's history (through `7f438a603`,
2026-07-28) and against current `main` — **the defect is still present on main**
(`dsa_backend.py:350`, `:820`, `:3206`).

### `#27091` is not what the code comment implies

It is a **merged PR**, not an issue: *"Unify full→SWA index translation in
init_forward_metadata; drop pool caches."* The `schedule_meta` remark the comment cites is
a **rejected-alternative note** explaining why draft-extend metadata must be planned
pre-pad. It says nothing about this deadlock. Our earlier reading of the citation was
correct but the handoff overstated it as "the thing to read first" — it isn't.

### `needs_cpu_seq_lens` history — the `False` is deliberate, and flipping it was already declined

| PR | Date | Content |
|---|---|---|
| #26128 | 2026-05-22 | Origin. Capability flag, default `True`; Triton opts out. |
| #29414 | 2026-06-26 | **DSA opts out** — *"now that all spec-v2 paths replay under CUDA graphs"*. That premise is exactly what fails on HIP. |
| **#29798** | 2026-07-01 | **Explicitly rejects option (b).** Verified verbatim via API. Its diff (+8/−0 on `dsa_backend.py`) **is** lines 806-813. |

> *"An earlier local workaround was to set `DeepseekSparseAttnMultiStepBackend.needs_cpu_seq_lens=True`.
> That fixes the assertion but makes all spec-v2 DSA draft decode materialize a CPU
> sequence-length mirror, including CUDA graph replay cases and FP8-style deployments that
> do not need it."* — #29798

Note #29798's scope: it repaired the eager fallback for batches exceeding
`--cuda-graph-max-bs` — a **global** property, entered by all ranks together. Upstream never
anticipated **per-rank asymmetric entry**. That is the gap we found.

---

## Upstream direction ENDORSES the sync-free fix

sglang **v0.5.15 release notes**:

> *"Spec V2 by default: zero-overhead scheduling via CUDA-graphable DSA draft-extend,
> **dropped D2H/H2D syncs**, fused metadata ops. +11% end-to-end TPS (#29413, #29343,
> #29166, #29077)."*

and the LMSYS blog it summarises:

> *"…made the DSA draft-extend path CUDA-graphable, **made `seq_lens_cpu` optional for DSA
> to drop the D2H sync**, removed the remaining H2D syncs…"*

**So removing this host dependency is upstream's own stated direction**, framed as a perf
win. Our contribution reframes it as also a correctness fix. That is the right PR framing:
*"extends the v0.5.15 sync removal to the eager `init_forward_metadata` path, where it
additionally fixes a DP-rank-divergence hang."*

## The anti-pattern is textbook, and well-sourced

- **NCCL:** *"Collective operations have to be called for each rank … using the same count
  and the same datatype … Failure to do so results in undefined behavior, including hangs."*
- **PyTorch/Meta fleet data:** *"Almost all NCCL watchdog timeouts are caused by collective
  desync (mismatch), not slowness"* and — describing our bug exactly — *"if different ranks
  enter different code paths, they can end up scheduling different collectives … some ranks
  get stuck at the subsequent CPU-GPU synchronization point."*
- **NVIDIA, sync-free CUDA-graph guidance:** *"Something that looks trivial in Python, like
  `if tensor.item() > 0:`, may trigger a host-device synchronization under the hood."*

DeepSeek/FlashMLA/TileLang publish **no** guidance on `max_seqlen_k` host syncs (genuine
negative) — but none of the reference designs *requires* a host-side max; they use
device-resident sentinels (`-1`) and clamped device tensors. A device-only or
static-upper-bound `max_seqlen_k` is consistent with them.

---

## Support-matrix reality

- **No consolidated support matrix exists.** Only `STANDALONE` and `NGRAM` are explicitly
  barred from `--enable-dp-attention`. EAGLE/MTP + DP + PD is shown by example — **Ascend
  NPU only**, no ROCm equivalent.
- **Neither blessed nor forbidden.** Do not expect upstream to reject this as "unsupported",
  but also do not expect prior validation. #32722 existing proves there is no CI for it.
- ⚠️ **Unverified but high-impact:** the GLM-5.2 cookbook reportedly disables MTP for AMD
  outright — *"MTP on gfx950 still depends on the spec-decode draft kernel, which isn't yet
  validated on this hardware."* Rests on a search summary of a JS-rendered page. **If true,
  we are the first to exercise this path on gfx950 and should expect more bugs behind this
  one.** Worth confirming manually before promising a complete fix.

## ⚠️ A SECOND, INDEPENDENT deadlock in this stack — must rule out

> EAGLE on gfx942/gfx950 requires **`--disable-custom-all-reduce`**: the aiter custom
> all-reduce kernel deadlocks during EAGLE verify at high concurrency.
> — sglang GLM-5.1 cookbook

Our runs do **not** currently set this flag. This is a distinct mechanism from the indexer
sync and could produce a superficially similar hang. **Add it to the next repro run** so a
persisting hang cannot be misattributed. (Related: #28815 / #31071 / PR #31478 — ROCm EAGLE
greedy-verify missing a TP broadcast, same workaround.)

---

## Recommended fix — converged across all three investigations

### Fix A (primary, ~6 lines): use the static page-table width, the codebase's own idiom

Replace the `else` arm at `dsa_backend.py:738-750`:

```python
else:
    # needs_cpu_seq_lens=False nulls the host mirror. Do NOT sync here: under
    # DP-attention only the occupied ranks would block, so the host falls out of
    # step with idle peers and the next collective goes ragged. Use the static
    # page-table width -- the same bound the graph path uses
    # (_graph_page_table_width) and the same idiom as triton/trtllm
    # ("over-allocate is safe"). Downstream kernels bound real reads by
    # cache_seqlens / seqlens_expanded, so a wider table only costs the
    # indexer's logits buffer.
    max_seqlen_k = self.req_to_token.shape[1]
```

**The idiom is established in-tree, three ways:**
- `triton_backend.py:704-708`, `:783-788`, `:837-841` — *"gpu_only: seq_lens_sum may be
  None; over-allocate is safe (ragged write)"*, falling back to `bs * self.max_context_len`.
- `trtllm_mha_backend.py:555-559` — *"Static upper-bound page-table width … the kernel
  bounds real KV reads by cache_seqlens, so this is a fixed loop bound only — never a host
  max / seq_lens_cpu D2H sync."*
- `dsa_backend.py:689-695` — **DSA's own graph path already does exactly this.**

**Over-sizing proven safe** (full consumer trace): the wide `page_table` is only indexed
*through* top-k, and top-k is masked per row by `lengths` derived from `cache_seqlens` —
independent of `max_seqlen_k`. `_topk_unfused` builds
`valid_mask = (col >= row_starts) & (col < row_starts + lengths)` and fills `-inf`;
`transform_index_page_table_decode_kernel` loads only under `mask = idx >= 0`. Columns
beyond the real length get `-inf`, are never selected, never dereferenced. Cost is the
indexer logits buffer only: at 128K context, bs=8, next_n=4 → ~16.8 MB/call vs a few MB at
realistic lengths. Bounded, but **must be benchmarked** — this is the one open cost question.

### Fix A2 (required, same change): lines 806-813 must be handled too

Independently found by task 1 and confirmed here. Those two unconditional syncs
**will deadlock identically even after line 746 is fixed**. Dataflow says both look dead for
DRAFT_EXTEND_V2 (`extend_seq_lens_cpu` is already supplied host-side at
`base_spec_worker.py:158`; `extend_prefix_lens_cpu` is read only in the `is_extend()` branch
and in `_cal_indexer_k_start_end`, which returns early for this mode) — **but
`init_forward_metadata` mutates `forward_batch` in place, so grep for post-call readers
before removing.** Verify by instrumentation, do not assume.

### Explicitly NOT the fix
- **(b) flip `needs_cpu_seq_lens=True`** — declined upstream by #29798, taxes all
  graph-replay deployments.
- **#32209's all-gather vote** — cannot see a host-side sync.

### Fix B (follow-up, real perf, out of scope for the hang)
Register DSA in the HIP draft-extend graph whitelist and add a `"hip"` key to
`Device2ExtendCudaGraphRunner`. Would close the eager-path gap entirely. Larger change;
`_apply_cuda_graph_metadata`'s non-fused branches already appear HIP-capable.

## Open questions before implementing
1. Benchmark the widened logits buffer (the only real cost of Fix A).
2. Confirm lines 806-813 are truly dead for DRAFT_EXTEND_V2 by instrumentation.
3. Add `--disable-custom-all-reduce` to rule out the second known ROCm EAGLE deadlock.
4. Manually verify the "MTP disabled for AMD in the GLM-5.2 cookbook" claim.

# r03 — code review of #33968 (`plan.md` step 4, first pass)

Reviewed against the rebased head `138901d918` on upstream main `56834422a1`.
Hardware validation for this PR is complete — see `writeback_33968.md`.

## Verdict

The fix is correct and minimal for the fault it targets, and the hardware A/B
backs it. **One issue is worth raising on the PR before it is flipped**, below.
It is not a blocker for the fault being fixed, but a reviewer will find it and
it is better to state it first.

## What the patch gets right

- **The mechanism matches the measurement.** `pool_host/dsa.py:159` builds
  `index_k_data_ptrs` from host `data_ptr()`s into a device-resident tensor, and
  `transfer_kv_all_layer_mla(dst_layers=...)` dereferences it from a kernel. With
  `hipHostRegister` the device mapping is at a different address, so the kernel
  faults at the host VA. Reproduced; see `writeback_33968.md`.
- **Switching the default, not just the `"cuda"` key, is necessary.** Verified
  directly: `torch.device("cuda:0")` is not dict-key-equal to `"cuda"`, and
  `dispatch_check.py` shows all three keys resolving to `alloc_with_host_register`
  on stock and to `alloc_with_pin_memory` on the patch. The commit message's
  justification is accurate.
- **No behaviour change off ROCm.** `_is_hip` is False on CUDA/XPU/MPS builds, so
  the default remains `alloc_with_host_register` and the `npu`/`musa` entries are
  untouched. The `defaultdict` shape is preserved.
- **The import follows existing idiom.** Every sibling in `pool_host/`
  (`base.py`, `dsa.py`, `mha.py`, `mla.py`, `mamba.py`) already does
  `from sglang.srt.utils import is_hip`. `is_hip()` is
  `torch.version.hip is not None` — no import cycle, no runtime cost.
- **Scope is clean.** `validate_A.py` PASS: equivalent to the proven local fix on
  every dispatch key and on the measured allocation, removes nothing, adds only
  `_ALLOC_MEMORY_FUNCS` / `_is_hip` / `is_hip`, and carries none of infera's local
  marker (`GLM52_ROCM_HOST_ALLOC`).

## The issue to raise: the patch silently bypasses a configured storage allocator

The two allocators do not treat the `allocator` argument the same way:

```python
def alloc_with_host_register(dims, dtype, device, pin_memory, allocator):
    buffer = allocator.allocate(dims, dtype=dtype, device=device)   # <- used
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer

def alloc_with_pin_memory(dims, dtype, device, pin_memory, allocator):
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer                                                    # <- ignored
```

`allocator` is annotated `None` on the pin_memory side, but **no call site ever
passes None**: every pool does
`self.allocator = get_allocator_from_storage(allocator_type)` (`base.py:146`,
`dsa.py:68`, `mha.py:710`, `mamba.py:63`), which returns a `HostTensorAllocator`,
or a `MooncakeHostTensorAllocator` / `UMBPHostTensorAllocator` / shm variant when
`--hicache-storage-backend` selects one (`kv_cache_builder.py:125`).

So after this patch, on ROCm, a deployment that configures a mooncake or mori
storage backend gets `torch.empty` instead of its allocator, with no warning.

**Measured consequence for the default allocator** (`allocator_bypass_check.py`):

```
host_register(default alloc)   pinned=True  nbytes=131072
pin_memory                     pinned=True  nbytes=131072

allocator state after host_register: (256, 512) torch.uint8
allocator state after pin_memory:    None None   <- untouched
```

Equivalent buffers; the allocator object simply never runs. For the **default**
`HostTensorAllocator` the only difference is `alloc_mmap` vs `torch.empty`, and
both yield a pinned host buffer — harmless. For **mooncake/mori** the allocator
exists precisely to hand back registered//shared memory, so bypassing it is not
harmless, and that path is untested here.

### Why this is not a blocker

- It is **pre-existing upstream behaviour**, not something this PR invents:
  `"npu"` and `"musa"` have always routed to `alloc_with_pin_memory` and have
  always ignored the allocator the same way. The PR extends an existing
  inconsistency to ROCm; it does not create it.
- The alternative — keeping `alloc_with_host_register` on ROCm — is exactly the
  configuration that faults, as measured. Bypassing the allocator is strictly
  better than a GPU memory access fault.
- The default path is measurably equivalent.

### What to say on the PR

State it plainly rather than let a reviewer discover it: that on ROCm a
configured non-default host allocator is now bypassed, that this mirrors what
`npu`/`musa` already do, and that reconciling the two allocators' handling of the
`allocator` argument is worth a follow-up but is out of scope here. Offer the
narrower alternative (route only when `allocator_type == "default"`, keep
host_register otherwise) and let the maintainers choose — noting that the
narrower form leaves the faulting path reachable.

## Second pass — deeper read (LSP + serena)

### The fix covers more pools than the commit message claims

The commit message names only `DSAIndexerPoolHost.index_k_data_ptrs`. In fact
**all four host pools** build the same device-resident table of host `data_ptr()`s
and all four resolve their allocator through the same dispatch:

| pool | table | dispatch site |
|---|---|---|
| `dsa.py` | `index_k_data_ptrs` (:159) | `:138` |
| `mha.py` | `k_data_ptrs` / `v_data_ptrs` (:119, :124, :759) | `:186`, `:786`, `:1109` |
| `mla.py` | `data_ptrs` (:102) | `:170`, `:202` |
| `mamba.py` | temporal / conv_state ptr tables (:115, :123) | `:136` |

So the one-line dispatch change fixes the fault for every pool at once. This is
a point **in the PR's favour** that the description undersells — worth adding.

### Pool-level verification, both arms

`scripts/pool_alloc_check.py` resolves the allocator exactly as the pools do
(`ALLOC_MEMORY_FUNCS[torch.device("cuda:0")]`), allocates through it, and measures
the property the tables depend on:

```
stock     dispatch[cuda:0] -> alloc_with_host_register
          host=0x7db5a343a000  devPtr=0x7db5a2c20000  same=False  pinned=True
          RESULT: UNSAFE -- kernel will fault

patched   dispatch[cuda:0] -> alloc_with_pin_memory
          host=0x77424fa50000  devPtr=0x77424fa50000  same=True   pinned=True
          RESULT: SAFE for the pointer tables
```

Note the stock buffer **is pinned** and still unsafe. Pinning is not the property
that matters here; address identity is. Anyone reading the code and reasoning
"`pin_memory=True` was already passed, so it must be fine" would get this wrong —
worth a sentence in the PR.

### The allocator-bypass finding is softer than it first appears

`mamba.py:138` already wraps the dispatched allocator and **skips it entirely**
for zero-element buffers:

```python
def alloc_func(dims, *, dtype, device, pin_memory, allocator):
    # conv-only linear attention has no ssm state: mmap can't map the
    # 0-element temporal buffer, so hand back a plain empty tensor.
    if np.prod(dims) == 0:
        return torch.empty(dims, dtype=dtype, device=device)
```

So bypassing the allocator indirection is already an accepted in-tree pattern
where the allocator cannot serve the request, not something this PR invents. That
does not remove the need to disclose it, but it does argue against treating it as
a defect.

### Type annotation is misleading, pre-existing

LSP reports `alloc_with_pin_memory(..., allocator: None)`, while every call site
passes a real `HostTensorAllocator`. The annotation has been wrong since before
this PR (`npu`/`musa` already route here). Not this PR's to fix; mention only if
a maintainer asks.

### Nothing else found

No circular import (`is_hip` is `torch.version.hip is not None`; every sibling in
`pool_host/` already imports it from `sglang.srt.utils`). No change off ROCm.
No other consumer of `alloc_with_host_register` anywhere in the tree — the
dispatch table is its only caller.

**Both review passes are complete. Ready to flip.**

## Post-flip: the CI red does NOT clear by itself

Flipped to ready 2026-08-24 05:32; `mergeable` went `BLOCKED` -> `MERGEABLE`.
The pr-gate checks are still red, and `plan.md` step 5.2 is wrong about why:

> *"The current red on all three is the `Block draft PR` repo policy, not a real
> failure — it should clear on its own once the PR is no longer a draft."*

It does not clear on its own, for two independent reasons, both verified:

**1. `ready_for_review` does not retrigger the workflows.** `pr-test.yml` declares
a bare `on: pull_request`, which defaults to `opened` / `synchronize` /
`reopened` only. Every run on this head is from 04:31 (the push); the flip at
05:32 produced none. The stale runs logged `PR Draft: true` because that is what
was true when they ran. The gate itself reads live state — `pr-gate.yml:30` calls
`github.rest.pulls.get` at runtime, not the event payload — so a *fresh* run would
see `draft: false`. There just is no fresh run. `gh run rerun` is refused:
`Must have admin rights to Repository`.

**2. A `run-ci` label is also required, and only maintainers can set it.**
`pr-gate.yml:58` enforces `require-run-ci`, and the run logged
`Require run-ci: true` against `PR Labels: [hicache]`. So even a fresh run would
fail the next gate. `gh pr edit --add-label run-ci` is refused:
`dorado269 does not have the correct permissions`.

This is the normal path, not a problem with this PR: every one of the last eight
merged PRs sampled carries `run-ci`, applied by a maintainer.

**Consequence for the remaining two PRs.** Do not treat "flip and watch the gate
go green" as a step that can be completed from this side. The reachable
definition of done is: validated, rebased, evidence posted, out of draft. Green CI
requires a maintainer to add `run-ci`, and a push (or maintainer rerun) to
retrigger. Worth noting in `pr.done.md` so the next session does not chase it.

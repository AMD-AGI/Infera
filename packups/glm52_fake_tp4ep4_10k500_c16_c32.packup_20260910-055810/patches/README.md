# Three fake-only patches: what / why / how

All three target SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a`. The measured launcher bind-mounted the complete modified files read-only over the immutable image. It did not modify the image or commit a source checkout. Baseline files are the exact original files from the earlier experiment; the CPU audit applies every patch with zero fuzz and checks byte equality with each shipped modified file.

| Exact patch | Original in source/baseline/ | Modified in source/ | Runtime path below /sglang/ |
|---|---|---|---|
|`fake_proposal.patch`|`eagle_disaggregation.py`|`eagle_disaggregation_fake_fix.py`|`python/sglang/srt/speculative/eagle_disaggregation.py`|
|`fake_index_elision.patch`|`kv_cache_configurator_original.py`|`kv_cache_configurator_fake_fix.py`|`python/sglang/srt/mem_cache/kv_cache_configurator.py`|
|`fake_topk_domain.patch`|`dsa_utils.py`|`dsa_utils_fake_fix.py`|`python/sglang/srt/layers/attention/dsa/utils.py`|

## 1. Fake proposal one-hot

**What:** initialize a full-vocabulary float32 `draft_probs` distribution with a point mass at the fake handoff's supplied dummy top-k token, and set `topk_p` to1. Publication to the overlap relay happens afterward so it carries the distribution.

**Why/context:** the established ROCm rejection-sampling path needs the first proposal distribution. The fake prebuilt PD handoff omitted it; the earlier debugging run reached a `torch.stack` with `None` instead of proposal probabilities. A normalized fake distribution makes that synthetic handoff structurally valid.

**How:** narrowly guard on transfer backend `fake` and rejection sampling enabled. Non-fake transfer and non-rejection behavior stay unchanged. The four-case verbatim CPU regression covers normalized one-hot, overlap relay, real-transfer no-change and non-rejection no-change. It requires the exact image's Torch/SGLang imports; this packup host lacks Torch, so it is supplied but not rerun here. Patch/syntax checks are independent of those imports.

**Limit:** this does not create valid prompt KV, choose acceptance3.61, transfer true prefill proposals, or repair/validate real PD.

## 2. Shared-index elision

**What:** extend the existing shared-topk index-K elision eligibility from aggregate/mix (`null`) mode to the fake decode target.

**Why/context:** unused index-K allocations constrained the older long-context capacity test. Real PD must preserve its wire/buffer layout, while fake handoff transfers/registers no real KV buffers. Reusing the existing mix eligibility avoids redundant target index storage without aliasing prompt KV pages or shortening inputs.

**How:** `_should_elide_dsa_index_k` retains no-HiCache, no-HiSparse and not-draft conditions. Its existing accounting/allocation call sites use the same eligibility. Add only `(mode == decode and backend == fake)`. The verbatim72-case truth-table test exercises3 modes ×3 backends ×2 draft ×2 HiCache ×2 HiSparse; original fails and patched passes in the offline audit.

**Limit:** prior long-context capacity gains motivate the change but are not measured as an isolated speedup in this short run. Real transport retains its original layout.

## 3. Consistent fused physical-topk domain

**What:** fake decode keeps the local physical-slot fused-topk convention across producer and consumers.

**Why/context:** the optimized gfx950 indexer can emit physical KV slots. The original HIP PD index-sharing condition disabled the corresponding fused consumer, treating those slots as logical request-table positions. That producer/consumer domain mismatch caused the previous fake long-context failure. Clipping indices or disabling the optimized indexer would hide or avoid the mismatch rather than preserve the intended contract.

**How:** add a fake-decode exception to `should_use_dsa_fused_topk`, preserving the existing fuse flag, seed/remap behavior and non-fake paths. The exact guard test is **48 cases**, not the96 mentioned in early historical prose:3 modes ×2 backends ×2 seed ×2 fuse flag ×2 remap. Original fails and patched passes offline.

**Limit:** no general claim about true-PD seed transfer/remapping. These three patches were reused unchanged here; there was no new kernel optimization. C32's192-row FlyDSL decline remains an explicit fallback, not something these patches remove.

## Applying versus replaying

For offline inspection, each patch uses standard `a/python/...` / `b/python/...` paths and `patch --batch --fuzz=0 -p1` from a staging source root. `scripts/audit_package.py` performs that staging entirely inside temporary audit space and checks exact output bytes. For actual replay use the generated launcher, which already mounts the three supplied modified files; do not patch the image or original experiment in place.

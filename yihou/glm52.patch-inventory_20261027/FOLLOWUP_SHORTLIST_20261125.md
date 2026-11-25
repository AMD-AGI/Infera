# GLM-5.2 upstream follow-up shortlist — 2026-11-25

## Audit scope and evidence

Live authored-PR discovery covered `sgl-project/sglang`, `AMD-AGI/Infera`, `kvcache-ai/Mooncake` and `ROCm/aiter`. Ten authored SGLang PRs and three Infera drafts remain open; there are no authored open Mooncake/Aiter PRs. External #45524 is retained as a serving dependency. Top-k #47872 remains owner-paused; its CI/reviews were not refreshed.

SGLang main was pinned to [`5b4ddbb609e9a6101e8c3620d22ade2059f77b62`](https://github.com/sgl-project/sglang/commit/5b4ddbb609e9a6101e8c3620d22ade2059f77b62). This is a source/control audit, not a new GPU or integrated-serving qualification. CPU probes ran actual extracted main functions: 24 collapsed-draft owner/length cases, the moved offloader's copy method, and the ROCm PD+DCP config gate. Raw evidence is preserved locally in `runtime/audit-main-20261125/`; the probe outcome is recorded below. No PR was closed or rewritten during this audit.

## Ready PRs to keep following

| Priority | PR | Purpose | Current state | Main comparison / next action |
|---|---|---|---|---|
| P0 | [#47278](https://github.com/sgl-project/sglang/pull/47278) | gfx950 FP8 sparse MLA split-K decode | Ready; head `9602092a`; Base passed; AMD passed; kkHuang-amd approved | Required historical kernel family. Still open and not in main; prioritize maintainer merge/final review. Do not restart passing CI for freshness. |
| P0 | [#47897](https://github.com/sgl-project/sglang/pull/47897) | Guarded gfx950 paged MQA logits with paired-row correctness | Ready; head `5a06f9e7`; no approval; latest Base ran zero tests and failed its finish check | Required historical kernel family; Gluon implementation is absent from main. Maintainer must approve this commit with `/tag-and-rerun-ci`, then recover Base. See [bot instruction](https://github.com/sgl-project/sglang/pull/47897#issuecomment-7123846157). Preserve schedule fallback, pair-tail and numerical qualifications. |
| P1 | [#47276](https://github.com/sgl-project/sglang/pull/47276) | gfx950 fused indexer logits/top-k for small/MTP batches | Ready; head `e95ec5ab`; Base passed; no human approval | Separate optional path, disabled in frozen llr37. Not superseded by #47897. Obtain kernel review and coordinate overlap with external #45307; preserve passing head. |
| P1 | [#45484](https://github.com/sgl-project/sglang/pull/45484) | Preserve `is_shuffled` through OffloaderV2 H2D copies | Ready; head `43dec26b`; conflicts; no approval | Main moved the file to `srt/layers/offloader.py`, but `_CpuParamOffloader.create_device_tensor()` still returns bare `.to(...)`. Actual CPU copy probe loses the tag. Rebase to the new path, update imports/test registration and run CPU/Base. Initial-weight tagging fixes #48219/#51463 do not preserve attributes on fresh copies. Not used by llr37. |
| P1 | [#46822](https://github.com/sgl-project/sglang/pull/46822) | Skip DSA indexing for an eager idle ROCm rank | Ready; head `11262e11`; mergeable; no approval | Main's eager idle early return still checks `_is_cuda` only. HIP extension remains missing; keep this focused fix and recover current-head CI/review. External graph-consistency #48302 changes a separate layer and does not replace it. Not an additional llr37 requirement. |
| P0, external | [#45524](https://github.com/sgl-project/sglang/pull/45524) | Wait for a PP producer before sending a computed non-final KV chunk | External, Ready; head `5818b0bf`; Base passed; no approval | Main `process_prefill_chunk` still calls `send_kv_chunk` directly in `elif computed`. Existing cached-prefix event fencing is a different branch. Keep following the external author; do not create a duplicate or alter their branch. |

## Draft PRs with remaining value

| Priority | PR | Purpose | Main comparison | Next action / limits |
|---|---|---|---|---|
| P2 | [#46819](https://github.com/sgl-project/sglang/pull/46819) | Cache DCP pack metadata and prepare row indices on the gather stream | Main still rebuilds a device `torch.tensor(metadata)` for every gather; row-index conversion remains outside gather-stream context. No equivalent cache implementation found. | Retain, but rebase conflicts and remeasure isolated performance against current main. Preserve producer dependency/stream lifetime and pointer/layout invalidation. DCP optimization, outside llr37 DCP1. |
| P2 | [#46826](https://github.com/sgl-project/sglang/pull/46826) | Enable ROCm DCP decode behind DCP1 prefill | Main still rejects PD+DCP on non-NPU platforms. CPU probe confirms ROCm decode rejection. The feature is not upstreamed. | Keep as a draft feature, not a ready guard deletion. Rebase while preserving new main NPU allowance and ROCm HiCache support; validate current generic collapsed-draft transport instead of depending on the obsolete #46824 implementation. Historical DP2×DCP2 graph correctness and e2e coverage remain; coordinate external #48302. |

## Owner-paused

| PR | Status | Action |
|---|---|---|
| [#47872](https://github.com/sgl-project/sglang/pull/47872) | PAUSE — standalone Triton top-k; owner requested redesign | Retain historical record. Do not query CI/reviews, retry, rebase, promote or merge until explicit resumption. Historical llr37 top-k family does not make this reworked port currently merge-ready. |

## Recommend retiring the old implementation

### #46824: sharded draft relayout

[PR #46824](https://github.com/sgl-project/sglang/pull/46824) is still Draft/conflicted, but its core purpose is now implemented by main's more general layout protocol:

- Merged [#48741](https://github.com/sgl-project/sglang/pull/48741) carries the decode pool's `rows_are_dcp_collapsed` through `draft_rows_dcp_collapsed` / `dst_draft_rows_dcp_collapsed` registration.
- [`build_dcp_token_transfer_plan`](https://github.com/sgl-project/sglang/blob/5b4ddbb609e9a6101e8c3620d22ade2059f77b62/python/sglang/srt/disaggregation/common/utils.py#L178) uses the target's source/destination rows for a collapsed draft. The actual current-main function passed 24 CPU cases across DCP2/DCP4, all ranks and partial/full token counts.
- Main's `prepare_dcp_token_item_lens(..., draft_rows_collapsed=...)`, Mooncake draft-copy planner and decode registration now express the layout rather than inferring the remote layout solely from a local ROCm model predicate.
- Merged [#51455](https://github.com/sgl-project/sglang/pull/51455) also packs a collapsed draft with its target rows.

Recommendation: **retire the original production patch from the active queue**. Before closing it, map/port any uniquely useful regression coverage to current main and verify byte-copy semantics; do not resurrect old bootstrap fields or backend rejection logic. CPU ownership-plan checks are not network transport or model/e2e proof. This audit recommends closure/supersession; it does not close the PR.

## Already completed or superseded — no merge work

- SGLang #33059, #46820, #46830 and #49732: merged. #49732 supersedes active ownership of external #46918; preserve attribution without pursuing two implementations.
- Mooncake #5005 and Infera #218: merged.
- SGLang #33968/#33970/#33973/#45483/#46828: already closed or replaced; retain historical links rather than reopening them. #45483's external #45156 remains a separate author's work.
- #33643: closed historical speculative-penalty work; not in the active merge queue. Any renewed task needs a fresh current-main issue audit.

## Separate projects and documentation

| PR | Disposition |
|---|---|
| [SGLang #51760](https://github.com/sgl-project/sglang/pull/51760) | Retain in a separate Muse Spark / InferenceX docs queue. Its two HiCache/SWA environment variables are still absent from main's cookbook fragment. It is neither a GLM-5.2 bugfix nor a 48k prerequisite. |
| [Infera #221](https://github.com/AMD-AGI/Infera/pull/221) | Historical kernel evidence catalog, Draft. Archive/provenance work, outside runtime/CI merge priority. Do not count as a deployment patch. |
| [Infera #228](https://github.com/AMD-AGI/Infera/pull/228) | Permanent Draft for view/track; retain as the team's reference, never promote/merge. |
| [Infera #252](https://github.com/AMD-AGI/Infera/pull/252) | Independent agent_sys/o11y design work. Outside this GLM queue; make no retirement decision based on SGLang main. |

## Operational handoff

The proposed GLM queue contains **eight ongoing implementation PRs including external #45524**, plus **one explicitly paused top-k row**. #46824 is a retirement candidate; other project/docs work is separate. Current automated mutation scope remains #47276/#47278/#47897; this audit does not silently reactivate other branches. Changes to that scope require a subsequent owner instruction. Base success is distinct from approval, complete platform CI, and integrated-image/numerical qualification.

All throughput/performance evidence must remain tied to its own revision and run. Historical 48k used simulated acceptance; do not attribute that whole-stack score to any individual PR or claim a new PR-only image is qualified by this source audit.

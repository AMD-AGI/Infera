# Next-turn PR collection

Defined by the owner on **2026-11-25**: the previous follow-up shortlist excluding #47278 and owner-paused #47872. The collection contains six owned PRs plus external/read-only #45524. #46824 remains a retirement candidate outside this collection.

## Rebase and CI admission snapshot

All six owned PRs were replayed onto SGLang main `5f241eb8ba6c` (the commit merging #47278; use the complete SHA in `maintenance/next_turn.json`). Each was pushed with an exact old-head lease, preserving yihou DCO and without coauthor/assistant trailers. All six are mergeable. External #45524 was not modified.

| PR | Purpose | State | Published head | run-ci | Maintainer action |
|---|---|---|---|---|---|
| [#47276](https://github.com/sgl-project/sglang/pull/47276) | Optional fused indexer logits/top-k | Ready | `d334464b7225` | Yes | `/approve-ci` on the new head |
| [#47897](https://github.com/sgl-project/sglang/pull/47897) | Paged MQA logits | Ready | `d6cb51b6500f` | Yes | `/approve-ci` on the new head |
| [#45484](https://github.com/sgl-project/sglang/pull/45484) | OffloaderV2 shuffled-weight tag preservation | Ready | `8e6eca78f9a2` | No | `/tag-and-rerun-ci` |
| [#46822](https://github.com/sgl-project/sglang/pull/46822) | Eager idle HIP indexer early return | Ready | `16f18d9a04e1` | No | `/tag-and-rerun-ci` |
| [#46819](https://github.com/sgl-project/sglang/pull/46819) | DCP gather metadata caching | Draft | `74116b2c056d` | No | `/tag-and-rerun-ci`; Draft gate still applies |
| [#46826](https://github.com/sgl-project/sglang/pull/46826) | ROCm PD+DCP decode enablement | Draft | `dba5922c0a46` | No | `/tag-and-rerun-ci`; Draft gate still applies |
| [#45524](https://github.com/sgl-project/sglang/pull/45524) | PP prefill producer-before-KV-send ordering (external) | Ready | `5818b0bf466c` | Yes | No change requested; preserve existing passing Base |

## Why labels alone are insufficient

Pinned main [5f241eb8ba6c](https://github.com/sgl-project/sglang/commit/5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef) uses `pull_request_target` with `approval-mode: commit`. For this low-permission author, the run-ci label must be present and the exact PR commit must carry a successful `ci/run-ci-approved` status. All six new owned heads currently lack that status. An authorized maintainer can use `/tag-and-rerun-ci` to label/approve, or `/approve-ci` when the label already exists. Bare UI labeling, ordinary retries and UI reruns do not approve the commit. See [the Base finish instruction](https://github.com/sgl-project/sglang/blob/5f241eb8ba6cb43b5dde2d0e57f3ab1c482202ef/.github/workflows/pr-test.yml#L538).

**#46819 and #46826 remain Draft.** Main skips tests on Draft PRs regardless of labels/approval. This task did not promote them; current performance/transport or graph/numerical/e2e blockers remain. Arrange CI admission when those Ready gates are satisfied, rather than treating labeling as functional-test execution.

## Compatibility and local validation

- #45484: follow main’s `srt/layers/offloader.py` move in the tests; migrate AMD suite registration. Three checked-in CPU tests passed.
- #46822: preserve the focused HIP extension and main’s current idle/result handling. Four checked-in CPU tests passed.
- #46819: retain main’s Mooncake test helpers and CUDA runner; migrate the obsolete AMD JIT suite to `base-a-jit-kernel-unit`. Two checked-in CPU metadata tests passed.
- #46826: preserve main’s NPU PD allowance and standalone ROCm HiCache behavior. The additional ROCm PD decode remains limited to no HiCache. Seven checked-in config tests passed, including both compatibility controls. Earlier #46824 merge-dependency references are superseded by generic collapsed-draft support in main; current runtime/graph qualification remains.
- #47276/#47897: kernel and launch-policy blobs are identical to their previous published heads. Each checked-in CPU host-dispatch test passed; #47897 retains the persistent-schedule Aiter fallback.

**18 CPU tests passed**, using the checked-in tests and actual production ASTs with optional GPU/server imports isolated. This is not full-package collection, GPU numerical validation or new e2e/throughput evidence. All PR-scoped hooks passed; the #46819 suite and #46826 formatting failures were corrected and the failed hooks rechecked. Historical GPU/performance evidence remains revision-specific.

Artifacts: ignored local `runtime/next-turn-20261125/` contains branch manifests, CPU runner/results, scoped-hook logs and final live PR/label/approval snapshots. No files or branches belonging to other people were rewritten.

## Holds and monitoring

#47278 is excluded from next-turn and is now in main. #47872 remains owner-paused. The existing scheduled kernel monitor retains its configured scope; this collection definition is not an implicit expansion of its retry allowlist. Temporary interactive write holds on #47276/#47897 are released after publication; fresh approval-gate requirements are recorded in the worker prompt.

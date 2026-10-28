@HaiShaw @bingxche I have rebased this existing PR onto main `449617bc94a5d4ae23ad62d0da2f842e711ca983`; the updated head is `d0062f8bce0c86de998ff36b8c106e2ab4bc6ca5`.

The new registered CPU regression runs the production `_get_topk_paged` method across legacy/preshuffle layouts, decode, target verification, draft-extend-v2, and empty/padding-only queries. Patched: **12 passed**. Restoring unpatched main: **10 failed, 2 unpadded controls passed**. All repository hooks applicable to the diff passed.

Reviewing current Aiter also caught a missing boundary in the old proposal: its SplitKV scheduling divides by the query count, so passing a zero-row slice is unsafe. The updated fix returns the correctly shaped all-`-1` indices without launching Aiter or top-k for zero real queries; the tests enforce that.

**This does not yet satisfy your requested AMD DP-attention accuracy results.** These are CPU contract tests, with Aiter/top-k stubbed. No current-head MI300/MI355 model-accuracy result is claimed. Isolated GPU requests did not obtain a node during this validation session; I have put the exact MI300, MI355, and stage-c suite commands in the PR body and retained them as outstanding pre-approval requirements. The existing `run-ci` label has now started CI for the rebased head; please ensure the requested nightly/stage-c AMD coverage executes as well. Skipped CI and the old downstream deployment results are not substitutes.

The body also records the related CP metadata-padding work in #32175 so the two contracts can be reconciled rather than independently layered.


CI triage on this head: corrected the legacy title to the current `[Fix] [AMD] dsa: ...` format after the title check flagged it. The [Intel SPR job](https://github.com/sgl-project/sglang/actions/runs/40473090335/job/122084833207) failed while building the Docker image with `no space left on device`, before any tests ran. That runner needs capacity recovery before a useful rerun; changing this indexer patch cannot resolve that infrastructure failure. AMD and the remaining base checks are still in progress.

# Remaining 48k upstream priorities — November 10, 2026

The reproduction reference is llr37-c104 (48,713 total tokens/s/GPU), with
three-run mean 48,204. These runs used simulated MTP acceptance 2.99; the
historical correctness caveats remain. No individual PR inherits that throughput
claim. See [deployment/source mapping](LLR37_DEPLOYMENT_PRIORITY.md),
[public reproduction](PUBLIC_REPRODUCE.md) and [live handoff](PR_HANDOFF.md).

## Required behavior and source families

| PR | Purpose | Verified state on November 10 | Next action |
|---|---|---|---|
| [#33059](https://github.com/sgl-project/sglang/pull/33059) | Align real/padded Aiter query rows | Merged | Use a revision/image containing the merge. |
| [#46820](https://github.com/sgl-project/sglang/pull/46820) | Prevent KV metrics backpressure from blocking scheduling | Ready; new head 3c12e225; Base and NVIDIA Extra active | Complete current-head Base and reviewer-requested NVIDIA Extra, then merge review. |
| [#46830](https://github.com/sgl-project/sglang/pull/46830) | Remove rank-divergent DSA host synchronization | Ready; current-head Base and AMD passed | Owner is arranging maintainer merge; preserve passing head. |
| [#45524](https://github.com/sgl-project/sglang/pull/45524) | Wait for PP producer before early KV transfer | External author; Base passed | Maintainer review/merge; retain historical source-equivalence qualification. |
| [#47278](https://github.com/sgl-project/sglang/pull/47278) | gfx950 split-K FP8 sparse MLA | Ready; main conflicts and missing required base; historical AMD passed | Reconcile four production kernel conflicts with numerical validation before new CI. This is the principal remaining code blocker. |
| [#47897](https://github.com/sgl-project/sglang/pull/47897) | gfx950 paged MQA logits | Ready; Base passed | Complete review and exact source/evidence mapping. |
| [#47872](https://github.com/sgl-project/sglang/pull/47872) | Opt-in Triton top-k | Ready; Base/AMD/Xeon/ARM64 passed | Complete numerical/dispatch/fallback review and maintainer merge. |

These ports cover historical behavior/source families. The frozen image already
contains patches; it does not need to wait for all merges. A current-main image
with these ports still needs combined-stack correctness and performance validation.

## Conditional and optional work

- [#46918](https://github.com/sgl-project/sglang/pull/46918) is the external
  absorb-BMM compiler-failure candidate for a stock-image path. Its Base was
  cancelled and main conflicts remain. The historical reproduction retains its
  frozen workaround, so this PR is conditional rather than an additional kernel
  throughput prerequisite.
- [#47276](https://github.com/sgl-project/sglang/pull/47276) is optional: the fused
  indexer path was off in llr37. Its repeated CI failure must not be presented as
  a blocker to reproducing the selected 48k recipe.
- The exact K01 overlay (including historical MoE), Aiter GEMM CSVs and role/site
  configuration remain required reconstruction inputs. They are not replaced by
  a list of PR URLs; see the deployment/source mapping above.

## Monitoring

Owner instruction: poll every three hours, `0 */3 * * *` UTC, with the existing
one-hour minimum between authorized retry requests. Preserve the separate daily
ledger schedule and permanent-Draft [Infera #228](https://github.com/AMD-AGI/Infera/pull/228).
Monitor Base by default; NVIDIA Extra for #46820 is explicitly required by the
owner/reviewer and must be followed even after Base passes. Preserve active runs
and successful heads. Current run details are refreshed by the maintenance worker.

# Packups — GLM-5.3 series integration, 2026-09-01/03

**This branch is not part of PR #151 and is not intended to merge.** It carries
the experiment-cluster packups alongside the code they were produced against, so
that a reader of the PR can reach the raw evidence without a separate handoff.

`main` + PR #151 = the deliverable. This branch = the deliverable **plus** the
evidence.

## What is here

| packup | what it covers | size |
|---|---|---:|
| `glm53flash.mxfp4.mix.packup_20260902` | **GLM-5.3-Flash-MXFP4, aggregated MIX.** The mission's first-priority model. Includes the shared-experts-fusion root cause and the fixed-length sweep. | 624K |
| `glm53flash.fp8.mix.packup_20260902` | **GLM-5.3-Flash-FP8, aggregated MIX.** Correctness, 8/8 AITER mHC lines, and the fusion **control arm** proving the gfx950 fusion path itself works. Also carries both HIP-IPC probe runs, including the cross-container one the PD work relies on. | 508K |

## Not here, and why

**The big-model work has no packup yet** — the alignment result (0.89–1.11× vs
the GLM-5.2 baseline across 8 matched points) and all three PD arms live in the
scratch workspace at
`/apps/yihou/glm53.series.workspace_20260901/bigmodel/`, not in a packed-up
directory. That is the largest result of the cluster and the largest gap in this
branch. It should be packed up before this work is considered closed.

**`glm53flash.mix.packup_20260830` is deliberately excluded.** It predates this
cluster and belongs to the earlier Flash bring-up, though this work depends on it
— it is the packup whose pin (`7fa1924c`) is cited in `.claude/CLAUDE.md` as the
fallback and as proof that a Flash overlay can serve. Include it if you want the
lineage; it is not a product of these experiments.

Other packups under `/apps/yihou/packups/` (`qwen36-27b.*`,
`glm52_kvcache_prefill.*`, `mix.{latency,stress}.*.spur`) belong to other tasks.

## One caveat that travels with the FP8 packup

Its two throughput numbers (99.70 / 456.68 tok/s) were **measured by the team
lead on the arm's own deployment**, not by the operator who owns the packup. That
operator declined to certify them, correctly, on the grounds that they did not
take the measurement and did not sample the neighbouring GPUs for contention
during it. The packup states this and keeps their name off those numbers; the
attribution should not be quietly simplified.

---
name: agent-sys-kernel-forge
description: Run, inspect, steer, resume or finish a GPU kernel optimization campaign with Hyperloom's embedded KernelForge, using an Infera operator_workset or an AKLR schema. Covers campaign preparation, monitoring, ablation, review and evidence handoff. Use after prior-work lookup for Forge kernel optimization; retain the parent task's integration and serving validation.
---

# Kernel optimization with Hyperloom Forge

Use the maintained Forge embedded in **AMD-AGI/Hyperloom**, at the pinned
`AKLR` revision in [sources.md](references/sources.md). The standalone
KernelForge repository and its `kernel-agents` interface are historical inputs.

Choose the input without changing its measurement contract:

| Input | Procedure | Authority for correctness and performance |
|---|---|---|
| Infera `operator_workset`, including example 03 and e2e-flow stage 4 | [workset.md](references/workset.md) | The workset's protected harness, then the parent task's independent validators |
| Native AKLR schema | [aklr.md](references/aklr.md) | Upstream `forge_task`, AKLR validation and its per-Point reports |

Both use [campaign.md](references/campaign.md) for preflight, watching, steering,
recovery and review. Read that and the selected procedure before launching.
Optimization alone does not establish a serving improvement: preserve the
parent task's engine integration, engagement checks and end-to-end comparison.

## External kernel knowledge

Read the pinned [AKLR KB entry](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/SKILL.md)
and follow its routing for the current operator, language and GPU. This is an
external reference: do not vendor its pages into Infera tasks, skills, examples
or handoffs. Pass the actual checkout's KB root and applicable language folder
with repeated `--skill-dir` on **every fresh invocation and resume**. A link in
the outer brief alone does not show that the inner Forge agents received it.

Record the AKLR revision, referenced paths and evidence of delivery. Keep
per-round matched references and unanswered questions in the campaign notes;
link to upstream pages rather than reproducing them. See the KB's
[language and architecture entrypoints](references/sources.md#external-kernel-knowledge).

## Deliver

Return the source revisions, input and driver identity, budget and actual run
state, per-case verdicts, review/ablation reports and the resulting commit or
patch. Keep Forge's search score separate from independently measured gains.
An unchanged, neutral or rejected kernel is a valid outcome; unavailable checks
remain explicitly unchecked. Knowledge-base edits and external publication are
separate work unless the invoking task already authorizes them.

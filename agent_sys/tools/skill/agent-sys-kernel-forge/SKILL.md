---
name: agent-sys-kernel-forge
description: Run, inspect, steer, resume or finish a GPU kernel optimization campaign with Hyperloom's embedded KernelForge, using an Infera operator_workset or an AKLR schema. Covers campaign preparation, monitoring, ablation, review and evidence handoff. Use after prior-work lookup for Forge kernel optimization; retain the parent task's integration and serving validation.
---

# Kernel optimization with Hyperloom Forge

Use **Hyperloom's embedded Forge** and preserve the input's measurement contract.

This skill is the maintained user entrypoint for Forge work in Infera. Read it
directly when optimizing a kernel, or use it as the kernel step of a parent task.
Provide the workset or native schema, assigned GPU, runtime, model/time budget
and expected deliverable; resolve missing allocation or budget facts before launch.

Shared campaign instructions, prompt templates and runtime helpers are owned by
this skill directory. The examples demonstrate its use; knowledge tasks compose
it into target selection, integration and serving evaluation. Those wrappers
own their input/output contracts and independent acceptance gates. They refer
here for Forge behavior rather than maintaining another campaign procedure.

1. **Resolve sources at session/mission start.** Read [sources.md](references/sources.md):
   use AKLR `main` and the Hyperloom revision that AKLR declares. Record the
   resolved revisions for this run. If an upstream skill moved, find its current
   entry and discuss the proposed replacement with the user before adopting it.
2. **Load only the matching input guide:**
   - Infera `operator_workset`, example 03 or e2e-flow stage 4:
     [workset.md](references/workset.md).
   - Native AKLR schema: [aklr.md](references/aklr.md), a short Infera adapter
     boundary; follow AKLR's linked workflow for its actual commands.
3. **Before launching, or when inspecting/resuming**, read [campaign.md](references/campaign.md)
   for the Infera monitoring, recovery and final-output requirements. Follow the
   resolved AKLR campaign skill for the shared lifecycle.

**You MUST consult the external [AKLR KB](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-kb/SKILL.md)
before starting a campaign**, using its resolved entry and the pages relevant to
the selected operator, language and GPU. Record the actual KB path/revision,
references consulted and how they affect the plan. Verify that the launch passes
those actual checkout paths to the inner agents with `--skill-dir` on fresh runs
**and resumes**; require their consultation/use evidence in the campaign record.
Path delivery alone is not consultation. If the KB cannot be consulted, resolve
its location or use the documented fallback before launching. Do not copy KB
pages into this skill or a handoff.

Deliver the final patch/commit, resolved source and driver identities, per-case
verdicts, review reports and unchecked gates. Forge's search score is separate
from independently measured gains; serving gains also require the parent task's
integration and end-to-end validation. Neutral or rejected results remain valid
outcomes. External publication and upstream KB edits follow the invoking task's
authorization.

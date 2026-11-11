---
name: agent-sys-kernel-forge
description: Run, inspect, steer, resume or finish a GPU kernel optimization campaign with Hyperloom's embedded KernelForge, using an Infera operator_workset or an AKLR schema. Covers campaign preparation, monitoring, ablation, review and evidence handoff. Use after prior-work lookup for Forge kernel optimization; retain the parent task's integration and serving validation.
---

# Kernel optimization with Hyperloom Forge

Use **Hyperloom's embedded Forge** and preserve the input's measurement contract.

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

Consult the external [AKLR KB](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-kb/SKILL.md)
for the selected operator, language and GPU. Pass its actual checkout paths to
the inner agents with `--skill-dir` on fresh runs **and resumes**. Record path
delivery and references used; do not copy KB pages into this skill or a handoff.

Deliver the final patch/commit, resolved source and driver identities, per-case
verdicts, review reports and unchecked gates. Forge's search score is separate
from independently measured gains; serving gains also require the parent task's
integration and end-to-end validation. Neutral or rejected results remain valid
outcomes. External publication and upstream KB edits follow the invoking task's
authorization.

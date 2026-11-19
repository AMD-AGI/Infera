# Campaign supervision in Infera

After resolving session sources, follow the selected AKLR checkout's
[campaign skill](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/forge-loop-campaign/SKILL.md)
for launch, steering, stop/resume, ablation and finalization commands. This
reference keeps only the Infera integration requirements and supervision deltas.

## Preflight

- Use the input's driver, target architecture and success criteria. Agree any
  missing model/time/resource budget before launching paid agents, including
  post-run review. Check runtime help for duration/profiling limits and label a
  reduced search as a smoke run.
- Run inside the intended ROCm container on an assigned idle card. Compare
  candidate and baseline on equivalent hardware; do not carry an MI300 result
  into a gfx950 claim. Keep all writable state on the task's approved storage.
- Use the session record's selected source paths and verify installed module,
  backend registry and lifecycle capabilities. Do not substitute an unrelated
  `kernel-agents` on PATH or install a different torch over the image's ROCm build.
- Pass the returned external `skill_dirs` on fresh runs and resumes. Check agent
  logs/prompt evidence for actual path delivery; a URL in the outer brief is not
  delivery to the inner agent. Record KB references and unanswered questions.

## Start and watch

Use an isolated git workspace; never optimize a sealed input or the serving
engine's stock arm. Record the owned process group/container and exact launch
command without credentials.

Before backgrounding, assign a duty teammate and establish a real timer for
**10-minute** checks, or the invoking user's interval. If no timer is available,
stay active for timed checks and say no unattended timer exists. Ending a
Claude Code turn does not create a wake-up.

Each check reads process/container liveness, log timestamps, round lessons,
dashboard and output files. Report changes, per-case numbers with their ruler,
unchecked gates and what is still alive at a stated time. A status label alone
is insufficient. Graph compilation may take about 30 minutes; inspect build
activity before diagnosing a quiet log as a stall. Use upstream
[report-progress](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/report-progress/SKILL.md)
when available, with the timer actually supported by this session.

## Stop and resume

Follow the resolved upstream stop/resume contract. Stop only the owned run and
its child agents; verify that no paid child remains. Preserve the source/session
record, driver, workspace and resume command. Reapply invocation-only settings,
including remaining budget and all external KB paths; fresh-only task arguments
must not be replayed as resume arguments. Prefer AKLR's generated resume wrapper
for native schemas. If a resumed crash repeats before any round completes,
leave it stopped and ask the user rather than retrying indefinitely.

## Review after the loop ends

Once the loop and children end, follow upstream ablation/finalization on the
kept change and read the reports. Preserve a no-improvement result when nothing
was kept. Use the actual configured git author and the target repository's
review/lint rules.

Measure the **final reviewed bytes**: workset mode refreshes its collected
kernel/patch before running independent validators; native mode uses AKLR's
stress, publish and per-Point validation. Never alter the driver, baseline or
cases to obtain a pass. Preserve historical measured evidence as historical.

Knowledge distillation is an optional upstream review proposal when requested,
not automatic publication or a copy of AKLR KB into Infera. Otherwise hand over
the referenced pages and unanswered questions.

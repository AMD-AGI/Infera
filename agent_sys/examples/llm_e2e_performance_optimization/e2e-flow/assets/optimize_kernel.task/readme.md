# optimize_kernel: compose the Forge skill with an Infera workset

Use the maintained
[`agent-sys-kernel-forge` skill](../../../../../tools/skill/agent-sys-kernel-forge/SKILL.md)
as the kernel-optimization procedure. Read its workset guide and campaign guide;
they own preparation, source/KB setup, execution, supervision, resume and review.
In this task, read `$PKG/assets/kernel_forge_skill/SKILL.md` and its matching
references; `PKG` is the staged package root defined below. The package embeds
the same skill through a relative directory link, so a global installation is
optional. In a direct session, use the installed skill or its migrated Infera
source. This brief defines only the adapter's inputs, output and acceptance.

## Inputs and runtime

- `operator_workset`: definitions, workloads, oracle, protected entrypoints,
  recorded baseline, mismatch policy and generated Forge adapter. Select exactly
  one operator; preserve all declared cases and the measurement protocol.
- `deploy_kit`: the environment record and source/engine context. Use the recorded
  image and assigned card. The transport records whether execution used the live
  container or an owned ephemeral one; preserve that distinction in the result.
- `AGENT_SYS_TASK_PACKAGE` (or `AGENT_SYS_DEMO_PACKAGE`): the actual staged package
  root, called `PKG` in the skill's workset guide. Adapter entrypoints are under
  `$PKG/assets/optimize_kernel.task/steps`.
- `E2E_WORK_ROOT` and `KFO_SCRATCH_ROOT`: approved writable local storage visible
  inside the measurement container. Keep a single task run directory throughout.
- `KFO_PYTHON`: the measurement-container interpreter. The control interpreter
  also needs the task dependencies, including PyYAML; configure
  `AGENT_SYS_DEMO_PYTHON` if necessary.
- The task's `forge_source_session`, source checkout paths, primary provider,
  model, time/round budgets and measurement-container variables are supplied by
  the parent graph. Resolve and validate them through the skill before launch.

The source/ABI adapter and handoff writer belong to this wrapper. Forge-specific
campaign helpers, shared preflight and the generated program template are owned
by the skill and embedded into package assets through relative links. Run the
skill's documented entrypoints; do not maintain a separate Forge command here.

## Measurement authority

The workset's recorded baseline is the denominator. Hardware and optimization
premise mismatches follow its abort policy; software mismatches follow its warning
policy. The premise gate's refusal must travel in the output, without a speedup
claim. Never replace the baseline, cases, oracle or acceptance thresholds to pass.

After a kept candidate's review, recollect the final source/patch and run the
workset's own correctness entrypoint on every case before its performance
entrypoint. Preserve its timing protocol and per-case RSD. Candidate hashes in
the test reports and delivered source must agree. The native Forge score is
search evidence; independent workset measurements decide the operator claim.
The parent serving integration and end-to-end comparison decide any serving gain.

Mock, smoke, no-improvement, abort and correctness-failure outcomes must be
labelled plainly. A no-improvement result can have an empty patch; a real run's
missing patch is incomplete. A mock cannot support an optimization claim.

## Output and acceptance

Publish one `kernel_optimization` content directory at
`AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION`. The adapter's handoff writer assembles
`items/codes/<operator>.packup_<date>/`, its apply manifest, final patch/source,
structured verdict and evidence. It derives ratios from the workset rather than
accepting a hand-authored search score as the claim.

Complete the generated packup documents: `README.md` with an honest Result,
`REPRODUCE.md` with actual commands and expected output, `environment.md` from
the recorded environment, and `notes.md` with findings and unresolved questions.
Also write the content-root README, with nonempty Purpose, Interface and Boundary
sections. Include source/KB references and use evidence, the reviewed candidate,
per-case measurements, review findings and unchecked gates; keep KB pages external.

Run the skill's adapter selfcheck, then let the parent graph apply
`check_environment`, `check_optimization_shape` and
`check_speedup_substantiated`. The independent validators remain authoritative.
On a failed gate, preserve the refusal and its evidence instead of substituting
numbers. Example03 stops at this operator result; e2e-flow's stage5 consumes it
for gated integration and serving comparison.

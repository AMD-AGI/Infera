# Native AKLR schema

This guide defines the Infera-to-AKLR boundary; it does not maintain a second
copy of AKLR's campaign manual. After source resolution from the skill entry,
follow the resolved checkout's authoritative
[Forge walkthrough](https://github.com/AMD-AGI/AKLR/blob/main/docs/forge_task.md),
[adapter README](https://github.com/AMD-AGI/AKLR/blob/main/harness/scripts/forge_task/README.md)
and [campaign skill](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/forge-loop-campaign/SKILL.md).

- Select the task's `aklr` mode and schema name/file. Do not convert it into an
  Infera workset. AKLR owns the schema, audit, ABI, determinism and timing rules.
- Prepare and run through upstream `forge_task` and its generated launch/resume
  wrappers, in a task-owned AKLR clone. The driver writes candidates there;
  concurrent campaigns need separate clones. Keep preflight evidence outside
  the empty run directory that prepare requires.
- Use the GPU and language dependencies required by the selected AKLR revision.
  Request assigned MI355 resources when its schema requires gfx950; an MI300
  smoke test cannot establish that schema's performance.
- Require the upstream stress and validation gates on the final reviewed bytes.
  Read the per-Point result with
  [aklr-read-validation](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-read-validation/SKILL.md);
  report unsupported Points and fallbacks instead of substituting a mean score.
- Hand Infera the published local schema/commit and the original AKLR reports.
  Do not run Infera's workset collector on a native AKLR workspace. Local
  `forge_task publish` does not authorize a push or pull request.

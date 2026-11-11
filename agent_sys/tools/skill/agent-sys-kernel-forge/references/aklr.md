# Native AKLR schema

Use the upstream adapter in the pinned AKLR checkout. Read its
[Forge task command contract](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/scripts/forge_task/README.md)
and [walkthrough](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/docs/forge_task.md).
The schema, audit, ABI, determinism and timing authority are AKLR's current
contract at that revision, not the Infera workset schema.

## Prepare

Complete [shared preflight](campaign.md#preflight). Native AKLR's documented
campaign targets gfx950; obtain an assigned MI355 from the user if unavailable.
Use one private AKLR clone per campaign and install AKLR plus the baseline and
language dependencies its README requires in the ROCm container. The driver
writes candidates into that clone, so concurrent campaigns must not share it.

Choose one schema and regime. If splitting workloads, keep original schemas
intact and write an explicit regime schema outside `aklr/ops`, using the path
shape the upstream adapter requires. Do not silently remove failing Points.
`RUN` below is a fresh run path; keep preflight evidence in a sibling task
directory so prepare receives an empty/nonexistent directory.
Use a backend supported by the pinned AKLR adapter (`flydsl`, `triton` or
`gluon`); Hyperloom's wider backend registry does not extend AKLR's adapter.

```bash
cd "$AKLR_REPO"
HIP_VISIBLE_DEVICES="$GPU" python harness/scripts/forge_task prepare \
  --schema "$SCHEMA" --run-dir "$RUN" --kernel-backend "$BACKEND"
```

Use `--schema-file` for an external regime schema. Non-GEMM operators require
`--starting-kernel` according to the adapter contract. Read `baseline.json`
before launch and identify workloads near the timer's floor using AKLR's own
measurement guidance; do not promise a visible gain there from another timer.

## Campaign and resume

Inspect the generated `forge_loop.sh`: it invokes Hyperloom Forge, declares the
kernel/driver/brief/backend and passes the AKLR KB paths, paired measurement and
the appropriate baseline anchor. The resume wrapper restores the saved task and
reapplies its invocation-only skill paths. Preserve those wrappers and use the
same agreed runtime interpreter, model, assigned GPU and remaining budget.

```bash
"$RUN/forge_loop.sh" --max-hours "$MAX_HOURS" --model "$FORGE_MODEL" \
  --gpu "$GPU" --lanes 1 --no-experience-kb --no-kb-warmstart --no-pr-kb
```

Follow [shared watch/steer/stop/recovery](campaign.md#start-and-watch). After
archiving the owned stop marker, resume with `forge_loop_resume.sh` and the
remaining budget; do not reuse fresh-only options. The `--no-` flags concern
Forge's shared stores: they do not disable the explicit external AKLR KB.

## Review, validate and publish locally

Wait for the campaign to end, then run the shared ablation/finalization steps.
Use AKLR's lint configuration. For language-specific review guides, follow the
upstream command contract but keep external AKLR KB pages as references: do not
copy them into this skill or its output. If a review command edits its guide
directory, do not give it the authoritative KB directory as an editable target.

The native driver's concurrency stress gate is required after finalization:

```bash
HIP_VISIBLE_DEVICES="$GPU" python "$RUN/workspace/driver.py" --stress
```

Require `stress: PASS`; a failed or hung check blocks local publication. It is
separate from validation on an idle GPU. Then, for an authorized kernel
delivery into the private AKLR clone:

```bash
python harness/scripts/forge_task publish "$RUN"
python -m aklr export check
python -m aklr validate --base origin/main "$PUBLISHED_SCHEMA" --out "$VALIDATION_DIR"
```

`PUBLISHED_SCHEMA` is the actual path printed by publish. Name that file
explicitly; an unscoped validation can test the whole repository. Read the
per-Point report with AKLR's
[aklr-read-validation](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-read-validation/SKILL.md)
procedure. Distinguish correctness/build findings, timing uncertainty and
dispatch decisions; a mean score does not establish every Point's verdict.

Return the schema, supported Points, fallbacks, source revisions, stress and
validation evidence, reviewed commit and ablation findings. Review provenance,
public execution-path reachability and measurement fairness following the
[AKLR integration/review references](sources.md#workflow-sources-and-relationships).
Local publish is not a push or pull request. Follow the invoking task's
authorization for external publication and any optional upstream KB proposal.

# Sources and session setup

## Select current sources once

At session/mission start, resolve **AKLR `main`**, then read that revision's
[Forge setup](https://github.com/AMD-AGI/AKLR/blob/main/docs/forge_task.md)
and [adapter instructions](https://github.com/AMD-AGI/AKLR/blob/main/harness/scripts/forge_task/README.md).
Use the Hyperloom commit, tag or branch they declare. When AKLR declares a branch,
resolve it to a commit and record both the declaration and actual SHA. The
inspected AKLR instructions declare Hyperloom's **`AKLR` branch**, not an
immutable Hyperloom hash.

Keep the resolved pair and source paths in one session record; reuse it during
the campaign and resume. Do not fetch a new pair for each round or silently
upgrade an existing campaign's state. `hyperloom_repo` and `aklr_repo` are source
locations, not permission to reset a shared checkout.

For cross-UID source checks, the helper uses an anonymous per-command Git
configuration overlay that trusts only the selected checkout and includes the
caller's existing global configuration. It writes no persistent trust settings
and adds no wildcard trust; this path requires Linux memfd support.

In the campaign container, with `RUN` an existing task metadata directory:

```bash
SESSION="$RUN/forge_sources.json"
python "$FORGE_SKILL/scripts/check_forge.py" \
  --hyperloom "$HYPERLOOM_REPO" --aklr "$AKLR_REPO" --backend "$BACKEND" \
  --resolve --session "$SESSION" --out "$SESSION"
```

The helper resolves session sources into owned storage when needed. **Use the
returned `hyperloom_repo`, `aklr_repo` and `skill_dirs`**, which may differ from
the supplied locations. Install the selected Hyperloom checkout with
`pip install -e "<selected Hyperloom>[forge]"`, then rerun the helper with the
same `--session "$SESSION"` for installed-module and command-capability checks.
Use `--installed-source <copy>` if the task installs an editable scratch copy;
preflight verifies its tracked source bytes and the imported module's origin.
Always name that same session file when writing a separate runtime report.

An enclosing knowledge task with a generated launcher can delegate to this
skill's `scripts/run_entrypoint.sh`. Supply `INFERA_REPO`, the selected
`HYPERLOOM_REPO` and `AKLR_REPO`, `FORGE_BACKEND`, absolute `FORGE_ENTRYPOINT`,
and `FORGE_PREFLIGHT_JSON`; use `FORGE_SESSION_JSON` for a separate existing
session record. For an editable copy, set `FORGE_INSTALLED_SOURCE` to the same
copy used with `--installed-source`. When unset, the launcher reuses
`installed.source_path` from an existing `FORGE_PREFLIGHT_JSON` report, or its
session record, reading it before either file can be overwritten. Every launch
revalidates that identity through preflight; a saved path does not bypass the
byte/module-origin checks or trust an arbitrary installed module. The helper
forwards the verified source/KB paths and `FORGE_INSTALLED_SOURCE` to the generated
launcher without changing fresh/resume arguments. It does not replace preparation above.
Knowledge-repo `current_forge.sh` files are compatibility delegates to it.

## If upstream changed or cannot be read

If a referenced skill is missing, search the resolved AKLR tree by its purpose,
frontmatter and callers; follow symlinks and plugin manifests. Report the old
path, candidate replacement and relevant behavior changes to the user, and
suggest adopting the new entry. **Get the user's choice before using a moved
entry**; a matching folder name alone is insufficient. For a relocated KB, pass
the approved path as `--accept-kb-path <repository-relative path>`.

Use the suggested snapshots below only when AKLR is inaccessible or discovery/
adaptation fails. Record that fallback, its reason and actual selected revisions;
do not present it as current AKLR main. If it also fails capability checks,
stop with the missing contract rather than inventing commands.

For an adaptation failure the helper cannot detect itself, select the suggested
pair explicitly with a reason and a **new** session record:

```bash
python "$FORGE_SKILL/scripts/check_forge.py" --resolve \
  --use-suggested "<what could not be accessed or adapted>" \
  --session "$RUN/fallback_sources.json" --out "$RUN/fallback_sources.json"
```

Supply existing `--hyperloom` and `--aklr` checkouts when they contain cached
fallback objects needed offline. This does not reset either caller checkout.

| Revision | Repository and role |
|---|---|
| `5d59c935e7e47410e1f84bc4c2a5077816d4071d` | Suggested **AKLR** fallback snapshot: workflow, adapter and externally referenced KB |
| `fa94e001b88d77e69d8d50973c868a8f4a11a15d` | Suggested **Hyperloom `AKLR`** fallback: embedded Forge compatible with that snapshot |
| `d265d9b85d09cfe25b0996712367559aac206651` | Previously inspected **Hyperloom `main`**, recorded for comparison; not a selected default or fallback |

These are revisions of two source repositories. None is an AKLR benchmark
baseline: each schema's contract and validation run define its baseline.

## Load upstream guidance for the current step

Use these live links to locate the current entry; once sources resolve, read
the corresponding files in that recorded checkout for a consistent campaign.

| When needed | Authority |
|---|---|
| Native schema preparation and local publication | [forge_task walkthrough](https://github.com/AMD-AGI/AKLR/blob/main/docs/forge_task.md) and [adapter README](https://github.com/AMD-AGI/AKLR/blob/main/harness/scripts/forge_task/README.md) |
| Launch, watch, steer, stop/resume, ablate and finalize | [forge-loop-campaign](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/forge-loop-campaign/SKILL.md) |
| Periodic evidence and liveness reports | [report-progress](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/report-progress/SKILL.md) |
| Native per-Point verdicts | [aklr-read-validation](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-read-validation/SKILL.md) |
| Kernel delivery/review | [aklr-port-generated-kernels](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-port-generated-kernels/SKILL.md), [aklr-review-kernel-pr](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-review-kernel-pr/SKILL.md), [aklr-bad-cases](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-bad-cases/SKILL.md) |
| Operator, language or GPU knowledge | [aklr-kb](https://github.com/AMD-AGI/AKLR/blob/main/harness/skills/aklr-kb/SKILL.md); follow its own routing |

The KB is external: use its checkout paths and link the pages used; do not copy
them. AKLR's native driver owns its audit/timer; Infera worksets retain their
protected harness and independent validators. Resource/image/release skills are
conditional on the task, not prerequisites of every local campaign.

Adapted from AKLR under its [MIT license](https://github.com/AMD-AGI/AKLR/blob/main/LICENSE); the required notice is retained in [NOTICE](../NOTICE).

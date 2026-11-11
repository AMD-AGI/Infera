# Shared campaign operations

Adapted from AKLR's `forge-loop-campaign`, `report-progress` and Forge task
walkthrough; source identities and license are in [sources.md](sources.md).

## Preflight

1. Name the input, immutable driver, kernel edit target, target GPU and success
   criteria. Read the actual baseline before spending agent/GPU hours. Separate
   decode and prefill regimes when one objective would hide their different
   bottlenecks; preserve the original input and declare any subset explicitly.
2. Reuse the invoking task's model, budget and resource authorization. If absent,
   settle the model and time/cost budget with the user before launching agents.
   The pinned CLI enforces a one-hour floor; budgets at most two hours disable
   profiling-driven Analysis and single-lane Plan Critic. Label these smoke runs
   rather than interpreting no improvement as a full search result.
3. Run inside the target ROCm container on an assigned idle GPU. Inspect GPU
   processes and memory before use; use the same card for baseline and candidate.
   Check the workset/schema architecture. MI300 is suitable for compatible local
   smoke tests; native AKLR's documented gfx950 run needs MI355 resources from
   the user. Do not substitute hardware while retaining old baseline claims.
4. Give each campaign its own writable workspace; native AKLR also needs its own
   AKLR clone because its driver writes candidates into `aklr/ops`. Keep source,
   caches, compiler output and logs under storage allowed by the user's rules.
   Check the mount and capacity first; do not use an implicit system temp path.
5. Set `HYPERLOOM_REPO` and `AKLR_REPO` to the pinned external checkouts in
   [sources.md](sources.md). Confirm a clean referenced runtime/KB. In the
   campaign interpreter, run this skill's `scripts/check_forge.py`:

   ```bash
   python "$FORGE_SKILL/scripts/check_forge.py" \
     --hyperloom "$HYPERLOOM_REPO" --aklr "$AKLR_REPO" \
     --backend "$BACKEND" --source-only --out "$RUN/source-preflight.json"
   python -m pip install -e "$HYPERLOOM_REPO[forge]"
   python "$FORGE_SKILL/scripts/check_forge.py" \
     --hyperloom "$HYPERLOOM_REPO" --aklr "$AKLR_REPO" \
     --backend "$BACKEND" --out "$RUN/runtime-preflight.json"
   ```

   `FORGE_SKILL` is the actual directory of this installed/read skill; `RUN`
   must already be an owned task directory for these evidence files. Native
   `forge_task prepare` instead receives a separate empty run directory. No
   `kernel-agents` fallback: the check verifies source pins, import origin,
   backend registry, required lifecycle commands and `--skill-dir` support.
   If the task installs an editable scratch copy, pass its path as
   `--installed-source`; the helper checks it against the pinned source bytes.
6. Deliver the **external** AKLR KB root and matching language folder listed in
   the preflight's `skill_dirs`. Use repeated `--skill-dir` flags and include
   the same flags on resume. For Gluon, the language folder is `triton`.
   Check agent logs or prompt evidence for the actual paths. Retain the KB's
   own lookup protocol and record references/unanswered questions, without
   copying its content into the handoff.

The helper makes no installation, launch, GPU allocation or publication. Do not
install torch over the ROCm image's torch. Profiling dependencies are Hyperloom's
separate `forge-profiling` extra, only when the selected image needs them.

## Start and watch

Follow the selected input's launch procedure. Give Forge an isolated git
workspace containing the intended editable files; never the parent task's
sealed input or the serving engine's stock arm. Use an explicit
`--kernel-backend`; the pinned CLI silently falls back for unsupported names,
so verify the registry before launch.

Keep launch, resume and steering commands in the run directory with no secrets.
For an unattended launch use a dedicated process group, tmux session or container
owned by this run, and record its real PID/group or container identity. Confirm
that the recorded handle belongs to the run before using it to stop anything.

Before backgrounding, assign a duty teammate and establish an available timer
to inspect the run every **10 minutes**. Record its handle and evidence paths.
If timer tools are unavailable, remain active and perform timed checks; report
that no unattended timer exists. Ending a turn does not create a wake-up.

At each check read the process/container status, log tail and timestamps,
`forge_experiments/dashboard.html`, per-round lessons and actual output files.
Long graph compilation can take about 30 minutes: inspect build/cache activity
before classifying a quiet log as a stall. Report measured progress, per-case
baseline/candidate results with their ruler, completed gates, unchecked items,
what is still alive at a stated time and the next action. A RUNNING label or PID
file alone is not liveness evidence. Report status changes promptly.

Steer from new evidence, with a focused direction and relevant external KB path:

```bash
python -m kernelforge.cli forge-say --workspace "$W" "$DIRECTION"
```

For an unexpected crash, inspect its cause and preserve the state before
resuming with the remaining budget. If the resumed attempt crashes again before
completing a round, leave it stopped and ask the user; do not loop indefinitely.

## Stop and resume

`touch "$W/.stop"` requests a stop after the round. An immediate stop must end
this run's owned process group/container, including its child agents. Killing
only the coordinator may leave paid agents running. Verify ownership and scope
before signaling; confirm afterward that this run has no remaining children.
Leave other users' processes and containers alone.

For resume, preserve a copy of the prior launch record and archive the owned
`.stop` marker within allowed storage. Use the exact existing workspace and
`--resume`; do not pass fresh-only `--kernel`, `--driver`, `--program-md-file`,
`--kernel-backend` or `--git-branch`. Reapply invocation-only settings: model,
remaining time, GPU, knowledge flags, all `--skill-dir` paths and any scoring
anchor/paired-keep policy the selected mode requires. Native AKLR's generated
`forge_loop_resume.sh` preserves its anchor and skill paths; inspect it before
use. Do not turn a completed or intentionally stopped run into an auto-retry.

## Review after the loop ends

Confirm the loop and children have ended, then price the kept rounds and review
the whole result in its isolated workspace:

```bash
python -m kernelforge.cli forge-ablate --workspace "$W" --gpus "$GPU"
python -m kernelforge.cli forge-finalize --workspace "$W" --gpu "$GPU" \
  --model "$FORGE_MODEL" --author "$AUTHOR"
```

`AUTHOR` is the workspace's real configured git identity. Add the target
repository's lint commands and suitable review guides when applicable. Several
ablation GPUs are appropriate only when assigned and idle. These commands spend
time beyond the search budget; include review in the agreed resource budget.
Read `ablation.md` and `finalize.md`; a green command alone is not review evidence.
If there is no kept change, report that outcome without inventing an ablation
gain or delivery commit.

Use the selected mode's independent checks on the **final reviewed bytes**.
For native AKLR, include its `--stress` and validation. For worksets, use their
own correctness/performance scripts and the parent's validators; do not assume
their driver implements AKLR's `--stress` option.

Read findings as evidence: per-case failures, correctness, actual fast-path
engagement, fallback behavior, timing noise and environment equivalence. Do not
change the driver, contract, baseline or case list to make a candidate pass.
Keep historical measurements and hashed handoffs intact, with a current
migration pointer rather than relabeling them as new runs.

`forge-distill` is an optional **upstream AKLR knowledge proposal** when that
feedback is requested. Its output is a review patch, never automatic knowledge
publication and never an excuse to vendor AKLR KB pages into Infera. Link the
proposal and its evidence; follow AKLR's `CONTRIBUTING.md` for any authorized
upstream edit. Otherwise hand over matched references and unanswered questions.

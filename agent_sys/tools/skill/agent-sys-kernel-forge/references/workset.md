# Infera operator_workset

Keep the existing Infera workset format. AKLR's `forge_task` adapts **AKLR
schemas** and must not reinterpret an Infera workset or replace its measurement.

## Prepare and run

Use the exported workset and the enclosing task's input/output contract. This
reference owns the workset campaign procedure. In the minimal example,
`examples/03_kernelforge_one_kernel` publishes the fixed workset and invokes
e2e-flow stage 4. Stage 3's `assets/lib/forge_export.py` generates the Forge
adapter from the workset's definitions and workloads; do not hand-edit a generated
`run_forge.sh`, driver or case list. The main `workset.yaml` remains the input
authority and the workset's harness remains protected.

Run the parent package with `agent-sys-task-runner` or agent_sys, supplying the
session-selected `hyperloom_repo` and `aklr_repo` paths and
`--var forge_source_session=<the same session file visible in the container>`.
This preserves the outer preflight's choice, including an explicit fallback.
The task defaults to `forge_agent_backend=claude`; set the provider and matching
`forge_model` explicitly for another client. `forge_max_rounds=0` leaves rounds
uncapped within `forge_max_hours`; a small round cap makes a bounded real smoke
run. On manual resume, repeat the provider/model, intended total round cap and
external KB paths as required by the selected runtime. Resume retains the
campaign's original primary provider, which can differ from an active fallback.
Upstream may still probe other relay members; an unavailable backup is recorded
separately from a failed campaign.
The task's `assets/optimize_kernel.task/readme.md` supplies its IO, environment
and acceptance contract. Its `steps/30_run_forge.sh` is an embedding of this
skill's `scripts/workset/30_run_forge.sh`: it stages source, checks the runtime
and launches the generated `forge.one_line`. Shared helpers and preflight code
are maintained here and linked into package assets; package staging materializes
those links. Use those entrypoints to preserve source containment and mock/no-claim
checks. The task's source/ABI adapter and handoff writer remain task-specific.
Both examples also embed this skill as `assets/kernel_forge_skill`, so the
staged task can read its entry and references without a global installation.
For a direct call to a canonical `scripts/workset` helper, supply the task's
adapter directory with `--adapter-steps "$S"` (or `KFO_ADAPTER_STEPS`). The
materialized task entrypoints retain their existing invocation interface.

Before launch, inspect the generated one-liner's `--dry-run` command and verify:

- `python -m kernelforge.cli forge-loop` resolves to the session-selected Hyperloom source;
- `--workspace` is the isolated engine-source copy, and `--kernel` is a file
  inside that checkout in the correct path frame;
- `--kernel-backend` matches the actual editable language, without `-fellow`;
- the protected driver's cases equal the workset's workload cases;
- repeated `--skill-dir` paths point to the selected external AKLR KB and the
  appropriate language entry, with evidence that the inner agents received them.

The exporter also writes `forge.program` (`forge_program.md`), a short original
Infera brief with the protected measurement contract and mandatory explicit KB
consultation/use evidence. The generated fresh launcher passes it through
`--program-md-file`; Hyperloom snapshots that context immutably and reuses it on
resume, so the launcher omits the fresh-only flag then. An explicitly supplied
caller brief replaces the default without duplicate program flags; the caller
must retain the measurement contract and KB consultation requirements. Delivery
of program context or inlined skill text is not evidence of actual page reads:
inspect the round's tool reads and matched/use or precise no-match record.

The maintained templates are this skill's
[`workset_program.md`](../assets/workset_program.md),
[`workset_launcher.sh.in`](../assets/workset_launcher.sh.in) and
[`workset_driver.py.in`](../assets/workset_driver.py.in). The example's exporter
populates them from workset definitions and workloads. Exported briefs, launchers
and drivers are generated outputs; update the canonical template or workset IO
adapter and regenerate them.

Do not reuse the stock serving checkout as Forge's workspace: the downstream
stock/candidate comparison needs an unchanged stock arm. Do not change a sealed
input while optimizing. The generated adapter's graph timing is Forge's search
signal; the workset's declared measurement mode remains the claim authority.

## Execute the Infera task adapter

This adapter consumes `deploy_kit` and `operator_workset` and publishes
`kernel_optimization`. In a task, let `PKG` be the actual staged package root
(`AGENT_SYS_TASK_PACKAGE`, or `AGENT_SYS_DEMO_PACKAGE`) and set:

```sh
S="$PKG/assets/optimize_kernel.task/steps"
W="${E2E_WORK_ROOT:?}/m4/$(date -u +%Y%m%dT%H%M%S)"
TMPDIR="${TMPDIR:-${KFO_SCRATCH_ROOT:?}/tmp}"; export TMPDIR
CONTROL_PYTHON="${AGENT_SYS_DEMO_PYTHON:-python3}"
mkdir -p "$W/state" "$W/forge" "${TMPDIR:?}"
```

Keep this same `W` for the whole run. Use task-approved local storage visible
inside its container. The control interpreter needs the task's Python
dependencies, including PyYAML. `KFO_PYTHON` names the interpreter inside the
measurement image; if it does not exist on the control side, select the control
interpreter with `AGENT_SYS_DEMO_PYTHON`.

Read and gate the inputs before spending campaign resources:

```sh
"$CONTROL_PYTHON" "$S/10_read_inputs.py" --out "$W/state/inputs.json"
"$CONTROL_PYTHON" "$S/20_premise_gate.py" --inputs "$W/state/inputs.json" --out "$W/state/premise.json"
```

Select one operator with `10_read_inputs.py --operator <id>` if needed. Require
`premise.held: true`. On a premise abort, skip the campaign and measurements;
write a handoff carrying the refusal with no performance claim. The workset's
recorded baseline and mismatch policy remain authoritative.

Run the campaign inside the intended image. From outside it, use the task's
existing transport:

```sh
"$S/run_in_container.sh" --workdir "$W" \
  "$S/30_run_forge.sh --inputs '$W/state/inputs.json' --workdir '$W/forge'"
```

Use paths visible to the measurement container. When already inside, invoke
the step directly. The transport uses the recorded live container when
available, otherwise an owned ephemeral container from the recorded image.
Preserve `observed_runtime.mode`; an ephemeral result carries the image state.
Respect the assigned card and the container's pin. Record its command and
source/KB preflight, then supervise using [campaign.md](campaign.md).

Require `forge_result.json` and, for a real run, an existing `engine.patch`.
An empty patch and `improved: false` are valid no-improvement results; a missing
patch is an incomplete run. `KFO_MOCK=1` creates a labelled incumbent-only
result and cannot support an optimization claim. A reduced budget must be
labelled as a smoke run. Finish any kept candidate with the review procedure
below before independent measurement.

After review and recollection, invoke these steps through the same transport
(or directly inside the measurement container), stopping on a failed gate:

```sh
"$S/40_correctness.sh" --inputs "$W/state/inputs.json" \
  --candidate "$W/forge/optimized_kernel.py" --out "$W/state/correctness.json"
"$S/50_performance.sh" --inputs "$W/state/inputs.json" \
  --candidate "$W/forge/optimized_kernel.py" --out "$W/state/performance.json"
"$S/60_write_handoff.py" --inputs "$W/state/inputs.json" --state "$W/state" \
  --out "$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION"
"$S/70_selfcheck.sh" --handoff "$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION"
```

Correctness must pass every declared case before timing. Keep the workset's
protocol, all performance cases and per-case RSD. On a correctness failure,
skip timing and write the failure verdict. The handoff writer derives ratios
from the recorded baseline; do not replace them with the search score.

Complete its generated `README.md`, `REPRODUCE.md`, `environment.md` and
`notes.md`, and write the content-root README required by the handoff contract.
Label mock, smoke, abort and no-improvement outcomes plainly. Include actual
commands, source/KB identity/use, final byte hashes, per-case results, review
findings and unmeasured boundaries. Selfcheck does not replace the independent
validators or a parent task's serving evaluation.

## Finish and deliver

Run ablation and final review in the same selected image and runtime. If the
ephemeral container changed, reinstall and verify the session-selected Forge
before running those commands. Inside the post-review container, set
`TASK_RUN_ROOT` to the existing task run root (`W`) and create a fresh CLI temp
directory before each review command group:

```bash
CLAUDE_CODE_TMPDIR=$(mktemp -d "$TASK_RUN_ROOT/forge/yihou-claude-review.XXXXXX")
export CLAUDE_CODE_TMPDIR
```

Never reuse a prior container's chowned Claude CLI cache.

After the resolved upstream campaign's ablation and review, refresh the workset
deliverables. Finalization can change the kept source after stage 30 first copied
the candidate and patch, so refresh those deliverables before measuring or
writing a handoff. In the same task container and environment as stage 30:

```bash
"$KFO_PYTHON" "$PKG/assets/optimize_kernel.task/steps/collect_forge_output.py" \
  --inputs "$TASK_RUN_ROOT/state/inputs.json" --workdir "$TASK_RUN_ROOT/forge"
```

`PKG` is the e2e-flow package used by the task; `TASK_RUN_ROOT` is the task's
local-disk run root (named `W` in its brief), not the Forge git workspace.
Pass these through the task's existing `run_in_container.sh` transport when
invoking from outside the container. The collector regenerates
`optimized_kernel.py` and `engine.patch` from `forge/engine_src` against the
recorded baseline, and refuses protected harness changes. Do not reuse stale
pre-review copies.

Then rerun correctness and performance on the final reviewed source through
the workset's entrypoints. Have the parent task write and validate its
`kernel_optimization` or `kernel_campaign_result` handoff in its existing format.

The Forge native summary, ablation and finalize reports are useful search/review
evidence. They do not replace independent baseline/candidate remeasurement,
source and GPU equivalence checks, or proof that the intended kernel executes.
Any serving gain additionally needs the parent's gated integration and
end-to-end comparison. An unchanged or rejected campaign must not acquire a
performance claim from a mock fixture or copied historical result.

Include runtime/KB preflight records, actual KB path delivery, source revisions,
commands, final patch/commit, per-case measurements, review verdicts, unchecked
gates and KB questions. Refer to historical examples for their lessons while
retaining their original code, environment and measurements as historical.

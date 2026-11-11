# Infera operator_workset

Keep the existing Infera workset format. AKLR's `forge_task` adapts **AKLR
schemas** and must not reinterpret an Infera workset or replace its measurement.

## Prepare and run

Use the enclosing task's brief and exported workset. In the minimal example,
`examples/03_kernelforge_one_kernel` publishes the fixed workset and invokes
e2e-flow stage 4. Stage 3's `assets/lib/forge_export.py` generates the Forge
adapter from the workset's definitions and workloads; do not hand-edit a generated
`run_forge.sh`, driver or case list. The main `workset.yaml` remains the input
authority and the workset's harness remains protected.

Run the parent package with `agent-sys-task-runner` or agent_sys, supplying the
pinned `hyperloom_repo` and `aklr_repo` paths. Read the actual
`assets/optimize_kernel.task/readme.md` before using its steps. Its real-run
`steps/30_run_forge.sh` stages source, installs/checks the runtime and launches
the operator's generated `forge.one_line`; it also maps native results into the
parent task's evidence format. Use that path instead of bypassing its environment,
source containment, handoff and mock/no-claim checks.

Before launch, inspect the generated one-liner's `--dry-run` command and verify:

- `python -m kernelforge.cli forge-loop` resolves to the pinned Hyperloom source;
- `--workspace` is the isolated engine-source copy, and `--kernel` is a file
  inside that checkout in the correct path frame;
- `--kernel-backend` matches the actual editable language, without `-fellow`;
- the protected driver's cases equal the workset's workload cases;
- repeated `--skill-dir` paths point to the pinned external AKLR KB and the
  appropriate language entry, with evidence that the inner agents received them.

Do not reuse the stock serving checkout as Forge's workspace: the downstream
stock/candidate comparison needs an unchanged stock arm. Do not change a sealed
input while optimizing. The generated adapter's graph timing is Forge's search
signal; the workset's declared measurement mode remains the claim authority.

## Finish and deliver

Apply [shared review](campaign.md#review-after-the-loop-ends) to the campaign
workspace. Finalization can change the kept source after stage 30 first copied
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

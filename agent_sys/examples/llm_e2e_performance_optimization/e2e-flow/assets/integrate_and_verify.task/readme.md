# `integrate_and_verify`

Put m4's optimised kernel in front of the real service, measure the same things
on an unpatched and a patched deployment, and write down whether the change is
safe to keep.

**One task, five leaves' worth of work.** `integration-demo` split this across
`serve_stock → measure_stock → serve_patched → measure_patched → compare`.
Mission M5.2 forbids that split — *"agent A 去把服务部署好，agent B 去使用：这是
不被允许的"* — so bring-up and use live here together.

## What that costs you, specifically

Three of those five edges carried an **argument** rather than a datum. The one
that mattered was `serve_patched ← measure_stock`, and it did not exist because
the patched bring-up needed the stock arm's numbers: it existed because **the
patched bring-up's first act is to destroy the stock deployment**, and wiring it
to `serve_stock` instead would have let the scheduler start the teardown while
the stock arm was still being measured. That edge is gone.

What replaces it is two things, and you are responsible for both:

1. **Step 6 below refuses to run** unless the stock arm's `steps.json` exists.
2. **`check_measurement_order` refuses the pair afterwards** if the arms overlap
   in time, if they ran different sequences, or if any step exited non-zero.

So the ordering below is not a suggested order. It is the measurement.

## Before you start

- **You are on the login node.** Everything touching a GPU goes through
  `assets/lib/remote.sh`, which dispatches on `$E2E_TRANSPORT`. Do not spell a
  transport yourself; do not `ssh`.
- **Two containers, and this stage is the one place in the flow allowed two**
  (CONTRACT.md §5, granted by mission G5.1). A container holds one state for its
  life, which is the whole reason the two-arm design exists.
- **Never `docker rm -f` a name you did not create.** Both held nodes are
  carrying other tenants' containers right now. Your two names are derived from
  `$E2E_CONTAINER`; touch nothing else.
- Write everything into the attempt zone (`$PWD`). The handoff directories are
  composed at step 9, not written into as you go.

## STEPS

Run these in order. Each names the command and what says it worked. Where a step
says **ABORT**, stop and report; do not carry on with a broken arm, because a
half-measured arm compared against a whole one is worse than no comparison.

### 0. Mock, if this stage is mocked

```sh
bash "$AGENT_SYS_TASK_PACKAGE/assets/lib/mock_m5.sh" arms \
     "$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml"
bash "$AGENT_SYS_TASK_PACKAGE/assets/lib/mock_m5.sh" report \
     "$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml"
```

**Only when `$E2E_MOCK_STAGES` contains `all`, `m5` or `stage5-integration`.**
Both commands print what they wrote. Then stop — steps 1 to 10 do not run.

*Accepted:* three output directories are non-empty and
`stock.measurement/items/env/steps.json` names six steps beginning with `serve`.

### 1. Read the overlay plan and the environment

```sh
cat "$AGENT_SYS_INPUT_PATCH_OVERLAY/items/result/mounts.json"
cat "$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml"
```

*Accepted:* `mounts.json` lists at least one mount, and every entry's
`sha256_stock` differs from its `sha256_patched`. If they are equal, **ABORT**:
the two arms would be byte-identical and every check downstream would pass for
the wrong reason. (`check_overlay_applies` has already refused this upstream, so
seeing it here means something changed under you.)

### 2. Bring up the **stock** arm

Record the wall-clock start first — step 9 needs it and there is no default:

```sh
SERVE_STOCK_T0=$(date -u +%Y-%m-%dT%H:%M:%S+00:00); SERVE_STOCK_S=$(date +%s)
E2E_ARM=stock \
E2E_OUTPUT_DIR="$PWD/arm.stock/deployment" \
E2E_CONTAINER="${E2E_CONTAINER}_stock" \
  bash "$AGENT_SYS_TASK_PACKAGE/assets/serve/round.sh"
SERVE_STOCK_SECONDS=$(( $(date +%s) - SERVE_STOCK_S ))
```

*Accepted:* exit 0, and `arm.stock/deployment/items/env/deployment.json` says
`"arm": "stock"` with `overlay.applied == 0`. A stock arm that mounted the
overlay is not a control.

*Watch out:* bring-up takes minutes and the log repeats `Health check failed`
throughout. That is the engine compiling, not a hang. `round.sh` decides when to
give up; do not shorten its wait.

### 3. Measure the **stock** arm

```sh
E2E_ARM=stock \
E2E_OUTPUT_ACCEPT="$PWD/arm.stock/accept" \
E2E_OUTPUT_BENCH="$PWD/arm.stock/bench" \
E2E_CONTAINER="${E2E_CONTAINER}_stock" \
  bash "$AGENT_SYS_TASK_PACKAGE/assets/accept/measure.sh"
```

The five steps inside — `smoke`, `needle`, `probe`, `lm_eval`, `bench_r<N>` — run
in a fixed order and are timestamped into `items/env/steps.json`. **Do not
reorder them and do not run any of them twice**: "round 1 is cold against this
trace" is only true if the same things happened before it on both arms.

*Accepted:* exit 0, and `steps.json` records all five with `rc: "0"`.

*If one step fails:* re-run **that step only** if the failure is clearly
environmental (a timeout on a saturated node), and say so in the handoff's
`watchout`. A step re-run changes what preceded the steps after it, so if you
re-run anything before `bench_r1`, re-run the rest of the arm too.

### 4. Generate the ad-hoc correctness cases (M5.4)

The frozen suite is in `assets/accept/` and ships with the package, which means
it can be satisfied by construction. `$E2E_ADHOC_CASES` cases are invented per
run so that the set cannot be prepared for — 免得作弊.

Write `arm.stock/accept/items/result/adhoc.json`:

```json
{"generated_at": "<ISO8601>",
 "generator": {"model": "<your model id>",
               "prompt": "<the exact instruction you gave yourself to invent these>"},
 "cases": [{"id": "adhoc-1",
            "prompt": "<what you sent to the endpoint>",
            "expectation": "<what a correct answer must contain or satisfy>",
            "answer": "<what came back>",
            "ok": true}]}
```

Send each case to `http://$E2E_NODE_IP:$E2E_PORT_ROUTER/v1/chat/completions`
through `remote.sh`'s `on`, the same way the frozen checks do.

**Four rules, each closing a way this requirement could be met without being
met**, and `check_acceptance` enforces all four:

- at least `$E2E_ADHOC_CASES` cases;
- the **generator prompt is recorded** — what was asked is half the evidence,
  and without it nobody can tell a hard case that was answered from an easy one
  that was substituted;
- no case repeats a frozen one, and none repeats another;
- **the patched arm runs the same cases with the same ids.** Different cases per
  arm is not a comparison, and it is the shape a regression could hide behind.

Make them checkable rather than clever: an arithmetic identity, a constrained
format, a fact stated in the prompt and asked back. A case whose correctness you
cannot decide from the answer text is a case that will be scored by opinion.

### 5. Tear the stock deployment down

```sh
bash "$AGENT_SYS_TASK_PACKAGE/assets/serve/reset_gpus.sh"
```

*Accepted:* VRAM returns to baseline on all eight cards. `reset_gpus.sh` kills
only processes that look like leftover inference workers and explicitly protects
the scheduler and the container runtime — **do not replace it with a broader
kill**, and do not proceed while memory is still held: a worker started beside a
leftover one fails bootstrap with `memory capacity is unbalanced`, which says
nothing about the real cause.

### 6. Bring up the **patched** arm

```sh
SERVE_PATCHED_T0=$(date -u +%Y-%m-%dT%H:%M:%S+00:00); SERVE_PATCHED_S=$(date +%s)
E2E_ARM=patched \
E2E_OUTPUT_DIR="$PWD/arm.patched/deployment" \
E2E_CONTAINER="${E2E_CONTAINER}_patched" \
E2E_PRIOR_STEPS="$PWD/arm.stock/accept/items/env/steps.json" \
  bash "$AGENT_SYS_TASK_PACKAGE/assets/serve/round.sh"
SERVE_PATCHED_SECONDS=$(( $(date +%s) - SERVE_PATCHED_S ))
```

`E2E_PRIOR_STEPS` is the replacement for the deleted edge: `round.sh` refuses to
start if the stock arm's step record is not there, and copies its sequence into
the patched deployment record as `preceded_by`.

*Accepted:* exit 0, `deployment.json` says `"arm": "patched"` with
`overlay.applied` equal to the number of mounts, **and the patch-live evidence is
present**: `env/docker_mounts.json`, `env/container_hashes.tsv`,
`env/marker_hits.tsv`.

*Why those three are collected here and not by the validator:*
`check_patch_live` re-hashes the patched file **inside the running container**,
and by the time a validator runs the deployment is gone. If they are missing,
the arm has to be re-run; there is no way to reconstruct them afterwards.

*If a container hash equals `sha256_stock`:* **ABORT.** The mount did not take
and this arm is measuring stock code. `__file__` will look right anyway — a bind
mount does not change the path inside the container.

### 7. Measure the **patched** arm

Identical to steps 3 and 4, with `E2E_ARM=patched`, the patched container name,
`arm.patched/...` output paths, and **the same ad-hoc cases and ids** as step 4.

*Accepted:* `steps.json` records the same five step names in the same order as
the stock arm's, all `rc: "0"`.

### 8. Tear the patched deployment down

As step 5.

### 9. Compose the two arm handoffs

```sh
for arm in stock patched; do
  python3 "$AGENT_SYS_TASK_PACKAGE/assets/lib/merge_arm.py" \
    --arm "$arm" \
    --from "$PWD/arm.$arm/deployment" \
    --from "$PWD/arm.$arm/accept" \
    --from "$PWD/arm.$arm/bench" \
    --out "$(eval echo \$AGENT_SYS_OUTPUT_$(echo ${arm}_MEASUREMENT | tr a-z A-Z))" \
    --serve-started "$(eval echo \$SERVE_${arm}_T0)" \
    --serve-seconds "$(eval echo \$SERVE_${arm}_SECONDS)" \
    --package "$AGENT_SYS_TASK_PACKAGE" \
    --environment "$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml"
done
```

The bring-up timestamps from steps 2 and 6 are required and have no default: a
guessed bring-up window would make `check_measurement_order`'s disjointness
check pass by construction.

*Accepted:* each prints `steps: serve -> smoke -> needle -> probe -> lm_eval ->
bench_r1`, and the two arms' windows do not overlap.

### 10. Compare, and write the report

```sh
python3 "$AGENT_SYS_TASK_PACKAGE/assets/compare.py" \
  --stock "$AGENT_SYS_OUTPUT_STOCK_MEASUREMENT" \
  --patched "$AGENT_SYS_OUTPUT_PATCHED_MEASUREMENT" \
  --overlay "$AGENT_SYS_INPUT_PATCH_OVERLAY" \
  --profiling-evidence "$AGENT_SYS_INPUT_PROFILING_EVIDENCE" \
  --kernel-optimization "$AGENT_SYS_INPUT_KERNEL_OPTIMIZATION" \
  --out "$AGENT_SYS_OUTPUT_INTEGRATION_REPORT" \
  --package "$AGENT_SYS_TASK_PACKAGE"
```

This is a program and you must not do its arithmetic yourself.
`check_no_regression` **recomputes every comparison from the raw numbers and
fails if its answer differs from the report's** — including when its own answer
is "accept" — so a hand-written verdict is caught rather than believed.

*Accepted:* exit 0, and `items/text.json` validates:

```sh
python3 "$AGENT_SYS_TASK_PACKAGE/assets/lib/schema.py" \
  --schema integration_report \
  --doc "$AGENT_SYS_OUTPUT_INTEGRATION_REPORT/items/text.json"
```

**A refused verdict is a result, not a failure of this task.** If `compare`
returns `accepted: false`, the report is still correct and complete and you
should still produce it; `check_no_regression` will stop the graph, which is the
validator working. Do not adjust a bar to turn a refusal into a pass — the 5%
and 10% bars were measured to be right (the within-arm round-to-round spread on a
steady node is ~2%), and a previous round widened them to 35%/30% in response to
two arms measured fifteen minutes and one co-tenant apart. That was the wrong
response; the missing control is a comparability gate at bring-up
(`../../../todo.md` T7).

## Report back

State, in this order: whether both arms completed, the two arms' time windows,
the verdict and its reasons, whether the patch was proven live (hashes and
marker counts), and anything you re-ran and why. If you aborted, say at which
step and what the evidence was.

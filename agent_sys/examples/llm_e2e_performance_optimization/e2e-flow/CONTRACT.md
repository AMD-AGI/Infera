# `e2e-flow` — the frozen cross-module contract

**Frozen 2026-09-03 by the leader, before any module work started.** Everything
in this file is what the five modules agree on so that they can be written in
parallel. A module owner who needs something here to change **asks the leader
and does not change it locally** — five owners silently disagreeing about a kind
name is the failure this document exists to prevent.

Authority is the repo-root `mission.md`. Each rule below cites the item it comes
from.

---

## 0. Why this package exists

The five stages already exist as five separate packages under
`../{deploy,profiling,analyze,kernel-opt,integration}-demo/`. **A handoff only
travels inside one run's graph**, so five packages are five runs and nothing
chains. This package is one graph: `main` → five non-leaf stages → leaves.

The five demos are **reference, not competition**. Their `assets/` are ~20k
lines of measured, debugged, cluster-proven `.py`/`.sh`. They move here and are
adapted. **Nothing about them is deleted, and nothing here is re-derived that
they already got right.**

---

## 1. The kind list — fifteen kinds, and no sixteenth without the leader

| # | kind | content_type | producer | consumers |
|---|---|---|---|---|
| 1 | `deploy_kit` | `code` | m1 | m2, m5 |
| 2 | `profiling_mode_off.bench_result` | `reproducible` | m2 · clean line | m2 · merge |
| 3 | `profiling_mode_on.bench_result` | `reproducible` | m2 · profiled line | m2 · merge |
| 4 | `profiling_mode_on.profile_result` | `reproducible` | m2 · profiled line | m2 · merge |
| 5 | `profiling_mode_on.kernel_table` | `structured_text` | m2 · profiled line | m2 · merge |
| 6 | `profiling_evidence` | `reproducible` | m2 · merge | m3, m5 |
| 7 | `kernel_worklist` | `structured_text` | m3 · rank | m3 · identify |
| 8 | `operator_identity` | `structured_text` | m3 · identify | m3 · build |
| 9 | `operator_workset` | `code` | m3 · build | m4, m5 |
| 10 | `kernel_optimization` | `code` | m4 | m5 |
| 11 | `patch_overlay` | `reproducible` | m5 · apply | m5 · integrate |
| 12 | `stock.measurement` | `reproducible` | m5 · integrate | m5 · packup |
| 13 | `patched.measurement` | `reproducible` | m5 · integrate | m5 · packup |
| 14 | `integration_report` | `structured_text` | m5 · integrate | m5 · packup |
| 15 | `e2e_packup` | `code` | m5 · packup (`is_end`) | — |

Down from 26 across the five demos. Every deletion is a mission item; see §7.

### 1.1 Naming — `${mode}.${result_type}` (M2.2)

A kind whose meaning depends on a mode **must** carry the mode as a
dot-prefixed component. `baseline` and `profiled` are gone: they named a role in
one package's story rather than a configuration.

- `profiling_mode_off` — profiler detached, **CUDA graph ON**. The numbers that
  mean something; this is what m5's stock arm must reproduce (M5.1.3.1).
- `profiling_mode_on` — profiler attached, **CUDA graph OFF**, because a graph
  launch hides the kernels the profiler is there to see.
- `stock` / `patched` — m5's two arms.

**Dots are legal and safe.** `_common.schema.json#/$defs/name` is
`^[A-Za-z_][A-Za-z0-9_.-]*$`, and `env_mgr/grants.py:450 _env_name` maps every
non-alphanumeric to `_` before uppercasing, so
`profiling_mode_on.bench_result` reaches a body as
`$AGENT_SYS_OUTPUT_PROFILING_MODE_ON_BENCH_RESULT`.

**The collision trap, checked once and never again by anything:** two kinds that
differ only in a separator — `stock.measurement` and `stock_measurement` — map
to the same variable name, and `_by_unique_kind` then **silently exports
neither** (`grants.py:435-447` keeps only names claimed by exactly one row). The
fifteen names above were checked; a new kind must be checked against all fifteen
before it is added.

---

## 2. Every handoff carries the environment (G5)

> *"整个流程的handoff都需要传递env"*

One rule, three spellings, because the content types differ:

| content_type | where `environment.yaml` goes |
|---|---|
| `reproducible` | `items/env/environment.yaml` — `env` is already a **required** item |
| `code` | `items/codes/environment.yaml` |
| `structured_text` | `items/env/environment.yaml`, `env` declared in the kind's `items_schema` |

It is the **same document with the same schema** in all fifteen. A validator
that wants to check it does not need to know which content type it is looking at
beyond picking the directory.

### 2.1 The `environment` document — promoted, not invented

The record mission M1.2.1 asks for already exists, unschema'd, inside today's
sealed handoffs as `content/items/env/deployment.json` and `context.json`. This
contract takes their union, splits it as the mission asks, and gives it a
schema at `assets/schemas/environment.schema.json`.

```yaml
schema_version: 1
fixed:                      # M1.2.1.1 — 可固化环境
  node: crsuse2-m2m-061
  node_ip: 10.245.159.129
  gpu_arch: gfx950
  gpu_count: 8
  image: infera/engine-sglang:gfx950-local
  image_id: sha256:...      # the digest, not only the tag
  dockerfile: scripts/Dockerfile.sglang   # path inside this handoff, or null
  rocm: 7.2.0
  model_name: Qwen/Qwen3.6-27B
  model_path: /shared_nfs/yihou/models/Qwen3.6-27B
  tp_size: 8
  scripts: {package: e2e-flow, commit: <sha>, entrypoints: [...]}
runtime:                    # M1.2.1.2 — 哪个机器的哪个 docker container
  slurm_jobid: '106250'
  container: yihou_e2e_flow_<run6hex>
  ports: {router: 8101, worker: 8102, etcd: 8103}
  endpoint: http://10.245.159.129:8101
  transport: spur           # spur | srun | local
  started_at: '2026-09-03T13:00:00Z'
```

`environment.md`, where a packup layout still wants one, becomes a **rendering**
of this document, not the record. Today it is checked by three regexes
(`deploy-demo/assets/check_deploy_kit.validator/check.py:71-80`), which is
exactly what M1.1.1 objects to.

---

## 3. Schemas — `assets/schemas/`, read by producer *and* validator (G2, M3.6)

> *"所有结构化的文档，尽量有自己的json schema, 该schema同时暴露给producer & validator"*

### 3.1 `items_schema` does **not** satisfy this, and that is measured

`handoff/content.py:184-197` validates a file or tree item by building
`{item_name: <filename string>}` and checking *that* against `items_schema`. The
file's **contents are never read**. It is an admission check at the seal
boundary (`store.py:448,501`), it is never exported to a body, and **no
validator in any of the five demos imports `jsonschema`** — all of them
hand-roll (`analyze-demo/…/check_workset_shape.validator/check.py:96`).

So this package carries its own schemas.

### 3.2 The layout

```
assets/schemas/
  environment.schema.json          # §2.1 — every kind
  deploy_kit.layout.yaml           # M1.1 — file/dir layout spec, not a JSON Schema
  bench_result.schema.json         # M2.2.1
  kernel_table.schema.json         # M2.9.3 / M3.5 — ONE definition, shared
  kernel_worklist.schema.json
  operator_identity.schema.json
  workset.schema.json              # M3.7 — the merged stage-3/stage-4 contract
  kernel_optimization.schema.json
  integration_report.schema.json
assets/lib/schema.py               # the ~40-line loader both sides import
```

`jsonschema>=4.18` is a declared agent_sys dependency
(`agent_sys/pyproject.toml:37`); 4.26.0 is importable here. Copy the idiom from
`agent_sys/spec_loader/validate.py:34-56` — `Draft202012Validator` plus a
`referencing` registry so schemas may `$ref` each other.

### 3.2a Every body is `#!/bin/sh` + `set -eu`, and the shebang is decoration

**agent_sys never consults a body's shebang.** It invokes one as
`["/bin/sh", entry]` — `validator/phase.py:147` and
`agent/backends/program.py:83`. On this host `/bin/sh` is **dash**:

```
$ /bin/sh -c 'set -euo pipefail; echo REACHED'
/bin/sh: 1: set: Illegal option -o pipefail     rc=2
```

So a body written `#!/usr/bin/env bash` + `set -euo pipefail` **exits 2 on line
1**, the phase reports UNREACHED rather than a verdict, and the failure reads as
the validator's rather than the shell's. Measured 2026-09-03 by m1 across all 31
skeleton bodies at once; the whole package was swept.

Write `#!/bin/sh` and `set -eu`. Where a body genuinely needs bash — today only
`assets/lib/mock.sh`, for `${!var}` — it is **invoked** as
`bash "$PKG/assets/lib/mock.sh" …` and guards on `$BASH_VERSION`, because
`. mock.sh` from a dash body is the natural thing to write and fails with an
unhelpful `Bad substitution`.

### 3.3 How both sides reach the same file

```sh
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
python3 "$PKG/assets/lib/schema.py" --schema bench_result --doc "$OUT/items/result/bench.json"
```

**Write both fallbacks in every `entry.sh`, task and validator alike.** A
validator's *input* phase gets the GLOBAL environment row and **never**
`AGENT_SYS_TASK_PACKAGE`
(`kernel-opt-demo/assets/check_workset_shape.validator/readme.md:47`). This has
already cost one run.

### 3.4 A `structured_text` handoff carries its own schema

`structured_text` has a built-in optional item `schema`. Every
`structured_text` kind here **copies its schema from `assets/schemas/` into
`items/schema`** at production time, and its validator checks that the copy is
byte-identical to the package's. The artefact is then self-describing *and*
provably not a private fork.

---

## 4. Validators (G3, G4)

1. **Program by default.** An AI validator has to justify itself; the reason is
   never "this was easier to write".
2. **At most three AI validators per handoff** (G3). More than three means one
   AI validator checking several criteria, not four validators.
3. **An AI validator's criteria are YAML**, not prose — name / brief / criterion
   per row, in the validator's `args` or in a file beside it (G3.1).
4. **An AI validator's `readme.md` is a `STEPS` section**: an ordered list of
   commands, each with its acceptance criterion (G4.2.1). The AI's job is to run
   them in order and read the results, not to invent a method.
5. Keep `dimension` / `strength` / `tags.cost` honest — `cost` is what orders a
   phase cheapest-first, and a `strong` verdict stops the graph.

### 4.0 The one trust chain in this package, and what holds it up

m4 is told to take its ground truth **strictly from the workset** and to abort
rather than re-measure when the premise differs (M4.3.5, reversing the old "do
not trust the workset's printed number" rule). **That instruction is only safe
because something has already run the workset's own tests on this hardware.**
That something is `check_workset_runs`.

The workset's evidence (`evidence/{correctness,performance}.json`) is written by
`build_workset`, which builds *and* measures — there is no separate
`verify_workset` task, because splitting build from measure across two agents is
the thing M2.5 forbids in the analogous case. So the evidence is the producer's
own claim.

**Therefore `check_workset_runs` must re-run at least one shape itself and check
its own number against the recorded one.** Reading the producer's evidence file
and grading its shape would make the whole chain a claim about a claim:
`build_workset` asserts a baseline, `check_workset_runs` confirms the assertion
is well-formed, and m4 then divides by it. That is the same failure
`check_no_regression` avoids by recomputing rather than reading a `verdict`
field, one stage earlier.

Consequence, and it is intended: **`build_workset` needs the shared container**
(its inputs already include `deploy_kit`), and `check_workset_runs` stays
`cost: gpu_hours`. If either is ever weakened, m4's
`check_speedup_substantiated` has to go back to re-measuring, and whoever
weakens it says so to the leader.

### 4.1 Shared validators are shared, not copied

`check_kernel_table` is **one** definition used by m2 and m3 (M3.5). The two
demos each carry a copy today with different `args`, which is one of the three
seams recorded in `handoff.analysis.md`. Same for the workset validators: they
live with m3's `operator_workset` and m4 **references** them (M4.4).

---

## 5. The runtime environment (G5.1, rule 7)

**Modules 1–4 share one container on one held node.** m1 brings it up and
records it in `environment.runtime`; m2, m3 and m4 exec into it.

**Module 5 is the designed exception.** Its two arms need two containers — a
container holds one state for its life, which is the entire reason the two-arm
design exists. `mission.md` G5.1 grants it: *"如果不行，再考虑换机器/启动不同的
docker container"*. m5 brings up both arms from the same image and the same
`environment.fixed`.

### 5.1 Bring-up and use are never split across agents (M2.5, M5.2)

> *"agent A 去把服务部署好，agent B 去使用：这是不被允许的"*

Consequence, and it is large: **there are no `serve_*` tasks and no
`deployment_*` handoffs anywhere in this package.** A task that needs a service
brings it up itself, in its own `readme.md` STEPS, and tears it down.

### 5.2 Cluster rules, standing, absolute

- All spur nodes share `/shared_nfs`. workspace / playground / handoff may live
  there at 777. **Nothing whose path lacks the substring `yihou` may ever be
  deleted.** "Remote" *is* this sharing.
- Never `docker rm -f` a container you did not create. Both held nodes are
  carrying other tenants' containers right now.
- Never `agent-sys run --clean` on a shared root — it removes **every** run.
- Every identifier bound on a shared host is a `--var`: container name, ports,
  workdir, served model name. `: "${VAR:=…}"`, never `export VAR=`.

---

## 6. Localisation — nothing site-specific in a spec (M2.1)

Out of every task `readme.md` and every step yaml:

- **how to reach a node.** `spur exec` / `srun --overlap` is dispatched by
  `assets/lib/remote.sh` on `$E2E_TRANSPORT`, and no readme spells either.
- **model facts.** No model name, path, context length, parser flag or TP size
  appears outside `shared.yaml`.
- **node facts.** Job id, hostname, IP are `--var`s with **no default** — a
  default is one allocation's answer shipped as everyone's, and it goes stale
  the hour the job ends.

---

## 7. What each module deletes, and the mission item that says so

| module | deleted | item |
|---|---|---|
| m2 | `serve_baseline`, `serve_profiled` tasks | M2.5, M2.3 |
| m2 | `deployment_baseline`, `deployment_profiled` kinds | M2.4 |
| m2 | `check_service_live` | M2.8.1 |
| m3 | `seed_table` task and its synthetic seed | M3.2 |
| m3 | the second `check_kernel_table` | M3.5 |
| m4 | `publish_workset` task | M4.2 |
| m4 | the standalone `workset` kind — merged into m3's `operator_workset` | M3.7, M4.1 |
| m4 | the "do not use the workset's printed number as denominator" rule | M4.3.5 — **reversed**: ground truth comes *strictly* from the workset; hardware/premise mismatch **aborts**, software mismatch **warns** |
| m5 | `seed_patch` — its input is now m4's `kernel_optimization` | M5.1 |
| m5 | `serve_stock`, `measure_stock`, `serve_patched`, `measure_patched`, `compare` → **one** AI task | M5.2 |
| m5 | six evidence kinds → `stock.measurement` + `patched.measurement` | consequence of M5.2 |

---

## 8. Deferred — recorded in `todo.md`, not built

`check_trace_coverage` against sglang source (M2.8.2) · `vendor_tuned` bucket
(M3.3) · one handoff per operator (M3.7.7) · AI-led + program-fixed analysis
(M3.8) · the patch mechanism should hack the registry rather than bind-mount
(M5.3 — *"但现在就这样吧"*, so `overlay_files` stays) · permission and
visibility management for the shared container (rule 7).

---

## 8a. Five owners, one worktree — how to commit without taking someone else's work

Raised by `checkpoint` 2026-09-03 and it is right: five owners write into **one**
shared checkout. At the moment it was raised, twelve modified and five untracked
files belonging to at least four different owners sat in the tree at once.

**`git add <dir>` is not the hazard's cure, and `git add` at all is part of it.**
`git add .../e2e-flow/` obeys "stage only paths under the package" to the letter
and sweeps four other owners' half-written files into one owner's commit. Worse,
**the index is itself shared state**: owner A's `git add` lands in the same index
owner B commits from a second later, so even correct per-file staging races.

### The rule

**Commit paths directly and never touch the index:**

```sh
git commit -s -m "..." -- \
  agent_sys/examples/.../e2e-flow/assets/check_yours.validator/check.py \
  agent_sys/examples/.../e2e-flow/assets/schemas/yours.schema.json
```

`git commit -- <pathspec>` commits the working-tree content of exactly those
paths and **ignores the index entirely**, so a concurrent `git add` by another
owner cannot be swept in. If two commits collide on `index.lock`, git says so;
wait a second and retry.

Then verify what you actually committed, rather than what you meant to:

```sh
git show --stat --name-only HEAD
```

Not one worktree per owner, which would be the structurally clean answer: work
is already in flight in this tree and moving it now would strand it. This is the
cheap correct fix, and the manifest below is what makes it checkable.

### The ownership manifest

Anything not listed is the **leader's**. A file with two claimants is a
conversation with the leader, not a race.

| owner | paths |
|---|---|
| leader | `CONTRACT.md` · `MOCK-MAP.md` · `README.md` · `main.yaml` · `shared.yaml` · `steps/common.yaml` · `assets/main.task/` · `assets/lib/{mock.sh,schema.py,env_render.py}` · `assets/schemas/{environment.schema.json,README.md}` · `../todo.md` |
| m1 | `steps/m1_deploy.yaml` · `assets/{check_deploy_kit,check_deploy_serves}.validator/` · `assets/{deploy_and_prove,m1_deploy}.task/` · `assets/schemas/deploy_kit.layout.yaml` · `assets/lib/zone.py` |
| m2 | `steps/m2_profiling.yaml` · `assets/{check_bench_result,check_trace_coverage,check_profiling_evidence,check_kernel_table}.validator/` · `assets/{run_profiling_mode_off,run_profiling_mode_on,merge_profiling_evidence,m2_profiling}.task/` · `assets/schemas/{bench_result,kernel_table}.schema.json` · `assets/{serve,load,analyze}/` · `assets/lib/{remote.sh,trace_stream.py}` |
| m3 | `steps/m3_analysis.yaml` · `assets/{check_worklist_shape,check_identity_resolved,check_workset_shape,check_workset_runs}.validator/` · `assets/{rank,identify,build_workset,m3_analysis}.task/` · `assets/schemas/{kernel_worklist,operator_identity,workset}.schema.json` |
| m4 | `steps/m4_kernel_opt.yaml` · `assets/{check_speedup_substantiated,check_optimization_shape}.validator/` · `assets/{optimize_kernel,m4_kernel_opt}.task/` · `assets/schemas/kernel_optimization.schema.json` |
| m5 | `steps/m5_integration.yaml` · `assets/{check_overlay_applies,check_patch_live,check_measurement_order,check_acceptance,check_bench_report,check_no_regression,check_packup_shape}.validator/` · `assets/{apply_patch,integrate_and_verify,packup,m5_integration}.task/` · `assets/schemas/integration_report.schema.json` · `assets/{accept,bench}/` · `assets/lib/{patchkit.py,eval_stats.py,store.py,redact.py,nodecall.py,container_roots.yaml}` |

`check_kernel_table` is **declared** in `steps/common.yaml` (leader's, because m2
and m3 share it) and its **body** is m2's. That split is deliberate: the shared
declaration is what stops the two-copies seam from reappearing, and the body has
one author.

**`assets/lib/` is the collision zone.** Announce a new file there to the leader
before landing it — three of us have already put something in it.

### The repo-root litter, which is a different and smaller problem

`glm5.2-dp8-tp8-workload-schema.tar`, `rank0/`, `.serena/`,
`handoff.analysis.md`, and a modified `agent_sys/docs/design.md` are the user's,
untracked, and outside this package. `git commit -- <paths>` cannot reach them,
so the rule above closes this one as a side effect.

## 9. The gate every change passes, in under a second

```sh
python3 -m agent_sys.cli.main show \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --var jobid=106250 --var node=crsuse2-m2m-061 --var node_ip=10.245.159.129 \
  --var model_name=Qwen/Qwen3.6-27B --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=infera/engine-sglang:gfx950-local
```

It loads and type-checks every yaml, derives the edge set from the handoff
wiring, checks it against every `froms`, and dispatches nothing. **Run it after
every edit.** `run --dry-run` is the next rung; `agent-sys run` with mock agents
is the one after that.

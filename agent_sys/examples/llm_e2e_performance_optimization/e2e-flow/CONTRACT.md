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

### 2.2 The absolute-path rule does not apply to this record

Carried validators inherit a rule from `analyze-demo` — *no absolute host path
in a handoff* — justified there by *"the seal refuses the whole delivery over
one"*. **Measured by m3 against the framework: that premise is false.**
`handoff/store.py:447` reads `# locality.check — NOT CALLED`, and `:494` gives
the reason: the shape heuristic read an HTTP access-log line as a filesystem
path and refused a correct artefact, **97% false positive on a real kit**.
Corroborated from the other side — the sealed `stage1-deploy/deploy_kit` carries
`/shared_nfs/...` in five content files and sealed cleanly.

This matters here and not only as wording: `environment.schema.json` **requires**
`model_path`, which is `/shared_nfs/yihou/models/Qwen3.6-27B` — an absolute path
by nature. A validator carrying the rule forward verbatim **rejects every
conforming handoff in this package**, which is how m3 found it, on a fixture.

So: keep the rule on its own merit — portability, a script carrying one host's
directory does not run on the next host — and **scope it to executable and
generated content (`.py`, `.sh`, `.json`, `.jsonl`), skipping the environment
record.** Do not justify it by the seal.

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

**And run `"${AGENT_SYS_DEMO_PYTHON:-python3}"`, never a bare `python3`.**
`cli/main.py:668` exports the interpreter the run itself is using. A validation
zone gets a policy-derived `PATH` on which `python3` resolves to
`/usr/bin/python3`, which on this host has **no `referencing`** and therefore
cannot import `assets/lib/schema.py`.

Measured: the body dies with `ModuleNotFoundError` **before writing
`verdict.json`**, and the phase reports *"nothing was decided"* rather than a
verdict — **a validator that cannot start looks exactly like one that was never
asked.** Found by m5 driving their leaves through the graph; twelve of the
twenty-one validators had it, across all five modules.

**`${VAR:-default}` and `${VAR-default}` are not the same test, and the colon is
the one that disarms a guard.** m1's, 2026-09-03, found while wiring
`replayed_from`.

`${VAR:-d}` substitutes when `VAR` is unset **or set-but-empty**. `${VAR-d}`,
without the colon, substitutes only when it is unset. So a caller that sets a
name to the empty string **on purpose** — meaning *"I am deliberately not this
thing"* — is overruled by the colon form and gets the default instead.

What that cost: `check_deploy_kit`'s `gate.sh` sets `MOCK_REPLAYED_FROM=""`
deliberately, because its fixture stands for a **real** bring-up and must face
the strict `environment.md` comparison. The empty string fell through to the
default, the fixture was marked *replayed*, the strict branch was skipped —
**and the planted `fixed.image` fault went unreported while the gate still
printed PASS.** The guard did not fail; it stopped testing the rule it exists to
test, and said nothing.

**A guard that silently disables a check is worse than one that fails**, and
this is the shell form that does it. Where empty is a meaningful value — a
deliberate "not set to anything" — use the colonless form, and prefer
`${VAR?message}` where absence is an error.

#### And **declaring a name with an empty default is not a no-op**

The same distinction one level up, and the leader shipped it as a live
regression the same afternoon it was written down.

`60bd848` added `E2E_STAGE: '${stage:-}'` to `runner`, to declare a name that
shared libraries were reading and nobody declared. **A declared-empty variable
is *present*.** So every body coping with the name's *absence* changed
behaviour that instant:

```
unset            environment.setdefault("E2E_STAGE", "m4")  ->  'm4'
declared empty   environment.setdefault("E2E_STAGE", "m4")  ->  ''
```

**Which forms are hazardous, measured — and the first version of this section
named the wrong one.** m5 checked it rather than accepting it, in both shells
and Python:

```
bash/dash   unset V; : "${V:=filled}"   ->  filled      SAFE
bash/dash   V=;     : "${V:=filled}"    ->  filled      SAFE — the colon form fills empty too
bash/dash   V=;     : "${V=filled}"     ->  (empty)     HAZARD
python      V=''; environ.setdefault    ->  ''          HAZARD
```

**`${VAR:=…}` is safe. The hazards are `${VAR=…}` — no colon — and Python's
`setdefault`.** This section originally said the opposite, which would have sent
every owner grepping for the harmless form and past the dangerous one. It is the
exact sibling of `c69c813` above: **same colon, opposite direction**, and the
leader got it backwards while writing the section about getting it backwards.

So the grep before adding a name to `shared.yaml` is `setdefault("<name>"` and
`${<name>=` — **not** `${<name>:=`.

**And it hit the only stage that was doing the right thing.** m4 was the sole
caller in the package setting `E2E_STAGE` at all — 21 other callers of
`env_render.py` set nothing — so declaring the name **took the one stage that
stamped `warnings[].stage` correctly and made it match the twenty-one that did
not.** Found by m4 running the leader's own new checker against their own agent
rather than assuming it passed; fixed in `7028275` by guarding on truthiness.

So: **before adding a name to `shared.yaml`, grep for `setdefault("<name>"` and
`${<name>=` in `assets/` — no colon on the second.** Coping-with-absence is a contract a declaration
breaks, and it breaks it silently and in the direction of the code that was
already correct.

#### A checker that makes people write worse comments is doing net harm

m4's, from the same commit. `check_agent_env.py` greps `.sh`/`.py` for
`E2E_[A-Z_]+`, so **a name in a comment counted as a read** — and their fix,
which *removed* a reader and left the name in the comment explaining why, was
reported as a problem. The repair they were pushed into was to reword the
comment: *"the fix made the comment slightly worse to keep a grep happy."*

Whole-line `#` comments are now stripped from code before matching. **Markdown
is exempt on purpose** — for a `kind: ai` closure the readme *is* the program,
so a variable named in prose is one the agent will try to read, and there is no
comment/code distinction to draw there.

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

### 4.2 Every `${...}` arg arrives as a **string**, and it has bitten twice

`args: {timeout_seconds: '${x:-3600}'}` reaches a body as the string `"3600"`,
not the integer. Both halves of that have now cost a run:

- **Truthiness.** `args.get("n") or 3` reads **3** when the spec says `0`,
  because `"0"` is truthy — and the `or` form *silently works* on the `${...}`
  form while failing on a genuine yaml integer `0`, which is the worst
  combination for ever finding it. Found by m3 on the one knob that can
  dismantle §4.0's trust chain; the guard refusing it was itself unreachable.
- **Arithmetic.** `time.time() + args["bringup_timeout_seconds"]` raises
  `TypeError: unsupported operand type(s) for +: 'float' and 'str'`. Measured in
  `check_deploy_serves` on the full mock run: the validator **crashed after a
  successful bring-up**, reported *"the check itself failed"*, and its teardown
  then failed on the same expression — so it also warned that containers and
  ports might be held. (They were not; checked on the node.)

**In a validator the arithmetic half produces no answer, not a wrong one.**
m2 proved it with teeth rather than asserting it: they copied
`check_trace_coverage`, removed exactly one `int()`, and ran it with real
string-typed args —

```
floor = args.get("min_gpu_kernels_per_rank", 1000)   # was int(...)
→ TypeError: '<' not supported between instances of 'int' and 'str'
→ rc=1, verdict.json absent
```

— which lands in **the dangerous category**: non-zero exit, no verdict, and the
phase reads a broken validator rather than a refused handoff. Same signature as
`check_deploy_serves`'s crash.

**Only the substituted half of an `args` block is affected**, which is why it
survives review: `spec_loader.variables.substitute` leaves a substituted scalar
a **string**, so `min_requests: '${min_requests:-50}'` arrives as `"50"` while a
literal `min_pct_total_sum: 80.0` beside it is still a float. Half the numbers
in one block are already typed.

**Read every numeric arg through `workset_io.arg_num`** (m3's), which coerces
and refuses an explicit `0` only when the spec means it to. `float`/`int` at the
point of use is not enough on its own — the truthiness half survives it.

### 4.3 One authority, two readers, one of them narrower

Named by m4 after the third instance in two days. **None of the three was wrong
logic**, and each is invisible to review because *both readers look correct in
isolation*:

- m3's `check_workset_runs` took the **workset** as authority for which checks
  to apply and the **report** as authority for which shapes were in scope — so
  a harness that silently measured nothing for one shape produced a clean PASS
  over a partial measurement.
- m4's `abort_on_mismatch` was unioned with the workset's list in
  `_check_premise` and read from the yaml's alone in `_check_ground_truth`, so
  a field the workset added to its own abort list went unenforced.
- m4's `_interpreter()` chose an interpreter that can import torch and both call
  sites used it as a yes/no probe and **discarded the value**.

Add the leader's: `schema.py` carried a comment asserting no schema used `$ref`
while three did. Same joint, different tissue — that one belongs to the
*justification-outliving-its-premise* family (§2.2, `container_roots.yaml`,
MOCK-MAP's `SGLANG_TORCH_PROFILER_DIR`), and the two families share a cause:
**a fact stated in one place and relied on in another, with nothing that fails
when they diverge.**

**What it is not**, and m3 drew the line: `min_shapes` in `args` and
`minItems: 3` in the schema are *two places mentioning one number*, where the
stricter wins by construction and the precedence is written down. That is fine.
**The fault is two readers with different reach**, not two mentions.

**Seven instances in two days across three owners**, and the last two are the
ones that matter most for how to look:

- m3 hardcoded an entrypoint's flag spelling in the harness *and* declared it in
  the manifest — so a workset declaring `--implementation` would have had m4
  passing it to a parser that only knew `--impl`. **The reader that could not be
  told was the harness.**
- `protocol` was declared in `workset.yaml`, echoed into the report by the
  harness, and compared by nothing — so m4 could re-measure under the manifest's
  protocol and divide by a baseline the report recorded under another. **Across
  two protocols that ratio looks entirely normal.**

**And the sharpest lesson is m3's, about the audit rather than the bug.** They
told the leader *"nothing else of mine reads one rule from two places"*, then
audited properly and found those two. **Claiming an audit is not one**, and the
difference between the two was two live defects that would have surfaced in m4's
transcript pointing at m3's code.

**What found each of them is the useful part, and it was never review.** A stub
that could *withhold* a shape, a workset that *added* a field, a colleague who
asked whether m4's case was really different from m3's. m3 had read the code
eight times. So: build fixtures that can take something away, and when a peer
asks whether your situation is the same as theirs, **check rather than reason**
— m4 assumed the interpreter exposure was "probably moot inside the shared
container" and it was in two places.

#### A half-parameterised identifier is worse than an unparameterised one

m1's, 2026-09-03, committed **while writing the fix for the same class**. They
parameterised etcd's port at its *producer* — `--listen-client-urls
http://0.0.0.0:${PORT}` — and left the literal in its *consumer*,
`--etcd-endpoint $MY_IP:2379`. The router died with `ConnectError`, **naming
neither the port nor the mismatch**, and they had to find it by hand.

**Because it looks fixed.** An unparameterised identifier is visibly hard-coded
and the next reader treats it accordingly; a half-parameterised one presents a
variable at the site anybody checks and a literal at the site nobody does. §4.3
is about one authority with two readers, and a port is exactly that — so
**fixing one end and not the other is this section's failure mode, not a
different one.**

The check is mechanical: having parameterised an identifier, grep the *value*
you removed, not the name you introduced.

#### When a symptom has candidate causes in more than one owner's work, reproduce before attributing

`checkpoint`'s, and it is the rule I would want the next effort to start with.
**Inference across an ownership boundary was wrong every time it was tried
today; measurement was right every time.**

The expensive instance was the leader's. Rung 0 refused three times at
`check_deploy_serves`; the cause was a missing `--var transport_env`, attributed
first to m1's GLM deployment holding GPUs, then to a missing `local` branch in
m2's `remote.sh`. Both readings were coherent, both were about somebody else's
work, and both were wrong. What settled it was copying the validator's zone and
running it under `env -i` — one command, and the diagnostic named the cause
outright.

m3 hit the mirror image within the hour: they inferred a general defect in m1's
records from a node mismatch **they had created themselves** with an ambient
`E2E_JOBID`, and flagged it as *worth checking rather than assuming*. The flag is
why it became a check instead of work handed to the person the ladder was
waiting on. The record was two commands away and consistent.

**The asymmetry is the point.** A wrong guess about your own code costs you a
few minutes. A wrong guess about a colleague's costs them an audit of work that
was never broken, and it arrives with your authority attached.

### 4.4 A fixture that is more convenient than production tests the fixture

m2's wording, kept nearly verbatim because the last sentence is the whole rule.

Every harness supplies its subject with inputs. When those are the *tidy*
version — empty args instead of the run's own, a module imported instead of
called, an interpreter that happens to have the dependency, a variable the real
caller does not set — **the harness cannot see the bug it was built after, and
it says so by returning clean.**

Three measured instances, **all found by the harness's own author**:

- layer B *imported* modules, while `schema.py` imports `jsonschema` inside
  `validate()`;
- layer C1 passed `args.json = {}`, so every `.get` returned a **typed** default
  and no string ever reached the arithmetic §4.2 is about;
- m4's stub kit honoured `${KFO_PYTHON:-python3}`, a variable the validator was
  not setting, so the kit could not watch the interpreter fail to arrive.

**And fix it everywhere the same convenience appears, not only where it bit.**
m2's, after finding the identical blind spot in a *second* harness
(`from_yaml.py` still `json.loads`-ed substituted values, so every threshold
arrived typed) hours after fixing it in the first: *"it is not that people write
bad fixtures, it is that a fixture gets fixed where the bug was found and not
everywhere the same convenience exists."*

**The test is: name the input the bug needs, and check the fixture delivers that
exact form.** Not a plausible one — that one. **And prove the probe can fail
before believing that it passed.**

#### The same root cause has three faces, and two of them invent a problem

All three are **the instrument's condition reported as the subject's verdict**.
Only the first is the one people guard against.

1. **A benign fixture reads as PASS.** The three instances above.
2. **A missing fixture reads as FAIL.** m2's, 2026-09-03: `from_yaml.py` pointed
   at a merge output directory that had been renamed, so `check_profiling_evidence`
   graded a path that did not exist and came back FAIL — immediately after a
   schema change, which is exactly when a false failure is most believable. Their
   clause: ***check the probe before believing a failure, not only before
   believing a pass.***
3. **A probe matching itself invents a failure that never happened.** m2's, the
   same afternoon: checking for leaked stub processes, `pgrep -af stub_router.py`,
   `pgrep -f 'python3.*stub_router'` and `grep -c 'stub_router.py 8'` each
   reported hits — **every one matching its own command line**, which contained
   the string. Three consecutive probes, three false alarms, one cause.

   **The reading that settles it is the one that cannot self-match**: the PID
   files the subject itself wrote, and the bound ports. Neither can contain the
   query. Nothing had leaked.

Face 2 has now caught the leader twice — most expensively when rung 0 returned
`check_deploy_kit: FAIL` on a stage that had been green, and the available
reading was "someone's commit regressed it". It had not: the leader had passed
`--var image=` a tag present on the node instead of the one the sealed kit
renders, and the validator refused correctly. Believing that failure would have
sent two owners auditing their commits for a defect that was in a command line.

4. **An instrument with no input at all reads PASS.** m3's, 2026-09-04, and it
   is the one nobody had a name for. `harness/_common.py` derived seven
   `E2E_<FIELD>` names for its `abort_on_mismatch` / `warn_on_mismatch` loops.
   **Not one of the seven was declared anywhere in the package**, so `_observed`
   returned `None` every time and **neither loop had ever been able to fire** —
   including the abort that stops a measurement being taken on a machine the
   workset's evidence did not come from, which is M4.3.5's premise.

   Faces 1–3 are all about a *fixture*: benign, missing, or self-matching.
   **Here there was no fixture of any kind** — the instrument read a channel
   that did not exist, and silence from a channel that does not exist is
   indistinguishable from silence meaning agreement. Its own docstring said so
   and made it sound safe: *"an unset variable means unknown, and unknown is
   not a mismatch."* Unknown was always.

   **And a second defect would have made the first meaningless even if it had
   fired.** The report's `environment` block took `gpu_arch`, `gpu_count`,
   `tp_size`, `container` and `image_id` **from the workset's own claim**,
   copied into the block whose whole job is to say where the run happened. A
   working abort would have compared a premise against a report that agreed
   with it by construction.

**Whichever face it wears, ask what the instrument would report if the subject
were fine — and make sure that is a different answer from the one you got.**

#### A paragraph asserting a behaviour is not one

m3's, and it is the cheapest signal in the codebase because it costs nothing to
look. `harness/_common.py`'s docstring read:

> *the abort is a behaviour, not a paragraph.*

**It was a paragraph** — and the sentence denying it had been read many times by
its own author. The same shape appeared twice more the same day: `redact.py`'s
error text telling m5 *"these absolute paths would still be refused by the
seal"* when the store does not call the seal's check, and `probes.yaml`'s
`direction` claiming the completion probe discriminates a case it is
structurally blind to.

**Three files, three owners, one failure: prose in a tool asserting a property
the code does not have, read past by everyone including the person who wrote
it.** The remedy is the same question §4.4 already asks and none of the three
authors asked of their own text: *what would this report if the subject were
broken?*

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

### 5.0 A container-written output is root-owned, and reading it works

Found by m3 on the first real GPU run of this package, and **every body that
runs work in a container hits it** — m1, m2, m4 and m5 the moment they run for
real.

The container runs as **root**, and it has to: a framework compiling kernels on
first call cannot write its cache as a user who does not exist inside the image.
So every file written into `$AGENT_SYS_OUTPUT_<KIND>` is root-owned.

**Reading is fine, and that is what makes it easy to miss.** The files are 644,
so `copy_out` works, the seal works, every validator reads them, and the run
goes green. What fails is *later*, and on the **next** run rather than the one
that caused it: the zone's own user cannot clean up.

`assets/lib/reclaim.sh <container> <path> [...]` chowns from **inside the same
container that created the files**, which is the only context with the
privilege. Idempotent, and a no-op when the container is gone, so a body calls
it in a `finally` without deciding first whether it will work.

Also export `PYTHONDONTWRITEBYTECODE=1` in any body that runs Python in a
container — root-owned `__pycache__` is the same problem arriving first.

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

**And never `git --amend`. Corrections get their own commit.** Raised by
`checkpoint` 2026-09-04 against their own near-miss, which is the only way this
one gets found.

The pathspec rule bounds the *blast radius*; it says nothing about `--amend`,
because **`--amend` rewrites whoever's commit happens to be at HEAD** — and HEAD
is moved by five owners. checkpoint amended their own checkpoint commit to fix
one wrong row; in the seconds between deciding and running it, the leader's
`8b87f41` landed, and the amend rewrote **that**, producing a commit that was the
leader's two files plus checkpoint's. Repaired with `git reset --mixed 8b87f41`
and verified back at its original SHA with both files and nothing lost.

The pathspec discipline is why the damage was one file rather than four owners'
work. But *"amend my last commit"* is a sentence with no true referent in a
shared worktree: **there is no "my last commit", only HEAD.**

A stale `index.lock` is the other half of this. Before removing one, establish it
is dead — created after the last successful commit, **zero bytes**, no holder
under `lsof`/`fuser`, no git process on the host. Establishing that is the whole
of the work; removing it is trivial afterwards.

Then verify what you actually committed, rather than what you meant to —
**and the obvious form of that check is broken.**

```sh
git log -1 --format='%h %s'          # is HEAD MINE?  ← the part that was missing
git show --stat --name-only HEAD     # and does it hold only my paths?
```

**`git show --stat --name-only HEAD` alone confirms the path and not the
commit.** Found by checkpoint, in their own procedure, against a commit of
mine:

1. their commit failed on `index.lock`;
2. in the seconds before the retry, **my** commit named a tree and swept their
   dirty `work.checkpoint.summary.md`;
3. their retry found nothing to commit for that path and **said nothing**;
4. their `--stat` check printed `work.checkpoint.summary.md` — **exactly what
   they expected to see** — because HEAD was my commit, holding their file.

So they reported "T+60 is committed" and it was not, by them; and they reported
that the `index.lock` retry "confirmed the guidance", when in fact **the retry
is not idempotent under contention and its no-op is silent.** Both reports were
false, and the check that should have caught it passed for the wrong reason.

`3b2ffde` is the artefact: my subject, 187 insertions, **nothing but their
file.** A reader trusting `%s` learns the opposite of what happened — which is
the third duplicate-subject pair today and the second where the duplicate is the
cross-owner one.

**This section's own verification step was the thing it was written to prevent.**

**A new file needs one narrow `git add` first**, and this is the one exception:
`git commit -- <path>` only reaches paths git already knows, so an untracked
file fails with *"pathspec did not match any file(s) known to git"*. Found by m1
on their first new file.

```sh
git add -- <the one new file>                       # never a directory
git commit -s -m "..." -- <all your paths>          # still ignores the rest of the index
```

**The rule protects others from you. It does not protect you from others.**
Reported by m2 after committing correctly by pathspec and still having their
work land inside another owner's commit. `git commit -- <paths>` bounds what
*your* commit takes; it does nothing about a file of yours sitting dirty in the
tree when somebody else names a directory. So: **commit early and often.** An
uncommitted file is the only thing that can be taken, and the window is however
long you leave it there.

The `git add` is narrow enough to be safe — it names one file, not a tree — and
the `git commit -- <paths>` that follows still commits working-tree content for
everything you name, so a concurrent add by another owner is still not swept in.

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
| m3 | `steps/m3_analysis.yaml` · `assets/{check_worklist_shape,check_identity_resolved,check_workset_shape,check_workset_runs}.validator/` · `assets/{rank,identify,build_workset,m3_analysis}.task/` · `assets/schemas/{kernel_worklist,operator_identity,workset}.schema.json` · `assets/lib/{workset_io.py,forge_export.py,csv_io.py,kernel_table.py,shapes.py,taxonomy.py,symbols.py,store.py,kernel_taxonomy.yaml}` |
| m4 | `steps/m4_kernel_opt.yaml` · `assets/{check_speedup_substantiated,check_optimization_shape}.validator/` · `assets/{optimize_kernel,m4_kernel_opt}.task/` · `assets/schemas/kernel_optimization.schema.json` · `assets/schemas/samples/` |
| m5 | `steps/m5_integration.yaml` · `assets/{check_overlay_applies,check_patch_live,check_measurement_order,check_acceptance,check_bench_report,check_no_regression,check_packup_shape}.validator/` · `assets/{apply_patch,integrate_and_verify,packup,m5_integration}.task/` · `assets/schemas/integration_report.schema.json` · `assets/{accept,bench}/` · `assets/lib/{patchkit.py,eval_stats.py,redact.py,nodecall.py,container_roots.yaml,merge_arm.py,mock_m5.sh}` |

`check_kernel_table` is **declared** in `steps/common.yaml` (leader's, because m2
and m3 share it) and its **body** is m2's. That split is deliberate: the shared
declaration is what stops the two-copies seam from reappearing, and the body has
one author.

**`../todo.md` is append-only for owners, and that is a correction to the row
above.** Raised by `checkpoint` 2026-09-03 against m1's `c16a5bb`, which added
T17 to a file the manifest assigns to the leader. checkpoint's reading is the
one I am taking: **the manifest was wrong, not m1.** The whole point of `todo.md`
is that a deferral gets recorded *at the moment it is found*, and the person who
finds it is mid-task — routing it through the leader means it is written later,
by someone who was not there, or not at all. A rule that is only obeyed by people
who are not busy is not a rule.

So: any owner may **append** a numbered item. Nobody but the leader may edit or
remove an existing one, because renumbering under five concurrent readers is how
a deferral silently becomes a different deferral. Commit it on its own or
alongside the work that produced it, by path, as above.

This is the second time the manifest has been wrong in the direction of
over-centralising, and the pattern is worth naming: **a file the leader writes
most of is not thereby a file only the leader may write.** Ownership here is
about who resolves conflicts, not about who is allowed to contribute.

**`assets/lib/` and `assets/bench/` are the two collision zones.** Announce a new
file in either to the leader before landing it — three of us have already put
something in `lib/`, and `bench/` holds `aiperf_replay.sh`, `pythonpath/` and
`summarise.py` shared between m1, m2 and m5 (`aiperf_synthetic.sh` is m1's).

A shared file carried across from a demo package **keeps that package's variable
prefix until somebody renames it, and the rename is the leader's.** Measured:
`assets/lib/remote.sh` arrived from `integration-demo` and its `_env_prelude`
forwarded `^(IT_|AGENT_SYS_)` to the remote side — so **no `E2E_*` variable
reached the far end of an `spur exec` at all**, and the symptom would have been
an unset variable on the remote host, naming neither the line nor the rename.
Reported by m1, who correctly declined to rename a shared file under four other
owners.

### The repo-root litter, which is a different and smaller problem

`glm5.2-dp8-tp8-workload-schema.tar`, `rank0/`, `.serena/`,
`handoff.analysis.md`, and a modified `agent_sys/docs/design.md` are the user's,
untracked, and outside this package. `git commit -- <paths>` cannot reach them,
so the rule above closes this one as a side effect.

## 9. The gate every change passes, in under a second

```sh
python3 -m agent_sys.cli.main show \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --var jobid=1 --var node=n --var node_ip=0.0.0.0 \
  --var model_name=m --var model_path=/p --var image=i
```

It loads and type-checks every yaml, derives the edge set from the handoff
wiring, checks it against every `froms`, and dispatches nothing. **Run it after
every edit.** `run --dry-run` is the next rung; `agent-sys run` with mock agents
is the one after that.

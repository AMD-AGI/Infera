# `e2e-flow` — the frozen cross-module contract

**Frozen by the package owner, before any module work started.** Everything
in this file is what the five modules agree on so that they can be written in
parallel. A module owner who needs something here to change **asks the package owner
and does not change it locally** — five owners silently disagreeing about a kind
name is the failure this document exists to prevent.

Authority is the repo-root `mission.md`. Each rule below cites the item it comes
from.

---

## 0. Why this package exists

The five stages began life as five separate task packages, one per stage. **A
handoff only travels inside one run's graph**, so five packages are five runs
and nothing chains. This package is one graph: `main` → five non-leaf stages →
leaves.

Their `assets/` — some 20k lines of measured, debugged, cluster-proven `.py`
and `.sh` — were carried over and adapted rather than re-derived. **This is a
refine of definitions, not a rewrite of bodies.**

---

## 1. The kind list — fifteen kinds, and no sixteenth without the package owner

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
`^[A-Za-z_][A-Za-z0-9_.-]*$`, and `env_mgr/grants.py _env_name` maps every
non-alphanumeric to `_` before uppercasing, so
`profiling_mode_on.bench_result` reaches a body as
`$AGENT_SYS_OUTPUT_PROFILING_MODE_ON_BENCH_RESULT`.

**The collision trap, checked once and never again by anything:** two kinds that
differ only in a separator — `stock.measurement` and `stock_measurement` — map
to the same variable name, and `_by_unique_kind` then **silently exports
neither** (`grants.py` keeps only names claimed by exactly one row). The
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
  node: <host>
  node_ip: <ip>
  gpu_arch: gfx950
  gpu_count: 8
  image: <registry>/<engine image>:<tag>
  image_id: sha256:...      # the digest, not only the tag
  dockerfile: scripts/Dockerfile.sglang   # path inside this handoff, or null
  rocm: 7.2.0
  model_name: <org>/<model>
  model_path: <path to the weights on that host>
  tp_size: 8
  scripts: {package: e2e-flow, commit: <sha>, entrypoints: [...]}
runtime:                    # M1.2.1.2 — 哪个机器的哪个 docker container
  slurm_jobid: '<job>'
  container: <prefix>_e2e_flow_<run6hex>
  ports: {router: 8101, worker: 8102, etcd: 8103}   # the band is a parameter
  endpoint: http://<ip>:<router port>
  transport: <spur | srun | local>
  started_at: '<ISO 8601, UTC>'
```

### 2.2 The absolute-path rule does not apply to this record

Carried validators inherit a rule — *no absolute host path in a handoff* —
justified by *"the seal refuses the whole delivery over one"*. **Measured
against the framework: that premise is false.**
`handoff/store.py` reads `# locality.check — NOT CALLED`, and `:494` gives
the reason: the shape heuristic read an HTTP access-log line as a filesystem
path and refused a correct artefact, **97% false positive on a real kit**.
Corroborated from the other side — a sealed `deploy_kit` carries absolute site
paths in five content files and sealed cleanly.

This matters here and not only as wording: `environment.schema.json` **requires**
`model_path`, which is an absolute path by nature. A validator carrying the rule
forward verbatim **rejects every conforming handoff in this package**, which is
how it was found, on a fixture.

So: keep the rule on its own merit — portability, a script carrying one host's
directory does not run on the next host — and **scope it to executable and
generated content (`.py`, `.sh`, `.json`, `.jsonl`), skipping the environment
record.** Do not justify it by the seal.

`environment.md`, where a packup layout still wants one, becomes a **rendering**
of this document, not the record. Today it is checked by three regexes
(the deploy stage's original `check_deploy_kit` body), which is
exactly what M1.1.1 objects to.

---

## 3. Schemas — `assets/schemas/`, read by producer *and* validator (G2, M3.6)

> *"所有结构化的文档，尽量有自己的json schema, 该schema同时暴露给producer & validator"*

### 3.1 `items_schema` does **not** satisfy this, and that is measured

`handoff/content.py` validates a file or tree item by building
`{item_name: <filename string>}` and checking *that* against `items_schema`. The
file's **contents are never read**. It is an admission check at the seal
boundary (`store.py`), it is never exported to a body, and **no
validator in any of the five demos imports `jsonschema`** — all of them
hand-roll (the analysis stage's original `check_workset_shape` body).

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
(`agent_sys/pyproject.toml`); 4.26.0 is importable here. Copy the idiom from
`agent_sys/spec_loader/validate.py` — `Draft202012Validator` plus a
`referencing` registry so schemas may `$ref` each other.

### 3.2a Every body is `#!/bin/sh` + `set -eu`, and the shebang is decoration

**agent_sys never consults a body's shebang.** It invokes one as
`["/bin/sh", entry]` — `validator/phase.py` and
`agent/backends/program.py`. On this host `/bin/sh` is **dash**:

```
$ /bin/sh -c 'set -euo pipefail; echo REACHED'
/bin/sh: 1: set: Illegal option -o pipefail     rc=2
```

So a body written `#!/usr/bin/env bash` + `set -euo pipefail` **exits 2 on line
1**, the phase reports UNREACHED rather than a verdict, and the failure reads as
the validator's rather than the shell's. Measured by m1 across all 31
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
`cli/main.py` exports the interpreter the run itself is using. A validation
zone gets a policy-derived `PATH` on which `python3` resolves to
`/usr/bin/python3`, which on this host has **no `referencing`** and therefore
cannot import `assets/lib/schema.py`.

Measured: the body dies with `ModuleNotFoundError` **before writing
`verdict.json`**, and the phase reports *"nothing was decided"* rather than a
verdict — **a validator that cannot start looks exactly like one that was never
asked.** Found by m5 driving their leaves through the graph; twelve of the
twenty-one validators had it, across all five modules.

**`${VAR:-default}` and `${VAR-default}` are not the same test, and the colon is
the one that disarms a guard.** m1's, found while wiring
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

The same distinction one level up, and the package owner shipped it as a live
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
not.** Found by m4 running the package owner's own new checker against their own agent
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
(the kernel-optimisation stage's original `check_workset_shape` readme). This has
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
weakens it says so to the package owner.

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

Add the package owner's: `schema.py` carried a comment asserting no schema used `$ref`
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
told the package owner *"nothing else of mine reads one rule from two places"*, then
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

m1's, committed **while writing the fix for the same class**. They
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

another owner's, and it is the rule I would want the next effort to start with.
**Inference across an ownership boundary was wrong every time it was tried
today; measurement was right every time.**

The expensive instance was the package owner's. Rung 0 refused three times at
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

### 4.4 A check that happens to be right and cannot be wrong is not a check

**m3's, and it is the sentence the rest of this section is
instances of.** They arrived at it after finding four in their own stage in one
day, each returning the correct answer for a reason that had nothing to do with
the subject:

| instrument | why it could not be wrong |
|---|---|
| `_observed` in the abort gate | read seven `E2E_*` names **nothing declared** |
| the report's `environment` block | **transcribed** the premise it was meant to test |
| `min_shapes` | counted shapes that were **never timed** |
| the `--environment` fallback | compared a document **with itself** |

Three of the four were **green**. None was a bug in the ordinary sense; each was
a working mechanism pointed at nothing.

m3's own correction to what they first proposed as the lesson is worth keeping,
because it is the difference between the small version and the general one:

> I told you the durable lesson was *"a paragraph asserting a behaviour is not
> one"*. It was the smaller version — that one is about **docstrings**, and
> every instance since has been about something **a machine reads**: a JSON
> field, a flag, a mount, a variable name.

**The operational form is the same question in every case: what would this
report if the subject were broken?** If the answer is "the same thing", the
instrument is decoration regardless of how carefully it was written.

### 4.4.1 A fixture that is more convenient than production tests the fixture

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

#### Eight ways a check can be satisfied without being a check

Each of these was paid for once here. The incident is not the durable part; the
rule is.

1. **The same root cause wears several faces, and some of them invent a
   problem.** Before naming a defect, check whether what you are looking at is
   the instrument rather than the subject.
2. **A paragraph asserting a behaviour is not that behaviour.** Prose in a
   readme is not a control; only something that can fail is.
3. **A search that can find itself returns a failure that looks like a clean
   result.** Never compose a check out of a pattern that appears in the command
   running it.
4. **A reader whose success path was never exercised** proves nothing when it
   passes: exercise the path that is supposed to work, not only the one that is
   supposed to fail.
5. **A refusal you agree with is a refusal nobody audits.** A verdict that
   confirms what you already believe gets the least scrutiny and deserves the
   same as any other.
6. **A bar can be neutralised from three distances — in the schema default, in
   the step's args, and in the launch line — and only the nearest is visible to
   the person who owns it.** Resolve every `${...}` against the command that
   will actually run, not against the file that declares it.
7. **Prove the path is live before believing a PASS is a hole.** A check that
   never executed and a check that executed and found nothing are the same
   green.
8. **When you re-run a probe narrower, keep the broader answer.** The narrowing
   discards exactly the information that would have contradicted it — and a
   narrow probe's answer can survive several readers, because the rest of its
   output is right.

### 4.5 Where a verdict's *author* is recorded, which is one place only

**`handoffs/<id>/v<N>/validation.yaml`.** Per handoff version, it holds one row
per validator: `validator`, `result`, `strength`, `dimension`, `task_id`, the
zone path and `at`.

**Nothing else in a run tree carries a validator's name.** Measured
while building `assets/lib/replay_root.py`:

- the validation zone holds `args.json`, `inputs.json`, `materials.json`,
  `verdict.json` — and **`verdict.json` is keyed by handoff id, not by
  validator**, so two validators' verdicts are distinguishable only by
  fingerprinting their `args.json`;
- `store/event/*.json` records `phase_done` and friends with `task_id` and
  `handoff_id`, and no validator name;
- `store/task/*.json` records `closure`, `agent_spec`, `history[].outcome` —
  the task, not the checks it ran.

**So the answer to *"did validator X pass on handoff Y in run Z"* has exactly
one source, and grepping run output for `REFUSED` is not it** — the package owner did
that and nearly reported four false failures, because a
`validator_report.txt` is written by the bodies that adopted `write_report` and
by no others, and its heading is a rendering rather than the record.

**The consequence that bit first:** *"this artefact passed three times"* is not
a statement about a run finishing, and it is not even a statement about three
green verdicts. It has to name **which validators**. `replay_root` found
`kernel_optimization` graded by `check_environment` alone in one run and by
three validators in two others — all green, all "passed", and **graded by a
different ruler**. Counting those as three would promote an artefact whose
stability was measured against a moving standard, so **the validator set is part
of the verdict** and a change in it is reported rather than averaged away.

### 4.6 A relative check cannot detect a fault both sides share

**m5, measured standalone on constructed inputs.** Not a fact about
one validator — a fact about **every cross-comparison in this package**, and the
reason some of them need an absolute bar beside them rather than a tighter
tolerance.

`stock_vs_m2_block` asks whether m5's stock arm reproduces m2's
`profiling_mode_off` bench within 10 %. Three cases through the real function:

| case | reconciliation | why |
|---|---|---|
| both engines healthy | **passes** | correct |
| one engine in eager decode | **refuses** | correct — throughput `-78.6 %`, ITL `+30.8 %` |
| **both engines in eager decode** | **passes** | the two arms genuinely agree, about a number 4.6× wrong |

**The third case is indistinguishable from the first.** Measured:

```
case1 keys     == case3 keys     : True
case1 verdicts == case3 verdicts : True
block mentions the graph ceiling : False
```

Same fields, same three `within_tolerance: True`, differing only in absolute
values that nothing bars. **No reader and no downstream consumer can tell them
apart.**

**And no comparison can be made to see it, including a better one.** The obvious
repair is to compare the engine configuration as well as the numbers. It does
not work:

| case | m2 ceiling | stock ceiling | argv differ? |
|---|---|---|---|
| healthy | 512 | 512 | **no** |
| both eager | 8 | 8 | **no** |
| one eager | 512 | 8 | yes |

In the case that matters **the two sides agree** — 8 == 8. A cross-comparison
sees nothing because nothing disagrees. Only an **absolute** bar catches it:
*the graph ceiling must be at least the concurrency the load achieved.*

**The operational form:** for every check that compares two things, ask **what
fault would both sides have?** If the answer is not "none", that fault needs a
bar on each side, not a comparison between them. It is §4.4's question — *what
would this report if the subject were broken?* — asked of a check whose subject
is a **pair**.

The implementation is `assets/lib/graph_ceiling.py`: one body, two call sites
(m2's bench, m5's arms), because this is the same predicate asked of different
engines and not one bar written twice — the distinction `min_requests` cost us.

**The detail that makes it a design lesson rather than a bug report:**
`merge_profiling_evidence/merge.py` already carries `env/` for every part, with
a comment giving the exact reason — *"the load configuration in it is what makes
the two benches comparable, and **is not recoverable from the numbers**."* The
data has been in both handoffs all along, under the same filename, **read by
nothing.** Writing down why a field must travel is not the same as wiring a
consumer to it.

### 4.1 Shared validators are shared, not copied

`check_kernel_table` is **one** definition used by m2 and m3 (M3.5). The two
demos each carry a copy today with different `args`, which is one of the three
seams recorded in `handoff.analysis.md`. Same for the workset validators: they
live with m3's `operator_workset` and m4 **references** them (M4.4).

---

## 5. The runtime environment (G5.1, rule 7)

**Modules 1–4 share one container on one held node.** m1 brings it up and
records it in `environment.runtime`; m2, m3 and m4 exec into it.

**"One container" means one DEPLOYMENT container. It does not forbid an
ephemeral measurement container**, and the two must not be conflated — that
conflation stopped rung 0.

A deployment container serves the model, m1 owns its lifetime, and nobody else
starts or stops one. A **measurement** container is apparatus: it runs a kernel,
prints a number and dies. In a **mock** chain no deployment container is ever
brought up, so a check that can only re-measure inside one cannot run at all —
and the mock e2e is a deliverable.

m3 established the shape first (`measure_in_container.sh`, and read 325–392
before copying it — the lifecycle notes there were learned the hard way):

```bash
: "${E2E_MEASURE_CONTAINER:=yihou_m3_measure_$$}"
docker run --rm --name "$E2E_MEASURE_CONTAINER" …   # trapped: reclaim, then rm -f
```

Own name, `--rm`, removed in a trap, started from the image the record names.
Never a name you did not create; never `docker rm -f` on anything else.

So any body or validator that must measure:

1. **prefer the container the record names**, when it is genuinely running — in a
   real chain that is the faithful thing, same engine state and same pins;
2. **fall back to an ephemeral one from the same image** when it is not;
3. **say in the report which of the two it used.** A number re-measured in a
   fresh container and one re-measured inside the live deployment are different
   claims, and a reader must not have to guess which they are holding.

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

### 5.2 Shared-host rules

**Every identifier this package binds on a shared host is a parameter, not a
constant** — container names, ports, work roots, the served model name. A
constant here is one machine's answer shipped as everyone's, and two runs on one
host then collide on a name neither of them chose.

**Before any bring-up, check the cards and the container list**, and check both:
a card that reads idle is not the same claim as a host with nothing running on
it, and a container that holds no memory yet may still be about to load.

**Who may stop what is site policy and is deliberately not written here.** This
document specifies the contract between the five stages; an operator's authority
over other tenants on a shared machine comes from the site, not from a package.

### 5.3 What a mock may and may not put in a handoff

**A mock may obtain a real fact by a route the producer does not use. It may not
assert a fact the producer does not have.**

Both halves were decided on the same afternoon, on two cases that
look identical from a distance and are not:

| field | real producer | may the mock write it? |
|---|---|---|
| `base_sha256` | `60_write_handoff.py` hashes the stock file from the engine tree | **yes** — `mock_adapt` extracts the same file from the image with `docker create` + `docker cp` and hashes that. **A different route to the same fact.** |
| `public_symbol` | the workset says `null` for a `call_site_fragment` operator | **no** — writing one makes the artefact assert something the real thing does not say, and arms `check_patch_live`'s `first_call` regex against a symbol that is not at the call site. |

The test is **not** where the fact comes from. It is whether the real producer
asserts the same fact. m4 and m5 both refused the second case unprompted and both
routed the first rather than deciding it, which is the reason the line got drawn
before either was landed.

**A mock that takes the first route must record which route it took**, naming the
image when it extracted from one. Two artefacts that agree on a hash obtained two
ways are stronger than either alone; two that agree and cannot say which route
they took are `protocol.timing` again — a field copied faithfully and understood
by nothing (T43).

**And the degraded route stays.** When no engine tree is reachable and no node
answers, `mock_adapt` hashes the replacement and *says so in the handoff*. That
note is why the 12:12 refusal was diagnosable in one read rather than an hour, and
**a mock that cannot run on a login node without an allocation is worse than one
honest about a degraded hash.** Delete the fallback and the mock acquires a
dependency the thing it is mocking does not have.

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

## 8. Deferred — recorded in `todo.md`, not built

`check_trace_coverage` against sglang source (M2.8.2) · `vendor_tuned` bucket
(M3.3) · one handoff per operator (M3.7.7) · AI-led + program-fixed analysis
(M3.8) · the patch mechanism should hack the registry rather than bind-mount
(M5.3 — *"但现在就这样吧"*, so `overlay_files` stays) · permission and
visibility management for the shared container (rule 7).

---

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

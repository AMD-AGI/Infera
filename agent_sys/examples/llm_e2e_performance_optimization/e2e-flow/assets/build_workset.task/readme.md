# build_workset — build one runnable optimisation workset per candidate operator

You are handed a resolved list of candidate operators and a profile. You produce
one `operator_workset`: for every candidate, a runnable definition, at least
three shapes, a correctness test and a performance test — and the **measured**
result of running both.

Everything mechanical is already a program. **Your job is to run the STEPS below
in order and to supply the two things a program cannot: a correctness reference
that is obviously right, and the identity of the incumbent implementation a
speedup will be measured against.** Do not invent a method, do not write your
own test runner, and do not edit the harness. Each step names the command and
the condition that decides whether it worked.

## What you must not do, and why each one has bitten

- **Do not edit `run_correctness.sh`, `run_performance.sh` or `_common.py`.**
  They are copied from the package and `check_workset_shape` compares the copy
  byte for byte. An agent that writes its own oracle controls its own result,
  and every number after that is unfalsifiable. This is the single rule that
  lets the next stage take your numbers as ground truth instead of re-measuring.
- **Do not write a reference by reading the kernel.** That is how a correctness
  gate comes to agree with a bug. Import the framework's own — it is usually in
  the project's test suite, and finding it is STEP 4's real work.
- **Do not use the same function for `reference` and `baseline`.** The reference
  is slow and obviously right; the baseline is the fast incumbent the served
  engine calls today. Identical means the speedup is 1.0 by construction, and
  `check_workset_shape` refuses it.
- **Do not put a number in `targets.speedup_baseline`.** Magpie's
  `Avg time (us)` is an in-service average across mixed batch sizes and
  contention with every other kernel; the standalone baseline is what
  `evidence/performance.json` records. They can differ severalfold.
- **Do not fill `solution` or `evaluation` in a workload line.** Those slots
  belong to the consumer. A workset that fills them asserts an answer it has not
  measured.

## STEPS

Throughout: `PKG="${AGENT_SYS_TASK_PACKAGE:?}"` and `OUT="${AGENT_SYS_OUTPUT_OPERATOR_WORKSET:?}"`;
`WS="$OUT/items/codes"`.

### STEP 1 — read the inputs

```sh
cat "$AGENT_SYS_INPUT_OPERATOR_IDENTITY/items/text.json"
cat "$AGENT_SYS_INPUT_OPERATOR_IDENTITY/items/env/environment.yaml"
ls "$AGENT_SYS_INPUT_PROFILING_EVIDENCE/items/"
```

**Acceptance:** you can name, for each operator, its `logical_operator`, its
`source_owner`, its `repository_language`, and how many shapes its `cases` list
carries. Note every operator with fewer than three — STEP 5 is where you fix
that, and it is easier to plan for now.

### STEP 2 — scaffold

```sh
python3 "$PKG/assets/build_workset.task/scaffold.py"
```

**Acceptance:** exit 0, and `ls "$WS"` shows `workset.yaml`, `definitions/`,
`workloads/`, `environment.yaml`, `run_correctness.sh`, `run_performance.sh`,
`_common.py`. The two `.sh` files are executable. The command prints which
definitions it scaffolded; every one of them now contains a `TODO(build_workset)`
sentinel in `reference` and `baseline`.

Re-running this is safe: it refreshes the scaffold and leaves any definition
whose sentinels you have already replaced.

### STEP 3 — find the framework's own reference and the incumbent call

For each operator, in the checkout named by `edit_target.repo_root_var`:

```sh
grep -rn "<the entry function from edit_target>" --include='*.py' "$REPO" | head
grep -rln "<the operator's own name>" "$REPO"/*test* "$REPO"/**/test_*.py 2>/dev/null | head
```

**Acceptance:** for each operator you can name **two** symbols and cite the file
and line for each —

1. the **reference**: a slow, obviously-correct implementation, ideally the one
   the project's own test checks this kernel against;
2. the **baseline**: the call the serving path actually makes today, which is
   the function `edit_target.entry_function` sits under or dispatches to.

If you cannot find a reference for an operator, say so in that operator's
`reference.rationale` and write one in plain PyTorch from the *published
semantics of the operation*, never from the kernel source. Recording that you
did is not optional — it is what tells the next reader the gate is weaker.

### STEP 4 — fill the Definition

For each `definitions/<op_type>/<name>.json`: replace the `reference` and
`baseline` sentinels, and fill `inputs`, `outputs` and any missing `axes`.

Both are **Python source strings ending in `def run(*args, **kwargs)`** —
flashinfer-bench's own convention; `../../../../../rank0/definitions/` holds two
worked examples and is the thing to imitate. `inputs` maps a name to
`{shape, dtype, description}` where each shape entry is an axis name or an
integer.

```sh
python3 - <<'PY'
import ast, json, pathlib, sys
bad = 0
for p in pathlib.Path("<WS>/definitions").rglob("*.json"):
    d = json.loads(p.read_text())
    for key in ("reference", "baseline"):
        src = d[key]
        if "TODO(build_workset)" in src:
            print(f"{p.name}: {key} is still the sentinel"); bad += 1; continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"{p.name}: {key} line {e.lineno}: {e.msg}"); bad += 1; continue
        if not any(getattr(n, "name", None) == "run" for n in tree.body):
            print(f"{p.name}: {key} defines no top-level run()"); bad += 1
    if d["reference"].strip() == d["baseline"].strip():
        print(f"{p.name}: reference and baseline are identical"); bad += 1
    if not d["inputs"]:
        print(f"{p.name}: inputs is empty"); bad += 1
sys.exit(1 if bad else 0)
PY
```

**Acceptance:** that command exits 0.

Also update each operator's `reference` and `baseline` blocks in `workset.yaml`
to `kind: imported` with the `module`, `symbol`, `source_file` and `line` you
cited in STEP 3, and a one-sentence `rationale`. Use `kind: written` with a
`path` only where STEP 3 forced you to.

### STEP 4a — declare the integration point

M5.1.1. `integration` is **not** `edit_target`: that one says where an optimiser
edits, this one says where a replacement is *installed* and what it may not
change. m5's `apply_patch` is a program only because this block exists — with
it, applying is a copy from a named source to a named target; without it, m5
reads an optimisation report and guesses.

Per operator, fill in `workset.yaml`:

- `target_files` — paths inside the engine, relative to `edit_target.repo_root_var`.
  The scaffold seeded these from `editable_sources`; widen or narrow deliberately.
- `public_symbol` — the entry point a replacement must still provide. The file
  may be rewritten wholesale; this must survive it.
- `signature` — the call signature that must hold.
- `invariants` — replace the sentinel. **What a replacement may not change,
  beyond the signature**, and only things you have checked. The worked example
  is the one to have in mind because none of it is inferable from the signature:
  the production call site for a sampler softmax is `logits[:] = torch.softmax(...)`,
  so the replacement must **write in place into a caller-provided `out`** — one
  that allocates is not substitutable there and would pass every correctness
  gate. An invariant nobody verified is worse than a missing one, because m5
  will rely on it.

```sh
python3 - <<'PY'
import pathlib, sys, yaml
doc = yaml.safe_load(pathlib.Path("<WS>/workset.yaml").read_text()); bad = 0
for op in doc["operators"]:
    i = op["integration"]
    if not i["target_files"]: print(f"{op['operator_id']}: no target_files"); bad += 1
    if not i["public_symbol"].strip(): print(f"{op['operator_id']}: no public_symbol"); bad += 1
    if any("TODO(build_workset)" in v for v in i["invariants"]):
        print(f"{op['operator_id']}: invariants still the sentinel"); bad += 1
sys.exit(1 if bad else 0)
PY
```

**Acceptance:** that command exits 0.

### STEP 5 — reach three shapes, and label the ones you added

`check_workset_shape` requires **at least three shapes per operator, and every
shape the operator is to be optimised for**. The scaffold carried across every
shape the profile observed. Where that is fewer than three, add cases that
exercise a different tile boundary or batch regime — and set
`observed: false` on each, with a `note` saying why you picked it.

Append one line to the operator's `workloads/<...>.jsonl` per shape you add, and
one entry to its `shapes` list in `workset.yaml`, **in the same order**.

```sh
python3 - <<'PY'
import json, pathlib, sys, yaml
ws = pathlib.Path("<WS>"); doc = yaml.safe_load((ws / "workset.yaml").read_text()); bad = 0
for op in doc["operators"]:
    lines = [json.loads(l) for l in (ws / op["workload"]).read_text().splitlines() if l.strip()]
    if len(op["shapes"]) < 3:
        print(f"{op['operator_id']}: {len(op['shapes'])} shapes, needs 3"); bad += 1
    if len(lines) != len(op["shapes"]):
        print(f"{op['operator_id']}: {len(lines)} workload lines vs {len(op['shapes'])} shapes"); bad += 1
    for i, (l, s) in enumerate(zip(lines, op["shapes"])):
        if l["workload"]["uuid"] != s["uuid"] or l["workload"]["axes"] != s["axes"]:
            print(f"{op['operator_id']}: line {i} does not match shapes[{i}]"); bad += 1
    if sum(1 for s in op["shapes"] if s.get("is_primary")) != 1:
        print(f"{op['operator_id']}: needs exactly one primary shape"); bad += 1
    if not any("performance" in s["role"] for s in op["shapes"]):
        print(f"{op['operator_id']}: no shape carries a performance role"); bad += 1
sys.exit(1 if bad else 0)
PY
```

**Acceptance:** that command exits 0.

### STEP 6 — export the KernelForge add-on

```sh
python3 "$PKG/assets/lib/forge_export.py" --workset "$WS"
```

**Acceptance:** exit 0, and for each operator
`operators/<id>/{run_forge.sh,invocation_spec.json,forge_task.yaml,tests/cases.json,scripts/forge_driver.py}`
exist, with `run_forge.sh` executable. Check the one-liner prints without
running:

```sh
"$WS"/operators/<id>/run_forge.sh --dry-run
```

It prints `kernel-agents forge-loop … --workspace <REQUIRED: the git checkout to
edit>`. **The placeholder is not a defect** — `--workspace` is required by
`forge-loop` and names the checkout it edits in place, which is a site fact this
workset cannot know, so it is refused rather than defaulted for the same reason
the measurement card is. Passing one substitutes it:
`run_forge.sh --dry-run --workspace <checkout>`. Without `--dry-run` and without
a workspace the script exits 2 and says so.

Do not hand-edit anything under `operators/`. If it is wrong, the base format is
wrong; fix that and re-run this step.

### STEP 7 — run the correctness test

**Do not run the entrypoints where you are standing.** You are on the host that
runs `agent-sys`, in a zone — not inside any container. That host has **no
torch**, measured: `spur exec <jobid> python3 -c "import torch"` →
`ModuleNotFoundError`. Only the containers have it. So `./run_correctness.sh`
run directly here fails at the first import, and on a host where it happened to
succeed it would measure on a card nobody chose.

Both steps go through one script, which starts an ephemeral measurement
container on the node and runs the entrypoint inside it:

```sh
"$PKG/assets/build_workset.task/measure_in_container.sh" "$WS" \
  ./run_correctness.sh --json evidence/correctness.json
```

**`E2E_MEASURE_GPU` must be set and it is not defaulted.** It is in your
environment if the run was launched with `--var measure_gpu=<n>`; check it
before you start, because the script refuses without it and you will have spent
the scaffold for nothing:

```sh
: "${E2E_MEASURE_GPU:?no measurement card — the run must pass --var measure_gpu=<n>}"
```

There is no fallback on purpose. Five owners share these nodes, and a card
someone else is serving from does not *fail* — it returns slower numbers that
`check_workset_runs` re-measures on the same card and agrees with. **A wrong
card produces a confident wrong answer, which is the one failure mode this
whole workset exists to prevent.**

**This is the same instrument `check_workset_runs` will use to re-measure you.**
That validator re-runs one shape through this same script and compares its own
number against your record; a producer measuring through a different arrangement
than the validator would not be re-measured, it would be re-*interpreted*. If
you find a reason to measure some other way, the validator is what you have to
change first, and that is a conversation with the leader rather than an edit.

You do **not** need `export PYTHONDONTWRITEBYTECODE=1` any more — the script
passes it into the container, along with `TMPDIR` and `TRITON_CACHE_DIR`, and
chowns the output back to you afterwards. It is set because the container runs
as root — it has to, since a framework compiling kernels on first call cannot
write its cache as anyone else — so any `__pycache__` left inside the handoff
would be root-owned and the runner could not copy or clean the output as its own
user. Measured: it does not fail on the run that creates it, only on the next
one.

**Acceptance:** exit 0. If it fails, read `evidence/correctness.json`: each shape
carries its `snr_db` and a `failure`. A failure here is almost always one of
three things and they are distinguishable —

- **the reference is wrong** — a low SNR on *every* shape, usually because the
  reference and the kernel are given differently pre-processed inputs. Check any
  shuffling, sorting or packing the real call path applies: the kernel gets the
  transformed tensors and the reference must get the originals.
- **`build_inputs` cannot build a dtype** — an explicit `SystemExit` naming the
  dtype. A packed or quantised input needs its own builder; put it in the
  Definition's `inputs` as a `scalar` where it is not really a tensor.
- **the gate is genuinely too tight** — a near-miss on a low-precision operator.
  Say so in the operator's `notes`; **do not lower `snr_db` to make it pass.**

### STEP 8 — run the performance test

Same instrument, same container image, same card — STEP 7's reasoning applies
unchanged and matters more here, because this step produces the number every
later stage divides by:

```sh
"$PKG/assets/build_workset.task/measure_in_container.sh" "$WS" \
  ./run_performance.sh --json evidence/performance.json
```

**Acceptance:** exit 0, and in `evidence/performance.json` every shape has five
`per_group_ms` entries and an `rsd` at or below 0.10. If `rsd` is above it the
node was not quiet — re-run rather than record it. A noisy baseline is worse
than no baseline: an optimiser working against it takes the first candidate that
lands on a fast sample for a win and chases noise for hours.

Then record two things in `workset.yaml`.

The evidence block:

```yaml
evidence:
  correctness_report: evidence/correctness.json
  performance_report: evidence/performance.json
  measured_on: {node: <hostname>, gpu_arch: <arch>, container: <name>, at: <ISO8601>}
```

And **transcribe the noise floor**. `run_performance.sh` computed it and wrote
it to `evidence/performance.json` as `noise_floor`; copy that number into every
operator's `noise_floor`. Do not round it down and do not substitute 1.05.

```sh
python3 -c "import json;print(json.load(open('evidence/performance.json'))['noise_floor'])"
```

Copy it into **two** places: every `operators[].noise_floor` **and**
`ground_truth.noise_floor`. Forgetting the second leaves the workset-wide bar at
the scaffold's placeholder, below every operator's own floor —
`check_workset_shape` refuses that, and it caught it in this package's own mock
producer before it caught it anywhere else.

It is `1 + 2.83 x rsd_max` — the two-sample 2-sigma separation at the spread
this run actually saw, because two measurements each with relative sd `s` differ
by more than `sqrt(2)*z*s` by chance alone. **m4 must not pick this number
itself**: a consumer choosing its own floor is a consumer choosing when to call
its own result significant. `check_workset_shape` compares the transcription
against the file.

### STEP 9 — validate what you built, before it is graded

```sh
python3 "$PKG/assets/lib/schema.py" --schema workset --doc "$WS/workset.yaml"
```

**Acceptance:** exit 0. It prints every problem at once with the field path, so
fix them in one pass rather than one run each.

### STEP 10 — the handoff's own README

`content_type: code` requires exactly three sections and refuses the seal
without them: **Purpose**, **Interface**, **Boundary**. Write
`"$OUT/README.md"` with those three headings.

- *Purpose*: what this workset is for, and which profile it came from.
- *Interface*: the two entrypoints, their four flags, and the statement that
  they are protected.
- *Boundary*: what it does **not** carry — every operator you excluded and why,
  every shape you added rather than observed, every gate you believe is weak,
  and any operator whose `status` is `partial` with the field that is missing.

**Acceptance:** all three headings present, and the *Boundary* section names at
least the excluded operators. A workset whose boundary section is empty is
claiming it has no limits, which has never been true of one.

## Done

`check_environment`, `check_workset_shape` and `check_workset_runs` grade this.
The last one **re-runs at least one shape itself** and compares against
`evidence/performance.json`, so a recorded number that does not reproduce on
this hardware fails — which is the point, and is why STEP 8 says re-run rather
than record a noisy result.

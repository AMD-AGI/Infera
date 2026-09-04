# `optimize_kernel` — one kernel, optimised, and proven with the workset's own tools

You are the **wrapper** between two systems that do not know about each other.
On one side `agent_sys` hands you an `operator_workset` and a `deploy_kit` and
expects a `kernel_optimization` back. On the other, **KernelForge** is an
autonomous optimisation loop with its own CLI and its own output tree.

**Everything you need to run is already a script.** Mission G4.2: the steps below
each have an executable and a stated acceptance criterion, and your job is to run
them in order, read what comes back, and stop when one refuses. **You sequence
and you read. You do not invent a method, and you do not substitute a number.**

Where a step's acceptance criterion fails, the honest outcome is a handoff that
says so. A run that aborts and records why is a *successful* run of this task; a
run that produces a plausible number is not.

---

## The one rule that reverses what the previous package said

The package this task descends from told you: *"Your baseline is the number you
measured on this host. Never the number printed in the workset."* **That is no
longer true.** Mission M4.3.5:

> 优化任务的 ground truth 本身就应该严格的从 workset 中来，如果最基础的硬件、
> 优化前提不一样，直接报错 abort。软件环境不太一样可以报 warning。

So:

- **The denominator of every ratio you report is the workset's own recorded
  baseline.** You do not measure a baseline and you do not choose one.
- **If the hardware or the optimisation premise differs from the workset's, you
  abort.** Not "adjust", not "note and continue". A speedup measured against a
  different architecture is not a smaller result, it is the answer to a
  different question, and it reads as entirely legitimate to everyone
  downstream.
- **If the software environment differs, you warn** — and the warning goes in
  the handoff, in `premise.verdict.warnings[]` and in
  `run_environment.warnings[]`, so it travels to m5.

STEP 2 decides this for you and writes the answer down. Do not second-guess it
in either direction.

---

## Where these steps run: **inside m1's container**

CONTRACT §5: *"Modules 1–4 share one container on one held node. m1 brings it up
and records it in `environment.runtime`; m2, m3 and m4 exec into it."*

**This is not a detail you can defer.** There is no host anywhere on this
cluster with torch — `spur exec <job> python3 -c "import torch"` is
`ModuleNotFoundError`, measured; only the containers have it. `KFO_PYTHON`
defaults to `/opt/venv/bin/python3`, which is a path **inside the image**. So a
step run on the host either dies immediately or, worse, measures nothing useful.

STEPs **3, 4, 5 and 6** need the container: the campaign needs the GPU and the
ROCm stack, the two measurements need torch, and STEP 6 hashes the stock engine
file out of the container tree. STEPs 1, 2 and 7 read JSON and run fine either
side.

Use the wrapper, which reads the container out of the same record every other
step reads:

```sh
S="$AGENT_SYS_TASK_PACKAGE/assets/optimize_kernel.task/steps"
"$S/run_in_container.sh" --workdir "$W" '<the step command>'
```

It **execs** into the recorded container when that container is running, and
never starts or removes *the deployment* — m4 did not create it. **When the
record's container is not running it starts an ephemeral one of its own** from
the image the record names: `--rm`, self-named, removed in a trap, and never a
name it did not create. That is the leader's ruling of 2026-09-04, because in a
mock chain nobody brings the deployment up and a step that could only run after
a real one could not be in the mock e2e at all. §5 conflated *the deployment*,
whose lifetime is m1's, with *the measurement apparatus*, which belongs to
whoever measures.

**The two are different claims and the handoff says which you got.** A number
measured inside the live deployment carries the engine's state; one measured in
a throwaway carries only the image's. The wrapper prints which, and records it
as `premise.observed_runtime.mode`.

Two refusals you will meet, both deliberate:

* **`HIP_VISIBLE_DEVICES` has no default.** This host is shared and cards 0–3
  have been a co-tenant's, so the wrapper refuses rather than picking one.
* **A card outside the container's own pin is refused, not substituted.** The
  pin is an *environment default* — m1's kit starts the container with
  `--device /dev/dri` whole and narrows it only with
  `--env HIP_VISIBLE_DEVICES` — so `docker exec -e HIP_VISIBLE_DEVICES=4` would
  silently widen it back out and measure on a card the deployment was never
  given. You asked about one card; an answer about another is not a smaller
  answer.

**If you are already executing inside the container, run the step directly** —
the wrapper is for crossing the boundary, not for being past it.

---

## STEPS

Run these in order. Each is `$AGENT_SYS_TASK_PACKAGE/assets/optimize_kernel.task/steps/<name>`,
and STEPs 3–6 go through `run_in_container.sh` as above.
Everything they write goes under `$KFO_SCRATCH_ROOT/<run>/`, which is local disk.

### STEP 1 — read the inputs and pin the run

```sh
./steps/10_read_inputs.py --out "$W/state/inputs.json"
```

Resolves the workset root, the operator, the `deploy_kit`'s `environment.yaml`,
and the entrypoints and protocol the later steps use. Nothing is chosen here;
everything is read.

**Acceptance:** exit 0, and `state/inputs.json` names exactly one `operator_id`,
one `correctness` and one `performance` entrypoint, and ≥ 3 performance shapes.
If it names two operators, the workset carries more than one and this task
optimises one — pass `--operator <id>`.

### STEP 2 — the premise gate. **Before anything is spent.**

```sh
./steps/20_premise_gate.py --inputs "$W/state/inputs.json" --out "$W/state/premise.json"
```

Compares the workset's `ground_truth.environment` against the environment this
run is actually in, field by field, using the workset's own
`abort_on_mismatch` / `warn_on_mismatch` lists.

**Acceptance:** exit 0 and `premise.json` has `"held": true`.

**On exit 2 — an abort — stop here.** Do not run forge. Do not measure anything.
Go straight to STEP 6 and write the handoff with the premise's own verdict in
it: `held: false`, `aborted_on` naming the fields, and **no
`evidence.performance.claim` at all**. The schema will not let you write a claim
in that state and you should not want to.

In `e2e-flow` this gate should normally pass, because m1 mints the environment
record and m2, m3 and m4 all run in the container m1 brought up. A failure here
usually means something is genuinely wrong with the wiring, not with the
hardware.

### STEP 3 — run KernelForge, through the workset's own one-liner

```sh
./steps/30_run_forge.sh --inputs "$W/state/inputs.json" --workdir "$W/forge"
```

The workset carries `operators/<operator_id>/run_forge.sh` — its
`forge.one_line`, generated by m3 from the same definition the tests use
(M3.7.6). **Run that. Do not assemble a `kernel-agents forge-loop` command
line yourself.** The workset knows its own target functions, its own driver and
its own architecture; a hand-written invocation is where `--gpu-target gfx942`
got copied onto a gfx950 host in the previous round and tuned for a chip nobody
was measuring.

The wrapper sets the five environment facts that otherwise fail *silently* —
see the pitfall table below — and copies the workset into a scratch git
repository first, because forge commits its keeps and
`$AGENT_SYS_INPUT_OPERATOR_WORKSET` is a sealed artefact.

**Acceptance:** exit 0 and `forge/forge_result.json` exists. `improved: false`
is a legitimate result and is **not** a failure of this step.

`KFO_MOCK=1` skips the campaign; the step then copies the seed to
`forge/optimized_kernel.py` behind a banner saying so and writes a
`forge_result.json` of nulls. A mock is not a small campaign, and every later
step and the schema itself treat it differently.

### STEP 4 — the workset's **correctness** test, on the candidate

```sh
./steps/40_correctness.sh --inputs "$W/state/inputs.json" \
    --candidate "$W/forge/optimized_kernel.py" --out "$W/state/correctness.json"
```

The workset's `entrypoints.correctness.cmd`, verbatim, with the candidate
selected. Not forge's internal SNR gate — that is forge grading its own
homework, and it is recorded separately.

**Acceptance:** exit 0, and `correctness.json` has `"passed": true` with an
entry for **every** shape the workset declares for correctness.

**A failure here ends the run.** Go to STEP 6 and write the handoff with the
failure in it. **Correctness is not a percentage**: a kernel that is right on
two shapes of three is wrong, and it must never reach STEP 5. That ordering is
the reason these are two steps and not one.

### STEP 5 — the workset's **performance** test, on the candidate

```sh
./steps/50_performance.sh --inputs "$W/state/inputs.json" \
    --candidate "$W/forge/optimized_kernel.py" --out "$W/state/performance.json"
```

The workset's `entrypoints.performance.cmd`, under the workset's own
`protocol` — you do not choose `groups` or `iters_per_group`, and comparing a
5-group sample against a 3-group one is not a comparison.

**Acceptance:** exit 0, and `performance.json` carries a figure for every
performance shape with `rsd` recorded per case.

Read the `rsd` before you read the medians. The baseline side is tight (~2% on
a steady node) and an optimised kernel has measured ~8% round to round on this
hardware — unexplained since 2026-08-31. A single sample of the loose side is
not a measurement.

### STEP 6 — write the handoff

```sh
./steps/60_write_handoff.py --inputs "$W/state/inputs.json" --state "$W/state" \
    --out "$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION"
```

Assembles the packup and the document. It computes the ratios **for** you, from
the workset's baseline and STEP 5's measurement, and it will refuse to write a
claim when the premise aborted, when forge was mocked, or when correctness
failed. That refusal is the schema's, not the script's opinion.

**Acceptance:** exit 0, and the output holds exactly one
`items/codes/<operator>.packup_<YYYYMMDD>/`.

You still write four documents yourself — `README.md`, `REPRODUCE.md`,
`environment.md`, `notes.md` — because they are the part a cold reader needs
and no script knows what surprised you. STEP 6 writes their skeletons and
leaves them for you. See *Writing the four documents* below.

### STEP 7 — grade yourself before you finish

```sh
./steps/70_selfcheck.sh --handoff "$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION"
```

Runs `check_optimization_shape`'s **own body** over what you just wrote, plus the
free half of `check_speedup_substantiated`. Same code that will grade you, so a
pass here is not a promise and a failure here is certain.

**Acceptance:** exit 0. If it names a problem, fix the problem — do not adjust
a number so the check stops complaining.

---

## Writing the four documents

STEP 6 leaves skeletons. Fill them.

- **`README.md`** needs a `## Result` section, and its first line must say what
  happened in words a reader cannot mistake. If forge was mocked it says **MOCK
  RUN — no optimization was performed**. If the budget was degraded it says
  **SMOKE TEST — degraded budget**. If the premise aborted it says so. A run
  that could be read as a complete campaign must be impossible to read as one.
- **`REPRODUCE.md`** — ordered, copy-pasteable commands and an *Expected output*
  section. It is checked for a floor of real command lines.
- **`environment.md`** — a rendering of the environment record, not a second
  source of truth. Numbers in it, or the check fails.
- **`notes.md`** — the gotchas and the wrong turns, **including yours**. This is
  the most valuable file in the kit and the one nobody can generate.

Be honest in `## Boundary` about what you did not measure, what shapes you did
not cover, and what you could not explain. An honest boundary is worth more than
a confident one, and the expensive validator re-measures in public.

---

## The pitfalls. Four of these fail silently

Ordered by how quiet they are. Each has already cost a day.

| # | trap | symptom | handled by |
|---|---|---|---|
| 1 | **`$TMPDIR` points at a directory that does not exist** | *every* HIP kernel launch segfaults with **no output**, while `torch.cuda.is_available()` still returns `True`. Cost 25 minutes of bisection down to a hand-written HIP program on 2026-09-02, because nothing in the failure names a filesystem | the agent spec's `env`, and every step script creates it |
| 2 | **`--max-hours <= 2.0` silently degrades the campaign** | forge drops Analysis to static-only and the implementer turn cap falls 500 → 100. Nothing warns you; the campaign runs and produces a report (`kernel_agents/cli.py:47`, `:1391` — strictly greater) | STEP 3 refuses to pass a value it was not told to, and sets `degraded` |
| 3 | **the floor on `--max-hours` is 1.0 and is enforced** | `click.BadParameter` below it. There is no five-minute forge run | STEP 3 clamps and records that it did |
| 4 | **writes to an NFS `$HOME` fail, two of the three quietly** | `~/.triton` and the experience KB fail silently; a `~/.cache` write once killed an sglang scheduler with an unremarkable `PermissionError` | `TRITON_CACHE_DIR` and `KNOWLEDGE_LOCAL_ROOT` in the agent spec. **Do not unset them and write nothing under `$HOME`** |
| 5 | **`rocprof-compute` dependency conflict degrades profiling** | one line of `kernel-agents status`, and nothing else. Forge pins `astunparse==1.6.3`/`kaleido==1.3.0`; ROCm 7.2's rocprofiler-compute needs `1.6.2`/`0.2.1` | run `pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt`; the conflict warning pip prints is expected |
| 6 | **`kernel-agents list`/`show` look in the wrong directory** | `No experiments found`, and you conclude the campaign produced nothing | pass `--dir "$W/forge/forge_experiments"` |
| 7 | **a non-clean `CLAUDE_CONFIG_DIR` crashes forge's backend probe** | `AttributeError: 'list' object has no attribute 'get'`. At least this one is loud and immediate | STEP 3 exports a clean empty one under `$KFO_SCRATCH_ROOT`, for the nested process only |

---

## Rules about this shared machine

- **Create only under `$KFO_SCRATCH_ROOT` and your output handoff.** Nowhere
  else.
- **Delete nothing you did not create.** No `rm -rf` on a path you were handed,
  no `docker rm -f` on a container you did not start — both held nodes are
  carrying other tenants' containers right now. If a directory is in your way,
  fail and say so.
- **Never write a recursive delete whose target is a variable.** `rm -rf "$d"/*`
  with `$d` unset is `rm -rf /*`. That happened on this class of host on
  2026-08-31 and destroyed another engineer's git history.
- **Nothing whose path lacks the substring `yihou` may be deleted, ever.**
- **Do not change `$HIP_VISIBLE_DEVICES`.** Other people are on the other cards.
- **Never pass an explicit mode when you create a directory.** Measured
  2026-09-01: a run created `results/raw_measurements/` at `0644`, wrote seven
  files into it, and could not read them back — a directory without its execute
  bit cannot be traversed *by anyone, including its owner*. It failed with
  `PermissionError` on a path it had just written, which reads like a sandbox
  problem and is not one.
- **Tear down anything you started.**

---

## What you are given

| variable | what it is |
|---|---|
| `$AGENT_SYS_INPUT_OPERATOR_WORKSET` | the workset handoff's **version directory** — files are under `content/items/codes/` |
| `$AGENT_SYS_INPUT_DEPLOY_KIT` | m1's kit; its `content/items/codes/environment.yaml` is the environment record this run is in |
| `$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION` | your output handoff's **content directory** — `content/` itself |
| `$AGENT_SYS_TASK_PACKAGE` | the staged copy of this package. Worked examples: `assets/schemas/samples/` |
| `$KFO_SCRATCH_ROOT` | **local disk**, writable, inside a `yihou/` directory. Everything temporary |
| `$KFO_KERNELFORGE_REPO`, `$KFO_MAX_HOURS`, `$KFO_FORGE_MODEL`, `$KFO_SNR_THRESHOLD`, `$KFO_MOCK` | forge's own knobs |
| `$HIP_VISIBLE_DEVICES` | the one GPU you may use |

Note the asymmetry in the first three, because it has bitten people: the
**input** variables point at the version directory, so `content/` is a hop
below them; the **output** variable points at `content/` itself. They look like
a set. They are one level apart.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That one file's
difference is the whole of what "an agent task" rather than "a program task"
means in this system.

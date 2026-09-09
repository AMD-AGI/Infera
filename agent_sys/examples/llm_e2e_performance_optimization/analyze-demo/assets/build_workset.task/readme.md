# build_workset — write the material KernelForge reads

You are given a ranked list of GPU kernels worth optimizing and, for each one,
where its source lives. Write the workset KernelForge needs to optimize them.

## What a workset is, and what it is not

**You are building the measuring apparatus, not the thing measured.**

The kernel source already exists inside the serving framework. KernelForge's
`forge-loop` edits it in place, in a git worktree, so a copy of it in this
handoff would be useless to it. What it cannot produce for itself is the thing
that decides whether an edited kernel is still correct and how much faster it
got — and it treats that driver as a **protected file** its optimizing agent may
not modify.

So: do not copy or rewrite kernel source. Do write the driver, the reference
implementation, the cases, and the briefs.

## Your inputs

| variable | what it holds |
|---|---|
| `$AGENT_SYS_INPUT_KERNEL_WORKLIST` | `items/text.json` — every profiled kernel, with the selected ones marked and their observed shapes as `cases` |
| `$AGENT_SYS_INPUT_OPERATOR_IDENTITY` | `items/text.json` — per operator: language, container root, source files, entry points, PyTorch reference candidates |
| `$AD_SGLANG_SRC`, `$AD_AITER_SRC` | read-only checkouts of the source trees, extracted from the serving image. **Open the files** `operator_identity` names |
| `$AD_GPU_TARGET`, `$AD_GPU_TYPE`, `$AD_FRAMEWORK`, `$AD_SNR_THRESHOLD`, `$AD_IMAGE` | facts about the target machine, to be copied into every exported file |

Work operator by operator, in rank order. Use the `operators` array of
`operator_identity` as your list — every entry gets a directory.

## Where to write

Into `$AGENT_SYS_OUTPUT_OPERATOR_WORKSET`. It already exists and you are granted
write on it. Do not create anything beside it: `claim/` and `manifest.yaml` are
the system's, and the manifest is what publishes the version.

```
README.md            Purpose / How to run / Result / Environment / Watch out
items/result         what you produced, as prose a reviewer reads first
items/env            the target machine and image, from the variables above
items/script         a shell script that runs every workset's run_forge.sh
items/code/<operator_id>/     one directory per operator, laid out below
items/watchout       what a consumer must not assume
```

All five of `result`, `env`, `script`, `code` and the README sections are
required: the kind is `content_type: reproducible`, and a seal without them
fails. A README section that says "to be filled in" is rejected too.

**`items/script` must be executable.** `agent/gate.py` holds
`EXECUTABLE_ITEMS = {script, command, entry}` and refuses the seal when one of
them is present and `os.access(path, os.X_OK)` is false:

```
output_not_executable: <handoff> declares 'script' executable, and it is not
```

So finish with

```sh
chmod +x "$AGENT_SYS_OUTPUT_OPERATOR_WORKSET/items/script"
chmod +x "$AGENT_SYS_OUTPUT_OPERATOR_WORKSET"/items/code/*/run_forge.sh
```

This is measured, not hypothetical: a complete workset was refused on exactly
this, fourteen minutes into a run, and the message the system then sends back is
`continue, do it until finished` — which does not name the missing bit, so the
agent that receives it has nothing to act on.

### One operator directory

```
README.md                  what this operator is, why it was selected, what is known and unknown
invocation_spec.json       Hyperloom invocation-spec schema v2
forge_task.yaml            KernelForge task definition
program.md                 the brief forge-loop passes to its fellow
run_forge.sh               the forge-loop command line, ready to run
reference/naive_torch.py   the PyTorch reference this operator is checked against
scripts/task_runner.py     shared: builds inputs, calls the operator, calls the reference
scripts/forge_driver.py    forge-loop's driver — the stdout contract below
scripts/standalone_driver.py  the same measurement for a human, free format
tests/cases.json           at least three correctness cases
provenance.json            what the profile said about this kernel
```

## The driver contract, verbatim

`scripts/forge_driver.py` is read by forge-loop as a black box over stdout. This
is not a style preference; preflight rejects a driver that deviates.

```
python forge_driver.py
    runs the complete correctness suite over every declared case and prints
    at least one of:
        SNR: 62.13 dB          preferred; forge gates on this against the threshold
        allclose: True         fallback

python forge_driver.py --warmup <n> --iters <n> --bench-mode
    prints the measured per-iteration time, and must include the line

        time_ms: 0.1861

    verbatim in that form. `verify_workset` parses that key; anything else it
    has to guess at from prose.

python forge_driver.py --profile-run
    one forward pass, for a profiler to sample
```

Case selection is by `--shape CASE_ID=<id>`. The driver **must** cover every
selector in `tests.driver_contract.case_selectors`; preflight rejects a driver
that covers fewer or more.

Put the whole of the above block into each `program.md` as well, so a fellow
reading only that file still knows the contract.

## Rules that are not negotiable

**Do not invent a source file.** `operator_identity` records how each path was
found, in `source_resolution_method` and `resolution_evidence`. Where it says
`agent_recovered`, the symbol could not be located by name — read the container
root it names and find the entry function yourself, and say in `program.md` how
you found it. If you cannot, leave `target_kernel_functions` empty and list
`edit_target.source_file` in `invocation_spec.json`'s `missing_fields`. An empty
field is recoverable; a wrong one is not.

**Prefer importing a reference to writing one.** Where `baseline_ref_symbol` is
non-empty, `identify` has already confirmed that `baseline_ref_file` defines it.
Import it. Reimplementing an operator you have only read is the most likely way
to make the correctness check agree with a bug. Write your own only where the
field is empty, and say so in the operator README.

**Never write an absolute path into any file under the output directory.** The
seal scans every file and refuses anything outside `/usr/`, `/opt/`, `/bin/`,
`/lib/`, `/etc/`, `/srv/`, `/workspace/`, `/app/`, `/var/lib/`, `/var/log/`,
`/run/`, `/proc/`, `/sys/`, `/dev/`. Container roots travel as the `${...}`
placeholders `operator_identity` carries, and `run_forge.sh` resolves them from
its own environment.

This is the single most common way to lose a finished handoff, and the trap is
not what it sounds like. A **relative** path can match too: the rule's lookbehind
does not exclude `>`, so writing

```
<operator_id>/scripts/forge_driver.py
```

in prose makes `/scripts/forge_driver.py` look absolute and refuses the whole
delivery. Measured — it is what threw away a complete workset once already.
Write `scripts/forge_driver.py` on its own instead.

Run the checker before you finish, and fix until it exits 0:

```sh
python "$AGENT_SYS_TASK_PACKAGE/assets/lib/check_locality.py" "$AGENT_SYS_OUTPUT_OPERATOR_WORKSET"
```

It applies the seal's own regexes and allow-list and prints the file and line of
every offender.

**At least three cases per operator.** Take them from the worklist's `cases`
array. Where the profile recorded fewer than three distinct shapes, add smaller
ones of the same form and mark them `synthetic: true` in `tests/cases.json` —
a synthetic case is a correctness case, not a performance one.

**Keep the two exported formats agreeing.** `invocation_spec.json` and
`forge_task.yaml` describe one operator; `check_workset_shape` re-derives the
shared fields and compares them. The mapping is:

| concept | `invocation_spec.json` | `forge_task.yaml` |
|---|---|---|
| identity | `logical_operator` | `task_id` |
| primary shape | `workload.task_group.cases[0].selector` | `shapes.primary` |
| other shapes | `workload.task_group.cases[1:]` | `shapes.validation` |
| source files | `implementation.sources` | `source_files` |
| gaps | `status` + `missing_fields` | a `TODO:` line per gap in `constraints` |

`invocation_spec.json` must carry `"schema_version": 2` and a `status` of
`complete` or `partial`. Set `partial` and fill `missing_fields` whenever
anything is unknown — the schema has that pair precisely so that incomplete
evidence can be stated instead of papered over.

## How to check your own work, and when to stop

Run all four. Each must exit 0.

```sh
python "$AGENT_SYS_TASK_PACKAGE/assets/lib/check_locality.py" "$AGENT_SYS_OUTPUT_OPERATOR_WORKSET"
python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('$AGENT_SYS_OUTPUT_OPERATOR_WORKSET/items/code/*/scripts/*.py')]"
python -c "import json,glob; [json.load(open(f)) for f in glob.glob('$AGENT_SYS_OUTPUT_OPERATOR_WORKSET/items/code/*/*.json')]"
test -x "$AGENT_SYS_OUTPUT_OPERATOR_WORKSET/items/script"
```

A driver that does not parse cannot be measured, a spec that does not load
cannot be read, a path the seal refuses discards everything, and a `script`
without its executable bit fails the completeness gate after all the work is
done.

**Then stop.** When every operator has its eleven files and the three checks
pass, you are done — say what you produced and end your turn. The system waits
for the whole graph to go quiet within a fixed budget, and a task that keeps
polishing past that point loses the work it already finished, unsealed. Do not
re-verify what you have already verified.

Order the work so that stopping early still delivers: finish one operator
completely before starting the next, rather than writing all the specs, then all
the drivers. Two complete worksets are useful; five half-written ones are not.

## Watch out

The profile you are working from is a **GLM-5.2 1P1D decode** capture, not
GLM-5.3-Flash. The shapes are real; the operator mix is not the target model's.
Say so in `items/watchout`.

`items/result` is where a reviewer looks first. Write what you actually
produced — how many operators, how many with a confirmed entry point, how many
with an imported reference against a written one, and what you could not
determine. An honest gap listed there is worth more than a complete-looking
directory that hides one.

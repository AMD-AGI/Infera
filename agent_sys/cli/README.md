# `cli` — the runnable proof that the components compose

The program's **single entry point over any task package**, and the composition
root: it wires the eight component packages together and runs a graph. It is not
one package's runner — no path is privileged and no package name is built in.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 17 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) — layout, the event stream, the three verbs, the criterion→test map (§14) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.8 — may import anything of ours; **nothing may import it** |
| Tests | `../tests/cli/` — none makes a model call, needs a credential, or needs a sandbox |

## Who uses it

Everyone above the library layer. `[project.scripts]` points here, so
`agent-sys` **is** this package. Nothing in the repository imports it; that is a
rule, not an accident, and `tests/interfaces/test_import_rules.py` enforces it.

## Usage

```bash
pip install -e agent_sys

agent-sys show                          # the graph, nothing dispatched
agent-sys run --dry-run                 # + every load-time check. This is what CI runs
agent-sys run --dry-run --with-broken   # exit 1, naming the offending file
agent-sys run                           # the whole thing. Needs credentials and a sandbox
agent-sys run --resume                  # continue the last run
agent-sys run --clean                   # remove every run and exit

agent-sys show --package agent_sys/examples/ok.algorithms_solve_grade.14 \
               --var n_problems=2
```

### The two flags that make it generic

| | |
|---|---|
| `--package DIR` | the package to load. `package.py` resolves it, or falls back to the checkout |
| `--var K=V` | repeatable. Sets a package variable, expanded by `spec_loader` as `${K}` or `${K:-default}` |

**`outside` is refused from `--var`, by name.** It is per-run, absolute, and only
the CLI knows it, so the CLI's value has to win — and a flag the user typed that
is silently dropped is worse than an error. Passing it is an argument error
naming the variable and why; a malformed `--var` with no `=` likewise. Both exit
2, which is what argparse already does for an unknown flag.

`show` passes no `outside` at all, so a package's `${outside:-…}` renders its
visibly-unfilled default. `show --var K=V` is the cheap way to check that a value
reaches a spec without preparing a run.

### Exit codes

| | |
|---|---|
| `0` | OK |
| `1` | a load error — the message names the file |
| `2` | a precondition is missing: no credentials, no sandbox, a bad argument |
| `3` | an expected failure did not happen |
| `4` | an unexpected failure |
| `5` | the run did not finish and nothing else said so |

**3 is the one to read twice.** A demo that prints "all good" because the sandbox
stopped blocking, or because a validator stopped failing, is the worst outcome
this artefact can produce; exit 3 is what stops it being silent.

## What is inside

| | |
|---|---|
| `main.py` | argparse, the three verbs, and the exit codes above |
| `build.py` | a closure becomes the root `Task` — `root_task`, `handoff_ids`, `wire` |
| `events.py` | the closed set of event kinds and their versioned payloads |
| `stream.py` | one stream, fanned out to the renderers. Never two writers |
| `render/human.py`, `render/machine.py` | the two renderings of that one stream |
| `environment.py` | the `Context` this package owns, handed to `env_mgr` |
| `package.py` | resolves `--package`, and the deliberately broken sibling behind `--with-broken` |
| `expectations.py` | what a package promises will fail, keyed by package directory |

## Two artefacts, and the split is the enforcement

```
cli/                                     the RUNNER — installed
examples/ok.filetree_grounded_report.4/  a TASK PACKAGE — YAML and data, not installed
examples/fail.dangling_handoff_kind.1/   deliberately broken. Only --with-broken loads it
```

Spec §1.1 says the demo may use nothing an out-of-repository task package could
not use. The moment an example holds an `__init__.py` it is importable and stops
looking like what an outside package looks like; `test_examples_has_no_init` is
what notices.

The broken package is a **sibling directory** rather than a broken file inside
the working one, because the YAML front end scans every `*.yaml` under a package
root — a broken document inside the working package would break every ordinary
run.

## Known leak

`expectations.py` is keyed by **package directory name**, which is a package name
living in the CLI. Its own docstring says so and says what closes it: a
`promises:` block in the package's `main.yaml`. That is a new schema and a change
on both sides of a module seam, so it is [`../docs/TODO.md`](../docs/TODO.md) and
not a quiet fix here.

**An empty set is a statement, not a gap.** It says *this package promises
nothing will fail*, and it exits `OK` rather than `UNEXPECTED_SUCCESS` — because
`ok: true` is the same byte for *no promise was made* and *every promise was
kept*, and the second is a much larger claim than a zero supports.

## Where the demo stops today

`describe`, with `ConfinementNotApplied` — [`../docs/ROADMAP.md`](../docs/ROADMAP.md)
§6.1, P0. That is a property of the roadmap item, not of this package, and
criterion 8 is unreachable until it lands.

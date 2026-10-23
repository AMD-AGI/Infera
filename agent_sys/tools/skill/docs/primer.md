# agent_sys in one page — what the skills rely on

Every statement here is read from the code (`spec_loader/`, `handoff/`,
`validator/`, `closure/`, `env_mgr/`, `agent/`) and cited in
`agent_sys/docs/` and each component's `docs/spec.md`.

## A task package

A directory with **`main.yaml`** and **`assets/`**. Every other `*.yaml` under
it (not under `assets/`) is scanned; `steps/` and `shared.yaml` are conventions.
A file holds one object or a list of them, told apart by `module:`:

| module | what it is | required keys |
|---|---|---|
| `agent` | who runs a leaf: `kind: program` (runs `entry.sh`) or `kind: ai` (Claude; the readme is its system prompt) | `name`, `kind` |
| `handoff` | a **kind** of artefact passed between tasks | `name`, `description`, `content_type`, `scope`, `validators` (≥1) |
| `validator` | a judge of one or more kinds | `name`, `brief`, `dimension`, `strength`, `inputs`, `tags{logic_source,cost}` |
| `task` | a unit of work (loaded as a *closure*) | `name`, `description`, `handoffs`, `validators`, `task{goal≤100, version, inputs, outputs}`; a leaf also `agent` |

Rules that bite:

- **`main` must be a non-leaf** (it has a `subgraph`); a leaf `main` crashes `show`.
- Inside one file, define a thing before you reference it; across files, any order.
- A leaf with an AI agent has `assets/<task>.task/readme.md` only — that file **is**
  the prompt. A program leaf adds `entry.sh` (run as `/bin/sh`, POSIX only).
- Grants: `read` for every input kind, `write` for every output kind
  (`{path: handoffs/<kind>, access: read|write, kind: <kind>}`). Pure
  boilerplate for the skills: the loader rejects a task without them, `astk
  scaffold` writes them, and nothing here enforces permissions or confinement.
- Subgraph edges are **derived** from kinds; `froms` must list them (and may add
  ordering-only edges). Listing order must be topological. Only the `is_end`
  entry's outputs leave a subgraph.
- `${VAR}` / `${VAR:-default}` in any value, from `--var K=V`. No nesting.
- The static check is `agent-sys show --package <dir>` (`astk show <dir>`).
  `agent-sys run` additionally needs the package inside a git repo with
  `extensions.preciousObjects` set — tell the user, never set it for them.

## A handoff

On disk the producer writes only `content/`:

    content/README.md          document-root sections, exact names, real text under each
    content/items/<key>        a file or a directory per item; items/items.json for small data

| content_type | README sections | required items |
|---|---|---|
| `reproducible` | Purpose, How to run, Result, Environment, Watch out | `result`, `env`, and one of `script`/`command` |
| `code` | Purpose, Interface, Boundary | `codes` |
| `structured_text` | Purpose, Schema | one of `text.json`/`text.yaml`/`text.xml` |
| `text` | Purpose | `content` |

The store **refuses to seal** an empty content dir, a missing/empty/placeholder
section (`[More Information Needed]`), or missing items. The gate refuses an
item named `script`, `command` or `entry` without the executable bit.
`items_schema` only checks item *names*. There is **no size limit** and no secret
scan in agent_sys — those are the skills' rules (`calibration.md`).

The store adds `manifest.yaml` (sha256 tree digest over `content/`, executable
bit included — copy with `cp -a`) and `validation.yaml` (verdict rows).

## A validator body

`assets/<name>.validator/entry.sh`, run as `/bin/sh` with cwd = a fresh zone:

    inputs.json      ["<handoff id>", ...]
    args.json        the validator's `args` (strings when they came from ${VAR})
    materials.json   {"<id>": "<content dir: README.md + items/>"}
    verdict.json     YOU WRITE IT: {"<id>": true|false}; no file = UNREACHED

Nothing is inherited from the caller's shell. An **output** check runs in the
producer's environment (`validator.environment.choose_configuration`'s
producer row: its `AGENT_SYS_*` paths, and the agent's `env:` block), plus
`AGENT_SYS_DEMO_PACKAGE`/`_PYTHON`/`_STORE`, with `HOME`, `TMPDIR`, `PWD` in the
zone; a validator that names an `agent:` gets that agent's `env:` instead.
(Read from `validator/phase.py`, where the producer row is marked LIVE; the
comment in e2e-flow's `m1_deploy.yaml` saying a validator never receives the
block predates it. Not re-measured in a run.) Pass what a validator needs in `args` anyway — args are the declared channel.
What the body prints is kept in its zone as `body_output.txt`; a body that
wants its reasons read writes them to `validator_report.txt`. Exit code does not matter; the verdict file does.

A task's validators = its closure `validators:` + every input/output kind's
`validators:`, run cheapest `tags.cost` first. **One failure fails the task, at
any strength.** agent_sys never retries a failed validation and never tells the
producer why — the skills do (task-runner), and say so.

## What a running leaf sees

    AGENT_SYS_TASK_PACKAGE     a copy of the package
    AGENT_SYS_INPUT_<KIND>     staged copy of an input's content (README.md, items/)
    AGENT_SYS_OUTPUT_<KIND>    the content dir to write
    AGENT_SYS_MY_{ZONE,WORKSPACE,PLAYGROUND,HANDOFFS,LOGS}
    + the agent's `env:` block (the only way a package passes its own variables)

`<KIND>` = the kind name, non-alphanumerics → `_`, upper-cased
(`profiling_mode_on.bench_result` → `PROFILING_MODE_ON_BENCH_RESULT`).

## Reusing a handoff made outside agent_sys

There is no import command. The supported route is a **program leaf that
publishes a fixture**: `cp -a "$AGENT_SYS_TASK_PACKAGE/assets/fixture/<kind>/." "$AGENT_SYS_OUTPUT_<KIND>/"`.
The kind's validators then judge it exactly as if a task had produced it
(precedent: `examples/03_kernelforge_one_kernel`, `publish_inputs`).

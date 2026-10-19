# `env_mgr` — all interaction with the operating system

Two jobs, separated by a wall that is enforced rather than intended:

1. **Provision an environment** from one self-contained YAML recipe —
   check / dry-run / install / bootstrap, over Python, apt, binaries and Claude
   plugins/MCP.
2. **Build and confine a zone** for a task — paths, grants, isolation, staging,
   and the remote side.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 22 acceptance criteria, of which 2–14 are CI-enforced |
| Design | [`docs/design.md`](docs/design.md) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.6 |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi` |
| Tests | `../tests/env_mgr/` |

## Who uses it

`agent` asks for a zone before it runs a body, and spawns through what it gets
back. `validator` asks for a validation environment. `cli` constructs the
`Context`. **This package imports `task_graph` and nothing else of ours**, and
nothing above it reaches around it to the operating system — which is the reason
the package exists.

## Usage — the recipe half

```bash
# The shipped recipe uses placeholder paths; point it at a real repo/workspace
# with --path (and --workspace) rather than editing the file.
uv run env-mgr check     env_mgr/recipes/sglang.repo.yaml --path /path/to/repo
uv run env-mgr dry-run   env_mgr/recipes/sglang.repo.yaml --path /path/to/repo
uv run env-mgr install   env_mgr/recipes/sglang.repo.yaml --path /path/to/repo --tag lsp
uv run env-mgr bootstrap env_mgr/recipes/sglang.repo.yaml --path /path/to/repo
```

Exit code 2 on any FAIL, else 0.

| installer | wraps | note |
|---|---|---|
| `uv` | [uv](https://docs.astral.sh/uv/) | ref form runs `uv pip install -e`; tool form runs `uv tool install` |
| `apt` | dpkg/apt-get | **detects and prints** the `apt-get` line. Never runs sudo |
| `bin` | a `check_cmd` plus a one-line install | standalone binaries with no project manifest |
| `oneline` | a single shell line | one line, so each stays inspectable in dry-run |
| `embed` | a multi-line shell body | only when control flow is genuinely needed |
| `claude` | `claude plugin` | Claude Code manages plugin state itself |
| `run_server` | a long-lived process | port-based MCP servers |

**Nothing is installed by this package directly.** Each installer shells out to
the mature tool above and can be swapped without touching the CLI or the recipe.

## The zone half

`EnvManager` has two methods — `prepare` for a task, `prepare_validation` for a
phase — and `prepare` returns a `Prepared` the caller **spawns through**. The
split is load-bearing: `prepare` checks, `spawn` applies, because on the
bubblewrap rung the confinement *is* the exec, and a caller that built its own
`argv` would silently run unconfined.

| | |
|---|---|
| `fs/` | domains, zones, layout, and staging a consumer's `content/` |
| `isolation/` | bubblewrap first, else Landlock, else refuse. `probe.py` is what decides |
| `grants.py` | a grant resolved to paths, and the `PATH` projected from the granted set |
| `paths.py`, `prefix.py` | the path-variable system, and the one shared root |
| `material.py` | rules, hooks and skills deployed into a zone |
| `remote/` | the far side: the connection, and the three tools an agent is handed |
| `sync.py`, `workspace.py`, `container.py`, `servers.py` | the rest of the substrate |
| `recipe.py`, `installers/`, `runner.py`, `report.py` | the recipe half |
| `o11y/` | the observability surface |

## Three things that will break a reimplementation

1. **A grant that resolves to nothing raises.** It does not return an empty set —
   an empty granted set is indistinguishable from a satisfied one, and that is
   how a confinement failure becomes a silent pass.
2. **`PATH` is projected from the granted set, never chosen.** The invariant is
   that `PATH` can never name a directory the kernel will refuse.
3. **A validation environment never inherits `os.environ`.** Anything a body
   needs must be declared, or it is not there.

## v1 limitations

- **Skip-with-warning is not implemented.** Each installer's own idempotent
  `check` covers the practical single-host case, so `--on-conflict weak` is a
  no-op: it skips conflict detection and installs. Only version-conflict
  detection under `fail` is active.
- **The workspace default is stubbed** — the default path, its warning, and
  user-bin symlinking are not wired up.
- **System apt is detect-and-print only.**

## Two exposures, named rather than hidden

**Installs run unconfined, and spec §4 does not say so.** An install writes
outside every zone by definition — confining it would defeat the purpose rather
than harden it — but §4 reads as though confinement is universal here, so the
next reader meets the exception by discovering it.
[`../docs/TODO.md`](../docs/TODO.md) item 13.

**`AGENT_SYS_NO_PERMISSIONS` defaults to on**, so grant enforcement is opt-in
today. [`../docs/ROADMAP.md`](../docs/ROADMAP.md) §6.1 is the P0 that changes it,
and §6.3 is the rebuild behind it.

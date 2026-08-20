# env_mgr

Layered environment manager for the agent work system. Driven by one
self-contained YAML recipe, it can **check / dry-run / install / bootstrap**
an environment (Python, apt, binaries, Claude plugins/MCP) and report per-item
status plus delivered artifacts (path/version/deps).

Design spec: `../docs/superpowers/specs/2026-08-17-env-mgr-design.md`.

## Installers and the mature tool each wraps (ai.env.md rule 4)

| installer | wraps | why |
|---|---|---|
| `uv`      | [uv](https://docs.astral.sh/uv/) | de-facto Python toolchain; ref form runs `uv pip install -e` against the project's own manifest, tool form runs `uv tool install` for standalone tools (serena). |
| `apt`     | dpkg/apt-get | standard Debian package DB; v1 only **detects + prints** the apt-get line (never sudo). |
| `bin`     | any check_cmd + install one-liner | for standalone binaries with no project manifest (uv-via-pip, pyright-via-npm). |
| `oneline` | a single shell line | declarative one-line actions; exactly one line so each is inspectable in dry-run. |
| `embed`   | a multi-line shell body | only when control flow is needed (serena per-project index). |
| `claude`  | `claude plugin` | Claude Code manages plugin state itself; we just add/list. |

Nothing is installed by env_mgr directly — each installer shells out to the
tool above and can be swapped without touching the CLI or the recipe.

## Usage

```bash
# The shipped recipe uses placeholder paths; point it at a real repo/workspace
# with --path (and --workspace) rather than editing the file.
uv run env-mgr check    env_mgr/recipes/sglang.repo.yaml --path /path/to/repo
uv run env-mgr dry-run  env_mgr/recipes/sglang.repo.yaml --path /path/to/repo
uv run env-mgr install  env_mgr/recipes/sglang.repo.yaml --path /path/to/repo --tag lsp
uv run env-mgr bootstrap env_mgr/recipes/sglang.repo.yaml --path /path/to/repo
```

Exit code: 2 on any FAIL, else 0.

## v1 limitations

- **Cross-layer skip-with-warning is not implemented** (design §4.1). env_mgr
  does not yet walk the parent chain to detect that an item is already
  satisfied by an upper layer and skip it with a warning. Each installer's own
  idempotent `check` covers the practical single-host case. Consequently
  `--on-conflict weak` is a **v1 no-op**: it skips cross-layer conflict
  detection entirely and proceeds with install (exit 0), whereas `fail`
  records the conflict and halts before install (exit 2). Only cross-layer
  *version-conflict detection* under `fail` is active.
- **workspace layer is stubbed** — the default `$HOME/workspace.infera.aiopt`
  path, its warning, and user-bin symlinking are not wired up yet.
- **system apt is detect-and-print only** — the `apt` installer never runs
  sudo; it prints the `apt-get install` line for you to run.

## Relationship to the former `../helper/` demo

`../helper/*.sh` was the original shell demo. It has been removed; its logic
now lives entirely in this recipe and the installers above.

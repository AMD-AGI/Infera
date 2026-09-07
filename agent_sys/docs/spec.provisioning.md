# Provisioning — how an agent gets its environment

| | |
|---|---|
| Status | **Normative for this round.** Written 2026-09-04 from the owner's rulings on PR 155 plus measurement |
| Scope | Everything installed or declared for an agent: recipes, add-ons, MCP servers, tools, skills, hooks, plugins |
| Spans | `env_mgr`, `agent`, `spec_loader` — which is why it is here and not in one component's `docs/` |
| Supersedes | the L1/L2/L3 vocabulary, entirely. There are no levels |

**Every rule below has either an owner ruling or a measurement behind it, and the
measurements are named.** Where the rulings underdetermined something, the
derivation is shown rather than the conclusion asserted.

---

## 1. The one rule everything else follows from

> **Declarative first, and separate processes over shared ones.**
>
> A thing is installed by **declaring it in a recipe**. It is delivered by a
> **file the harness reads** or a **process of its own**. Python code that adds
> capability to a running agent, and code that runs inside the `agent_sys`
> process, are both exceptions requiring justification (§6).

The one thing that is *not* installed by a recipe is an agent's own `.claude/`
tree, which is copied. §3.

## 2. Recipes come in three layers, and the layer is where the file is

| layer | where | how it is found |
|---|---|---|
| **default** | `env_mgr/default.env_recipe.yaml` | never named; always applies |
| **task package** | `<package>/assets/main.env_recipe.yaml` | auto-detected, one fixed spelling |
| **agent** | `<agent assets>/env_recipe.<agent>.yaml` | auto-detected, any `_stems` permutation |

**There is no `layer` field on an item and there must not be one.** The layer is
carried by the path, and a field restating it would be a second writer of one
fact. A recipe carrying a stale `layer:` key is **rejected** with a dated
migration message (`recipe.py:73`) rather than silently passed into `Item.spec`.

**`env_mgr/recipes/*.yaml` are demos** — the namespace of things you *name* in
`recipes: [x]`. The default is the one you never name, which is why it is not in
that directory.

### 2.1 They concatenate; they do not override

Default → package → agent, in that order, **additive**. A more specific layer
*adds* items; it does not replace them. Re-running an install is cheap because
every installer gates on `check` before `install`.

**Nothing detects a version conflict between layers**, and this is a known gap,
not an oversight: `detect_conflicts` is scoped to one `run()` and `_run_recipe`
spawns one child process per recipe file, so three layers are three independent
checks. Closing it means parsing all three in the parent — the in-process
coupling the subprocess design exists to avoid. Also measured: `detect_conflicts`
fires only on **incompatible version constraints**, never on a repeated name, so
two layers both declaring `uv` is not an error and should not be.

### 2.2 Absence

**Declared and absent is an error. Undeclared and absent is simply absent.**
(`material.py:62-86`'s existing rule.) There is no third case. Both agent-level
systems — its own recipe and its `.claude/` tree — may be absent independently.

## 3. Where an installed thing lands

Two questions, in order. Neither is declared; both are derived.

> **1. Is it AI material — a `.claude/` tree the agent harness reads as its own
> configuration?** If not, it installs **system-wide**, once.
> **2. If it is: did the *agent* declare it?** Yes → **project level**.
> Declared by the task package → **user level**.

| what | where | Claude Code calls it |
|---|---|---|
| binaries, language packages, OS packages | system-wide; the agent_sys root where the installer accepts a prefix | — |
| a `.claude/` tree from `main`/`default` | the agent_sys root's Claude config | **user level** |
| a `.claude/` tree under an agent's own assets | the agent's workspace root | **project level** |

**These are Claude Code's own two scopes and adopting them is the point.** A
harness that already distinguishes user from project does not need a second
hierarchy laid over it.

**The copy route is for row three only.** Everything else is installed by a
recipe.

Two consequences, stated because they are behaviour and not restatement: user
level is **shared across a run's agents**, and it **outlives the run** (the
agent_sys root is deliberately outside any run root — see `TODO.md` 4i).

## 4. Add-ons

`env_mgr/addons/<name>/` — what `agent_sys` itself ships for agents. **Inside
`env_mgr`, not beside it**, because `package-data` needs an owning package;
proven by building a wheel and counting members, not by the build succeeding.

**Installed only by recipe.** There is no declaration key on `AgentSpec`. A
recipe locates an addon by importing `env_mgr` — `PYTHONPATH` is pinned to the
package root and `run_cmd` inherits it — so no path pointing outside the zone
needs exporting.

## 5. MCP servers and tools

**The transport decides the mechanism**, and it is not a preference:

| transport | who starts it | how it is declared |
|---|---|---|
| **stdio** | **the harness spawns it** — that is what stdio means | a `.mcp.json` entry |
| **port-based (HTTP/SSE)** | **`env_mgr`, via the `run_server` installer** | a recipe item |
| in-process | — | **see §6** |

`run_server` maintains a registry at `<layout.run>/servers.json`, keyed to the
run's lifetime. A duplicate declaration finds the entry and reports **`warn`**
without starting anything. Servers are stopped when the run ends.

**The guarantee is *"stopped on normal and handled-error exit"*, not "always".**
`SIGTERM` has no handler and `SIGKILL` cannot have one. `PR_SET_PDEATHSIG` was
measured to close the `SIGKILL` case and is the wrong tool at the spawn site,
because the spawning process is a recipe child that exits within seconds.
`TODO.md` 4j carries both halves.

A port already held: same binary → `warn`, different → `fail`, **and a holder
owned by another uid → `fail`**, because its command line cannot be read at all
and therefore can never be "basically the same". The identity key is the
**declared program token from the item's own `command`**, matched against the
holder's `/proc/<pid>/cmdline` — asking *"is this the thing I was about to
start?"* rather than *"what is this process?"*, which has no answer when every
candidate reports as `python3`.

### 5.1 `.claude/` and the SDK overlap, and how that is resolved

Measured from the installed `claude_agent_sdk`:

- **`setting_sources`** defaults to loading **all** filesystem sources (user,
  project, local). `[]` is *SDK isolation mode*.
- **`strict_mcp_config=True`** ignores everything the CLI would otherwise load,
  *"e.g. project `.mcp.json`, user/global settings, plugin-provided servers"*.
- **Same-name collisions have no SDK-level arbitration.** The fields are plain
  dicts.

So the overlap is **real, known to the SDK, and resolved by switches rather than
by precedence** — additive by default. Because the SDK arbitrates nothing,
**whoever merges two sources owns the collision**: `claude_sdk.py:375-386` names
it and refuses rather than overwriting, since the model addresses servers as
`mcp__<server>__<tool>` and a silent replacement makes one side's tools vanish.

**Derived, and it resolves a claim that looked contradicted:** `.mcp.json` is a
**project-scope** filename — the CLI and SDK both say *"project `.mcp.json`"*,
and `--mcp-config` exists precisely to load one from elsewhere. A zone's
`$CLAUDE_CONFIG_DIR` is **user** scope. So `agent_assets.py:287`'s *"placing it
would put a file in the zone that nothing reads"* and the SDK's *"the CLI would
otherwise load project `.mcp.json`"* are **about different locations and both
true**. If the declarative route is ever wanted for MCP, its destination is the
**workspace root**, not the config directory.

## 6. Two things that need justification, and one exception

Both rules are about one temptation: reaching for Python because it is nearer
than a recipe. Neither forbids; each requires the reason to be written and to be
*"the other routes do not work"*.

> **Adding an MCP server or a tool to an agent from Python code** needs a
> justification that no declarative route works.
>
> **Running an MCP server inside the `agent_sys` process** needs a justification
> that no separate process works.

The second is the stronger: an in-process server is third-party or cross-module
code executing in the process that supervises every agent, with its memory, file
descriptors and credentials. There is no boundary to fail closed, and
`Installer`'s contract cannot express delivering one — it returns `Outcome`s, and
a live Python object does not survive a subprocess.

**The standing exception** is `env_mgr/remote/tools.py` — `env_remote_run`,
`env_remote_push`, `env_remote_pull`. It is delivered by **injection**:
`claude_sdk.py:393` puts a live `create_sdk_mcp_server` object into
`ClaudeAgentOptions`, and **nothing is written to disk**, so no installer can
carry it. It works and has a live user. **Closing condition**: reprovide the
three as a standalone server started by `run_server`, after which this section
has no exception. See `ROADMAP.md` §6, *"If `agent_sys` must ever serve MCP
itself"*, which carries both halves: the owner's sketch for the case where
`agent_sys` would have to be the server, and the argument that `run_server`
already suffices — which is why the closing condition above is reachable today
and is not waiting on that sketch.

**A second exception added by analogy to this one is the first rule being
ignored.** The point of writing it down is that the next case argues on its own
merits.

**The in-process `ToolDef` route for *component-supplied* tools is deleted.** An
add-on ships a server that runs on its own.

## 7. How an agent knows it is working remotely

Not from prose. `prepare.py:645` returns the three remote tools if the zone has a
far side and `()` if it does not — **whether `env_remote_run` is in the toolbox
is the answer** — and `AGENT_SYS_*_REMOTE` mirrors every local path name.

The reason this is a tool surface rather than a described procedure is in the
module's own docstring: *"an agent given a natural-language description of how to
sync a directory will improvise, and the improvisation will be wrong in a way
nobody notices."*

## 8. What is deliberately not settled here

- **`env_mgr/recipes/*.yaml` do not ship in a wheel.** `recipes: [serena]` in
  `examples/env_checker` cannot resolve from a wheel install. Recorded in
  `temp/bugs/2026-09-04-*`; the one-line `package-data` candidate is unrun.
- **Cross-layer version conflicts** — §2.1.
- **A registry sweep for runs killed by a signal** — `TODO.md` 4j.
- **Installs run unconfined** and §4 of `env_mgr`'s spec does not say so —
  `TODO.md` 4k. Documentary, arguably required, unchanged.

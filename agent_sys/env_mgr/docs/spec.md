# Environment Manager — Specification

| | |
|---|---|
| Status | Draft, revised after review |
| Revision | 3 — 2026-08-26. The read rule is an allow-list, because neither mechanism can enforce a deny-list (§4.2, §4.5). A default granted system set (§4.5.1). Isolation criteria are CI-enforced (§10). (rev. 2: Review of PR #132: isolation is OS-enforced, not prefix-matched; `env_mgr` is a provider of mechanisms, not a process; playground is unsynced scratch; sync is a one-time job at task start. rev. 1: widened scope from environment recipes) |
| Date | 2026-08-24 |
| Scope | All interaction with the Linux system: storage, workspaces, isolation, local↔remote mapping |
| Source | The task definition §7; a survey of agent-harness isolation mechanisms (§4) |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |

---

## 1. Purpose

`env_mgr` owns **everything the system does to the operating system**. Storage,
the agent's working environment, the remote machine where experiments run, and
the isolation that makes the other components' authority rules real rather than
intended.

### 1.1 It is a provider of mechanisms, not a process

**`env_mgr` is not a daemon, a thread, or a single object that maintains
everything.** It is a collection of pieces, each internally consistent, offered
to other components as mechanisms and tools.

This matters because the alternative reading — "the environment manager manages
the environment" — implies a running thing that owns global state and must be
consulted. It is closer to a library: a filesystem module that answers path
questions, a zone module that confines a process, a sync module that copies a
tree once. What makes it a component rather than a grab-bag is that each
mechanism is coherent on its own, and they share one fact: **the path**.

The shipped `env_mgr` — an environment manager driven by one YAML recipe,
doing check / dry-run / install / bootstrap — is one of those pieces and is
reused rather than reimplemented (§9). It is no longer held unchanged: §9's
table says which modules this round moved and why.

### 1.2 In scope

- All storage and persistence.
- The filesystem manager: domain registration, containment checks, local↔remote
  mapping.
- **Isolation** — the OS-level confinement that makes a permission boundary real.
- Remote access.
- The agent's local and remote environments: workspace, playground, handoff
  storage.
- Preparing an environment before a task runs.
- Its own metadata and configuration.

### 1.3 Out of scope

**The runtime environment of the program under test.** sglang, vllm, and infera
environments — and whether one is consistent before and after a change — are not
this component's business:

- Consistency and its rules belong to the handoffs and validators that name them.
  Concretely: **a versioned handoff carrying executable content must record its
  environment** (handoff spec §3.2), and that record is the handoff's, not
  `env_mgr`'s.
- Deployment guides and cluster conventions enter as **knowledge handoffs**, not
  as code here. §7.

Also out of scope: an agent's own process and durability, and what a backend does
internally.

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **The path is the fact** | Permission, storage, and mapping are all expressed against paths, in one component |
| 2 | **The boundary is the kernel's, not ours** | A hook is a first gate and a diagnostic. Confinement is an OS mechanism. §4 |
| 3 | **Fail closed** | Cannot canonicalise a path, cannot obtain a sandbox, cannot decide — deny. §4.3 |
| 4 | **Write narrowly, read broadly** | A task may not write outside its zones; reads get a generous but **declared** set. Both are allow-lists — the mechanisms enforce nothing else (§4.2). §4.5 |
| 5 | **Work on a copy** | An agent copies a handoff into its playground and works there. §6.3 |
| 6 | **A mechanism, not a manager** | §1.1 |
| 7 | **Reuse what shipped** | The recipe and installer machinery is reused, not reimplemented. It is no longer *frozen*, for **two** independent reasons, both 2026-09-04: the layer model was removed from it by design, and `installers/claude.py`'s plugin check was fixed — a check that could never pass, held in place by the fence along with two tests encoding a CLI output format that does not exist. Either alone would have retired the fence. §9 |
| 8 | **Adopt Claude Code's user/project split; invent no levels of our own** | Non-AI installs go system-wide. AI material splits exactly as the harness already splits it: package-declared is *user level*, agent-declared is *project level*. There is no layer field and no layer vocabulary. §9.1 |

---

## 3. What `env_mgr` owns

| Sub-module | Owns |
|---|---|
| **metadata** | Its own configuration: domain registrations, local↔remote mappings, sync strength. Persisted; auto-detected where possible |
| **filesystem** | Domain registration, containment checks, local↔remote mapping. §5 |
| **isolation** | OS-level confinement of an agent process. §4 |
| **remote** | How to reach a remote machine. §5.4 |
| **agent environment** | Workspace, playground, and their local/remote correspondence. §6 |
| **handoff storage** | The read/write entry point. §6.3 |

### 3.1 Metadata

Populated three ways, in decreasing order of preference: **auto-detected** where
discoverable (is this path on NFS? is this a git worktree?); **produced by a
designated task**, for answers that need something run — remote reachability and
access method are exactly this, and arrive as a knowledge handoff (§7);
**declared**, for the rest.

---

## 4. Isolation

The load-bearing section. It is specified against measured behaviour, not
intuition.

### 4.1 A hook alone does not isolate anything

The earlier revision specified confinement as a `PreToolUse` hook matching a
path prefix. **Both halves of that are broken**, and both were demonstrated
rather than argued:

| Claim | Finding |
|---|---|
| A hook can gate filesystem access | **No.** An agent writes `reader.py` and runs `python3 reader.py`. The hook receives `{"tool_name": "Bash", "command": "python3 reader.py"}` — **no file path appears in the JSON at all**, so there is nothing to match. Reads and writes outside the zone succeeded. Anthropic documents this: deny rules "don't apply to arbitrary subprocesses that read or write files indirectly" |
| Prefix matching identifies a governed path | **No.** `startswith(zone)` is CVE-2025-54794 in Claude Code itself, CVSS 9.1, described as "prefix matching instead of canonical path comparison". Three defeats reproduced: a sibling directory `…-EVIL/x.txt`; a symlink inside the zone pointing out; `zone/inner/../../outside`. All three passed the prefix check |

An unguessable generated prefix does not repair this. **It is
security-by-obscurity**: the value is not secret from the adversary that matters,
because the confined agent holds it. It was recovered three ways — `pwd`,
`readlink -f /proc/self/cwd`, and a `glob` on the parent directory, which also
enumerated a neighbouring zone. Keep the marker for readability and
accident-avoidance; **do not let it buy any confidence**.

### 4.2 The boundary is an OS sandbox

With kernel-level confinement applied, the same script that defeated the hook is
blocked at every attempt — including from a child `bash -c`, because the
restriction is **inherited across `exec` and cannot be dropped by a subprocess**.

**The mechanism chain, in order:**

| Order | Mechanism | What it does | Requires |
|---|---|---|---|
| 1 | **bubblewrap** (`bwrap`) | Namespace-based: out-of-zone paths are not in the process's filesystem view at all. Also isolates network and PID | The `bwrap` binary |
| 2 | **Landlock** | Kernel LSM: the process declares its own reachable set; the kernel enforces it on the process and every descendant | Linux ≥ 5.13, practically ≥ 6.2 |
| 3 | **Neither** | **Fail closed — refuse to start the agent** | — |

Not "warn and continue". An agent started without confinement is an agent
running with the operator's full privileges while the system reports it is
sandboxed, and that is worse than refusing. Claude Code exposes exactly this as
`failIfUnavailable`; Codex refuses rather than running unsandboxed.

Both Claude Code and Cursor state plainly that their in-process layers are **not
a security boundary** — Cursor says so in three separate places and deleted its
command denylist entirely after four bypasses, concluding that for every denied
command there are infinitely many undenied ones with the same effect.

#### The mechanisms are allow-lists, and that shapes §4.5

This is not an implementation detail; it decides what a rule can say.

[Landlock](https://docs.kernel.org/userspace-api/landlock.html) has **no deny
rules at all**. A ruleset declares which rights it handles and rules grant
subsets back, so it is deny-by-default; layers combine by **intersection** and
restrictions can only be added; and a hierarchy grant covers everything beneath
it and **cannot be holed**. "Everything except X" is not expressible.

Under bubblewrap it is expressible only at construction time — and the sandbox is
built **once, at task start** (§8), so a zone that appears later could never be
excluded from an already-running process.

Both facts point the same way: **a rule of the form "deny these, allow the rest"
cannot be enforced by either mechanism.** §4.5's read rule is an allow-list
because that is the only shape the boundary can actually take.

### 4.3 Path checks: canonical containment, failing closed

Where a path check is still made — the hook's first gate, and the sandbox's own
policy construction — it is:

```
p == z  or  p.startswith(z + os.sep)      # on realpath-resolved paths
```

Four rules, each fixing a specific documented failure:

1. **Resolve first.** `realpath` both sides before comparing.
2. **The trailing separator is load-bearing.** `z + os.sep` is what stops
   `zone-EVIL` matching `zone` — the CVE-2025-54794 defeat.
3. **Canonicalisation fails closed.** Use `Path.resolve(strict=True)` and treat
   **any** exception as deny. `os.path.realpath` does *not* raise on a broken or
   looping symlink — it silently returns a partly-resolved path, and a
   `try/except` falling back to the raw string reproduces CVE-2026-50549 (CVSS
   9.8) verbatim. For a file that does not exist yet, resolve its **parent**.
4. **Reject NUL bytes.**

**Canonicalise per check, at use time** — not once when the policy is built.
Resolving attacker-mutable components early is itself a TOCTOU bug; Codex fixed
exactly this.

### 4.4 Two rules about where policy comes from

**Never derive the zone root from model-supplied input.** Anchor it to what the
harness started with. A model-controlled `cwd` / `working_directory` was
CVE-2025-59532 in Codex and CVE-2026-50548 in Cursor, both CVSS 9.8.

**The policy lives outside the agent's writable set.** If an agent can write the
file that grants its own permissions, there is no boundary. Precedent:
CVE-2026-48124 (hooks executed from a workspace-local settings file) and
CVE-2026-26268 (writing `.git/hooks` escaped the sandbox, CVSS 9.9, no user
interaction). Also protect `.git/config`, `.git/hooks`, and shell rc files.

### 4.5 The two directional rules

| Direction | Rule |
|---|---|
| **Write** | A task's executor may not write outside its zones. Local or remote, no exception |
| **Read** | An **allow-list**: a generous granted set, and nothing outside it |

The asymmetry is deliberate: writes are how an agent affects the world and how it
could corrupt another task's materials; reads are mostly how it does its job. So
the read set is **broad** — but broad means *generously declared*, not open.

An earlier revision said "a governed path the permissions do not cover is denied;
everything else is allowed". That is a deny-list, and §4.2 records why neither
mechanism can enforce one. The intent behind it survives intact: an agent must
reach ordinary system locations, because refusing that damages the work. That is
a granted set, and a granted set is exactly what Landlock expresses natively.

**The granted read set:**

| Granted | |
|---|---|
| System hierarchies, read-execute | The default set (§4.5.1) — what makes a working agent possible |
| Its own zone | Its subtree, per §5.1 |
| Whatever else its permissions name | |
| **Nothing else** | In particular: no sibling zone, and no validation's materials |

**The read rule enforces the validator boundary** (validator spec §8): a
validation's materials are simply not in the granted set, so the producer cannot
reach them.

The allow-list also settles an objection the deny-list could not answer. The
sandbox is built once, at task start (§8), while zones keep appearing as tasks
are dispatched — under a deny-list, a zone created later could never be added to
an already-running process's denied set. Under an allow-list there is nothing to
add: **anything not granted at construction is already unreachable**, including
everything that does not exist yet.

#### 4.5.1 The default granted system set

Read-execute on the standard system hierarchies — `/usr`, `/lib`, `/lib64`,
`/bin`, `/sbin`, `/etc`, `/proc`, and the interpreter and toolchain paths a task
declares through its `env`.

It is a **default in configuration, not a constant in code**, because it is a
policy decision with a security cost and a site may need to narrow or widen it.
Two things about where the line falls:

- Read-execute on `/usr` and `/etc` is what every surveyed harness grants, and a
  task cannot run a Python interpreter without it.
- A **home directory is not in the default set.** It is where credentials,
  SSH keys, and other tasks' scratch live, and granting it by default would make
  the zone boundary decorative.

Without a default, every task would declare the same twenty paths, and the
twenty-first omission would be found at run time.

### 4.6 The honest ceiling

Stated in the spec because it decides what the system may claim:

> The OS sandbox is necessary and is the only thing measured to stop the scripted
> bypass. It is **not inviolable** — four sandbox escapes were disclosed against
> one harness in 2026 alone, rated 8.5–9.9. For genuinely untrusted input the
> serious answer is a VM or microVM per task, which is what the harnesses that
> take isolation seriously do.

The alpha runs trusted-but-fallible agents against a known workload, so a process
sandbox is the right point on the curve. That is a judgement about the threat
model, not a claim about the mechanism.

---

## 5. The filesystem manager

### 5.1 Domains and the nested layout

A **domain** is a registered region with a name, a root, and a kind:
`handoff_storage`, `playground`, or `workspace`. Registration is idempotent —
reloading an existing domain rather than recreating it is what lets a playground
survive a restart (§6.2).

**Storage is nested, following the task tree.** A subtask's storage lives inside
its parent's:

```
<root>/task.<uuid>.<version>.<hash>/
  ├── handoffs/
  ├── workspace/
  ├── playground/
  ├── logs/
  └── task.<child-uuid>.<version>.<hash>/     ← a subtask, nested
        └── …
```

Because permissions are a versioned task attribute covering the task's own
subtree (`task_graph` spec §3.2.2), **"may this task reach that path" is
containment** — which is the same question §4.3 already answers, computed the
same way. The nesting is what makes one mechanism serve both.

**The zone id is the runtime `uuid.version`** — the task's own identity, not a
separate namespace. Plus a per-level hash component, which is for readability and
collision-avoidance only (§4.1).

### 5.2 Local↔remote mapping

| Strength | Mechanism | Meaning |
|---|---|---|
| **strong** | Same NFS mount | The two paths are the same bytes |
| **strong** | Same mount | As above |
| **weak** | `rsync` | Two copies, synchronised explicitly |

Recorded per mapping, not inferred at use time.

### 5.3 Sync is a one-time job, at a defined point

**Not a continuous reconciliation.** At **task start**, everything is created on
both sides and local and remote are made identical. After that, nothing syncs
by itself.

**Sync is per-agent / per-case, never the whole root.** Syncing a root would move
material belonging to tasks that have nothing to do with this one, and would take
time proportional to the system rather than to the work.

What is *not* synced: **the playground** (§6.2).

### 5.4 Remote access

**ssh** and **docker exec** in v1. `kubectl exec` and `slurm`/`spur` exec are on
the roadmap.

How to detect and reach a remote is **knowledge produced by a designated global
task** and delivered as a knowledge handoff (§7) — not configuration hard-coded
here.

### 5.5 Remote↔local operations are tool calls

**The whole remote↔local surface is exposed to agents as tool calls**, not as a
procedure described in prose.

An agent given a natural-language description of how to sync a directory will
improvise, and the improvisation will be wrong in a way nobody notices. A tool
call has a schema, a name, and a result, and the agent uses it reliably.

---

## 6. The agent's environment

### 6.1 Local

| | |
|---|---|
| **workspace** | A **worktree** cut from the main repository, with dependency repositories cloned inside. Working directly in the main checkout is not permitted |
| **playground** | Scratch. §6.2 |
| **handoff storage** | The entry point of §6.3 |
| **logs** | Where the o11y system writes. Under the same permission rules as everything else |

A worktree rather than a clone: several agents get isolated checkouts sharing one
object store, and a branch one creates is visible to the operator without a
fetch.

### 6.2 The playground is unsynced scratch

**The playground is an agent's `/tmp`** — work in progress, temporary,
non-standard, unwashed content. Run logs, debug scripts, half-finished notes.

Three consequences, all deliberate:

- **It is not synced between local and remote.** Their contents can be completely
  different, and that is correct: each side's scratch is about the work happening
  on that side.
- **It survives a resume**, because it is on the filesystem. **Its contents are
  not guaranteed** — an agent may find what it left, or may not, and must not
  depend on it. That is what "playground" means.
- **The agent gets root permission within it.** It is scratch; constraining its
  internal layout would buy nothing and would make it useless for the thing it
  exists for.

The agent's resume story rests on its **history** (owned by the harness) and its
**workspace**. The playground helps when it happens to still be there.

### 6.3 Handoff storage entry

1. **A handoff is local-independent** — no local paths, dependencies declared
   (handoff spec §7).
2. **An agent copies before using.** Read from storage into the playground, work
   on the copy, never edit the stored artefact.
3. **An agent reaches only what its task's permissions cover** (§5.1).

Rule 2 is what makes a re-run comparable to the run before it.

### 6.4 Remote

Mirrors local — remote workspace, playground, handoff storage — related by §5.2's
mapping, except the playground, which is not mapped at all (§6.2). Both
directional rules (§4.5) apply on both sides.

---

## 7. What `env_mgr` does not own

**The runtime environment of the program under test.**

| Concern | Owner |
|---|---|
| Consistency of that environment before and after a change | The handoff and validator implementations that name it. A handoff with executable content records its environment |
| Deployment guides, cluster conventions, how to reach a remote | **Knowledge handoffs**, injected like any other |

`env_mgr` is itself a *consumer* of knowledge handoffs: when it discovers an
environment or deploys itself, it reads the same conventions an agent would.
Which implies **a small set of system-level tasks, agents, and handoffs** whose
job is to produce and maintain those conventions — part of the system, not
configuration beside it.

`env_mgr` does own the agent's *own* dependencies, which is what the shipped
recipe machinery already does.

---

## 8. Preparing an environment

Before a task's executor runs:

| Prepared | Detail |
|---|---|
| **Storage** | The task's nested directory, created on both sides and made identical (§5.3) |
| **Handoffs** | The task's declared inputs, copied into the playground |
| **Workspace** | The worktree cut, dependency repositories cloned |
| **Playground** | Created, or reloaded if this is a resume |
| **Isolation** | The sandbox obtained and applied. **If it cannot be obtained, the task does not start** (§4.2) |

The last row is a hard requirement, not a best effort.

---

## 9. Relationship to the shipped `env_mgr`

Reused, not rewritten — but **no longer frozen**, and that distinction is the
point of the table. Until 2026-09-04 these modules were held *byte-identical* by
a test, which was a scope fence for the round that built the new subsystems and
not a quality gate. The fence is retired with criterion 22 (§10), so "reused"
now means what it says: the same modules, changed where the design says to.

| Shipped | Status |
|---|---|
| `outcome.py`, `report.py`, `registry.py`, `versions.py` | **Reused unchanged.** They solve agent-dependency provisioning, which stays in scope. Nothing this round needed to touch them |
| `installers/` | **Changed**, and it is the reason the fence had to go rather than an exception to it: `claude.py::_present_names` was a check that could never pass, and two of its tests encoded a `claude plugin list` format the CLI does not produce. The fence was holding a wrong test in place |
| `recipe.py`, `runner.py` | **Changed** by the removal of the layer field, §9.1. `recipe.py` loses the field, its validation and its `_CLI_KEYS` entry, and gains a dated migration guard that rejects a stale `layer:` rather than letting it fall through into `Item.spec`; `runner.py`'s version-conflict Outcome loses the label it keyed on |
| `layer.py` | **Deleted.** `LAYER_ORDER` and `layer_index`, and neither was load-bearing: `layer_index` had no production caller at all, and the field had exactly one runtime reader — a conflict label that each recipe's own subprocess had already reduced to a single key |
| `cli.py` | **Extended** with domain and zone inspection sub-commands |

New: the filesystem manager (§5), isolation (§4), the agent environment (§6),
the handoff storage entry (§6.3), and metadata (§3.1).

**Decoupling is structural**: nothing new imports the installer machinery, and
nothing in the installer machinery learns about domains or zones.

Also needed, and recorded in [`../../docs/TODO.md`](../../docs/TODO.md): **a
submodule that sets up the Claude Code SDK** from an API key and endpoint in
config, so a fresh machine can run an agent without hand-setup.

The shipped README records three v1 limitations. The stubbed workspace default is
superseded by §6.1; the other two stand.

### 9.1 Where an installed thing lands

**There is no layer field and no level vocabulary.** An earlier revision of this
design had five, then four; both were removed, because the destination is
*derived* and an author restating it is a second writer of a fact the file path
already carries. The derivation is two questions.

> **1. Is it AI material — a `.claude/` tree the agent harness reads as its own
> configuration?** If not, it installs **system-wide**, once, for everyone.
> **2. If it is: did the *agent* declare it?** If yes, **project level**.
> If it was declared by the task package, **user level**.

| what | where | Claude Code calls it |
|---|---|---|
| binaries, language packages, OS packages, any tool an agent shells out to | system-wide. Where the installer accepts a prefix, **the agent_sys root**; where it does not (`apt`, a system interpreter), wherever it lands | — |
| a `.claude/` tree declared in `main.yaml` or `default.yaml` | the agent_sys root's Claude config | **user level** |
| a `.claude/` tree under an agent's own asset directory | the agent's workspace root | **project level** |

**The second and third rows are not our invention — they are Claude Code's own
two scopes**, and adopting them is the whole point: a harness that already
distinguishes user-scope from project-scope does not need a parallel hierarchy
laid over it. Question 2 is answered by *which directory the file was found in*,
so nothing has to be declared.

Two consequences that follow from the rows above and are stated because they are
behaviour changes, not restatements:

- **User level is shared across the agents of a run.** A skill declared in
  `main.yaml` is visible to every agent, not copied per agent. That is what user
  scope means; if a skill must be one agent's only, the agent declares it.
- **User level outlives a run.** The agent_sys root is deliberately not under a
  run root (PR 154: a resident daemon has to outlive any single run), so
  package-declared material persists into the next run. **Left as measured, not
  designed around** — see `../../docs/TODO.md`.

`serena` is the worked example, because it is both: the **binary** is one
installation every agent uses, and the `.mcp.json` that names it — with that
agent's own `--project` — is per-agent. They are two things, not one thing in two
places, and only the second is a copy.

**The shared root is not defined here.** `agent_sys` has exactly one, introduced
by PR 154: `AGENT_SYS_HOME`, defaulting to `~/.infera_agent_sys` and laid out like
`~/.local` (`bin/ share/ state/ run/`). This section adds a rule about *which*
things go there; it does not add a second root, and a module that needs the path
takes it from that owner rather than recomputing it. Until PR 154 merges, the
constant does not exist in this tree — see [`../../docs/TODO.md`](../../docs/TODO.md).

Two properties follow, and both are the reason the root is a single knob:

- **Nothing installs into `/usr/local/bin` or `~/.local/bin`.** A destination
  outside the root is a defect, not a fallback.
- **"Do not change host state" stays satisfiable by configuration**, because one
  variable relocates every install. A tool whose upstream default writes to
  `$HOME` (`uv`, serena) therefore needs its own variable pinned into the root —
  the pin is what makes the rule true, not the intention.

---

## 10. Acceptance criteria

**Criteria 2–14 are CI-enforced**, in `tests/env_mgr`, on every commit. None of
them needs an AI agent or an API key: they need a subprocess and a filesystem.
Criterion 6's "an agent writes a script" is incidental — what is being tested is
that a subprocess cannot escape, and `python3 -c` is a subprocess.

This matters because these are the system's central safety claims. Leaving them
to the demo would mean the one thing the design turns on is checked by hand, on
a developer's machine, when someone remembers.

**When no sandbox mechanism is available, the suite fails.** It does not skip.
A green suite that silently proved nothing is the same failure mode §4.2 refuses
for the runtime, and it would be worse here: the tests exist precisely to catch
the case where confinement is not working. The CI image guarantees `bwrap` or a
Landlock-capable kernel, or the suite is red.



1. A domain is registered, reloaded idempotently, and its kind determines its
   layout.
2. **A subtask's storage is nested inside its parent's**, and a task's reach is
   decided by canonical containment against its own subtree.
3. **`startswith` is not the check.** Each of the three documented defeats — a
   sibling directory sharing the zone's name prefix, a symlink inside the zone
   pointing outside, and a `..` traversal — is denied.
4. **Canonicalisation fails closed.** A path whose resolution raises — a broken
   or looping symlink — is denied, not passed through as a raw string.
5. A path containing a NUL byte is rejected.
6. **A scripted bypass is blocked.** An agent writes a Python script that opens a
   file outside its zone and executes it; the read and the write both fail. The
   same script succeeds *inside* the zone, so the confinement does not break the
   work.
7. **The confinement is inherited.** A child process spawned by the agent —
   `bash -c` — is equally confined and cannot drop the restriction.
8. **No sandbox means no start.** With `bwrap` absent and Landlock unavailable,
   the task refuses to start rather than running unconfined, and says so.
9. **The sandbox chain degrades in order**: `bwrap` when present, Landlock
   otherwise, refusal when neither.
10. The zone root is never taken from agent-supplied input: an agent that
    proposes a `working_directory` outside its subtree is denied.
11. **The policy is not writable by the agent it governs**, and neither are
    `.git/hooks`, `.git/config`, or shell rc files.
12. **A read outside the granted set is denied**, including a path in no
    governed region at all. The granted set is the default system hierarchies,
    the task's own zone, and whatever its permissions name (§4.5).
13. **A producing task cannot read a validation's checking standard**, resolved
    entirely by §5.1's containment.
14. **A sibling zone created after this task's sandbox was built is
    unreachable**, with no sandbox rebuild — the property the allow-list buys and
    a deny-list could not provide (§4.5).
15. Sync runs **once, at task start**, makes both sides identical, and is scoped
    to the task rather than the root. Nothing syncs afterwards without a call.
16. **The playground is not synced**, and local and remote contents may differ
    after a run.
17. A playground survives a resume; nothing depends on its contents having
    survived.
18. Remote↔local operations are callable by an agent as **tool calls** with
    schemas.
19. An agent copies a handoff into its playground and works on the copy; the
    stored artefact is byte-identical afterwards and its digest still verifies.
20. A workspace is a worktree, not a clone, and the main checkout is unmodified.
21. `env_mgr` reads its cluster conventions from a knowledge handoff; changing
    that handoff changes its behaviour without a code change.
22. **The shipped recipe and installer machinery keeps working**: `pytest
    agent_sys/tests/env_mgr` passes.

    **Revised 2026-09-04, and the earlier wording is kept here because the
    change is the point.** It read *"is **untouched**: … passes unchanged"*, and
    a test asserted that first clause literally — `git diff HEAD` over eight
    paths had to be empty. That was a **scope fence** for the round that built
    the new subsystems: do not rewrite the shipped machinery while adding to it.
    It was not a quality gate, and it was never meant to outlive its round.

    It has now been reached from the other side. The layer model is being
    removed by design (§9.1), and `installers/claude.py::_present_names` was
    found to be a check that can never pass — a fix that breaks **none** of the
    machinery's tests, because none of them covered it. Two of those tests turned
    out to encode a `claude plugin list` format the CLI does not produce, so the
    byte fence was holding a wrong test in place.

    So the fence is retired, not quietly widened: the owner ruled this round a
    design-level change, *"没用的测试去掉，该补的测试补上"*. What remains is the
    second clause, which is the property anyone actually wanted — **the tests
    pass**, with the tests themselves corrected where they were wrong.

---

## 11. Open questions

| Item | Status |
|---|---|
| **Sandbox on non-Linux** | The chain is Linux-specific. macOS has Seatbelt, which both surveyed harnesses use. Out of scope for the alpha's target, and it will be the first portability question |
| **What the sandbox does to remote execution** | §4 confines a local process. An `ssh` session's confinement is the remote side's problem, and nothing specifies it. The honest alpha answer is that remote execution is less isolated than local, and that should be written down rather than discovered |
| **Zone identity across machines** | Whether a zone id is the same on both sides or the mapping translates it. Same-id is simpler and assumes the operator laid both sides out identically |
| **Sync conflict** | §5.2's weak mapping is `rsync`, which has a direction. Which side wins when both changed is unspecified — and "the caller decides" will lose data eventually. Recorded in [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) |
| **Concurrent tasks in one domain** | Nesting makes each task's storage its own, so the common case is handled. Two runs of the *same* task is the case that is not |
| **Auto-detection scope** | §3.1 prefers auto-detection. What is actually detectable — NFS yes, worktree yes, remote reachability only by trying — has not been enumerated, and the boundary decides how much an operator must declare |
| **System-level tasks** | §7 concludes that maintaining cluster conventions implies system-level tasks, agents, and handoffs. None is specified, and they are the first thing that will need one |

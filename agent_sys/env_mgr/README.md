# env_mgr

Environment manager for the agent work system. Driven by one
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

- **Skip-with-warning is not implemented** (design §4.1). env_mgr does not yet
  detect that an item is already satisfied elsewhere and skip it with a
  warning. Each installer's own idempotent `check` covers the practical
  single-host case. Consequently `--on-conflict weak` is a **v1 no-op**: it
  skips conflict detection entirely and proceeds with install (exit 0), whereas
  `fail` records the conflict and halts before install (exit 2). Only
  *version-conflict detection* under `fail` is active. (This bullet described
  walking a chain of *layers*; the layer model is gone — `docs/spec.md` §9.1 —
  and what is unimplemented is the skip, not the chain.)
- **the workspace default is stubbed** — the default
  `$HOME/workspace.infera.aiopt` path, its warning, and user-bin symlinking are
  not wired up yet. The default appears nowhere in this tree.
- **system apt is detect-and-print only** — the `apt` installer never runs
  sudo; it prints the `apt-get install` line for you to run.

## Relationship to the former `../helper/` demo

`../helper/*.sh` was the original shell demo. It has been removed; its logic
now lives entirely in this recipe and the installers above.

---

# Above the wall: paths, zones, isolation

Everything above is the **shipped installer machinery**, reused rather than
reimplemented (spec §9, criterion 22 — no longer *frozen*: the layer model was
removed from it on 2026-09-04). Everything below is `docs/design.md` §2's subtree:
`meta.py`, `fs/`, `isolation/`, `grants.py`, `workspace.py`, `material.py`,
`sync.py`, `remote/`, `prepare.py`, and the CLI's two new sub-commands.

**The decoupling wall is enforced, not intended.** Nothing new imports the
installer machinery and nothing in the installer machinery learns about domains
or zones; `cli.py` is the single allowed exception.
`tests/env_mgr/test_imports.py` walks the AST and asserts both directions,
because "structural" is a claim about the import graph and an import graph is
checkable.

**That test was itself checked, and it had a gap.** It classified modules
against two hand-written lists, and `harness.py` was added above the wall
without being added to either — so neither direction applied to it. Injecting
`from env_mgr import harness` into `installers/bin.py` left the file at 41
passed. The list of what is *above* is now derived rather than typed, so the
rule reads **below may import only from below** and a new module is covered from
the moment it exists; a partition test keeps the two sides exhaustive. Probe:
`scratch/impl-2026-08/env_mgr/p8_the_wall_had_a_gap.py`.

## What a consumer is staged: `content/`, not the version directory

`fs/layout.py::stage` copies a stored version into the zone (spec §6.3 rule 2 —
an agent works on a copy). It copied the whole of `v<N>/` until `handoff`
measured what that put in front of an **independent validator**: `manifest.yaml`,
`validation.yaml`, and the producing agent's own `claim/self_check.yaml`. It now
copies the version's `content/`, which is the same answer `handoff.copy_out`
already gave — the wide copy was a second answer to a settled question.

The shape matches `copy_out` exactly: `copytree(<v>/content, dst)`, so the
artefact's files land **at** the mapped path with no `content/` level inside it.
`validator` records that path in `materials.json`, so it is a contract, and
preserving a `content/` level would have made every body quote `handoff`'s
directory vocabulary to reach its own inputs. Pinned by
`test_a_staged_input_is_the_artefact_itself_not_a_content_subdirectory`, with the
exposure itself named by
`test_staging_does_not_hand_a_consumer_the_producers_own_claim`.

**Withholding the manifest is the point, not a side effect** — but the route it
leaves is open to *in-process* code, not to a body, and that was measured after
the change rather than before it. `handoff.get_manifest` verifies a digest where
a staged copy does not, so `validator`'s own Python and `agent/gate.py` are
strictly better served by the store; `validator` confirmed their prior-verdict
path is `store.read_verdicts` and never reads the staged tree.

A **confined body** is a different matter. Measured
(`scratch/impl-2026-08/env_mgr/p11_can_a_body_reach_the_store_root.py`), granted
its zone and its inputs' `content/`:

| | |
|---|---|
| staged copy | `errno=0` |
| store root, listing | **`EACCES`** |
| another version's `manifest.yaml` | **`EACCES`** |

against an unconfined control where all four succeed. Nothing in `prepare` or
`policy.py` grants the store root — `ctx.store_root` is used only by `output_env`
and by `stage`. **So for a body both routes are closed, and the store route was
already closed before this change**: what narrowing removed is the *accidental*
route, a manifest that happened to ride along in a directory copy.
`examples/demo/logic/store.py`'s F-D5 fallback — read `AGENT_SYS_DEMO_STORE`,
walk to a manifest — does not work under confinement and did not before.

A version with no `content/` stages nothing rather than falling back to the wide
copy (`test_a_version_directory_with_no_content_stages_nothing`); no write route
in `handoff.store` produces one, so a fallback would only ever fire when the
layout is unexpected. **The probe that measured this carried that fallback and
therefore under-reported** — it showed five green suites for a narrowing the
implementation does not perform, and the four tests the real change took red had
been asserting against a store layout that never exists.
`scratch/impl-2026-08/env_mgr/p10_what_narrowing_stage_would_cost.py` records it.

## The machine these numbers came from

Every measurement in this file was taken here. **They are environment facts and
cannot be re-derived from the code**, so they are written down rather than
summarised: a later reader on a different kernel needs to know which of these
still apply to them.

| | |
|---|---|
| Kernel | Linux **6.5.0-45-generic** |
| Python | CPython **3.13.13** (the target is **3.10**; a 3.10-only failure will not surface here) |
| **Landlock ABI** | **3** |
| **`bwrap`** | **absent** — so rung 1 is never selected and every end-to-end confinement test runs on Landlock |
| Landlock layer cap | **16**; the 17th `restrict_self` is `E2BIG` |

**Three consequences that shape the design, not just the tests:**

**`restrict_self` restricts only the calling thread below ABI 8.** `all_threads()`
arrives at ABI 8 and does not exist here — and it would be the wrong thing to
want anyway, because it restricts *every* thread in the process, which is the
supervisor. Measured with the guard removed, in one process:

```
worker thread, writing outside   denied errno=13
worker thread, its own zone      writable
MAIN thread, writing outside     WRITABLE      <- the process is not confined
subprocess of the worker         denied        <- but a child is, and so is a grandchild
```

Row 3 is the honesty problem and **row 1 is the killer**: the thread that applies
confinement is confined **irreversibly**, so a runner thread that must write the
store afterwards is crippled — and it surfaces as a store bug, not a sandbox one.
That measurement is why step 7 split (below).

**The layer cap is a property of the running kernel, not of the API.** The
current `landlock_restrict_self(2)` man page says 64; the v6.1 kernel
documentation says 16. Both are right, for different kernels — the limit was
raised. Check yours; do not trust either number. It never binds here anyway,
because the supervisor spawns each executor directly so every executor carries
one layer.

**A grandchild inherits the domain.** Measured: a child confined the way `spawn`
confines it, spawning a grandchild **itself with no wrapper**, gets `EACCES`
against an `rc=0` unconfined control. That is what makes an AI harness's
self-spawned CLI confinable in principle — see the F8 note below.

## Step 7 split: `prepare` checks, `spawn` applies

Design §11.1 ended `prepare` with `apply()` — *"confinement last"*. It no longer
does, and the reason is the thread measurement above: **`agent`'s runner is
threaded by construction, and a thread that confines itself cannot finish its own
job.** `interfaces.md` §5.15.

| | |
|---|---|
| `prepare` | **checks** a mechanism exists (`select`) and refuses if not — so *no isolation, no start* now refuses **before** the workspace is cut rather than after |
| `Prepared.spawn(argv, **kw)` | **applies** it, in the child, between fork and exec |

**"Confinement last" survives in the form that mattered.** That rule existed so
the supervisor and every prior process stay outside the domain; moving the
syscall into the child achieves it **by construction rather than by ordering**,
which is stronger than the sequence that expressed it. The evidence is small and
concrete: the test for it needs **no fork of its own**, where the pre-split
version had to fork to avoid poisoning pytest.

`Prepared.confinement` is therefore **a prediction, not a report** — what `spawn`
will realise. It is safe to record because `spawn` cannot silently skip it: with
no mechanism it raises, and a `Prepared` reaching it carrying none raises too.

**Why `spawn` and not a wrapper.** `wrap_argv` works because bubblewrap *is* the
exec, so its confinement crosses fork/exec **as data in a command line**. Landlock
is a syscall against a live thread; there is nothing to put in an argv. The child's
whole job is `prctl` + `restrict_self`, because a ruleset **fd survives fork** and
is built in the parent — the smallest post-fork footprint available, which matters
because `preexec_fn` is documented unsafe in exactly a threaded process. Measured
at 150 rounds under four contending threads: no hangs. **That establishes the
footprint, not the absence of the hazard** — a fork deadlock is probabilistic and
CPython warns on principle. If a child ever hangs before exec, the answer is a
helper binary rather than a Python `preexec_fn`.

## Libraries adopted, and why

| Concern | Considered | Chosen | Why |
|---|---|---|---|
| **Landlock** | `rust-landlock` (not Python), `pylandlock`, a hand-written `ctypes` binding | **own `ctypes` binding**, `isolation/landlock.py` | **There is no maintained Python binding.** The three syscalls — `landlock_create_ruleset`, `landlock_add_rule`, `landlock_restrict_self` — have **no libc wrapper**, so *any* Python binding is `syscall(2)` by number regardless of who writes it; there is no library to wrap. The file is promoted from the measuring instrument at `scratch/design/probes-envmgr/landlock.py` that took every measurement the design cites |
| **Sandbox mechanism** | writing our own namespace code | **the `bwrap` binary, else Landlock** | bubblewrap is the mature tool and the mechanism is a *process*, so there is nothing to bind: `isolation/bwrap.py` builds an argument vector and nothing else. Codex migrated *to* a bundled bwrap after shipping Landlock, which is the same direction the chain orders them |
| **Sync** | `mutagen`, `unison`, `syncthing`, `rsync` | **`rsync`**, via `subprocess` | Spec §5.2 names it, and what is needed is a one-shot copy at task start, not a reconciler. The three reconcilers all solve the continuous problem we explicitly do not have. §9.3 adds the one thing rsync cannot do — see below |
| **Remote access** | `fabric`, `paramiko`, plain `ssh` / `docker exec` | **plain `ssh` and `docker exec`**, behind one Protocol | Two mechanisms, three methods. A library would add a dependency to hide a `subprocess` call, and neither `fabric` nor `paramiko` expresses `docker exec` at all |
| **Workspace** | `GitPython`, `pygit2`, the `git` CLI | **the `git` CLI**, via `subprocess` | The operations are `clone --shared`, `checkout`, `fetch` and `config`. `extensions.preciousObjects` and `--no-hardlinks` are plumbing flags a binding would have to pass through verbatim, and the CLI is the interface git itself documents |
| **Tree comparison** | `dirhash`, a digest library | **`filecmp` and `hashlib`**, stdlib | `handoff` owns digests (design §1.2); this module owns copies, and "is this copy byte-identical" is `filecmp.cmpfiles(shallow=False)` |
| **Config persistence** | `pydantic`, `tomllib`, `json` | **`json`**, stdlib | `meta.py` holds three flat records. Reaching for pydantic here would put a validation layer between `env_mgr` and its own file for no gain — and a *knowledge handoff* carrying the same shape is read by the same function |
| **Test harness** | a fork-per-test plugin, helper binaries | **one `conftest.py` fixture** | Landlock restriction is irreversible and inherited, so a test that sandboxes the pytest process poisons every later test. The kernel deleted its own per-test opt-in (`TEST_F_FORK` is now an alias for `TEST_F`) because *isolation belongs to the runner*; pytest does not fork, so this supplies it once |

**No new runtime dependency is added by this subtree.** Everything above is
either stdlib or a CLI that must exist anyway.

## The three things that will break a reimplementation

Each cost a measurement, and each fails in a way that names the wrong cause.

1. **`/dev/urandom` must be granted.** The Claude backend is a standalone Bun
   binary that aborts in 3 ms without it, prints *"oh no: Bun has crashed"* and
   hands the operator a crash-report URL **for the wrong project**. Granting the
   whole of `$HOME` read-write does not fix it; one character device does.
2. **Granting `/etc` does not give you DNS.** `/etc/resolv.conf` is a symlink to
   `../run/systemd/resolve/stub-resolv.conf`, and **Landlock rules apply to the
   resolved path**. One missing file, two tools, and symptoms with nothing in
   common: an immediate clean `rc=2` from `getent`, and a three-minute hang from
   the backend.
3. **A file target needs a different rights mask.** Handing a directory-only
   right (`MAKE_REG`, `READ_DIR`, …) for a non-directory is `EINVAL`, not an
   ignored bit — so an implementation that grants uniformly dies on the first
   `/dev/null` it is handed, with no indication which path caused it.

A fourth, about tests rather than about the kernel: **`returncode != 0` cannot
tell a blocked read from a failed exec.** Every denial in this suite is asserted
as `EACCES` against a named path, and every one carries a positive control.

## The Landlock layer cap, measured

`materials/07-env_mgr.md` records the man page saying 64 and 16 measured here,
and calls the pair unreconciled. Re-measured for this implementation
(`scratch/impl-2026-08/env_mgr/p1_layer_cap.py`):

```
landlock ABI = 3
layer 17: refused, errno 7 (Argument list too long)
MEASURED CAP = 16 layers stacked successfully
```

**Reconciled: both numbers are right, for different kernels.** The limit was
`LANDLOCK_MAX_NUM_LAYERS = 16` and was raised to 64 in a later kernel; the
current `landlock_restrict_self(2)` man page documents 64, and the v6.1 kernel
documentation documents 16. This kernel is 6.5 and stacks 16. So the number is a
property of the running kernel, not of the API, and
`isolation/landlock.py::MAX_LAYERS` says so.

It binds under exactly one architecture, and this design avoids it: **the
supervisor spawns each executor directly**, so every executor carries one layer
and task depth is not capped.

## `EnvManager` has two methods, and the rule that said one is amended here

Design §11.4 said **"one method, and it stays one"**, and named the hazard:
*"a second is how the runner would start making environment decisions."*
`prepare_validation` is the second, and this is the amendment rather than a
reinterpretation of the rule to fit it.

**What forced it.** Two modules were answering *where does a validation go* —
`validator` allocating with `mkdtemp`, and `layout.validation_zone` placing a
sibling. The `mkdtemp` zone is unreachable from a confined producer, so
criterion 13 looked satisfied; it was satisfied **by accident of location**,
because nothing granted `/tmp`, and not because anything decided a placement.
An accident is not a property, and the next person to move that zone root would
not have known they were removing one.

**What decides the shape is not the hazard's wording but what the object is: a
`Context` bound once.** A validation zone needs `ctx.domains` and
`ctx.store_root` — the *same* bound context — so a second registered component
would bind one configuration twice, and one fact with two writers is what the
rule was protecting against in the first place. A second method on the object
that already holds the context is the smaller promise.

**The guard survives the amendment.**
`test_env_manager_exposes_exactly_these` pins the *set* rather than the count,
so a third method still fails a test and still needs a decision. Deleting the
guard would have removed the pressure; leaving it at one would have blocked a
correct change.

```python
prepare_validation(task, execution, phase) -> ValidationZone(root, phase, materials)
```

Resolved by name, never imported — `validator` may not import this package, and
an import edge is permanent where a name lookup is not. `phase` is read for its
value, the same structural read this module already uses for
`task_graph.Access`. `materials` are **copies** staged out of the store, not a
grant on it, so a validation cannot edit what it validates.

**It confines nothing, and that is a boundary rather than an omission** — a
phase runner calling it is the supervisor, and confining there would confine the
supervisor. That was the whole of `interfaces.md` §5.15 when this method landed.

**§5.15 has since been settled and this paragraph's original reason is gone.**
It said `prepare` applies Landlock to its own process, which stopped being true
with the step-7 split; and it said who confines a validation body was an open
third question, which `spawn` answers — the same call confines a validator's
`entry.sh` and a program task's argv, which is why the entry was retitled from
*"what confines a validation body"* to *"what applies a policy to a body"*.

**What is still missing is small and nobody has asked for it**: a validation body
runs unconfined today, because `ValidationZone` carries no `Policy` and therefore
no `spawn`. The mechanism exists and the placement is right — the separation
rests on **placement**, which is a property, rather than on the accident of
location it rested on before. It does not yet rest on the kernel. Building the
policy half unasked would be guessing at what a validation may reach, which is
`validator`'s question and not this module's.

## Three seams found by implementing, and where each of them ended up

Each was raised naming both sides rather than fixed quietly, and each was built
so that the declaration in force at the time kept working. Two are now in the
contract; one is still open.

| # | Found | Now |
|---|---|---|
| 1 | `interfaces.md` §4.6 and `protocols.py` declared `prepare(task, execution)`; design rev. 4 §11.5 declared `prepare(task, execution, agent_spec)`, because `agent` spec §3.1's `env` and design §3.4's `rules` / `hooks` / `skills` name this module as their consumer and had no route to arrive through | **In the contract, §4.6 rev. 5.** Built as `agent_spec: Any = None`, so every two-argument call still works — `test_env_manager_satisfies_the_frozen_two_argument_call` |
| 2 | `Prepared` had five fields and nowhere to put the environment `material.deploy` computes, so `prepare` dropped it | **In the contract, §4.6 rev. 5**, as `environment: Mapping[str, str]` |
| 3 | **The rung-1 obligation was uncallable.** §4.6 told the caller to run `bwrap_argv(prepared.policy, availability, argv)`, and `Availability` is on neither `Prepared` nor anything `agent` may import (§4.4). Handing over raw material for the caller to assemble something that is this module's to compute | **Settled, §5.11.** `Prepared.wrap_argv(argv)` — one call, `Availability` stays here |
| 4 | `SyncReport` is `(sent, received, conflicts)` in `protocols.py` and `(copied, deleted, conflicts)` in design §9.3 | **Open, and cosmetic.** Built to the frozen names. `sent` counts files transferred and `received` counts files deleted, which fits neither name well |
| 5 | `Context` has seven fields and design §7.1.1 needs an eighth: *"a declared `repos` entry is a key into the run configuration, and `Context` carries the mapping"* | **Open.** `prepare` reads `ctx.repo_locations` with `getattr`, so a seven-field `Context` still works and a task declaring `repos` simply cannot be prepared until the root supplies the map |

`wrap_argv` returns `argv` unchanged under Landlock and the bubblewrap command
under rung 1. It raises `NoConfinement` when there is nothing to wrap with,
**including the binary having vanished since probe time**: resolved at exec
rather than remembered, which is the same rule as canonicalising per check.
`bwrap` is absent on this machine, so the branch is tested against a fake binary
and asserts the argv.

**`spawn` is the general form and `wrap_argv` is its bubblewrap half.**
`wrap_argv`'s shape does not carry to Landlock: bubblewrap *is* the exec, so its
confinement crosses fork/exec **as data in a command line**, while Landlock is a
syscall against a live thread and must run in the child. So the caller gets a
spawn, and branches on the mechanism nowhere.

`protocols.Prepared` and `prepare.Prepared` are two declarations of one shape —
the duplication `engineer_principle.md` §1 names, admissible for the same reason
`Policy.with_` and `Zone.contains` are, because a declaration's bodies are `...`
and a `NamedTuple` cannot carry a real method and stay one.
**`test_prepared_matches_the_declared_surface` is the price of that**, and
without it the honest move would be to have only one.

### One output resolves to two granted paths, and `v<N>/` itself to none

The user's ruling, and it resolves a contradiction between two earlier ones: the
`done_by_self_check` claim was ruled a **sibling** of `content/`, and the grant
was separately ruled to narrow **to** `content/`. Under both the agent cannot
write the thing it is asked to write.

```
v<N>/content/     the artefact          granted
v<N>/claim/       the producer's claim  granted, write only
v<N>/manifest.yaml                      reachable by neither
v<N>/validation.yaml                    reachable by neither
```

**The narrowing is not tidiness.** Under `interfaces.md` §4.14 the manifest *is*
the seal, so an agent granted `v<N>/` could write `manifest.yaml` and publish its
own unsealed version. That is a new exposure created by §4.14, not the old
"a digest is not a security boundary" one.

**`claim` is this module's name** — the user ruled the destination and left the
name here. A *directory* rather than a file, so a second claim needs no second
ruling. Measured
(`scratch/impl-2026-08/env_mgr/p9_a_sibling_claim_dir_survives_seal.py`): a
sibling survives `seal` and leaves the digest byte-identical at
`718d7aeb31a76c32…` — the same value `handoff` measured from the other side,
which is what makes the two comparable. Their probe found the alternative moves
it to `0b3a5c9f…`, so a claim inside `content/` would make the producer's
self-assessment part of the artefact's **identity**.

**A read grant gets `content/` alone**, because a consumer has nothing to claim.
That forecloses a body reading its input's `manifest.yaml`. That was raised as
an open question and **`handoff` has answered it: nothing needs to.** Checked
from this side rather than relayed — `agent/gate.py:91` is the only
`get_manifest` caller outside `handoff` and its tests, and `runner.py:574`
reaches it *after the executor returns*, supervisor-side. By design too: spec
§6.3 has a consumer work on a copy and `copy_out` verifies the digest before
returning, so integrity arrives as content the body can trust rather than a
manifest it must check. It reopens only for a consumer needing **provenance**,
and the answer there is an operation on the store, not a wider grant.

The two spellings of these directory names are pinned across the packages by
`tests/interfaces/test_handoff_layout.py::test_the_two_spellings_of_the_granted_subdirectories_agree`
— §8.1's forced duplication with a drift test, because a divergence here is a
**dispatch failure** rather than a cosmetic one.

**Both directories are created by `handoff.allocate`, and `grants.py` creates
nothing.** §4.18 ruled it — *the allocator creates every directory it expects to
be granted* — and this module is what found it, by measuring both outcomes for a
granted path that does not exist: non-optional is a `FileNotFoundError` that
kills every output-producing dispatch, because `landlock.py:198` opens every
granted path and `Granted.optional` defaults `False`; optional drops the rule
silently. **The agent cannot create it either**, since `mkdir` inside `v<N>/`
needs write on `v<N>/` — which is exactly what the narrowing removed.

`grants.py` did create `claim/` for one commit, as a bridge, because the user
left the *name* here and `handoff` could not act without one. `store.py:334`
does it now and the bridge is gone. **A resolver with a side effect is a
resolver a test cannot call twice**, and `allocate`'s `os.mkdir` is not
`exist_ok`, so a racing creation would turn a dispatch into `FileExistsError`.

### An output's directory needed a declared name — `demo`'s F-D17

Since §4.14 the allocator creates `v<N>/content/` at dispatch and `grants.py`
grants it, so the path a body must write to **exists and is reachable**. It had
no *name*: it lived only in `prepared.policy.granted`, which no body ever sees.
A demo body died with `KeyError` and the run reported `declared output … was
never delivered` — the completeness gate naming an absence with no cause.

`prepare` now exports one variable per output:

```
AGENT_SYS_OUTPUT_<KIND> = <store>/<hid>/v<N>/content
```

**Keyed by kind, because that is the author's only handle.** A closure declares
`outputs: ['facts']` — a list of kind names, not slots. Keying by `HandoffId`
would name the directory by a uuid minted at submit, which no author can write
down. This is `AGENT_SYS_TASK_PACKAGE`'s argument one slot over: a path known
only at prepare time that the body cannot compute and must have.

**Exported and granted agree by construction**, not by care — the value is the
`content/` subdirectory `_version_paths` grants, and
`test_the_exported_path_is_one_the_policy_grants` asserts the containment. An
exported path we did not grant would be the evaporating allow-list one level up:
the body failing on our own instruction.

**One kind naming two output slots is exported for neither, and that is a hole
rather than a resolution.** Discriminating them by `HandoffId` would invent a
naming scheme **no author can write against** — with `outputs: ['facts',
'facts']` the author cannot address one of them either. Choosing one silently is
precisely the failure this export exists to remove. Nothing is exported, the
body's own refusal fires, and the gap is named here.

### The switch: `AGENT_SYS_NO_PERMISSIONS`, and it is **on by default**

Ruled by the user, twice. Set it — **or leave it unset, which is the default
since 2026-08-30** — and this run performs **no permission management at all**.

```sh
# nothing to do: no confinement, no grant enforcement
agent-sys run --package …

# enforce, which is now the opt-in
AGENT_SYS_NO_PERMISSIONS=0 agent-sys run --package …
```

`0`, `false`, `no`, `off` and empty all mean *the switch is off*, i.e. enforce.
Anything else, including absence, means no permission management. The double
negative is the price of not introducing a second variable that could contradict
the first; `interfaces.md` §4.22f has the argument.

Read in exactly one place — `prepare.permissions_enforced()` — and passed
down as an argument everywhere else, because a switch with three readers is
three switches. `test_the_switch_is_read_in_exactly_one_place` asserts that over
the **AST**, excluding docstring mentions: three modules name the variable in
prose, so a substring search answers a different question.

| step | off |
|---|---|
| `grants.resolve_all(enforce=False)` | best-effort, **never raises** |
| `select` / `confinement_for` | **not called at all**, `confinement=None` |
| ~~`layout.stage(narrow=False)`~~ | **reversed — the switch does not widen staging.** See below |

**The third ruled row was reversed, by applying the ruling's own line.**
Widening `stage` moves every staged input **down one level** — the artefact's
files land at `<materials>/<hid>/v<N>/content/…` instead of at
`<materials>/<hid>/v<N>/…` — and `examples/demo/bin/render.py:67` reads the
narrow shape. So the switch would have broken a body **by moving its input**,
presenting as a body reading one level short rather than as a switch. `task_graph`
measured it (`probe_narrow_staging.py`) and it collided with a `demo` fix landed
the same hour.

And widening buys no permission property in this mode: with nothing confined a
body can read the store directly (`p11`'s unconfined control succeeds on all
four reads), so narrowing the *copy* denies it nothing it could not already
reach. **What is left is a path convention, which is materialisation.**
`stage(narrow=)` stays, tested, for whoever wants the wide shape.

**Materialisation is not permission management** and is untouched: staging,
`material.deploy`, the workspace, `staged_package` and the whole of
`environment` make a file appear where a task needs it. Turning any of them off
would break the run rather than unrestrict it.

**Two rows nobody named, both the switch being wrong in the dangerous
direction.** `spawn` falls back to `select(probe())` when `confinement is None`,
and `bwrap` is absent here — so without a guard **the kill switch failed closed
in the one mode whose whole point is not to**, and it would have surfaced as
`NoConfinement` from inside `spawn` on a run that had explicitly asked for none.
And `AGENT_SYS_NO_PERMISSIONS=0` reads as **on**: treating every non-empty
string as *set* would disable enforcement for exactly the operator trying to say
no.

`Prepared.permissions_enforced: bool = False` reports it — the default follows
the switch's, so a hand-built `Prepared` claims the ordinary case, which is now
the unenforced one. `confinement is None`
would otherwise mean *unconfined* for two reasons — no mechanism, or the switch
— and §4.17a is that a fact a reader must infer is a fact a reader can miss.

#### What the switch does to `task_graph`'s version seam: **nothing**

`main` asked, twice, and the answer moved between the askings — so here is the
one that holds. Measured after the `narrow` reversal:

```
                permissions_enforced=True     permissions_enforced=False
hole            []                            []
written         ['README.md', 'items']        ['README.md', 'items']
```

**Byte-identical. The switch does not reach staging at all**, so a green run
with it on is neither more nor less evidence about the seam than one with it
off. `task_graph`'s prediction stands verbatim: **an empty input directory, no
error.**

**Two claims made here before this, both wrong, both mine.**

*"A narrowed `stage` skips a never-written version, so the body sees a missing
path — one notch louder."* No: skipping needs `content/` to be **absent** and
`handoff.allocate` always creates it, so `os.path.isdir` is true on an empty
directory and it stages. Reasoned from the skip branch without checking what the
allocator leaves behind.

*"With the switch off, an input directory holding exactly `content/` and
`claim/` is a hole — unambiguous, because a real staged input never contains
those names."* Wrong twice over. `task_graph` measured that under `narrow=False`
a **written** artefact has the same two top-level names, so the discriminator
was depth and not names; and the reversal removed the mode entirely, so there is
no signature left to look for.

Both were me improving on a prediction that needed no improving, and both came
from reasoning over control flow instead of running it.

**There *is* a discriminator, and it is `task_graph`'s.** I had closed this as
unanswerable — *"nothing distinguishes a version that was allocated and never
written from one that is empty, at the point where `stage` looks."* True
locally, and the inference is sound anyway:

```
empty   content: seal -> "nothing was written to <…>/v0/content. That directory
                          is the agent's grant and it is empty, so this attempt
                          produced no content at all"
                 list_versions=[] latest=None
written content: seal -> None (sealed), list_versions=[0] latest=0
```

`handoff/store.py:434` refuses to publish empty content, so **a sealed version
always has non-empty content, and an empty staged input is therefore always a
hole** — never a legitimately empty artefact. The discriminator exists; it is
simply not local to `stage`, and nothing needs adding to the store.

So the debugging line does exist after all:

> **An empty staged input directory is always a hole.** If a body reports no
> input, compare `Execution.input_versions[hid]` against the version that has a
> `manifest.yaml`.

**And the way I nearly missed it is the day's shape again.** My first probe
wrapped `seal` in `try/except` and printed *"ACCEPTED"* — `seal` **returns** a
refusal reason rather than raising. The same output printed `list_versions=[]`
one line down, which said the version was not published. **The control produced
the correct signal and I read past it.**

### Holes have no reaper, and that is named rather than absorbed

§4.14 pre-allocates an output version at dispatch, so a failed attempt leaves a
`v<N>/` that `latest` skips. **Correctness needs no reaper** — a hole is skipped
rather than compacted, which is the whole point of the design. **Disk hygiene
does**, and nothing owns it: the directory is created by `handoff.allocate`, the
`claim/` inside it by `grants.py`, and neither has a lifecycle hook that fires
on a failed attempt. Recorded here so it is not mistaken for done.

### Relocating `CLAUDE_CONFIG_DIR` is what made a confined agent fail to log in

`material.deploy` points `CLAUDE_CONFIG_DIR` at a per-attempt directory inside
the zone, so that a run's transcript does not change with the reviewer's
dotfiles. **It also moves away the `env` block that holds the endpoint and the
auth header**, and the CLI then answers `Not logged in · Please run /login`.
That is this module's characteristic failure once more — *the symptom names the
wrong cause* — and it is why an AI task had never authenticated.

Three arms, one prompt, one machine
(`scratch/impl-2026-08/env_mgr/p7_relocated_config_loses_auth.py`):

| arm | `CLAUDE_CONFIG_DIR` | block injected | |
|---|---|---|---|
| A | the operator's own | — | `rc=0` `OK` |
| B | an empty directory | no | **`rc=1` `Not logged in · Please run /login`** |
| C | an empty directory | yes | `rc=0` `OK` |

**B is what every confined agent got.** `demo`'s preflight runs the same binary
**unconfined** against the operator's own config dir, so it passed and the task
then failed — the check and the run were not testing the same thing.

`harness.py` is the other half of the relocation: *if we move the config, we
carry what the config provided.* No library — `json` and the file the CLI itself
reads. Three properties are load-bearing rather than incidental:

- **An allow-list, and the operator's own.** Only keys the block *names* are
  forwarded; a supervisor variable absent from it does not travel, so this is
  not a wholesale environment leak into the sandbox.
- **The live value beats the file**, because an operator who exported an
  override for this session meant it and the file would silently undo it.
- **Values never reach a message.** The block routinely holds a subscription
  key; the one raise (a settings file that exists and does not parse) carries
  the path and the parser's complaint and nothing else.

`CLAUDE_CONFIG_DIR`, `CLAUDE_CODE_TMPDIR`, `TMPDIR` and `PATH` are never
forwarded. The first three would undo the relocation that made carrying the
block necessary; `PATH` is *derived from the granted set* precisely so that it
can never name a directory the kernel will refuse, and a settings file written
for an unconfined machine names several. An agent spec's **declared** `env` may
still set any of them — an author saying so outranks a default.

### The AI-backend hole: not "unconfinable", and the word was doing damage

Reported by `agent` as F8. The SDK spawns the `claude` CLI **itself**, so no
caller ever sees that argv and neither `wrap_argv` nor `spawn` can reach it.
`agent` refuses to start such a task, which is the right call and the only thing
standing there.

**This note said "unconfinable under bubblewrap", and both halves were wrong.**
Kept as a correction rather than silently fixed, because the reason a finding is
recorded goes stale faster than the finding, and a stale reason is what makes
people stop looking.

**Wrong about the reach.** Since the step-7 split `prepare` confines nothing, so
a child not started through `spawn` is unconfined under **every** mechanism, not
only rung 1. But nothing was lost: pre-split, Landlock did confine the runner's
thread and a self-spawned CLI *did* inherit — reachable only from a
single-threaded caller, and `apply()` refuses above one thread, which the runner
always is. **In any configuration that could run, an AI task was never confined.**
The split made it honest, not worse.

**Wrong about the word.** Measured
(`scratch/impl-2026-08/env_mgr/p5_grandchild_inherits.py`): a **grandchild**
inherits the domain.

```
unconfined grandchild             rc=0    (read succeeded)
grandchild of a confined child    rc=13   (EACCES)
```

The child was confined the way `spawn` confines it and then spawned the
grandchild itself, with no wrapper — the SDK's exact shape. So the true statement
is **"not confinable in-process"**: if the harness ran in a `spawn`-ed child, the
CLI it spawns would be confined with no `cli_path` shim, no argv interception and
no cooperation from the SDK — the three things everyone assumed were required.

**And it is not free, which is `agent`'s half and could not be seen from here.**
Out-of-process costs **level 2 entirely**, not a handle: `interrupt()`,
`instruct()` and `query()` are all calls on an in-process `ClaudeSDKClient`;
`monitor`'s `Pushable` *is* that handle, so the push path degrades to
escalate-only; and `interrupt()`'s drain reads `terminal_reason` off the message
stream, so a control channel alone would not do. That is a decision about what
the system is — a confined agent that cannot be interrupted may well be worth
more than an unconfined one that can — and it is on the roadmap rather than open,
so that *soluble* is not read as *small*.

## The path environment-variable system, and the four names it refuses

`refine.task_package.define.md` item 3 asks for *"一套路径环境变量系统"* — package
root, workspace, handoff root, playground; local and remote; and per-agent `my_*`
slots. `env_mgr/paths.py` is that system. This section records what was found
before it was designed, because two of the findings changed the answer.

### It is a sixth call site, not a sixth kind of source

`Prepared.environment` had five contributors — derived `PATH`,
`AGENT_SYS_TASK_PACKAGE`, `grants.output_env`, `grants.input_env`,
`material.deploy`. The second of those was **already a zone-path variable**,
exported by one hand-written line at `prepare`'s step 6a, and this requirement is
that line generalised: the package is one of the zone's directories and had a
name, while `workspace/`, `playground/`, `handoffs/` and `logs/` had none. So the
kinds stay five and the zone-path kind grows from one name to six.
`PACKAGE_ENV_VAR` moved to `paths.py` and is re-exported from `prepare`, because
one fact may not have two writers and `tests/cli/test_isolation_shown.py`
imports it from there.

**Per-agent components did not add a sixth kind either**, and the reason is
worth stating because it looks like one. `agent_assets.install` contributes
`AGENT_SYS_AGENT_ASSETS`, and it reaches `Prepared.environment` *through*
`material.deploy` — the fifth contributor, whose whole job is already *what this
agent needs*. What components genuinely added is not a sixth environment source
but **two destinations that are not an environment at all**:
`Prepared.mcp_servers` and more entries in `Prepared.tools`. That is why
`material.deploy` returns a `Deployed` value now instead of a `dict[str, str]` —
the mapping had nowhere to put a nested server declaration or a live Python
object, and a second function returning them would be one act split in two.

| variable | value | the user's name for it |
|---|---|---|
| `AGENT_SYS_MY_ZONE` | `<zone>` | — (see below) |
| `AGENT_SYS_TASK_PACKAGE` | `<zone>/package` | `task_package_root`, unchanged |
| `AGENT_SYS_MY_WORKSPACE` | `<zone>/workspace` | `my_agent_workspace` |
| `AGENT_SYS_MY_PLAYGROUND` | `<zone>/playground` | `my_agent_playground` |
| `AGENT_SYS_MY_HANDOFFS` | `<zone>/handoffs` | — (`等等`) |
| `AGENT_SYS_MY_LOGS` | `<zone>/logs` | — (`等等`) |
| `AGENT_SYS_AGENT_ASSETS` | `<zone>/package/<AgentSpec.assets>` | — (added with per-agent components) |
| `AGENT_SYS_INSTALL_REPORT` | `<zone>/logs/agent_assets.install.json` | — (ditto) |
| `<any of the above>_REMOTE` | the same path under `sync.remote_root(zone, mapping)` | `*_romote` |

`AGENT_SYS_AGENT_ASSETS` is **the one name in the family whose value is not a
zone subdirectory**, so `paths.py` owns its spelling and `agent_assets.install`
binds it. It still obeys the family's rule — exported and granted agree — because
the staged package is inside the zone and `prepare` grants the zone recursively.
It is not derived from `AGENT_SYS_TASK_PACKAGE` by a body, because the relative
part is the agent spec's and a body has no route to an agent spec.

**Every name in this table is a path inside the zone**, and that is now
without exception. `AGENT_SYS_ADDONS_ROOT` used to be here, naming
`agent_sys/env_mgr/addons/` and defended as *the same rule run the other way* —
`isolation/policy.py::addon_grants` composed a `READ_EXEC` grant on it under the
identical condition that emitted the name. It went with the `agent_plugins:`
declaration key (`docs/spec.provisioning.md` §4): an add-on is installed by a
recipe, the recipe runs unconfined and copies what it needs into the zone, and
nothing confined reaches back out. Deleting the last exported out-of-zone path
is what the removal was for.

A name whose directory does not exist is **not exported**: the zone's
subdirectories are one per registered domain kind, so a run with no `PLAYGROUND`
domain has no `<zone>/playground`, and naming it would instruct a body to use a
path that is not there. That is `grants.output_paths`' rule — absent, never
present-and-empty. The `_REMOTE` half is absent entirely when no mapping covers
the zone, which is **every production configuration today**:
`cli/environment.py` passes an empty mapping and states that this is legitimate
rather than a degradation, and `remote/tools.py` has no production caller either.

### The four `*_root` names are refused, and it is a measurement

`agent_workspace_root`, `agent_handoff_root` and `agent_playground_root` resolve
to registered **domain** roots, which sit outside the zone — and a fourth
candidate, `Context.store_root`, sits outside it too. Measured against a real
Landlock ruleset built from exactly the policy `prepare` composes, with an
in-zone positive control succeeding in the same confined child
(`scratch/ui-yaml-2026-08/w2/p13_are_the_root_paths_reachable.py`):

```
CONTROL my_agent_workspace       errno=0  OK
agent_handoff_root (zone tree)   errno=13 EACCES
agent_workspace_root (domain)    errno=13 EACCES
agent_playground_root (domain)   errno=13 EACCES
ctx.store_root                   errno=13 EACCES
```

All four read cleanly unconfined, so this is a denial and not an absence.
Exporting them would break the rule stated above for `AGENT_SYS_OUTPUT_<KIND>` —
*exported and granted agree by construction* — for four of the eleven names the
user listed. `AGENT_SYS_MY_ZONE` is the one root in the user's sense that **is**
granted, and it is exported in their place.

A second finding, independent of confinement: **the WORKSPACE and PLAYGROUND
domain roots are read by nothing.** `DomainRegistry.register` creates the
directory; the only place a `Domain.root` then leaves the registry is
`storage_root()`, which filters to `HANDOFF_STORAGE`, and `register`'s own
idempotency check. Nothing iterates the registry and nothing calls `get`. So even
unconfined, three of the four have no established meaning to export. Reported
rather than invented.

And a vocabulary hazard worth one line: `DomainKind.HANDOFF_STORAGE` names the
domain that roots the **zone tree**, while the store the artefacts live in is
`Context.store_root`, a separate field. `cli/environment.py` sets them to two
different directories. "Handoff root" therefore has three candidate referents,
and `AGENT_SYS_MY_HANDOFFS` carries the third — the staged-inputs directory
inside the zone, which is the only one a body can read.

### `my_*` is indexed by the zone, and could not be indexed by the agent

The requirement's `my_*` prefix needs an identifier for a per-agent slot. Four
candidates, and only one survives:

| candidate | generated | stable across a retry or resume? | filesystem-safe |
|---|---|---|---|
| `AgentId` | `AgentMgr.instantiate`, per launch | **No.** A fresh id every call, and its docstring says criterion 21 *forces* that — after a resume the stack top must report a different agent | yes, a uuid |
| `TaskId` | at task construction | yes | yes, a uuid |
| `Execution.attempt` | `push_execution`, per attempt | changes by design | yes, an int |
| a run id | does not exist | — | — |

`AgentId` is therefore unusable: a `my_agent_workspace` keyed on it would be a new
empty directory on every attempt, and spec §6.2's *"a playground survives a
resume"* would be false by construction. What indexes the slot is the **zone**,
whose directory name is already `task.<TaskId>.<attempt>.<hash>` — the stable
identity and the varying attempt, in the one name. So `my_*` reads as *this
attempt's*, which is what makes it different from a root, and no new identifier
was introduced.

A run id exists nowhere in `agent_sys`; `cli/environment.py` mints a
timestamp-plus-uuid stamp for a demo directory and nothing else uses it.

### Libraries: nothing was adopted, because nothing is expanded here

The obvious reading of "a path variable system" is that it needs a substitution
syntax, and the survey below is why it does not. **This module produces a
name→path table; it expands nothing.**

| Considered | Verdict |
|---|---|
| `os.path.expandvars` | Reads `os.environ` and nothing else — its signature is `expandvars(path)`, with no mapping parameter, so it cannot expand against a prepared environment that does not yet exist in this process |
| `string.Template` | The right shape if expansion were needed: `${name}`, `safe_substitute(mapping)`, `idpattern` `[_a-z][_a-z0-9]*`. Recorded as the answer for whoever needs one — `spec_loader`'s `${TASK_PACKAGE_ASSERT_DIR}` is that caller, not this module |
| POSIX shell parameter expansion | **Already in use and already sufficient.** `examples/demo/bodies/produce/entry.sh` writes `${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?…}}` — default-if-absent and fail-loudly-if-absent, which is the whole measured need, performed by `/bin/sh` at no cost to us |
| XDG base directories | The naming *pattern* is worth copying and the mechanism is not: `_HOME` for the single directory, `_DIRS` for a search path. We have no search path, so every name here is singular |
| Kubernetes downward API | The closest prior art for the `my_*` half: `env.valueFrom.fieldRef` injects a pod's *own* identity, on the stated principle that a container should be able to learn about itself without coupling to the platform's API. Its documented caveat is ours too — the values are static for the process's lifetime, which is why the index has to be the zone and not something re-minted mid-attempt |
| bubblewrap / OCI | No prior art at all. `--bind SRC DEST` names paths positionally; neither runtime has a vocabulary for *"the root of this sandbox's scratch"* |

So no dependency is added, and the reason is recorded rather than assumed.

## Deviations from the spec, carried from the design

`docs/design.md` §16.1 reports seven; the three that change what a test asserts:

- **D1 — the workspace is `git clone --shared`, not a worktree.** A worktree's
  index lives in the main repository, so the only configuration in which an
  agent can commit lets it write `<main>/.git/hooks` — where the hook then
  **runs**. That is CVE-2026-26268, and criterion 11 exists to prevent it. §6.1's
  *purpose* is preserved exactly; its mechanism is not. `prepare` enforces
  `extensions.preciousObjects` on the main repository as a precondition, because
  without it an ordinary `git gc` there destroys the agent's history.
- **D5 — a validation's materials are a sibling of the producing task's zone.**
  The spec's layout is five things and none of them is a validation, and the only
  place it has room is inside the producing subtree, which is reachable.
- **D7 — criteria 9, 17 and 21 are not satisfiable as written.** 9 is decomposed
  into three unit tests over an injected `Availability` plus one end-to-end,
  because no machine runs all three branches. 17's *"nothing depends on its
  contents having survived"* is a property of all future code and is a review
  rule. **21 is now half-satisfied**: `meta.from_knowledge` reads cluster
  conventions out of a versioned handoff and `test_conventions_come_from_a_knowledge_handoff`
  proves changing the handoff changes behaviour with no code change — the
  consumption route is real; the system-level task that would *produce* one
  remains unspecified.

**And criterion 13 has since joined that set, for a reason D7 could not have
anticipated.** The design believed D5's sibling placement closed it, and for the
**zone** route it does. `interfaces.md` §5.19 then ruled a task package granted
read-execute — a body is a launcher, so the package must be readable either way
— and a package holds `validators/`. So a producing task can reach the standard
it is judged against through the **package**. **One route closed, one open.**

**§4.16 reversed that grant to staging, and criterion 13's number did not move
— re-derived rather than assumed.** The tempting reading of *stage, not grant*
is that the package route closed with it. It did not: staging relocates the
standard from `<package>/validators/` to `<zone>/package/validators/`, and the
zone is granted read-**write**, so the copy is *more* reachable than the grant
it replaced. §4.16 says so itself — *"until [`TODO.md` 4a] holds, staging moves
the leak rather than closing it"*. So the honest number is unchanged at **21½
of 22**, and the strict `xfail` moved with the mechanism rather than being
retired: `test_a_staged_package_still_carries_the_validators`.

**What did change is that the mechanism which will close it now exists.**
`stage_package` takes an **allow-list** of package-relative paths, and the list
shape is the point rather than an ergonomic choice: criterion 14 holds because a
sibling zone nobody anticipated is *absent from a list*, and a deny-list over
the package would give exactly the anticipated-only guarantee §4.5 rejects — a
validator directory added next month would not be on it. When `TODO.md` 4a makes
a task's executable set nameable, `Context.package_stage` carries it and
criterion 13 closes by the same construction that already closes 14. The
allow-list is populated by nobody today, which is why the number is 21½.

## Every acceptance criterion, and the test that holds it

`docs/spec.md` §10's 22 criteria. **Names checked against the tree**, not against
the design's plan — the plan named tests before they existed and three of them
ended up called something else.

**Four criteria are not fully closed** — 9, 13, 17 and 21 — and each says so in
its own row rather than in a footnote, because a mapping that reads 22/22 over a
half-covered property is the failure this suite spends its time finding in other
people's code. Two of the four are *half* rather than *absent*, and the rows say
which half.

| # | Criterion | Test |
|---|---|---|
| 1 | domain registered, reloaded idempotently, kind decides layout | `test_register_idempotent`, `test_reload_preserves_playground`, `test_kind_decides_layout`, `test_register_rejects_a_changed_root`, `test_get_names_the_candidates`, `test_storage_root_needs_exactly_one` |
| 2 | subtask nested in parent; reach is containment | `test_subtask_nested_under_parent`, `test_reach_is_containment`, `test_the_zone_has_the_four_directories`, `test_a_retry_gets_its_own_zone` |
| 3 | `startswith` is not the check — **two layers** | userspace: `test_sibling_prefix_denied`, `test_symlink_out_denied`, `test_dotdot_denied`, `test_inside_is_allowed`. kernel: `test_startswith_defeats_denied_by_the_kernel` |
| 4 | canonicalisation fails closed | `test_broken_symlink_denied`, `test_symlink_loop_denied`, `test_nonstrict_resolve_is_not_used` |
| 5 | NUL byte rejected | `test_nul_byte_rejected`, `test_valueerror_is_caught_too` |
| 6 | scripted bypass blocked, same script inside succeeds | `test_scripted_bypass_denied`, `test_same_script_inside_zone_succeeds` |
| 7 | confinement inherited | `test_bash_child_inherits`, `test_second_ruleset_cannot_widen` |
| 8 | no sandbox, no start | `test_no_mechanism_refuses_to_start`, `test_refusal_names_the_reason`, `test_nothing_in_the_package_catches_noconfinement`, `test_no_confinement_propagates_out_of_prepare` |
| 9 | chain degrades in order — **decomposed, D7** | `test_prefers_bwrap`, `test_falls_back_to_landlock`, `test_refuses_when_neither` over an injected `Availability`, plus `test_end_to_end_against_the_declared_mechanism`. **No machine runs all three branches**: `bwrap` is absent so rung 1 cannot run, and a Landlock-capable kernel cannot be made to look incapable |
| 10 | zone root never from agent input | `test_zone_root_not_from_agent_input`, `test_a_traversal_proposal_is_denied`, `test_a_proposal_inside_the_subtree_is_honoured`, `test_tool_takes_no_zone_argument`, `test_a_path_argument_cannot_leave_the_zone` |
| 11 | policy, `.git/hooks`, `.git/config`, shell rc not writable | `test_policy_not_writable_by_agent`, `test_main_git_hooks_denied`, `test_main_git_config_denied`, `test_shell_rc_denied`, `test_a_read_write_grant_would_permit_the_hook_write` (negative control), `test_the_agents_own_hooks_are_its_own` (the bounded residual) |
| 12 | read outside the granted set denied | `test_ungranted_read_denied`, `test_ungoverned_path_denied`, `test_the_grant_is_what_denies_it` (negative control), `test_granted_read_does_not_widen_beyond_dac` |
| 13 | producer cannot read a validation's standard — **one route closed, one open** | zone route: `test_validation_is_a_sibling_not_a_descendant`, `test_producer_cannot_read_validation`, `test_the_placement_is_what_denies_it` (negative control). **package route: `test_a_staged_package_still_carries_the_validators`, a strict `xfail`** — §4.16 stages a copy instead of granting the root, and the copy still carries `validators/` until `TODO.md` 4a names a task's executable set. `test_a_staged_package_is_reachable_and_the_original_is_not` holds what staging *did* buy |
| 14 | a sibling zone created later is unreachable, no rebuild | `test_sibling_zone_created_later_unreachable`, `test_no_rebuild_required` |
| 15 | sync once, at start, scoped to the task | `test_sync_once_at_start`, `test_destination_matches_source`, `test_scoped_to_task_not_root`, `test_direction_is_required` |
| 16 | playground not synced | `test_playground_not_synced`, `test_playground_dir_created_empty` |
| 17 | playground survives a resume — **half, D7** | `test_playground_survives_resume`. *"Nothing depends on its contents having survived"* is a property of all future code, not an observable of a run: a review rule, not a test |
| 18 | remote operations are tool calls with schemas | `test_remote_tools_have_schemas`, `test_tool_call_round_trip`, `test_push_and_pull_round_trip` |
| 19 | agent works on a copy; the stored artefact is unchanged | `test_agent_works_on_a_copy`, `test_stored_artefact_byte_identical`, `test_copy_out_refuses_to_copy_onto_itself` |
| 20 | shared object store, main checkout unmodified — **D1**, not "is a worktree" | `test_workspace_shares_object_store`, `test_main_checkout_unmodified`, `test_the_agent_can_commit`, `test_collect_returns_work_by_a_supervisor_side_fetch`, `test_cut_refuses_a_main_repository_without_precious_objects`, `test_precious_objects_blocks_the_prune` |
| 21 | conventions from a knowledge handoff, no code change | `test_conventions_come_from_a_knowledge_handoff`, `test_a_missing_knowledge_handoff_is_the_empty_default`. **The consumption half only** — the system-level task that would *produce* one is unspecified, so the test builds the artefact. The design recorded this as untestable; it is half-testable |
| 22 | the shipped machinery **keeps working** | `test_cli_subcommands_preserve_shipped_shapes`, plus the machinery's own tests. **Revised 2026-09-04** (`fc200a2`): the criterion read *untouched*, and a test — test_the_shipped_modules_are_byte_identical, named here **without backticks on purpose**, because `test_every_test_the_readme_cites_exists` scans backticked `test_*` names and cannot tell *citing a test as cover* from *naming one that was removed* — asserted that literally, over `git diff HEAD`. That was a scope fence for the round that built the new subsystems, and this round is a design-level change to the machinery itself, so the fence is retired. The **65** is a 2026-08-30 snapshot, not a live count. See `docs/spec.md` §10 criterion 22 for the full reason |

**Beyond the criteria**, three suites hold properties nothing else would catch:
`test_imports.py` (the decoupling wall, both directions, plus `fs/path.py`
importing only `os` and `pathlib`), `test_threads.py` (§4.4(c), the ABI-3 thread
scope), and `test_task_graph_agreement.py` (the seam driven with the **real**
`task_graph` types, so the stub in `tests/env_mgr/stubs.py` cannot rot).

### A note on how these tests are built, because it is not optional here

**Every denial is asserted as `EACCES` against a named path**, never as a
non-zero exit status. A design measurement produced a false PASS from
`returncode != 0` because both children were failing to exec the interpreter
rather than being denied.

**Every denial carries a positive control** — the same operation succeeding
inside the zone, in the same confined process — which rules out *"the binary
could not run"*.

**And the load-bearing ones carry a negative control**: the arrangement under
which the denial would *not* happen. Without it, *"it is denied"* and *"it would
be denied whatever we did"* are the same green — which is how criterion 13's
separation was held by an accident of location for a week before anyone noticed.

## Running the isolation tests

```bash
pytest agent_sys/tests/env_mgr
```

**When no sandbox mechanism is available the suite fails. It does not skip.**
CI declares the mechanism rather than discovering it:

```bash
ENV_MGR_TEST_MECHANISM=landlock ENV_MGR_TEST_ABI=3 pytest agent_sys/tests/env_mgr
```

Absent those variables a developer's machine auto-detects and runs; the hard
failure is CI-side, which is how `rust-landlock` arranges the same thing. Skips
are permitted for environmental variation orthogonal to the property under test
— `rsync` absent, `bwrap` absent — and **never for the property**.

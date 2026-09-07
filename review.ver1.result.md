# PR #132 review — ver1

- PR: https://github.com/AMD-AGI/Infera/pull/132
- Branch: `dev.yihou.aiopt.all.1`
- Range reviewed: `HEAD~5..HEAD` (2c42d8c..bd5281e series), ~1500 lines across
  `agent_sys/env_mgr`, `agent_sys/cli`, `agent_sys/agent`
- Tool: `/code-review 132 max`, findings re-read against the source afterwards.
  Items marked ✅ were verified first-hand by reading the file; the rest are
  reported by the review pass and not independently re-derived.

## Critical — silent data loss / permission escape

### 1. `agent_sys/env_mgr/remote/tools.py:101` — `env_remote_push` / `env_remote_pull` never constrain `remote` ✅

The same commit introduced `_inside_remote` and applied it only to
`env_remote_run`'s `cwd`. `remote` is a bare string in both schemas (no
`description`, no `pattern`) and goes straight to `conn.push` / `conn.pull`.

Failure: `env_remote_push(path='payload', remote='/etc/cron.d/')` writes outside
the far-side zone; `env_remote_pull(remote='/home/other/.ssh/id_rsa', path='k')`
exfiltrates an arbitrary far-side file into the zone. Criterion 10's "the zone
root is never taken from agent-supplied input" holds for the local argument and
is silently false for the remote one.

### 2. `agent_sys/env_mgr/sync.py:215` — conflict pre-pass silently skipped for `REMOTE_TO_LOCAL` across a host boundary ✅

`remote_dst=far_is_dst and bool(prefix)`.

Failure: with `direction=REMOTE_TO_LOCAL` and an `Ssh` transport, `src` is the
far-side path and `remote_dst` is `False`, so `conflicts(src, dst)` runs
`os.path.isdir(src)` on the **local** machine. Either it returns `()` (path
absent locally → silent pass) or a directory with that same absolute path exists
locally and the wrong tree is compared. `rsync -a --delete` then overwrites the
local zone and the one guard against both-sides-changed data loss never ran.
Contradicts `_conflicts_across`'s own "never a silent skip". No production caller
uses this direction yet, so it ships untested.

### 3. `agent_sys/env_mgr/meta.py:106` — `mapping_roots()` rewrite changes which `remote_root` wins ✅

New body is `far_roots()` (all mappings, later entry wins) filtered by the weak
key set, instead of the last **weak** mapping's value.

Failure: `(RemoteMapping('/work', '/data/weak', WEAK), RemoteMapping('/work',
'/mnt/shared', STRONG))` used to yield `{'/work': '/data/weak'}` and now yields
`{'/work': '/mnt/shared'}` — the shared mount declared strong precisely because
nothing should be copied to it becomes the `rsync -a --delete` target, and
`check_delete_scope` validates the wrong root too.

### 4. `agent_sys/env_mgr/remote/connection.py:213` — `sync_transport` tests `not target` before `transport` ✅

`{transport: 'shh', target: ''}` (typo) or `{transport: 'docker', target: ''}`
falls through to `LocalConnection`. `sync` then runs `rsync -a --delete` against
`remote_root` as a *local* path — creating and destroying a directory on this
machine — with no diagnostic, despite the docstring's "An unknown transport
raises here, at composition, rather than at the first copy."

## High — functionally broken

### 5. `agent_sys/cli/main.py:515` — a legal docker mapping makes the CLI unstartable ✅

`sync_transport` is called for *every* mapping, not only sync-relevant ones, and
raises `ValueError` for `docker` — a value `RemoteMapping.target`'s own comment
("host for ssh, container for docker exec") and `tools.py`'s docstring ("a
container is a valid tool target and an invalid sync transport") both bless.
`main()` catches `PackageNotFound`, `SpecInvalid`, `CredentialsMissing` /
`NoConfinement` / `RepositoryNotPrepared`, `PrepareRefused` /
`UnresolvedGrant` — not `ValueError`. Result: raw traceback and a
non-`PRECONDITION` exit code. The reachability question has been collapsed onto
the sync question.

### 6. `agent_sys/env_mgr/prepare.py:496` — `zone_env(remote_zone_root=...)` still resolves against weak-only `ctx.mapping` ✅

`_remote_tools`, forty lines below, switched to `ctx.far_roots` for exactly the
reason that resolving against `ctx.mapping` is wrong.

Failure: strong mapping (one mount, two machines — the R1b configuration
`_remote_tools`'s docstring names). `ctx.mapping` is empty, so `remote_root`
returns `None` and every `AGENT_SYS_*_REMOTE` variable is omitted, while
`_remote_tools` successfully hands the agent `env_remote_run` / `push` / `pull`.
The agent has remote tools and no environment variable telling it where the far
side is.

### 7. `agent_sys/env_mgr/sync.py:278` — cross-host `sync` is once-only per zone

`_conflicts_across` refuses whenever `test -e dst` succeeds, and `sync` itself
creates `dst` with `mkdir -p` on the first run. Any second `prepare` for the same
zone (resume, retry after a transient failure, re-run of the same attempt) hits
`test -e` → rc=0 → unconditional `PrepareRefused` telling the operator to "Remove
the far-side path, or sync a fresh zone". The state that triggers the refusal was
created by the previous successful run of this same function.

### 8. `agent_sys/env_mgr/sync.py:228` — `-e " ".join(rsh)` re-introduces the argv-boundary bug bd5281e just fixed ✅

`Ssh('gpu01', options=('-o', 'ProxyCommand=ssh -W %h:%p bastion'))` →
`-e 'ssh -o ProxyCommand=ssh -W %h:%p bastion'`; rsync word-splits it, so `-o`
gets `ProxyCommand=ssh` and `-W`, `%h:%p`, `bastion` become stray rsh arguments.
Same class for any option containing a space. Should be `shlex.join(rsh)`.
(`connection.py:_rsync` has the same pre-existing line.)

## Medium

### 9. `agent_sys/agent/backends/claude_sdk.py:125` — in-place mutation of the caller's config

`options = dict(self.config.get("options") or {})` (line 283) is a *shallow*
copy, so `options.setdefault("mcp_servers", {})[_TOOL_SERVER] = server` writes
into the config's own nested dict. Attempt 1 (with a far side) installs a server
closed over attempt 1's `Zone` and `Ssh`; attempt 2 shares the config, has
`assignment.tools == ()`, skips the block — and still starts with attempt 1's
stale zone-bound server present, plus a live reference keeping attempt 1's
connection alive. A user-configured MCP server named `env_mgr` is overwritten
with no warning.

### 10. `agent_sys/agent/backends/claude_sdk.py:102` — remote tool calls have no timeout

`defn.call` runs under `asyncio.to_thread` wrapping a non-cancellable
`subprocess.run(timeout=None)`, so a wedged ssh session pins a worker thread for
the process lifetime and the SDK's own turn timeouts cannot reclaim it.
`Ssh.run` / `DockerExec.run` / `LocalConnection.run` all accept `timeout`;
`tools.py:97` passes none.

### 11. `agent_sys/env_mgr/protocols.pyi:114` — stub and implementation disagree on a default

Stub declares `Prepared.permissions_enforced: bool = True`; `protocols.py` and
`prepare.py` both declare `False`. This diff edited the stub (adding `tools`) and
left the divergence. The field's own docstring says the two declarations "must
agree, or the two halves of the seam disagree about what an omitted field
means". A type checker concludes an omitted value means enforcement is on; at
runtime it is off. `tests/interfaces/` evidently does not compare defaults.

## Low — consistency / stale documentation

### 12. `agent_sys/env_mgr/sync.py:219` — asymmetric, partly dead `playground_dst` creation

The remote branch creates `playground_dst` before rsync *and* again after; the
local branch creates it only after. `--delete` without `--delete-excluded` does
not remove receiver-side content matched by `--exclude=playground/**`, so the
post-copy `mkdir -p` already suffices and its comment ("Repeated after the copy
because `--delete` removes it") is not what rsync does. The pre-copy call costs
an extra remote round trip per sync and leaves two branches that create the same
directory on different schedules.

### 13. `agent_sys/cli/main.py:481` — retained comment now states the opposite of the code ✅

Lines ~481-487 still read "`mapping_roots()` and not the `RemoteMapping`s: it …
drops `transport`/`target`, which nothing anywhere reads yet … deliberately not
enough to reach another host — that needs a `Connection` on `Context`, and it is
not here." Six lines later the new comment says "`transports` is the reader
`transport` and `target` never had", and a `Connection` is put on `Context`.

## Suggested order

1 → 2 → 3 / 4 → 5 / 6. #1 is the only finding with security semantics; #2 and #3
are silent data destruction; #5 makes a legal configuration unstartable.

---

# Whole-package scan of `agent_sys` (beyond the PR diff)

Scope: all non-test, non-example code under `agent_sys/` (~26k lines). Not
limited to this PR's diff.

Method:

1. Full pyright run over `agent_sys` (LSP backend; 335 files, 570 errors,
   7 warnings), triaged by rule to the classes that can be real defects.
2. Mechanical comparison of every `.pyi` / `.py` pair: class field annotations
   **and** default values, via `ast`.
3. Pattern scan for `shell=True`, bare `except`, `except: pass`, mutable default
   arguments, `is` against a literal, `eval`/`exec`/`os.system`/`pickle`/
   `yaml.load`/`mktemp`, `os.environ[...] =`, `assert` in production paths,
   locks, atomic-write and file-locking primitives.
4. Serena symbol lookup (`find_symbol`, referencing symbols) plus source reads
   to confirm or reject each candidate.

Every item below was confirmed by reading the source.

## High

### A1. `agent_sys/agent/backends/claude_sdk.py:451` + `:567` — `interrupt` is defeated by its own lock

`_await` is `with self._loop_lock: self._loop.run_until_complete(...)`. The
attempt thread holds that lock for the whole model turn at `:423`
(`self._await(self._final_message())`). `interrupt()` at `:451` begins with
`self._await(self._client.interrupt())` from the monitor's thread, so it blocks
on the lock until the turn it is meant to cut short has finished on its own.

The interrupt is therefore inert in exactly the situation it exists for — a turn
that is running away. The docstring's "the lock is what keeps the two off the
loop at once" is accurate about the mechanism and silent about the consequence.
The shape that works is a loop owned by a dedicated thread plus
`asyncio.run_coroutine_threadsafe`.

### A2. `agent_sys/agent/backends/claude_sdk.py:222` — the event loop is never closed; one leak per attempt

`__init__` calls `asyncio.new_event_loop()`. The whole file references `_loop`
at only 222, 223, 567 and 568 — there is no `close()`, and `ClaudeSdkBackend`
does not override `stop()` (the base `backend.py:468` only drains the inbox and
settles). A backend is constructed fresh on every deploy
(`selection.py:188` → `runner.py:694`, once per task attempt), so each attempt
leaks a loop with its epoll descriptor and self-pipe pair. A long multi-task run
walks toward `EMFILE`.

### A3. Finding #9 above is process-wide, not per-attempt

`selection.py:188` passes `decl.config` — the object owned by the agent-spec
declaration and shared by every attempt of every task using that spec — straight
into the backend, and `claude_sdk.py` copies it only shallowly (`self.config =
dict(config or {})`, and `options = dict(...)`). So
`options.setdefault("mcp_servers", {})[_TOOL_SERVER] = server` writes into the
declaration's own nested dict. The consequence is not merely that attempt 2 sees
attempt 1's residue: **every later task using that agent spec starts with the
MCP server bound to the first task's zone**, and the live reference keeps that
attempt's `Ssh` connection alive.

## Medium

### A4. No file locking anywhere; `task_graph` store `update` is check-then-write — open question

`fcntl` / `flock` have zero hits in the package. `task_graph/store.py:_write`
does tmp → `Path.replace` (per-record atomic, no `fsync`), and `update()` checks
`path.exists()` and then overwrites wholesale, so two processes doing
read-modify-write on one key silently last-writer-wins with no detection.
`handoff/store.py` is the counter-example and gets it right: `os.mkdir(stage)`
as an atomic allocator and `os.rename` to seal.

**Deliberately left open.** If `agent_sys` is single-process by design, this is
only a missing `fsync`. If multi-process or shared-filesystem operation is
intended, it is a real gap. That is a question about intent, not about the code.

### A5. `agent_sys/validator/phase.py:906` — wrong return annotation on a documented seam

`_placed_root` is declared `-> tuple[Path, tuple[str, ...]]`, but both returns
(`:936`, `:937`) produce `dict(placed.materials)` / `{}`, and the consumer
`_declare_materials` (`:239`) is typed `Mapping[Any, str]`. Runtime is correct
and the annotation is not — and this map is precisely the "a multi-input
validator finds which copy is which by handoff id" contract, so a reader going
by the signature sees a sequence of paths instead.

## Low

### A6. `agent_sys/validator/phase.py:662` — `assert` on a production path

`assert prior is not None` is stripped under `python -O`, after which `prior[0]`
raises `TypeError`. If `history.may_skip` already guarantees non-`None`, the
assertion is dead either way; either make it an explicit branch or drop it.

### A7. `agent_sys/monitor/buffer.py:88` — `get()` waits once instead of on a predicate

`if not self._order and not self._closed: self._cond.wait(timeout)` is a single
wait, not a loop. A spurious wakeup, or a `notify` taken by another consumer,
makes `get` return `None` well before `timeout` elapsed, which the caller reads
as "nothing to do for that long". The practical cost is an early spin; the
`timeout` contract is still broken.

### A8. The `.pyi` default divergence is a single site, not a class of problem

Comparing annotations and defaults across every `.pyi` / `.py` pair, the only
genuine disagreement in the package is finding #11's
`Prepared.permissions_enforced` (stub `True`, implementation `False`). The other
three diffs (`Context.far_roots`, `Context.transports`, `Prepared.tools`) are
stub `...` placeholders and are correct.

Useful as a negative result: fixing one line closes it, and one default-value
assertion in `tests/interfaces/test_stub_agreement.py` keeps it closed.

## Checked and cleared

- **`env_mgr/installers/base.py:31` `shell=True`.** `cmd` comes from
  `item.spec['run']`, whose only source is `load_recipe(args.recipe)` at
  `env_mgr/cli.py:74` — a recipe the operator passes explicitly. It does not
  travel through a spec package and an agent cannot reach it. Not an injection
  surface.
- **`handoff/containment.py` and `env_mgr/fs/path.py`.** Containment is careful:
  `..` refused by policy before resolving, `z + os.sep` for the trailing
  separator defeat, NUL and wildcard rejection, fail-closed canonicalisation, and
  `resolve_for_check` deliberately not falling back for a broken symlink or a
  symlink loop. No bypass found.
- No bare `except`, no `except: pass`, no mutable default arguments, no `eval` /
  `exec` / `os.system` / `pickle` / `yaml.load` / `mktemp`, no `os.environ`
  mutation.

## On the pyright baseline

570 errors, of which 284 are `reportAttributeAccessIssue` and 202
`reportArgumentType`. Almost all come from `.pyi` protocols and `.py`
implementations being two *nominally distinct* type families —
`cli/main.py:565` is the representative sample, where
`monitor.protocols.EventRecord` and `monitor.record.EventRecord` are judged
incompatible. Harmless at runtime, but it means type checking currently emits
almost no usable signal here: this sweep salvaged 15 diagnostics worth reading
out of 570. Making the stubs declare only `Protocol`s, rather than re-declaring
the concrete classes, is what would turn pyright back into a guardrail.

### A9. `env_mgr/remote/connection.py:93` — `SyncReport.sent` counts lines, not files

Found while looking for duplicated implementations (see D3 below). `_rsync`
computes:

```python
sent = sum(1 for line in proc.stdout.splitlines()
           if line.startswith("Number of regular files transferred"))
```

That counts *matching lines*, so it is always 0 or 1 — never the file count.
`sync.py:290`'s `_stat` parses the integer after the colon and is the correct
version of the same parse. `Connection.push` / `Connection.pull` are the only
users, and their `SyncReport` is what `env_remote_push` / `env_remote_pull`
hand back to the model via `report._asdict()`, so the agent is told "1 file
transferred" for any non-empty copy.

---

# Compressibility analysis of `agent_sys`

Measured with `ast` + `tokenize`, not estimated. Docstrings are taken as the
first string expression of a module/class/function (internal blank lines
included); `#` comments come from `tokenize` with those falling inside a
docstring removed; blank lines exclude docstring-internal blanks, so nothing is
double-counted.

## D. Duplicate implementations and reuse

### D1. `protocols.py` re-declares what the implementation modules already declare, and `.pyi` is a third copy

| package | protocols.py | protocols.pyi | top-level names | also defined elsewhere | duplicated lines |
|---|---|---|---|---|---|
| agent | 361 | 99 | 13 | **13** | 283 |
| closure | 188 | 48 | 8 | 8 | 115 |
| env_mgr | 551 | 132 | 18 | 7 | 286 |
| handoff | 350 | 119 | 17 | 3 | 31 |
| monitor | 347 | 109 | 11 | 2 | 57 |
| spec_loader | 409 | 101 | 18 | 6 | 67 |
| validator | 258 | 98 | 13 | 2 | 53 |
| **total** | **2464** | **706** | 98 | **41** | **892** |

41 names carry three declarations (implementation + `protocols.py` +
`protocols.pyi`). `agent/` is the extreme case: all 13 names are duplicated,
including a verbatim copy of the `AgentStatus` enum (`backend.py:100` vs
`protocols.py:74`, identical members).

**Nine of them are two nominally distinct classes**, not a `Protocol`
structurally describing an implementation: `BackendUnsupported`,
`BackendUnavailable`, `AgentStatus`, `Kind`, `Zone`, `Policy`, `Prepared`,
`ValidationZone`, `PhaseOutcome`. Verified at runtime:

```
BackendUnsupported same class? False
raise agent.backend.BackendUnsupported → except agent.protocols.BackendUnsupported: NOT caught
```

`AgentStatus` is safe only by accident — `str, Enum` makes cross-class `==` and
`hash` agree.

**Not a live defect in-tree.** Every import of those nine names, in source and
in tests, comes from the implementation module; nothing imports them from
`protocols.py`. The hazard is for an outside consumer following the package's
own advertised surface — `agent/protocols.py`'s docstring says "What leaves
`agent/`" and its `__all__` lists all thirteen.

**Suggested fix.** Keep only real `Protocol`s in `protocols.py`; re-export
concrete classes, enums and exceptions (`from .backend import AgentStatus,
BackendUnsupported`). The same change removes the root cause of most of the 570
pyright errors (284 `reportAttributeAccessIssue` + 202 `reportArgumentType` are
almost entirely this), and retires part of the conformance suite that exists to
hold the three copies in agreement (`tests/interfaces/`, 2064 lines).

### D2. "Is this path under that root" has six independent implementations

| Site | Technique |
|---|---|
| `env_mgr/fs/path.py:83 contained` | `resolve(strict=True)` + `z + os.sep` — the canonical one |
| `env_mgr/fs/path.py:130 contained_syntactically` | `normpath`, no filesystem (far side) |
| `env_mgr/sync.py:58 _under` | `normpath` + `startswith(root + os.sep)` |
| `env_mgr/sync.py:116` | inline `zone.root.startswith(prefix + os.sep)` |
| `handoff/containment.py:54` | `resolve()` + `is_relative_to` |
| `validator/boundary.py:135 _inside` | `realpath` + `startswith(base + os.sep)` |
| `validator/separation.py:62` | `realpath`, **deliberately the opposite fail direction** |

The last two diverge for documented reasons (fail-closed vs fail-open) and
should not be merged. The two in `sync.py` are unexplained weaker rewrites —
`normpath` only, no `resolve`, so a symlink walks straight through — and should
call `fs/path.contained`.

Related naming collision: `_inside` is `validator/boundary.py:127` (returns
`bool`) and `env_mgr/remote/tools.py:32` (returns a path).

### D3. rsync command construction is written twice, and the copy carries a bug

`env_mgr/remote/connection.py:83 _rsync` and `env_mgr/sync.py:222-235` both do
`shutil.which`, `-a --stats`, `-e " ".join(rsh)`, trailing-separator
normalisation, `check=True`, and a parse of `Number of regular files
transferred` — with the two parses disagreeing, the copy being wrong (finding
A9 above). Extracting one `_run_rsync(...) -> SyncReport` fixes A9 and finding
#8's `shlex.join` at the same time.

### D4. Mechanical duplication is otherwise negligible

Structurally identical function bodies (≥3 statements, identifiers abstracted,
AST-hashed): **zero** matches between source files; the single hit pairs
`cli/main.py:745` with a conftest fixture. Identical 5-line code windows: 30,
all of them the `protocols.py` transcription in D1.

**Conclusion: this codebase has no ordinary copy-paste problem. All compressible
duplication traces to one decision — declaring the same thing three times.**

## E. Line statistics

| group | files | total | code | docstring | `#` | blank |
|---|---|---|---|---|---|---|
| source (non-test, non-example `.py`) | 131 | 28,170 | 11,546 | 10,334 | 3,145 | 3,145 |
| stubs (`.pyi`) | 7 | 706 | 550 | 0 | 42 | 114 |
| tests | 173 | 41,394 | 21,303 | 10,366 | 1,907 | 7,818 |
| examples | 24 | 4,704 | 2,610 | 1,096 | 421 | 577 |
| **total** | **335** | **74,974** | **36,009** | **21,796** | **5,515** | **11,654** |

- Source prose (docstring + `#`) is **13,479 lines = 47.8% of the file bytes,
  1.17× the code**.
- tests 41,394 : source 28,170 ≈ **1.47 : 1**.
- Repository-wide: 48% code, 36% prose, 16% blank.

## F. Comment blocks longer than 3 lines

A block is one docstring, or one run of consecutive `#` lines (broken by any
line of code).

### Source (non-test)

| kind | blocks | lines | blocks >3 | lines in them | saved at 3 |
|---|---|---|---|---|---|
| docstring | 960 | 10,334 | **731** | 10,022 | **7,829** |
| `#` runs | 740 | 3,145 | **239** | 2,292 | **1,575** |
| **total** | 1,700 | 13,479 | **970** | 12,314 | **9,404** |

**76% of source docstrings are longer than 3 lines**, and blocks over 3 lines
occupy 12,314 lines — 43.7% of the whole source tree.

Size distribution of the >3 blocks:

| lines | docstring | `#` runs |
|---|---|---|
| 4–6 | 172 | 115 |
| 7–12 | 272 | 66 |
| 13–25 | 211 | 51 |
| 26–50 | 65 | 6 |
| 51+ | **11** | 1 |

### Savings at different caps

| cap | source saved | % of source | tests saved | total saved | % of repo |
|---|---|---|---|---|---|
| **3** | **9,404** | **33.4%** | 6,347 | **15,751** | **21.0%** |
| 4 | 8,434 | 29.9% | 5,287 | 13,721 | 18.3% |
| 5 | 7,524 | 26.7% | 4,360 | 11,884 | 15.9% |
| 6 | 6,705 | 23.8% | 3,538 | 10,243 | 13.7% |
| 8 | 5,434 | 19.3% | 2,383 | 7,817 | 10.4% |
| 10 | 4,418 | 15.7% | 1,699 | 6,117 | 8.2% |

**Answer to the question as asked: compressing every block to 3 lines saves
9,404 lines in source (33.4%), and 15,751 lines repository-wide (21.0%).**

### One reservation, stated rather than assumed

The five requested elements (what / why / how / context / plain-language
explanation) do not fit in three lines here, because in this codebase the *why*
usually carries measured evidence and rejected alternatives —
`claude_sdk.py:_options`'s 20-line docstring records an unresolved defect about
which CLI binary counts, and compressing it to three lines deletes it. A layered
reading:

- **The 287 blocks of 4–6 lines** (30% of all >3 blocks) are mostly padding;
  compressing to 3 is near-lossless and saves ~970 lines.
- **The 83 blocks of 26+ lines** are design records (prior-art comparisons,
  measurements, rejected options). Those belong in an ADR or design document —
  *moving* them out saves ~3,500 lines and loses nothing.
- **The ~600 blocks of 7–25 lines** are the ones that need per-block judgement.

Under the more conservative combination — compress 4–6, relocate 26+, cap 7–25
at six lines — the realistic source-side saving is roughly 5,000–6,000 lines
(18%–21%) with no loss of design rationale.

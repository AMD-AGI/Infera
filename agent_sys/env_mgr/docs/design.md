# Environment Manager — Design

| | |
|---|---|
| Status | Draft — stage two of spec → design → test & code |
| Revision | 4 — 2026-08-27. **`prepare` gains `agent_spec`** (§11.1, §11.5). The spec-key-to-runtime trace found four agent-spec keys — `env`, `rules`, `hooks`, `skills` — whose declared consumer is this module and whose signature could not receive them. (rev. 3: 2026-08-27. **Dependency repositories are per task** (§7.1.1), following `closure` spec §2.5 rev. 9; the main repository stays global. (rev. 2: 2026-08-27. **The stage-three consistency pass.** `Access` is renamed `Mode`, because `task_graph` owns that name and `prepare()` mixed both in one `Policy` (§5.3.1). A grant that resolves to nothing now **raises** instead of returning an empty granted set (§6.1). `Context` is defined rather than used undefined, and gains the two attributes §6 needs (§11.1). Step 6's `handoffs.stage` becomes `layout.stage_handoffs`, a module that exists (§11.1). The environment manager's registration name is fixed (§11.4). (rev. 1: 2026-08-27. Initial))) |
| Implements | [`spec.md`](spec.md) rev. 3, acceptance criteria 1–22 |
| Language | Python ≥ 3.10. `ctypes` for the Landlock syscalls; no third-party sandbox dependency |

---

## 1. Scope

This document turns [`spec.md`](spec.md) into files, classes, and interfaces. It
adds no requirements. Where it makes a choice the spec left open, the choice is
stated here; where implementing the spec exposed a contradiction, §16 says so
rather than papering over it.

The spec's 22 acceptance criteria are the definition of done. §14 maps every one
to a named test, and says plainly which three cannot be satisfied as written.

**This module is unusual in the opposite way to `closure`.** Its spec is the one
written "against measured behaviour, not intuition" (spec §4), so the research
could not be a survey of opinions — it had to **re-measure the mechanisms the
spec turns on**. Twenty-three probes later, the finding is not that the spec is
wrong about the mechanisms. It is right about all of them. The finding is that
**several of its own sections cannot hold at the same time**, and the largest of
those re-creates the vulnerability another of its sections exists to prevent
(§7.1).

Evidence is in `scratch/design/findings-envmgr-mine.md` (M1–M23),
`findings-envmgr-selftests.md` (S1–S5), `findings-envmgr-sbxtest.md` (F1–F16),
with probes in `scratch/design/probes-envmgr/`. Where this document says
"measured", there is a script.

**This document specifies interfaces, not bodies.** A method is a signature and a
sentence. A body appears only where the ordering of steps *is* the design
decision — §4.5's construction sequence, §6.2's resolution, and §11.1's
preparation order, and nowhere else.

### 1.1 What this module owns

- **The path as the single fact** — canonical containment, domains, and zones
  (§3), which is the one thing every other piece here shares.
- **Isolation**: the mechanism chain, its selection, the Landlock binding, and
  what "fail closed" means at each of two tiers (§4).
- **The granted set** (§5), including the part of it the spec's default cannot
  supply, and the cost of restricting reads at all.
- **Resolving a permission grant** into real locations (§6) — the step
  [`../../closure/docs/design.md`](../../closure/docs/design.md) D2 handed over.
- **The agent's workspace** (§7), where the spec's own two requirements collide.
- **The storage layout** (§8), including the place a validation's materials go,
  which the spec's layout has no room for.
- **Sync** (§9) and **remote access as tool calls** (§10).
- **Preparing an environment** before an executor runs (§11).
- The shipped recipe and installer machinery, unchanged (§12).

### 1.2 What it does not

| Deferred to | What |
|---|---|
| `handoff` design §4, §6.2 | **The store, its layout, and digests.** This module grants access to `<root>/<hid>/v<N>/` and computes nothing about its contents |
| `closure` design §6.3 | **The covering relation.** `covers()` decides what a grant grants; §6.3 here is about not disagreeing with it |
| `task_graph` design §3.5 | **Owning `Permissions`.** It is carried there and interpreted here, and that split is the whole reason neither package imports the other |
| `validator` design §8 | **What a validation checks and how.** This module only guarantees the producer cannot reach it (§8.3) |
| `agent` design §8.7 | **Which `claude` CLI the backend runs**, except that §11.2 reports the prepared interpreter so the decision has an input |
| The program under test | **sglang, vllm, infera environments.** Spec §1.3 and §7. A versioned handoff records its own environment |
| [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) | **Sync conflict resolution**, beyond §9.3's detection |

---

## 2. Layout and the import graph

```
env_mgr/
├── cli.py                  EXTENDED with sub-commands (§12.2). The only shipped file that changes
├── recipe.py  layer.py  runner.py  outcome.py  report.py  registry.py  versions.py
├── installers/             ── all unchanged (§12.1)
│
├── meta.py                 configuration: domains, mappings, sync strength (spec §3.1)
├── fs/
│   ├── path.py             canonical containment, failing closed. §3.2
│   ├── domain.py           Domain: name, root, kind; idempotent registration. §3.3
│   ├── zone.py             Zone: one task attempt's region. §3.4
│   └── layout.py           the nested layout, and where a validation goes. §8
├── isolation/
│   ├── probe.py            which mechanism is available. §4.2
│   ├── policy.py           granted set → a mechanism-independent policy. §5.3
│   ├── landlock.py         the ctypes binding. §4.4
│   ├── bwrap.py            the bubblewrap argument builder. §4.3
│   └── apply.py            the chain, and the two-tier gate. §4.5
├── grants.py               resolving a Grant into locations. §6
├── workspace.py            the clone with alternates. §7
├── material.py             deploying an agent's rules/hooks/skills. §11.5
├── sync.py                 the one-shot job. §9
├── remote/
│   ├── connection.py       ssh and docker exec behind one Protocol. §10.2
│   └── tools.py            the tool-call surface. §10.3
└── prepare.py              the composition of all of it. §11
```

### 2.1 The decoupling wall

Spec §9: "**Decoupling is structural**: nothing new imports the installer
machinery, and nothing in the installer machinery learns about domains or zones."

That is enforced, not intended:

```
      meta ◄── fs.domain ◄── fs.layout ◄── fs.zone
                   ▲              ▲            ▲
       fs.path ────┘              │            │
                                  │            │
   isolation.policy ◄── isolation.apply ◄──────┤
       ▲       ▲              ▲                │
  grants ──────┘        isolation.probe        │
       ▲                isolation.landlock     │
  workspace, sync, remote ─────────────────────┤
                                               │
                              prepare ─────────┘
                                 ▲
                               cli.py
        ─────────────────── the wall ───────────────────
   recipe, layer, runner, outcome, report, registry, installers/
```

`cli.py` is the only module above the wall that may import from below it, and it
does so exactly as it does today. **A test asserts the wall in both directions**
(§14.4), because "structural" is a claim about the import graph and an import
graph is checkable.

`fs/path.py` imports only `os` and `pathlib`. It is the bottom of the graph and
the only module three others depend on, which is what "the path is the fact"
means in an import graph.

---

## 3. The path is the fact

Spec principle 1. Permission, storage, and mapping are all expressed against
paths, in one component. This section is that one component.

### 3.1 One check, three callers, and it is not the enforcement mechanism

The most important thing to say first, because it changes what the rest of §3 is
*for*.

**Measured (M2): the kernel already denies all three documented `startswith`
defeats, with no userspace path check involved.** A sibling `zone-EVIL`, a
symlink inside the zone pointing out, and `zone/../outside` are each EACCES under
Landlock, because the kernel evaluates the resolved path at open time.

So `fs/path.py` is **not** what stops an attack. It has three real callers:

| Caller | Why it needs the check |
|---|---|
| §5.3, policy construction | To decide which paths are handed to the kernel. A wrong answer here grants too much *before* the kernel sees anything |
| §6.3, grant resolution | To refuse a grant whose literal form and canonical form disagree |
| The `PreToolUse` hook | A first gate and a diagnostic — spec principle 2. It attributes a denial to a tool call, which the kernel cannot do |

Stating this matters for §14: criterion 3 ("`startswith` is not the check") can be
satisfied by unit-testing a comparison function, which would prove nothing about
confinement. §14.2 tests it at both layers and says which is which.

### 3.2 `contained()` — four rules, and one the spec's wording misses

```python
def contained(path: str | os.PathLike, zone: str | os.PathLike) -> bool:
    """True iff `path` is `zone` or lies beneath it, on resolved paths.

    Fails closed: any failure to resolve either side returns False.
    """
```

Spec §4.3's four rules, each confirmed first-hand (M8):

1. **Resolve first**, both sides.
2. **The trailing separator is load-bearing**: `p == z or p.startswith(z + os.sep)`.
   Measured: `zone-EVIL/x` passes a bare `startswith` and fails this.
3. **Canonicalisation fails closed.** `Path.resolve(strict=True)`, and treat any
   exception as deny. Measured: `os.path.realpath` returns a partly-resolved
   path for both a broken symlink and a symlink loop **without raising** — and so
   does `Path.resolve()` with its default `strict=False`, which is the trap,
   because it is what an implementation reaches for. For a path that does not
   exist yet, resolve its **parent**.
4. **Reject NUL bytes.**

**The clause the spec's wording misses.** Rule 3 says "treat **any** exception as
deny", and rule 4's NUL raises `ValueError` — not `OSError`. A handler written
the natural way, `except OSError: return False`, lets a NUL through to whatever
runs next. So:

```python
except (OSError, ValueError):      # ValueError is rule 4, and it is not an OSError
    return False
```

One line, and it is the difference between rules 3 and 4 composing and rule 4
being dead code.

**Canonicalise per check, at use time**, never once when the policy is built —
spec §4.3, and resolving attacker-mutable components early is itself a TOCTOU
bug.

### 3.3 `Domain` — registration, idempotent

```python
class DomainKind(str, Enum):
    HANDOFF_STORAGE = "handoff_storage"
    PLAYGROUND      = "playground"
    WORKSPACE       = "workspace"

class Domain(NamedTuple):
    name: str
    root: str                    # absolute, resolved, no trailing separator (§6.3)
    kind: DomainKind

class DomainRegistry:
    def register(self, name: str, root: str, kind: DomainKind) -> Domain:
        """Idempotent. Re-registering an existing name with the same root and
        kind returns the existing Domain and touches nothing on disk -- which is
        what lets a playground survive a restart (spec §6.2). A different root
        or kind for a live name is an error, not an update."""
    def get(self, name: str) -> Domain: ...
    def __iter__(self) -> Iterator[Domain]: ...
```

`get` names the candidates on a miss, following `env_mgr/registry.py:27` — which
already does exactly this (`have {sorted(REGISTRY)}`, M15) and is the precedent
main design O5 cites.

The kind decides the layout, and only that: §8.2's table.

### 3.4 `Zone` — one attempt's region

```python
class Zone(NamedTuple):
    task_id: TaskId
    attempt: int                 # §11.3 -- a zone belongs to an attempt, not a task
    root: str                    # <parent>/task.<uuid>.<version>.<hash>/
    def contains(self, path: str) -> bool: ...   # -> fs.path.contained
```

**The zone id is the runtime `uuid.version`** (spec §5.1) — the task's own
identity, not a separate namespace, plus a per-level hash for readability. The
hash buys no confidence: spec §4.1 already settled that an unguessable prefix is
security-by-obscurity, recovered three ways by the agent that holds it.

Because the layout nests (§8.1) and permissions cover the task's own subtree
(`task_graph` spec §3.2.2), **"may this task reach that path" is `contains`** —
one function serving both questions, which is what the nesting is for.

---

## 4. Isolation

The load-bearing section. Spec §4 specifies it against measured behaviour; this
one implements it against the same, plus what the survey found about how such
things are built.

### 4.1 What the chain actually degrades

Spec §4.2 orders the chain bubblewrap → Landlock → refuse. The design keeps that
order, and the survey supports it: Codex migrated *away* from Landlock to a
bundled bubblewrap after shipping the Landlock version (F16).

**But the two rungs are not the same kind of confinement, and the spec presents
them as though they were.** Measured (M1): this kernel is **Landlock ABI 3**,
which has filesystem rights, `REFER` and `TRUNCATE` — and no network restriction
at all, which arrives at ABI 4. Spec §4.2's own table says bubblewrap "also
isolates network and PID". So falling from rung 1 to rung 2 silently drops
network and PID isolation.

The design's position: **the chain degrades in preference *and* in properties,
and the difference is reported, not hidden.**

```python
class Confinement(NamedTuple):
    mechanism: Literal["bwrap", "landlock"]
    filesystem: bool             # always True -- it is the reason we are here
    network: bool                # bwrap: True. landlock: abi >= 4
    pid: bool                    # bwrap: True. landlock: never
    abi: int | None              # landlock only
```

`prepare()` returns this (§11.2) and the o11y record carries it, so "this run was
network-isolated" is an answerable question afterwards rather than an assumption.
§16 O1.

### 4.2 Selection is injectable, because criterion 9 is otherwise untestable

```python
class Availability(NamedTuple):
    bwrap: str | None            # path to the binary, or None
    landlock_abi: int | None     # or None

def probe() -> Availability:
    """The real one: shutil.which('bwrap'), and landlock_create_ruleset(
    NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)."""

def select(av: Availability) -> Literal["bwrap", "landlock"]:
    """bwrap when present, Landlock otherwise. Raises NoConfinement when
    neither -- and the caller does not catch it (§4.6)."""
```

`select` takes an `Availability` rather than calling `probe()` itself. That is the
whole trick, and it exists because of a measurement: **criterion 9 has no machine
on which all three branches run** (M16). `bwrap` is absent here, so rung 1 cannot
be exercised; and there is no ordinary way to make a Landlock-capable kernel look
incapable, so rung 3 cannot be exercised wherever rung 2 works.

With `select` taking its input as an argument, the three branches are three
one-line unit tests, and one end-to-end test runs against whatever the machine
actually has.

**And the survey says that is only half the answer** (F7, F8). Nobody achieves
"fail, never skip" by probing at run time — the kernel selftests ship
`CONFIG_SECURITY_LANDLOCK=y` as part of the test, and `rust-landlock` pins
`LANDLOCK_CRATE_TEST_ABI` per CI runner and **asserts the kernel matches it**,
booting UML kernels to cover the rest. The generalisation, which this design
adopts:

> **The mechanism and its ABI are a declared, asserted input to CI — not a
> discovered condition.** §14.1.

### 4.3 bubblewrap — an argument list, not a library

`bwrap.py` builds an argument vector. There is no binding to write; the
mechanism is a process.

```python
def argv(policy: Policy, *, bwrap: str) -> list[str]:
    """--ro-bind / --bind / --dev / --proc / --tmpfs from a Policy (§5.3),
    plus --unshare-net and --unshare-pid, which is where rung 1's extra
    properties come from (§4.1)."""
```

One thing to inherit from bubblewrap's own test arguments (F12): it distinguishes
`--ro-bind`, which hard-fails on a missing source, from `--ro-bind-try`, which
tolerates it — and uses `-try` for exactly the paths that legitimately do not
exist on some hosts (`/lib64`, `/sbin`, `/nix/store`). §5.4 makes that a property
of the granted entry rather than of the mechanism, so both rungs agree.

### 4.4 Landlock — a ctypes binding, ~120 lines, and three things that bite

There is no libc wrapper for any of the three syscalls, so this is `syscall(2)`
by number: 444 / 445 / 446 on x86-64. A working instrument already exists at
`scratch/design/probes-envmgr/landlock.py` and every measurement in this document
was taken with it.

```python
def abi_version() -> int: ...
def build(policy: Policy) -> Ruleset: ...
def restrict(ruleset: Ruleset) -> RestrictionStatus: ...
```

Three details that are not obvious and each cost a measurement:

**(a) The rights mask depends on what the target *is*.** Handing a directory-only
right (`MAKE_REG`, `READ_DIR`, `MAKE_DIR`, …) for a non-directory target is
`EINVAL`, not an ignored bit (M5). So:

```python
FILE_RIGHTS = A_EXECUTE | A_WRITE_FILE | A_READ_FILE | A_TRUNCATE
if not os.path.isdir(path):
    access &= FILE_RIGHTS
```

This is not a guess. `rust-landlock` does the same thing by `fstat` on the
`O_PATH` fd, and the comment two lines above it reads `// Linux would return
EINVAL.` (`src/fs.rs:316`, F13). An implementation that grants uniformly dies on
the first `/dev/null` it is handed, with `EINVAL` and no indication which path
caused it.

**(b) The rights mask also depends on the ABI.** Handing a bit the kernel does not
know is `EINVAL`. `BY_ABI` masks `handled_access_fs` down to what the running
kernel accepts.

**(c) `restrict_self()` restricts only the calling thread below ABI 8.**
`all_threads()` arrives at ABI 8; at ABI 3 it does not exist, and in a best-effort
implementation it is **silently dropped, leaving sibling threads unrestricted
while the status still reports enforced** (F15). This is a live hazard for a
Python process — so §4.5 restricts **before any thread is started**, and §14.3
asserts it. §16 O2.

### 4.5 Construction: fail-open by default is the ecosystem's answer, and not ours

The sequence *is* the decision, so it appears as a body:

```python
def apply(policy: Policy, av: Availability, *, tier: Tier) -> Confinement:
    mechanism = select(av)                      # raises NoConfinement -> §4.6
    if mechanism == "bwrap":
        return Confinement("bwrap", True, True, True, None)   # applied by exec
    ruleset = landlock.build(policy)            # (1) every entry, or an error
    status = landlock.restrict(ruleset)         # (2) irreversible from here
    if status.enforced is Enforced.NOTHING:     # (3) the hard gate, both tiers
        raise NoConfinement("ruleset enforced nothing")
    if tier is Tier.TEST and status.enforced is not Enforced.FULLY:
        raise NoConfinement(f"partial enforcement: {status.dropped}")   # (4)
    return Confinement.of(status)
```

**Step (1) is where this design departs from the ecosystem's default, on
purpose.** `rust-landlock`'s `path_beneath_rules` — the helper Codex uses for
`/`, for `/dev/null`, and for every writable root — does `Err(_) => None` on a
path it cannot open: **no rule, and no error** (F14). It is documented and
deliberate, and under deny-by-default a vanished grant is not an escalation. But
it means an allow-list typo silently evaporates, and spec principle 3 is "cannot
canonicalise a path, cannot obtain a sandbox, cannot decide — **deny**". A grant
that names a path we cannot open is a policy we cannot honour, and §5.4 decides
per entry whether that is fatal or expected — never per implementation accident.

**Steps (3) and (4) are the two-tier gate, and the spec conflates them.** Every
surveyed project splits it (F15): production does best-effort construction and
errors only if the ruleset ended up enforcing *nothing*; the test suite uses a
hard requirement and asserts full enforcement. `rust-landlock` states it under a
heading called "Test strategy" and adds that "applications should only check that
no error is returned".

Spec §4.2 states one rule for both tiers. This design implements two, and §16 D3
reports the difference rather than quietly picking one. The reason to keep the
strict tier at all is criterion 9's own logic: a suite that passes under partial
enforcement is a suite that cannot tell you enforcement degraded.

### 4.6 No sandbox, no start — and nobody catches `NoConfinement`

Spec §4.2: not "warn and continue". Criterion 8.

```python
class NoConfinement(RuntimeError):
    """No mechanism, or a mechanism that enforced nothing. Never caught inside
    this package. §11.1 lets it propagate and the task does not start."""
```

The survey's most useful negative result is that **the project closest to this
one decided the other way, in the weakest available form** (F10): Codex's
sandbox tests begin `if should_skip_bwrap_tests().await { eprintln!(…); return; }`
— a bare early return, so the test reports **green** on a machine with no sandbox
at all. It is named here because a reviewer will otherwise assume it was not
considered.

An agent started without confinement is an agent running with the operator's full
privileges while the system reports it is sandboxed. Refusing is better, and it
is only better if nothing anywhere converts the refusal into a warning.

### 4.7 The honest ceiling

Restated from spec §4.6 because it decides what this system may claim: the OS
sandbox is necessary and is the only thing measured to stop the scripted bypass.
It is **not inviolable**. For genuinely untrusted input the serious answer is a
VM per task. The alpha runs trusted-but-fallible agents against a known workload,
so a process sandbox is the right point on the curve — a judgement about the
threat model, not a claim about the mechanism.

Two additions this design's measurements justify:

- **Ordering is part of the boundary.** Landlock also hooks ptrace, and domain
  membership decides: a process that existed *before* confinement is protected
  from the confined one, and a child spawned after is not (M10). The supervisor's
  environment — which on a real machine holds API keys — is protected **because
  it started first**, not because of any filesystem rule. That is a real property
  and a fragile-looking one, and §11.1 depends on it.
- **A grant never widens access beyond DAC** (M11): `/etc` granted, `/etc/shadow`
  still denied. The granted set intersects the uid's existing rights; it is not a
  capability.

---

## 5. The granted set

### 5.1 Why we restrict reads at all, when nobody else does

The survey's most surprising result, and it deserves its own subsection because
it is the one place this design knowingly costs itself something (F12).

**No surveyed sandbox restricts reads.** Codex's entire filesystem policy is
three lines — `/` read-only, `/dev/null` read-write, plus the caller's writable
roots — and when a narrower read policy is asked for it **refuses rather than
approximates**:

> `Restricted read-only access is not supported by the legacy Linux Landlock
> filesystem backend.`

That is a coherent position, and taking it would make M3 and M7 disappear
entirely. This design does not take it, for one reason: **criteria 12 and 13
require read restriction, and criterion 13 is the producer/validator separation**
— the property the whole system exists to have (`validator` spec §8.1 names this
component as its mechanism). A system that grants `/` read-only has a producer
that can read its own validation's checking standard.

So reads are restricted, it works (P2, P3 measured every denial), and §5.2 and
§5.4 are the price.

### 5.2 The default granted system set, corrected

Spec §4.5.1 grants read-execute on `/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`,
`/etc`, `/proc`, "and the interpreter and toolchain paths a task declares through
its `env`", and states that a home directory is deliberately not in it.

Measured, that set does not start a program:

| Missing | Measured failure |
|---|---|
| **`/dev` is absent from the list entirely**, and `/dev/null` needs **write** | `fatal: could not open '/dev/null' for reading and writing: Permission denied` — git dies before reaching any repository question. M5 |
| **The interpreter is under `$HOME`** on any conda / pyenv / uv / venv install | `PermissionError: [Errno 13] … '/home/…/miniconda3/bin/python3'`, raised by `subprocess` in the *parent*, naming the interpreter rather than the sandbox. M3 |

So the default set is:

```python
DEFAULT_SYSTEM_SET = (
    Granted("/usr",  Access.READ_EXEC),
    Granted("/lib",  Access.READ_EXEC, optional=True),
    Granted("/lib64", Access.READ_EXEC, optional=True),
    Granted("/bin",  Access.READ_EXEC),
    Granted("/sbin", Access.READ_EXEC, optional=True),
    Granted("/etc",  Access.READ_EXEC),
    Granted("/proc", Access.READ_EXEC),
    Granted("/dev/null",    Access.READ_WRITE),      # ← write. M5
    Granted("/dev/zero",    Access.READ_EXEC),
    Granted("/dev/urandom", Access.READ_EXEC),
    Granted("/dev/tty",     Access.READ_WRITE, optional=True),
)
```

`/dev/{null,zero,full,random,urandom,tty}` is the node set Codex asserts must
survive its sandbox (F12), which is a useful cross-check on a list that is
otherwise a guess.

It remains **a default in configuration, not a constant in code** (spec §4.5.1) —
a site may narrow or widen it. And **the interpreter is not in it**: §11.2
resolves `sys.base_prefix` of the interpreter the task declares and adds it as a
task-level grant, so the failure mode is a missing declaration at prepare time
rather than an `EACCES` at exec time.

### 5.3 `Policy` — mechanism-independent

```python
class Mode(Flag):                # NOT `Access` — that name is task_graph's (§5.3.1)
    READ_EXEC  = auto()
    READ_WRITE = auto()

class Granted(NamedTuple):
    path: str
    mode: Mode
    optional: bool = False       # §5.4

class Policy(NamedTuple):
    granted: tuple[Granted, ...]
    def with_(self, *more: Granted) -> Policy: ...
```

#### 5.3.1 Why it is `Mode` and not `Access`

Rev. 1 called this `Access`, and `task_graph` design §3.5 also declares an
`Access` — `class Access(str, Enum): READ, WRITE`. The stage-three consistency
pass found that **`prepare()` mixes both in one `Policy`**: `Granted(zone.root,
Access.READ_WRITE)` uses this one, while the grants resolved from `Permissions`
carry that one. One name, two types, one call site, and `READ_WRITE` is not a
member of `task_graph`'s at all.

They are genuinely different things and both should exist:

| | Type | Answers |
|---|---|---|
| `task_graph.Access` | `str, Enum` — `READ` \| `WRITE` | *What did the package author declare?* A grant is read or it is write |
| `env_mgr.Mode` | `Flag` — `READ_EXEC` \| `READ_WRITE` | *What rights does the kernel get?* Combinable, and `READ_EXEC` has no declaration-side meaning at all — it exists because M3 measured that the interpreter's own prefix must be executable |

So this module renames its own, because the declared vocabulary is the shared one
and the kernel vocabulary is local. `grants.resolve` (§6.1) is the seam that maps
one to the other, and that mapping is now visible rather than being a name
collision that happened to type-check because neither is annotated.

One `Policy` builds either a `bwrap` argv (§4.3) or a Landlock ruleset (§4.4).
Nothing above `isolation/` knows which mechanism will consume it, which is what
makes §14.5's criteria testable against both.

The granted set is assembled in exactly the order spec §4.5 lists:

1. `DEFAULT_SYSTEM_SET` (§5.2),
2. the task's own zone, read-write (§3.4),
3. whatever its permissions name, resolved by §6,
4. nothing else.

### 5.4 `optional` — the one flag, and why it is per entry

`bwrap` exposes this as two flags (`--ro-bind` versus `--ro-bind-try`), used side
by side in its own tests for exactly the paths that legitimately do not exist on
some hosts (F12, F14). Landlock's ecosystem helper makes the opposite choice
silently for every path (F14).

Neither default is right for us, because the two cases are genuinely different:

| Entry | If the path does not exist |
|---|---|
| `/lib64` on a merged-`/usr` distro | **expected** — `optional=True`, skipped |
| A path a task's permissions named | **an error** — the policy cannot be honoured, and a typo must not evaporate |

So `optional` is a property of the granted entry, decided where the entry is
created, and §4.5 step (1) raises for any non-optional entry it cannot open. Spec
principle 3 gets the last word, and it gets it per entry rather than per
mechanism.

### 5.5 An allow-list makes an ungranted file look *broken*, not absent

A consequence of §5.1's choice, measured three times in one tool before it was
isolated (M7). Same tool, same option, two paths:

| `GIT_CONFIG_GLOBAL` points at | result |
|---|---|
| a path that does not exist | **rc=0** |
| a path that exists but is not granted | **rc=128**, `fatal: unknown error occurred while reading the configuration files` |

Real programs treat an optional file's absence as normal and a permission error
on it as fatal. So under §5.1, **every path a tool merely *probes* becomes a new
hard failure**, and the fix is per tool, not per sandbox:
`GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_SYSTEM=/dev/null`,
`core.excludesFile=/dev/null` were the three git needed.

```python
# workspace.py -- and this list is the shape of the problem, not the end of it
NEUTRALISED_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
```

This is honest but not complete: nothing enumerates what the *next* tool probes,
and it is the kind of list that grows by field report. §16 O3.

---

## 6. Resolving a grant

`closure` design D2 handed this step over: a grant references a handoff **kind
name**, `Permissions` never holds a `HandoffId`, and "where a runtime component
needs the instance, it resolves the name at that point — which is `env_mgr`, at
the moment it builds the zone".

### 6.1 The mapping is already on the runtime object

Measured (M13). `task_graph/models.py:107`:

```python
class Handoff(Model):
    id: HandoffId
    type: str = ""          # ← the kind name
```

So resolution needs no manifest read and no store access:

```python
def resolve(grant: Grant, task: Task, execution: Execution,
            handoffs: Mapping[HandoffId, Handoff], store_root: str) -> tuple[Granted, ...]:
    """kind name -> the instances of that kind this attempt actually has ->
    <store_root>/<hid>/v<N>/, per handoff design §6.2.

    Raises UnresolvedGrant if `grant.kind` matches no handoff on this task."""

def resolve_all(task: Task, execution: Execution, ctx: Context) -> tuple[Granted, ...]:
    """Every grant on `task.permissions`, flattened. `ctx` supplies the handoff
    mapping and the store root; §11.1 is its only caller."""
```

**`resolve_all` is what `prepare()` calls**, and rev. 1 named it there while
defining only `resolve`. The two differ in more than arity: `resolve` needs a
handoff mapping and a store root that `Context` carries and a single `Grant` does
not know about, so the wrapper is where the context is unpacked. Found in the
stage-three consistency pass.

#### The hole, and the half of it that is this module's to close

`Handoff.type` defaults to `""` and nothing requires it to be a registered kind
name (§16 O4, and `task_graph` O4 from the other side). Rev. 1 recorded that a
handoff whose type was never set matches no kind-named grant, **so the grant
silently covers nothing and the agent gets an empty granted set instead of an
error** — and then left it there.

Half of that is not this module's: who fills `type` is a `task_graph` spec
question, reported in the stage-three pass and not yet answered.

**The other half is, and rev. 2 closes it.** `resolve` **raises
`UnresolvedGrant`** naming the grant, the kind, and the task, rather than
returning `()`. The declared grant is the author's statement that this task needs
that artefact; a grant that resolves to nothing is a graph that will fail at the
first read, in a way that looks like the agent's fault.

This is `env_mgr` M7 turned around. M7's finding is that *an ungranted file looks
broken, not absent* — the sandbox's characteristic failure is a confusing error
somewhere else. An empty granted set is the same failure one level up, and it is
the version we control: the fix costs one raise, and it makes whichever route
`task_graph` eventually takes for `type` **loud when it is forgotten** rather than
silent. `prepare()` does not catch it, for the same reason it does not catch
`NoConfinement` (§4.6): the task does not start.

### 6.2 The version comes from the attempt

`Task.inputs` is a list of ids; the *versions* live on `Execution.input_versions`
(`models.py:158`), and a retry pushes a new `Execution`. Since a grant resolves to
`<root>/<hid>/v<N>/`, **`N` is attempt-dependent** — which is why `resolve` takes
an `Execution` and why §11.3 rebuilds the zone per attempt. §16 D6.

### 6.3 Two interpreters of one string, and how they are made to agree

`closure` design §6.3 fixes the covering relation at **exact string equality** and
forbids any other component from interpreting a path in a way that function does
not — citing kubernetes#122154, where a second interpreter made a static check
wrong in both directions and the maintainers shipped the wrongness rather than
open the grammar.

§3.2 requires realpath resolution. Those are two interpreters, and measured (M9)
they disagree on **every** case tried:

| pair | exact equality | after realpath |
|---|---|---|
| `zone` vs `zone/` | different | **same** |
| `zone` vs `zone/.` | different | **same** |
| symlink vs its target | different | **same** |
| `zone/../outside` vs `outside` | different | **same** |

Four for four, and each disagreement is in the forbidden direction: `covers()`
says "not covered" while this module would grant.

**The resolution is two checks at two times, and it makes the two agree by
construction rather than by care.**

```python
def canonical_syntax(path: str) -> bool:
    """Load time, no filesystem: absolute; no '.' or '..' segment; no trailing
    separator; no NUL; no wildcard character. Purely syntactic, so it is
    checkable when no zone exists yet."""

def canonical_here(path: str) -> bool:
    """Zone-build time: Path(path).resolve(strict=True) == path. Fails closed."""
```

A grant whose literal form is not already its canonical form is **rejected**, at
the earliest of the two moments it can be. Then the only strings that ever reach
either interpreter are ones where literal form *is* canonical form, and exact
equality and realpath cannot disagree about them.

`canonical_syntax` belongs in the schema that admits grant paths, so it fails at
load with the rest of a spec's errors. `canonical_here` cannot: symlinks do not
exist until the layout does. §16 D4.

The wildcard clause is not an anticipation of a feature. It is the closed side of
`closure` §6.3's grammar: if the schema never admits `*`, no component can be the
one that gives it meaning.

---

## 7. The workspace

### 7.1 The spec's two requirements cannot both hold

The central finding of this module, measured three ways (M6) and then measured
again for what it costs (M17).

| | |
|---|---|
| Spec §6.1 | the workspace is "a **worktree** cut from the main repository", not a clone, so agents share one object store |
| Spec §4.5 | "A task's executor may not write outside its zones. Local or remote, **no exception**" |

A worktree's `.git` is a file — `gitdir: <main>/.git/worktrees/<name>` — and the
main repository is outside the zone:

| Grant on the main repository | `status` / `log` | `add` / `commit` |
|---|---|---|
| not granted | `fatal: not a git repository: <main>/.git/worktrees/workspace` | — |
| **read-only** | works | **denied** — `fatal: Unable to create '<main>/.git/worktrees/workspace/index.lock'` |
| **read-write** | works | works |

**A worktree's index lives in the main repository**, so staging is a write outside
the zone. The only configuration in which an agent can commit is one where it can
write the shared repository.

**And that configuration re-creates the vulnerability criterion 11 exists to
prevent.** Confined with exactly that grant:

| attempt | result |
|---|---|
| write `<main>/.git/hooks/pre-commit` | **SUCCEEDED** |
| append to `<main>/.git/config` | **SUCCEEDED** |
| `git commit` afterwards | `rc=0`, **the agent's hook ran** |
| write another task's branch ref | **SUCCEEDED** |
| delete an object from the shared store | **SUCCEEDED** |
| *control:* read another task's zone | denied |
| *control:* read another task's worktree | denied |

The controls are the important row: **the zone boundary itself holds.** Every one
of these reaches through the shared repository that §6.1 requires. Criterion 11
names `.git/hooks` and `.git/config` explicitly, and spec §4.4 gives the reason —
CVE-2026-26268, CVSS 9.9, no user interaction.

### 7.1.1 The dependency repositories are per task

`closure` spec §2.5 rev. 9 gives a task spec a `repos` key — the dependency
repositories its work needs, `sglang`, `mooncake`, `aiter`. **The main repository
stays global** (one per run, in `Context`); the dependency list does not.

That split is the user's and it is the right shape for this module: the main
repository is what the *system* is working on, so `prepare()` cuts a workspace
from it every time; the dependencies are what *this task* needs, and a task that
needs none should not pay for cloning three.

```python
def cut(main_repo: str, zone: Zone, *, branch: str,
        repos: Sequence[str] = ()) -> Workspace:
    """As §7.2, plus one clone-with-alternates per declared dependency."""
```

`repos` is read from the task spec through `task.closure` (`task_graph` spec
§3.2.5), which is the same route the body takes and adds no field to the runtime
model.

**Each dependency is cut the same way the main repository is** — §7.2's clone
with alternates, under a read-only source — so §7.1's finding applies unchanged
and none of them can be written by the agent. The `extensions.preciousObjects`
precondition applies to each, which is a real cost: it is a `git config` write on
every repository a task names, not just on one.

**Nothing resolves a name to a location.** A declared `repos` entry is a key into
the run configuration, exactly as a resource pool name is. Where `sglang` lives is
a deployment fact and `Context` carries the mapping.

### 7.2 A clone with alternates, and it costs nothing it appeared to

```python
def cut(main_repo: str, zone: Zone, *, branch: str) -> Workspace:
    """git clone --shared --no-hardlinks <main_repo> <zone>/workspace.

    Objects are READ from the main repository via
    .git/objects/info/alternates and WRITTEN locally. The main repository is
    granted READ-ONLY (§5.3), and stays that way."""
```

Measured under a read-only main repository (M18): `status`, `log`, `cat-file` of a
blob from the shared store, `add`, `commit`, `checkout -b` — all rc=0. Writes to
`<main>/.git/hooks` and `<main>/.git/config` — denied. Objects are not copied:
main `.git` 468 KiB, clone `.git` 176 KiB.

So §6.1's *stated purpose* — "several agents get isolated checkouts sharing one
object store" — and §4.5's write rule hold simultaneously. This is a deviation
from §6.1's mechanism and criterion 20's wording; §16 D1.

**Its two apparent costs both have measured answers** (M23):

**Cost 1, the gc hazard, is real and total.** `man git-clone` warns the borrower
"will become corrupt", and names ordinary `git commit` in the *source* as the
trigger. Reproduced: main deletes a branch, runs `gc --prune=now`, and the
borrower is left at

```
-- git log:   fatal: bad object HEAD
-- git fsck:  error: refs/heads/feature: invalid sha1 pointer 83bfa08…
              missing blob 2491e062994c322e740761da7e6480668c43bb59
```

Not degraded. The agent's entire workspace history is unreadable.

**Its mitigation is built into git, and verified:**

```
[extensions]
    preciousObjects = true       # in the MAIN repository
```

```
fatal: cannot prune in a precious-objects repo
fatal: cannot delete packs in a precious-objects repo
-- borrower still healthy?  28c175c feat / 8e91775 base
```

`prepare()` sets it when it first sees a main repository, and refuses to cut a
workspace from one where it is not set — because the alternative is an agent's
work destroyed by someone else's routine housekeeping. The escape hatch, if a
workspace must outlive the arrangement: `git repack -a` plus dropping
`objects/info/alternates` makes it self-contained (objects 512 → 2.5K at probe
size) and `fsck` clean.

**Cost 2, "a branch is visible to the operator without a fetch", comes back.**
Spec §6.1 gives that as the second reason for a worktree. Measured three ways:

| how work returns to main | result |
|---|---|
| agent pushes, main read-only | **rejected** — `remote unpack failed: unable to create temporary object directory` |
| agent pushes, main writable | OK — but that is §7.1's forbidden grant |
| **main fetches from the agent's clone** | **OK** |

The third row is the design: **the write happens on the main side, performed by
the supervisor, not by the confined agent.** §4.5 holds unmodified, and the
property becomes one supervisor-side `git fetch` instead of a property of the
layout.

```python
def collect(ws: Workspace, main_repo: str) -> str:
    """Supervisor-side. `git fetch <ws.path> <branch>:<branch>` into the main
    repository. Runs OUTSIDE the agent's confinement, by the process that
    already holds write access. Returns the fetched ref."""
```

### 7.3 The residual, stated because it is bounded and real

The agent can write `<zone>/workspace/.git/hooks` — its own. That is inside its
zone, it is not an escape, and an agent that can run code in its own zone can do
that anyway. Criterion 11's protection is about the policy file and about *shared*
repository state, and both hold under §7.2.

---

## 8. Storage layout

### 8.1 Nested, following the task tree

Spec §5.1, unchanged:

```
<root>/task.<uuid>.<version>.<hash>/
  ├── handoffs/     ├── workspace/    ├── playground/    ├── logs/
  └── task.<child-uuid>.<version>.<hash>/     ← a subtask, nested
```

The nesting is what makes containment (§3.2) answer both "is this path in the
zone" and "may this task reach that path", because permissions cover the task's
own subtree recursively (`task_graph` spec §3.2.2) and the subtree *is* the
nesting.

### 8.2 Kind decides layout, and only that

| `DomainKind` | Layout under a task |
|---|---|
| `HANDOFF_STORAGE` | `handoffs/`, and `handoff` design §6.2's `<hid>/v<N>/` beneath it |
| `WORKSPACE` | `workspace/`, a clone with alternates (§7.2) |
| `PLAYGROUND` | `playground/`, empty, and the agent owns everything inside it (spec §6.2) |

`logs/` is not a domain: it is created with the zone and granted read-write like
anything else in it, which is what spec §6.1's last row asks for.

### 8.3 Where a validation's materials go — the layout has no room, and criterion 13 needs one

Criterion 13 says a producer cannot read a validation's checking standard,
"resolved entirely by §5.1's containment". `validator` spec §8.1 names this
component as the mechanism.

**But §5.1's layout is five things and none of them is a validation** (M22). And
the one placement the layout would suggest is the wrong one: anything under the
producing task's directory is inside its subtree, and therefore reachable.

So the layout gains one rule:

> **A validation's materials are a sibling of the producing task's zone, never a
> descendant of it.**

```
<root>/
  ├── task.<uuid>.…/                  ← the producing task
  └── validation.<uuid>.<phase>.…/    ← its validation. A SIBLING
```

Nothing else is needed. Under §5.3's granted set, a sibling is not granted, and
measured (P2, P12) a confined process cannot read a sibling, cannot traverse into
one, and cannot even `mkdir` outside its own zone. Criterion 13 is then true for
the reason the criterion claims — containment — once the layout puts the
materials somewhere containment excludes. §16 D5.

### 8.4 The layer limit, and the architecture it constrains

Measured (M20), confirmed against the kernel's own `LANDLOCK_MAX_NUM_LAYERS 16`
(S5): the 16th `restrict` returns `E2BIG`. Layers intersect, so a subtask's
executor *can* be confined further than its parent's.

**This binds under exactly one architecture.** If a subtask's executor is a
descendant process of its parent's executor, task-tree depth is hard-capped at
16 — and `task_graph` treats depth as unbounded. If the supervisor spawns each
executor directly, every executor carries exactly one layer and the limit never
binds.

This design takes the second: **`prepare()` is called by the supervisor, once per
attempt, and the executor is the supervisor's child.** That is also what §4.7's
ordering property requires — the supervisor must already exist, outside the
domain, for its environment to be protected. §16 O5.

---

## 9. Sync

### 9.1 A one-time job, at a defined point

Spec §5.3. Not a reconciliation loop.

```python
def sync(zone: Zone, mapping: Mapping, *, direction: Direction) -> SyncReport:
    """Once, at task start. Scoped to this task's subtree, never the root.
    Excludes the playground (§9.4)."""
```

### 9.2 "Identical" has a direction, and the spec does not say which

Measured (M21). `rsync -a --delete` does make two trees equal — by destroying
everything the destination had that the source did not. A remote-only file is
gone. **There is no symmetric mode.**

So `Direction` is a required argument, not a default:

```python
class Direction(str, Enum):
    LOCAL_TO_REMOTE = "local_to_remote"
    REMOTE_TO_LOCAL = "remote_to_local"
```

Spec §5.3's "local and remote are made identical" is implemented as "the
destination is made identical to the source, and the caller names which is
which".

### 9.3 Conflict is detected here, because `rsync` cannot detect it

Both sides edited the same file, local's mtime deliberately older:

| invocation | what the remote holds afterwards |
|---|---|
| `rsync -a` | `LOCAL edit` — the newer side silently discarded |
| `rsync -a --update` | `REMOTE edit` — picked by mtime, which is a guess |
| `rsync -a --checksum` | `LOCAL edit` — silently discarded |

**No flag reports that both had changed.** This settles the shape of spec §11's
open question — it cannot be answered by choosing a flag.

Since sync runs **once, at task start** (§9.1), the honest v1 answer is small and
achievable:

```python
class SyncReport(NamedTuple):
    copied: int
    deleted: int
    conflicts: tuple[str, ...]   # existed on both sides, differing, before the copy
```

Detection is a pre-pass comparing the two sides before anything is written; when
`conflicts` is non-empty, `prepare()` refuses. That is not conflict *resolution* —
resolution stays in [`../../docs/ROADMAP.md`](../../docs/ROADMAP.md) — but it
converts silent data loss into a stopped task, which is the difference the open
question is actually about. §16 O6.

### 9.4 Scope, and an honest note about why

Spec §5.3 gives two reasons for per-task scope: correctness (not touching another
task's material) and time ("proportional to the system rather than to the work").

Measured at 20 tasks × 50 files: whole root 1000 files in **108 ms**, one task 50
files in **53 ms**. Only 2×, because rsync's fixed startup dominates at this size.
The scaling argument is presumably right at real sizes; at the size measured, the
**correctness** argument is the one that holds. Recorded because a design that
cites a performance reason should say when it measured one.

**The playground is excluded** (spec §6.2, criterion 16). `--exclude playground/`
omits the contents but still creates the directory on the far side, empty — which
is consistent with §6.4 giving the remote its own playground, and is what
criterion 16's assertion must actually say.

---

## 10. Remote access

### 10.1 Two mechanisms in v1

`ssh` and `docker exec`. `kubectl exec` and `slurm`/`spur` are roadmap (spec
§5.4). How to detect and reach a remote is **knowledge produced by a designated
task and delivered as a knowledge handoff** (spec §7), not configuration here.

### 10.2 One Protocol

```python
class Connection(Protocol):
    def run(self, argv: Sequence[str], *, cwd: str | None = None,
            timeout: float | None = None) -> CompletedProcess: ...
    def push(self, local: str, remote: str) -> SyncReport: ...
    def pull(self, remote: str, local: str) -> SyncReport: ...
```

Three methods, because that is what §9 and §11 call. The transport-specific
options that always leak through such an abstraction are held in the `Mapping`
that `meta.py` persists, not passed per call — so the leak is in configuration,
where it is inspectable, rather than in a signature.

### 10.3 The surface is tool calls, with schemas

Spec §5.5, criterion 18: the whole remote↔local surface is exposed to agents as
**tool calls**, not as a procedure described in prose, because "an agent given a
natural-language description of how to sync a directory will improvise, and the
improvisation will be wrong in a way nobody notices".

```python
def tools(conn: Connection, zone: Zone) -> tuple[ToolDef, ...]:
    """env_remote_run, env_remote_push, env_remote_pull -- each with a JSON
    schema, each closed over this attempt's zone so a path argument cannot
    name another task's material."""
```

Closing over the zone is what makes criterion 10 true on this surface too: the
zone root is never taken from agent-supplied input, because the tool does not
accept one.

### 10.4 The far side is less confined, and that is written down

Spec §11 lists this as open. Sharpened here rather than resolved: §4 confines a
**local** process. A command sent over `ssh` runs in whatever the far side
provides, and nothing in this design confines it.

**So the honest statement is that remote execution is less isolated than local**,
and `Confinement` (§4.1) is reported per side so a run's record says so rather
than implying otherwise. §16 O7.

---

## 11. Preparing an environment

### 11.1 The order is the design

```python
def prepare(task: Task, execution: Execution, agent_spec: AgentSpec,
            ctx: Context) -> Prepared:
    zone   = layout.create(task, execution, ctx.domains)      # 1
    policy = Policy(DEFAULT_SYSTEM_SET).with_(
                 Granted(zone.root, Mode.READ_WRITE),         #    §5.3.1
                 *grants.resolve_all(task, execution, ctx),   # 2  §6
                 *ctx.interpreter_grants)                     # 3  §5.2
    ws     = workspace.cut(ctx.main_repo, zone, branch=...)   # 4  §7.2
    report = sync.sync(zone, ctx.mapping, direction=...)      # 5  §9
    if report.conflicts:
        raise PrepareRefused(report.conflicts)                #    §9.3
    layout.stage_handoffs(task, execution, zone, ctx)         # 6  §8.3
    material.deploy(agent_spec, zone)                         # 6b §11.5
    conf   = isolation.apply(policy, probe(), tier=ctx.tier)  # 7  §4.5 -- LAST
    return Prepared(zone, ws, policy, conf, report)
```

Rev. 1 wrote step 6 as `handoffs.stage(...)`, naming a module §2's file listing
does not contain. It is `fs/layout.py`'s: staging an input into the zone is
placing a directory at a path the layout decides, and a second module owning that
would be a second answer to *where does this go*.

```python
class Context(NamedTuple):
    """Everything `prepare` needs that is not on the task or the attempt.
       Built once by the composition root; see interfaces.md §2."""
    domains: DomainRegistry              # §3.3
    handoffs: Mapping[HandoffId, Handoff]   # for §6's kind -> instance step
    store_root: str                      # handoff design §6.2's <root>
    main_repo: str                       # §7.2
    mapping: Mapping[str, str]           # §9, spec §3.1's declared sync mapping
    interpreter_grants: tuple[Granted, ...]  # §5.2 -- M3's measured requirement
    tier: Tier                           # §4.5's two-tier gate
```

Rev. 1 used five of those attributes and defined none of them. `handoffs` and
`store_root` are the two the stage-three pass added: §6.1's `resolve` needs both
and `resolve_all` had nowhere to get them.

**Step 7 is last, and that is load-bearing twice over.** Everything before it
writes outside the zone by design — creating the zone, cutting the workspace,
staging handoffs — and none of it is possible afterwards. And §4.7's ordering
property depends on it: the supervisor and every process that already exists are
outside the resulting domain, and therefore protected from the agent by Landlock's
ptrace hook.

`NoConfinement` from step 7 is not caught (§4.6). The task does not start.

### 11.2 What `Prepared` carries

```python
class Prepared(NamedTuple):
    zone: Zone
    workspace: Workspace
    policy: Policy
    confinement: Confinement     # §4.1 -- mechanism, and whether net/pid are isolated
    sync: SyncReport
```

`confinement` is in the return value because §4.1 refuses to let the chain's
degradation be invisible, and because `agent` design O2 needs the prepared
interpreter path to decide which `claude` CLI the backend runs — that decision is
this module's (spec §9, [`../../docs/TODO.md`](../../docs/TODO.md) item 6), and
this is the input to it.

### 11.3 Per attempt, not per task

`prepare` takes an `Execution`. §6.2 is the reason: grants resolve to
`<root>/<hid>/v<N>/` and `N` lives on the attempt, so a retry has a different
granted set. Spec §4.5's "the sandbox is built once, at task start" is true of a
single attempt and not of a task with two. §16 D6.

The playground is **reloaded, not recreated**, when the zone already exists — that
is criterion 17 and spec §6.2's "it survives a resume". Nothing may depend on its
contents; §14.6 says why that half is not testable.

### 11.4 What the composition root registers, and under what name

`agent` design §7.1 says its `Runner` *"resolves `agent_specs`, `env_mgr` and the
validator's phase runner from the registry, by name, at use time"*. **No document
registered this module under any name**, which the stage-three consistency pass
found while assembling the one normative composition root
([`../../docs/interfaces.md`](../../docs/interfaces.md) §2). Rev. 1 of this
document never mentions the component registry at all, because everything above
`prepare()` was somebody else's caller.

```python
r.register("env_mgr", EnvManager(ctx))
```

```python
class EnvManager:
    """The registered component. One method the runner calls, and it is prepare."""
    def __init__(self, ctx: Context) -> None: ...
    def prepare(self, task: Task, execution: Execution,
                agent_spec: AgentSpec) -> Prepared: ...
```

A thin object over §11.1's function, and it exists for one reason: `Context` is
composition-time configuration — the domains, the store root, the main repository,
the sync mapping, the tier — and threading it through the runner would make the
runner carry configuration it has no opinion about. The object binds the context
once at the root and the runner passes only what varies per dispatch.

**It has no other method.** Everything else in this document is either below
`prepare` or is CLI surface (§12.2), and adding a second method here is how the
runner would start making environment decisions.

---

### 11.5 The agent's material, and the parameter rev. 2 did not have

`agent` spec §3.1 says `env` is *"resolved by `env_mgr`"*, and `agent` design §3.4
says `rules`, `hooks` and `skills` are handed *"to `env_mgr` to deploy"*. **Rev. 2's
`prepare(task, execution, ctx)` could not receive any of them** — no parameter, no
`Context` field. Four keys, one named consumer, no route. Found by the
spec-key-to-runtime trace, and it is the same shape as the composition root's two
missing registrations.

```python
def deploy(agent_spec: AgentSpec, zone: Zone) -> None:
    """Place this agent's rules, hooks and skills into the zone, and satisfy its
    declared `env`. Paths in the task package, in Claude Code's canonical form;
    this module places them and parses none of them."""
```

**The spec arrives as a parameter, not through a registry.** `prepare` grows
`agent_spec: AgentSpec` and `Context` does not gain the spec registry, because
the caller already has it: `agent`'s `Runner` resolves `agent_specs` by name and
is what calls `prepare`. Giving this module a registry handle would let it resolve
*any* spec, which is a larger authority than the job needs and would put the spec
layer on the wrong side of §1.2's split.

**It runs at step 6b — before confinement, after the zone exists.** Deploying is
writing into the zone, and step 7 makes writing impossible. It is beside handoff
staging because it is the same kind of act: putting something the executor will
need where the executor can reach it.

**`env` is where this module's existing machinery meets the new parameter.** A
declared environment requirement is what `recipe.py` and `installers/` already
resolve (§12.1) — the one part of the user's interface brief that was already
built. What is new is only that the requirement now has a route from the agent
spec to them.

**This module still parses nothing.** `rules`, `hooks` and `skills` are paths in
Claude Code's canonical form (`agent` spec §4.5); converting between harness
formats is an independent module that does not exist (`agent` design §3.5). A file
is placed, not read.

## 12. The shipped package

### 12.1 Unchanged, and a test says so

`recipe.py`, `layer.py`, `installers/`, `runner.py`, `outcome.py`, `report.py`,
`registry.py`, `versions.py`. 65 tests pass today and criterion 22 requires them
to keep passing untouched.

`registry.py` already lists candidates on a miss (M15) — the precedent main design
O5 cites — so nothing here needs it changed.

### 12.2 `cli.py` gains sub-commands, and the 65 tests still pass

Spec §9 extends the CLI with domain and zone inspection. The shipped parser is
`env-mgr <stage> <recipe>` with `stage` a positional `choices=STAGES`, and all 15
CLI tests call `main(["check", recipe, …])`.

Measured (M12): converting the four stages into sub-parsers and adding `domain`
and `zone` beside them parses **all six shipped call shapes identically**, sets
the same `stage` / `recipe` / `--json` attributes, and preserves `SystemExit(2)`
on an invalid stage.

```
env-mgr check|dry-run|install|bootstrap <recipe> [flags]     ← unchanged
env-mgr domain [name]                                        ← new
env-mgr zone [task-id]                                       ← new
```

One caveat, recorded because it is the only observable difference: a *global* flag
placed **before** the sub-command would no longer parse. No shipped test does
that, and no documented invocation does either.

---

## 13. Build versus adopt

| Concern | Considered | Chosen | Why |
|---|---|---|---|
| Landlock binding | `rust-landlock` (not Python), a `ctypes` binding, `pylandlock` | **own `ctypes`, ~120 lines** | There is no maintained Python binding. The instrument that took every measurement in this document is already written and is that size. The three syscalls have no libc wrapper, so any binding is `syscall(2)` by number regardless |
| Sandbox mechanism | Write our own namespace code | **`bwrap` binary, else Landlock** | Spec §4.2. Codex moved *to* bundled bwrap after shipping Landlock (F16), which is the same direction |
| Rights masking | Grant uniformly and catch `EINVAL` | **`fstat` and mask per target type** | M5 measured the `EINVAL`; `rust-landlock/src/fs.rs:316` does exactly this and its comment names the same errno (F13) |
| Rule construction | `path_beneath_rules`-style skip-on-missing | **raise, unless `optional`** | F14: the ecosystem default is silently fail-open, and spec principle 3 is fail-closed. §5.4 |
| Workspace | worktree; full clone; **clone with alternates** | **clone with alternates** | §7.1 measured that a worktree cannot satisfy §4.5; a full clone copies the object store the spec wants shared |
| Sync | `mutagen`, `unison`, `syncthing`, **`rsync`** | **`rsync`** | Spec §5.2 names it, and §9.1 needs a one-shot copy, not a reconciler. §9.3 adds the one thing it cannot do |
| Remote | `fabric`, `paramiko`, plain `ssh` | **plain `ssh` / `docker exec` behind §10.2** | Two mechanisms, three methods. A library would add a dependency to hide a `subprocess` call |
| Test harness | fork per test; helper binaries; **a sandboxed re-run of the test body** | **§14.3** | S4/F1: the kernel deleted its per-test opt-in because isolation belongs to the runner. F6 is the cheapest pytest analogue |

---

## 14. Test plan

`tests/env_mgr/` gains the new suites beside the shipped 65. Criteria 2–14 are
CI-enforced on every commit (spec §10) and need a subprocess and a filesystem —
no agent, no API key.

### 14.1 The environment is a declared input, not a discovered condition

Spec §10: "When no sandbox mechanism is available, the suite fails. It does not
skip."

The survey's sharpest structural finding is that **nobody achieves that by probing
at run time** (F7, F8). The kernel ships `CONFIG_SECURITY_LANDLOCK=y` with its
selftests; `rust-landlock` pins `LANDLOCK_CRATE_TEST_ABI` per CI runner and
asserts the kernel matches. So:

- CI declares `ENV_MGR_TEST_MECHANISM` (`bwrap` or `landlock`) and, for Landlock,
  `ENV_MGR_TEST_ABI`.
- `test_environment.py::test_declared_mechanism_is_the_one_present` asserts the
  machine matches the declaration, and **fails** otherwise.
- Absent the variables — a developer's machine — it auto-detects and runs. The
  hard failure is CI-side, exactly as `rust-landlock` arranges it.

**The gate lives in one place.** bubblewrap's `BWRAP_MUST_WORK` is the right shape
and leaks: the variable appears in one file, and the Python half of its own suite
skips unconditionally — passing green in the CI job that sets it (F9). A single
session-scoped fixture is the whole defence against repeating that.

**And skipping is allowed for exactly one thing.** The kernel's selftests skip 15
times, always for optional *filesystems* (`overlayfs`, `tracefs`), never for
Landlock (S2). The rule adopted: **skip for environmental variation orthogonal to
the property under test; never skip the property.**

### 14.2 Denials are asserted by errno against a named path

The pattern is not a preference. My own instrument produced a **false PASS** from
a `returncode != 0` check, because both children were failing to exec the
interpreter rather than being denied (M4). The kernel's suite has zero
occurrences of `ASSERT_NE(0, …)` on an access check; every site is
`ASSERT_EQ(EACCES, test_open(path, flags))` with the helper returning errno (S3,
F4). Codex's `expect_denied` is `assert_ne!(exit_code, 0)` and cannot tell a
denial from a missing shell (F11).

```python
def attempt(path: str, mode: str) -> int:
    """0 on success, else errno. The only way this suite observes an access."""
```

Two reinforcements taken from the survey:

- **A positive control.** The same operation must succeed *before* enforcement in
  the same process (F3). That alone rules out "the binary could not run".
- **A distinct errno for the permitted case.** bubblewrap passes a deliberately
  bad pointer so a permitted syscall fails with a *different known* errno rather
  than succeeding, making "allowed vs denied" two distinct non-zero codes and
  leaving 126/127 free to mean "the harness itself is broken" (F5). Used for the
  cross-`exec` cases in §14.3.

### 14.3 The harness forks; tests do not

Landlock restriction is irreversible and inherited, so a test that sandboxes the
pytest process poisons every later test. The kernel's answer is that **isolation
belongs to the runner**: `TEST_F_FORK` was deprecated into an alias for `TEST_F`
because the harness forks everything (S4, F1).

pytest does not fork, so this design supplies it once:

```python
@pytest.fixture
def sandboxed(request):
    """Runs the decorated test body in a forked child, applies the policy there,
    and marshals the failure text back over a pipe. The parent never restricts
    itself. Modelled on bubblewrap's tests/test-helper.py (F6)."""
```

Two details taken verbatim from `rust-landlock` (F2): a crash in the child must
never be scored a pass, and the child must `os._exit()` so handlers inherited
from pytest do not run inside the sandbox.

Cross-`exec` cases (criterion 7) use a helper script whose **exit status is the
errno**, per §14.2.

`test_threads.py::test_restriction_precedes_any_thread` asserts §4.4(c): at ABI 3
`all_threads()` does not exist, so a sibling thread started before `restrict`
would remain unrestricted while the status still reports enforced.

### 14.4 The decoupling wall

`test_imports.py` walks the module graph and asserts both directions of §2.1:
nothing new imports the installer machinery, and nothing in the installer
machinery imports `fs`, `isolation`, `grants`, `workspace`, `sync`, or `remote`.
`cli.py` is the single allowed exception.

### 14.5 Criteria

| # | Test | File |
|---|---|---|
| 1 | `test_register_idempotent`, `test_reload_preserves_playground`, `test_kind_decides_layout` | `test_domain.py` |
| 2 | `test_subtask_nested_under_parent`, `test_reach_is_containment` | `test_layout.py` |
| 3 | `test_sibling_prefix_denied`, `test_symlink_out_denied`, `test_dotdot_denied` — **twice**: against `contained()` and against a confined subprocess, because §3.1 shows the kernel answers all three without it | `test_path.py`, `test_confine.py` |
| 4 | `test_broken_symlink_denied`, `test_symlink_loop_denied`, `test_nonstrict_resolve_is_not_used` | `test_path.py` |
| 5 | `test_nul_byte_rejected`, `test_valueerror_is_caught_too` (§3.2) | `test_path.py` |
| 6 | `test_scripted_bypass_denied` asserting **EACCES against the target**, and `test_same_script_inside_zone_succeeds` with the interpreter granted (M3) | `test_confine.py` |
| 7 | `test_bash_child_inherits`, `test_second_ruleset_cannot_widen` | `test_confine.py` |
| 8 | `test_no_mechanism_refuses_to_start`, `test_refusal_names_the_reason` | `test_chain.py` |
| 9 | `test_prefers_bwrap`, `test_falls_back_to_landlock`, `test_refuses_when_neither` — unit tests over `select(Availability)`, plus one end-to-end against the declared mechanism (§4.2, §14.1) | `test_chain.py` |
| 10 | `test_zone_root_not_from_agent_input`, `test_tool_takes_no_zone_argument` (§10.3) | `test_policy.py`, `test_tools.py` |
| 11 | `test_policy_not_writable_by_agent`, `test_main_git_hooks_denied`, `test_main_git_config_denied`, `test_shell_rc_denied` | `test_workspace.py` |
| 12 | `test_ungranted_read_denied`, `test_ungoverned_path_denied` | `test_confine.py` |
| 13 | `test_validation_is_a_sibling_not_a_descendant`, `test_producer_cannot_read_validation` (§8.3) | `test_layout.py` |
| 14 | `test_sibling_zone_created_later_unreachable`, `test_no_rebuild_required` | `test_confine.py` |
| 15 | `test_sync_once_at_start`, `test_destination_matches_source`, `test_scoped_to_task_not_root` | `test_sync.py` |
| 16 | `test_playground_not_synced`, `test_playground_dir_created_empty` (§9.4) | `test_sync.py` |
| 17 | `test_playground_survives_resume` — **half of the criterion only**; §14.6 | `test_layout.py` |
| 18 | `test_remote_tools_have_schemas`, `test_tool_call_round_trip` | `test_tools.py` |
| 19 | `test_agent_works_on_a_copy`, `test_stored_artefact_byte_identical` — needs `handoff`'s digest; §15 orders it | `test_handoff_entry.py` |
| 20 | `test_workspace_shares_object_store`, `test_main_checkout_unmodified` — **not** "is a worktree"; §16 D1 | `test_workspace.py` |
| 21 | — **no artefact exists**; §14.6 | — |
| 22 | The shipped 65, unchanged, plus `test_cli_subcommands_preserve_shipped_shapes` (§12.2) | `tests/env_mgr/` |

### 14.6 Three criteria that cannot be satisfied as written

Stated here rather than quietly approximated.

| # | Why |
|---|---|
| **9** | No machine runs all three branches (M16). Split into three unit tests over an injected `Availability` plus one end-to-end against the declared mechanism. The *composition* is tested; the three-way degradation on one host is not, and cannot be |
| **17** | "nothing depends on its contents having survived" is a property of all future code, not an observable of a run. The survival half is tested; the non-dependence half is a review rule, not a test |
| **21** | Requires a knowledge handoff carrying cluster conventions, and spec §11 concedes the system-level tasks that would produce one are unspecified. There is nothing to test against until one exists |

Criteria 18 and 19 are testable but cross-module: 18 needs the tool surface and 19
needs `handoff`'s digest. §15 orders them last for that reason.

---

## 15. Implementation order

Each step leaves the suite green.

| # | Step | Unblocks |
|---|---|---|
| 1 | `fs/path.py` and its tests — criteria 3, 4, 5 | everything; it is the bottom of §2's graph |
| 2 | `isolation/landlock.py` + `probe.py` + `policy.py`, promoting the probe instrument | criteria 6, 7, 12, 14 |
| 3 | `isolation/apply.py` and the chain — criteria 8, 9 | the §14.3 fixture, which every later confinement test needs |
| 4 | `fs/domain.py`, `fs/zone.py`, `fs/layout.py` — criteria 1, 2, 13 | §8.3's sibling rule |
| 5 | `grants.py` — §6, and `canonical_syntax` into the grant schema | criterion 10 |
| 6 | `workspace.py` — criteria 11, 20 | the module's largest deviation, and the one most worth reviewing early |
| 7 | `sync.py` — criteria 15, 16 | |
| 8 | `remote/` — criterion 18 | needs `agent`'s tool surface |
| 9 | `prepare.py` — criterion 17, and the composition | needs 1–8 |
| 10 | `cli.py` sub-commands — criterion 22 | last, because it is the only shipped file touched |
| 11 | The handoff entry — criterion 19 | needs `handoff` implemented |

Steps 1–3 are the safety claims and should land before anything that depends on
them looks finished.

---

## 16. Deviations, and new open questions

### 16.1 Deviations from the spec

The spec set is agreed and a design does not amend it. Each of these is reported.

| # | Deviation | Why | Effect |
|---|---|---|---|
| **D1** | **The workspace is a `git clone --shared`, not a worktree.** Contradicts §6.1's mechanism and criterion 20's wording | §7.1 measured that a worktree's index lives in the main repository, so the only configuration in which an agent can commit lets it write `<main>/.git/hooks` — where the hook then **runs**. That is CVE-2026-26268, which §4.4 cites as criterion 11's reason for existing | §6.1's stated *purpose* is preserved exactly; its mechanism changes. Both apparent costs have measured answers (§7.2), one of which — `extensions.preciousObjects` in the main repository — becomes a precondition `prepare()` enforces |
| **D2** | **§4.5.1's default granted set is extended**: `/dev/null` read-**write**, the `/dev` character devices, and the task's declared interpreter | Measured: without `/dev/null` writable, git dies before any repository question; with the interpreter under `$HOME`, `subprocess` fails at exec naming the interpreter rather than the sandbox (M5, M3) | The spec's own clause ("the interpreter and toolchain paths a task declares") is what makes this an extension rather than a contradiction — but the default is never sufficient alone, which the spec does not say |
| **D3** | **Fail-closed is implemented at two tiers**, where §4.2 states one rule | Every surveyed project splits it: production errors only on `NotEnforced`; the test suite demands full enforcement (F15). Stating one rule for both is stricter than anything surveyed | §4.5 steps (3) and (4). The strict tier is kept because a suite that passes under partial enforcement cannot detect degradation |
| **D4** | **A grant path must be canonical, and is rejected otherwise** — a rule the spec does not have | §6.3 measured that exact-equality and realpath disagree on all four forms tried, always in the direction `closure` §6.3 forbids. Requiring canonical form makes the two interpreters agree by construction | Syntactic half belongs in the grant schema (load time); the realpath half runs at zone build. Adds a load-time failure mode that did not exist |
| **D5** | **A validation's materials are a sibling of the producing task's zone** — an addition to §5.1's layout | Criterion 13 says containment resolves the property, but §5.1's layout has no place for a validation, and the only place it has room is inside the producing subtree, which is reachable (M22) | Small, but it is an addition rather than a reading, and criterion 13 is untrue without it |
| **D6** | **The zone is built per attempt, not per task** | Grants resolve to `<root>/<hid>/v<N>/` and `N` lives on `Execution`, not `Task` (M14) | §4.5's "the sandbox is built once, at task start" is true of one attempt. A retry rebuilds |
| **D7** | **Criteria 9, 17 and 21 are not satisfied as written**, and §14.6 says so rather than approximating | No machine runs criterion 9's three branches; criterion 17's second half is not an observable; criterion 21 has no artefact | Reported. 9 is decomposed, 17 is half-tested, 21 is blocked on the system-level tasks spec §11 already lists as unspecified |

### 16.2 New open questions

| # | Question |
|---|---|
| **O1** | **The chain's two rungs are different kinds of confinement.** bubblewrap isolates network and PID; Landlock at ABI ≤ 3 isolates neither and cannot touch the network before ABI 4. §4.1 reports the difference in `Confinement`, but nothing decides whether a task that *needs* network isolation may run on rung 2 at all. That is a policy question and it is not this document's to settle |
| **O2** | **Below ABI 8, `restrict_self()` restricts only the calling thread.** §11.1 restricts before any thread starts and §14.3 asserts it, which is sufficient today. It stops being sufficient the moment anything in the executor's startup path spawns a thread first, and the failure is silent — the status still reports enforced |
| **O3** | **Nothing enumerates what the next tool probes.** §5.5's three git variables were found by running git under confinement. The list grows by field report, and there is no mechanism that would have predicted them. A per-tool neutralisation table has no owner |
| **O4** | **`Handoff.type` defaults to `""` and nothing requires it to be a registered kind.** A handoff whose type was never set matches no kind-named grant, so the grant covers nothing and the agent gets an empty granted set rather than an error. The fix belongs in `task_graph` or in admission, not here |
| **O5** | **Whether executors nest as processes decides whether task depth is capped at 16.** §8.4 chooses supervisor-spawned executors, which avoids the cap. Nothing outside this document records that the choice has that consequence, and `task_graph` treats depth as unbounded |
| **O6** | **§9.3 detects a conflict and refuses; it does not resolve one.** Refusing is right for a one-shot sync at task start. It is not right for whatever eventually wants to sync mid-task, and that caller does not exist yet |
| **O7** | **Remote execution is less isolated than local, and now says so.** §10.4 reports it per side rather than resolving it. The moment a validation runs remotely, criterion 13 stops being enforced by anything this document specifies |

## 17. o11y

Side-cars that watch a run. **The rule that outranks every feature in this
chapter: o11y may never fail the thing it observes.** Every failure is one
`log.warning` and a skip, and there is a test per mode holding that line.

### 17.1 AgentsView

[AgentsView](https://github.com/kenn-io/agentsview) is an external Go binary
that reads Claude Code's JSONL transcripts and serves search, analytics and
token-cost views. `agent_sys` reaches its backend through `claude-agent-sdk`,
which spawns the `claude` CLI, which writes exactly those transcripts — so the
two fit with no glue on either side. The whole integration is *where the
transcripts land* and *which directory the panel reads*.

**AgentsView's own code is never modified.** Every knob is one it publishes:
`--port`, `CLAUDE_PROJECTS_DIR`, `AGENTSVIEW_DATA_DIR`, `disabled_agents`.

#### The prefix

`~/.infera_agent_sys`, laid out like `~/.local` (`bin/ share/ state/ run/`),
owned by `env_mgr` and named by the `AGENT_SYS_*` family in `prefix.py`. It
exists because the two obvious alternatives are both wrong: `/usr/local/bin` is
host state we promised not to touch, and `~/.local/bin` is the user's. Upstream's
`install.sh` is not used — it hardcodes `/usr/local/bin` with no override point.
The recipe (`recipes/agentsview.o11y.yaml`, `installer: bin`,
`importance: suggested`) installs a pinned release and verifies its published
`SHA256SUMS`.

`state/claude` is deliberately not under a run root: the daemon outlives any
single run, so the directory it reads must be a stable path.

#### Session scoping — five gates

The panel must show only the sessions `agent_sys` produced. **The constraint
that outranks the feature: the user's own Claude Code must be untouched.**

| | |
|---|---|
| 1 | `CLAUDE_CONFIG_DIR=$AGENT_SYS_CLAUDE_HOME` in the **child's** environment dict, never in our `os.environ`. `material.deploy` sets its own per-attempt value, so `<zone>/config/projects` is symlinked into the prefix — credentials and settings stay the zone's, only the output is shared |
| 2 | `CLAUDE_PROJECTS_DIR` points the panel at that one root |
| 3 | `disabled_agents` switches off the other 60 providers. Pinned and hand-maintained; `check_disabled_agents` warns when it has drifted from the installed binary **in either direction** — a name the binary dropped breaks `serve` loudly, a name it gained leaks silently |
| 4 | `AGENTSVIEW_DATA_DIR` is ours, so a user's own archive and settings are untouched |
| 5 | `HOME` is redirected into the prefix for the binary's own subprocesses. AgentsView derives every provider's *default* root from `HOME`, so this needs no list and cannot go stale when upstream adds a provider we have never heard of. Found while measuring, and stronger than gate 3 |

#### Lifecycle, port, failure

Started at the end of the `env_mgr` deploy path and left **resident**
(`daemon_idle_timeout = "0s"`; the default 20m would empty the panel for anyone
opening the URL after their run). Default port `18888`; resolution order is
`--agentsview-port`, then `AGENTSVIEW_PORT`, then the default, and an unusable
value falls back with a warning rather than failing. `--no-agentsview`,
`--dry-run` and `--clean` make no external call at all.

**A taken port is a warning and a skip, never a relocation.** We bind-probe
before launching precisely because `serve` would otherwise move quietly to the
next free port, and a panel on 18889 is a panel nobody knows the address of.
`--replace` is passed for the same reason: without it, `serve --background
--port N` silently attaches to any daemon already alive for this data directory
and reports *its* port, exit 0, `N` ignored.

**Reuse requires proof of ownership.** A live AgentsView on the port is not
evidence it is ours — a user's own daemon lists every session on the machine.
Two gates: it answers `/api/v1/agents` with 200 and JSON (a status code is not
an identity), **and** a live `daemon.<pid>.json` in our data directory names
that port. That record is AgentsView's own artefact, read never written; because
the data directory is ours alone, one found there was written by a daemon we
configured. It is removed on a clean stop, so only an unclean death leaves a
stale one, and that is caught by checking the pid.

**No validation path may start a daemon.** Measured: `health`, `projects` and
`session list` all autostart one on a port AgentsView picks — the delegation
this component exists to prevent, happening where nobody is watching.
`doctor sync` does not, and answers the same question. `AGENTSVIEW_NO_DAEMON=1`
does not rescue them; it makes them refuse outright.

**Success goes to the event stream** (`EventKind.O11Y_PANEL`), not `logging`:
this package never configures `logging`, so an info record reaches nobody while
`log.warning` still reaches stderr through `lastResort`.

#### Known limitation

The zone symlink's behaviour under an enforcing policy is **untested, because
currently untestable**: `agent_sys` refuses to start any AI task under
`AGENT_SYS_NO_PERMISSIONS=0` today, before the executor runs, so nothing ever
traverses the link. That refusal predates this feature — measured with paired
arms differing in one file, both failing identically. If a confined child ever
does follow it, the prefix is under `$HOME`, which `DEFAULT_SYSTEM_SET` does not
grant; the likely repair is a grant on `$AGENT_SYS_CLAUDE_HOME/projects`, which
is a permissions decision and is deliberately not taken here. Read this as
"untested", never as "safe".

### 17.2 One project per run

AgentsView derives a project from the session's **deepest** path segment. Every
agent attempt runs in its own zone, and zones nest, so one run's sessions arrive
as several unrelated projects — measured on a real nested fixture, four sessions
of one run as `0_11e34171`, `0_f6daeb1b`, `task.main.b869ddf0_…` and
`task.solve_a.8c8fb4c1_…`.

**Renaming the directories cannot fix this, and that is the whole reason this
section exists.** PR #156 put the closure name into every runtime directory,
which made those strings readable; it did not join them, because a nested child
task fragments off from its parent however prettily both are named. The only
filesystem fix would be putting every attempt of a run under one directory,
which is precisely what zone isolation exists to prevent.

So `env_mgr/o11y/mapping.py` posts **one `explicit` mapping over the run root**
at run start, through AgentsView's own settings API. The dependency stays
unmodified.

#### What was measured before any of it was written

Every property below came from a container probe against a real v0.42.0, not
from the API's shape or a field's name. Three of them changed the code.

| | |
|---|---|
| **`explicit` is the only usable layout** | The other legal value, `repo_dot_worktrees`, matched **zero** sessions across thirteen prefix shapes — including a genuine `<repo>/.worktrees/<name>` tree the archive had correctly identified. Why is **unsettled**; upstream source at `ff8fb4e8` would settle it. We do not use it |
| **`explicit` is depth-independent** | One mapping over `runs/<A>` caught that run's sessions at depth 3 *and* 5, and only that run's. Nesting is a non-issue — which is what makes the whole design work |
| **`Origin` is mandatory** | Without it a mutating call answers a plain-text `403 Forbidden`, not the JSON error shape. It reads exactly like a missing endpoint |
| **`machine` must be read, never assembled** | A mapping written with a `machine` the daemon does not recognise matches nothing and says nothing. It comes from the daemon's own `local_machine`. The recon itself ran in a container whose hostname was not the host's — which is how a hard-coded `gethostname()` would have shipped broken |
| **Names normalise `-` → `_`** | Everything else round-trips, including `@ : / + # % .`, spaces and non-ASCII; there is no length cap. We normalise before posting so the string we send is the string the panel shows |
| **No `apply`, `preview`, `reclassify` or token** | A mapping that exists *before* ingest labels the session at sync time. Those calls are only for sessions already in the archive |
| **`409` is success** | `POST` is not idempotent; uniqueness is `(machine, path_prefix)`, not including layout. A re-run of the same run id conflicts with its own row |
| **Prefix boundaries are segment-safe, longest wins** | `run-AAA` does not capture `run-AAAX`; a specific row beats a general one |
| **Classification reads the recorded cwd, not the disk** | A session whose directory never existed classified and mapped normally; one whose directory was moved away survived a `sync --full`. **Zone teardown after a run is harmless to the panel** |
| **AgentsView does not look downward** | A zone is a non-git directory containing `git clone --shared` at `<zone>/workspace` whose alternates point outside it. Byte-for-byte identical classification to a bare plain directory: `repository_path=''`, `worktree_relationship='unknown'`. It never reaches `alternates` |

#### Retention, which is a policy and not hygiene

One row per run, and **no garbage collection** — deliberately. Deleting a row
does not disturb the panel on `apply`, but the next `sync --full` re-derives
labels from the mapping table and that run's sessions revert to their directory
names. **Keeping the label is keeping the row.** So any GC is a decision about
how long old runs stay named, not a tidiness measure, and a GC that silently
un-names last month's runs is worse than a table of small rows.

**Measured, and it is why no GC is needed.** The cost is `O(rows × sessions)`,
not `O(rows)` — visible only by repeating the whole curve at a different session
count, where the same row counts cost 3.5–4× more. It works out at ~0.75 µs per
(row × session), holding across a 20× range of the product: ~2.6 s added to a
`sync --full` at 5000 rows over 800 sessions, and nothing measurable below ~1000
rows.

**But `sync --full` is not the path the daemon runs.** Paired arms over an
identical 800-session archive differing only in the mapping table: idle 0.991 s
against 1.023 s, one new session 1.150 s against 1.189 s. **+32 ms and +39 ms at
5000 rows**, both inside the spread within either arm. Incremental sync — file
watch and poll timer — is unaffected.

Since runs and sessions grow together the mapping term is quadratic in run
count, but it loses to the linear re-parse cost of a full sync until roughly
**15 000 runs**, by which point a full sync takes ~19 minutes for reasons that
have nothing to do with mappings. If a GC is ever wanted, the trigger is
full-sync latency, not table size.

**One thing left unsettled, and it is the number a GC would turn on:** whether
`enabled=false` rows still cost a scan. Disabling rather than deleting is the
attractive shape for a GC — deleting un-names the run — and that measurement is
what would decide it. Re-run the scaling harness after flipping the rows via
`PUT /{id}`.

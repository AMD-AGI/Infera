"""What leaves `env_mgr/`, above the wall.

The safety-critical module. Its spec is the one written against measured
behaviour rather than intuition, and this surface follows: every guarantee here
has a probe behind it in `scratch/design/probes-envmgr/`.

Everything in this file sits **above** the decoupling wall. Nothing here imports
the installer machinery (`recipe`, `layer`, `runner`, `outcome`, `report`,
`registry`, `versions`, `installers/`), and nothing there learns about domains or
zones. A test asserts the wall in both directions.

Declarations only. See `docs/interfaces.md` §4.6.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from enum import Enum, Flag, auto
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, Protocol

from task_graph.ids import HandoffId, TaskId

__all__ = [
    "Confinement",
    "Context",
    "Domain",
    "DomainKind",
    "DomainRegistry",
    "EnvManager",
    "Granted",
    "Mode",
    "NoConfinement",
    "Policy",
    "Prepared",
    "PrepareRefused",
    "SyncReport",
    "Tier",
    "UnresolvedGrant",
    "ValidationZone",
    "Zone",
    "contained",
]


# --------------------------------------------------------------------------- #
# Errors. None of these is caught by the runner: criterion 14 is
# "no isolation, no start", and each of the three means the task does not start.


class NoConfinement(RuntimeError):
    """No mechanism could confine this task. The chain is bubblewrap, then
    Landlock, then **refuse** — there is no third rung."""


class PrepareRefused(RuntimeError):
    """Sync found a conflict it cannot resolve. `rsync` cannot express "make both
    sides identical" and cannot report the case: `--delete` destroys
    destination-only content, and with both sides edited `-a` and `--checksum`
    silently discard one while `--update` guesses by mtime. Detection is ours."""


class UnresolvedGrant(ValueError):
    """A declared grant names a handoff kind this task has no instance of.

    **Raised rather than resolving to an empty granted set**, which is what rev. 1
    did. `Handoff.type` defaults to `""` and nothing yet requires it to be a
    registered kind, so an unfilled type silently matched no grant and the agent
    received an empty set instead of an error.

    The declared grant is the author's statement that this task needs that
    artefact; a grant covering nothing is a graph that fails at the first read, in
    a way that looks like the agent's fault. Whichever route the open question
    takes, forgetting it is now loud.
    """


# --------------------------------------------------------------------------- #
# The path is the fact


def contained(path: str, zone: str) -> bool:
    """True iff `path` is `zone` or lies beneath it, **on resolved paths**.

    Four rules, each measured:

    1. Resolve first, both sides.
    2. The trailing separator is load-bearing — `p == z or p.startswith(z + sep)`.
       `zone-EVIL/x` passes a bare `startswith` and fails this.
    3. Canonicalisation fails closed. `resolve(strict=True)`, and **any**
       exception denies. `os.path.realpath` returns a partly-resolved path for a
       broken symlink *and* for a symlink loop without raising, and so does
       `Path.resolve()` at its default `strict=False` — which is the trap,
       because it is what an implementation reaches for.
    4. Reject NUL bytes. A NUL raises `ValueError`, **not** `OSError`, so
       `except OSError` alone makes rule 4 dead code.

    Canonicalised per check, at use time — resolving attacker-mutable components
    early is itself a TOCTOU bug.

    **This is not the enforcement mechanism.** Measured: the kernel already denies
    all three documented `startswith` defeats with no userspace check involved.
    This function decides which paths are handed to the kernel, refuses a grant
    whose literal and canonical forms disagree, and gives the hook a diagnostic
    the kernel cannot — attributing a denial to a tool call.
    """
    ...


class DomainKind(str, Enum):
    HANDOFF_STORAGE = "handoff_storage"
    PLAYGROUND = "playground"
    WORKSPACE = "workspace"


class Domain(NamedTuple):
    name: str
    root: str  # absolute, resolved, no trailing separator
    kind: DomainKind


class DomainRegistry(Protocol):
    def register(self, name: str, root: str, kind: DomainKind) -> Domain:
        """Idempotent: re-registering a name with the same root and kind returns
        the existing `Domain` and touches nothing on disk, which is what lets a
        playground survive a restart. A *different* root or kind for a live name
        is an error, not an update."""
        ...

    def get(self, name: str) -> Domain:
        """Names the candidates on a miss."""
        ...

    def __iter__(self) -> Iterator[Domain]: ...


class Zone(NamedTuple):
    """One task **attempt's** region.

    Not one task's. Grants resolve to `<root>/<hid>/v<N>/` and `N` lives on
    `Execution`, so a retry has a different granted set and rebuilds.
    """

    task_id: TaskId
    attempt: int
    root: str

    def contains(self, path: str) -> bool: ...


# --------------------------------------------------------------------------- #
# Isolation


class Mode(Flag):
    """What rights the **kernel** gets. Combinable.

    Deliberately not named `Access` — that name is `task_graph`'s and means what
    the *author declared*, read or write. `READ_EXEC` has no declaration-side
    meaning at all: it exists because the interpreter's own prefix must be
    executable, and every ordinary Python install is under `$HOME`, which the
    default granted set excludes.

    A Landlock rule whose target is a *file* EINVALs if given directory-only
    rights, so the mask depends on what the target is — `fstat` and mask per
    target, which is what the canonical Rust binding does for the same errno.
    """

    READ_EXEC = auto()
    READ_WRITE = auto()


class Granted(NamedTuple):
    path: str
    mode: Mode
    optional: bool = False


class Policy(NamedTuple):
    """Mechanism-independent. One `Policy` builds either a bubblewrap argv or a
    Landlock ruleset, and nothing above `isolation/` knows which will consume it.

    `optional` is per entry rather than global. bubblewrap exposes exactly this
    choice as two flags, `--ro-bind` versus `--ro-bind-try`, and uses both side by
    side; Landlock's ecosystem helper makes the fail-open choice silently for
    every path, so a typo in an allow-list evaporates.
    """

    granted: tuple[Granted, ...]

    def with_(self, *more: Granted) -> Policy: ...


class Confinement(NamedTuple):
    """What was actually achieved — reported, never assumed.

    **The chain degrades in properties, not only in preference.** bubblewrap also
    isolates network and PID; Landlock below ABI 4 isolates neither and cannot
    touch the network at all. Falling from rung 1 to rung 2 silently drops both
    unless something says so, so this is in `Prepared` and in the o11y record.
    """

    mechanism: Literal["bwrap", "landlock"]
    filesystem: bool
    network: bool
    pid: bool
    abi: int | None


class Tier(str, Enum):
    """Fail-closed is two rules at two tiers, everywhere it was surveyed.

    Production constructs best-effort and errors only if the ruleset ended up
    unenforced; the suite demands full enforcement, because a suite that passes
    under partial enforcement cannot detect degradation.
    """

    PRODUCTION = "production"
    STRICT = "strict"


class SyncReport(NamedTuple):
    sent: int
    received: int
    conflicts: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Preparing an environment


class Context(NamedTuple):
    """Everything `prepare` needs that is not on the task or the attempt.

    Built once by the composition root and bound into `EnvManager`. Threading it
    through the runner instead would make the runner carry configuration it has
    no opinion about.
    """

    domains: DomainRegistry
    #: **A live view, not a snapshot**, and the type does not say so — which is
    #: worth one line because the obvious reading of `Mapping` is *a dict you
    #: have*, and a `dict` here is empty for ever.
    #:
    #: A `Context` is built **before** `build_registry`, since `EnvManager(ctx)`
    #: is one of its arguments — and every handoff in the run is declared
    #: *inside* it, by `submit`. So anything captured at construction predates
    #: every handoff it would need to resolve. Supply a read-through mapping
    #: over the handoff manager, bound once the root returns.
    handoffs: Mapping[HandoffId, Any]
    store_root: str
    main_repo: str
    mapping: Mapping[str, str]
    interpreter_grants: tuple[Granted, ...]
    tier: Tier
    #: **The agent backend's CLI, absolute — declared, not discovered.**
    #:
    #: Measured: the SDK's `_find_cli` checks its own **bundled** binary first
    #: and returns before it ever calls `shutil.which("claude")`, so on this
    #: machine an agent runs `2.1.251` from the SDK while this package's recipe
    #: configured plugins into `2.1.246` on `PATH`. Two binaries, two versions,
    #: and the agent silently lacks the plugins that were installed for it.
    #:
    #: Declared rather than re-resolved because `PATH` can differ between the
    #: process that ran the recipe and the process that runs the agent, and two
    #: independent resolutions is the ambiguity one layer down. If it ever has
    #: to survive between invocations, `meta.py` is where it would persist.
    agent_cli: str | None = None
    #: **The task package, staged rather than granted.** `interfaces.md` §4.16.
    #:
    #: F19's third position. A grant on the package root is what opened
    #: criterion 13's second route — `validators/` lives in the package, so a
    #: producing task could read the standard it was about to be judged against
    #: without ever leaving its own permissions. Under a scope where permission
    #: management exists only to stop agents cross-contaminating, that route is
    #: anti-gaming rather than cross-contamination, and containment is what the
    #: spec says resolves it.
    #:
    #: So `prepare` copies this into the zone and tells the body where the copy
    #: went. **A package-relative path resolved against this original root now
    #: points outside every grant**, which is the seam `demo` and `agent` have
    #: to meet — see this package's README.
    package: str | None = None
    #: The package-relative paths to stage, or `None` for the whole package.
    #:
    #: **An allow-list, and the list shape is the point.** Criterion 14 holds
    #: because a zone nobody anticipated is *absent from a list*; a deny-list
    #: would give the anticipated-only guarantee §4.5 rejects. `None` is today's
    #: honest state and not a convenience: until `TODO.md` 4a separates a task's
    #: executable set from the validators', nothing can name that set, the copy
    #: carries `validators/` too, and criterion 13's second route is **moved
    #: rather than closed**.
    package_stage: tuple[str, ...] | None = None
    #: **How to reach the far side of each mapping, keyed by `local_root`.** A
    #: key with no entry means both ends are on this machine, which is every
    #: configuration before R1 and stays the default — so an empty mapping here
    #: is not a degradation.
    #:
    #: **A superset of `mapping`'s keys, deliberately.** `mapping` is
    #: `Meta.mapping_roots()` and is weak-only, because strength answers *must
    #: bytes be copied*. This answers *can I reach the far side*, which is a
    #: different question: a **strong** mapping is one mount seen by two
    #: machines and still has a far side — often the only one with the GPU on
    #: it. So `sync` finds nothing here for a strong mapping and never looks,
    #: while the tool surface does.
    #:
    #: A **sibling of `mapping` rather than a widening of it**: `sync.remote_root`
    #: owns the walk over `mapping` and is cited by `paths.zone_env`, which wants
    #: *where does this zone live over there* and has no use for a transport.
    #: Widening the value type would have made every reader of that walk carry a
    #: fact only `sync` needs.
    #:
    #: Typed `Any` rather than `SyncTransport` for one reason: `env_mgr.protocols`
    #: is imported by `env_mgr.remote.connection`, so naming the class here would
    #: be a cycle. `sync` is the only reader and it has the real type.
    transports: Mapping[str, Any] = MappingProxyType({})
    #: **Where the far side is, for every mapping**, keyed by `local_root`. The
    #: twin of `transports`: that one says how to reach it, this one says where
    #: it is. `mapping` is the weak-only subset and is `sync`'s input; the tool
    #: surface reads this, because a **strong** mapping's `remote_root` is not
    #: in `mapping` at all and tools resolved against `mapping` would point at
    #: nothing for exactly the configuration R1b uses.
    far_roots: Mapping[str, str] = MappingProxyType({})


class Prepared(NamedTuple):
    """What the runner is handed.

    The last field and the method are **rev. 5**, and both were found by the
    only caller trying to use this. Neither is tidiness.

    `environment` carries what `material.deploy` computes — `CLAUDE_CONFIG_DIR`,
    `CLAUDE_CODE_TMPDIR`, and the agent spec's own `env`. A five-field frozen
    tuple had nowhere to put them, so they were dropped and the runner could not
    see them. Measured: with `~/.claude` granted, a confined agent read the
    **operator's personal** `CLAUDE.md` and obeyed its language rule. Pointing
    `CLAUDE_CONFIG_DIR` into the zone is what removes the `$HOME` grant
    entirely. It is a read-only mapping: a mutable default on a `NamedTuple` is
    shared by every instance, which is one edit from one task's environment
    leaking into another's.

    The rejected alternative was `agent` calling `material.deploy` itself, which
    puts an environment decision in the runner — the thing *one method, and it
    stays one* exists to prevent.
    """

    zone: Zone
    workspace: Any
    policy: Policy
    #: **`None` is a value, not an absence** — it is how this says *unconfined*,
    #: and four sites branch on it: `prepare.py:286`, `agent/runner.py:749` and
    #: `:1343`, and `demo`'s §4.17a banner.
    #:
    #: It was accurate as `Confinement` until `ad730a2`. Before the kill switch
    #: `prepare` ended `conf = confinement_for(select(av), …)` and `select`
    #: **raises** `NoConfinement`, so no `Prepared` this module produced could
    #: carry `None`; the switch made it reachable and the type did not follow.
    #: Not drift discovered — a divergence introduced, with a consumer already
    #: depending on the new value.
    #:
    #: The comment nine lines below survived the whole time explaining what
    #: `confinement is None` means, which is the part worth remembering: a
    #: reader who checked this file found the semantics documented **and** a
    #: type saying the value could not occur, and prose reads as the more
    #: considered of the two.
    confinement: Confinement | None
    sync: SyncReport
    environment: Mapping[str, str] = MappingProxyType({})
    #: The resolved CLI this environment was provisioned for, or `None` when the
    #: `Context` declared none. A backend pins it and **refuses** if it cannot —
    #: silently running a different binary is what this field exists to stop.
    agent_cli: str | None = None
    #: **False when this run performed no permission management at all**, the
    #: user's `AGENT_SYS_NO_PERMISSIONS` kill switch. Stated rather than
    #: inferred: `confinement is None` would otherwise mean *unconfined* for two
    #: different reasons — no mechanism on the machine, or the switch — and
    #: §4.17a is that a fact a reader must infer is a fact a reader can miss.
    #:
    #: **Default `False` since 2026-08-30**, tracking the switch's own default
    #: (§4.22f). The declaration and `prepare.Prepared` must agree, or the two
    #: halves of the seam disagree about what an omitted field means.
    permissions_enforced: bool = False
    #: Every output slot with a version pinned → its `content/` directory,
    #: **keyed by `HandoffId`**. `agent` states each declared output, its kind
    #: and its resolved path in the conversation, and neither an environment
    #: variable nor a readme can carry a per-attempt path to a model.
    #:
    #: Absent means *no resolved path*, which `agent` renders rather than skips:
    #: an agent told about two of three outputs writes two and finishes.
    output_paths: Mapping[HandoffId, str] = MappingProxyType({})
    #: Where the task package was staged (§4.16), or `None` when none was
    #: configured. The same value reaches a **body** through
    #: `environment["AGENT_SYS_TASK_PACKAGE"]`, which is right for a body — it
    #: reads its own process environment, where the name is the interface. This
    #: field is for `agent.Runner`, which resolves a package-relative `entry`
    #: into an argv and must not reach into a mapping for a key spelled by a
    #: name copied across a boundary neither side checks.
    #:
    #: **Not `package`:** `Context.package` is the original checkout and this is
    #: the copy. Same type, different path; one name for both would make
    #: substituting either silent.
    staged_package: str | None = None
    #: **`remote.tools.ToolDef`s for this attempt's far side, or `()`.**
    #:
    #: On the returned value and **not** a third `EnvManager` method, which is
    #: `interfaces.md` §4.6's own precedent: `wrap_argv` sits here for the same
    #: reason, and `test_env_manager_exposes_exactly_these` pins the component's
    #: method set at two so that a third is a decision rather than a drift.
    #:
    #: Spec §5.5 is why this exists at all: the remote surface reaches an agent
    #: as **tool calls**, because *"an agent given a natural-language
    #: description of how to sync a directory will improvise, and the
    #: improvisation will be wrong in a way nobody notices"*. Until this field
    #: there was no route from `remote/tools.py` to any backend, so criterion 18
    #: was built and unreachable.
    #:
    #: Typed loosely for the same reason `Assignment.confinement` is: it crosses
    #: to `agent`, which may not import `env_mgr`.
    tools: tuple[Any, ...] = ()

    def spawn(self, argv: Sequence[str], **popen_kwargs: Any) -> Any:
        """Start `argv` **confined**, and hand back the process. One verb.

        `wrap_argv`'s shape does not carry over to Landlock: bubblewrap *is* the
        exec, so its confinement crosses the fork/exec boundary as **data** in a
        command line, while Landlock is a syscall against a live thread and must
        be executed in the child, after fork, before exec. So the caller gets a
        spawn rather than a wrapper, and branches on the mechanism nowhere.

        The child's whole job is two syscalls, deliberately. Forking a threaded
        process hands the child locks held by threads that do not exist in it —
        the documented reason `preexec_fn` is unsafe — and the runner is
        threaded by construction. A ruleset fd survives fork, so it is built in
        the parent and the child only restricts.

        Raises `NoConfinement` when there is no mechanism. This is where *no
        isolation, no start* lands for a caller that could not be confined in
        its own process.
        """
        ...

    def wrap_argv(self, argv: Sequence[str]) -> list[str]:
        """The executor's command line, confined. **Ask, do not assemble.**

        On the bubblewrap rung `apply()` confines nothing, because **bwrap *is*
        the exec** — so the policy only becomes real when something runs this.
        The caller cannot build it: a bwrap argv needs the policy *and* the
        binary, and `Availability` is not a type `agent` may import. Handing
        over the raw material would be this module publishing its internals so
        somebody else can do its job.

        Returns `argv` unchanged under Landlock, where the process was already
        confined when `prepare` returned. Raises `NoConfinement` when there is
        nothing to wrap with — including the binary having vanished since probe
        time, which is resolved here rather than remembered, the same rule as
        canonicalising per check.

        **It does not close the hole underneath it.** Under bubblewrap an *AI*
        backend cannot be wrapped by anyone, because the SDK spawns its own CLI
        and no caller ever sees that argv.
        """
        ...


class ValidationZone(NamedTuple):
    """Where a validation's materials go, and what was put there.

    `root` is a **sibling** of the producing task's zone, never a descendant —
    design D5, and criterion 13 is untrue without it, because anything under the
    producing task's directory is inside its subtree and permissions cover a
    task's own subtree recursively. Two modules were found answering *where does
    a validation go*; this is the one that owns the layout.

    `materials` are **copies** staged out of the store, which is what dissolves
    the seam where a body was handed handoff ids as strings in a zone with
    nothing pointing at the store. Copies rather than a grant, because a
    validation must not be able to edit what it is validating.

    It maps **handoff id → staged path**. A validator taking more than one input
    must know which copy is which — the binding is many-to-many and a verdict is
    per handoff — and the two sides' orderings do not correspond, so a bare list
    could not be zipped against one. Recovering the association by parsing
    `<materials>/<hid>/v<N>` would make this module's directory shape a contract
    another package quotes; handing over the association instead is the same
    rule as handing over an ordering rather than a sort key.
    """

    root: str
    phase: str
    materials: Mapping[HandoffId, str]


class EnvManager(Protocol):
    """The registered component, `env_mgr`. **One method, and it stays one.**

    A second is how the runner would start making environment decisions.
    """

    def prepare(self, task: Any, execution: Any, agent_spec: Any = None) -> Prepared:
        """Create the zone, build the policy, cut the workspace, sync, stage the
        handoffs, and **check** that a mechanism exists — `Prepared.spawn`
        applies it.

        **Step 7 split**: it once applied the confinement here, and a threaded
        runner cannot survive that — the thread that applies it can no longer
        write the store, irreversibly. So it checks and refuses early, before
        the workspace is cut, and the syscall happens in the child.

        The rule that made "confinement last" matter survives and is stronger
        for the move: the supervisor and every prior process stay outside the
        domain **by construction** rather than by ordering, so Landlock's ptrace
        hook protects the environment holding the API keys either way.

        Raises `NoConfinement`, `PrepareRefused` or `UnresolvedGrant`. The caller
        catches none of them.
        """
        ...

    def prepare_validation(self, task: Any, execution: Any, phase: Any) -> ValidationZone:
        """Place a validation's zone and stage what it validates.

        Resolved by name, never imported: `validator` may not import this
        package, and an import edge is permanent where a name lookup is not.

        **It confines nothing**, and that is a boundary rather than an omission.
        `prepare` applies Landlock to *its own process*, because the executor is
        that process's child; a phase runner calling this is the supervisor, and
        confining it would confine the supervisor. Who applies a policy to a
        validation *body* is a third question, and this does not quietly answer
        it.
        """
        ...

    def place_zone(self, task: Any, execution: Any) -> Zone:
        """Create this attempt's zone and **nothing else**.

        For a task that will never execute. A non-leaf's main phase is run by
        unfolding, so it reaches no path that calls `prepare` and never gets a
        zone — and a subtask's storage nests *inside its parent's*, so a parent
        without one is a parent whose children cannot be placed. No nested graph
        runs until something calls this.

        Repairing it with `prepare` is wrong three ways, two measured: it would
        cut a workspace and stage handoffs for a task that never runs; it would
        end in confinement of a thread that is about to be handed back; and that
        confinement would be refused anyway, because an attempt thread plus the
        main thread is already two.

        Confines nothing, cuts nothing, stages nothing.
        """
        ...

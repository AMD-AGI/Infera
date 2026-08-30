"""The composition root.

`docs/interfaces.md` §2 is normative for this listing, and it is wider than this
package: six of the names it registers are owned by modules `task_graph` may not
import. **Those are wired through a guarded, function-local import** — module
scope is pydantic, `spec_loader` and this package, per §4.7 rev. 5, so
`import task_graph` never reaches a sibling and a sibling that is still
declaration-only is skipped with a line in the log rather than an `ImportError`
at start-up. `test_bootstrap.py` asserts both halves.

Registration order is free: components resolve at use time, not construction
time. The one real constraint is that the packages load before the two
whole-catalogue passes, and that both run before the closure index freezes.
"""

import logging
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from itertools import chain
from typing import Any

from spec_loader import SpecInvalid, failed_names, format_problems, rejected
from task_graph.agent import AgentMgr
from task_graph.graph import check_graph
from task_graph.handoff import HandoffMgr
from task_graph.models import SUBGRAPH_AGENT_SPEC
from task_graph.policy import DepthFirstPolicy, SchedulePolicy
from task_graph.registry import Registry
from task_graph.resource import GpuMgr, ResourceMgr, TokenMgr
from task_graph.runner import FakeRunner, TaskRunner
from task_graph.scheduler import Scheduler
from task_graph.store import MemoryStoreMgr, StoreMgr
from task_graph.task import TaskMgr

__all__ = ["build_registry", "RegistryViews"]

log = logging.getLogger(__name__)


def build_registry(
    *,
    store: StoreMgr | None = None,
    runner: TaskRunner | None = None,
    policy: SchedulePolicy | None = None,
    resources: Sequence[ResourceMgr] | None = None,
    registries: Any = None,
    packages: Sequence[Any] = (),
    env: Any = None,
    strict_level: Any = None,
    monitors: Sequence[Any] | None = None,
    config_order: Sequence[str] = (),
    handoff_root: str | None = None,
    knowledge_root: str | None = None,
) -> Registry:
    """Wire a system. Every default is overridable, which is how a test
    substitutes a fake runner, a spy manager, or a different policy.

    The default runner stays `FakeRunner` deliberately: the suite written against
    a fake whose completion the test drives is the regression guard for
    everything else, and swapping the default would rewrite it. The real runner
    is passed in, by the demo and by the whole-system CLI.

    **`registries` is the same shape, and for the same reason.** The root cannot
    construct what only the caller can configure, and four symptoms of that had
    accumulated on the five spec registries: `handoff_root`, `knowledge_root`,
    `strict_level`, and `HandoffSpecRegistry`'s escape-hatch flag having no route
    at all — spec §5.3 gives it a command-line switch, and nothing in the
    assembled system could turn it on. Passing the five closes that without a
    parameter per flag, and takes four guarded sibling imports out of this file
    with it.

    **Additive on purpose.** Omitting it constructs exactly what this function
    constructed before, so no caller has to move. The default goes when `demo`
    passes the five, and not before — changing a signature under the first real
    caller is the shape §4.11 just named.
    """
    r = Registry()

    # 1 ── the runtime managers. None of these reads a spec.
    r.register("store_mgr", store or MemoryStoreMgr())
    r.register("handoff_mgr", HandoffMgr(r))
    r.register("task_mgr", TaskMgr(r))
    r.register("agent_mgr", AgentMgr(r))
    for pool in resources if resources is not None else _default_resources(r):
        r.register(f"resource:{pool.name}", pool)
    r.register("policy", policy or DepthFirstPolicy())

    # 2-5 ── the two handoff stores, the five spec registries, the package load,
    # and the two whole-catalogue passes. Every one belongs to another module.
    _wire_specs(
        r,
        registries=registries,
        packages=packages,
        handoff_root=handoff_root,
        knowledge_root=knowledge_root,
    )

    _bridge_agent_specs(r)

    # 6 ── the executors. They resolve specs, so they are built after loading.
    _wire_executors(r, env=env, strict_level=strict_level, config_order=config_order)
    r.register("runner", runner or FakeRunner())
    _wire_monitors(r, monitors=monitors)

    # 7 ── the scheduler, last. A statement rather than a constraint: a graph
    # cannot be assembled from specs that have not been admitted.
    r.register("scheduler", Scheduler(r))
    return r


def _bridge_agent_specs(r: Registry) -> None:
    """Make every admitted agent spec instantiable — `demo` F-D2.

    Two tables, and until now nothing joined them: `agent_specs` holds the spec
    *documents* `load_package` admitted, and `AgentMgr`'s table is the vocabulary
    `Scheduler.submit` checks with `is_registered`. A graph whose spec loaded
    cleanly still got `unknown agent spec 'collect'; registered: []` at submit.

    **It belongs here because it is a fact about the catalogue, not about any
    graph.** It does not vary per graph, no graph-builder has an opinion about
    it, and every graph needs it identically. The sharper form: this function
    has already run both whole-catalogue passes and raised on a fatal problem,
    so returning a registry that cannot dispatch anything after a *successful*
    load would mean the root had not finished composing.

    The name only. `Agent.config` is left empty by the task definition, and
    passing the admitted document would copy the whole agent spec into every
    minted `Agent` — the spec stays in `agent_specs`, resolved by name at use
    time, which is the discipline `Task.agent_spec` already follows.
    """
    agent_mgr = r.get("agent_mgr")
    # **The built-in first, and unconditionally.** A non-leaf's `agent_spec` is
    # `SUBGRAPH_AGENT_SPEC` because the author no longer writes one, and
    # `submit` gates every task on `is_registered`. It is registered here rather
    # than in `AgentMgr.__init__` for the reason this function exists: the
    # vocabulary `submit` checks with is the composition root's to assemble, and
    # a manager that pre-seeded its own table would be a second writer of it.
    #
    # Registered even when `agent` is declaration-only, unlike the loop below —
    # a graph can be built and submitted with no agent specs admitted at all,
    # and a non-leaf in one still has to pass the gate.
    agent_mgr.register(SUBGRAPH_AGENT_SPEC)
    if "agent_specs" not in r:
        return  # `agent` is declaration-only; the same condition every sibling has
    for name in sorted(r.get("agent_specs").names()):
        if name == SUBGRAPH_AGENT_SPEC:
            # An authored spec of that name is not an error and is not silently
            # dropped: registering it again is a no-op on the same key, and the
            # only reader that could tell them apart (`runner.agent_spec_of`) is
            # unreachable for a non-leaf. Said out loud because the name is now
            # part of the vocabulary and an author picking it should know.
            log.warning(
                "agent spec %r shares the name the system uses for a non-leaf; "
                "a leaf naming it will run it, a non-leaf will not",
                name,
            )
        agent_mgr.register(name)


def _default_resources(r: Registry) -> list[ResourceMgr]:
    return [GpuMgr(r, capacity=8), TokenMgr(r, capacity=1_000_000)]


SPEC_REGISTRIES = (
    ("HandoffSpecRegistry", "handoff", "handoff_specs"),
    ("ValidatorSpecRegistry", "validator", "validator_specs"),
    ("TaskSpecRegistry", "closure", "task_specs"),
    ("AgentSpecRegistry", "agent", "agent_specs"),
    ("ClosureRegistry", "closure", "closures"),
)


def _wire_specs(
    r: Registry,
    *,
    registries: Any,
    packages: Sequence[Any],
    handoff_root: str | None,
    knowledge_root: str | None,
) -> None:
    """Steps 2 through 5 of `interfaces.md` §2, when the modules exist.

    **The tail of this function is the whole-catalogue phase, and it is a shape
    rather than a list of errands.** Everything after `load_package` is work no
    single package can do for itself, because each of them needs a registry some
    *other* package filled — which is why it accumulates here and nowhere else.
    Three so far, and they come in two kinds:

    | | |
    |---|---|
    | passes that **report** | `check_closures`, `check_graph` — they return `Problem`s and raise nothing, so a catalogue with two faults reports both |
    | passes that **effect** | `_bridge_agent_specs` — it registers, and the root would otherwise return something that cannot dispatch |

    Both must run after **every** package and before the scheduler, and the
    second kind is the one to watch: a check that is skipped reports nothing and
    looks like a clean catalogue, while an effect that is skipped leaves a system
    that fails later and elsewhere. Two of this week's defects were the first
    kind and one was the second.

    A fourth is expected — `agent.AgentSpecRegistry.check_knowledge`, which
    returns `Problem`s and so belongs with the first kind. It is not called yet:
    its `mandatory` argument is a run-config flag with no route into this
    function, which is the gap this phase keeps producing rather than a detail
    of that pass.

    Steps 4 and 5 are genuinely ordered and it is the one real constraint in the
    root: `load_package` needs the five registries; `check_closures` needs
    *every* package loaded; `check_graph` runs after it and takes what it
    rejected in `skip`; `freeze()` comes after both, because the reverse index
    is built over the closures that passed.
    """
    if registries is not None:
        # The caller built them, so this file constructs none of the four
        # sibling types and guesses none of their configuration. A missing
        # attribute raises here rather than degrading — `Registries` declares
        # all five, and a view that cannot answer is a wiring fault, not a
        # tolerable absence.
        for _, _, key in SPEC_REGISTRIES:
            r.register(key, getattr(registries, key))
        parts = _optional({"FilesystemStore": "handoff"})
    else:
        parts = _optional({"FilesystemStore": "handoff", **{n: m for n, m, _ in SPEC_REGISTRIES}})
    # The five spec registries come first. `interfaces.md` §2 lists the stores
    # ahead of them and §2.2 says registration order is free — but a store now
    # takes a `KindSource` that reads `handoff_specs`, so building them in the
    # listed order would read top-down as though the store could resolve a kind
    # before the registry existed. It resolves lazily and would work either way;
    # the swap is for whoever reads it next.
    for name, _, key in SPEC_REGISTRIES:
        if name in parts:
            r.register(key, parts[name]())

    # One implementation, two roots. A root that was not supplied leaves the
    # name unregistered: an artefact store rooted at a default nobody chose is
    # worse than a loud `KeyError` at the first resolution.
    if "FilesystemStore" in parts:
        store_cls, kinds = parts["FilesystemStore"], _KindSource(r)
        if handoff_root is not None:
            r.register("handoff_store", store_cls(handoff_root, kinds=kinds))
        if knowledge_root is not None:
            r.register("knowledge_store", store_cls(knowledge_root, kinds=kinds))

    if not packages:
        return
    missing = [k for k in ("handoff_specs", "task_specs", "closures") if k not in r]
    if missing:
        raise RuntimeError(
            f"cannot load {len(packages)} task package(s): the spec registries "
            f"{sorted(missing)} are not available. Their modules are not implemented yet."
        )
    views = RegistryViews(r)
    loaded = _optional({"load_package": "spec_loader", "check_closures": "closure"})
    closures, task_specs = r.get("closures"), r.get("task_specs")

    reports = [loaded["load_package"](pkg, views) for pkg in packages]
    failed = failed_names(reports)
    problems = list(chain.from_iterable(rep.problems for rep in reports))
    if "check_closures" in loaded:
        # `interfaces.md` §2 rev. 5: one `HandoffSpecRegistry` receives every
        # package, so its own report *is* the whole-catalogue answer, and the
        # `merged(reports)` the listing used to call for cannot be written — a
        # `LoadReport` has no `without_validator` to fold.
        #
        # **A plain call, deliberately.** This was `getattr(..., lambda: None)`,
        # and the two sides had spelled the accessor differently: the default
        # produced `None`, `check_closures` returns early on `None`, and an
        # escape-hatch admission went unreported in the assembled system with
        # three suites green. `load_report` is not optional — `handoff_specs` is
        # a registry this root requires and the guard above already refused
        # without one — so a rename should raise here rather than degrade to
        # silence. A test stub that wants tolerance answers the method.
        handoff_report = r.get("handoff_specs").load_report()
        problems += loaded["check_closures"](
            views, handoff_report, skip=_names_for(closures, failed)
        )
    if "agent_specs" in r:
        # `agent` spec §3.6 checks 3 and 4 — criterion 2's whole mechanism, and
        # it ran nowhere: `agent`'s own tests call it directly, which is why it
        # was green. **`mandatory` stays False and that is not the gap it looks
        # like**: the flag chooses fatal-versus-warn, and warn *is* the spec's
        # default, so the pass now does its default job. What has no route is
        # the strict half — spec §3.5's run-config knob, which needs a decision
        # about how run configuration reaches this function at all, since
        # `strict_level`, `config_order` and the two roots are the same question
        # asked four times. Reported; not invented here.
        #
        # **A plain call.** This was guarded by `hasattr(..., "check_knowledge")`
        # — written an hour after I removed the `getattr(..., lambda: None)` two
        # functions up, for the same defect, and it slipped past the test that
        # forbids a three-argument `getattr` because `hasattr` is a different
        # spelling of the same quiet skip. A rename in `agent` would have
        # stopped the pass silently. The tolerance it bought was for a test
        # stub, and a stub that wants tolerance answers the method.
        _, knowledge = r.get("agent_specs").check_knowledge(r.get("handoff_specs"))
        problems += knowledge
    skip = failed | rejected(problems)
    problems += check_graph(_catalogue(task_specs), skip=_names_for(task_specs, skip))
    fatal = [p for p in problems if p.fatal]
    if fatal:
        raise SpecInvalid(format_problems(fatal))
    # **Non-fatal problems are reported, not dropped**, and this line is a
    # defect repair rather than a nicety. `Problem.fatal=False` has one producer
    # — `closure`'s check 3, for a closure built from a kind admitted under the
    # escape hatch — and reporting it is what `closure` criterion 6 *is*. This
    # function computed those problems and then filtered them away, so the
    # escape-hatch admission reached nobody even after `handoff` renamed the
    # accessor and I removed the `getattr` default that had been swallowing it.
    # Three fixes on one path, and the value was still going nowhere at the end
    # of it: a plausible value produced and discarded, which is the same family
    # as the four.
    if remaining := [p for p in problems if not p.fatal]:
        log.warning(
            "%d admitted with reservations:\n%s", len(remaining), format_problems(remaining)
        )
    closures.freeze()


def _names_for(registry: Any, origins: AbstractSet[str]) -> frozenset[str]:
    """Turn a set of `Problem.origin`s into the spec **names** a `skip` filters on.

    `failed_names` and `rejected` answer in origins — a `Problem` has carried one
    since it was frozen — while every registry is keyed by name and both passes
    filter with `if name in skip`. Neither side is wrong alone, and the bridge is
    `origin_of`, which only the composition root can apply because it is the only
    place holding both the problems and the registries.
    """
    return frozenset(n for n in registry.names() if registry.origin_of(n) in origins)


def _wire_executors(
    r: Registry, *, env: Any, strict_level: Any, config_order: Sequence[str]
) -> None:
    """`env_mgr` and `phase_runner` — the two names finding C1 found missing.

    `EnvManager` has no default: its constructor takes a `Context` that only the
    caller can supply, so an unpassed `env` leaves the name unregistered rather
    than guessing one.
    """
    if env is not None:
        r.register("env_mgr", env)
    parts = _optional({"PhaseRunner": "validator", "StrictLevel": "validator"})
    if "PhaseRunner" in parts:
        level = strict_level
        if level is None and "StrictLevel" in parts:
            level = parts["StrictLevel"].DEFAULT
        r.register("phase_runner", parts["PhaseRunner"](level))
    if config_order:
        r.register("config_order", tuple(config_order))


def _wire_monitors(r: Registry, *, monitors: Sequence[Any] | None) -> None:
    """`monitor:<name>`, plus the global budget and the thread excepthook.

    The excepthook is process-global and therefore nobody's constructor: an
    uncaught exception in a thread prints a traceback and vanishes — the process
    lives, the exit code is unchanged, and producers see nothing.
    """
    parts = _optional(
        {
            "Budget": "monitor",
            "PusherMonitor": "monitor",
            "DEFAULT_MONITOR_NAME": "monitor",
            "Recorder": "monitor",
            "monitor_for": "monitor",
        }
    )
    if "monitor_for" in parts:
        # Registered rather than imported by the scheduler, which may not reach
        # a sibling. `monitor` owns resolving `Task.monitor_spec` — the default
        # name, and the message naming the offending value — so the scheduler
        # asks for the resolver by name and this file never re-implements
        # `or DEFAULT_MONITOR_NAME`. **Not in `interfaces.md` §2.1's table**;
        # reported, because a registered name nobody wrote down is how two of
        # this week's defects started.
        r.register("monitor_for", parts["monitor_for"])
    if "Budget" in parts:
        r.register("budget", parts["Budget"]())
    if "Recorder" in parts:
        # `interfaces.md` §2.5: a registered row rather than a literal `...`.
        # Rev. 4 wrote `install_excepthook(recorder=..., sink=...)`, which meant
        # the root always had to build one and nothing did.
        r.register("recorder", parts["Recorder"](r.get("store_mgr")))
    chosen = monitors
    if chosen is None and {"PusherMonitor", "DEFAULT_MONITOR_NAME"} <= parts.keys():
        chosen = [parts["PusherMonitor"](parts["DEFAULT_MONITOR_NAME"], r)]
    for monitor in chosen or ():
        r.register(f"monitor:{monitor.name}", monitor)
    # `install_excepthook(recorder, sink)` is still NOT called here, and the
    # reason has changed. It is no longer that the arguments are unbuildable —
    # `recorder` is a row above and `NullUserSink` exists. It is that the hook is
    # **process-global**: `threading.excepthook` is one slot for the whole
    # interpreter, and a library function that installs one takes a decision
    # belonging to whoever owns the process. Two registries built in one test
    # session would fight over it. `interfaces.md` §5.9 is open on this; whoever
    # owns the entry point calls it with `r.get("recorder")`.


# --------------------------------------------------------------------------- #
# The guarded import. `interfaces.md` §4.7 says this package imports pydantic,
# and §2 hands it a composition root that constructs six other packages' types.
# Both cannot hold at module scope; deferring the import is what reconciles
# them, and it is reported rather than assumed to be what was meant.


def _optional(wanted: dict[str, str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for symbol, module in wanted.items():
        try:
            found[symbol] = getattr(__import__(module, fromlist=[symbol]), symbol)
        except (ImportError, AttributeError):
            log.debug("composition root: %s.%s is not available yet", module, symbol)
    return found


class _KindSource:
    """`hid -> HandoffKind`, which no single component can answer.

    `interfaces.md` §2.6. A store's `put` needs the kind for three things — the
    README's required sections, the `items` check against `items_schema`, and
    `Manifest.kind` — and the mapping has two halves living in two packages that
    do not import each other:

    | half | owner |
    |---|---|
    | `hid -> Handoff.type` | `task_graph.HandoffMgr.type_of` |
    | `type -> HandoffKind` | `handoff.HandoffSpecRegistry.kind_of` |

    Only the root holds both, which is why `handoff.store.KindSource` is a
    Protocol rather than a concrete type and why this class is here rather than
    in either package.

    **`None` is the honest answer for an unresolvable id, and it is not a
    fallback.** A store without a kind reads but does not publish: `put` raises
    and names the wiring, rather than checking no README sections, validating
    `items` against nothing, and recording `kind: ""` — which is what it did
    before it raised, with all 135 of `handoff`'s tests green because every one
    injects a resolver.
    """

    __slots__ = ("_r",)

    def __init__(self, registry: Registry) -> None:
        self._r = registry

    def kind_for(self, hid: Any) -> Any:
        type_name = self._r.get("handoff_mgr").type_of(hid)
        specs = self._r.get("handoff_specs")
        # `kind_of` raises `SpecNotFound` on an unknown name and the Protocol
        # asks for `None`, so the membership test is the conversion — not a
        # swallowed exception.
        return specs.kind_of(type_name) if type_name and type_name in specs else None


class RegistryViews:
    """A `Registries` — the read-only view over the five spec registries.

    The Protocol lives in `spec_loader` and no module in the set claims the
    concrete class. It is built here because the component `Registry` is the one
    object that holds all five, and resolving them at attribute access keeps the
    view honest if a test swaps one after wiring.
    """

    _KEYS = {
        "handoff": "handoff_specs",
        "validator": "validator_specs",
        "task": "task_specs",
        "agent": "agent_specs",
        "closure": "closures",
    }

    def __init__(self, registry: Registry) -> None:
        self._r = registry

    @property
    def handoff_specs(self) -> Any:
        return self._r.get("handoff_specs")

    @property
    def validator_specs(self) -> Any:
        return self._r.get("validator_specs")

    @property
    def task_specs(self) -> Any:
        return self._r.get("task_specs")

    @property
    def agent_specs(self) -> Any:
        return self._r.get("agent_specs")

    @property
    def closures(self) -> Any:
        return self._r.get("closures")

    def for_kind(self, kind: str) -> Any:
        try:
            key = self._KEYS[kind]
        except KeyError:
            raise KeyError(f"no spec registry for kind {kind!r}") from None
        return self._r.get(key)


def _catalogue(task_specs: Any) -> dict[str, Any]:
    return {name: task_specs.get(name) for name in task_specs.names()}

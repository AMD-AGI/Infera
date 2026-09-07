"""The load checks, and the pass that runs them.

Every check appends; none raises. A closure with a typo'd kind *and* a missing
agent should report both, for the reason main design §3.6 gives about the loader
generally: a loader that dies on the first bad spec makes fixing a package an
N-round trip.

Nothing here catches an exception to decide. `kind not in regs.handoff_specs` is
a membership test, not a `try: get() except SpecNotFound`. `SpecNotFound` exists
for a caller that wanted the spec; a checker that wanted the answer asks the
question.

**Where this pass runs is part of its design.** Not inside `load_package`: that
function runs once per package, so with two packages the pass would fire with the
second package's specs in no registry — and cross-package references are a
supported case. Dagster hit the same ordering problem and its fix has the same
shape, with the reason in a source comment: `# Late validate all jobs' resource
requirements are satisfied, since they may not be applied until now`. So it runs
at the composition root, once, after every package is loaded. That also fixes the
import: `spec_loader` may not import this package, and `bootstrap` already
imports everything.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, Protocol, runtime_checkable

from spec_loader import validator_agent_of
from spec_loader.protocols import ClosureDoc, Problem, Registries, SpecInconsistent, TaskSpec

from .model import (
    agent_of,
    body_of,
    declared_handoffs,
    grants_of,
    has_subgraph,
    named_inputs,
    named_kinds,
    named_outputs,
    phase_validators,
    subgraph_of,
    task_of,
)

__all__ = ["READ", "WRITE", "HandoffLoadReport", "check_closure", "check_closures", "covers"]

#: The two access levels a grant declares. These are the *values* of
#: `task_graph.Access`, restated as strings because this package may not import
#: `task_graph` — the same shape as `monitor`'s locally-declared `Pushable`, and
#: for the same reason. A grant is carried by `task_graph` and interpreted here.
READ = "read"
WRITE = "write"


@runtime_checkable
class HandoffLoadReport(Protocol):
    """`handoff`'s escape-hatch report, declared locally.

    Its real home is `handoff`, and this package may not import that one. The
    duplication is one attribute wide and is the price of the import rule; the
    Protocol is `runtime_checkable` so a caller passing the wrong report is a
    `TypeError` at the boundary rather than an `AttributeError` three frames in.

    Not to be confused with `spec_loader.LoadReport`, which is
    `(admitted, problems)`. Two dataclasses, one name, one shared field name,
    different meanings — both are in scope in this file.
    """

    admitted: Sequence[str]
    without_validator: Sequence[str]


# --------------------------------------------------------------------------- #
# Messages.
#
# Everyone clears "name both sides". The differentiator, from the survey, is a
# **computed repair** drawn from what is actually in scope — Dagster's model:
# "or change the required key to one of the following keys which points to an
# IOManagerDefinition: [...]". A notable negative from the same survey, recorded
# because it corrects an assumption this module would otherwise make: a tracker
# search on the Kubernetes coverage message found three issues and none
# complained it was unreadable. Message legibility is not where the pain is.
# Correctness of the check is.


def _candidates(registry: Any) -> list[str]:
    return list(registry.names())


def _hint(name: str, known: Sequence[str]) -> str:
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.7)
    return f"\n  hint: {close[0]!r} is close." if close else ""


def _known(known: Sequence[str]) -> str:
    return ", ".join(known) or "nothing"


def _problem(origin: str, path: str, keyword: str, message: str, *, fatal: bool = True) -> Problem:
    return Problem(origin=origin, path=path, keyword=keyword, message=message, fatal=fatal)


# --------------------------------------------------------------------------- #
# Check 6 — the covering relation.
#
# The sharpest finding in the survey is a failure this check is directly exposed
# to. Kubernetes' `Covers` wrongly rejects a legal delegation (kubernetes#122154)
# because `resourceNames` has no glob support while the CSR admission plugin had
# given `example.com/*` a meaning; the maintainer's reply was that the glob "is a
# semantic specific to the CSR admission plugin, it is not part of the
# authorization API or RBAC", followed by `/remove-kind bug`. They shipped the
# wrongness rather than open the grammar.
#
# So: the covering relation is ONE function, in THIS module, and it is TOTAL over
# the syntax the schema admits. No other component may interpret a grant in a way
# this function does not. `env_mgr` resolves paths to real locations and enforces
# containment; what it must not do is decide that some path *form* grants
# something this function thinks it does not.
#
# The alpha's *name* grammar is the smallest one that is total: exact string
# equality, no wildcards. That is Android lint's held-side model, and it is the
# only grammar with no ambiguous case. Kubernetes asserts the fail-closed
# direction with a named test — `TestCoversEnumerationNotCoveringVerbStar`, where
# an exhaustive enumeration on the grant side does not cover a `*` on the
# requirement side.
#
# **Access is a different axis, and there WRITE implies READ.** That is not a
# grammar extension; it is a two-element order on a closed enum, and it is what
# the enforcement layer does regardless of what is declared: `env_mgr` translates
# a grant into a Landlock `Mode`, and a write grant on a directory without read
# and execute is not usable, because a file cannot be created in a directory that
# cannot be traversed. `interfaces.md` §3.1 renamed the kernel-side type to `Mode`
# for exactly this reason — `Access` is what an author *declared*, `Mode` is what
# the kernel gets, and `READ_EXEC` has no declaration-side meaning.
#
# **This relation exists twice and the duplication is forced.** `task_graph`
# ships `Permissions.covers` over pydantic objects, and its docstring names this
# check as the reason it exists. This one runs at load, where there is no `Task`
# and a task spec is a `Mapping` — building a `Permissions` from it would mean
# importing `task_graph`, which §4.5 forbids. So one relation is obliged to have
# two bodies, one over objects and one over raw dicts. Rev. 1 of this module
# matched access exactly and the two disagreed silently for a day.
# `tests/interfaces/test_covers_agreement.py` is the price of the duplication
# and the only thing that keeps them honest.


def covers(task: TaskSpec, kind: str, access: str) -> bool:
    """Whether this task's declared permissions grant `access` on `kind`.

    Exact string equality on the kind name; **`WRITE` covers a `READ`
    requirement**, and the reverse does not hold. Byte-for-byte the answer
    `task_graph.Permissions.covers` gives, and a test asserts that over every
    pair.

    Adding a wildcard to the *name* grammar is a change to this function and to
    the schema that admits its syntax, in that order — and to nothing else.
    """
    for grant in grants_of(task):
        if grant.get("kind") != kind:
            continue
        held = grant.get("access", READ)
        if held == access or held == WRITE:
            return True
    return False


def _held(task: TaskSpec) -> str:
    grants = [f"{g.get('kind')}({g.get('access', READ)})" for g in grants_of(task) if g.get("kind")]
    return ", ".join(grants) or "none"


def check_permissions_cover(task: TaskSpec, origin: str, closure: str) -> list[Problem]:
    """Check 6 — the one only a closure can perform.

    It needs the task's handoffs and its permissions together, and neither
    registry sees both. It reads nothing outside this document, which is why it
    runs last and takes no `Registries`: a closure whose kind does not resolve
    still gets a coverage message, because the grant is about a *name* the author
    wrote and can fix whether or not the kind exists.

    With exact-equality covering, the decision atom is `(kind, access)` and the
    message atom is the same one, so there is nothing to decompose for the
    machine and nothing to re-compact for the human. Kubernetes needs both forms
    and added the re-compaction later, from a `TODO`; that stops being free the
    moment a wildcard arrives.
    """
    problems: list[Problem] = []
    for kind, access, role in (
        *((k, READ, "consumes") for k in named_inputs(task)),
        *((k, WRITE, "produces") for k in named_outputs(task)),
    ):
        if covers(task, kind, access):
            continue
        problems.append(
            _problem(
                origin,
                "$.task.permissions.grants",
                "covers",
                f"closure {closure!r} {role} handoff {kind!r} but its permissions "
                f"grant no {access} for it.\n"
                f"  grants held: {_held(task)}\n"
                f"  hint: add a {access} grant for {kind!r}, or remove it from the "
                f"task's {'inputs' if access == READ else 'outputs'}.",
            )
        )
    return problems


# --------------------------------------------------------------------------- #
# Check 7 — the body.


def check_body(task: TaskSpec, origin: str, closure: str) -> list[Problem]:
    """Existence of the declaration, never its content.

    What a script does and whether a readme is any good are not load-time
    questions. **Nor is whether the paths resolve on disk**: this pass is handed
    registries and an opaque origin label, and resolving a body path would need
    the package root — which is deliberately unreachable from here, because the
    loader's no-path property is structural rather than disciplinary. That half
    of the design's check 7 belongs to whoever holds the `TaskPackage`; see
    `README.md`.
    """
    problems: list[Problem] = []
    body = body_of(task)

    readme = body.get("readme")
    if not isinstance(readme, str) or not readme:
        problems.append(
            _problem(
                origin,
                "$.task.body.readme",
                "required",
                f"closure {closure!r} declares no `readme.md` for its task. "
                f"Required of every task, leaf or not: a task nobody can read is "
                f"a step nobody can review, and that holds for a non-leaf too.",
            )
        )

    if body.get("entry") and has_subgraph(task):
        problems.append(
            _problem(
                origin,
                "$.task.body.entry",
                "oneOf",
                f"closure {closure!r} declares both an `entry.sh` and a subgraph, "
                f"and they are mutually exclusive: a task contains a task graph, "
                f"or it is a leaf that does the work itself. A non-leaf's work "
                f"*is* its subgraph.\n"
                f"  hint: `readme.md` is required either way — the exclusion is "
                f"between `entry.sh` and the subgraph, not between `body` and it.",
            )
        )
    return problems


# --------------------------------------------------------------------------- #
# Check 8 — a subgraph entry names a declared closure.


def check_subgraph_targets(
    task: TaskSpec, regs: Registries, origin: str, closure: str
) -> list[Problem]:
    """Every closure a subgraph entry names must be in the catalogue.

    **Asked for by `task_graph`, and it is genuinely this pass's rather than
    theirs**: the question is cross-registry, and `check_graph` walks task specs
    without holding `regs.closures`. Reaching the closures registry from
    `build_registry` and handing it in would put a second cross-registry question
    into a function whose whole argument for existing is that it only walks task
    specs.

    **`Task.unfold` raises on the same fault and stays.** Not redundancy: it
    catches the case no load-time pass can see, because `replace_with`
    instantiates a closure named at run time by a monitor's decision
    (`task_graph` criterion 51). What this buys is that the *declared* case never
    reaches it — a subgraph nested three deep would otherwise fail hours into a
    run, after work has been done and paid for, one typo at a time instead of
    alongside every other reason the graph is not admissible.

    The message enumerates the catalogue, which `task_graph`'s raise also does
    and which is the difference between "no such closure" and "you wrote
    `collect_trace` and the catalogue has `collect_traces`".
    """
    problems: list[Problem] = []
    known = _candidates(regs.closures)
    for index, entry in enumerate(subgraph_of(task)):
        target = entry.get("closure") if isinstance(entry, Mapping) else None
        if not isinstance(target, str) or not target:
            problems.append(
                _problem(
                    origin,
                    f"$.task.subgraph[{index}]",
                    "required",
                    f"closure {closure!r} declares a subgraph entry naming no "
                    f"closure. An entry is `{{closure, froms, is_start?, is_end?}}`, "
                    f"and the closure name is what `unfold` instantiates.",
                )
            )
            continue
        if target not in regs.closures:
            problems.append(
                _problem(
                    origin,
                    f"$.task.subgraph[{index}]",
                    "resolves",
                    f"closure {closure!r} names subtask closure {target!r}, which "
                    f"is not declared.\n"
                    f"  known closures: {_known(known)}"
                    f"{_hint(target, known)}",
                )
            )
    return problems


# --------------------------------------------------------------------------- #
# The seven checks over one closure. The order is the design, so it is a body.


def check_closure(
    doc: ClosureDoc,
    regs: Registries,
    *,
    origin: str,
    name: str,
    handoff_report: HandoffLoadReport,
) -> list[Problem]:
    """Checks 2-7 over one closure. Check 1 is the schema, done before this pass.

    `handoff_report` is required here for the same reason it is required on the
    pass: a default would let a caller reach check 3 with nothing to intersect and
    get silence back.
    """
    problems: list[Problem] = []
    task = task_of(doc)

    # 2. Resolution first, and declaration second. Both failures are about one
    #    kind name and the author's next action differs: an unresolved kind means
    #    "you typed it wrong, or the file is missing", while an undeclared one
    #    means "add it to `handoffs`". Reporting the second when the first is true
    #    sends them to the wrong file.
    #
    #    The check is one-directional. A declared kind the task does not name is
    #    LEGAL, because a closure may declare a kind its subgraph uses internally.
    declared = set(declared_handoffs(doc))
    known_kinds = _candidates(regs.handoff_specs)
    for kind in named_kinds(task):
        if kind not in regs.handoff_specs:
            problems.append(
                _problem(
                    origin,
                    "$.task",
                    "resolves",
                    f"closure {name!r} names handoff kind {kind!r}, which does not "
                    f"resolve.\n"
                    f"  known kinds: {_known(known_kinds)}"
                    f"{_hint(kind, known_kinds)}",
                )
            )
        elif kind not in declared:
            problems.append(
                _problem(
                    origin,
                    "$.handoffs",
                    "declared",
                    f"closure {name!r} names handoff kind {kind!r} on its task but "
                    f"does not declare it.\n"
                    f"  declared: {_known(sorted(declared))}\n"
                    f"  hint: add {kind!r} to the closure's `handoffs`.",
                )
            )

    # 3. Only over kinds that resolved. An escape-hatch admission is a REPORT,
    #    not a failure: a kind with no validator is unadmittable in the first
    #    place, so anything in `without_validator` is already known and already
    #    permitted. This does not re-derive the coverage.
    problems += _escape_hatch_report(task, handoff_report, origin, name)

    # 4. Absent and present-and-wrong are both errors, and they are different
    #    errors. The schema's `required` catches the first for any document that
    #    reached it; this catches it for any document that did not.
    #
    #    **Absent is only an error for a leaf** — main spec §4.8 narrowed at
    #    rev. 10, and `closure.schema.json` with it: `agent` left `required` and
    #    an if/else reinstates it unless `task.subgraph` is present and
    #    non-empty. `has_subgraph` is `bool(subgraph_of(task))`, which is that
    #    condition exactly, so the two gates cannot drift apart by rewording.
    #    Measured before narrowing: this branch fired on an agent-less non-leaf
    #    that the schema had already admitted, so the schema and the load check
    #    disagreed and the ruling was unreachable in practice
    #    (`scratch/ui-yaml-2026-08/w5/probe_agentless_nonleaf_load_check.py`).
    #
    #    Present-and-wrong does **not** narrow. An author may still name one on
    #    a non-leaf, and a name that does not resolve is still a typo.
    agent = agent_of(doc)
    known_agents = _candidates(regs.agent_specs)
    if not agent and not has_subgraph(task):
        problems.append(
            _problem(
                origin,
                "$.agent",
                "required",
                f"closure {name!r} names no agent spec and declares no subgraph, "
                f"so it is a leaf and something has to run it. What varies is the "
                f"spec's `kind` — `ai`, `human`, or `program`. A plain executable "
                f"is a `kind: program` spec, which costs a name and a command line "
                f"and buys one dispatch path and one answer to 'what ran this "
                f"task'.\n"
                f"  a non-leaf needs none of this: its work is its subgraph, and "
                f"the system supplies the executor.\n"
                f"  known agent specs: {_known(known_agents)}",
            )
        )
    elif agent and agent not in regs.agent_specs:
        # `agent and` is load-bearing now that the branch above narrowed. It was
        # unreachable on a falsy `agent` while the first branch caught every
        # one; with the non-leaf case falling through, an absent key reaches
        # here as `agent_of`'s `""` and reported "names agent spec '', which
        # does not resolve". Measured, not foreseen.
        problems.append(
            _problem(
                origin,
                "$.agent",
                "resolves",
                f"closure {name!r} names agent spec {agent!r}, which does not "
                f"resolve.\n"
                f"  known agent specs: {_known(known_agents)}"
                f"{_hint(agent, known_agents)}",
            )
        )

    # 5. Resolves, then is-the-right-kind. The second message is only reachable
    #    when the first passed, which is what makes it specific.
    known_validators = _candidates(regs.validator_specs)
    for validator in phase_validators(doc):
        if validator in regs.validator_specs:
            continue
        if validator in regs.task_specs:
            problems.append(
                _problem(
                    origin,
                    "$.validators",
                    "kind",
                    f"closure {name!r} names {validator!r} as a phase validator, "
                    f"but {validator!r} is a general task, not a validator.\n"
                    f"  known validator specs: {_known(known_validators)}",
                )
            )
        else:
            problems.append(
                _problem(
                    origin,
                    "$.validators",
                    "resolves",
                    f"closure {name!r} names phase validator {validator!r}, which "
                    f"does not resolve.\n"
                    f"  known validator specs: {_known(known_validators)}"
                    f"{_hint(validator, known_validators)}",
                )
            )

    # 6 and 7. The two checks that read nothing outside this document.
    problems += check_permissions_cover(task, origin, name)
    problems += check_body(task, origin, name)

    # 8. Cross-registry again, and last because it is the newest and the least
    #    load-bearing: `Task.unfold` catches the same fault at run time, and this
    #    only moves the declared case to load.
    problems += check_subgraph_targets(task, regs, origin, name)
    return problems


def _escape_hatch_report(
    task: TaskSpec,
    handoff_report: HandoffLoadReport,
    origin: str,
    name: str,
) -> list[Problem]:
    # No `if handoff_report is None: return []`. The absent case is refused at
    # the entry point, loudly, and this function is reached only with a report —
    # "no escape-hatch admissions" is an empty `without_validator`, and "nobody
    # told me" is not representable.
    without = set(handoff_report.without_validator)
    hit = [kind for kind in named_kinds(task) if kind in without]
    if not hit:
        return []
    return [
        _problem(
            origin,
            "$.handoffs",
            "escape_hatch",
            f"closure {name!r} is assembled from handoff "
            f"{'kinds' if len(hit) > 1 else 'kind'} {', '.join(repr(k) for k in hit)}, "
            f"admitted under the no-validator escape-hatch flag. The closure loads; "
            f"nothing it produces is checked by a validator of its own kind.",
            fatal=False,
        )
    ]


def _bind_phase_validators(regs: Registries, names: Sequence[str]) -> None:
    """Tell `validator_specs` which closures name each phase validator.

    **`docs/interfaces.md` §5.4's third edge kind, and the one `users_of` could
    not see.** A closure's phase validators are a property of the task, so the
    handoff specs cannot carry them — a `users_of` counting only handoff-kind
    bindings reports a validator two closures run in every output phase as used
    by nothing. That is Airflow #58058's false-negative deadness, and dbt#14436 is
    a second independent instance.

    **Here rather than at the composition root**, and the argument is
    `_admit_task_specs`': the root calling it would mean `bootstrap` reading
    `phase_validators` off a closure document, which is exactly the leaked
    knowledge §4.5 keeps out of this package's neighbours — and it would duplicate
    the loop this pass already runs with `skip` applied, so the layering gate
    comes free here and would have to be restated there.

    **Called directly, not through `getattr`.** `bind_phase` is not on the
    `SpecRegistry` Protocol, so a five-dict stub must grow it — and that is the
    point. A `getattr(..., None)` here would make a missing binding silent in the
    assembled system, which is the shape that hid `handoff`'s `load_report`
    mismatch and would hide this one as *under-reported `users_of`* rather than as
    an error. `closure` design D4 chose a query over a silent wrong answer once
    already; this is the same choice one level out.

    It records the edge as *declared*, not as resolved. Whether the name resolves
    is check 5's question and is reported there; "this closure names X" is true of
    the document either way, and a `users_of` that hid unresolvable users would be
    answering a different question from the one it is asked.
    """
    bind_phase = regs.validator_specs.bind_phase
    for name in names:
        bind_phase(name, phase_validators(regs.closures.get(name)))


def _check_validator_agents(regs: Registries) -> list[Problem]:
    """Every validator that names an agent spec must name one that resolves.

    **`validator` §8.2 row 1**: a validator bound to a real agent takes that
    agent's declared `env`. The name is optional, and absent is the ordinary case
    — it takes the global row and is no fault at all.

    **Present-and-unresolvable is fatal, and the reason is not consistency with
    check 4.** `validator` put the wrinkle well and then caught themselves on it:
    a validator whose agent does not resolve still has a working global-row
    environment, so it is *not* unusable the way a task with no agent is. True —
    and that is an argument about the **absent** case. An author who wrote
    `profilr` wanted row 1 and would get row 4, silently, with a working
    environment that is not the one they configured. The symptom is a validator
    that **runs**, in the wrong environment, producing a verdict somebody
    trusts — which is worse than most of this family, because a validator's whole
    job is to be the thing you believe.

    **A whole-catalogue pass rather than a row in the per-closure loop.** The
    fault is a property of the *validator spec*, not of any closure that names
    it, and three consequences follow. It is reported once rather than once per
    naming closure. It is keyed to `validator_specs.origin_of(name)` — the file
    the author has to open — rather than to a closure's origin, which would be
    the wrong file. And it catches a validator bound only to a handoff kind,
    which no closure names and which the per-closure form would miss entirely
    while having exactly the same defect.

    **`spec_loader.validator_agent_of`, not `spec["agent"]`.** The key belongs to
    `validator`'s document, and this is the first check here whose *document* is a
    third party's. It is in the leaf because `closure` may not import `validator`
    — measured, not assumed — and it is named `validator_agent_of` rather than
    `agent_of` because a leaf exporting two accessors of one name over two
    document types is a shadowing hazard nobody downstream could alias around.
    """
    problems: list[Problem] = []
    known = _candidates(regs.agent_specs)
    for name in sorted(regs.validator_specs.names()):
        agent = validator_agent_of(regs.validator_specs.get(name))
        if agent is None or agent in regs.agent_specs:
            continue
        problems.append(
            _problem(
                regs.validator_specs.origin_of(name),
                "$.agent",
                "resolves",
                f"validator {name!r} names agent spec {agent!r}, which does not "
                f"resolve. It would fall back to the global environment — a "
                f"working one that is not the one this validator was configured "
                f"with, and nothing would say so at run time.\n"
                f"  known agent specs: {_known(known)}"
                f"{_hint(agent, known)}",
            )
        )
    return problems


# --------------------------------------------------------------------------- #
# The pass.


def check_closures(
    regs: Registries,
    handoff_report: HandoffLoadReport,
    *,
    skip: AbstractSet[str] = frozenset(),
) -> list[Problem]:
    """Every closure in `regs.closures`, in sorted name order, except those in
    `skip`. Returns problems; **raises only on a wiring fault** — see below.

    **Sorted name order**, so a package with two broken closures reports them the
    same way twice — the determinism rule OPA states as `util.KeysSorted`.

    `skip` is the layering gate: a closure whose own spec already failed is not
    checked again, because "your task's handoff kind does not resolve" on top of
    "your schema is broken" is noise. Kubernetes CRD validation does exactly
    this, and gives the reason — CEL validation error messages that are not
    actionable.

    `handoff_report` is a fact about *this* load, while the five registries
    outlive it, which is why it is a parameter rather than a field on
    `Registries`. It is also typed in `handoff`, and this package may not import
    that one.

    **It is required, it has no `None` default, and a `None` reaching it raises.**
    Rev. 1 wrote `handoff_report: HandoffLoadReport | None = None` and returned
    early on `None`, and that default is what hid a real defect for as long as it
    was hidden: the composition root reached for an accessor `handoff` had not
    named yet, `getattr(..., lambda: None)()` produced `None`, check 3 returned
    early, and an escape-hatch admission went unreported **in the assembled
    system** while all three packages' suites stayed green. `load_report()` can no
    longer return `None`, so a `None` here is a wiring fault and nothing else.
    `docs/interfaces.md` §4.11.

    **It is the closure pass, and it does four things in one call.** It admits
    each closure's nested task spec, records the closure→phase-validator edge
    with `validator_specs`, runs the checks, and builds the reverse index. They
    are one function because a caller that had to remember to also call three
    others would eventually not, and because this is the only moment at which the
    five registries and the whole closure catalogue are both in hand — `freeze()`
    takes no argument, and nobody else may key a task spec or see that edge.
    """
    if handoff_report is None:
        raise TypeError(
            "check_closures(handoff_report=None): the composition root always "
            "supplies one — `handoff_specs.load_report()` never returns None — so "
            "a None here means the registry registered under 'handoff_specs' is "
            "not a HandoffSpecRegistry, or the root reached for an accessor that "
            "does not exist. Silently skipping check 3 is what hid exactly this "
            "once already (docs/interfaces.md §4.11)."
        )

    problems: list[Problem] = []
    closures = regs.closures

    # **`origin_of` and `_build_index` are called directly, and the `getattr`
    # guards that used to be here are gone.** Neither is on the `SpecRegistry`
    # Protocol — `origin_of` is on the shared registry base, `_build_index` is
    # intra-package — so a guard looked prudent. It was not.
    #
    # Measured before removing them: no test in this package ever supplied a
    # `closures` lacking either, so both fallbacks were **dead**. And they were
    # not harmless dead code. `build_registry` now takes `registries=` from the
    # caller, so a `closures` without `origin_of` became reachable — and the
    # fallback silently labelled every `Problem` with the closure's *name* where
    # a file path belongs, which is indistinguishable from a real origin in a
    # message and is the one thing `docs/design.md` §6.2 asks these messages to
    # carry. A degradation nothing raises on, which is the shape this package
    # spent a day removing from other people's code and had two of its own.
    #
    # So they are obligations now. The five-dict stub carries `origin_of`, the
    # same way it carries `bind_phase` and for the same reason.
    origin_for = closures.origin_of
    admitted = [name for name in sorted(closures.names()) if name not in skip]

    # 1. Admit the nested task specs, as a pass of their own and before any
    #    check. Check 5 asks whether a phase validator name is a *general task*,
    #    and that question has no answer until every task spec is present — with
    #    one loop it would depend on where the closure sorted.
    problems += _admit_task_specs(regs, admitted, origin_for)

    # 1b. Record the third edge kind, for the same reason and in the same place.
    _bind_phase_validators(regs, admitted)

    # 1c. The validator catalogue, which is not per-closure — see the docstring.
    problems += _check_validator_agents(regs)

    # 2. Check.
    for name in admitted:
        problems += check_closure(
            closures.get(name),
            regs,
            origin=origin_for(name),
            name=name,
            handoff_report=handoff_report,
        )

    # 3. Index, over every closure the pass saw.
    closures._build_index(regs)
    return problems


def _admit_task_specs(
    regs: Registries,
    names: Sequence[str],
    origin_for: Callable[[str], str],
) -> list[Problem]:
    """Key each closure's nested task spec into `task_specs`, under the closure's
    name.

    **Nothing else can do this, and until it was done nothing did.** A task spec
    is nested inside its closure and carries no `name`, so `spec_loader` does not
    discover one — `task` is deliberately not a discoverable kind. The
    consequence, found by `task_graph`: `check_graph` walked an empty catalogue,
    so criteria 50 and 53 were green and inert. This package owns both registries
    and the nesting, so the write is here, in the pass the composition root
    already calls at the only correct moment.

    The two registries share a key space by decision (`task_registry.py`), and
    this is the call that makes that true rather than merely stated.

    `SpecInconsistent` becomes a `Problem` rather than propagating, because the
    pass raises nothing. It is reachable: the shared base rejects the *reverse*
    collision — one spec admitted under two names — so two closures whose tasks
    are byte-identical land here. That is design O6's "a task spec reused by two
    closures is inexpressible", surfacing as a message instead of a silent
    second key.
    """
    problems: list[Problem] = []
    for name in names:
        origin = origin_for(name)
        try:
            regs.task_specs.add(name, task_of(regs.closures.get(name)), origin=origin)
        except SpecInconsistent as clash:
            problems.append(
                _problem(
                    origin,
                    "$.task",
                    "duplicate",
                    f"closure {name!r} cannot key its task spec: {clash}\n"
                    f"  hint: a task spec is registered under its closure's name, so "
                    f"two closures cannot share one task. Give them different tasks, "
                    f"or make one closure of them.",
                )
            )
    return problems

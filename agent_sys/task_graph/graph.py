"""The graph-level load checks.

Two criteria say "at load", and `task_graph` has no load step: `submit` is the
entry point and it sees one task at a time. Neither neighbour will take these —
`closure` spec §4.1 defers graph-level checks to nobody and main design §6.3
declines them explicitly — so this module claims them, on the argument that the
graph is this module's subject.

They run over task **specs**, not over `Task` objects, and that is forced: the
catalogue is static while `submit` accepts tasks at any time, so a runtime pass
could never see a complete graph. The composition root loads every package and
*then* registers the scheduler, which is the only moment when every spec is
present and nothing has run.

`Problem` comes from `spec_loader`, not from a type of this module's own: one
report format across every load-time check is the rule, and a second shape here
would make the whole-run error output depend on which pass found the fault.

Check 3 arrived later and from another package's specification — `closure`
criterion 12, `froms` — and its home is argued at `_check_subgraph_froms` rather
than here, because the argument is about that check and not about this file.
"""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

from spec_loader.protocols import Problem, TaskSpec
from task_graph.models import DerivedEdge, SubgraphEntry, derived_edges, subgraph_entries

__all__ = ["check_graph"]


def check_graph(
    specs: Mapping[str, TaskSpec],
    *,
    skip: AbstractSet[str] = frozenset(),
) -> list[Problem]:
    """Every graph-level check over the declared catalogue.

    Returns problems and raises nothing, so a catalogue with two faults reports
    both. `skip` is the layering gate: a closure whose own spec already failed is
    not walked again, because its handoff names may not resolve and a second
    error on top of a broken schema is noise.
    """
    problems: list[Problem] = []
    names = [name for name in sorted(specs) if name not in skip]
    for name in names:
        problems += _check_leaf_only_acquisition(name, specs[name])
        problems += _check_subgraph_containment(name, specs[name], specs, skip)
        problems += _check_subgraph_froms(name, specs[name], specs)
    return problems


def _check_leaf_only_acquisition(name: str, spec: TaskSpec) -> list[Problem]:
    """Check 1 — a task declaring a subgraph declares no `resources`.

    A parent holding a lease while its subtasks queue for the same pool is
    hold-and-wait, and two tasks in one graph competing for one pool deadlock on
    it. Airflow shipped exactly this as `SubDagOperator`, diagnosed it, and
    deleted it; its own load-time guard filtered only single-slot pools, so a
    two-slot pool with a parent holding one and a child needing two parsed
    clean. This forbids the class rather than an instance of it.
    """
    entries = subgraph_entries(spec)
    resources = spec.get("resources") or {}
    if not entries or not resources:
        return []
    return [
        Problem(
            origin=name,
            path="$.resources",
            keyword="leaf_only_acquisition",
            message=(
                f"task {name!r} declares resources {dict(resources)!r} and expands into a "
                f"subgraph of {len(entries)} tasks. Only a leaf may acquire; a parent holding "
                f"a lease while its subtasks queue for the same pool is the deadlock this rule "
                f"prevents. Move the declaration onto the subtasks that do the work."
            ),
        )
    ]


def _check_subgraph_containment(
    name: str,
    spec: TaskSpec,
    specs: Mapping[str, TaskSpec],
    skip: AbstractSet[str],
) -> list[Problem]:
    """Check 2 — no handoff produced inside a subgraph is consumed outside it.

    Except through the end entry subtask's outputs, which is the declared
    boundary. The violation is representable here only because handoff ids are
    one flat namespace: everywhere else containment is enforced by *scope*, so
    the inner name simply does not exist outside and the mistake cannot be
    written. Scoping ids to a subgraph is the alternative with precedent and is
    a specification question.

    **"Outside" is transitive; "produced inside" and "exported" are not.** The
    asymmetry is deliberate and each half of it is forced by a different fact.

    - `inside` must be the whole descendant set — the direct entries, plus the
      entries of *their* subgraphs, recursively. A grandchild is strictly inside
      its grandparent's subgraph, so consuming a kind produced there blocks
      nobody: the message this check would print is its own refutation.
      "Cancelling inside `main` would silently block `review_x`" is false when
      `review_x` is reached through `main` -> `grade` -> `review_x`, because
      cancelling inside `main` cancels `grade` and `grade` cancels `review_x`.
      There is no outside observer to protect. A closure catalogue may name an
      ancestor, so the walk carries a visited set and a cycle terminates.
    - `produced_inside` and `exported` stay one level deep, because
      **every nesting level runs its own check**: `check_graph` calls this for
      each spec in the catalogue, so a kind produced by a grandchild and consumed
      by a genuine outsider is caught when the check runs on the *child*'s own
      subgraph, where that grandchild is a direct entry. Widening the producer
      set too would report one fault twice, from two parents, naming two
      different subgraphs as its origin — and the shallower report is the less
      useful one, since the boundary the author must export through is the
      child's end entry, not the grandparent's.
    """
    entries = subgraph_entries(spec)
    if not entries:
        return []

    inside = _descendants(entries, specs)
    produced_inside: dict[str, str] = {}
    exported: set[str] = set()
    for entry in entries:
        member = specs.get(entry.closure)
        if member is None:
            continue
        for kind in _kinds(member, "outputs"):
            produced_inside.setdefault(kind, entry.closure)
            if entry.is_end:
                exported.add(kind)

    problems: list[Problem] = []
    for other in sorted(specs):
        if other in skip or other == name or other in inside:
            continue
        for kind in _kinds(specs[other], "inputs"):
            producer = produced_inside.get(kind)
            if producer is None or kind in exported:
                continue
            problems.append(
                Problem(
                    origin=other,
                    path="$.inputs",
                    keyword="subgraph_containment",
                    message=(
                        f"task {other!r} consumes handoff kind {kind!r}, which is produced by "
                        f"{producer!r} inside {name!r}'s subgraph and is not exported through "
                        f"its end entry subtask. Cancelling inside {name!r} would silently "
                        f"block {other!r}. Export the kind from the end entry subtask, or "
                        f"produce it outside the subgraph."
                    ),
                )
            )
    return problems


def _descendants(
    entries: Sequence[SubgraphEntry],
    specs: Mapping[str, TaskSpec],
) -> set[str]:
    """Every closure strictly inside a subgraph, at any depth.

    A closure whose spec is absent from the catalogue contributes itself and
    stops: `check_graph`'s `skip` gate and an unresolved name are both ordinary,
    and a missing member has its own problem reported elsewhere.

    The membership test doubles as the visited set, so a catalogue in which `a`'s
    subgraph names `b` and `b`'s names `a` terminates instead of recursing for
    ever. Nothing forbids that catalogue at this point in the load — the checks
    that would reject it run per-spec and report rather than raise, so this one
    has to survive being handed it.
    """
    inside: set[str] = set()
    pending = [entry.closure for entry in entries]
    while pending:
        closure = pending.pop()
        if closure in inside:
            continue
        inside.add(closure)
        member = specs.get(closure)
        if member is not None:
            pending += [entry.closure for entry in subgraph_entries(member)]
    return inside


def _check_subgraph_froms(
    name: str,
    spec: TaskSpec,
    specs: Mapping[str, TaskSpec],
) -> list[Problem]:
    """Check 3 — `froms` agrees with the derived edges, and points backwards.

    **Why here and not in `closure/check.py`.** The fact is about the *graph* a
    subgraph declares, and this module's docstring already claims that ground.
    Two things make the choice forced rather than a preference:

    - The cross-check has to *run* the derivation, and the derivation is
      `task_graph`'s (`models.derived_edges`, called by `Task._instantiate`).
      `closure` may not import this package — `closure/check.py:50` says so in as
      many words — so a check hosted there would have to re-implement the
      producer walk. That is `engineer_principle.md` §3's exact failure mode and
      §1's "never let an invariant have two writers": when the derivation
      changes, the copy in the checker is the one nobody updates, and it fails by
      going *quiet*, which is the worst available failure for a checker.
    - It needs task specs and nothing else. `closure.check_subgraph_targets` is
      the near neighbour and is the counter-example that fixes the rule: its own
      docstring justifies living there because the question is *cross-registry*
      and `check_graph` does not hold `regs.closures`. This question is the
      opposite — it holds only task specs, and `check_closure` is handed one
      document at a time and cannot see the members' specs at all.

    **What is checked, and in which direction.**

    | | |
    |---|---|
    | a derived edge `froms` does not declare | **error** |
    | a `froms` edge no handoff derives | **reported, not fatal** |
    | a name that is not an entry of this subgraph | **error** |
    | a name that is not an *earlier* entry | **error** — the topological rule |

    The second row is where `closure` spec §2.7 asks for something it cannot
    have. It calls the check "two-directional", then names the exception that
    makes the second direction unenforceable: `froms` exists so that *a
    dependency that shares no handoff* can be written down, and such an edge is
    by definition derived by nothing. So the two mistakes §2.7 wants caught are
    not symmetric. "Wired a handoff and did not notice an edge appeared" is
    derived-and-undeclared, and is decidable — it is an error. "Removed a handoff
    and did not notice an edge vanished" is declared-and underived, and is
    **indistinguishable** from the feature; rejecting it would delete the
    feature, so it is reported non-fatally and the message states both readings.

    **The topological rule costs one index comparison, and only `froms` can
    break it.** `derived_edges` walks the list in order and only ever links to an
    earlier producer, so a derived edge points backwards by construction; a
    declared one is the only kind that can point forward. That also means no
    cycle is representable once this check passes — see `README.md` for why
    `graphlib.TopologicalSorter` is therefore not adopted.
    """
    entries = subgraph_entries(spec)
    if not entries:
        return []

    duplicated = sorted({e.closure for e in entries} & _repeats(entries))
    if duplicated:
        # Reported and then abandoned. `froms` names entries by their `closure`,
        # so with a name used twice there is no fact of the matter about which
        # entry an edge points at, and every check below would be guessing. One
        # honest problem beats a page of them derived from a guess.
        return [
            Problem(
                origin=name,
                path="$.subgraph",
                keyword="froms_ambiguous",
                message=(
                    f"task {name!r} lists {_names(duplicated)} more than once in its "
                    f"subgraph. `froms` names an entry by its `closure`, so a repeated "
                    f"name makes every edge touching it ambiguous and the cross-check "
                    f"against the derived edges cannot run. Give each entry a distinct "
                    f"closure, or make one closure of the repeats."
                ),
            )
        ]

    index_of = {entry.closure: i for i, entry in enumerate(entries)}
    edges = derived_edges(entries, specs)
    problems: list[Problem] = []
    for i, entry in enumerate(entries):
        problems += _check_froms_resolve(name, entries, entry, i, index_of)
        problems += _check_froms_covers_derived(name, entries, entry, i, edges[i])
        problems += _check_froms_beyond_derived(name, entries, entry, i, edges[i])
    problems += _check_marks_agree(name, entries)
    return problems


def _check_froms_resolve(
    name: str,
    entries: Sequence[SubgraphEntry],
    entry: SubgraphEntry,
    i: int,
    index_of: Mapping[str, int],
) -> list[Problem]:
    problems: list[Problem] = []
    for declared in entry.froms:
        target = index_of.get(declared)
        if target is None:
            problems.append(
                Problem(
                    origin=name,
                    path=f"$.subgraph[{i}].froms",
                    keyword="froms_resolves",
                    message=(
                        f"task {name!r} subgraph entry {i} ({entry.closure!r}) declares "
                        f"froms {declared!r}, which is not an entry of this subgraph.\n"
                        f"  entries: {_names([e.closure for e in entries])}"
                    ),
                )
            )
        elif target >= i:
            problems.append(
                Problem(
                    origin=name,
                    path=f"$.subgraph[{i}].froms",
                    keyword="froms_order",
                    message=(
                        f"task {name!r} subgraph entry {i} ({entry.closure!r}) depends on "
                        f"{declared!r}, which is entry {target} — "
                        + (
                            "itself.\n"
                            if target == i
                            else f"later in the list.\n"
                            f"  the edge {declared!r} -> {entry.closure!r} points backwards "
                            f"against the listing, so the listing is not a topological "
                            f"order.\n"
                        )
                        + f"  move {entry.closure!r} after {declared!r}."
                    ),
                )
            )
    return problems


def _check_froms_covers_derived(
    name: str,
    entries: Sequence[SubgraphEntry],
    entry: SubgraphEntry,
    i: int,
    edges: Sequence[DerivedEdge],
) -> list[Problem]:
    declared = set(entry.froms)
    problems: list[Problem] = []
    for edge in edges:
        producer = entries[edge.producer_index].closure
        if producer in declared:
            continue
        problems.append(
            Problem(
                origin=name,
                path=f"$.subgraph[{i}].froms",
                keyword="froms_mismatch",
                message=(
                    f"task {name!r} subgraph entry {i} ({entry.closure!r}) consumes handoff "
                    f"kind {edge.kind!r}, which entry {edge.producer_index} ({producer!r}) produces, "
                    f"so an edge {producer!r} -> {entry.closure!r} exists that `froms` does "
                    f"not declare.\n"
                    f"  declared: {_names(entry.froms)}\n"
                    f"  add {producer!r} to entry {i}'s froms, or stop consuming "
                    f"{edge.kind!r}."
                ),
            )
        )
    return problems


def _check_froms_beyond_derived(
    name: str,
    entries: Sequence[SubgraphEntry],
    entry: SubgraphEntry,
    i: int,
    edges: Sequence[DerivedEdge],
) -> list[Problem]:
    """The non-fatal half. See `_check_subgraph_froms` for why it cannot be an
    error: an edge no handoff supports is the one thing `froms` buys.

    Only names that already resolved to an earlier entry are considered. A name
    that resolves to nothing, or forwards, has its own error, and a second
    problem on top of it is the noise `skip` exists to avoid one level up.
    """
    index_of = {e.closure: n for n, e in enumerate(entries)}
    derived = {entries[edge.producer_index].closure for edge in edges}
    unmatched = [
        f
        for f in entry.froms
        if f not in derived and index_of.get(f, i) < i  # resolves, earlier
    ]
    if not unmatched:
        return []
    return [
        Problem(
            origin=name,
            path=f"$.subgraph[{i}].froms",
            keyword="froms_underived",
            fatal=False,
            message=(
                f"task {name!r} subgraph entry {i} ({entry.closure!r}) declares froms "
                f"{_names(unmatched)}, and no handoff wiring produces "
                f"{'those edges' if len(unmatched) > 1 else 'that edge'}. Two readings, and "
                f"nothing here can tell them apart: it is a dependency that shares no "
                f"handoff, which is what `froms` is for and is legal — or a handoff was "
                f"removed and the edge was left behind. Not an error, because rejecting it "
                f"would delete the feature."
            ),
        )
    ]


def _check_marks_agree(name: str, entries: Sequence[SubgraphEntry]) -> list[Problem]:
    """`is_start` / `is_end` against the edges `froms` declares.

    **The positional defaults cannot be the ones that break.** `is_start`
    defaults true only at index 0, and index 0's `froms` must be empty because
    there is no earlier entry to name; `is_end` defaults true only at the last
    index, which nothing later can name. So both defaults land on a genuine root
    and a genuine sink, *because* of the topological rule — a violation here is
    always an explicit mark, and this check needs no way to tell the two apart.

    Both directions matter for a different reason. `is_start` is observational —
    spec §3.2.1, "dispatching it means the subgraph has begun" — so a mark on an
    entry with a predecessor announces a beginning that already happened.
    `is_end` is not observational: `monitor/base.py:663` has the end subtask's
    completion tell the parent's monitor that the subgraph has finished, and
    `models.py`'s `mine.get(kind) if entry.is_end` wires the parent's outputs to
    it. Marking an entry that has a successor therefore reports completion, and
    transitions the parent, while declared work is still ahead.
    """
    has_successor = {f for entry in entries for f in entry.froms}
    problems: list[Problem] = []
    for i, entry in enumerate(entries):
        if entry.is_start and entry.froms:
            problems.append(
                Problem(
                    origin=name,
                    path=f"$.subgraph[{i}].is_start",
                    keyword="mark_disagrees",
                    message=(
                        f"task {name!r} subgraph entry {i} ({entry.closure!r}) is marked "
                        f"`is_start` and declares froms {_names(entry.froms)}. Dispatching "
                        f"the start entry is what 'the subgraph has begun' means, and this "
                        f"one cannot be dispatched until its predecessors have run. Mark "
                        f"the entry that has no predecessor."
                    ),
                )
            )
        if entry.is_end and entry.closure in has_successor:
            successors = _names([e.closure for e in entries if entry.closure in e.froms])
            problems.append(
                Problem(
                    origin=name,
                    path=f"$.subgraph[{i}].is_end",
                    keyword="mark_disagrees",
                    message=(
                        f"task {name!r} subgraph entry {i} ({entry.closure!r}) is marked "
                        f"`is_end`, and {successors} declare it in their froms. Completing "
                        f"the end entry announces the subgraph as finished and hands the "
                        f"parent its outputs, so marking one with work still ahead of it "
                        f"finishes the subgraph early. Mark the entry nothing depends on."
                    ),
                )
            )
    return problems


def _repeats(entries: Sequence[SubgraphEntry]) -> set[str]:
    seen: set[str] = set()
    twice: set[str] = set()
    for entry in entries:
        (twice if entry.closure in seen else seen).add(entry.closure)
    return twice


def _names(names: Sequence[str]) -> str:
    return ", ".join(repr(n) for n in names) or "nothing"


def _kinds(spec: TaskSpec, key: str) -> list[str]:
    return list(spec.get(key) or ())

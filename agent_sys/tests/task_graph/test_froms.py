"""`froms` and the topological listing order — `closure` criterion 12.

The criterion's last sentence is the one that shapes this file: *"a test that
only checks `froms` against itself would pass with the derivation deleted"*. So
every cross-check test here is written against a subgraph whose edge exists
**only** in the handoff wiring — delete `derived_edges` and the reject tests go
green-and-silent, which is the failure they are here to catch. `test_derivation`
at the bottom is the mechanical half of the same claim.

The marks half is the trap this wave was warned about, and the answer is
`test_the_positional_defaults_cannot_disagree`: the topological rule makes the
positional defaults provably consistent with `froms`, so only an explicit mark
can contradict one.
"""

import pytest

from task_graph.graph import check_graph
from task_graph.models import derived_edges, subgraph_entries

from .conftest import closure_doc, make_task, new_handoffs, task_specs, with_closures


def catalogue(subgraph) -> dict:
    """`collect` produces `trace`; `summarise` consumes it. One derived edge, and
    it comes from the wiring rather than from anything either entry declares."""
    return {
        "pipeline": closure_doc("pipeline", outputs=["report"], subgraph=subgraph),
        "collect": closure_doc("collect", outputs=["trace"]),
        "summarise": closure_doc("summarise", inputs=["trace"], outputs=["report"]),
    }


def entry(closure, froms, **marks) -> dict:
    return {"closure": closure, "froms": list(froms), **marks}


def problems(subgraph) -> list:
    return check_graph(task_specs(catalogue(subgraph)))


# ------------------------------------------------------ the cross-check


def test_froms_omitting_a_derived_edge_is_rejected_naming_both():
    """The core of criterion 12, and the mistake §2.7 says actually happens:
    wiring a handoff and not noticing an edge appeared."""
    (problem,) = problems([entry("collect", []), entry("summarise", [])])

    assert problem.keyword == "froms_mismatch"
    assert problem.fatal
    assert problem.path == "$.subgraph[1].froms"
    assert "'collect'" in problem.message and "'summarise'" in problem.message
    assert "'trace'" in problem.message, "the message must name the kind that makes the edge"


def test_froms_declaring_the_derived_edge_is_silent():
    assert problems([entry("collect", []), entry("summarise", ["collect"])]) == []


def test_a_declared_edge_no_handoff_supports_is_reported_but_not_fatal():
    """The one thing `froms` buys — a dependency that shares no handoff — and
    the reason the check cannot be symmetric. `closure` spec §2.7 calls the
    cross-check "two-directional" and then names this exception, which is the
    same shape as the mistake it wants caught (a handoff removed, the edge left
    behind) and is not distinguishable from it here.
    """
    # `summarise` first, so nothing has produced `trace` when it consumes one:
    # the wiring derives no edge at all, and the only edge is the declared one.
    (problem,) = problems([entry("summarise", []), entry("collect", ["summarise"])])

    assert problem.keyword == "froms_underived"
    assert not problem.fatal, "rejecting this would delete the feature"
    assert "'summarise'" in problem.message


# ------------------------------------------------------ the listing order


def test_a_froms_naming_a_later_entry_is_rejected_naming_the_edge():
    """The topological rule, which is this repository's and not adopted."""
    found = problems([entry("collect", ["summarise"]), entry("summarise", [])])
    order = [p for p in found if p.keyword == "froms_order"]

    (problem,) = order
    assert problem.path == "$.subgraph[0].froms"
    assert "'summarise'" in problem.message and "'collect'" in problem.message
    assert "entry 1" in problem.message, "the message must name the entry it points at"


def test_a_froms_naming_itself_is_rejected():
    found = problems([entry("collect", []), entry("summarise", ["summarise"])])
    (problem,) = [p for p in found if p.keyword == "froms_order"]
    assert "itself" in problem.message


def test_a_froms_naming_no_entry_of_this_subgraph_is_rejected():
    found = problems([entry("collect", []), entry("summarise", ["collect", "elsewhere"])])
    (problem,) = [p for p in found if p.keyword == "froms_resolves"]
    assert "'elsewhere'" in problem.message
    assert "'collect'" in problem.message, "the message must enumerate what is in scope"


def test_a_repeated_closure_is_reported_and_stops_the_rest():
    """`froms` names an entry by its `closure`, so a name used twice has no
    referent. Reporting the ambiguity and abandoning the subgraph beats a page
    of edge problems derived from a guess about which entry was meant."""
    found = problems([entry("collect", []), entry("collect", []), entry("summarise", ["collect"])])
    assert [p.keyword for p in found] == ["froms_ambiguous"]
    assert "'collect'" in found[0].message


# ------------------------------------------------------ the marks


def test_the_positional_defaults_cannot_disagree():
    """**The answer to the interaction.** `is_start` defaults true only at index
    0, whose `froms` must be empty because there is no earlier entry to name;
    `is_end` defaults true only at the last index, which nothing later can name.
    The topological rule therefore makes both defaults land on a genuine root and
    a genuine sink, and a mark can only contradict `froms` when it is written by
    hand. Nothing below declares a mark.
    """
    chain = [entry("collect", []), entry("summarise", ["collect"])]
    assert problems(chain) == []

    entries = subgraph_entries(task_specs(catalogue(chain))["pipeline"])
    assert (entries[0].is_start, entries[0].froms) == (True, ())
    assert (entries[1].is_end, entries[1].froms) == (True, ("collect",))


def test_an_explicit_is_start_on_an_entry_with_predecessors_is_rejected():
    found = problems([entry("collect", []), entry("summarise", ["collect"], is_start=True)])
    (problem,) = [p for p in found if p.keyword == "mark_disagrees"]
    assert problem.path == "$.subgraph[1].is_start"
    assert "'collect'" in problem.message


def test_an_explicit_is_end_on_an_entry_with_a_successor_is_rejected():
    """Not cosmetic. `monitor/base.py:663` has the end subtask's completion tell
    the parent's monitor the subgraph has finished, and `_instantiate` wires the
    parent's outputs to it — so this reports completion, and transitions the
    parent, with declared work still ahead."""
    found = problems([entry("collect", [], is_end=True), entry("summarise", ["collect"])])
    (problem,) = [p for p in found if p.keyword == "mark_disagrees"]
    assert problem.path == "$.subgraph[0].is_end"
    assert "'summarise'" in problem.message


# ------------------------------------------------------ the derivation


def test_derived_edges_always_point_backwards():
    """Why no topological *sort* is needed, and why `graphlib` is not adopted:
    the derivation only ever links to an earlier producer, so the derived graph
    is acyclic and its listing order is already topological. Only a declared
    `froms` can break either, and one index comparison catches it."""
    subgraph = [entry("collect", []), entry("summarise", ["collect"])]
    specs = task_specs(catalogue(subgraph))
    entries = subgraph_entries(specs["pipeline"])

    edges = derived_edges(entries, specs)
    assert all(edge.producer_index < i for i, group in enumerate(edges) for edge in group)
    assert edges == ((), ((0, "trace"),))


def test_unfold_derives_the_handoff_edge_and_adds_the_declared_one(registry, scheduler):
    """The union, and the reason for it: without it `froms` would be checked and
    then discarded, and the handoff-free dependency it exists to express would
    reach no `Task`."""
    subgraph = [
        entry("collect", []),
        entry("audit", []),
        entry("summarise", ["collect", "audit"]),
    ]
    docs = catalogue(subgraph)
    docs["audit"] = closure_doc("audit")  # no handoffs at all: no derivable edge
    with_closures(registry, docs)
    (report,) = new_handoffs(1)
    parent = make_task(outputs=[report], closure="pipeline", kinds={report: "report"})
    scheduler.submit(parent)

    collect, audit, summarise = parent.unfold()
    assert summarise.depends_on == [collect.id, audit.id]
    assert collect.depends_on == [] and audit.depends_on == []


def test_unfold_refuses_a_froms_that_names_no_earlier_entry(registry, scheduler):
    """Load is the gate, and `unfold` still fails loudly — the same split
    `check_subgraph_targets` documents against `unfold`'s own raise, and it is
    what a closure named at run time by a monitor runs into."""
    from task_graph.models import TaskStateError

    docs = catalogue([entry("collect", []), entry("summarise", ["nowhere"])])
    with_closures(registry, docs)
    (report,) = new_handoffs(1)
    parent = make_task(outputs=[report], closure="pipeline", kinds={report: "report"})
    scheduler.submit(parent)

    with pytest.raises(TaskStateError, match="nowhere"):
        parent.unfold()

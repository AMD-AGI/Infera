"""Reading a key out of a spec, for the modules that share the key.

Two accessors, and they are here for the reason the five schemas are: this
package already declares every key in the system, so a reader of one adds no
interpretation of a package's *content*. `Body` was declared three times in
Python over one `$defs.body`, and `subgraph`'s key twice.

**The line this package does not cross, stated once because it is easy to blur.**
`spec_loader` may *declare and expose* the vocabulary; it may not *act* on it
during a load. Exporting `body_of` for a caller is declaration-side. Having
`load_package` reach into an admitted closure for its `task` key would be
action-side — the loader changing what it does based on what a document contains
— and that is what main spec §4.4 makes structural. The two look alike and are
not: one adds a name to the contract, the other adds behaviour to the pipeline.

Bodies live here rather than in `protocols.py`, which is declarations-only by
construction (`docs/interfaces.md` §8) so that importing it costs nothing and a
circular import is impossible. The signatures are declared there, beside
`render` and `validate`, which are the same shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .protocols import Body, ClosureDoc, TaskSpec

__all__ = ["body_of", "subgraph_of", "task_of", "validator_agent_of"]


def body_of(spec: Mapping[str, Any]) -> Body:
    """A task's or a validator's declared body. `{}` when it declares none.

    Returns the mapping as written rather than a constructed object, which is
    `docs/design.md` §4.1 applied one level down: a spec is a plain `dict`
    throughout, and a constructor would have to coerce. The concrete harm is not
    hypothetical — the dataclass version this replaces turned a task with **no
    body** into `Body(readme="")`, an object that is truthy and reports a body
    that is present and empty. `{}` is falsy, so `if body_of(task):` means what
    it looks like.

    Tolerant of a malformed document on purpose. Problems are collected rather
    than raised (`docs/design.md` §3.6), so a checker runs over documents that
    never passed the schema, and an accessor that raised would take the checker
    down with the first bad spec. **An empty result is therefore not evidence of
    absence** when the schema has not run — the schema is the gate, and this is
    not a second one.
    """
    body = spec.get("body")
    return cast(Body, body) if isinstance(body, Mapping) else cast(Body, {})


def subgraph_of(task: TaskSpec) -> tuple[Mapping[str, Any], ...]:
    """The declared expansion, **as written**. `()` for a leaf.

    Entries come back unnormalised: no mark is defaulted here, because
    `is_start` / `is_end` mean something only once an entry has become a
    `task_graph.SubgraphEntry`, and that type is not this package's to name.
    Splitting it that way is `engineer_principle.md` §3 — this package owns that
    the key exists and is called `subgraph`; `task_graph` owns what an entry
    *means*, and normalises on top of this.

    **The key and the entry shape are named by no specification.** `task_graph`
    chose `task.subgraph` as `[{closure, is_start?, is_end?}]`, with absent marks
    defaulting to first and last, and documented it as a convention. Moving the
    accessor here gives the key one reader; it does **not** promote the
    convention to a rule, and nobody should cite this module as the place it
    became one.

    Non-mapping entries are passed through untouched rather than coerced, for
    `body_of`'s reason: whoever normalises decides what a bare string means, and
    the schema is what rejects one.
    """
    raw = task.get("subgraph")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(raw)


def task_of(doc: ClosureDoc) -> TaskSpec:
    """The task spec inside a closure document — `closure` spec §2, key `task`.

    A task spec is not independently loadable: it is declared inside its closure
    under this key, so anything holding a closure document and wanting the task
    spec must reach through it. Three packages did, and each wrote its own
    reader — `closure.task_of`, `agent.body.task_of`, `task_graph._task_of` —
    because the direct edges are forbidden: `task_graph` may not import
    `closure`, and `closure` imports `task_graph`, so that one would be a cycle.

    **The duplication was admissible on `Pushable`'s terms and is now removable,
    which is better.** `docs/interfaces.md` §8 prices two declarations at one
    drift test; it does not say a checked duplication is as good as no
    duplication. One writer beats two-plus-a-test, so
    `tests/interfaces/test_task_of_agreement.py` goes when the last copy does.

    Returns the nested object **itself, never a copy**: every caller reads
    further into what it gets back, and `Task.unfold` compares identity.

    `{}` when the key is absent or is not a mapping — `body_of`'s tolerance, for
    `body_of`'s reason. The concrete failure this must not hide is on record:
    `task_graph`'s `check_graph` shipped reading `doc["task"][...]` where it was
    handed the inner spec already, found nothing under the wrong key, and
    returned no problems for a catalogue that violated two criteria. Green, and
    inert. An empty result here is not evidence of absence.
    """
    task = doc.get("task")
    return task if isinstance(task, Mapping) else {}


def validator_agent_of(spec: Mapping[str, Any]) -> str | None:
    """The agent spec a **validator** names, or `None`.

    `validator` spec §8.2 row 1: a validator bound to a real agent with a
    declared environment uses *that one*. `closure`'s pass resolves the name
    against `agent_specs`, and it may import only this package — so the reader
    lives here or `closure` hardcodes a key of `validator`'s document.

    **Not `agent_of`, and the broken naming pattern is the point.**
    `closure.agent_of` reads a *closure document* and returns `str`; this reads
    a *validator spec* and returns `str | None`. Both take a `Mapping` and
    neither raises on the wrong document, so handing a closure doc to this one
    yields a plausible string rather than an error. Inside `closure`'s file that
    collision can be aliased away; exported from the leaf, which everyone
    imports, it cannot. The other three accessors here are `<key>_of` because
    their key is unambiguous across documents; this one is not, so the name says
    which document it reads.

    **`str | None` where `closure.agent_of` is `str`**, and that is the specs
    disagreeing rather than an inconsistency: `agent` is *required* on a closure
    document and *optional* on a validator spec, where absence is the ordinary
    case and takes §8.2's global row.

    `""` and a non-string both give `None` — absent, not a declaration nobody
    made. `minLength: 1` in the schema means neither survives a document that
    passed it, and `body`'s `entry: ""` is on record as the bug that shape
    causes: an empty path read as *no entry*, and a programmatic validator
    silently ran as agent-bodied.

    **The annotation is `Mapping[str, Any]` and not an alias, deliberately.**
    `TaskSpec` and `ClosureDoc` are `TypeAlias = Mapping[str, Any]` — comments
    with a type's syntax — so an alias naming the wrong document costs nothing
    at run time and misleads every reader before they reach this docstring. A
    `ValidatorSpec` alias would be worse still: `validator.ValidatorSpec` is a
    pydantic model, so the leaf would export a second importable name for a
    different thing, which is the collision this function's name exists to
    avoid, one level up.
    `validator` found this one: the function whose entire purpose is to say which
    document it reads was annotated with the name of a different one, and an
    annotation is the half of that job a reader meets first, in a hover, before
    any docstring.
    """
    agent = spec.get("agent")
    return agent if isinstance(agent, str) and agent else None

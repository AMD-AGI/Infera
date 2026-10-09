"""The validator registry — one of the four the main design reserves.

`spec_loader.SpecRegistry` is the Protocol and `spec_loader.BaseSpecRegistry` is
its implementation; this subclasses the second, which is design §10.1's *"the
base supplies the dict, the collision policy, and the error shape; this subclass
adds its own load-time checks and its own indexes."* The override point is
`_validate`, which runs **before anything is stored**, so this class never has to
remember to check before calling up rather than after.

**A duplicate name raises.** Criterion 3, and the shape is `fsspec`'s: an error by
default, an identical re-registration a no-op. The alternative is on record as a
mistake — Great Expectations logs `Overwriting declaration` and proceeds, and
Inspect AI's `registry_add` is a bare dict assignment with no check at all.
pandera is the system that raises, with a message that names the collision.

The base also rejects the **reverse** collision — one spec under two names — and
that is worth knowing here rather than discovering: criterion 12's two
parameterised instances differ in `name` and `args`, so they are not
byte-identical and are unaffected.

**The two indexes are separate objects that disagree**, and every system surveyed
keeps them separate for that reason: dbt's `manifest.json` and `run_results.json`
share only `unique_id`; Bazel has three graphs and none is "what executed";
Airflow keeps its serialised DAG and its task instances apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from spec_loader import BaseSpecRegistry
from spec_loader.protocols import SpecInconsistent
from validator.protocols import Dimension
from validator.spec import ValidatorSpec, admit

__all__ = ["RunRecord", "RunState", "ValidatorSpecRegistry"]

#: The reference kinds "who uses this" enumerates. Derived from one table rather
#: than restated at each call site: Airflow's asset orphanage reported live
#: assets dead because materialisation through an `AssetAlias` was not one of the
#: four tables its join covered (#58058), and dbt's `build_node_edges` silently
#: drops any edge whose target is outside a hand-chained set (#14436). Whoever
#: adds a fourth edge kind adds it here and nowhere else. `docs/interfaces.md`
#: §5.4 records that nothing yet owns making this derived rather than declared.
#:
#: **`closure` was missing until `demo` found it**, and its absence was the exact
#: bug the paragraph above cites: a validator two closures run reported as used by
#: nothing. `interfaces.md` §5.4 names three edge kinds — *"a handoff kind naming
#: a validator; a composite naming a member; a closure naming a phase validator"*
#: — and this tuple had two. Quoting Airflow #58058 and then committing it is
#: worth recording rather than quietly correcting.
EDGE_KINDS = ("handoff_kind", "composite", "closure")


class RunState(str, Enum):
    """**"Never run" is a state, not a set difference.**

    Stryker is the model — `Pending`, `No coverage` and `Ignored` are first-class,
    serialised and assertable. The counter-example is Great Expectations, whose
    suite success is `successful == evaluated` with `evaluated = len(results)`: a
    tautology whose denominator counts results produced, so **an empty suite
    reports `success=True`**. That is the bug criterion 14 exists to forbid.
    """

    NEVER_RUN = "never_run"
    RAN = "ran"


@dataclass(frozen=True)
class RunRecord:
    """The historical index's answer. `runs` is a count, not a boolean, so "runs
    constantly" and "ran once" are distinguishable as criterion 14 asks."""

    validator: str
    runs: int
    first_at: datetime
    last_at: datetime


class ValidatorSpecRegistry(BaseSpecRegistry):
    """`kind = "validator"`. The base's name table, plus admission and two indexes."""

    kind = "validator"

    def __init__(self) -> None:
        super().__init__()
        self._models: dict[str, ValidatorSpec] = {}
        #: validator name -> (edge kind, the thing that names it)
        self._edges: dict[str, set[tuple[str, str]]] = {}
        self._runs: dict[str, RunRecord] = {}

    # -------------------------------------------------------------- admission

    def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """§9.3 checks 1 and 2, run before anything is stored.

        Checks 3, 4 and 5 need the *other* registry and so run in the closure
        pass, not here — that is design §10.3's split, and D4's reason for
        criterion 11 joining them there.
        """
        admitted = admit(spec, origin=origin)
        if admitted.name != name:
            raise SpecInconsistent(
                f"{origin}: registered as {name!r} but the spec names itself {admitted.name!r}"
            )

    def _admitted(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """The composite edges, recorded once a spec is genuinely stored.

        Not in `_validate`, which runs before the collision check, and not in an
        `add` override, which also runs on the byte-identical re-registration
        `add` treats as a no-op. This hook runs on neither path — `spec_loader`
        added it after two registries hit the same trap from different sides, and
        it is `engineer_principle.md` §4.1's rule: one that lives inside cannot
        be got wrong from outside.

        **It fixed no live bug here**, and that is worth saying rather than
        implying otherwise: `_edges` holds sets, so a double-record was already
        invisible. What it removes is an ordering this class had to remember. The
        double-append it prevents is real for a list-backed index, so
        `test_a_re_registration_does_not_double_an_edge` pins the property
        against the day this one stops being a set.
        """
        self._edges.setdefault(name, set())
        for member in self.spec(name).members:
            self._edges.setdefault(member, set()).add(("composite", name))

    def spec(self, name: str) -> ValidatorSpec:
        """The typed view of an admitted record.

        Two accessors rather than one because the base's contract is a `Mapping`
        — four registries, one shape — while everything inside this package wants
        the model. Memoised on first read; `get` raises for an unknown name.
        """
        if name not in self._models:
            self._models[name] = admit(self.get(name), origin=self.origin_of(name))
        return self._models[name]

    # --------------------------------------------------------- the static index

    def bind(self, handoff_kind: str, validators: Iterable[str]) -> None:
        """Record that a handoff kind names these validators — *for the artefact*.

        Called by the closure pass, which is the only place both registries are
        loaded. An edge to a validator this registry does not hold is still
        recorded — the disagreement is the useful part, and dropping it is the
        silent set intersection that makes dbt's `child_map` incomplete with no
        error.
        """
        self._record("handoff_kind", handoff_kind, validators)

    def bind_phase(self, closure: str, validators: Iterable[str]) -> None:
        """Record that a closure names these as its **phase** validators.

        A different edge from `bind`, and the difference is the one
        `closure.schema.json` spells out: a phase validator is *"a property of the
        task rather than of any one handoff kind"*, while a kind's own validators
        run whenever that kind is produced. One validator can be reached both
        ways, and `users_of` has to say which.

        Two verbs rather than one with a flag, because the caller always knows
        which it is holding — a flag would be this registry asking its callers to
        classify on its behalf.
        """
        self._record("closure", closure, validators)

    def _record(self, edge: str, source: str, validators: Iterable[str]) -> None:
        for name in validators:
            self._edges.setdefault(name, set()).add((edge, source))

    def users_of(self, name: str) -> list[str]:
        """Static: what names this validator now, across every edge kind.

        The question is *who names this **and how***, which is what "what breaks
        if I change this" actually wants — a bare list of names cannot say whether
        a closure runs it as a phase validator or a kind binds it to an artefact.

        **The tagged string is meant for reading, and it punishes a careless
        parse without being lossy.** Measured over pathological names — `a:b`,
        `::`, `:y`, `x:` — `split(":", 1)[1]` round-trips **exactly**, because no
        edge tag contains a colon, so the first colon is always the separator.
        The naive `split(":")[1]` does not, and returns `a` for `a:b`.

        So the format is injective and the correct recovery is one character from
        the wrong one. Nothing parses it today, and if something ever needs the
        parts the fix is to return structured pairs rather than to teach every
        caller which split to reach for.

        **An earlier revision of this docstring called the format lossy and cited
        it as a reason `closure.closures_using_validator` is not derivable from
        this. Both were wrong** — `closure` reported the naive parse's behaviour
        as the format's, I reproduced their snippet instead of measuring
        `users_of`, and the claim survived into a commit. The two queries are kept
        apart on a different argument, which `interfaces.md` §4.5 records: this
        one is fed from both edge kinds and answers *who names this and how*
        across them, theirs answers *which closures name it as a phase validator*,
        typed, within one.
        """
        return sorted(f"{kind}:{who}" for kind, who in self._edges.get(name, set()))

    def list_by_dimension(self, dimension: Dimension) -> list[str]:
        """Criterion 15 — so "nothing checks trustworthiness on this kind" is
        answerable rather than a thing somebody has to notice."""
        return sorted(n for n in self.names() if self.spec(n).dimension is dimension)

    def dimensions_present(self) -> set[Dimension]:
        return {self.spec(n).dimension for n in self.names()}

    # ----------------------------------------------------- the historical index

    def record_run(self, name: str, *, at: datetime | None = None) -> None:
        """One execution. Accepts a name this registry does not hold.

        Airflow tombstones rather than deleting — `REMOVED = "Task vanished from
        DAG before it ran"` is a *state* and the reconciliation is bidirectional.
        dbt does the opposite and it is worse: an id present in `run_results` and
        gone from the manifest is dropped by a silent set intersection with no
        error, warning or count. **"Has this ever run" is meaningless if deletion
        erases the answer.**
        """
        moment = at or datetime.now(timezone.utc)
        prior = self._runs.get(name)
        self._runs[name] = RunRecord(
            validator=name,
            runs=1 if prior is None else prior.runs + 1,
            first_at=moment if prior is None else prior.first_at,
            last_at=moment,
        )

    def has_ever_run(self, name: str) -> RunRecord | None:
        """Historical: from the observed executions. `None` means never."""
        return self._runs.get(name)

    def run_state(self, name: str) -> RunState:
        return RunState.RAN if name in self._runs else RunState.NEVER_RUN

    def never_run(self) -> list[str]:
        """Registered and never executed — the validators nobody should trust."""
        return sorted(n for n in self.names() if n not in self._runs)


def index_bindings(registry: ValidatorSpecRegistry, kinds: Mapping[str, Sequence[str]]) -> None:
    """Feed the static index from the handoff registry's side, in one pass."""
    for kind, validators in kinds.items():
        registry.bind(kind, validators)

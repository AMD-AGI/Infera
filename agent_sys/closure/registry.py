"""`ClosureRegistry`, and the frozen reverse index behind its six queries.

Neither registry here is a component `Registry`. That one resolves collaborators
late and permits replacement — a test swaps an implementation after wiring, which
is a requirement. A spec registry is a name table that refuses one: two specs
claiming a name is a fault, and one validator under two names would run twice and
record two verdicts against a single handoff version.

Both registries subclass `spec_loader.BaseSpecRegistry`, which holds the dict,
the fsspec collision policy, `SpecNotFound` with its candidate list, and
`origin_of`. A fifth bespoke registry would be a fifth policy to keep in step,
and `docs/design.md` §5.3 records what that costs: Kubernetes' `runtime.Scheme`
is one struct holding seven typed maps with **three different collision
policies**, one of which panics and one of which overwrites unconditionally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from spec_loader import BaseSpecRegistry

from . import query
from .model import agent_of as _agent_of
from .query import Index

__all__ = ["ClosureRegistry"]


class ClosureRegistry(BaseSpecRegistry):
    """The closure table, plus the reverse index and its six queries."""

    kind: ClassVar[str] = "closure"

    def __init__(self) -> None:
        super().__init__()
        self._index: Index | None = None
        self._frozen = False

    # -- registration ------------------------------------------------------ #

    def add(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        if self._frozen:
            raise RuntimeError(
                f"closure registry is frozen; {name!r} from {origin!r} cannot be "
                f"admitted. The index is built over the closures that passed the "
                f"load pass, and an index that can outlive its build is one "
                f"somebody eventually has to purge."
            )
        super().add(name, spec, origin=origin)

    def _build_index(self, regs: Any) -> None:
        """Build the reverse index over every admitted closure.

        Called by `check_closures`, which is the only caller that holds the five
        registries — `freeze()` takes no argument, and the four universes the
        reverse queries validate against (kinds, agents, validators, closures)
        have to be captured from somewhere. Intra-package, and therefore not a
        promise to anybody outside.

        A snapshot rather than a live handle: nothing here holds a reference to
        another module's mutable state after the pass returns.
        """
        self._index = query.build_index(self._specs, regs)

    def freeze(self) -> None:
        """Refuse further registration.

        Called once by the composition root, after the closure pass, over the
        closures that passed — the pass runs first, and a fatal problem raises
        before this line, so reaching it means every admitted closure passed.

        **Frozen structurally, not by convention.** Sphinx is the argument for
        making mutation impossible rather than discouraged: its index outlives
        its build, and the price is a `clear_doc` *and* a `merge_domaindata`
        obligation on every owner, third-party extensions included.
        """
        if self._index is None:
            raise RuntimeError(
                "freeze() before check_closures(); the reverse index is built by "
                "the load pass, which is the only caller holding the five "
                "registries the reverse queries validate against"
            )
        self._frozen = True

    # -- the six read-only queries ----------------------------------------- #
    #
    # None mutates, and none is called at run time. The universe of every answer
    # is the loaded catalogue: a task package that is not loaded is not in it.
    # Unlike Bazel's `rdeps`, which makes the caller name the universe and whose
    # own documentation shows the failure being a confident *empty* answer, we
    # cannot get that universe wrong — only narrow.

    def handoff_kinds(self, closure: str) -> tuple[str, ...]:
        """Every kind this closure touches, inputs and outputs.

        Raises `SpecNotFound` if `closure` is not a closure.
        """
        return query.handoff_kinds(self.get(closure))

    def validators_for(self, closure: str) -> tuple[str, ...]:
        """Every validator that will run: the phase validators, plus the
        per-handoff ones joined through the handoff registry.

        The join is the caller's alternative, and not having to write it is what
        this method is for.
        """
        return self._require_index().validators_for(closure, self.get(closure))

    def closures_using_kind(self, kind: str) -> tuple[str, ...]:
        """Reverse. Raises `SpecNotFound` if `kind` is not a known handoff kind;
        returns `()` for a known kind no closure uses.

        **"Not found" and "found, used by nothing" are different answers.** dbt
        has this right in the data — a pre-populated `[]` — and loses it at all
        six call sites, which guard with `if unique_id in child_map`; the
        user-facing message then has to hedge across typo, unused, and disabled
        alike. Cargo keeps them apart by resolving the name against the catalogue
        *before* touching the graph, so the two never share a code path, and that
        is what happens here.
        """
        return self._require_index().closures_using_kind(kind)

    def closures_using_agent(self, agent: str) -> tuple[str, ...]:
        """Same, for an agent spec. Raises `SpecNotFound` for an unknown one."""
        return self._require_index().closures_using_agent(agent)

    def closures_using_validator(self, name: str) -> tuple[str, ...]:
        """Reverse, for a **phase** validator: which closures name it as one.

        **This docstring used to say `users_of` "structurally cannot see" this
        edge, and that stopped being true when `check_closures` wired
        `bind_phase`** — which this package did itself, so the sentence went stale
        because of a change made here. `validator`'s `users_of` now spans every
        edge kind and tags each entry with which one, so a validator two closures
        run in every output phase is no longer reported as used by nothing.
        Airflow #58058 and dbt#14436 are what that failure looked like elsewhere.

        **So the two are not one fact twice, and the difference is the reason
        this survived a withdrawal** (`docs/interfaces.md` §4.5): `users_of` is
        fed from both sides and answers *who names this, and how*, across kinds;
        this answers *which closures name it as a phase validator*, typed, in one
        kind. Recovering the second from the first means one package parsing
        another's display format — `split(":", 1)`, which does work and is not
        the same thing as a typed answer.
        """
        return self._require_index().closures_using_validator(name)

    def agent_of(self, closure: str) -> str:
        """The agent spec name. Always present, because `agent` is required."""
        return _agent_of(self.get(closure))

    def _require_index(self) -> Index:
        if self._index is None:
            raise RuntimeError(
                "the reverse index has not been built; run check_closures() over "
                "this registry first"
            )
        return self._index

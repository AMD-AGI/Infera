"""The name table the four spec registries share.

`docs/design.md` §5.1: main spec §4.1 accepts the cost of four places to change
when the loading mechanism changes, and names the fix if it hurts — *"a shared
**loader** the four registries call, not a shared registry"*. This class is that
shared base. It holds the dict, the collision policy, and the error shape;
everything a kind does differently belongs in the subclass.

`SpecRegistry` in `protocols.py` is the *Protocol* — declaration only, so
subclassing it yields no dict and no policy, and four packages would each write
their own. `BaseSpecRegistry` is the implementation of that Protocol and is what
`docs/interfaces.md` §3's "four registries subclass it" needs to exist.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .protocols import SpecInconsistent, SpecNotFound

__all__ = ["BaseSpecRegistry"]


class BaseSpecRegistry:
    """Name to admitted spec, with duplicate-is-an-error.

    Deliberately the opposite of `task_graph.Registry`, which overwrites so a
    test can swap a component after wiring (`docs/design.md` §5.3). Two objects,
    two jobs, and the collision policies are irreconcilable — which is why they
    are not one class.

    `fsspec`'s shape: error by default, a byte-identical re-registration a no-op.
    The alternative is on record as a mistake twice over — Great Expectations
    logs "Overwriting declaration" and proceeds, Inspect AI assigns with no check
    at all.
    """

    #: Which of the five kinds this registry holds. Subclasses set it.
    kind: str = "spec"

    def __init__(self) -> None:
        self._specs: dict[str, Mapping[str, Any]] = {}
        self._origins: dict[str, str] = {}
        #: Canonical form of each admitted spec, to the name holding it. This is
        #: what makes the *reverse* collision detectable — see `add`.
        self._identities: dict[str, str] = {}

    # -- writing ----------------------------------------------------------- #

    def add(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """Admit a spec.

        Raises `SpecInconsistent` if `name` is held by a *different* spec; a
        byte-identical re-registration is a no-op, so loading one package twice
        is harmless and loading two that disagree is not.

        **The reverse collision is rejected too**: the same spec admitted under
        two names. `pluggy` does this and the reason transfers exactly — for
        them it silently doubles hook invocations, and here one validator under
        two names would run twice and record two verdicts against one handoff
        version (`docs/design.md` §5.2).

        **That branch is unreachable through both supported entry paths, and it
        stays anyway.** `load_package` keys by `doc["name"]`, so it cannot admit
        one document under two keys; and every subclass so far checks the spec's
        own `name` against the key in `_validate`, which fails first with a more
        specific message. So do not write a test that reaches it through either
        — it cannot pass. It is the last line of defence for a registry populated
        by something else, which is `pluggy`'s reason too.

        Both messages name the two origins, because the whole difficulty of a
        collision is finding the other side. OPA#7806 is a user who still could
        not tell which of two bundles was misconfigured with symmetric blame
        alone.
        """
        self._validate(name, spec, origin=origin)
        identity = _canonical(spec)

        held = self._specs.get(name)
        if held is not None:
            if _canonical(held) == identity:
                return
            raise SpecInconsistent(
                f"{self.kind} {name!r} is already held by a different spec\n"
                f"  first:  {self._origins[name]}\n"
                f"  second: {origin}"
            )

        twin = self._identities.get(identity)
        if twin is not None:
            raise SpecInconsistent(
                f"{self.kind} {name!r} is byte-identical to {twin!r}, "
                f"which is already admitted\n"
                f"  first:  {self._origins[twin]}\n"
                f"  second: {origin}\n"
                f"  hint: one spec under two names runs twice and records two "
                f"results for one artefact."
            )

        self._specs[name] = spec
        self._origins[name] = origin
        self._identities[identity] = name
        self._admitted(name, spec, origin=origin)

    def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """A subclass's own load-time checks, run **before anything is stored**.

        The override point, so a subclass never has to remember to call its
        checks before `super().add(...)` rather than after.

        **Signal a rejection by raising any `ValueError`.** `SpecInvalid` and
        `SpecInconsistent` are the two this package provides and are the right
        choice when they fit, but a module's own exception is equally fine as
        long as it is a `ValueError` — which every one in the tree already is.
        `load_package` turns any of them into a `Problem`.

        That sentence used to *enumerate* the two, and the enumeration was the
        defect. `handoff` put the general form better than the incident does:
        **a contract naming what callers must raise is a contract that grows a
        new violation every time somebody adds a class; one naming a base they
        already inherit cannot.** The enumeration cost a whole multi-package
        load the day `validator` invented a third type — not because they were
        careless, but because the set was open and nothing said so.

        Do **not** subclass `SpecInvalid` merely to be catchable. `handoff`
        measured that six of their ~20 `Malformed` raises are load-time spec
        faults and the rest are runtime artefact faults, so the kinship would
        assert that a device node in a content tree is a spec that failed its
        schema. A type that lies to reach a catch clause is the same fault this
        paragraph is about, pointing the other way.

        Each module's spec lists its own — `handoff` §8 has five, `validator`
        §9.3 has five. None of them is here, because none of them is shared.

        Three properties an override must respect — three traps at one seam
        rather than one trap three times. Two were found the hard way, by
        different registries; the third was found by a reader of the paragraph
        below, before writing anything, which is what a warning like this is
        for:

        **It must be side-effect-free.** It runs before the collision check, so
        a hook that recorded an edge would leave one pointing at a spec that was
        then rejected as a duplicate. Record in `_admitted`, which runs only when
        a spec is actually stored.

        **Only a *fatal* fault may raise.** A report-severity finding — a handoff
        kind admitted under the escape-hatch flag — must be admitted *and*
        reported (`handoff` spec §5.3), and a raise here means it is never
        admitted, so it is never reported either.

        **It runs on a re-registration that then returns as a no-op**, so
        anything expensive here runs once per *occurrence* rather than once per
        spec. Parse to check; parse again in `_admitted` if you also want to
        cache, rather than caching from here.

        **Do not "fix" that by returning early on the byte-identical branch
        before this runs.** It is the obvious optimisation and it is wrong at any
        price: a kind's own load-time checks would then run or not depending on
        whether some *other* package happened to load the same spec first, which
        makes them conditional on package ordering. The check being unconditional
        is a property worth more than the cost — `handoff` measured the cost at
        ~104 ms over 100 realistic kinds, beside a render step already costing
        ~2.3 s serial, and the per-distinct-schema cache belongs in the
        subclass's own `_validate` where the expensive field is.
        """

    def _admitted(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """Called once, after a spec is genuinely stored. The place for indexes.

        The counterpart to `_validate`, and it exists because the obvious place
        to build an index is the wrong one twice over. `_validate` runs before
        the collision check; and `add` returns as a **no-op** on a byte-identical
        re-registration *after* `_validate` has already run, so a subclass
        indexing there appends twice for one spec. Main spec §4.3 makes the same
        kind vendored in two packages a supported case, so that is a real path —
        `handoff` was guarding on `len(self)` changing to survive it.

        Here neither can happen: this runs after the duplicate checks and only on
        the branch that stores. A subclass guarding on `len(self)` is working
        around the absence of this hook and can drop the guard.

        **Whether a subclass was *bitten* depends on its container**, which is
        why the hook is worth more than the bug count suggests: `validator`'s
        index holds sets, so a double-record was invisible there and the override
        was safe by accident rather than by design. A hook that makes the wrong
        thing unrepresentable beats one that happens not to matter yet.
        """

    # -- reading ----------------------------------------------------------- #

    def get(self, name: str) -> Mapping[str, Any]:
        """Raises `SpecNotFound`, naming the kind, the name, and the candidates.

        pytest sets the bar — `fixture 'x' not found` followed by every
        available fixture — and it is already a convention here:
        `env_mgr/registry.py` raises `unknown installer {name!r} (have {...})`.
        """
        try:
            return self._specs[name]
        except KeyError:
            have = ", ".join(self.names()) or "nothing admitted"
            raise SpecNotFound(f"no {self.kind} named {name!r} (have: {have})") from None

    def names(self) -> list[str]:
        """Every admitted name, sorted."""
        return sorted(self._specs)

    def origin_of(self, name: str) -> str:
        """Where `name` was loaded from — the label, never opened.

        Inert provenance, and the one getter this class has. It exists because
        `docs/design.md` §6.2's report rule is *name both sides*, and a
        cross-registry pass holds neither side's file path otherwise.
        """
        if name not in self._origins:
            have = ", ".join(self.names()) or "nothing admitted"
            raise SpecNotFound(f"no {self.kind} named {name!r} (have: {have})")
        return self._origins[name]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} kind={self.kind!r} n={len(self._specs)}>"


def _canonical(spec: Mapping[str, Any]) -> str:
    """A spec's identity, for comparison only.

    `sort_keys` plus no whitespace, so two documents that differ only in key
    order are one spec. Not a digest: nothing persists this, and `handoff` owns
    the one hash in the system that anybody depends on.
    """
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=repr)

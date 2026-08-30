"""The handoff kind registry, its reverse index, and the two-way agreement check.

One of the four spec registries (main design §5.1). It holds the dict, the
collision policy and the error shape from `SpecRegistry`; what is added here is
the kind's own checks, the **reverse index** spec §8 asks for, and the check
that a kind and a validator agree about their binding.

Duplicate registration is an **error**, deliberately the opposite of
`task_graph.Registry`, which overwrites so a test can swap a component after
wiring. Two specs claiming one name is a fault: one validator admitted under
two names would run twice and record two verdicts against one handoff version.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from handoff import kind as kind_mod
from handoff.errors import BindingConflict, Malformed
from handoff.protocols import HandoffKind, HandoffLoadReport, Scope
from spec_loader import BaseSpecRegistry
from spec_loader.protocols import Problem, SpecInvalid, SpecNotFound

__all__ = ["HandoffSpecRegistry"]

#: The validator spec key naming the kinds a validator binds to. **`inputs`,
#: not `binds_to`**: `ValidatorSpec` declares `inputs: tuple[str, ...]` and that
#: model is `extra="forbid"`, so a spec carrying `binds_to` is rejected at
#: admission. Rev. 1 of two designs had two names for one field, and the
#: agreement check — the thing criterion 10 is about — read the one that cannot
#: exist.
BINDS_KEY = "inputs"


class HandoffSpecRegistry(BaseSpecRegistry):
    """Admitted handoff kinds, answering questions rather than exposing a dict.

    The dict, the collision policy, `get`'s candidate list and `origin_of` come
    from `BaseSpecRegistry`, which is `spec_loader`'s implementation of the
    `SpecRegistry` Protocol. What is added here is what a handoff kind does
    differently: its own load-time checks, the reverse index, and the two-way
    agreement check.

    Before that base existed this class held its own copy of the collision
    policy. Two writers of one rule is exactly `engineer_principle.md` §1's
    "never let an invariant have two writers", and the copy is now gone.
    """

    #: Reads in the message, not a lookup key — `Registries.for_kind` is keyed
    #: on `SpecSource.kind` ("handoff") and never on this. So it is spelled the
    #: way main design §5.4's example prints it: `no handoff kind named 'x'`.
    kind = "handoff kind"

    def __init__(self, *, allow_no_validator: bool = False) -> None:
        super().__init__()
        #: Spec §5.3's escape hatch: off by default, permits an absent
        #: validator, disables no existing one, and reports every kind it lets
        #: through by name.
        self._allow_no_validator = allow_no_validator
        self._kinds: dict[str, HandoffKind] = {}
        self._by_validator: dict[str, list[str]] = {}
        self._without_validator: list[str] = []

    # ---- SpecRegistry ----

    def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """The base's hook: this kind's own checks, **before** anything is stored.

        Raising is the contract — `load_package` turns a `SpecInvalid` into a
        `Problem` and carries on, so one refused kind does not hide the other
        nine.

        **Only a *fatal* problem raises.** With the escape-hatch flag set, a
        kind naming no validator produces a report-severity finding and must
        still be *admitted*, because criterion 12 requires it to appear in the
        report — and a kind that raised here would never reach `report()`.
        That is the whole reason `Problem.fatal` exists.
        """
        problems = self.check(spec, origin=origin)
        fatal = [p for p in problems if p.fatal]
        if fatal:
            raise SpecInvalid("\n".join(p.message for p in fatal))

        built = kind_mod.from_spec(spec, origin=origin)
        if built.name != name:
            raise SpecInvalid(f"{origin}: admitted as {name!r} but $.name is {built.name!r}")

    def _admitted(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """The base's post-store hook: build the indexes here, and only here.

        This class previously overrode `add` and guarded on `len(self)`
        changing, because `_validate` runs before the collision check and
        because the base returns as a no-op on a byte-identical
        re-registration — so indexing in the hook appended twice for one spec,
        and main spec §4.3 makes the same kind vendored in two packages a real
        path rather than a hypothetical.

        `_admitted` runs only on the branch that stores, so the double-append
        is **unrepresentable** rather than guarded against, and the guard is
        gone. Two registries hit that trap independently, which is why it
        became a hook rather than a warning in a docstring.
        """
        built = kind_mod.from_spec(spec, origin=origin)
        self._kinds[name] = built
        for validator in built.validators:
            self._by_validator.setdefault(validator, []).append(name)
        if not built.validators:
            self._without_validator.append(name)

    # ---- the kind's own load-time checks ----

    def check(self, spec: Mapping[str, Any], *, origin: str) -> list[Problem]:
        """Spec §8's checks 2, 4 and 5, as `Problem`s. Check 3 is the closure pass's.

        Collected rather than raised, and the escape-hatch finding is
        **`fatal=False`** — the one report-severity problem in the system.

        Where that non-fatal problem actually reaches a caller is worth naming,
        because it is not from here. `load_package` builds `Problem`s only from
        exceptions raised out of `add`, and `_validate` raises on fatal faults
        only — so the escape-hatch fact leaves this package as
        `load_report().without_validator`, and `closure/check.py`'s check 3 is
        what turns it into the `fatal=False` `Problem` (`closure` spec §4). Two
        hops, one writer each. The copy returned here is for a caller that
        wants the whole finding list; nothing in the pipeline reads it.
        """
        try:
            built = kind_mod.from_spec(spec, origin=origin)
        except Malformed as exc:
            return [Problem(origin=origin, path="$", keyword="required", message=str(exc))]

        problems = [
            Problem(origin=origin, path="$.items_schema", keyword="format", message=m)
            for m in kind_mod.check(built, origin=origin, allow_no_validator=True)
        ]
        if not built.validators:
            problems.append(
                Problem(
                    origin=origin,
                    path="$.validators",
                    keyword="minItems",
                    message=(
                        f"{origin}: handoff kind {built.name!r} names no validator"
                        + (
                            " — admitted under the bring-up flag and reported"
                            if self._allow_no_validator
                            else ". A kind with no validator cannot be admitted (spec §5.3)"
                        )
                    ),
                    fatal=not self._allow_no_validator,
                )
            )
        return problems

    # ---- questions, not fields ----

    def kind_of(self, name: str) -> HandoffKind:
        """The admitted kind, as the value the store and the checks need."""
        try:
            return self._kinds[name]
        except KeyError:
            raise SpecNotFound(
                f"no handoff kind named {name!r} (have: {', '.join(self.names()) or 'none'})"
            ) from None

    def validators_for(self, name: str) -> list[str]:
        return list(self.kind_of(name).validators)

    def kinds_for(self, validator: str) -> list[str]:
        """Which kinds name this validator — the reverse index.

        Built at admission rather than searched per call, because spec §8 asks
        for it by name and because the agreement check needs it once per
        validator, not once per query.
        """
        return sorted(self._by_validator.get(validator, ()))

    def can_satisfy_required(self, name: str) -> bool:
        """Whether a kind may sit behind a `fixed.required` input.

        Criterion 15, and it is a question rather than a `scope` getter for the
        reason `engineer_principle.md` §3 gives: every caller comparing a tag
        against the same constant is one branch copied N times. **`addons`
        cannot satisfy a required input** — if it could, the declared interface
        would be advisory (spec §4.2).
        """
        return self.kind_of(name).scope in (Scope.FIXED_REQUIRED, Scope.FIXED_OPTIONAL)

    def load_report(self) -> HandoffLoadReport:
        """The escape-hatch report — a value, not a log line.

        **Named `load_report`, not `report`.** `task_graph/bootstrap.py` already
        calls this accessor by that name, and it matches the type it returns.
        The two names were three (a `merged(reports)` in `interfaces.md` §2 was
        the third), and a composition root reaching for a method that is not
        there did not fail — `getattr(..., lambda: None)` turned it into `None`,
        and `closure`'s check 3 then skipped itself in silence.

        Criterion 12 asserts that a kind admitted without a validator appears
        by name in the startup report *and* the run record, and an assertion
        over a log capture is a test of the logging configuration.
        """
        return HandoffLoadReport(
            admitted=self.names(), without_validator=sorted(self._without_validator)
        )

    # ---- the two-way agreement check ----

    def check_bindings(self, validators) -> None:  # noqa: ANN001 - a SpecRegistry
        """Spec §5.1: the binding is recorded on both sides, and a mismatch **crashes**.

        Runs in the closure pass, not at admission: `check_trace_shape` cannot
        be verified to bind back to `trace` while the validator registry may
        still be empty.

        A silently-resolved conflict means one of the two records is lying and
        nobody finds out which — so this raises. The check is four lines and
        the message is the work: both sides named, both origins, the specific
        differing element rather than "they differ", a fix naming both as
        repair points, deterministic ordering, and a hint where one is
        plausible.

        **No precedent was found for this check.** SQLAlchemy's
        `back_populates` does not verify that the two sides agree; GraphQL
        Federation deleted the requirement in Fed 2. `design.md` D2 and O6.
        """
        for name in self.names():
            for vname in self.validators_for(name):
                if vname not in validators:
                    raise SpecNotFound(
                        f"{self.origin_of(name)}: handoff kind {name!r} names "
                        f"validator {vname!r}, which does not resolve "
                        f"(have: {', '.join(validators.names()) or 'none'})"
                    )
                bound = tuple(validators.get(vname).get(BINDS_KEY) or ())
                if name in bound:
                    continue
                raise BindingConflict(name, vname, self._conflict_message(name, vname, bound))

    def _conflict_message(self, name: str, vname: str, bound: tuple[str, ...]) -> str:
        near = sorted(k for k in bound if k != name)
        hint = (
            f"\n  hint: {near[0]!r} also exists — one of the two was renamed and the other not."
            if near and near[0] in self
            else ""
        )
        return (
            f"handoff kind {name!r} and validator {vname!r} disagree\n"
            f"  {self.origin_of(name)}  validators: {list(self.validators_for(name))}\n"
            f"  <{vname}>  {BINDS_KEY}: {list(bound)}\n"
            f"  differing: {vname!r} binds to {list(bound)}, and {name!r} is not "
            f"in that list\n"
            f"  fix: either add {name!r} to {vname}'s {BINDS_KEY}, or remove "
            f"{vname!r} from {name}'s validators. Both are valid repairs."
            f"{hint}"
        )

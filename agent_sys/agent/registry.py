"""`AgentSpecRegistry` — design §4.

The fourth of the four registries the main design reserves. Spec §3.6's four
load-time checks, and where each lands:

| # | Check | Where |
|---|---|---|
| 1 | The YAML validates; the name is unique | `BaseSpecRegistry`; the schema pass ran earlier |
| 2 | **Every declared backend resolves** | `_validate()`, via `backends.resolve` |
| 3 | **Every knowledge handoff kind resolves** | `check_knowledge`, a second pass |
| 4 | **Knowledge coverage is reported** | the same pass |

**Check 3 is a separate pass** for the reason the main design §6 gives: a
registry cannot see another registry's contents during its own load. It takes
the handoff registry as an argument rather than reaching for it.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agent.backend import BackendUnsupported
from agent.backends import resolve
from agent.spec import AgentSpec, KnowledgeRef, KnowledgeReport, types_absent
from spec_loader import BaseSpecRegistry
from spec_loader.protocols import Problem, SpecInvalid

__all__ = ["AgentSpecRegistry", "KnowledgeWarning"]


class KnowledgeWarning(UserWarning):
    """Spec §3.5's default: missing knowledge warns, naming what is absent.

    A hard requirement blocks bring-up and a silent absence is how zero-shot
    agents happen. A warning plus an opt-in gate is the shape that survives
    both.
    """


class AgentSpecRegistry(BaseSpecRegistry):
    """The fourth of the four. `spec_loader` supplies the collection.

    **The duplicate policy is inherited, not restated.** The first revision
    implemented it here because `spec_loader` was declaration-only and wave 1's
    one unbendable rule is to import the Protocol and never a sibling's
    in-flight implementation. `BaseSpecRegistry` has landed, so the four methods
    are gone.

    The base also rejects the **reverse** collision — one spec admitted under
    two names — and for an agent spec that path is unreachable, because
    `_validate` gets there first: the spec carries its own `name`, so admitting
    it under a second registry name fails the name check with a message that
    says which two names disagree rather than which two origins collided. Both
    are right; the earlier one is more specific, and this is recorded so nobody
    later "fixes" the ordering to reach the base's message.

    Spec §3.6's check 2 goes in `_validate`, which runs **before anything is
    stored**; that is the whole point of the hook, and it is why this class has
    no `add`. The parsed model is cached in `_admitted`, which runs only on the
    branch that stores — `_validate` also runs on a byte-identical
    re-registration that then returns as a no-op, so caching there would parse
    twice and, worse, cache a spec that a later collision check rejected.
    """

    kind = "agent spec"

    def __init__(self) -> None:
        super().__init__()
        self._models: dict[str, AgentSpec] = {}

    # ---- the two hooks the base gives a subclass -------------------------- #

    def _validate(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        """Spec §3.6 checks 1 and 2. Side-effect-free, as the base requires.

        Raises `SpecInvalid` naming the offending value if the model does not
        build or a declared backend does not resolve — criterion 1. Checks 3 and
        4 are a separate pass and cannot run here: a registry cannot see another
        registry's contents during its own load.
        """
        self._check_backends(self._model(name, spec, origin), origin)

    def _admitted(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        self._models[name] = self._model(name, spec, origin)

    # ---- what this registry adds ----------------------------------------- #

    def spec(self, name: str) -> AgentSpec:
        """The parsed model, which is what the runner and selection want.

        A whole operation rather than a raw record plus a parse at every call
        site (`engineer_principle.md` §3).
        """
        self.get(name)  # raises SpecNotFound, naming the candidates
        return self._models[name]

    def check_knowledge(
        self, handoff_specs: Any, *, mandatory: bool = False
    ) -> tuple[list[KnowledgeReport], list[Problem]]:
        """Checks 3 and 4, over every admitted spec.

        `mandatory` is the run-config flag of spec §3.5. **The report is a value
        and both modes read the same one**, so they cannot drift into
        disagreeing about what is missing — criterion 2 asserts exactly that:
        the same spec loads in one mode and is rejected in the other.

        A knowledge ref marked `required` is fatal in both modes; one that is
        not is fatal only under the flag. That is what reconciles criterion 1
        ("an unresolvable knowledge handoff is rejected at load") with criterion
        2 ("missing knowledge warns by default"), and it is the only reader
        `KnowledgeRef.required` has.
        """
        reports: list[KnowledgeReport] = []
        problems: list[Problem] = []
        for name in self.names():
            model = self._models[name]
            report = KnowledgeReport(
                spec=name,
                missing=[ref for ref in model.knowledge if ref.kind not in handoff_specs],
                types_absent=types_absent(model.knowledge),
            )
            reports.append(report)
            problems += self._knowledge_problems(report, mandatory)
        return reports, problems

    # ---- internals -------------------------------------------------------- #

    def _model(self, name: str, spec: Mapping[str, Any], origin: str) -> AgentSpec:
        try:
            model = AgentSpec.of(spec)
        except ValidationError as exc:
            raise SpecInvalid(f"agent spec {name!r} from {origin!r}: {exc}") from exc
        if model.name != name:
            raise SpecInvalid(
                f"agent spec registered as {name!r} names itself {model.name!r} (from {origin!r})"
            )
        return model

    def _check_backends(self, model: AgentSpec, origin: str) -> None:
        """Check 2. **Resolution, not availability** — see the module docstring."""
        for decl in model.backends:
            try:
                resolve(decl.backend_entry, key=decl.key, err=decl.err)
            except BackendUnsupported as exc:
                raise SpecInvalid(
                    f"agent spec {model.name!r} from {origin!r} declares backend "
                    f"{decl.key!r} as {decl.backend_entry!r}, which does not resolve: {exc}"
                ) from exc

    def _knowledge_problems(self, report: KnowledgeReport, mandatory: bool) -> list[Problem]:
        problems: list[Problem] = []
        for index, ref in enumerate(report.missing):
            problems.append(self._problem(report.spec, index, ref, mandatory))
        if report.missing and not mandatory:
            warnings.warn(report.render(), KnowledgeWarning, stacklevel=3)
        return problems

    @staticmethod
    def _problem(spec: str, index: int, ref: KnowledgeRef, mandatory: bool) -> Problem:
        return Problem(
            origin=spec,
            path=f"$.knowledge[{index}].kind",
            keyword="knowledge",
            message=(
                f"agent spec {spec!r} names knowledge handoff kind {ref.kind!r} "
                f"({ref.knowledge_type}), which no handoff spec registers"
            ),
            fatal=mandatory or ref.required,
        )

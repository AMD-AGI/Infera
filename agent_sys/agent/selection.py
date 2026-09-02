"""Backend selection — design §6.

The single most consequential thing the prior art says about implementing spec
§3.3's three sources:

> **Observing the selection must be separable from making it, from day one.**

matplotlib learned it three times. `get_backend()` *resolved and committed* the
choice, so a later `use()` became a silent no-op (#12362); resolution also
destroyed the caller's open figures (#23298). All three bugs are one shape:
selection as a side-effecting operation disguised as a read.

So selection returns everything it learned, and answering "which backend?" is
never what performs the selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from agent.backend import Assignment, BackendUnsupported, Executor
from agent.backends import resolve
from agent.spec import AgentSpec, BackendDecl, Kind

__all__ = ["BackendUnavailable", "Rejection", "Selection", "select_backend"]

#: What a `kind: program` agent selects, and the one place the kind decides
#: anything. Design §7.2.1: "this is one line of dispatch and not a second
#: control flow."
PROGRAM_ENTRY = "program"


class BackendUnavailable(RuntimeError):
    """Nothing in the chain could run here.

    **Carries every rejection and its reason.** keyring #316 is the cautionary
    case: an uncaught exception from inside a probe made `import keyring`
    unusable, the fix broadened the catch, and every rejection reason vanished
    with it. You get the reasons or you get a broad catch, never both, unless
    the probe returns a structured result.
    """

    def __init__(self, spec_name: str, rejected: Sequence[Rejection]) -> None:
        self.spec_name = spec_name
        self.rejected = tuple(rejected)
        tried = "; ".join(f"{r.key}: {r.reason}" for r in self.rejected) or "nothing was tried"
        super().__init__(f"no usable backend for agent spec {spec_name!r} — {tried}")


@dataclass(frozen=True)
class Rejection:
    key: str
    reason: str  # the adapter's own words


@dataclass(frozen=True)
class Selection:
    """Everything the selection learned, not just the answer.

    `source` is what makes criterion 3 assertable without reading logs, and it
    is what keyring lacks: its issue #632 asks for exactly this — *"show which
    backend is being used and why"* — and the `diagnose` command added in reply
    names neither.

    **A dataclass rather than the pydantic model design §6.1 sketches**, because
    `backend` is a live executor: a validating model over a running object is a
    schema for something that has no schema. `Rejection` follows it for
    symmetry.

    **`backend` is typed `Executor`, not `AgentBackend`.** A `kind: program`
    spec selects a `ProgramExecutor`, which has no level 2 by construction
    (design §9), and the runner holds level 1 only (§7.1). `agent/protocols.py`
    narrows it to `AgentBackend`; see `README.md` for the report.
    """

    backend: Executor
    key: str
    source: Literal["cli", "spec", "config"]
    rejected: tuple[Rejection, ...] = field(default_factory=tuple)


def select_backend(
    spec: AgentSpec,
    *,
    override: str | None,
    config_order: Sequence[str],
    assignment: Assignment,
) -> Selection:
    """Choose a backend, and report what was tried and why it was not chosen.

    Raises `BackendUnavailable` if nothing is usable.

    **`assignment` is required and has no default**, which is `interfaces.md`
    §4.11 and not a style choice. The probe *is* the constructor (§6.4), so the
    assignment is how an executor receives its `readme`, `entry`, `zone` and
    `environment` — a caller that omitted it would build an agent with no
    instruction, no entry point and no zone, and **it would start and do
    nothing**. `_deploy` always supplies one and nothing else calls this, so a
    default could only ever make a wrong call possible. Rev. 1 had
    `| None = None`; `closure`'s review found it, and it is the fourth instance
    of a fallback whose reason had expired.

    **Nothing is cached.** keyring caches with `@once` and matplotlib resolves
    once and freezes; both are wrong for us for a reason neither of them has —
    `env_mgr` deploys the environment, so a probe taken before deployment is
    taken at the one moment it is guaranteed to be wrong. Selection happens per
    dispatch, and the probes are an import and a `PATH` lookup.
    """
    rejected: list[Rejection] = []

    # 1 ── the CLI override. Pins the run, and does not fall through.
    if override is not None:
        declared = _declared(spec, override)
        decl = declared or BackendDecl(key=override, backend_entry=override)
        return Selection(backend=_probe(decl, assignment), key=decl.key, source="cli")

    # 1b ── the kind, which is one line of dispatch (design §7.2.1).
    if spec.kind is Kind.HUMAN:
        raise BackendUnavailable(
            spec.name, [Rejection(key="human", reason="kind: human is declared and unimplemented")]
        )
    if spec.kind is Kind.PROGRAM:
        decl = _declared(spec, PROGRAM_ENTRY) or BackendDecl(
            key=PROGRAM_ENTRY, backend_entry=PROGRAM_ENTRY
        )
        return Selection(backend=_probe(decl, assignment), key=decl.key, source="spec")

    # 2 ── the spec's own order. The order IS the preference.
    for decl in spec.backends:
        try:
            return Selection(
                backend=_probe(decl, assignment),
                key=decl.key,
                source="spec",
                rejected=tuple(rejected),
            )
        except BackendUnsupported as exc:
            rejected.append(Rejection(key=decl.key, reason=str(exc)))

    # 3 ── the global fallback, from the whole-system config.
    for key in config_order:
        decl = _declared(spec, key)
        if decl is None:
            decl = BackendDecl(key=key, backend_entry=key)
        elif any(r.key == key for r in rejected):
            continue  # already tried in source 2, and it said no
        try:
            return Selection(
                backend=_probe(decl, assignment),
                key=decl.key,
                source="config",
                rejected=tuple(rejected),
            )
        except BackendUnsupported as exc:
            rejected.append(Rejection(key=key, reason=str(exc)))

    raise BackendUnavailable(spec.name, rejected)


def _declared(spec: AgentSpec, key: str) -> BackendDecl | None:
    """The declaration with this key, if the spec has one.

    Design D6: an override **need not** name a declared backend — the case that
    most needs pinning is a backend the spec's author did not foresee. A key
    that does match uses that declaration's `config` and `err`.
    """
    for decl in spec.backends:
        if decl.key == key:
            return decl
    return None


def _probe(decl: BackendDecl, assignment: Assignment) -> Executor:
    """Construct the backend and let it decide whether it can run here.

    keyring's shape, adopted: **one expression carries both facts.** Its
    `viable` is literally "reading `priority` did not raise", and there is no
    separate `is_available()` to drift out of sync with the ordering — drift is
    the failure mode a boolean probe plus a rank invites.

    Every adapter's constructor is `(key, config, assignment)` and raises
    `BackendUnsupported` naming the cause.
    """
    cls = resolve(decl.backend_entry, key=decl.key, err=decl.err)
    try:
        return cls(decl.key, decl.config, assignment)
    except BackendUnsupported:
        raise
    except Exception as exc:
        raise BackendUnsupported(decl.key, "run here", str(exc)) from exc

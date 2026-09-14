"""Fixtures, and the stubs that stand in for packages still being written.

`docs/implementation-stage.md` §4.1: a wave-1 package is written against the
Protocol, not against a neighbour's in-flight implementation. **If you need a
neighbour's behaviour to run a test, satisfy the Protocol with a stub in your own
`tests/`.** Everything in this file that is not `validator`'s is such a stub:
`MemoryHandoffStore` satisfies `handoff.protocols.HandoffStore`, and
`DictSpecRegistry` satisfies `spec_loader.protocols.SpecRegistry`.

The real `task_graph` is used as shipped — it exists, it is 358 green tests, and
faking it would make `test_invisibility.py` assert against a fake scheduler.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from handoff.protocols import Verdict
from spec_loader.protocols import SpecInconsistent, SpecNotFound
from task_graph.bootstrap import build_registry
from task_graph.ids import HandoffId
from validator.protocols import PhaseKind, StrictLevel
from validator.registry import ValidatorSpecRegistry

# --------------------------------------------------------------------------- #
# Stubs for the two neighbours


class MemoryHandoffStore:
    """`handoff.HandoffStore`'s verdict half, in a dict.

    Only the operations this module calls are implemented; the rest raise, so a
    test that starts depending on one fails loudly rather than silently
    exercising a stub nobody meant to write. **`allocate` is here because that
    is exactly what happened**: `interfaces.md` §4.14 made the scheduler pin an
    output's version at dispatch, so every dispatch through a registry holding
    this stub now reaches the store, and the loud half fired as designed.
    Added by `task_graph`, whose change forced it.
    """

    def __init__(self) -> None:
        self.verdicts: dict[tuple[HandoffId, int], list[Verdict]] = {}
        self.allocated: dict[HandoffId, int] = {}

    def allocate(self, hid: HandoffId) -> int:
        """The next version for this slot. No directory: nothing here has a root.

        A counter rather than `max(...) + 1` over anything, for the same reason
        the real store uses `mkdir`: two attempts must never be handed the same
        number, and a retry must not be pinned to the version the previous one
        left behind.
        """
        nxt = self.allocated.get(hid, -1) + 1
        self.allocated[hid] = nxt
        return nxt

    def list_versions(self, hid: HandoffId) -> list[int]:
        """**Always empty, and that is the truth about this suite rather than a
        shortcut.** The real store calls a version published once its manifest
        exists, and the manifest is written by `seal`; nothing in `tests/validator`
        seals. `FakeRunner.produce` writes the *slot* — `open_next` then
        `HandoffVersion.seal` on `handoff_mgr` — and never touches a store, so no
        directory here has ever been published and the honest answer is `[]`.

        Added because `PhaseRunner._targets` began asking. It is deliberately not
        given a `publish` hook to make it answer otherwise: the version an output
        phase reads is the subject of
        `test_output_version_against_the_real_store.py`, and a stub that models
        publication would be this suite deciding its own answer to the question
        that file exists to ask (`interfaces.md` §8.11g).

        **This stub's tolerance is why the defect it now guards could not surface
        here.** `read_verdicts` below returns `[]` for a version nobody wrote; the
        real store *raises* `Malformed` for a version that is not published, which
        is the exception the bring-up run died on.
        """
        return []

    def latest(self, hid: HandoffId) -> int | None:
        """`None`, for `list_versions`' reason: nothing here is ever published."""
        return None

    def record_verdict(self, hid: HandoffId, version: int, verdict: Verdict) -> None:
        self.verdicts.setdefault((hid, version), []).append(verdict)

    def read_verdicts(self, hid: HandoffId, version: int) -> list[Verdict]:
        return list(self.verdicts.get((hid, version), ()))

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - the loud half
        raise NotImplementedError(f"MemoryHandoffStore is a verdict-only stub; {name} is not here")


class DictSpecRegistry:
    """`spec_loader.SpecRegistry` over a dict. Duplicate-is-an-error, as the base
    promises — a stub that overwrote would hide criterion 3 rather than test it."""

    def __init__(self, kind: str, records: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.kind = kind
        self._items: dict[str, Mapping[str, Any]] = dict(records or {})

    def add(self, name: str, spec: Mapping[str, Any], *, origin: str) -> None:
        if name in self._items and self._items[name] != spec:
            raise SpecInconsistent(f"{origin}: {name!r} already defined")
        self._items[name] = spec

    def get(self, name: str) -> Mapping[str, Any]:
        try:
            return self._items[name]
        except KeyError:
            raise SpecNotFound(
                f"no {self.kind} named {name!r}; have {sorted(self._items)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


class StubClosureRegistry:
    """`closure.ClosureRegistry`'s one method this module calls.

    `validators_for(closure)` returns **the union** — the closure's phase
    validators plus the per-handoff ones joined through the handoff registry.
    That join is `closure`'s and the stub models it as one already-computed set,
    because that is exactly what the real one hands over: asking for the parts
    and joining them here was the defect `demo` found.
    """

    def __init__(self) -> None:
        self._sets: dict[str, tuple[str, ...]] = {}

    def declare(self, closure: str, validators: Sequence[str]) -> None:
        merged = list(self._sets.get(closure, ()))
        merged += [v for v in validators if v not in merged]
        self._sets[closure] = tuple(merged)

    def validators_for(self, closure: str) -> tuple[str, ...]:
        try:
            return self._sets[closure]
        except KeyError:
            raise SpecNotFound(f"no closure named {closure!r}") from None


class RecordingExecutor:
    """Stands in for the agent-bodied runner `agent` design O6 has not chosen.

    Writes exactly the same `verdict.json` a script body writes, which is the
    property `test_agent_bodied_and_script_bodied_validators_are_substitutable`
    is about.
    """

    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[str] = []
        self.agents: list[Any] = []

    def run_body(
        self, spec: Any, env: Any, handoffs: Mapping[HandoffId, Any], registry: Any
    ) -> Any:
        """Mints a fresh **unbound** agent per call and returns its id.

        The real one does the same, and unbound is the mechanism rather than an
        accident: binding it to the producing task would re-create by the back
        door exactly the coupling criterion 10 forbids.
        """
        from task_graph.ids import AgentId

        self.calls.append(spec.name)
        env.verdict_file.write_text(json.dumps({str(h): self.result for h in handoffs}))
        minted = AgentId.new()
        self.agents.append(minted)
        return minted


# --------------------------------------------------------------------------- #
# Spec records


def validator_record(
    name: str,
    *,
    inputs: Sequence[str] = ("trace",),
    dimension: str = "completeness",
    strength: str = "strong",
    cost: str = "seconds",
    logic_source: str = "external_static",
    readme: str = "readme.md",
    entry: str | None = "entry.sh",
    materials: Sequence[str] = (),
    members: Sequence[str] = (),
    reduce: str | None = None,
    args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One admissible validator record, as the loader would hand it over.

    **A composite gets no `body`**, because its implementation is its members and
    `admit` rejects a spec carrying both. Naming `members` here is therefore
    enough to build a valid composite record; the caller does not also have to
    remember to suppress the body.
    """
    record: dict[str, Any] = {
        "name": name,
        "brief": f"{name}: a check, one line",
        "inputs": list(inputs),
        "dimension": dimension,
        "strength": strength,
        # `domain` is omitted rather than written empty: `spec_loader`'s schema
        # gives it `minItems: 1`, so an explicit `[]` is a fault while an absent
        # key is not. A real spec would not write one either.
        "tags": {"logic_source": logic_source, "cost": cost},
        "members": list(members),
        "reduce": reduce,
        "args": dict(args or {}),
    }
    if not members:
        # `entry` and `materials` are **omitted** when absent, not written as
        # `None` / `[]`. `spec_loader.Body` types `entry` as `str` and the schema
        # gives it `minLength: 1`, so `null` is as invalid as `""` — an absent key
        # is the only spelling of "this check is not programmatic".
        body: dict[str, Any] = {"readme": readme}
        if entry is not None:
            body["entry"] = entry
        if materials:
            body["materials"] = list(materials)
        record["body"] = body
    return record


def write_body(root: Path, *, verdict: bool | Mapping[str, bool] = True) -> Path:
    """A package root holding a `readme.md` and an `entry.sh` that writes a verdict."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "readme.md").write_text("# a check\n")
    literal = "True" if verdict is True else "False" if verdict is False else repr(dict(verdict))
    (root / "entry.sh").write_text(
        "#!/bin/sh\nset -eu\npython3 - <<'PY'\n"
        "import json, pathlib\n"
        "zone = pathlib.Path.cwd()\n"
        "ids = json.loads((zone / 'inputs.json').read_text())\n"
        f"want = {literal}\n"
        "out = want if isinstance(want, dict) else {i: want for i in ids}\n"
        "(zone / 'verdict.json').write_text(json.dumps(out))\n"
        "PY\n"
    )
    return root


# --------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture
def package_root(tmp_path: Path) -> Path:
    return write_body(tmp_path / "pkg")


@pytest.fixture
def zone_root(tmp_path: Path) -> Path:
    return tmp_path / "zones"


@pytest.fixture
def store() -> MemoryHandoffStore:
    return MemoryHandoffStore()


@pytest.fixture
def registry(store: MemoryHandoffStore) -> Any:
    """A real `task_graph` registry, plus the three names this module resolves.

    Built through `bootstrap.build_registry` so the scheduler, the managers and
    the fake runner are the shipped ones; the spec registries are added on top,
    because the composition root that registers them is `task_graph` rev. 12's.
    """
    r = build_registry()
    r.register("handoff_store", store)
    r.register("handoff_specs", DictSpecRegistry("handoff"))
    r.register("validator_specs", ValidatorSpecRegistry())
    r.register("closures", StubClosureRegistry())
    return r


@pytest.fixture
def dispatched(registry: Any) -> Any:
    """A real task, really dispatched, with one typed output already sealed.

    Built through the shipped scheduler rather than by hand: `task.current` and
    its `agent_id` come from a genuine dispatch, which is what makes the phase
    runner's attribution and `test_invisibility.py`'s spy assertions mean
    anything.

    The handoff's kind travels through `Task.kinds`, which `submit` passes to
    `declare(..., types=...)` — `docs/interfaces.md` §5.1, closed 2026-08-27. The
    phase reads `Handoff.type` and nothing here restates it.
    """
    from task_graph.models import Task

    registry.get("agent_mgr").register("producer")
    hid = HandoffId.new()
    task = Task(agent_spec="producer", outputs=[hid], kinds={hid: "trace"}, closure=CLOSURE)
    registry.get("scheduler").submit(task)
    registry.get("runner").produce(registry, task.id)
    return task


#: The closure the `dispatched` task came from. A task's validator set is its
#: closure's, asked of `closures` — so a fixture that binds a kind must also put
#: the result into the closure's set, exactly as `closure`'s own index does.
CLOSURE = "produce_trace"


def bind_kind(
    registry: Any, kind: str, validators: Sequence[str], *, closure: str = CLOSURE
) -> None:
    """Register a handoff kind naming its validators, across all three indexes.

    The third is the one that makes the phase run them: `PhaseRunner` asks
    `closures.validators_for(task.closure)` for the whole set rather than
    deriving it from the kinds, so a kind bound and not declared on the closure
    is bound to nothing that runs.
    """
    registry.get("handoff_specs").add(
        kind, {"name": kind, "validators": list(validators)}, origin=f"<{kind}>"
    )
    registry.get("validator_specs").bind(kind, validators)
    registry.get("closures").declare(closure, validators)


__all__ = [
    "DictSpecRegistry",
    "MemoryHandoffStore",
    "PhaseKind",
    "RecordingExecutor",
    "StrictLevel",
    "bind_kind",
    "validator_record",
    "write_body",
]

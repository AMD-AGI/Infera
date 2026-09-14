"""Which store version an output phase checks, driven against the real store.

**`tests/validator/conftest.py::MemoryHandoffStore` cannot ask this question**,
and that is why the defect survived 169 green tests in this directory. It keys
verdicts by `(hid, version)` in a dict and returns `[]` for a pair nobody wrote;
the real `FilesystemStore` *raises* `Malformed` for a version that is not
published. So a phase reading the wrong number is invisible against the stub and
fatal against the store.

The wrong number was `handoff_mgr.get(hid).latest.version` — `HandoffMgr`'s
**slot** version, where every consumer of a `Target` wants the **store**
directory version (`interfaces.md` §5.12). The two counters advance on different
events: the store on **every dispatch**, because `task_graph`'s `_pin_outputs`
allocates unconditionally so `env_mgr` has a path to grant; the slot on **every
agent write**. One dispatch that does not write is enough to part them.

**Both tests here therefore assert the two numbers differ before asserting
anything about the phase.** With slot == store the subject is absent and a pass
would mean nothing — `interfaces.md` §8.11g, *a working instrument pointed at
the safe case*, which is precisely what the rest of this directory was.

Measured on `scratch/demo2-2026-08/bringup/n1`, whose non-leaf `main` is the
first in the tree to declare an output. `examples/demo/`'s `main` declares
`outputs: []`, so no parent ever pinned and slot == store == 0 everywhere.

Tests may import `handoff` — `tests/interfaces/test_handoff_layout.py` states the
terms — and the precedent for a whole file spent on the real store is
`tests/agent/test_gate_against_the_real_store.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from handoff import content as content_mod
from handoff.protocols import HandoffKind
from handoff.store import CONTENT_DIR, FilesystemStore, version_dir
from task_graph.bootstrap import build_registry
from task_graph.ids import HandoffId
from task_graph.models import SUBGRAPH_AGENT_SPEC, Task
from task_graph.store import MemoryStoreMgr
from tests.validator.conftest import (
    DictSpecRegistry,
    bind_kind,
    validator_record,
)
from validator.phase import PhaseRunner
from validator.protocols import PhaseKind, StrictLevel
from validator.registry import ValidatorSpecRegistry

KIND = "directions"
CLOSURE = "choose_directions"
CHECK = "check_directions"


class _OneKind:
    """`handoff.store.KindSource` over a single kind, as the gate's test does."""

    def __init__(self, kind: HandoffKind) -> None:
        self._kind = kind

    def kind_for(self, hid: HandoffId) -> HandoffKind | None:
        return self._kind


def _kind() -> HandoffKind:
    """`structured_text`, the shape `n1`'s `directions` declares."""
    return HandoffKind(
        name=KIND,
        content_type="structured_text",
        items_schema={
            "type": "object",
            "required": ["text.json"],
            "properties": {"text.json": {"type": "string"}},
            "additionalProperties": False,
        },
        validators=(CHECK,),
        scope="fixed.required",
        version=None,
    )


def _write_content(root: Path) -> Path:
    """A publishable `content/` for `structured_text`: README plus `text.json`."""
    items = root / content_mod.ITEMS_DIR
    items.mkdir(parents=True, exist_ok=True)
    (items / "text.json").write_text(json.dumps({"areas": ["sorting"]}), encoding="utf-8")
    (root / "README.md").write_text(
        "# Directions\n\n## Purpose\n\nThe areas a course covers.\n\n"
        "## Schema\n\nOne object with an `areas` list.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    return FilesystemStore(tmp_path / "handoffs", kinds=_OneKind(_kind()))


@pytest.fixture
def registry(store: FilesystemStore) -> Any:
    """`tests/validator`'s registry with the **real** store in the stub's place."""
    from tests.validator.conftest import StubClosureRegistry

    r = build_registry(store=MemoryStoreMgr())
    r.register("handoff_store", store)
    r.register("handoff_specs", DictSpecRegistry("handoff"))
    r.register("validator_specs", ValidatorSpecRegistry())
    r.register("closures", StubClosureRegistry())
    r.get("agent_mgr").register("teacher")
    # The spec a non-leaf runs under, supplied by the system rather than the
    # author (`task_graph.models.SUBGRAPH_AGENT_SPEC`). `n1`'s `main` uses it.
    r.get("agent_mgr").register(SUBGRAPH_AGENT_SPEC)
    return r


@pytest.fixture
def diverged(registry: Any, store: FilesystemStore, tmp_path: Path):
    """`n1`'s state, built through the shipped scheduler and the real store.

    One handoff and **two tasks declaring it as an output**, which is not a
    contrivance: `task_graph/models.py::_instantiate` gives the end entry *the
    parent's own* handoff id — ``hid = mine.get(kind) if entry.is_end`` — so a
    non-leaf that declares an output and the subtask that fills it are two tasks
    over one slot, and `_pin_outputs` runs for both.

    Dispatching the parent first burns store v0 as a hole; the leaf takes v1 and
    is the only one that publishes there. `FakeRunner.produce` then advances the
    **slot** to 0, because it is the first write.

    Returns `(parent, leaf, hid)`.
    """
    hid = HandoffId.new()
    parent = Task(
        agent_spec=SUBGRAPH_AGENT_SPEC,
        outputs=[hid],
        kinds={hid: KIND},
        closure=CLOSURE,
        parent=None,
        is_end=True,
    )
    scheduler = registry.get("scheduler")
    scheduler.submit(parent)

    leaf = Task(
        agent_spec="teacher",
        outputs=[hid],
        kinds={hid: KIND},
        closure=CLOSURE,
        parent=parent.id,
        is_end=True,
    )
    scheduler.submit(leaf)

    # What the leaf's body does inside its grant: write into the directory that
    # was allocated for it, then let the seal publish it in place.
    leaf_pin = leaf.current.output_versions[hid]
    _write_content(version_dir(store.root, hid, leaf_pin) / CONTENT_DIR)
    assert store.seal(hid, leaf_pin, producer=leaf.id) is None, "the fixture must publish"
    registry.get("runner").produce(registry, leaf.id)  # the slot side, and only now

    return parent, leaf, hid


def _phase(tmp_path: Path) -> PhaseRunner:
    root = tmp_path / "pkg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "readme.md").write_text("# a check\n")
    (root / "entry.sh").write_text(
        "#!/bin/sh\nset -eu\npython3 - <<'PY'\n"
        "import json, pathlib\n"
        "zone = pathlib.Path.cwd()\n"
        "ids = json.loads((zone / 'inputs.json').read_text())\n"
        "(zone / 'verdict.json').write_text(json.dumps({i: True for i in ids}))\n"
        "PY\n"
    )
    return PhaseRunner(StrictLevel.DEFAULT, zone_root=tmp_path / "zones", package_root=root)


def _bind(registry: Any) -> None:
    registry.get("validator_specs").add(
        CHECK, validator_record(CHECK, inputs=[KIND]), origin=f"{CHECK}.jsonnet"
    )
    bind_kind(registry, KIND, [CHECK], closure=CLOSURE)


def test_the_fixture_really_parts_the_slot_from_the_store(diverged, registry, store) -> None:
    """**The non-vacuity control, and it is the point of this file.**

    Everything below is a claim about *which of two numbers* is read. If they
    agree, both readings pass and the two tests underneath prove nothing —
    §8.11g's last row, an instrument aimed at the safe case. This fails the day
    the fixture stops producing the divergence, which is the day to come back
    here rather than to trust the green underneath.
    """
    parent, leaf, hid = diverged
    slot = registry.get("handoff_mgr").get(hid).latest.version
    parent_pin = parent.current.output_versions[hid]
    leaf_pin = leaf.current.output_versions[hid]

    assert (parent_pin, leaf_pin) == (0, 1), "the parent burned v0, the leaf took v1"
    assert slot == 0, "the slot counts writes, and there has been exactly one"
    assert slot != leaf_pin, "slot and store must differ or nothing below is tested"
    assert store.list_versions(hid) == [1], "only the leaf's version is published"


def test_a_leafs_output_phase_reads_its_own_store_version_not_the_slot(
    diverged, registry, tmp_path
) -> None:
    """**Fault one.** `_targets` used the slot, so the leaf checked v0 — a hole.

    Before the fix this raised `Malformed: cannot read verdicts of <hid> v0: it
    is not published (published: [1])` out of `history.priors`, which is the
    exception `n1` died on with *"final directions: output_validating"*.
    """
    _, leaf, hid = diverged
    _bind(registry)
    store = registry.get("handoff_store")

    outcome = _phase(tmp_path).run_phase(PhaseKind.OUTPUT, leaf, registry)

    assert outcome.passed is True
    assert [r.version for r in outcome.ran] == [leaf.current.output_versions[hid]]
    assert [v.validator for v in store.read_verdicts(hid, 1)] == [CHECK]


def test_a_non_leafs_output_phase_reads_the_version_its_subgraph_published(
    diverged, registry, tmp_path
) -> None:
    """**Fault two**, which survives fault one's fix on its own.

    A non-leaf pins a version and never writes it, so `output_versions` names a
    hole. It holds no reference to the number its end entry published — §5.12's
    gap one level up — so the phase asks `handoff` which version *is* published
    rather than trusting a pin nothing filled.

    Measured before the fix on `n1`: with fault one repaired the leaf succeeded
    and the parent then died at `validator/phase.py:660` on *"cannot read
    verdicts of … v0"*, stranded in `output_validating` for the whole run.
    """
    parent, _, hid = diverged
    _bind(registry)
    store = registry.get("handoff_store")

    outcome = _phase(tmp_path).run_phase(PhaseKind.OUTPUT, parent, registry)

    assert outcome.passed is True
    assert [r.version for r in outcome.ran] == [1], "the published version, not the parent's pin"
    assert parent.current.output_versions[hid] == 0, "and its own pin is still the hole"
    assert [v.validator for v in store.read_verdicts(hid, 1)] == [CHECK]

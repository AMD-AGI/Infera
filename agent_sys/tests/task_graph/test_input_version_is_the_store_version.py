"""Which number `Execution.input_versions` carries, against the real store.

**The field is read as a store version by two modules and was written as a slot
version by this one.** `env_mgr/grants.py::_versions` merges the two halves —

    versions = dict(execution.input_versions)
    versions.update(execution.output_versions)

— and resolves every entry under ``<store>/<hid>/v<N>/``. `output_versions` is a
store directory version by construction (`Scheduler._pin_outputs` allocates it),
so `input_versions` had to be one too. It was not: `_dispatch_pass` filled it
from `handoff_mgr.latest(hid).version`, which is `HandoffMgr`'s **slot** version.
One field, two currencies, in one dictionary.

`interfaces.md` §5.12 names the two counters and records that the reference
between them has no owner. This is the third place in one stage where a caller
spent one as the other, after `validator.PhaseRunner._targets`' output branch and
its input branch.

**Measured** on the first full `examples/demo2` run to reach the last task
(`scratch/demo2-2026-08/runs/full3.log`), handoff `52c75d0a`, kind `scores`:
store `v0` a hole, store `v1` published, the non-leaf `grade` pinned v0 and never
wrote it, its end entry `score` wrote v1 — and `optimise` recorded
`input_versions: 0`. Thirteen of fourteen tasks had already succeeded.

**The counters agree whenever every handoff is dispatched exactly once**, which
is every graph this repository had before a non-leaf declared an output. So
`examples/demo/` cannot show this and neither can a fixture built to its shape,
which is why the first test below asserts the divergence before anything else
looks at it (`interfaces.md` §8.11g).

`MemoryStoreMgr` is the task/handoff record store and is unrelated;
`handoff_store` is the artefact store, and only the real `FilesystemStore`
distinguishes a published version from an allocated hole.
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

KIND = "scores"


class _OneKind:
    """`handoff.store.KindSource` over a single kind."""

    def __init__(self, kind: HandoffKind) -> None:
        self._kind = kind

    def kind_for(self, hid: HandoffId) -> HandoffKind | None:
        return self._kind


def _kind() -> HandoffKind:
    return HandoffKind(
        name=KIND,
        content_type="structured_text",
        items_schema={
            "type": "object",
            "required": ["text.json"],
            "properties": {"text.json": {"type": "string"}},
            "additionalProperties": False,
        },
        validators=("check_scores",),
        scope="fixed.required",
        version=None,
    )


def _write_content(root: Path) -> Path:
    items = root / content_mod.ITEMS_DIR
    items.mkdir(parents=True, exist_ok=True)
    (items / "text.json").write_text(json.dumps({"per_student": []}), encoding="utf-8")
    (root / "README.md").write_text(
        "# Scores\n\n## Purpose\n\nWhat each student scored.\n\n"
        "## Schema\n\nOne object with a `per_student` list.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    return FilesystemStore(tmp_path / "handoffs", kinds=_OneKind(_kind()))


@pytest.fixture
def registry(store: FilesystemStore) -> Any:
    r = build_registry(store=MemoryStoreMgr())
    r.register("handoff_store", store)
    r.get("agent_mgr").register("scorer")
    r.get("agent_mgr").register("optimiser")
    r.get("agent_mgr").register(SUBGRAPH_AGENT_SPEC)
    return r


@pytest.fixture
def diverged(registry: Any, store: FilesystemStore):
    """`examples/demo2`'s state at the moment `optimise` was dispatched.

    A non-leaf and its end entry declare one handoff as an output — which is not
    a contrivance: `models.py::_instantiate` gives the end entry *the parent's
    own* id, ``hid = mine.get(kind) if entry.is_end``, so `_pin_outputs` runs for
    both. The parent burns store v0 as a hole; the leaf takes v1 and publishes
    there. The slot advances only on the write, so it is 0.

    Returns `(parent, leaf, hid)`.
    """
    hid = HandoffId.new()
    scheduler = registry.get("scheduler")

    parent = Task(
        agent_spec=SUBGRAPH_AGENT_SPEC,
        outputs=[hid],
        kinds={hid: KIND},
        closure="grade",
        parent=None,
        is_end=True,
    )
    scheduler.submit(parent)

    leaf = Task(
        agent_spec="scorer",
        outputs=[hid],
        kinds={hid: KIND},
        closure="score",
        parent=parent.id,
        is_end=True,
    )
    scheduler.submit(leaf)

    leaf_pin = leaf.current.output_versions[hid]
    _write_content(version_dir(store.root, hid, leaf_pin) / CONTENT_DIR)
    assert store.seal(hid, leaf_pin, producer=leaf.id) is None, "the fixture must publish"
    registry.get("runner").produce(registry, leaf.id)  # the slot side, and only now

    return parent, leaf, hid


def test_the_fixture_really_parts_the_slot_from_the_store(diverged, registry, store) -> None:
    """**The non-vacuity control.**

    Every claim below is about *which of two numbers* is recorded. With slot ==
    store there is no subject and a pass would mean nothing — which is exactly
    the state every graph in this repository was in before a non-leaf declared
    an output.
    """
    parent, leaf, hid = diverged
    slot = registry.get("handoff_mgr").latest(hid).version

    assert (parent.current.output_versions[hid], leaf.current.output_versions[hid]) == (0, 1), (
        "the parent burned store v0 as a hole, the leaf took v1"
    )
    assert slot == 0, "the slot counts writes, and there has been exactly one"
    assert slot != leaf.current.output_versions[hid], (
        "slot and store must differ or nothing below is tested"
    )
    assert store.list_versions(hid) == [1], "only the leaf's version is published"


def test_a_consumer_pins_the_published_store_version_not_the_slot(
    diverged, registry, store
) -> None:
    """`optimise`'s case, and the whole of the fix.

    The consumer is dispatched after the divergence, so `_pin_inputs` has to
    choose. `store.latest` answers *which version is published*, filtered on the
    manifest, so the parent's unsealed hole is invisible.
    """
    _parent, _leaf, hid = diverged
    consumer = Task(
        agent_spec="optimiser",
        inputs=[hid],
        kinds={hid: KIND},
        closure="optimise",
        parent=None,
    )
    registry.get("scheduler").submit(consumer)

    pinned = consumer.current.input_versions[hid]
    assert pinned == 1, "the published store version, not the slot"
    assert pinned != registry.get("handoff_mgr").latest(hid).version, (
        "and it is specifically not the slot number, which is 0"
    )
    assert pinned in store.list_versions(hid), "a pinned input must name a published version"


def test_an_input_with_nothing_published_contributes_no_entry(registry) -> None:
    """Silence, not a zero.

    A task whose input has never been published is not ready, and recording a
    version for it would say it is. This is the same silence the old
    `handoff_mgr.latest(hid) is None` produced, kept deliberately: `_ready` is
    what decides eligibility, and `_pin_inputs` must not pre-empt it with an
    invented number.
    """
    hid = HandoffId.new()
    waiting = Task(
        agent_spec="optimiser",
        inputs=[hid],
        kinds={hid: KIND},
        closure="optimise",
        parent=None,
    )
    registry.get("scheduler").submit(waiting)

    assert waiting.current is None or hid not in (waiting.current.input_versions or {})

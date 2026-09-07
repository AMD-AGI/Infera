"""Fixtures for `tests/cli`.

**No test in this directory makes a model call, requires credentials, or
requires a sandbox**, and `demo` design §11 is why that is a property rather
than a limitation: CI *loads* the example on every commit and a human *runs* it.
That is Airflow's arrangement — every one of its shipped-example tests is
parse-only — and it dissolves the tension between spec §1's *"the first thing to
break when one of them drifts"* and spec §5's *"CI does not run it"*.

So the graph here is built and driven by `task_graph.FakeRunner`, whose
completion a test drives explicitly. What that cannot cover is named in the
tests that would have covered it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cli import build, package
from cli.stream import Stream
from task_graph import Task, build_registry


@pytest.fixture(scope="session")
def package_root() -> Path:
    """The demo task package, found the way the CLI finds it.

    Session-scoped: `locate` walks the filesystem and every test wants the same
    answer.
    """
    return package.locate()


@pytest.fixture()
def registry(package_root: Path, tmp_path: Path) -> Any:
    """A loaded system with the default `FakeRunner` and no environment.

    `env` is not passed, so `env_mgr` is left unregistered — which is
    `docs/interfaces.md` §2.4's behaviour and exactly what a test wants: a
    `Task` that reached an executor here would mean the fake was doing something
    a fake must not.
    """
    system = build_registry(
        packages=[package.task_package(package_root)],
        handoff_root=str(tmp_path / "handoffs"),
        knowledge_root=str(tmp_path / "knowledge"),
    )
    return system


@pytest.fixture()
def graph(registry: Any) -> list[Task]:
    """The root plus its unfolded subgraph, added but **not submitted**.

    `add` rather than `submit` is what `show` and `--dry-run` do: it supplies the
    registry a transition needs without placing anything in a pool.
    """
    root = build.root_task("main", registry)
    task_mgr = registry.get("task_mgr")
    task_mgr.add(root)
    tasks = [root, *root.unfold()]
    for subtask in tasks[1:]:
        task_mgr.add(subtask)
    build.wire(tasks)
    return tasks


@pytest.fixture()
def submitted(registry: Any) -> Task:
    """The **root only**, handed to the scheduler. Returns it.

    Not the unfolded graph, and the difference is load-bearing: a non-leaf
    unfolds inside `Task.enter_phase(RUNNING)`, so a caller that unfolds *and*
    submits gets the subgraph twice — four tasks become seven, and the second
    `consume` waits on a handoff nothing will ever produce. Measured here, and
    it is the shape `cli/main.py` had until this fixture found it.

    So: `show` and `--dry-run` unfold by hand and submit nothing; a run submits
    the root and lets the phase transition do it.
    """
    root = build.root_task("main", registry)
    registry.get("scheduler").submit(root)
    return root


@pytest.fixture()
def stream() -> Stream:
    return Stream()


def by_closure(tasks: list[Task], name: str) -> Task:
    """The one task instantiated from a closure. Raises rather than returning
    `None`, because a `None` here folds into an attribute error three lines
    later and names nothing."""
    found = [task for task in tasks if task.closure == name]
    if len(found) != 1:
        raise AssertionError(f"expected one {name!r} task, found {len(found)}")
    return found[0]

"""Fixtures for the loader's tests.

Every test builds its own registries; nothing is process-global
(`docs/design.md` §9). The `Registries` double is five `BaseSpecRegistry`
subclasses rather than five dicts, because these tests exercise the collision
policy that a dict does not have — the Protocol's own docstring says a dict is
enough for a package that only reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from spec_loader import BaseSpecRegistry, YamlPackage


class _Registry(BaseSpecRegistry):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind


class FakeRegistries:
    """A `Registries` satisfying the Protocol, with no composition root."""

    def __init__(self) -> None:
        self.handoff_specs = _Registry("handoff kind")
        self.validator_specs = _Registry("validator")
        self.task_specs = _Registry("task spec")
        self.agent_specs = _Registry("agent spec")
        self.closures = _Registry("closure")

    def for_kind(self, kind: str) -> BaseSpecRegistry:
        try:
            return {
                "handoff": self.handoff_specs,
                "validator": self.validator_specs,
                "task": self.task_specs,
                "agent": self.agent_specs,
                "closure": self.closures,
            }[kind]
        except KeyError:
            raise KeyError(f"no registry for spec kind {kind!r}") from None


@pytest.fixture
def registries() -> FakeRegistries:
    return FakeRegistries()


class PackageBuilder:
    """Write YAML into a package tree, then hand back the package.

    The two mandatory names are created up front (main spec §4.3), because a
    test about discovery or about a schema should not have to restate the
    package's shape — and a test *about* the mandatory names says so by not
    using this builder.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "assets").mkdir(parents=True, exist_ok=True)
        self.write("main.yaml", MAIN)
        self.asset("main.md", "the root task")

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def asset(self, relative: str, text: str = "") -> Path:
        return self.write(f"assets/{relative}", text)

    def one(self, module: str, name: str, body: str) -> Path:
        """One object in a file of its own, under a directory per kind.

        The layout a package *may* use and need not — `test_package.py`'s
        criterion-4 test builds the opposite one by hand and compares.

        A `${name}.md` asset comes with it, because that is how a body is
        declared now: the fixtures show the convention rather than binding a
        path by hand, which is what `demo` spec's best practice asks for and
        what would otherwise make every fixture emit an explicit-binding
        warning.
        """
        self.asset(f"{name}.md", f"what {name} is")
        return self.write(f"{module}s/{name}.yaml", f"module: {module}\nname: {name}\n{body}")

    def package(self, variables: Mapping[str, Any] | None = None) -> YamlPackage:
        return YamlPackage(root=self.root, variables=variables or {})


@pytest.fixture
def builder(tmp_path: Path) -> PackageBuilder:
    return PackageBuilder(tmp_path / "pkg")


# --------------------------------------------------------------------------- #
# Minimal well-formed documents, one per kind. Each is the smallest document its
# schema admits, so a test that adds one key is testing that key.

#: The mandatory entry, in its smallest legal form.
#:
#: **A leaf, and it declares no subgraph.** `main.yaml` is where a package's
#: outermost graph goes (main spec §4.3) and a package with one task has a
#: degenerate one; `YamlPackage` checks that the *file* is there and does not
#: also demand a `subgraph` key, because no specification says a one-task
#: package is ill formed. It names an agent nobody defines, which is legal here
#: on purpose: `load_package` runs no cross-registry check (`design.md` D8).
#:
#: Its body is found by convention from `assets/main.md`, like every other
#: fixture here.
MAIN = """\
module: task
name: main
description: the mandatory entry
agent: main_agent
handoffs: []
validators: []
task:
  goal: hold the package together
  version: "1"
  inputs: []
  outputs: []
"""

HANDOFF = """\
description: a captured kernel trace
content_type: reproducible
scope: fixed.required
validators: [check_trace_shape]
"""

VALIDATOR = """\
brief: every kernel in the trace has a recorded shape
dimension: completeness
strength: strong
inputs: [trace]
tags: {logic_source: external_static, cost: seconds}
"""

AGENT = """\
kind: program
description: runs the trace collector
"""

CLOSURE = """\
description: run the collector and capture one trace
agent: tracer
handoffs: [trace]
validators: [check_trace_shape]
task:
  goal: collect a kernel trace from one e2e run
  version: "1"
  inputs: []
  outputs: [trace]
"""

#: The four modules a user writes and one document each. **`closure` is not one
#: of them**: a user writes `module: task` and the closure is what comes out
#: (`closure` spec §2).
ONE_OF_EACH = {
    "handoff": ("trace", HANDOFF),
    "validator": ("check_trace_shape", VALIDATOR),
    "agent": ("tracer", AGENT),
    "task": ("collect_trace", CLOSURE),
}

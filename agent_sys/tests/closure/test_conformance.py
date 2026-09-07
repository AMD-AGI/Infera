"""What this package promises the rest of the system, checked rather than remembered.

`docs/interfaces.md` §4.5 gives `closure` one import — `spec_loader` — and zero
run-time resolutions. Both are claims about code, and both are checkable.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import closure
from closure.check import check_closures
from closure.protocols import check_closures as declared_check_closures
from closure.registry import ClosureRegistry
from closure.task_registry import TaskSpecRegistry
from spec_loader import protocols as spec_protocols

from .conftest import Regs

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "closure"


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_closure_imports_spec_loader_and_nothing_else_of_ours() -> None:
    """The import rule matters more here than anywhere.

    This is the module whose whole job is looking at four other modules' objects,
    so it is where an import would be easiest to justify and hardest to remove.
    It reaches them through `Registries`.

    `tests/interfaces/test_import_rules.py` asserts the same thing for every
    package. This one is here so the failure names *this* package's reason.
    """
    ours = {"handoff", "validator", "agent", "task_graph", "env_mgr", "monitor", "demo"}
    for path in sorted(PACKAGE.rglob("*.py")):
        illegal = _imports(path) & ours
        assert not illegal, (
            f"{path.relative_to(ROOT)} imports {sorted(illegal)}. `closure` reaches "
            f"the other four modules through `Registries`, by name, at call time."
        )


def test_nothing_is_resolved_at_run_time() -> None:
    """Criterion 8's structural half — `closure` resolves nothing by name.

    A component `Registry.get("...")` call anywhere in this package would be a
    run-time resolution, which `docs/interfaces.md` §4.5 gives this module none
    of. The spy in `test_authority.py` proves the other direction: that nobody
    resolves *us* from a scheduler frame.

    `registry.get(name)` on a *spec* registry is a different thing and is what
    the checks do all day, so the test looks for the component-registry names
    rather than for the method.
    """
    components = {
        "store_mgr",
        "handoff_mgr",
        "task_mgr",
        "agent_mgr",
        "policy",
        "handoff_store",
        "knowledge_store",
        "env_mgr",
        "phase_runner",
        "runner",
        "budget",
        "scheduler",
    }
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value in components:
                raise AssertionError(
                    f"{path.relative_to(ROOT)}:{node.lineno} names the component "
                    f"{node.value!r}. Nothing here resolves at run time."
                )


def test_the_registries_satisfy_the_spec_registry_protocol() -> None:
    """`_BaseSpecRegistry` is provisional: the real base belongs in `spec_loader`
    and is declaration-only today. This is what makes the swap safe."""
    protocol = spec_protocols.SpecRegistry
    wanted = {n for n in vars(protocol) if not n.startswith("_")} | {"__contains__"}

    for registry in (ClosureRegistry(), TaskSpecRegistry()):
        for name in wanted:
            assert callable(getattr(registry, name, None)) or name == "kind", (
                f"{type(registry).__name__} does not satisfy SpecRegistry.{name}"
            )
        assert isinstance(registry.kind, str)
        assert inspect.signature(registry.add).parameters.keys() == inspect.signature(
            protocol.add
        ).parameters.keys() - {"self"}


def test_the_registry_satisfies_the_closure_registry_protocol() -> None:
    registry = ClosureRegistry()
    for name in (
        "handoff_kinds",
        "validators_for",
        "closures_using_kind",
        "closures_using_agent",
        "closures_using_validator",
        "agent_of",
        "freeze",
    ):
        assert callable(getattr(registry, name, None)), f"ClosureRegistry.{name} is missing"


def test_check_closures_matches_its_declaration() -> None:
    """`protocols.py` is frozen by construction. If the implementation and the
    declaration drift, one of the two was edited alone.

    **Defaults are compared, not just names.** Rev. 1 of this test compared names
    only, and that is exactly why it stayed green while the implementation
    carried a `handoff_report=None` default the declaration did not have — the
    default that hid a real defect. A signature test that ignores defaults is not
    a signature test.
    """
    declared = inspect.signature(declared_check_closures)
    actual = inspect.signature(check_closures)
    assert list(actual.parameters) == list(declared.parameters)
    for name, param in actual.parameters.items():
        assert param.default == declared.parameters[name].default, (
            f"check_closures({name}=...) defaults to {param.default!r} in the "
            f"implementation and {declared.parameters[name].default!r} in "
            f"protocols.py"
        )
    assert actual.parameters["skip"].kind is inspect.Parameter.KEYWORD_ONLY
    assert actual.parameters["handoff_report"].default is inspect.Parameter.empty


def test_a_missing_handoff_report_is_loud() -> None:
    """`docs/interfaces.md` §4.11: no `| None = None` on a parameter the
    composition root always supplies, and no early return on it.

    The composition root calls `handoff_specs.load_report()`, which never returns
    `None`. So a `None` arriving here is a wiring fault — the wrong object under
    `handoff_specs`, or a root reaching for an accessor that does not exist — and
    it is precisely the case that once went unnoticed with three green suites.
    """
    with pytest.raises(TypeError, match="handoff_report"):
        check_closures(Regs(), None)


def test_the_package_exports_exactly_what_the_contract_lists() -> None:
    assert set(closure.__all__) == set(closure.protocols.__all__)
    for name in closure.__all__:
        assert hasattr(closure, name), f"closure.__all__ names {name!r}, which is absent"

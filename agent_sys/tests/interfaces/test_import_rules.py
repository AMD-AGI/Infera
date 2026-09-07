"""The import graph is a claim, and a claim about an import graph is checkable.

`docs/interfaces.md` §4 gives each package a row saying what it may import. Every
one of those rows exists because a specific thing goes wrong when it is broken,
and the reasons are in the design documents. These tests are the enforcement, so
the rows do not become aspirations — which is what happened to the composition
root, and what the stage-three consistency pass had to unpick.

They walk the AST rather than grepping the source. `"scheduler" in runner.py` is
`True` today, from two docstring mentions, so a substring check would fail for
the wrong reason.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Every top-level package this repository's `agent_sys` declares.
OURS = {
    "spec_loader",
    "handoff",
    "validator",
    "agent",
    "closure",
    "env_mgr",
    "task_graph",
    "demo",
    "monitor",
}

#: What each package may import *from this repository*. `docs/interfaces.md` §4.
ALLOWED: dict[str, set[str]] = {
    # The leaf. The moment it imports `handoff` to understand a handoff spec,
    # "the loader does not interpret a package's content" stops being structural.
    "spec_loader": set(),
    "handoff": {"spec_loader", "task_graph"},
    "validator": {"spec_loader", "handoff", "task_graph", "monitor"},
    # The runner reports every phase boundary — planned and not — to a monitor.
    "agent": {"spec_loader", "task_graph", "monitor"},
    # The module whose whole job is looking at four other modules' objects, and
    # therefore the one where an import would be easiest to justify and hardest
    # to remove. It reaches them through `Registries`.
    "closure": {"spec_loader"},
    "env_mgr": {"task_graph"},
    "task_graph": {"spec_loader"},
    # One way, and the awkward half is deliberate: the monitor needs `instruct`
    # on a live agent, and declares `Pushable` locally rather than import `agent`.
    # `tests/interfaces/test_pushable.py` is what keeps the two shapes in step.
    "monitor": {"task_graph"},
}


def _sources(pkg: str) -> list[Path]:
    return [p for p in (ROOT / pkg).rglob("*.py") if "scratch" not in p.parts]


def _repo_files(*suffixes: str) -> list[Path]:
    """Every file with one of these suffixes, `scratch/` pruned rather than filtered.

    `scratch/` is gitignored working space that may hold cloned third-party
    trees — `pyproject.toml` excludes it from pytest collection for the same
    reason. Walking it and discarding the result afterwards took the one test
    that does this from ~2 s to ~27 s.
    """
    found: list[Path] = []
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {"scratch", ".git", "__pycache__"}]
        found += [Path(parent) / f for f in files if f.endswith(suffixes)]
    return found


def _imported_packages(path: Path) -> set[str]:
    """Top-level package names this file imports, ours only."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found & OURS


@pytest.mark.parametrize("pkg", sorted(ALLOWED))
def test_package_imports_only_what_it_may(pkg: str) -> None:
    for path in _sources(pkg):
        got = _imported_packages(path) - {pkg}
        illegal = got - ALLOWED[pkg]
        assert not illegal, (
            f"{path.relative_to(ROOT)} imports {sorted(illegal)}; "
            f"{pkg} may import {sorted(ALLOWED[pkg]) or 'nothing of ours'} "
            f"— docs/interfaces.md §4"
        )


def test_spec_loader_is_the_leaf() -> None:
    """Stated separately because it is the one rule that enforces a spec claim.

    Main spec §4.4 says the loader does not read, audit, or constrain a package's
    source. This is the import half of that.

    **The other half got stronger at rev. 10 and moved, so the citation here had
    to change rather than be retyped.** It used to read *"`validate(data: bytes,
    ...)` having no path parameter is half of that"*. `validate` now takes a
    parsed document, and the property it demonstrates is no longer an ordering
    convention inside `load_package` — *render, then check* — but a **type
    boundary**: what crosses the seam is a `SpecDocument`, which has no field
    through which a path could arrive. Two tests guard that, and neither belongs
    in a file about imports:

    - `tests/spec_loader/test_validate.py::test_validate_takes_no_path`, the
      signature;
    - `tests/spec_loader/test_package.py::test_load_package_opens_no_file`, the
      behaviour — `load_package` handed a package with no directory behind it,
      failing if anything is opened. It could not have been written before, when
      `load_package` called `render` and `render` read every source.

    Naming them rather than asserting a third copy here: one fact, one writer
    (`engineer_principle.md` §1).
    """
    for path in _sources("spec_loader"):
        assert not _imported_packages(path) - {"spec_loader"}, (
            f"{path.relative_to(ROOT)} imports from this repository. "
            f"spec_loader is the leaf and must stay one."
        )


def test_no_source_format_survives_the_deletion() -> None:
    """Main spec criterion 17, and this file is where its import half belongs.

    *"No `.jsonnet` or `.libsonnet` file remains in the tree, nothing imports
    `_jsonnet` or `rjsonnet`, and neither is a declared dependency. Stated as a
    criterion rather than left to a grep because a deletion that is 95% done is
    the state in which a second format quietly comes back."*

    An import rule is exactly what the middle clause is, and this file already
    walks an AST for exactly that shape. The last `import _jsonnet` in the tree
    was `tests/validator/test_reference.py`'s, which rendered the three shipped
    general specs; they are YAML now.

    The other two clauses are asserted here too because they are one deletion and
    splitting them across three files is how a 95%-done one survives. **`scratch/`
    is excluded**: it is gitignored working space that may hold cloned
    third-party trees, and `pyproject.toml` already excludes it from collection
    for the same reason.

    **It walks every `.py` under the root and not `_sources(pkg)`, and the first
    draft did the latter.** `_sources` walks the nine directories in `OURS`, and
    `tests/` is not one of them — so the guard could not see the single file the
    clause was written about. Measured rather than reasoned: with `import
    _jsonnet` planted back into `tests/validator/test_reference.py` the first
    draft printed `1 passed`. A working instrument pointed at the safe case,
    which `docs/interfaces.md` §8.11g has thirteen recorded instances of; the
    probe that caught this one is
    `scratch/ui-yaml-2026-08/w3/probe_criterion_17_guard.py` and it is kept so
    the next person can re-aim the instrument rather than trust it.
    """
    banned = {"_jsonnet", "rjsonnet", "jsonnet"}

    offenders = []
    for path in _repo_files(".py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names: set[str] = set()
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = {node.module.split(".")[0]}
            if names & banned:
                offenders.append(f"{path.relative_to(ROOT)} imports {sorted(names & banned)}")
    assert not offenders, "\n".join(offenders)

    sources = _repo_files(".jsonnet", ".libsonnet")
    assert not sources, "\n".join(str(p.relative_to(ROOT)) for p in sorted(sources))

    declared = (ROOT / "pyproject.toml").read_text()
    for line in declared.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" in stripped.split('"')[0]:
            continue
        assert not stripped.startswith(('"jsonnet', '"rjsonnet')), stripped


def test_nothing_imports_demo() -> None:
    """`demo` criterion 15: the demo is downstream of everything and upstream of
    nothing. If a component ever needs it, the demo has stopped being an example
    and has become a dependency."""
    for pkg in OURS - {"demo"}:
        for path in _sources(pkg):
            assert "demo" not in _imported_packages(path), f"{path.relative_to(ROOT)} imports demo"


def test_env_mgr_wall_holds_downward() -> None:
    """`env_mgr` spec §9: nothing in the installer machinery learns about domains
    or zones.

    The wall is asserted in both directions; this is the direction that matters,
    because the shipped 65 tests are below it and criterion 22 requires them to
    keep passing untouched.
    """
    below = {
        "recipe",
        "layer",
        "runner",
        "outcome",
        "report",
        "registry",
        "versions",
        "installers",
    }
    above = {
        "meta",
        "fs",
        "isolation",
        "grants",
        "workspace",
        "sync",
        "remote",
        "prepare",
        "protocols",
    }

    for name in below:
        path = ROOT / "env_mgr" / f"{name}.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                head = node.module.split(".")[0]
                assert head not in above, (
                    f"env_mgr/{name}.py imports {node.module}, which is above "
                    f"the wall — env_mgr design §2.1"
                )

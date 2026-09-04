# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The decoupling wall, in **both** directions. Design §14.4.

Spec §9: *"Decoupling is structural: nothing new imports the installer
machinery, and nothing in the installer machinery learns about domains or
zones."* "Structural" is a claim about the import graph, and an import graph is
checkable — so it is checked rather than intended.

The AST is walked rather than the source grepped: a substring check would match
the word in a docstring and fail for the wrong reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "env_mgr"

#: The shipped installer machinery. Criterion 22 requires its 65 tests to keep
#: passing untouched, and this is the structural half of that.
#:
#: **This is the only list, and it is closed.** It names a set that finished
#: growing before this work started, so enumerating it cannot fall behind. It
#: can still *shrink*: `layer` was removed with the layer model, and a name left
#: here for a module that no longer exists would put the wall's `_above()` — a
#: derived set — permanently one name out of step with reality.
BELOW = {"recipe", "runner", "outcome", "report", "registry", "versions", "installers"}

#: `cli.py` is the only module above the wall that may import from below it, and
#: it does so exactly as it does today.
EXCEPTION = "cli"

#: Neither side of the wall; package plumbing with no imports of its own to rule
#: on. Named so that `test_every_module_is_on_one_side_of_the_wall` stays a
#: partition rather than acquiring a silent remainder.
NEITHER = {"__init__", "__main__", EXCEPTION}


def _above() -> set[str]:
    """Everything design §2 adds — **derived, not enumerated.**

    This was a hand-written list of eleven names, and the omission it allowed is
    the reason it is no longer one. `harness.py` was added above the wall and
    nobody added it here, so it was in neither set: it was never checked for
    importing the shipped machinery, and — worse — a module *below* the wall
    could import it and the test whose entire job is catching that stayed green.
    Measured before the change: injecting ``from env_mgr import harness`` into
    `installers/bin.py` left this file at 41 passed.

    A list that must be extended by hand each time the package grows is a list
    that will be one commit stale, and a wall test that is one commit stale is
    indistinguishable from no wall test. So the rule inverts: **below may import
    only from below**, and above is whatever is left. A new module is checked
    from the moment it exists, and the default for an unclassified name is the
    safe one — forbidden to the shipped machinery — rather than invisible.
    """
    found: set[str] = set()
    for path in ROOT.iterdir():
        if path.is_dir() and (path / "__init__.py").exists():
            found.add(path.name)
        elif path.suffix == ".py":
            found.add(path.stem)
    return found - BELOW - NEITHER


ABOVE = _above()


def _modules(names: set[str]) -> list[Path]:
    out: list[Path] = []
    for name in names:
        path = ROOT / f"{name}.py"
        if path.exists():
            out.append(path)
        package = ROOT / name
        if package.is_dir():
            out.extend(package.rglob("*.py"))
    return out


def _heads(path: Path) -> set[str]:
    """Top-level names this file imports from within `env_mgr`, however spelled.

    Both ``from .fs import x`` and ``from env_mgr.fs import x`` reach the same
    module, and a rule that saw only one spelling would be one refactor from
    silent.

    **Four spellings, and the fourth was missing.** ``from env_mgr import
    harness`` names the module in ``names`` rather than in ``module``, so the
    branch that split ``node.module`` saw only ``env_mgr`` and the reader
    reported nothing at all. That is the spelling `material.py` actually uses,
    and it is the one a below-the-wall module would most naturally copy.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            head, _, rest = node.module.partition(".")
            if head == "env_mgr":
                # ``from env_mgr.fs import x`` names it in `module`;
                # ``from env_mgr import harness`` names it in `names`.
                found |= {rest.split(".")[0]} if rest else {a.name for a in node.names}
            else:
                found.add(head)
        elif isinstance(node, ast.ImportFrom) and node.level and not node.module:
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "env_mgr" and len(parts) > 1:
                    found.add(parts[1])
    return found


@pytest.mark.parametrize("path", _modules(ABOVE), ids=lambda p: p.name)
def test_nothing_new_imports_the_installer_machinery(path: Path) -> None:
    illegal = _heads(path) & BELOW
    assert not illegal, (
        f"{path.relative_to(ROOT.parent)} imports {sorted(illegal)} from below the "
        f"wall — env_mgr spec §9, design §2.1"
    )


@pytest.mark.parametrize("path", _modules(BELOW), ids=lambda p: p.name)
def test_nothing_below_the_wall_learns_about_domains_or_zones(path: Path) -> None:
    illegal = _heads(path) & ABOVE
    assert not illegal, (
        f"{path.relative_to(ROOT.parent)} imports {sorted(illegal)} from above the "
        f"wall — the shipped machinery must not learn about domains or zones"
    )


def test_every_module_is_on_one_side_of_the_wall() -> None:
    """A wall with a gap in it is a fence.

    `BELOW` is enumerated because it is closed; everything else is derived. This
    asserts the two really do partition the package, so the failure mode that
    hid `harness.py` — a module in neither set, checked by neither direction —
    cannot come back by someone adding a file.
    """
    every = {
        p.name if p.is_dir() and (p / "__init__.py").exists() else p.stem
        for p in ROOT.iterdir()
        if p.suffix == ".py" or (p.is_dir() and (p / "__init__.py").exists())
    }
    unclassified = every - BELOW - ABOVE - NEITHER
    assert not unclassified, (
        f"{sorted(unclassified)} is on neither side of the wall, so no direction "
        f"of this test applies to it"
    )
    assert not BELOW & ABOVE, sorted(BELOW & ABOVE)
    missing = BELOW - every
    assert not missing, f"BELOW names {sorted(missing)}, which no longer exists"


def test_a_module_below_the_wall_may_not_import_one_above_it_however_new() -> None:
    """The regression this file's inversion exists for, stated on its own.

    Not a substitute for the parametrised direction above — it names the
    specific module that was invisible, so that if `harness` is ever moved or
    renamed this fails by name rather than by silently having nothing to check.
    """
    assert "harness" in ABOVE, "harness.py moved; this test now guards nothing"
    for path in _modules(BELOW):
        assert "harness" not in _heads(path), (
            f"{path.relative_to(ROOT.parent)} imports env_mgr.harness — the "
            f"shipped machinery must not learn what the operator's harness "
            f"configured"
        )


def test_cli_is_the_only_module_that_may_cross() -> None:
    """Stated as its own test because it is the exception, and an exception that
    nothing names is indistinguishable from a leak."""
    heads = _heads(ROOT / f"{EXCEPTION}.py")
    assert heads & BELOW, "cli.py stopped importing the shipped machinery"
    # It reaches the new surface lazily, inside the sub-command handler, so the
    # four shipped stages pay nothing for it.
    source = (ROOT / f"{EXCEPTION}.py").read_text()
    assert "from .inspection import" in source


def test_fs_path_is_the_bottom_of_the_graph() -> None:
    """It imports only ``os`` and ``pathlib``, and three modules sit on it.
    That is what "the path is the fact" means as an import edge."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse((ROOT / "fs" / "path.py").read_text())):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"os", "pathlib", "__future__"}, sorted(imported)


def test_env_mgr_imports_nothing_of_ours_but_task_graph() -> None:
    """`docs/interfaces.md` §4.6. Restated inside this package's own suite so it
    fails here first, where the person who broke it is looking."""
    ours = {"spec_loader", "handoff", "validator", "agent", "closure", "monitor", "demo"}
    for path in ROOT.rglob("*.py"):
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
        assert not found & ours, f"{path.name} imports {sorted(found & ours)}"


def test_every_test_the_readme_cites_exists() -> None:
    """The criterion-to-test mapping is a claim, and a claim is checkable.

    `README.md` maps all 22 acceptance criteria to named tests, and that mapping
    is the deliverable rather than a formality. A mapping nobody checks decays
    the same way a §5.x-settled-but-§4.x-stale row does — it keeps reading as
    true while the thing it names has been renamed or deleted.

    So this fails the moment the README cites a test that does not exist. It
    deliberately does **not** require the reverse: most tests hold a design
    decision or a deviation rather than a criterion, and demanding every one be
    cited would turn the mapping into an inventory.
    """
    import re

    readme = (ROOT / "README.md").read_text()
    cited = set(re.findall(r"`(test_\w+)`", readme))
    assert cited, "the README cites no tests at all; the mapping has been lost"

    here = Path(__file__).parent
    real: set[str] = set()
    for path in here.glob("test_*.py"):
        real |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))

    missing = sorted(cited - real)
    assert not missing, (
        f"README.md cites {len(missing)} test(s) that do not exist: {missing}. "
        f"Either the test was renamed and the mapping was not, or the criterion "
        f"lost its cover."
    )

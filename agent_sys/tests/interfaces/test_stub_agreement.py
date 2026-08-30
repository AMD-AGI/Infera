"""The `.py` and the `.pyi` are two records of one fact, so something checks them.

`docs/interfaces.md` §8 ships both: the `.py` is importable at runtime and
carries the reasons, the `.pyi` is the shape a type checker reads. That is two
declarations of one contract, which is the failure `engineer_principle.md` §1
names and which the stage-three consistency pass found fifteen instances of.

These tests are what makes the duplication safe rather than merely convenient.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGES = ("spec_loader", "handoff", "validator", "agent", "closure", "env_mgr", "monitor")
ROOT = Path(__file__).resolve().parents[2]


def _surface(path: Path) -> dict[str, str]:
    """Every public top-level name, mapped to a normalised signature.

    A class maps to its bases plus its own public members; a function to its
    argument list and return annotation. Docstrings and bodies are ignored,
    which is the whole point — the stub has neither.
    """
    tree = ast.parse(path.read_text())
    out: dict[str, str] = {}

    def sig(fn: ast.FunctionDef) -> str:
        return ast.unparse(fn.args) + " -> " + (ast.unparse(fn.returns) if fn.returns else "None")

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            out[node.name] = sig(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            members = []
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("__"):
                    members.append(f"{sub.name}{sig(sub)}")
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    members.append(f"{sub.target.id}: {ast.unparse(sub.annotation)}")
            bases = ",".join(ast.unparse(b) for b in node.bases)
            out[node.name] = f"({bases}) " + " ".join(sorted(members))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                out[node.target.id] = ast.unparse(node.annotation)
    return out


@pytest.mark.parametrize("pkg", PACKAGES)
def test_stub_declares_the_same_surface(pkg: str) -> None:
    """The two files agree on names and on signatures.

    If this fails, one of them was edited and the other was not. Regenerate the
    stub rather than hand-patching it — the `.py` is the source.
    """
    impl = _surface(ROOT / pkg / "protocols.py")
    stub = _surface(ROOT / pkg / "protocols.pyi")

    assert set(impl) == set(stub), (
        f"{pkg}: only in .py {sorted(set(impl) - set(stub))}; "
        f"only in .pyi {sorted(set(stub) - set(impl))}"
    )
    disagree = {k: (impl[k], stub[k]) for k in impl if impl[k] != stub[k]}
    assert not disagree, f"{pkg}: signatures differ for {sorted(disagree)}"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_all_is_exhaustive_and_resolvable(pkg: str) -> None:
    """Every name in `__all__` exists, and nothing public is left out of it.

    `__all__` is what `docs/interfaces.md` §1.2 calls *frozen*: it is the list a
    reader consults to know what changing something costs. A public name missing
    from it is a seam nobody knows is a seam.
    """
    mod = __import__(f"{pkg}.protocols", fromlist=["protocols"])
    declared = set(mod.__all__)

    missing = [n for n in declared if not hasattr(mod, n)]
    assert not missing, f"{pkg}.__all__ names {missing}, which do not exist"

    public = {
        n
        for n, v in vars(mod).items()
        if not n.startswith("_")
        and getattr(v, "__module__", f"{pkg}.protocols") == f"{pkg}.protocols"
    }
    assert public <= declared, f"{pkg}: public but not in __all__: {sorted(public - declared)}"

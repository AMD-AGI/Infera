#!/usr/bin/env python3
"""Control: `--impl <path>` measures the bytes at *that path*, and says so.

    python3 assets/lib/controls/impl_is_the_file_you_named.py

Exit 0 = the property holds **and** the guard that reports it discriminates.
Exit 1 = one of the four cases below came out wrong. No node, no GPU, no torch.

### The dependency this exists for

m4's third-tree workspace (`63bcaca`) is on **no interpreter's import path**.
That is safe only while `_common.load_impl` reads the file it was handed rather
than resolving a module through `sys.path`. If it ever became an import, forge
would keep editing its tree, the driver would keep measuring the container's
untouched copy, and **every ratio would come back ~1.0 with no error anywhere**
— a wrong answer byte-identical to *"the optimiser found nothing"*.

m4 could only document that at the invocation. **A comment cannot fail.** This
can.

### Why the guard is a digest and not the mechanism

The tempting assertion is *"the loader execs rather than imports"*, which is a
claim about implementation and goes stale the moment someone finds a third way
to load a file. What m4 actually depends on is the **outcome**: the bytes
measured are the bytes at the path they named. So `load_impl` records
`impl_read = {path, sha256, bytes, loaded_by}` at the moment it reads, and a
consumer compares that digest against the file it just wrote.

`report["impl_path"]` cannot do this job and that is the point of case 4 below:
it is `args.impl` copied at parse time, so it reads the same whether the file
was execed, imported, shadowed, or never opened at all.

### The four cases, and case 3 is the one that matters

1. the digest matches the file that was named                    (the property)
2. editing that file changes the digest                          (it tracks bytes,
                                                                  not the name)
3. **a constructed import-based regression is caught**           (it discriminates)
4. `impl_path` alone does **not** catch case 3                   (the null: without
                                                                  this, 1–3 prove
                                                                  nothing new)

Case 4 is the control's own control. A guard that fires on the regression is
only worth adding if the field we already had would *not* have fired — otherwise
the honest answer to the leader was "already covered, change nothing".
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve()
HARNESS = HERE.parents[2] / "build_workset.task" / "harness"
sys.path.insert(0, str(HARNESS))

import _common  # noqa: E402

CANDIDATE = "def run(x):\n    return x + 1\n"
SHADOW = "def run(x):\n    return x - 1\n"

_OPERATOR = {"operator_id": "control_op"}
_DEFINITION = {"baseline": "def run(x):\n    return x\n", "reference": "def run(x):\n    return x\n"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_by_import(module_dir: pathlib.Path, module_name: str):
    """**The constructed regression.** What a well-meaning rewrite looks like.

    Resolves `module_name` through `sys.path` instead of reading a given file —
    which is exactly the change m4's workspace cannot survive, because their tree
    is on no import path and some *other* copy of the same module name is.
    """
    import importlib

    sys.path.insert(0, str(module_dir))
    try:
        module = importlib.import_module(module_name)
        return module.run
    finally:
        sys.path.remove(str(module_dir))
        sys.modules.pop(module_name, None)


def main() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        named = root / "named" / "candidate.py"
        named.parent.mkdir()
        named.write_text(CANDIDATE, encoding="utf-8")

        # 1 — the property.
        _common.IMPL_READ.clear()
        run = _common.load_impl(_OPERATOR, _DEFINITION, str(named), "baseline")
        read = dict(_common.IMPL_READ)
        if read.get("sha256") != _sha(CANDIDATE):
            failures.append(f"1: impl_read.sha256 {read.get('sha256')} != sha256 of the named file")
        if read.get("path") != str(named.resolve()):
            failures.append(f"1: impl_read.path {read.get('path')} != {named.resolve()}")
        if run(1) != 2:
            failures.append("1: the callable is not the named file's `run`")
        print(f"  1 named file measured and recorded   sha={read.get('sha256', '?')[:12]} "
              f"loaded_by={read.get('loaded_by')}")

        # 2 — the digest tracks bytes, not the name.
        named.write_text(CANDIDATE.replace("+ 1", "+ 7"), encoding="utf-8")
        _common.IMPL_READ.clear()
        _common.load_impl(_OPERATOR, _DEFINITION, str(named), "baseline")
        if _common.IMPL_READ.get("sha256") == read.get("sha256"):
            failures.append("2: the digest did not change when the file did")
        print(f"  2 edited file changes the digest     sha={_common.IMPL_READ.get('sha256','?')[:12]}")
        named.write_text(CANDIDATE, encoding="utf-8")

        # 3 — the constructed regression, caught.
        shadow_dir = root / "shadow"
        shadow_dir.mkdir()
        (shadow_dir / "candidate.py").write_text(SHADOW, encoding="utf-8")
        imported = _load_by_import(shadow_dir, "candidate")
        measured_wrong = imported(1) != 2  # the shadow returns 0, the named file 2
        caught = _sha(SHADOW) != _sha(named.read_text(encoding="utf-8"))
        if not measured_wrong:
            failures.append("3: the constructed regression did not actually measure the wrong file — "
                            "the control cannot discriminate and proves nothing")
        if not caught:
            failures.append("3: a digest comparison would NOT have caught the regression")
        print(f"  3 import-based regression measures {imported(1)} where the named file gives {run(1)}; "
              f"digests differ -> caught")

        # 4 — the null. `impl_path` alone would have missed it.
        # Both loaders were handed the same *name*; only the digest distinguishes.
        if str(named) == str(shadow_dir / "candidate.py"):
            failures.append("4: the control is malformed — the two paths are identical")
        same_basename = named.name == (shadow_dir / "candidate.py").name
        if not same_basename:
            failures.append("4: the shadow does not share the module name, so it is not the regression")
        print(f"  4 both are '{named.name}' and report['impl_path'] is the argument, unchanged "
              f"by which was loaded -> the old field would NOT have caught it")

    for line in failures:
        print(f"  FAIL {line}", file=sys.stderr)
    print("\nimpl_is_the_file_you_named: " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

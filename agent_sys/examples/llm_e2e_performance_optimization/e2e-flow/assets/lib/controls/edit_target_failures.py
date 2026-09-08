#!/usr/bin/env python3
"""Control: the two ways `edit_target` goes wrong, exercised against known inputs.

    python3 assets/lib/controls/edit_target_failures.py

Exit 0 = each case behaves as recorded below. No node, no GPU, no torch.

### Why this exists

`edit_target` was hit from two directions on 2026-09-04 and both were scheduled
to be discovered *during* the first rung-3 run, on a held node, three stages
downstream of their cause:

| | what moved | what stayed | found by |
|---|---|---|---|
| **A** | the path | the offsets | m5 — k004's workset names `mixed_moe_gemm_2stage.py`, the identity says `moe_gemm_2stage.py`, and k004 keeps its own 3079/3275 |
| **B** | the offsets | the path | m4 — the workset records line 183, the fragment is at 207 in the deployed image |

The question worth answering standalone is not *"can they happen"* but **"does
either produce a result that looks like a correct null?"** A resolver that
refuses loudly is a scheduling problem. One that silently resolves to the wrong
span, while everything downstream validates, is what costs a rung.

### What this found, and B is not what it looked like

**A refuses, once the workset carries provenance.** `from_identity` records what
the identity said; `check_workset_shape._check_target_paths` compares. Case A1
shows it refusing. **A2 is the honest limit**: a workset with no `from_identity`
— every artefact sealed before 2026-09-04, k004's included — is UNVERIFIED and
reported as a note, not a refusal. Nothing can compare against a record that was
never made.

**B is a mock-only defect, and that is the finding.** `scaffold.py` — the real
producer — emits **no line number at all**; there is nothing to go stale. The
183 that misled m4 is a constant in `mock_adapt.py`, wrong by 24 lines against
the image, and it appears **twice**: once as `entry_function_line` and once
inside `resolution_evidence`'s prose, which repeats the number as its own
justification. B3 pins the value so a reintroduction fails here rather than in
a campaign.

**And nothing in this stage grades a line number, by design.** m4 established
that an offset must never be a locator — they anchor on the fragment text — so
the right treatment is to stop emitting a wrong one and to say in the schema
that it is advisory. A validator arm that checked the number against a file
would need the image, which no validator in this package has.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
PKG = HERE.parents[2]
os.environ.setdefault("AGENT_SYS_TASK_PACKAGE", str(PKG.parent))
sys.path.insert(0, str(PKG / "lib"))

_spec = importlib.util.spec_from_file_location(
    "check_workset_shape", PKG / "check_workset_shape.validator" / "check.py"
)
_shape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shape)

RIGHT = "aiter/ops/flydsl/kernels/moe_gemm_2stage.py"
WRONG = "aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"


def _operator(**over):
    op = {
        "operator_id": "moe_gemm_mfma_moe2_afp4_wfp4",
        "edit_target": {
            "source_file": RIGHT,
            "editable_sources": [RIGHT],
            "entry_function": "compile_mixed_moe_gemm2",
            "from_identity": {"source_file_path": [RIGHT], "kernel_id": "k004"},
        },
        "integration": {"target_files": [RIGHT]},
    }
    op["edit_target"].update(over.pop("edit_target", {}))
    op.update(over)
    return op


def _run(label, operator, expect):
    problems, notes = [], []
    _shape._check_target_paths(operator, problems, notes)
    got = "REFUSE" if problems else "accept"
    ok = got == expect
    detail = (problems or notes or [""])[0]
    print(f"  {'OK ' if ok else 'BAD'} {label:54} {got:7} {detail[:76]}")
    return ok


def main() -> int:
    ok = True
    print("A — the path moved, the offsets stayed (m5's half)")
    ok &= _run("A0 null: workset agrees with the identity", _operator(), "accept")
    moved = _operator()
    moved["edit_target"]["source_file"] = WRONG
    moved["edit_target"]["editable_sources"] = [WRONG]
    moved["integration"]["target_files"] = [WRONG]
    ok &= _run("A1 path moved, provenance recorded", moved, "REFUSE")
    sealed = copy.deepcopy(moved)
    sealed["edit_target"].pop("from_identity")
    ok &= _run("A2 path moved, NO provenance (every sealed artefact)", sealed, "accept")

    print("\nB — the offsets went stale, the path was right (m4's half)")
    stale = _operator(edit_target={"entry_function_line": 183})
    ok &= _run("B1 a wrong line number, nothing else changed", stale, "accept")

    from_scaffold = set()
    src = (PKG / "build_workset.task" / "scaffold.py").read_text(encoding="utf-8")
    block = src.split('"edit_target": {', 1)[1].split("},", 1)[0]
    for line in block.splitlines():
        if '":' in line:
            from_scaffold.add(line.strip().split('"')[1])
    b2 = "entry_function_line" not in from_scaffold and "device_function_line" not in from_scaffold
    print(f"  {'OK ' if b2 else 'BAD'} {'B2 the real producer emits no line number':54} "
          f"{'confirmed' if b2 else 'IT DOES — B is reachable in production'}")
    ok &= b2

    # **Parsed, not grepped, and the first version of this was grepped.** A text
    # search cannot tell a live constant from a comment explaining why it was
    # removed — so the moment the fix landed with a comment naming the field,
    # B3 kept failing. That is the same defect this control exists to catch, in
    # the control: an instrument that answers a question adjacent to the one
    # asked. `ast` looks for the key in a dict literal, which is the fact.
    import ast

    mock_src = (PKG / "build_workset.task" / "mock_adapt.py").read_text(encoding="utf-8")
    emitted: set[str] = set()
    for node in ast.walk(ast.parse(mock_src)):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    emitted.add(key.value)
    b3 = "entry_function_line" not in emitted and "device_function_line" not in emitted
    print(f"  {'OK ' if b3 else 'BAD'} {'B3 the mock emits no line number either':54} "
          f"{'confirmed (no such key in any dict literal)' if b3 else 'the constant is back in mock_adapt.py'}")
    ok &= b3

    print("\nedit_target_failures: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

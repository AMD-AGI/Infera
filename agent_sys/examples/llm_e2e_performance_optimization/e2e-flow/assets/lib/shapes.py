#!/usr/bin/env python3
"""Parsing Magpie's `Input Shapes` column into forge-loop case selectors.

The column holds one or more observed call signatures, separated by `; `. Each
signature is a chain of tensor shapes joined by `x`, each shape written
`[d0,d1,...]`. Scalars appear as `[n]`. An empty cell means the profiler
recorded no shapes for that kernel, which happens for kernels launched outside
an aten op.

Example, verbatim from the sample profile:

    [288,6144]x[33,4096,3072]x[33,6144,1024]x[288,9]; [256,6144]x[33,4096,3072]x...

The case selector shape is Hyperloom's: a `CASE_ID` plus named dimensions.
`_task_group_contract.py` uses `CASE_SELECTOR_KEY = "CASE_ID"` and KernelForge's
preflight rejects a driver that does not cover every declared selector, so the
names produced here are a contract and not a label.
"""

from __future__ import annotations

import re

CASE_SELECTOR_KEY = "CASE_ID"

_TENSOR = re.compile(r"\[([0-9,\s]*)\]")


def parse_input_shapes(cell: str) -> list[list[list[int]]]:
    """`Input Shapes` -> one entry per observed signature, each a list of shapes."""
    cell = (cell or "").strip()
    if not cell:
        return []
    signatures = []
    for chunk in cell.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        tensors = []
        for match in _TENSOR.finditer(chunk):
            body = match.group(1).strip()
            if not body:
                tensors.append([])
                continue
            try:
                tensors.append([int(d.strip()) for d in body.split(",") if d.strip()])
            except ValueError:
                continue
        if tensors:
            signatures.append(tensors)
    return signatures


def dedupe(signatures: list[list[list[int]]]) -> list[list[list[int]]]:
    """Distinct signatures, first occurrence order preserved.

    Order matters: `build_task_group_shape_cases` treats the first case as
    primary, and the profiler emits the highest-frequency signature first.
    """
    seen = set()
    out = []
    for signature in signatures:
        key = repr(signature)
        if key in seen:
            continue
        seen.add(key)
        out.append(signature)
    return out


def named_dimensions(signature: list[list[int]], category: str) -> dict:
    """Give the leading dimensions names a driver can select on.

    Hyperloom infers M/N/K for GEMM and QTOKENS/QHEADS/KVHEADS/HEADSIZE for
    attention. The naming below covers the same two families and falls back to
    positional `D0_0` style names, which are still selectable even when the
    semantic name is unknown. A wrong semantic name would be worse than a
    positional one, because a driver author would trust it.
    """
    if not signature:
        return {}

    first = signature[0]
    if category in {"gemm", "moe_gemm"} and len(signature) >= 2 and len(first) >= 2:
        a, b = signature[0], signature[1]
        if len(a) >= 2 and len(b) >= 2:
            return {"M": a[-2], "K": a[-1], "N": b[-1]}

    if category == "attention" and len(first) >= 3:
        names = ["QTOKENS", "QHEADS", "HEADSIZE"]
        out = {name: first[index] for index, name in enumerate(names) if index < len(first)}
        if len(signature) >= 2 and len(signature[1]) >= 2:
            out["KVHEADS"] = signature[1][-2]
        return out

    out = {}
    for t_index, tensor in enumerate(signature[:3]):
        for d_index, dim in enumerate(tensor[:4]):
            out[f"D{t_index}_{d_index}"] = dim
    return out


def build_cases(cell: str, category: str, max_cases: int) -> list[dict]:
    """`Input Shapes` -> the `cases` list carried by both output formats.

    Returns `[]` when the cell is empty. A caller that gets `[]` must exclude
    the kernel: a workset with no shapes has no correctness test.
    """
    signatures = dedupe(parse_input_shapes(cell))[:max_cases]
    cases = []
    for index, signature in enumerate(signatures, start=1):
        case_id = f"case_{index:03d}"
        dimensions = named_dimensions(signature, category)
        cases.append(
            {
                "case_id": case_id,
                "selector": {CASE_SELECTOR_KEY: case_id, **dimensions},
                "shapes": signature,
                "is_primary": index == 1,
            }
        )
    return cases


def argument_records(signature: list[list[int]], dtypes: list[str] | None = None) -> list[dict]:
    """Hyperloom `invocation.arguments[]` records for one signature.

    `dtype` is `""` when the symbol name carried no evidence for that position;
    the caller adds `invocation.arguments[i].dtype` to `missing_fields` in that
    case rather than guessing.
    """
    dtypes = dtypes or []
    records = []
    for index, shape in enumerate(signature):
        records.append(
            {
                "path": f"args[{index}]",
                "position": index,
                "shape": shape,
                "dtype": dtypes[index] if index < len(dtypes) else "",
                "raw": "[" + ",".join(str(d) for d in shape) + "]",
            }
        )
    return records

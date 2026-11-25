###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: adapted from sglang_inproc/tool2_perop/shape_export.py (our SGLang port's stand-in for AKLR's
# `AKLR_EXPORT=1` recorder, AKLR @ 33689332). vLLM changes: aiter callables are proxied where vLLM binds
# them (`vllm.*` module namespaces and the `vllm._aiter_ops.rocm_aiter_ops` facade's static/class methods);
# every call record carries the tool-2 row(s) executing when it was made (`rows`), so a kernel author gets
# the API, its argument shapes and the operator row it belongs to; `write_shapes` adds the per-row operator
# shape file (`operator_shapes.json`) that feeds an operator workset (handoff kbk-workset-glm52-decode).

"""Operator shape export (AKLR_EXPORT equivalent) for vLLM.

During the one eager recording forward only (never inside a timed graph), with AKLR_EXPORT=1:
- `api` records: every aiter callable bound into a loaded `vllm.*` module and every static/class method of
  `rocm_aiter_ops`; each call logs the API, the call site, every argument's spec, and the active tool-2 rows;
- `op` records: every dispatcher op outside aten/prim (`vllm::*`, `aiter::*`, ...) seen by a TorchDispatchMode;
- `module` records: every hooked tool-2 operator's input specs and weight specs.
Identical records are counted, not repeated. Output per rank: `<dir>/rank<r>/calls.pid<pid>.<run>.jsonl`,
`summary.pid<pid>.<run>.json`. If `aklr` is importable, AKLR's own recorder is used (same switch).
Counts are Python dispatches during one forward, never time weights.
"""

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def enabled() -> bool:
    return os.environ.get("AKLR_EXPORT", "0") not in ("", "0", "false", "False")


def aklr_available() -> bool:
    try:
        import aklr  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def spec(value, depth: int = 0):
    """JSON-safe description of one argument."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""), "stride": list(value.stride())}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and depth < 2:
        return [spec(v, depth + 1) for v in value[:16]]
    if isinstance(value, dict) and depth < 2:
        return {str(k): spec(v, depth + 1) for k, v in list(value.items())[:16]}
    if hasattr(value, "__dataclass_fields__") and depth < 2:
        return {"<type>": type(value).__name__, **{k: spec(getattr(value, k, None), depth + 1) for k in list(value.__dataclass_fields__)[:12]}}
    return f"<{type(value).__name__}>"


def _is_aiter_callable(obj) -> bool:
    mod = getattr(obj, "__module__", None) or ""
    return callable(obj) and not isinstance(obj, type) and (mod == "aiter" or mod.startswith("aiter."))


class Recorder:
    def __init__(self, out_dir: str, run: str, rows_now=None):
        self.out_dir = Path(out_dir)
        self.run = run
        self.counts: dict[str, int] = {}
        self.rows: dict[str, dict] = {}
        self.rows_now = rows_now or (lambda: [])

    def add(self, row: dict) -> None:
        key = json.dumps(row, sort_keys=True)
        if key not in self.rows:
            self.rows[key] = row
        self.counts[key] = self.counts.get(key, 0) + 1

    def module_call(self, name: str, mod, args, kwargs) -> None:
        weights = {n: spec(p) for n, p in list(mod.named_parameters(recurse=False))[:8]} if isinstance(mod, torch.nn.Module) else {}
        self.add({"kind": "module", "op": name, "class": f"{type(mod).__module__}.{type(mod).__qualname__}",
                  "args": [spec(a) for a in args], "kwargs": {k: spec(v) for k, v in kwargs.items()}, "weights": weights})

    def _api(self, api, site, a, kw):
        self.add({"kind": "api", "api": api, "site": site, "rows": list(self.rows_now()),
                  "args": [spec(x) for x in a], "kwargs": {k: spec(v) for k, v in kw.items()}})

    @contextlib.contextmanager
    def proxy_aiter(self):
        """Wrap aiter callables bound into loaded vllm modules, and rocm_aiter_ops methods; restore on exit."""
        patched, seen = [], {}
        for mod_name, mod in list(sys.modules.items()):
            if not mod_name.startswith("vllm.") or mod is None:
                continue
            for attr, obj in list(vars(mod).items()):
                if not _is_aiter_callable(obj):
                    continue
                qual = getattr(obj, "__qualname__", attr)
                api = f"{obj.__module__}.{attr if '<locals>' in qual else qual}"

                def wrapped(*a, _o=obj, _api=api, _site=mod_name, **kw):
                    self._api(_api, _site, a, kw)
                    return _o(*a, **kw)

                setattr(mod, attr, wrapped)
                patched.append((mod, attr, obj, False))
                seen[api] = seen.get(api, 0) + 1
        try:
            from vllm._aiter_ops import rocm_aiter_ops

            for attr, raw in list(vars(rocm_aiter_ops).items()):
                if not isinstance(raw, (staticmethod, classmethod)) or attr.startswith("is_") or attr.startswith("_"):
                    continue
                fn = getattr(rocm_aiter_ops, attr)

                def wrapped(*a, _o=fn, _api=f"vllm._aiter_ops.rocm_aiter_ops.{attr}", **kw):
                    self._api(_api, "vllm._aiter_ops", a, kw)
                    return _o(*a, **kw)

                setattr(rocm_aiter_ops, attr, staticmethod(wrapped))
                patched.append((rocm_aiter_ops, attr, raw, True))
                seen[f"rocm_aiter_ops.{attr}"] = 1
        except Exception:  # noqa: BLE001 -- no ROCm facade: module-level proxies only
            pass
        try:
            yield seen
        finally:
            for owner, attr, obj, _ in reversed(patched):
                setattr(owner, attr, obj)

    @contextlib.contextmanager
    def dispatcher_ops(self):
        """Record non-aten dispatcher ops (custom kernels registered with torch.library)."""
        from torch.utils._python_dispatch import TorchDispatchMode

        rec = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                ns = getattr(func, "namespace", "") or str(func).split("::")[0]
                if ns not in ("aten", "prim", "prims", "_c10d_functional", "c10d"):
                    rec.add({"kind": "op", "api": str(func.name() if hasattr(func, "name") else func),
                             "rows": list(rec.rows_now()), "args": [spec(a) for a in args],
                             "kwargs": {k: spec(v) for k, v in kwargs.items()}})
                return func(*args, **kwargs)

        with _Mode():
            yield

    def save(self, extra: dict | None = None) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        path = self.out_dir / f"calls.pid{pid}.{self.run}.jsonl"
        with open(path, "w") as f:
            for key, row in self.rows.items():
                f.write(json.dumps({**row, "count": self.counts[key]}) + "\n")
        by_kind = {k: sum(c for key, c in self.counts.items() if self.rows[key]["kind"] == k) for k in ("api", "module", "op")}
        summary = {
            "run": self.run, "pid": pid, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "distinct_calls": len(self.rows), **{f"{k}_calls": v for k, v in by_kind.items()},
            "apis": sorted({r["api"] for r in self.rows.values() if r["kind"] == "api"}),
            "ops": sorted({r["api"] for r in self.rows.values() if r["kind"] == "op"}),
            **(extra or {}),
        }
        (self.out_dir / f"summary.pid{pid}.{self.run}.json").write_text(json.dumps(summary, indent=2))
        return path


def default_dir(result_dir: str | None = None) -> str:
    return os.environ.get("AKLR_EXPORT_DIR") or str(Path(result_dir or ".") / "aklr_export")


@contextlib.contextmanager
def export_scope(ops: dict, run: str, result_dir: str | None = None, rank: int = 0, rows_now=None):
    """Record shapes during one eager forward when AKLR_EXPORT is set; yields the Recorder (or None)."""
    if not enabled():
        yield None
        return
    if aklr_available():
        from aklr.core.runtime.dispatch import DispatcherRegistry

        yield None
        recorder = DispatcherRegistry.default().recorder
        if recorder is not None:
            recorder.save()
        return
    rec = Recorder(str(Path(default_dir(result_dir)) / f"rank{rank}"), run, rows_now)
    handles = []
    for name, mod in ops.items():
        if isinstance(mod, torch.nn.Module):
            handles.append(mod.register_forward_pre_hook(lambda m, a, k, _n=name: rec.module_call(_n, m, a, k), with_kwargs=True))
    proxied = {}
    try:
        with rec.proxy_aiter() as proxied, rec.dispatcher_ops():
            yield rec
    finally:
        for h in handles:
            h.remove()
    rec.save({"aiter_bindings_proxied": len(proxied), "rank": rank})


# ----------------------------------------------------------------------------------------------
# Per-row operator shapes (machine-readable; input of an operator workset).


def _tensors(value, out: list, depth: int = 0) -> list:
    if isinstance(value, torch.Tensor):
        out.append(value)
    elif isinstance(value, (list, tuple)) and depth < 3:
        for v in value:
            _tensors(v, out, depth + 1)
    elif isinstance(value, dict) and depth < 3:
        for v in value.values():
            _tensors(v, out, depth + 1)
    elif hasattr(value, "__dataclass_fields__") and depth < 3:
        for k in value.__dataclass_fields__:
            _tensors(getattr(value, k, None), out, depth + 1)
    return out


def gemm_dims(tokens: int | None, weight: torch.Tensor | None) -> dict:
    """M/N/K of `y[M,N] = x[M,K] @ W[N,K]^T` from a 2-D weight; packed 4-bit weights double K."""
    if weight is None or weight.dim() != 2 or tokens is None:
        return {}
    n, k = weight.shape
    packed = weight.dtype in (torch.uint8, torch.int8) and getattr(weight, "_bench_fp4", False)
    return {"M": int(tokens), "N": int(n), "K": int(k * (2 if packed else 1)), "weight_dtype": str(weight.dtype).replace("torch.", "")}


def write_shapes(path: Path, header: dict, rows: list[dict]) -> Path:
    """`operator_shapes.json`: {"schema": "tool2-operator-shapes/1", "context": header, "rows": [...]}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "tool2-operator-shapes/1", "context": header, "rows": rows}, indent=1, default=str))
    return path

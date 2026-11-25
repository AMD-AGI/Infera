"""Operator shape export: the AKLR_EXPORT equivalent of tool 2 for an image without the `aklr` package.

AKLR's recorder logs every call that reaches an AKLR-routed API (an `@api` function or an aiter entry point
proxied by `aklr_proxy`) to `calls.pid<p>.<run>.jsonl`, and `python -m aklr export` turns those Points into
per-op Definitions/Workloads. The pinned SGLang image ships no `aklr`, so this module records the same
call-level facts itself, during the one eager recording forward only (never inside a timed graph):

- `api` records: every aiter callable bound into an `sglang.*` module namespace is wrapped for the duration
  of the recording forward; each call logs the API name, the sglang call site, and every argument's spec
  (tensor shape/dtype/stride, scalars verbatim, other objects by type name);
- `op` records: every dispatcher op outside `aten`/`prim` (e.g. `aiter::fused_moe_`, `aiter::gemm_a8w8_blockscale`,
  `sglang::*`) seen by a TorchDispatchMode during that forward, with its argument specs;
- `module` records: every hooked tool-2 operator (`kind|op`) logs its input specs and its weight specs.

Identical (api, site, args) calls are counted, not repeated. Output, one JSON object per line:
`<AKLR_EXPORT_DIR>/calls.pid<pid>.<run>.jsonl` plus `summary.pid<pid>.<run>.json`. If `aklr` is importable
and AKLR_EXPORT=1 the AKLR recorder is used instead (same switch as upstream).
Counts are Python dispatches during one forward, not time weights.
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
    return f"<{type(value).__name__}>"


def _is_aiter_callable(obj) -> bool:
    mod = getattr(obj, "__module__", None) or ""
    return callable(obj) and not isinstance(obj, type) and (mod == "aiter" or mod.startswith("aiter."))


class Recorder:
    def __init__(self, out_dir: str, run: str):
        self.out_dir = Path(out_dir)
        self.run = run
        self.counts: dict[str, int] = {}
        self.rows: dict[str, dict] = {}
        self.patched: list = []

    def add(self, row: dict) -> None:
        key = json.dumps(row, sort_keys=True)
        if key not in self.rows:
            self.rows[key] = row
        self.counts[key] = self.counts.get(key, 0) + 1

    def module_call(self, name: str, mod, args, kwargs) -> None:
        weights = {
            n: spec(p) for n, p in list(mod.named_parameters(recurse=False))[:8]
        }
        self.add(
            {
                "kind": "module",
                "op": name,
                "class": f"{type(mod).__module__}.{type(mod).__qualname__}",
                "args": [spec(a) for a in args],
                "kwargs": {k: spec(v) for k, v in kwargs.items()},
                "weights": weights,
            }
        )

    @contextlib.contextmanager
    def proxy_aiter(self):
        """Wrap aiter callables bound into loaded sglang modules; restore on exit."""
        seen = {}
        for mod_name, mod in list(sys.modules.items()):
            if not mod_name.startswith("sglang.") or mod is None:
                continue
            for attr, obj in list(vars(mod).items()):
                if not _is_aiter_callable(obj):
                    continue
                qual = getattr(obj, "__qualname__", attr)
                # decorator wrappers (aiter's torch_compile_guard) carry a <locals> qualname: use the bound name
                api = f"{obj.__module__}.{attr if '<locals>' in qual else qual}"

                def wrapped(*a, _o=obj, _api=api, _site=mod_name, **kw):
                    self.add(
                        {
                            "kind": "api",
                            "api": _api,
                            "site": _site,
                            "args": [spec(x) for x in a],
                            "kwargs": {k: spec(v) for k, v in kw.items()},
                        }
                    )
                    return _o(*a, **kw)

                setattr(mod, attr, wrapped)
                self.patched.append((mod, attr, obj))
                seen[api] = seen.get(api, 0) + 1
        try:
            yield seen
        finally:
            for mod, attr, obj in reversed(self.patched):
                setattr(mod, attr, obj)
            self.patched.clear()

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
                    rec.add(
                        {
                            "kind": "op",
                            "api": str(func.name() if hasattr(func, "name") else func),
                            "args": [spec(a) for a in args],
                            "kwargs": {k: spec(v) for k, v in kwargs.items()},
                        }
                    )
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
        summary = {
            "run": self.run,
            "pid": pid,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "distinct_calls": len(self.rows),
            "api_calls": sum(c for k, c in self.counts.items() if self.rows[k]["kind"] == "api"),
            "module_calls": sum(c for k, c in self.counts.items() if self.rows[k]["kind"] == "module"),
            "op_calls": sum(c for k, c in self.counts.items() if self.rows[k]["kind"] == "op"),
            "apis": sorted({r["api"] for r in self.rows.values() if r["kind"] == "api"}),
            "ops": sorted({r["api"] for r in self.rows.values() if r["kind"] == "op"}),
            **(extra or {}),
        }
        (self.out_dir / f"summary.pid{pid}.{self.run}.json").write_text(json.dumps(summary, indent=2))
        return path


def default_dir(result_dir: str | None = None) -> str:
    return os.environ.get("AKLR_EXPORT_DIR") or str(Path(result_dir or ".") / "aklr_export")


@contextlib.contextmanager
def export_scope(ops: dict, run: str, result_dir: str | None = None, rank: int = 0):
    """Record shapes during one eager forward when AKLR_EXPORT is set; yields the Recorder (or None)."""
    if not enabled():
        yield None
        return
    if aklr_available():
        # Upstream path: AKLR's own recorder (load_routes + aklr_proxy were installed at setup).
        from aklr.core.runtime.dispatch import DispatcherRegistry

        yield None
        recorder = DispatcherRegistry.default().recorder
        if recorder is not None:
            recorder.save()
        return
    rec = Recorder(str(Path(default_dir(result_dir)) / f"rank{rank}"), run)
    handles = []
    for name, mod in ops.items():

        def hook(m, args, kwargs, _n=name):
            rec.module_call(_n, m, args, kwargs)

        handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
    proxied = {}
    try:
        with rec.proxy_aiter() as proxied, rec.dispatcher_ops():
            yield rec
    finally:
        for h in handles:
            h.remove()
    rec.save({"aiter_bindings_proxied": len(proxied), "rank": rank})

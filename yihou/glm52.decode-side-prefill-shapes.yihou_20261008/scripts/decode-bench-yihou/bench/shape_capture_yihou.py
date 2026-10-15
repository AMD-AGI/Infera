"""Opt-in operator-shape capture for one measured speculative iteration.

Records, for every nn.Module call and every attention-backend forward made while
armed, the input/output tensor shapes and dtypes, tagged with the runner and the
ForwardBatch mode that issued it (TARGET_VERIFY, DRAFT_EXTEND_V2, DECODE, ...).

Only meaningful with --disable-cuda-graph: a replayed graph runs no Python, so
no hook fires. Stdlib-only at import time; torch is imported lazily.
"""
import json
import re
from collections import OrderedDict
from pathlib import Path


def add_shape_cli_args(parser):
    group = parser.add_argument_group("shape capture (opt-in; default off)")
    group.add_argument("--shape-capture-step", type=int, default=-1,
                       help="measured iteration index to capture shapes on; -1 disables")
    group.add_argument("--shape-capture-ranks", default="0",
                       help="comma-separated TP ranks that write a shape file")


def _describe(value, depth=0):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""),
                "stride": list(value.stride()) if value.dim() <= 4 else None}
    if depth >= 2:
        return None
    if isinstance(value, (list, tuple)):
        items = [_describe(item, depth + 1) for item in value]
        return items if any(item is not None for item in items) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _strip(described):
    if isinstance(described, dict) or not isinstance(described, (list, tuple)):
        return described
    return [item for item in described if item is not None]


class ShapeCapture:
    """Hooks are installed for exactly one iteration, then removed."""

    def __init__(self, args, rank, log):
        self.step = args.shape_capture_step
        self.enabled = self.step >= 0 and str(rank) in args.shape_capture_ranks.split(",")
        self.rank, self.log = rank, log
        self.out = Path(args.result_dir) / f"shapes_rank_{rank}_yihou.jsonl"
        self.handles, self.restores = [], []
        self.context = {"runner": None, "mode": None}
        self.records = []
        self.params_seen, self.hooked = set(), set()

    def _wrap_runner(self, label, runner):
        original = runner.forward
        capture = self

        def forward(forward_batch, *args, **kwargs):
            previous = dict(capture.context)
            capture.context.update(
                runner=label, mode=forward_batch.forward_mode.name,
                batch_size=int(forward_batch.batch_size),
                num_tokens=int(forward_batch.input_ids.shape[0]) if forward_batch.input_ids is not None else None,
                seq_lens=forward_batch.seq_lens_cpu.tolist() if forward_batch.seq_lens_cpu is not None else None,
                extend_seq_lens=(forward_batch.extend_seq_lens_cpu
                                 if getattr(forward_batch, "extend_seq_lens_cpu", None) is not None else None),
            )
            capture.records.append({"event": "forward_begin", **capture.context})
            try:
                return original(forward_batch, *args, **kwargs)
            finally:
                capture.context = previous

        runner.forward = forward
        self.restores.append(lambda: runner.__dict__.pop("forward", None))

    def _wrap_attn_backend(self, label, backend):
        capture = self
        for name in ("forward", "forward_extend", "forward_decode"):
            original = getattr(backend, name, None)
            if original is None:
                continue

            def make(original=original, name=name):
                def wrapped(*args, **kwargs):
                    capture.records.append({
                        "event": "attn_backend", "backend": type(backend).__name__, "method": name,
                        "owner": label, **{k: capture.context.get(k) for k in ("runner", "mode")},
                        "args": _strip(_describe(list(args))),
                        "kwargs": {k: _describe(v) for k, v in kwargs.items() if _describe(v) is not None},
                    })
                    return original(*args, **kwargs)
                return wrapped

            setattr(backend, name, make())
            self.restores.append(lambda backend=backend, name=name: backend.__dict__.pop(name, None))

    def _hook_model(self, label, model):
        capture = self

        def pre(module, args, kwargs, qualname):
            record = {
                "event": "module", "module": qualname, "class": type(module).__name__,
                "owner": label, **{k: capture.context.get(k) for k in ("runner", "mode")},
                "args": _strip(_describe(list(args))),
                "kwargs": {k: _describe(v) for k, v in kwargs.items() if _describe(v) is not None},
            }
            if id(module) not in capture.params_seen:
                # Direct parameters/buffers only: weight shape + storage dtype (MXFP4/FP8 packing visible here).
                capture.params_seen.add(id(module))
                tensors = {**dict(module.named_parameters(recurse=False)),
                           **dict(module.named_buffers(recurse=False))}
                record["params"] = {name: _describe(t) for name, t in tensors.items()}
            capture.records.append(record)

        def post(module, args, kwargs, output, qualname):
            capture.records.append({"event": "module_out", "module": qualname,
                                    **{k: capture.context.get(k) for k in ("runner", "mode")},
                                    "output": _strip(_describe(output))})

        for qualname, module in model.named_modules():
            # Target and draft share some modules (rotary, embed/lm_head): hook each object once.
            if not qualname or id(module) in self.hooked:
                continue
            self.hooked.add(id(module))
            self.handles.append(module.register_forward_pre_hook(
                lambda m, a, k, q=qualname: pre(m, a, k, q), with_kwargs=True))
            self.handles.append(module.register_forward_hook(
                lambda m, a, k, o, q=qualname: post(m, a, k, o, q), with_kwargs=True))

    def _wrap_triton(self):
        """Record every Triton/Gluon kernel launch: name, named tensor args, scalar args, grid."""
        try:
            from triton.runtime.jit import JITFunction
        except ImportError:
            return
        capture, original = self, JITFunction.run

        def run(fn, *args, grid, warmup, **kwargs):
            if not warmup:
                names = list(getattr(fn, "arg_names", []))
                named = dict(zip(names, args))
                named.update(kwargs)
                tensors = {k: _describe(v) for k, v in named.items() if hasattr(v, "shape")}
                scalars = {k: v for k, v in named.items()
                           if isinstance(v, (int, float, bool, str)) and not hasattr(v, "shape")}
                capture.records.append({
                    "event": "triton", "kernel": getattr(getattr(fn, "fn", None), "__name__", str(fn)),
                    **{k: capture.context.get(k) for k in ("runner", "mode")},
                    "tensors": tensors, "scalars": scalars,
                    "grid": list(grid) if isinstance(grid, (tuple, list)) else "callable",
                })
            return original(fn, *args, grid=grid, warmup=warmup, **kwargs)

        JITFunction.run = run
        self.restores.append(lambda: setattr(JITFunction, "run", original))

    # Module-global helpers the DSA indexer calls by name; patching the module attribute is
    # enough because they are resolved at call time. Missing names are skipped silently.
    FUNCTION_TARGETS = {
        "sglang.srt.layers.attention.dsa.dsa_indexer": (
            "act_quant", "aiter_paged_mqa_logits", "indexer_k_quant_and_cache",
            "fp8_mqa_logits", "sgl_fp8_mqa_logits", "fp8_paged_mqa_logits"),
    }

    def _wrap_functions(self):
        import importlib
        capture = self
        for module_name, attrs in self.FUNCTION_TARGETS.items():
            module = importlib.import_module(module_name)
            for attr in attrs:
                original = getattr(module, attr, None)
                if not callable(original):
                    continue

                def make(original=original, attr=attr):
                    def wrapped(*args, **kwargs):
                        result = original(*args, **kwargs)
                        capture.records.append({
                            "event": "function", "function": f"{module_name}.{attr}",
                            **{k: capture.context.get(k) for k in ("runner", "mode")},
                            "args": _strip(_describe(list(args))),
                            "kwargs": {k: _describe(v) for k, v in kwargs.items() if _describe(v) is not None},
                            "output": _strip(_describe(result)),
                        })
                        return result
                    return wrapped

                setattr(module, attr, make())
                self.restores.append(lambda module=module, attr=attr, original=original:
                                     setattr(module, attr, original))

    def arm(self, runners):
        """runners: {label: ModelRunner}. Wraps forward, hooks model + attn backend."""
        seen = set()
        for label, runner in runners.items():
            if id(runner) in seen:
                continue
            seen.add(id(runner))
            self._wrap_runner(label, runner)
            self._hook_model(label, runner.model)
            backend = getattr(runner, "attn_backend", None)
            if backend is not None:
                self._wrap_attn_backend(label, backend)
        self._wrap_triton()
        self._wrap_functions()
        self.log(self.rank, f"shape capture armed on {sorted(runners)}; {len(self.handles)} hooks")

    def disarm_and_write(self, extra):
        for handle in self.handles:
            handle.remove()
        for restore in reversed(self.restores):
            restore()
        self.handles, self.restores = [], []
        with self.out.open("w") as f:
            f.write(json.dumps({"event": "meta", **extra}) + "\n")
            for record in self.records:
                f.write(json.dumps(record) + "\n")
        self.log(self.rank, f"shape capture wrote {len(self.records)} records to {self.out}")


LAYER = re.compile(r"\.layers\.(\d+)\.")


def summarize(path):
    """Collapse per-layer records into unique (runner, mode, module-pattern, shapes) rows."""
    rows = OrderedDict()
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record["event"] not in ("module", "attn_backend"):
            continue
        name = record.get("module") or f"<{record['backend']}.{record['method']}>"
        pattern = LAYER.sub(".layers.*.", name)
        key = (record.get("runner"), record.get("mode"), pattern,
               json.dumps(record.get("args")), json.dumps(record.get("kwargs"), sort_keys=True))
        if key in rows:
            rows[key]["count"] += 1
            layer = LAYER.search(name)
            if layer:
                rows[key]["layers"].append(int(layer.group(1)))
        else:
            layer = LAYER.search(name)
            rows[key] = {"runner": key[0], "mode": key[1], "module": pattern,
                         "class": record.get("class", record.get("backend")), "args": record.get("args"),
                         "kwargs": record.get("kwargs"), "count": 1,
                         "layers": [int(layer.group(1))] if layer else []}
    return list(rows.values())

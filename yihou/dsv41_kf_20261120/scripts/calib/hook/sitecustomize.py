"""Bounded rocprofiler-sdk json-tool window for a running vLLM (V1, model runner v2) worker.

Port of the Muse hook (infera.muse yihou/muse_agentx_opt_20261027/research/calib_profiler/hook/sitecustomize.py,
2026-11-13) from SGLang ModelRunner.forward to vLLM `vllm.v1.worker.gpu.model_runner.GPUModelRunner`:
  * execute_model -> one step record, stage DECODE (every scheduled request runs 1 + its spec tokens), MIXED (at
    least one request runs a prefill chunk), DUMMY (DP padding / dummy_run); token counts recorded per step;
  * sample_tokens -> stage SAMPLE (sampling, rejection, DSpark `speculator.propose` = the draft forward).
Kernel -> step attribution is by the launching HIP API call's CPU time (CLOCK_BOOTTIME) inside a step's [t0, t1].

Loaded through PYTHONPATH in every Python process. Does nothing unless CALIB_ENABLE=1.
Control files in CALIB_TRACE_DIR: start_window (arm; the window opens at the next execute_model), finalize
(write trace-<pid>.json, then finalized-<pid>). Outputs per process: trace-<pid>.json, steps-<pid>.json, calib-<pid>.log.

Env: CALIB_ENABLE=1, CALIB_SDK_BUILD=<build dir>, CALIB_TRACE_DIR=<absolute dir>, CALIB_STEPS=60 (non-dummy
execute_model calls per window), CALIB_MAX_WINDOW_S=30.
"""
import os
import sys


def _chain_shadowed_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in list(sys.path):
        try:
            if not entry or os.path.abspath(entry) == here:
                continue
            cand = os.path.join(entry, "sitecustomize.py")
            if os.path.isfile(cand):
                import runpy

                runpy.run_path(cand, run_name="sitecustomize_chained")
                return cand
        except Exception as exc:  # never break interpreter startup
            sys.stderr.write(f"[calib] chained sitecustomize {entry} failed: {exc!r}\n")
            return None
    return None


def _setup():
    import ctypes
    import json
    import threading
    import time
    from pathlib import Path

    root = Path(os.environ["CALIB_TRACE_DIR"])
    sdk = Path(os.environ["CALIB_SDK_BUILD"])
    steps_target = int(os.environ.get("CALIB_STEPS", "60"))
    max_window_s = float(os.environ.get("CALIB_MAX_WINDOW_S", "30"))
    pid = os.getpid()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["ROCPROFILER_TOOL_OUTPUT_FILE"] = str(root / f"trace-{pid}.json")
    logf = open(root / f"calib-{pid}.log", "a", buffering=1)

    def now():
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)

    def log(msg):
        logf.write(f"{now()} {msg}\n")

    import torch  # loads HIP -> rocprofiler-register -> json-tool tool_init

    tool = ctypes.CDLL(str(sdk / "lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so"))
    tool.json_tool_pause_until_graph.restype = None
    tool.json_tool_finalize_idle_worker.restype = None
    tool.json_tool_start.restype = ctypes.c_int
    tool.json_tool_stop.restype = ctypes.c_int
    maps = [l.split()[-1] for l in open("/proc/self/maps") if "rocprofiler-sdk" in l]
    log(f"init argv={sys.argv[:3]} sdk_maps={sorted(set(maps))}")

    lock = threading.Lock()
    st = {"state": "pre", "t_open": None, "t_close": None, "n_steps": 0, "steps": [], "close_req": False,
          "dp_rank": None}
    tool.json_tool_pause_until_graph()

    def close_window(reason, sync=True):
        # caller holds lock
        if sync:
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                log(f"sync failed {exc!r}")
        rc = tool.json_tool_stop()
        st["t_close"] = now()
        st["state"] = "done"
        out = {"pid": pid, "dp_rank": st["dp_rank"], "engine": "vllm", "clock": "CLOCK_BOOTTIME",
               "window_start_ns": st["t_open"], "window_end_ns": st["t_close"], "close_reason": reason,
               "n_steps": st["n_steps"], "steps": st["steps"]}
        tmp = root / f".steps-{pid}.json.tmp"
        tmp.write_text(json.dumps(out))
        tmp.rename(root / f"steps-{pid}.json")
        log(f"window closed reason={reason} rc={rc} records={len(st['steps'])} steps={st['n_steps']}")

    def describe(so, dummy):
        if dummy or so is None:
            return {"stage": "DUMMY", "reqs": 0, "tokens": 0, "prefill_tokens": 0, "prefill_reqs": 0,
                    "spec_tokens": 0, "max_q": 0}
        nst = so.num_scheduled_tokens
        spec = getattr(so, "scheduled_spec_decode_tokens", {}) or {}
        pre_tok = pre_req = spec_tok = mx = 0
        for rid, n in nst.items():
            k = len(spec.get(rid, ()))
            spec_tok += k
            mx = max(mx, n)
            if n > 1 + k:
                pre_tok += n
                pre_req += 1
        return {"stage": "MIXED" if pre_req else "DECODE", "reqs": len(nst),
                "tokens": int(so.total_num_scheduled_tokens), "prefill_tokens": pre_tok, "prefill_reqs": pre_req,
                "spec_tokens": spec_tok, "max_q": mx}

    def maybe_open(self):
        if st["state"] == "armed":
            with lock:
                if st["state"] == "armed":
                    pc = getattr(getattr(self, "vllm_config", None), "parallel_config", None)
                    st["dp_rank"] = getattr(pc, "data_parallel_rank", None)
                    rc = tool.json_tool_start()
                    st["t_open"] = now()
                    st["state"] = "open"
                    log(f"window open rc={rc} dp_rank={st['dp_rank']}")

    def maybe_close():
        with lock:
            if st["state"] == "open" and (st["n_steps"] >= steps_target or st["close_req"]):
                close_window("steps" if st["n_steps"] >= steps_target else "max_window_s")

    def wrap_execute(orig):
        def execute_model(self, scheduler_output, *a, **k):
            maybe_open(self)
            if st["state"] != "open":
                return orig(self, scheduler_output, *a, **k)
            dummy = bool(k.get("dummy_run", False))
            t0 = now()
            r = orig(self, scheduler_output, *a, **k)
            t1 = now()
            try:
                rec = describe(scheduler_output, dummy)
                rec.update({"t0": t0, "t1": t1})
                st["steps"].append(rec)
                if rec["stage"] != "DUMMY":
                    st["n_steps"] += 1
            except Exception as exc:
                log(f"label failed {exc!r}")
            # The window closes after the matching sample_tokens, so the last step's draft forward is inside it.
            return r

        execute_model._calib_wrapped = True
        return execute_model

    def wrap_sample(orig):
        def sample_tokens(self, *a, **k):
            if st["state"] != "open":
                return orig(self, *a, **k)
            t0 = now()
            r = orig(self, *a, **k)
            t1 = now()
            st["steps"].append({"stage": "SAMPLE", "t0": t0, "t1": t1})
            maybe_close()
            return r

        sample_tokens._calib_wrapped = True
        return sample_tokens

    def wrap_dummy(orig):
        # DP padding steps go through execute_model(dummy_run=True) inside the runner; the worker-level entry is
        # wrapped too so that a dummy step of an idle DP rank is labelled even if the runner path changes.
        def execute_dummy_batch(self, *a, **k):
            if st["state"] != "open":
                return orig(self, *a, **k)
            t0 = now()
            r = orig(self, *a, **k)
            st["steps"].append({"stage": "DUMMY_BATCH", "t0": t0, "t1": now()})
            return r

        execute_dummy_batch._calib_wrapped = True
        return execute_dummy_batch

    def patch_runner(mod):
        cls = getattr(mod, "GPUModelRunner", None)
        if cls is None or getattr(cls.execute_model, "_calib_wrapped", False):
            return
        cls.execute_model = wrap_execute(cls.execute_model)
        cls.sample_tokens = wrap_sample(cls.sample_tokens)
        log("patched GPUModelRunner.execute_model/sample_tokens")

    def patch_worker(mod):
        cls = getattr(mod, "Worker", None)
        f = getattr(cls, "execute_dummy_batch", None) if cls is not None else None
        if f is None or getattr(f, "_calib_wrapped", False):
            return
        cls.execute_dummy_batch = wrap_dummy(f)
        log("patched Worker.execute_dummy_batch")

    import importlib.abc
    import importlib.machinery

    targets = {"vllm.v1.worker.gpu.model_runner": patch_runner, "vllm.v1.worker.gpu_worker": patch_worker}

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner, fn):
            self.inner, self.fn = inner, fn

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            try:
                self.fn(module)
            except Exception as exc:
                log(f"patch failed {module.__name__} {exc!r}")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target_mod=None):
            fn = targets.get(name)
            if fn is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, fn)
            return spec

    pending = False
    for name, fn in targets.items():
        if name in sys.modules:
            fn(sys.modules[name])
        else:
            pending = True
    if pending:
        sys.meta_path.insert(0, _Finder())

    def control():
        last_pause = 0.0
        while True:
            time.sleep(0.2)
            s = st["state"]
            if s == "pre" and time.monotonic() - last_pause > 2.0:
                # Safety net: if tool_init ran after our first pause, stop dispatch tracing again.
                tool.json_tool_pause_until_graph()
                last_pause = time.monotonic()
            if s == "pre" and (root / "start_window").exists():
                with lock:
                    if st["state"] == "pre":
                        st["state"] = "armed"
                        log("armed")
            if s == "open" and st["t_open"] and (now() - st["t_open"]) / 1e9 > max_window_s:
                st["close_req"] = True
                time.sleep(5.0)
                with lock:
                    if st["state"] == "open":  # no step arrived: close from this thread
                        close_window("max_window_s_idle", sync=False)
            if (root / "finalize").exists() and s in ("done", "pre", "armed"):
                with lock:
                    if st["state"] in ("armed", "open"):
                        close_window("finalize_requested", sync=False)
                    tool.json_tool_finalize_idle_worker()
                (root / f"finalized-{pid}").write_text("done\n")
                log("finalized")
                return

    threading.Thread(target=control, name="calib-control", daemon=True).start()


if os.environ.get("CALIB_ENABLE") == "1" and os.environ.get("CALIB_SKIP") != "1":
    try:
        _setup()
    except Exception as _exc:
        sys.stderr.write(f"[calib] setup failed, profiler window disabled: {_exc!r}\n")

_chain_shadowed_sitecustomize()

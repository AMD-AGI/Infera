"""Per-step recorder for vLLM v1 GPU model runner (v2 runner, vLLM 0.34): diagnostics only.

Loaded in every Python process through vpd_steplog.pth; does nothing unless VPD_STEPLOG_DIR is set. When
vllm.v1.worker.gpu.model_runner is imported it wraps GPUModelRunner.execute_model and the module's
dispatch_cg_and_sync_dp, and writes one CSV line per execute_model call to $VPD_STEPLOG_DIR/steps-r<dp>-<pid>.csv:

  S  wall  dt_ms  cpu_ms  rank  dummy  reqs  toks  maxq  n_dec  n_small  n_big  big_toks  n_new  n_new_small
     new_big_toks  n_preempted  mode  desc_toks  desc_reqs  uniform  ctx_sum  ctx_max
    ctx_*   context (computed tokens before the step) of the scheduled requests: sum and largest
    dt_ms   time since the previous call on this process (the step cadence under async scheduling)
    cpu_ms  time spent inside execute_model
    n_dec / n_small / n_big   requests scheduled with exactly decode_query_len / fewer / more than 16 tokens
    n_new_small  new requests with <= decode_query_len tokens (a request entering decode after its remote KV)
    mode / desc_*  the CUDA graph descriptor agreed across DP ranks for this step
and every VPD_STEPLOG_MEM_EVERY (500) steps a memory line:
  M  wall  rank  free_b  total_b  reserved_b  allocated_b  max_reserved_b
Errors are swallowed: the recorder must never change what the engine does.
"""

import os
import sys

_DIR = os.environ.get("VPD_STEPLOG_DIR")
_TARGET = "vllm.v1.worker.gpu.model_runner"


def _patch(m):
    import atexit
    import functools
    import time

    runner_cls = m.GPUModelRunner
    orig_exec = runner_cls.execute_model
    orig_dispatch = m.dispatch_cg_and_sync_dp
    mem_every = int(os.environ.get("VPD_STEPLOG_MEM_EVERY", "500"))
    st = {"desc": None, "prev": None, "buf": [], "last_flush": time.monotonic(), "f": None, "n": 0, "live": False}

    @functools.wraps(orig_dispatch)
    def dispatch(*a, **k):
        out = orig_dispatch(*a, **k)
        st["desc"] = out[0]
        return out

    m.dispatch_cg_and_sync_dp = dispatch

    def _flush_at_exit():
        if st["buf"]:
            try:
                f = _file(st.get("rank", -1))
                f.write("".join(st["buf"]))
                f.flush()
            except Exception:
                pass

    atexit.register(_flush_at_exit)

    def _file(rank):
        if st["f"] is None:
            os.makedirs(_DIR, exist_ok=True)
            path = os.path.join(_DIR, f"steps-r{rank}-{os.getpid()}.csv")
            st["f"] = open(path, "a", buffering=1 << 20)
            st["f"].write(
                "#S,wall,dt_ms,cpu_ms,rank,dummy,reqs,toks,maxq,n_dec,n_small,n_big,big_toks,n_new,n_new_small,"
                "new_big_toks,n_preempted,mode,desc_toks,desc_reqs,uniform,ctx_sum,ctx_max\n"
                "#M,wall,rank,free_b,total_b,reserved_b,allocated_b,max_reserved_b\n"
            )
        return st["f"]

    def _record(self, so, dummy, t0, t1):
        rank = st["rank"] = getattr(self, "dp_rank", -1)
        dq = getattr(self, "decode_query_len", 0) or 0
        nst = getattr(so, "num_scheduled_tokens", None) or {}
        vals = list(nst.values())
        n_dec = n_small = n_big = big = 0
        for v in vals:
            if v == dq:
                n_dec += 1
            elif v < dq:
                n_small += 1
            if v > 16:
                n_big += 1
                big += v
        new = getattr(so, "scheduled_new_reqs", None) or []
        n_new_small = new_big = 0
        for r in new:
            v = nst.get(getattr(r, "req_id", None), 0)
            if v <= dq:
                n_new_small += 1
            elif v > 16:
                new_big += v
        pre = getattr(so, "preempted_req_ids", None) or ()
        # Context of the scheduled requests (computed tokens before this step).
        ctx = [getattr(r, "num_computed_tokens", 0) or 0 for r in new]
        cached = getattr(so, "scheduled_cached_reqs", None)
        if cached is not None:
            ctx.extend(getattr(cached, "num_computed_tokens", None) or ())
        d = st["desc"]
        mode = getattr(getattr(d, "cg_mode", None), "name", "?") if d is not None else "-"
        dt = (t0 - st["prev"]) * 1e3 if st["prev"] is not None else 0.0
        st["prev"] = t0
        st["buf"].append(
            f"S,{time.time():.3f},{dt:.2f},{(t1 - t0) * 1e3:.2f},{rank},{int(dummy)},{len(vals)},"
            f"{getattr(so, 'total_num_scheduled_tokens', 0)},{max(vals) if vals else 0},{n_dec},{n_small},{n_big},{big},"
            f"{len(new)},{n_new_small},{new_big},{len(pre)},{mode},{getattr(d, 'num_tokens', '')},"
            f"{getattr(d, 'num_reqs', '')},{getattr(d, 'uniform_token_count', '')},{sum(ctx)},{max(ctx, default=0)}\n"
        )
        st["n"] += 1
        if mem_every > 0 and st["n"] % mem_every == 0:
            try:
                import torch

                free, total = torch.cuda.mem_get_info()
                st["buf"].append(
                    f"M,{time.time():.3f},{rank},{free},{total},{torch.cuda.memory_reserved()},"
                    f"{torch.cuda.memory_allocated()},{torch.cuda.max_memory_reserved()}\n"
                )
            except Exception:
                pass
        now = time.monotonic()
        if len(st["buf"]) >= 2000 or now - st["last_flush"] > 5:
            f = _file(rank)
            f.write("".join(st["buf"]))
            f.flush()
            st["buf"].clear()
            st["last_flush"] = now

    @functools.wraps(orig_exec)
    def execute_model(self, scheduler_output, *a, **k):
        st["desc"] = None
        t0 = time.perf_counter()
        out = orig_exec(self, scheduler_output, *a, **k)
        t1 = time.perf_counter()
        try:
            dummy = bool(k.get("dummy_run", False))
            # Start-up profiling and graph capture are dummy runs too; record from the first real step on.
            if not dummy:
                st["live"] = True
            if st["live"]:
                _record(self, scheduler_output, dummy, t0, t1)
        except Exception as e:  # noqa: BLE001
            if not st.get("warned"):
                st["warned"] = True
                sys.stderr.write(f"[vpd_steplog] record failed: {e!r}\n")
        return out

    runner_cls.execute_model = execute_model
    sys.stderr.write(f"[vpd_steplog] patched {_TARGET} in pid {os.getpid()} -> {_DIR}\n")


if _DIR:
    import importlib.abc

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            orig_exec_module = loader.exec_module

            def exec_module(module):
                orig_exec_module(module)
                try:
                    _patch(module)
                except Exception as e:  # noqa: BLE001
                    sys.stderr.write(f"[vpd_steplog] patch failed: {e!r}\n")

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

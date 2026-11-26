"""Keep vLLM's DP engines stepping in lockstep instead of pausing when idle (decode DP2 deadlock, EXPERIMENT 4.11).

A DP rank with no work runs a dummy batch so that its peer's forward finds a partner for every MoE collective. After a
finish-sync all-reduce (DPEngineCoreProc._has_global_unfinished_reqs, every dp_sync_interval steps) finds every rank
idle, the wave pauses: an idle rank no longer runs dummy batches and waits for input until a START_DP_WAVE wakes it.
In the R17-A deadlock (11-26 06:04, per-step logs flushed every step) rank 0 had paused (its last step: 0 tokens, no
forward, no dummy batch after it) while rank 1 entered a real 6-token step whose collectives then waited forever (both
GPUs busy, every Python thread idle), i.e. rank 1 started work that never woke rank 0. With this patch the all-reduce
still runs on its schedule (so the collectives stay matched) but the engines always count as running, so an idle rank
keeps running dummy batches and can never be absent from a peer's step; the cost is dummy forwards while decode is
idle. vLLM's own fault-tolerance mode does the same when its coordinator is disabled ("never idle-pause").
Loaded through dp_never_pause.pth; does nothing unless VLLM_DP_NEVER_PAUSE=1.
"""

import os
import sys

_ON = os.environ.get("VLLM_DP_NEVER_PAUSE") == "1"
_TARGET = "vllm.v1.engine.core"


def _patch(m):
    cls = m.DPEngineCoreProc
    orig = cls._has_global_unfinished_reqs
    seen = {"n": 0}

    def _has_global_unfinished_reqs(self, local_unfinished):
        real = orig(self, local_unfinished)
        if not real and seen["n"] < 3:
            seen["n"] += 1
            sys.stderr.write(f"[dp_never_pause] pid {os.getpid()} dp_rank {getattr(self, 'dp_rank', '?')}: "
                             f"all ranks idle at step {self.step_counter}; staying in lockstep\n")
        return True

    cls._has_global_unfinished_reqs = _has_global_unfinished_reqs
    sys.stderr.write(f"[dp_never_pause] patched {_TARGET}.DPEngineCoreProc in pid {os.getpid()}\n")


if _ON:
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
                    sys.stderr.write(f"[dp_never_pause] patch failed: {e!r}\n")

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

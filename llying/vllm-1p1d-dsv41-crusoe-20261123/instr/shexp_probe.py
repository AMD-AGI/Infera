"""Log why vLLM's MoE runner does (not) run the shared experts on its auxiliary stream: diagnostics only.

R15 set VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=65536 on the prefill leg, yet its profile shows every shared-expert
GEMM on the main stream (EXPERIMENT 4.11). This wraps SharedExperts._determine_shared_experts_order and, for the first
SHEXP_PROBE_N (4) calls per process with more than 256 tokens, writes the decision and every input it depends on to
stderr: the threshold seen, the aux stream, the modular kernel's internal-overlap answer, the disable conditions.
Loaded through shexp_probe.pth; does nothing unless SHEXP_PROBE=1. Errors are swallowed.
"""

import os
import sys

_ON = os.environ.get("SHEXP_PROBE") == "1"
_N = int(os.environ.get("SHEXP_PROBE_N", "4"))
_TARGET = "vllm.model_executor.layers.fused_moe.runner.shared_experts"


def _patch(m):
    import vllm.envs as envs

    orig = m.SharedExperts._determine_shared_experts_order
    seen = {"n": 0}

    def wrapped(self, hidden_states):
        order = orig(self, hidden_states)
        try:
            if hidden_states.shape[0] > 256 and seen["n"] < _N:
                seen["n"] += 1
                try:
                    mk = self._mk_can_overlap_shared_experts()
                except Exception as e:  # noqa: BLE001
                    mk = repr(e)
                sys.stderr.write(
                    f"[shexp_probe] pid {os.getpid()} tokens {hidden_states.shape[0]} -> {order}; "
                    f"threshold {envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD} "
                    f"(env {os.environ.get('VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD')}), "
                    f"stream {self._stream is not None}, mk_overlap {mk}, "
                    f"disable_overlap {self._disable_shared_experts_overlap}, "
                    f"layer {type(self._layer).__name__}\n"
                )
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[shexp_probe] log failed: {e!r}\n")
        return order

    m.SharedExperts._determine_shared_experts_order = wrapped
    sys.stderr.write(f"[shexp_probe] patched {_TARGET} in pid {os.getpid()}\n")


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
                    sys.stderr.write(f"[shexp_probe] patch failed: {e!r}\n")

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

"""Make vLLM's auxiliary stream (vllm.utils.torch_utils.aux_stream) a high-priority stream.

The MoE runner puts the shared experts on aux_stream() (VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD), yet in the prefill
profile their GEMMs ran on the same hardware queue as the main stream (EXPERIMENT 4.11): HIP maps streams round-robin
onto GPU_MAX_HW_QUEUES (4) hardware queues in creation order, and the engine creates ~200 streams, so the aux stream can
share the main stream's queue and serialise behind it, although the FlyDSL GEMMs launch on the current stream
(aiter/ops/flydsl/gemm_a8w8_bpreshuffle_8wave.py) and a two-stream microbenchmark hides them fully under the MoE
all-gather (rccl_overlap_test.py). High-priority streams come from their own queue pool, so this aux stream cannot share
the normal-priority main stream's queue. Loaded through aux_prio.pth; does nothing unless AUX_STREAM_PRIORITY=1.
"""

import os
import sys

_ON = os.environ.get("AUX_STREAM_PRIORITY") == "1"
_TARGET = "vllm.utils.torch_utils"


def _patch(m):
    import torch

    state = {"stream": None}

    def aux_stream():
        from vllm.platforms import current_platform

        if state["stream"] is None and current_platform.is_cuda_alike():
            lo, hi = torch.cuda.Stream.priority_range()
            state["stream"] = torch.cuda.Stream(priority=hi)
            sys.stderr.write(f"[aux_prio] pid {os.getpid()}: aux stream {state['stream'].cuda_stream:#x} "
                             f"priority {hi} (range {lo}..{hi})\n")
        return state["stream"]

    m.aux_stream = aux_stream
    sys.stderr.write(f"[aux_prio] patched {_TARGET}.aux_stream in pid {os.getpid()}\n")


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
                    sys.stderr.write(f"[aux_prio] patch failed: {e!r}\n")

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())

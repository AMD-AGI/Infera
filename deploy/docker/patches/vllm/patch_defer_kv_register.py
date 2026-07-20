#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Defer the Mooncake KV registration until AFTER warmup + cudagraph capture.

Baked into the engine image by Dockerfile.vllm (the `for f in vllm-patches/*.py`
loop). Idempotent and self-locating. Patches two vLLM files.

WHY: with bare ibv_reg_mr (dma-buf GPUDirect removed — see
mooncake_cpp/apply_mooncake_cpp_patches.sh), registering the KV pool at its normal
point (inside gpu_model_runner.initialize_kv_cache, BEFORE compile_or_warm_up_model)
still trips a decode-boot crash at high util: one TP worker's
compile_or_warm_up_model returns None and vLLM's
`max(t.language_model for t in compilation_times)` aggregation dies with
`AttributeError: 'NoneType' object has no attribute 'language_model'`. Deferring the
Mooncake register_kv_caches to the very END of compile_or_warm_up_model (after all
warmup, cudagraph capture, and the post-capture sampler/pooler dummy runs) avoids
it. Registration still completes before the engine reports ready, so P<->D pairing
is unaffected (validated: Kimi-K2.6 util0.8 stable + DeepSeek-V4-Pro correct).

Always on. Run inside a container with vLLM installed:
    docker exec <ctr> python3 patch_defer_kv_register.py
Exits 0 (no-op) if an anchor isn't present.
"""

import importlib.util as u
import os
import py_compile
import sys

spec = u.find_spec("vllm")
if not spec:
    print("defer-kv: vllm not found")
    sys.exit(0)
V = os.path.dirname(spec.origin)

# ---- 1) gpu_model_runner.py: stash the kv_caches instead of registering now ----
f1 = os.path.join(V, "v1/worker/gpu_model_runner.py")
s1 = open(f1).read()
if "[infera-defer]" not in s1:
    anchor1 = "            else:\n                kv_transfer_group.register_kv_caches(kv_caches)\n"
    if anchor1 not in s1:
        print("defer-kv: model_runner anchor not found — skipping")
        sys.exit(0)
    repl1 = (
        "            else:\n"
        "                # [infera-defer] stash; register at END of compile_or_warm_up_model\n"
        "                self._infera_deferred_kv_caches = kv_caches\n"
        "                self._infera_deferred_kv_group = kv_transfer_group\n"
    )
    s1 = s1.replace(anchor1, repl1, 1)
    tmp = f1 + ".tmp"
    open(tmp, "w").write(s1)
    py_compile.compile(tmp, doraise=True)
    os.replace(tmp, f1)
    print("defer-kv: patched gpu_model_runner.py")
else:
    print("defer-kv: gpu_model_runner already patched")

# ---- 2) gpu_worker.py: do the deferred registration right before the return ----
f2 = os.path.join(V, "v1/worker/gpu_worker.py")
s2 = open(f2).read()
if "[infera-defer]" not in s2:
    anchor2 = "        return CompilationTimes("
    if s2.count(anchor2) != 1:
        print(f"defer-kv: gpu_worker anchor count={s2.count(anchor2)} (need 1) — skipping")
        sys.exit(0)
    inject2 = (
        "        # [infera-defer] all warmup/capture/sampler forwards done — NOW register\n"
        "        # the KV with Mooncake (deferred from initialize_kv_cache to dodge the\n"
        "        # compile_or_warm_up_model None-aggregation crash at high util).\n"
        "        _dkv = getattr(self.model_runner, '_infera_deferred_kv_caches', None)\n"
        "        _grp = getattr(self.model_runner, '_infera_deferred_kv_group', None)\n"
        "        if _dkv is not None and _grp is not None:\n"
        "            from vllm.logger import init_logger as _il\n"
        "            _il(__name__).info('[infera-defer] registering KV with Mooncake at END of warmup+capture')\n"
        "            _grp.register_kv_caches(_dkv)\n"
        "            self.model_runner._infera_deferred_kv_caches = None\n" + anchor2
    )
    s2 = s2.replace(anchor2, inject2, 1)
    tmp = f2 + ".tmp"
    open(tmp, "w").write(s2)
    py_compile.compile(tmp, doraise=True)
    os.replace(tmp, f2)
    print("defer-kv: patched gpu_worker.py")
else:
    print("defer-kv: gpu_worker already patched")
print("defer-kv: DONE")

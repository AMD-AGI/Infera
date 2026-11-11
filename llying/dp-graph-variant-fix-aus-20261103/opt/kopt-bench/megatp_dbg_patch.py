#!/usr/bin/env python3
"""Instrument MegaMoeTP in place (inside a throw-away container, on the image's own aiter source): with
AITER_MEGAMOE_TP_DBG=1 at compile time, thread 0 of every CTA stores s_memrealtime (100 MHz) at five points into a
debug buffer passed in the fused-tail slot tn_res (unused without the tail): 0 kernel entry, 1 after quantizing this
rank's rows, 2 compute warps done with their GEMM units, 3 all roles done, 4 after finish. Layout: buf[cta * 8 + k].
The instrumented kernel gets a "_dbg" name suffix so it never shares a compile-cache entry with the normal one.
Host side: engine.dbg_buf (int64 tensor of n_cta * 8) is passed as tn_res when no tail is given.
Every edit asserts its anchor occurs exactly once.

Usage (in the container): python3 megatp_dbg_patch.py
"""
import importlib.util
import os


def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{os.path.basename(path)}: anchor found {n} times: {old[:70]!r}"
        s = s.replace(old, new)
    open(path, "w").write(s)


base = os.path.dirname(importlib.util.find_spec("aiter").origin) + "/ops/flydsl/kernels/mega_moe_tp"
kern = base + "/mega_moe_tp_kernel.py"
eng = base + "/mega_moe_tp.py"
if "_dbg_now" in open(kern).read():
    print("already patched")
else:
    edit(kern, [
        ("import inspect\n", "import inspect\nimport os\n"),
        ("from .schedule import build_schedule\n",
         "from .schedule import build_schedule\nfrom .common import now as _dbg_now\n"
         "from ..communication_ops_utils import store_i64_global_system as _dbg_st64\n"),
        ("    @traced\n    def roles(L, tid, a, epoch):\n",
         "    DBG = os.environ.get(\"AITER_MEGAMOE_TP_DBG\", \"0\") == \"1\"\n\n"
         "    @traced\n    def dbg_mark(tid, a, k):\n"
         "        if const_expr(DBG and not TN):\n"
         "            if tid == i32(0):\n"
         "                _dbg_st64(a[\"tn_res\"] + fx.Int64(i32(gpu.block_id(\"x\"))) * fx.Int64(64)"
         " + fx.Int64(8 * k), _dbg_now())\n\n"
         "    @traced\n    def roles(L, tid, a, epoch):\n"),
        ("            compute_units(L, tid, a)\n            comm_help(L, tid, a, epoch)\n",
         "            compute_units(L, tid, a)\n            dbg_mark(tid, a, 2)\n            comm_help(L, tid, a, epoch)\n"),
        ("        tid = fx.Int32(gpu.thread_id(\"x\"))\n",
         "        tid = fx.Int32(gpu.thread_id(\"x\"))\n        dbg_mark(tid, a, 0)\n"),
        ("        quant_chunks(L, tid, NTT, a, i32(0), i32(NCHA))\n        gpu.barrier()\n",
         "        quant_chunks(L, tid, NTT, a, i32(0), i32(NCHA))\n        gpu.barrier()\n        dbg_mark(tid, a, 1)\n"),
        ("        if const_expr(TN):\n            wait_vm(0)\n        gpu.barrier()\n",
         "        if const_expr(TN):\n            wait_vm(0)\n        gpu.barrier()\n        dbg_mark(tid, a, 3)\n"),
        ("        finish(tid, a, epoch)\n", "        finish(tid, a, epoch)\n        dbg_mark(tid, a, 4)\n"),
        ("flyc.kernel(name=name, known_block_size",
         "flyc.kernel(name=name + (\"_dbg\" if DBG else \"\"), known_block_size"),
    ])
    edit(eng, [
        ("            *(ptr(t) for t in (tail or (None, None, None))),\n",
         "            *(ptr(t) for t in (tail or (getattr(self, \"dbg_buf\", None), None, None))),\n"),
    ])
    print("patched", kern, eng)

#!/usr/bin/env python3
"""E0 -- H1 mechanism micro-repro (no model, one GPU).

H1: sglang pins a large HiCache host pool with cudaHostRegister (hipHostRegister on
ROCm) in CHUNKS (sglang/srt/mem_cache/pool_host/common.py `_cuda_host_register`,
limit SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB, default 256), but the JIT HiCache
kernels translate only the tensor BASE (runtime.cuh `get_device_accessible_ptr` ->
hipHostGetDevicePointer(base)) and then index linearly. If separate registrations
get non-contiguous device aliases, dev(base)+off for `off` past chunk 0 is either
unmapped (GPU memory access fault) or aliases some other mapping (silent wrong data).

Parent mode (default) spawns one child per case so a GPU fault cannot kill the run:
  chunked  (limit=--chunk-gib, granularity=--gran-bytes): T0, T1@chunk1, T2@chunk1, T2@chunk2
  single   (limit >= size; one registration)            : T0, T2@chunk1, T2@chunk2
Touch kinds (each a tiny triton kernel that reads --nread bytes and checksums them):
  T0  dev(base)+off inside chunk 0            -> must pass
  T1  hipHostGetDevicePointer(base+off)       -> per-chunk translation, must pass
  T2  dev(base)+off inside chunk k (k>=1)     -> what the JIT kernels do; H1 predicts fault/mismatch
Every child first prints the address probe: for each chunk, dev(base)+off vs dev(base+off).

Output: per-case JSON lines prefixed with `RESULT ` and a final `VERDICT ...` line.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

GIB = 1024**3


def _log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- child
def _hip():
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (  # as engram.py
        find_loaded_library,
    )

    path = find_loaded_library("libamdhip64") or "libamdhip64.so"
    lib = ctypes.CDLL(path)
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    lib.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    return lib, path


def _dev_ptr(lib, host_ptr):
    out = ctypes.c_void_p()
    rc = lib.hipHostGetDevicePointer(ctypes.byref(out), ctypes.c_void_p(host_ptr), 0)
    return rc, (out.value or 0)


def _pattern_byte(off):
    return (off * 131 + 7) & 0xFF


def child(args):
    # The env var is read at call time (EnvInt.get -> os.getenv), but set it before
    # the import anyway so either reading style takes the override.
    os.environ["SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB"] = str(args.chunk_gib)

    import torch
    import triton
    import triton.language as tl

    from sglang.srt.mem_cache.pool_host import common as phc
    from sglang.srt.mem_cache.storage.mmap.mmap_allocator import alloc_mmap

    @triton.jit
    def _read_sum(addr, n, out_ptr, BLOCK: tl.constexpr):
        p = addr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        offs = tl.arange(0, BLOCK)
        x = tl.load(p + offs, mask=offs < n, other=0).to(tl.int64)
        tl.store(out_ptr, tl.sum(x, axis=0))

    torch.cuda.set_device(0)
    size = int(args.size_gib * GIB)
    gran = args.gran_bytes
    size -= size % gran  # whole granules, like a real page-first pool

    t0 = time.time()
    buf = alloc_mmap((size,), torch.uint8)  # same allocator HostTensorAllocator uses
    t_alloc = time.time() - t0
    t0 = time.time()
    # the exact routine sglang calls (alloc_with_host_register -> _cuda_host_register)
    phc._cuda_host_register(buf, gran if not args.legacy_single else None)
    t_reg = time.time() - t0
    ranges = getattr(buf, phc._CUDA_HOST_REGISTERED_RANGES_ATTR)
    base = buf.data_ptr()
    _log(
        f"[child] case={args.case} size={size} ({size / GIB:.2f} GiB) chunk_gib={args.chunk_gib} "
        f"gran={gran} nreg={len(ranges)} alloc={t_alloc:.1f}s register={t_reg:.1f}s"
    )

    lib, hip_path = _hip()
    rc, dbase = _dev_ptr(lib, base)
    _log(f"[child] libamdhip64={hip_path} host_base={base:#x} dev_base={dbase:#x} rc={rc}")
    if rc != 0 or not dbase:
        raise SystemExit(f"hipHostGetDevicePointer(base) failed rc={rc}")

    # optional second opinion: the AOT op make_kernel_ptr_table uses
    aot = None
    try:
        from sgl_kernel.kvcacheio import get_device_accessible_ptr as aot  # noqa
    except Exception as e:  # pragma: no cover
        _log(f"[child] sgl_kernel.kvcacheio.get_device_accessible_ptr unavailable: {e!r}")

    probe = []
    for k, (rptr, rsize) in enumerate(ranges):
        for tag, off in (("start", rptr - base), ("mid", rptr - base + rsize // 2),
                         ("last", rptr - base + rsize - args.nread)):
            rc_k, d = _dev_ptr(lib, base + off)
            row = {  # noqa
                "chunk": k, "where": tag, "off": off,
                "reg_host": rptr, "reg_size": rsize,
                "dev_base_plus_off": dbase + off, "dev_of_host_off": d, "rc": rc_k,
                "contiguous": (rc_k == 0 and d == dbase + off),
                "delta": (d - (dbase + off)) if rc_k == 0 else None,
            }
            if aot is not None:
                try:
                    row["aot_view_ptr"] = int(aot(buf[off:off + 1], 0))
                except Exception as e:
                    row["aot_view_ptr"] = repr(e)
            probe.append(row)
    for r in probe:
        _log(
            "[probe] chunk={chunk} {where:5s} off={off:#x} dev(base)+off={dev_base_plus_off:#x} "
            "dev(base+off)={dev_of_host_off:#x} rc={rc} contiguous={contiguous} delta={delta}".format(**r)
        )

    res = {"case": args.case, "touch": args.touch, "touch_chunk": args.touch_chunk,
           "nreg": len(ranges), "size": size, "dev_base": dbase, "host_base": base,
           "same_va": dbase == base,
           "probe_noncontig": [r["chunk"] for r in probe if not r["contiguous"]]}

    if args.offs:
        # explicit offset list (E2): GiB floats, "end", or "bnd[+-]BYTES" where bnd is the
        # first registration boundary. Touched in the given order via dev(base)+off (T2
        # arithmetic) and, for comparison, the per-address translation is printed.
        bnd = ranges[0][1]
        res["boundary"] = bnd
        res["touches"] = []
        n = args.nread
        out = torch.zeros(1, dtype=torch.int64, device="cuda")
        for tok in args.offs.split(","):
            tok = tok.strip()
            if tok == "end":
                off = size - n
            elif tok.startswith("bnd"):
                off = bnd + int(tok[3:] or 0)
            else:
                off = int(float(tok) * GIB)
            off = max(0, min(off, size - n))
            pat = torch.tensor([_pattern_byte(off + i) for i in range(n)], dtype=torch.uint8)
            buf[off:off + n].copy_(pat)
            expect = int(pat.to(torch.int64).sum())
            rc_k, d = _dev_ptr(lib, base + off)
            addr = dbase + off
            _log(f"[touch] T2 tok={tok} off={off:#x} ({off / GIB:.3f} GiB) addr={addr:#x} "
                 f"dev(base+off)={d:#x} contiguous={d == addr} -> launching")
            t0 = time.time()
            _read_sum[(1,)](addr, n, out, BLOCK=triton.next_power_of_2(n))
            torch.cuda.synchronize()
            got = int(out.item())
            st = "ok" if got == expect else "MISMATCH"
            _log(f"[touch] T2 tok={tok} expect={expect} got={got} -> {st} ({time.time() - t0:.2f}s)")
            res["touches"].append(dict(tok=tok, off=off, contiguous=(d == addr), status=st))
        res["touch_status"] = "ok" if all(t["status"] == "ok" for t in res["touches"]) else "MISMATCH"
    elif args.touch != "none":
        k = 0 if args.touch == "T0" else args.touch_chunk
        # Touch offset = middle of chunk k of the REFERENCE chunked layout, so the
        # single-registration control reads exactly the same bytes as the chunked case.
        ref = (args.ref_chunk_gib * GIB // gran) * gran
        off = k * ref + ref // 2
        if off + args.nread > size:
            res["touch_status"] = f"skip(off {off:#x} beyond size)"
        else:
            n = args.nread
            # host-side pattern at the target bytes, then check the GPU sees the same bytes
            host_view = buf[off:off + n]
            pat = torch.tensor([_pattern_byte(off + i) for i in range(n)], dtype=torch.uint8)
            host_view.copy_(pat)
            expect = int(pat.to(torch.int64).sum())
            if args.touch == "T1":
                rc_k, addr = _dev_ptr(lib, base + off)
                assert rc_k == 0, rc_k
            else:
                addr = dbase + off
            out = torch.zeros(1, dtype=torch.int64, device="cuda")
            _log(f"[touch] {args.touch} chunk={k} off={off:#x} addr={addr:#x} n={n} -> launching")
            _read_sum[(1,)](addr, n, out, BLOCK=triton.next_power_of_2(n))
            torch.cuda.synchronize()
            got = int(out.item())
            res.update(touch_addr=addr, touch_off=off, expect=expect, got=got,
                       touch_status="ok" if got == expect else "MISMATCH")
            _log(f"[touch] {args.touch} chunk={k} expect={expect} got={got} -> {res['touch_status']}")
    _log("RESULT " + json.dumps(res))
    phc._cuda_host_unregister(buf)


# ---------------------------------------------------------------------------- parent
def parent(args):
    big_chunk = int(args.size_gib) + 8  # limit >= size -> exactly one registration
    cases = [
        ("chunked", args.chunk_gib, "T0", 0),
        ("chunked", args.chunk_gib, "T1", 1),
        ("chunked", args.chunk_gib, "T2", 1),
        ("chunked", args.chunk_gib, "T2", 2),
        ("single", big_chunk, "T0", 0),
        ("single", big_chunk, "T2", 1),
        ("single", big_chunk, "T2", 2),
    ]
    os.makedirs(args.logdir, exist_ok=True)
    results = []
    for name, chunk, touch, kc in cases:
        tag = f"{name}.{touch}.c{kc}"
        cmd = [sys.executable, os.path.abspath(__file__), "--child",
               "--case", name, "--chunk-gib", str(chunk), "--size-gib", str(args.size_gib),
               "--gran-bytes", str(args.gran_bytes), "--nread", str(args.nread),
               "--touch", touch if args.touch else "none", "--touch-chunk", str(kc),
               "--ref-chunk-gib", str(args.chunk_gib)]
        if name == "single" and args.single_legacy:
            cmd.append("--legacy-single")
        logf = os.path.join(args.logdir, f"e0.{tag}.yihou.log")
        _log(f"=== {tag}: {' '.join(cmd)}  (log {logf})")
        t0 = time.time()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", timeout=args.timeout)
            rc, out = p.returncode, p.stdout
        except subprocess.TimeoutExpired as e:
            rc, out = "TIMEOUT", (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        with open(logf, "w") as f:
            f.write(out)
        res = None
        for line in out.splitlines():
            if line.startswith("[probe]") or line.startswith("[touch]") or line.startswith("[child]"):
                _log("   " + line)
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
        fault = "Memory access fault" in out
        if res is None:
            res = {"case": name, "touch": touch, "touch_chunk": kc}
        res["tag"], res["rc"], res["gpu_fault"], res["secs"] = tag, rc, fault, round(time.time() - t0, 1)
        if fault:
            res["touch_status"] = "GPU_FAULT"
            for line in out.splitlines():
                if "Memory access fault" in line:
                    res["fault_line"] = line.strip()
                    break
        elif rc != 0 and "touch_status" not in res:
            res["touch_status"] = f"CHILD_FAIL(rc={rc})"
        _log(f"   -> rc={rc} status={res.get('touch_status')} fault={fault} "
             f"noncontig_chunks={res.get('probe_noncontig')} same_va={res.get('same_va')}")
        _log("SUMMARY " + json.dumps(res))
        results.append(res)

    def st(tag):
        for r in results:
            if r["tag"] == tag:
                return r.get("touch_status")
        return None

    chunked_noncontig = any(r.get("probe_noncontig") for r in results if r["case"] == "chunked")
    single_noncontig = any(r.get("probe_noncontig") for r in results if r["case"] == "single")
    nreg = max((r.get("nreg") or 0) for r in results if r["case"] == "chunked")
    t2_bad = [t for t in ("chunked.T2.c1", "chunked.T2.c2") if st(t) not in ("ok", None) and not str(st(t)).startswith("skip")]
    ctrl_ok = st("single.T2.c1") == "ok" and st("single.T0.c0") == "ok"
    base_ok = st("chunked.T0.c0") == "ok" and st("chunked.T1.c1") == "ok"

    if nreg < 2:
        v = f"INCONCLUSIVE: chunked case produced nreg={nreg} (<2); raise --size-gib or lower --chunk-gib"
    elif not args.touch:
        v = ("H1-PROBE: non-contiguous device aliases across chunks" if chunked_noncontig and not single_noncontig
             else "H1-PROBE: aliases contiguous across chunks (mechanism NOT present at this size)" if not chunked_noncontig
             else "INCONCLUSIVE: single registration is also non-contiguous by probe")
    elif chunked_noncontig and t2_bad and ctrl_ok and base_ok:
        v = f"H1 MECHANISM CONFIRMED: chunked dev(base)+off fails {t2_bad} ({[st(t) for t in t2_bad]}); single-registration control and per-chunk translation pass"
    elif not chunked_noncontig and not t2_bad and ctrl_ok and base_ok:
        v = "H1 MECHANISM REFUTED at this size/chunk: chunked aliases contiguous and dev(base)+off reads correct bytes in every chunk"
    else:
        v = (f"INCONCLUSIVE: chunked_noncontig={chunked_noncontig} single_noncontig={single_noncontig} "
             f"t2_bad={t2_bad} ctrl_ok={ctrl_ok} base_ok={base_ok}; see SUMMARY lines")
    _log("VERDICT " + v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--case", default="chunked")
    ap.add_argument("--size-gib", type=float, default=24.0)
    ap.add_argument("--chunk-gib", type=int, default=8, help="SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB (int GiB, >=1)")
    # Illustrative page-first granule (page_size 64 x 78 layers x 576 B). Any granule that
    # does not divide 1 GiB mimics production, where chunk = floor(limit/gran)*gran.
    ap.add_argument("--gran-bytes", type=int, default=64 * 78 * 576)
    ap.add_argument("--nread", type=int, default=256)
    ap.add_argument("--touch", default="none", help="child: none|T0|T1|T2 ; parent: any non-'none' enables touches")
    ap.add_argument("--touch-chunk", type=int, default=1)
    ap.add_argument("--offs", default="", help="child: explicit touch list, e.g. 0,200,300,end,bnd-4096,bnd+4096")
    ap.add_argument("--ref-chunk-gib", type=int, default=8, help="chunk layout that defines touch offsets")
    ap.add_argument("--legacy-single", action="store_true", help="child: registration_granularity=None (legacy single call)")
    ap.add_argument("--single-legacy", action="store_true", help="parent: use the legacy path for the single-registration control")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--logdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out.e0.yihou"))
    args = ap.parse_args()
    if args.child:
        child(args)
    else:
        args.touch = args.touch != "none"
        parent(args)


if __name__ == "__main__":
    main()

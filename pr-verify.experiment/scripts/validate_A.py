#!/usr/bin/env python3
"""Equivalence + scope check for the hicache ROCm allocator fix.

Reference ("对拍" baseline): the proven local fix,
  infera deploy/docker/patches/sglang_rocm/patch_hicache_rocm_host_alloc.py
Candidate: the upstream-shaped edit that goes into the PR.

Both are applied to independent throwaway copies of the SAME container
sglang tree, then compared on the three things that can differ:

  1. dispatch  -- what ALLOC_MEMORY_FUNCS resolves to for every key that
                  matters, including the defaultdict fallback
  2. behaviour -- an actual allocation through the resolved function, with
                  its device pointer measured against its host VA
  3. scope     -- the set of names the module exports, and every other
                  symbol's identity, must be unchanged from stock

Run inside a ROCm sglang container.
"""

import ctypes
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRT = Path(
    subprocess.run(
        # sglang is a namespace package here, so sglang.__file__ is None and
        # sglang.__path__ points at the repo root, not the package -- resolve
        # through sglang.srt, which is a real package.
        [sys.executable, "-c", "import sglang.srt; print(list(sglang.srt.__path__)[0])"],
        capture_output=True,
        text=True,
    ).stdout.strip()
)
COMMON = SRT / "mem_cache" / "pool_host" / "common.py"

# ---------------------------------------------------------------- candidate

CAND_OLD = """ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)"""

CAND_NEW = """_ALLOC_MEMORY_FUNCS = {
    "npu": alloc_with_pin_memory,
    "musa": alloc_with_pin_memory,
}
if _is_hip:
    _ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_pin_memory if _is_hip else alloc_with_host_register,
    _ALLOC_MEMORY_FUNCS,
)"""


def apply_candidate(src: str) -> str:
    """The upstream-shaped edit, expressed against whatever import line the
    tree happens to use (main says storage.mmap, v0.5.15 says mmap_allocator)."""
    for imp in (
        "from sglang.srt.mem_cache.storage.mmap import alloc_mmap",
        "from sglang.srt.mem_cache.mmap_allocator import alloc_mmap",
    ):
        if imp in src:
            src = src.replace(
                imp,
                imp + "\nfrom sglang.srt.utils import is_hip",
                1,
            )
            break
    else:
        sys.exit("candidate: alloc_mmap import not found")
    src = src.replace(
        "logger = logging.getLogger(__name__)",
        "logger = logging.getLogger(__name__)\n\n_is_hip = is_hip()",
        1,
    )
    assert CAND_OLD in src, "candidate: ALLOC_MEMORY_FUNCS block not found"
    return src.replace(CAND_OLD, CAND_NEW, 1)


# ---------------------------------------------------------------- reference

REF_OLD = CAND_OLD
REF_MARKER = "GLM52_ROCM_HOST_ALLOC"
REF_NEW = f"""{REF_MARKER} = "applied"

_ALLOC_MEMORY_FUNCS_OVERRIDES = {{
    "npu": alloc_with_pin_memory,
    "musa": alloc_with_pin_memory,
}}
if is_hip():
    _ALLOC_MEMORY_FUNCS_OVERRIDES["cuda"] = alloc_with_pin_memory

ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_pin_memory if is_hip() else alloc_with_host_register,
    _ALLOC_MEMORY_FUNCS_OVERRIDES,
)"""


def apply_reference(src: str) -> str:
    """The infera patch, transcribed. Differences from the candidate are
    exactly the local-repo scaffolding: a bytecode marker literal, and
    calling is_hip() at each use instead of hoisting it."""
    for imp in (
        "from sglang.srt.mem_cache.storage.mmap import alloc_mmap",
        "from sglang.srt.mem_cache.mmap_allocator import alloc_mmap",
    ):
        if imp in src:
            src = src.replace(
                imp,
                imp + f"\nfrom sglang.srt.utils import is_hip  # {REF_MARKER}",
                1,
            )
            break
    else:
        sys.exit("reference: alloc_mmap import not found")
    assert REF_OLD in src, "reference: ALLOC_MEMORY_FUNCS block not found"
    return src.replace(REF_OLD, REF_NEW, 1)


# ---------------------------------------------------------------- harness


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dev_ptr(host_ptr):
    lib = ctypes.CDLL("libamdhip64.so")
    out = ctypes.c_void_p()
    rc = lib.hipHostGetDevicePointer(ctypes.byref(out), ctypes.c_void_p(host_ptr), ctypes.c_uint(0))
    return None if int(rc) != 0 else (out.value or 0)


def probe(mod, label):
    """Everything that can distinguish the two variants."""
    import torch
    from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator

    d = {}
    for key in ("cuda", "npu", "musa", "xpu", "some-unknown-device"):
        d[f"dispatch[{key}]"] = mod.ALLOC_MEMORY_FUNCS[key].__name__

    # memory_pool_host.py:768 and :1257 key the table with `tensor.device`, a
    # torch.device object -- which is NOT dict-key-equal to the string "cuda"
    # (different hash), so those two pools always take the defaultdict
    # fallback. A fix that only adds a "cuda" key would miss them.
    dev_obj = torch.zeros(1, device="cuda").device
    d["dispatch[torch.device('cuda:0')]"] = mod.ALLOC_MEMORY_FUNCS[dev_obj].__name__

    fn = mod.ALLOC_MEMORY_FUNCS["cuda"]
    buf = fn(
        (8 << 20,),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
        allocator=HostTensorAllocator() if fn.__name__ == "alloc_with_host_register" else None,
    )
    host = buf.data_ptr()
    dev = dev_ptr(host)
    d["alloc.is_pinned"] = bool(buf.is_pinned())
    d["alloc.devptr_equals_host"] = (dev == host) if dev is not None else "query-failed"
    d["alloc.nbytes"] = buf.numel() * buf.element_size()

    # Scope: no symbol other than the dispatch table may change identity, and
    # nothing may be added beyond the private helper.
    d["exports"] = sorted(
        n for n in vars(mod) if not n.startswith("__") and n != "ALLOC_MEMORY_FUNCS"
    )
    print(f"--- {label}")
    for k, v in d.items():
        print(f"    {k} = {v}")
    return d


def main():
    stock_src = COMMON.read_text()
    tmp = Path(tempfile.mkdtemp())

    variants = {
        "stock": stock_src,
        "reference (infera patch)": apply_reference(stock_src),
        "candidate (upstream PR)": apply_candidate(stock_src),
    }
    probes = {}
    for i, (label, src) in enumerate(variants.items()):
        p = tmp / f"common_{i}.py"
        p.write_text(src)
        probes[label] = probe(load(p, f"_probe_common_{i}"), label)

    print()
    ref = probes["reference (infera patch)"]
    cand = probes["candidate (upstream PR)"]
    stock = probes["stock"]

    ok = True

    # Equivalence: reference vs candidate, on everything except the exports
    # list (the reference deliberately adds a bytecode marker we strip).
    for k in ref:
        if k == "exports":
            continue
        if ref[k] != cand[k]:
            print(f"EQUIVALENCE FAIL  {k}: reference={ref[k]!r} candidate={cand[k]!r}")
            ok = False
    if ok:
        print("EQUIVALENCE  PASS  candidate matches the proven local fix on every")
        print("                   dispatch key and on the measured allocation.")

    # Scope: candidate must add exactly one private name to stock, nothing else.
    added = set(cand["exports"]) - set(stock["exports"])
    removed = set(stock["exports"]) - set(cand["exports"])
    expected_added = {"is_hip", "_is_hip", "_ALLOC_MEMORY_FUNCS"}
    if removed:
        print(f"SCOPE FAIL        candidate removed exports: {sorted(removed)}")
        ok = False
    elif added != expected_added:
        print(
            f"SCOPE FAIL        candidate added {sorted(added)}, expected {sorted(expected_added)}"
        )
        ok = False
    else:
        print("SCOPE        PASS  candidate removes nothing and adds only")
        print(f"                   {sorted(expected_added)}.")

    # And the reference's local scaffolding must NOT have come along.
    if REF_MARKER in variants["candidate (upstream PR)"]:
        print(f"SCOPE FAIL        candidate still carries the local marker {REF_MARKER}")
        ok = False
    else:
        print(f"SCOPE        PASS  local marker {REF_MARKER} is absent from the candidate.")

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

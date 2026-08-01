#!/usr/bin/env python3
"""Standalone micro-repro of the DSA-indexer hicache write-back kernel.

WHY THIS EXISTS: one e2e boot of the real deployment costs ~8 min of model load
plus ~2 min of needle traffic. The faulting code -- transfer_kv_all_layer_mla on
the layer_first/kernel path -- needs none of that. It needs a device indexer
buffer, a pinned host buffer, and two pointer tables. So we reproduce it here in
~30 s and sweep several hypotheses in a single GPU session.

It rebuilds EXACTLY what DSAIndexerPoolHost + DSATokenToKVPool construct:

  device:  index_k_with_scale_buffer = [ zeros((dev_pages, page_size*(128+4)), uint8) ] * layer_num
  host:    layer_first -> zeros((layer_num, indexer_page_num, page_stride), uint8)
           allocated by alloc_mmap + cudaHostRegister  (NOT torch pin_memory)

Then it calls the same kernel the engine calls: transfer_kv_all_layer_mla.

The kernel dereferences the HOST pointers from inside a GPU kernel, so the host
buffer must be *mapped* into the device address space, not merely page-locked.
That is what the --host-alloc variants probe.

Exit 0 = no fault. A GPU memory access fault aborts the process (that IS the signal).
"""
import argparse
import ctypes
import math
import mmap
import sys

import torch

PAGE_SIZE = 64
LAYER_NUM = 78
INDEX_HEAD_DIM = 128
QUANT_BLOCK = 128
SIZE_PER_TOKEN = INDEX_HEAD_DIM + INDEX_HEAD_DIM // QUANT_BLOCK * 4  # 132
DTYPE = torch.uint8
PAGE_STRIDE = SIZE_PER_TOKEN * PAGE_SIZE * 1  # 8448

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]

# hipHostRegister flags (hip_runtime_api.h)
HOST_REGISTER_DEFAULT = 0x0
HOST_REGISTER_PORTABLE = 0x1
HOST_REGISTER_MAPPED = 0x2


def alloc_mmap_host(dims, shared=True, populate=True, dtype=DTYPE):
    """Mirror sglang's alloc_mmap: MAP_SHARED|MAP_ANONYMOUS|MAP_POPULATE."""
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()
    alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE
    flags = mmap.MAP_ANONYMOUS | (mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE)
    if populate:
        flags |= getattr(mmap, "MAP_POPULATE", 0x08000)
    ptr = _libc.mmap(None, alloc_bytes, mmap.PROT_READ | mmap.PROT_WRITE,
                     flags, -1, 0)
    if ptr is None or ptr == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap failed")
    array = (ctypes.c_byte * alloc_bytes).from_address(ptr)
    return torch.frombuffer(array, dtype=dtype, count=math.prod(dims)).reshape(dims)


def host_register(buf, flags):
    cudart = torch.cuda.cudart()
    n = buf.numel() * buf.element_size()
    rc = cudart.cudaHostRegister(buf.data_ptr(), n, flags)
    if int(rc) != 0:
        raise RuntimeError(f"cudaHostRegister rc={int(rc)} flags={flags:#x} bytes={n}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-tokens", type=int, default=3380992,
                    help="device pool tokens (spur faulting run: 3380992)")
    ap.add_argument("--host-tokens", type=int, default=712256,
                    help="host pool tokens (spur faulting run: 712256)")
    ap.add_argument("--layers", type=int, default=LAYER_NUM)
    ap.add_argument("--npages", type=int, default=64)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--case", default="head",
                    choices=["head", "tail", "gap", "sweep"])
    ap.add_argument("--host-alloc", default="mmap_default",
                    choices=["mmap_default", "mmap_mapped", "mmap_portable_mapped",
                             "pin_memory", "mmap_private", "mmap_nopopulate"],
                    help="how the host buffer is allocated/registered")
    ap.add_argument("--no-dev-alloc", action="store_true",
                    help="skip the big device buffer (isolates host mapping)")
    args = ap.parse_args()

    dev = "cuda"
    torch.cuda.init()

    dev_pages = (args.dev_tokens + PAGE_SIZE + 1) // PAGE_SIZE
    anchor_page_num = args.host_tokens // PAGE_SIZE + 1
    anchor_size = anchor_page_num * PAGE_SIZE
    indexer_page_num = (anchor_size + PAGE_SIZE + 1) // PAGE_SIZE

    host_bytes = indexer_page_num * PAGE_STRIDE * args.layers
    print(f"case={args.case} host_alloc={args.host_alloc} layers={args.layers}")
    print(f"device: tokens={args.dev_tokens} pages={dev_pages}")
    print(f"host:   anchor_page_num={anchor_page_num} indexer_page_num={indexer_page_num} "
          f"({host_bytes/1e9:.2f} GB)")

    free, total = torch.cuda.mem_get_info()
    print(f"device free={free/1e9:.1f}GB / {total/1e9:.1f}GB", flush=True)

    if args.no_dev_alloc:
        # One tiny device buffer per layer; still a valid src for the kernel.
        dev_pages = max(dev_pages, args.npages + 1)
        dev_bufs = [torch.zeros((args.npages + 1, PAGE_STRIDE), dtype=DTYPE, device=dev)
                    for _ in range(args.layers)]
        dev_pages = args.npages + 1
    else:
        need = dev_pages * PAGE_STRIDE * args.layers
        if need > free * 0.5:
            args.layers = max(1, int(free * 0.35 / (dev_pages * PAGE_STRIDE)))
            print(f"!! scaling layers to {args.layers} to fit device memory")
            host_bytes = indexer_page_num * PAGE_STRIDE * args.layers
        print(f"allocating {args.layers} device buffers "
              f"({dev_pages*PAGE_STRIDE*args.layers/1e9:.2f} GB) ...", flush=True)
        dev_bufs = [torch.zeros((dev_pages, PAGE_STRIDE), dtype=DTYPE, device=dev)
                    for _ in range(args.layers)]
    dev_ptrs = torch.tensor([b.data_ptr() for b in dev_bufs],
                            dtype=torch.uint64, device=dev)

    print(f"allocating host buffer {host_bytes/1e9:.2f} GB via {args.host_alloc} ...",
          flush=True)
    dims = (args.layers, indexer_page_num, PAGE_STRIDE)
    if args.host_alloc == "pin_memory":
        host_buf = torch.zeros(dims, dtype=DTYPE, device="cpu", pin_memory=True)
    elif args.host_alloc == "mmap_private":
        host_buf = alloc_mmap_host(dims, shared=False)
        host_register(host_buf, HOST_REGISTER_DEFAULT)
    elif args.host_alloc == "mmap_nopopulate":
        host_buf = alloc_mmap_host(dims, populate=False)
        host_register(host_buf, HOST_REGISTER_DEFAULT)
    elif args.host_alloc == "mmap_mapped":
        host_buf = alloc_mmap_host(dims)
        host_register(host_buf, HOST_REGISTER_MAPPED)
    elif args.host_alloc == "mmap_portable_mapped":
        host_buf = alloc_mmap_host(dims)
        host_register(host_buf, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED)
    else:  # mmap_default -- exactly what sglang does today
        host_buf = alloc_mmap_host(dims)
        host_register(host_buf, HOST_REGISTER_DEFAULT)

    host_refs = [host_buf[i] for i in range(args.layers)]
    host_ptrs = torch.tensor([x.data_ptr() for x in host_refs],
                             dtype=torch.uint64, device=dev)
    print(f"host ptr[0]={host_refs[0].data_ptr():#x}", flush=True)

    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla

    def run(dst_pages, tag):
        src = torch.arange(len(dst_pages), dtype=torch.int64, device=dev) % dev_pages
        dst = torch.tensor(dst_pages, dtype=torch.int64, device=dev)
        print(f"  [{tag}] dst[0]={dst_pages[0]} dst[-1]={dst_pages[-1]} "
              f"n={len(dst_pages)}", flush=True)
        for _ in range(args.iters):
            transfer_kv_all_layer_mla(
                src_layers=dev_ptrs, dst_layers=host_ptrs,
                src_indices=src, dst_indices=dst,
                item_size=PAGE_STRIDE, num_layers=args.layers,
            )
        torch.cuda.synchronize()
        print(f"  [{tag}] OK", flush=True)

    n = args.npages
    if args.case == "head":
        run(list(range(n)), "head")
    elif args.case == "tail":
        run(list(range(anchor_page_num - n, anchor_page_num)), "tail@anchor")
    elif args.case == "gap":
        run([indexer_page_num - 1] * n, "last_indexer_page")
    else:
        run(list(range(n)), "head")
        run(list(range(anchor_page_num - n, anchor_page_num)), "tail@anchor")
        run([indexer_page_num - 1] * n, "last_indexer_page")

    print("ALL OK -- no fault")
    return 0


if __name__ == "__main__":
    sys.exit(main())

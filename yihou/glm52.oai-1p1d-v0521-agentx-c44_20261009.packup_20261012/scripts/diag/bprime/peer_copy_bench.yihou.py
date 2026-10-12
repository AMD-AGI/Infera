#!/usr/bin/env python3
"""B' microbench (one process, 2 visible GPUs: cuda:0 = src (prefill), cuda:1 = dst (decode)).

Same shape as te_conc: L layers x P pages x PAGE bytes, every page sent to a random per-layer
slot -> L*P scattered PAGE-sized copies (default 32 x 2048 x 41,984 B = 2.56 GiB, 65,536 blocks).
Arms (each timed warm, each verified byte-exact on the DST GPU):
  memcpy_loop   : hipMemcpyAsync per block on one stream (today's per-op model, single process)
  memcpy_batch  : ONE hipMemcpyBatchAsync call (CLR 7.2.0 loops ihipMemcpy internally)
  kernel_g{N}   : one triton peer-copy kernel on cuda:0 storing into cuda:1 over XGMI,
                  N programs (CU-bounded variants), 8-byte vectors
Stale-L2 scenario per arm: dst is filled with 0xAA and READ on the dst GPU (pulls lines into
its caches) right before the copy; after host sync the dst GPU verifies every byte.
Also: KV checksum (sglang adler32_strided_checksum) cost over the same pages + determinism +
single-byte corruption detection.
Prints one RESULT json line per arm and a VERDICT line.
"""
import ctypes, json, os, sys, time

import torch
import triton
import triton.language as tl

L = int(os.environ.get("PB_LAYERS", "32"))
P = int(os.environ.get("PB_PAGES", "2048"))
PAGE = int(os.environ.get("PB_PAGE", str(64 * 656)))
GRIDS = [int(x) for x in os.environ.get("PB_GRIDS", "8,32,256,1024").split(",")]
REPS = int(os.environ.get("PB_REPS", "3"))
assert PAGE % 8 == 0
NELEM = PAGE // 8


def log(*a):
    print(*a, flush=True)


hip = ctypes.CDLL("libamdhip64.so")
hip.hipGetErrorString.restype = ctypes.c_char_p


def ck(name, rc):
    if rc != 0:
        raise RuntimeError(f"{name} rc={rc} {hip.hipGetErrorString(rc).decode()}")


assert torch.cuda.device_count() >= 2, "need HIP_VISIBLE_DEVICES=<src>,<dst>"
ck("hipSetDevice0", hip.hipSetDevice(0))
can = ctypes.c_int(0)
ck("hipDeviceCanAccessPeer", hip.hipDeviceCanAccessPeer(ctypes.byref(can), 0, 1))
rc = hip.hipDeviceEnablePeerAccess(1, 0)
log(f"peer access 0->1 can={can.value} enable_rc={rc}")  # 704 = already enabled is fine

S, D = torch.device("cuda:0"), torch.device("cuda:1")
gen = torch.Generator().manual_seed(1234)
perms = [torch.randperm(P, generator=gen) for _ in range(L)]
src = [torch.empty((P, PAGE), dtype=torch.uint8, device=S) for _ in range(L)]
dst = [torch.empty((P, PAGE), dtype=torch.uint8, device=D) for _ in range(L)]


def pattern(l, dev):
    p = torch.arange(P, device=dev, dtype=torch.int64)
    return ((l * 131 + p * 7 + 1) % 251 + 1).to(torch.uint8)


for l in range(L):
    # varying bytes within a page too, so a shifted copy cannot pass
    base = pattern(l, S)[:, None].to(torch.int32)
    col = torch.arange(PAGE, device=S, dtype=torch.int32)[None, :] % 13
    src[l].copy_(((base + col) % 251 + 1).to(torch.uint8))
torch.cuda.synchronize(S)

# block lists (host + device copies)
src_addrs, dst_addrs = [], []
for l in range(L):
    sb, db = src[l].data_ptr(), dst[l].data_ptr()
    for p, q in enumerate(perms[l].tolist()):
        src_addrs.append(sb + p * PAGE)
        dst_addrs.append(db + q * PAGE)
N = len(src_addrs)
src_addr_t = torch.tensor(src_addrs, dtype=torch.int64, device=S)
dst_addr_t = torch.tensor(dst_addrs, dtype=torch.int64, device=S)
GiB = N * PAGE / 2**30
log(f"blocks={N} PAGE={PAGE} total={GiB:.2f} GiB")

# expected dst content, built on the dst GPU once
expected = []
for l in range(L):
    e = torch.empty((P, PAGE), dtype=torch.uint8, device=D)
    e[perms[l].to(D)] = src[l].to(D)
    expected.append(e)
torch.cuda.synchronize(D)


def stale_prepare():
    for t in dst:
        t.fill_(0xAA)
    s = sum(int(t[:, :64].sum().item()) for t in dst)  # read on the dst GPU -> lines cached
    torch.cuda.synchronize(D)
    return s


def verify():
    torch.cuda.synchronize(D)
    bad = sum(int((dst[l] != expected[l]).any(dim=1).sum().item()) for l in range(L))
    return bad


stream = ctypes.c_void_p()
ck("hipSetDevice0", hip.hipSetDevice(0))
ck("hipStreamCreate", hip.hipStreamCreateWithFlags(ctypes.byref(stream), 1))  # non-blocking


def arm_memcpy_loop():
    t0 = time.perf_counter()
    for s_, d_ in zip(src_addrs, dst_addrs):
        rc = hip.hipMemcpyAsync(ctypes.c_void_p(d_), ctypes.c_void_p(s_), ctypes.c_size_t(PAGE), 4, stream)
        if rc:
            ck("hipMemcpyAsync", rc)
    ck("hipStreamSynchronize", hip.hipStreamSynchronize(stream))
    return time.perf_counter() - t0


DA = (ctypes.c_void_p * N)(*dst_addrs)
SA = (ctypes.c_void_p * N)(*src_addrs)
SZ = (ctypes.c_size_t * N)(*([PAGE] * N))


def arm_memcpy_batch():
    fail = ctypes.c_size_t(0)
    t0 = time.perf_counter()
    rc = hip.hipMemcpyBatchAsync(DA, SA, SZ, ctypes.c_size_t(N), None, None, ctypes.c_size_t(0),
                                 ctypes.byref(fail), stream)
    ck(f"hipMemcpyBatchAsync(failIdx={fail.value})", rc)
    ck("hipStreamSynchronize", hip.hipStreamSynchronize(stream))
    return time.perf_counter() - t0


@triton.jit
def _peer_copy_kernel(src_addr, dst_addr, n_items, NELEM: tl.constexpr, BLOCK: tl.constexpr,
                      NPROG: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    for item in range(pid, n_items, NPROG):
        s = tl.load(src_addr + item).to(tl.pointer_type(tl.int64))
        d = tl.load(dst_addr + item).to(tl.pointer_type(tl.int64))
        for off in range(0, NELEM, BLOCK):
            m = (off + offs) < NELEM
            v = tl.load(s + off + offs, mask=m)
            tl.store(d + off + offs, v, mask=m)


def make_kernel_arm(nprog):
    def arm():
        with torch.cuda.device(S):
            t0 = time.perf_counter()
            _peer_copy_kernel[(nprog,)](src_addr_t, dst_addr_t, N, NELEM=NELEM, BLOCK=1024,
                                        NPROG=nprog, num_warps=4)
            torch.cuda.synchronize(S)
            return time.perf_counter() - t0
    return arm


arms = [("memcpy_loop", arm_memcpy_loop), ("memcpy_batch", arm_memcpy_batch)]
arms += [(f"kernel_g{g}", make_kernel_arm(g)) for g in GRIDS]
results = []
for name, fn in arms:
    stale_prepare()
    try:
        fn()  # warm (JIT / first use)
    except Exception as e:
        log(f"RESULT {json.dumps(dict(arm=name, error=repr(e)[:300]))}")
        continue
    walls, bads = [], []
    for _ in range(REPS):
        stale_prepare()
        walls.append(fn())
        bads.append(verify())
    w = min(walls)
    r = dict(arm=name, blocks=N, GiB=round(GiB, 3), wall_min=round(w, 4), walls=[round(x, 4) for x in walls],
             us_per_block=round(w / N * 1e6, 3), GiB_s=round(GiB / w, 2), bad_pages=bads,
             byte_exact=all(b == 0 for b in bads))
    results.append(r)
    log("RESULT " + json.dumps(r))

# checksum cost (sglang's built-in PD KV checksum kernel)
try:
    from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
    idx = torch.arange(P, dtype=torch.int64, device=S)
    ptrs = [t.data_ptr() for t in src]
    def ck_once():
        torch.cuda.synchronize(S)
        t0 = time.perf_counter()
        v = adler32_strided_checksum(ptrs, [PAGE] * L, [idx] * L)
        return v, time.perf_counter() - t0
    v0, _ = ck_once()
    vals, ts = zip(*[ck_once() for _ in range(REPS)])
    det = all(v == v0 for v in vals)
    src[3][17, 5] ^= 0x01
    v_bad, _ = ck_once()
    src[3][17, 5] ^= 0x01
    v_back, _ = ck_once()
    r = dict(arm="kv_checksum_adler32", GiB=round(GiB, 3), wall_min=round(min(ts), 4),
             GiB_s=round(GiB / min(ts), 2), deterministic=det, detects_1byte=v_bad != v0,
             restores=v_back == v0, value=hex(v0))
    results.append(r)
    log("RESULT " + json.dumps(r))
except Exception as e:
    log(f"RESULT {json.dumps(dict(arm='kv_checksum_adler32', error=repr(e)[:300]))}")

ok = all(r.get("byte_exact", True) for r in results if "arm" in r and r["arm"] != "kv_checksum_adler32")
log("VERDICT " + ("PASS byte-exact all copy arms" if ok else "FAIL (see RESULT lines)"))

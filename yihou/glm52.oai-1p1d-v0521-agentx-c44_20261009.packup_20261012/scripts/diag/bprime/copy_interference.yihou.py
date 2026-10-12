#!/usr/bin/env python3
"""H-X1 microbench: does a busy src GPU slow the prefill->decode peer copy?

One process, 2 visible GPUs (cuda:0 = src/prefill, cuda:1 = dst/decode), peer access enabled.
Copy = exactly what mooncake HipTransport does per call in e2e: NB blocks of BLK bytes
(default 75 x 9 MiB ~= 0.66 GiB), one hipMemcpyAsync each on NON-BLOCKING streams round-robin
(mooncake StreamPool, hipStreamNonBlocking), then hipEventSynchronize per block.
Background load on the SRC GPU (a second thread), variants:
  idle          : nothing
  gemm_null     : bf16 GEMM loop on the legacy default (NULL) stream
  gemm_stream   : bf16 GEMM loop on a torch-created stream (sglang forward_stream style)
  spin_stream   : long full-GPU spin kernels (SPIN_MS each) on a torch stream (long attention-like)
  hostcopy      : concurrent pinned-host D2H + H2D (CI_HOST_MB chunks) on 2 torch streams (HiCache-like)
Run the whole thing twice from the wrapper: HSA_ENABLE_SDMA unset (default) and =0, so the
copy engine (SDMA vs blit kernel) is visible from the numbers.
Prints RESULT json per variant (copy wall, GiB/s, bytes verified) + the env.
"""
import ctypes, json, os, threading, time

import torch
import triton
import triton.language as tl

NB = int(os.environ.get("CI_BLOCKS", "75"))
BLK = int(os.environ.get("CI_BLOCK_BYTES", str(9 << 20)))
NSTREAMS = int(os.environ.get("CI_STREAMS", "64"))
REPS = int(os.environ.get("CI_REPS", "3"))
SPIN_MS = float(os.environ.get("CI_SPIN_MS", "200"))
GEMM_N = int(os.environ.get("CI_GEMM_N", "8192"))


def log(*a):
    print(*a, flush=True)


hip = ctypes.CDLL("libamdhip64.so")
hip.hipGetErrorString.restype = ctypes.c_char_p


def ck(n, rc):
    if rc:
        raise RuntimeError(f"{n} rc={rc} {hip.hipGetErrorString(rc).decode()}")


S, D = torch.device("cuda:0"), torch.device("cuda:1")
ck("setdev", hip.hipSetDevice(0))
rc = hip.hipDeviceEnablePeerAccess(1, 0)
log(f"env HSA_ENABLE_SDMA={os.environ.get('HSA_ENABLE_SDMA', '<unset>')} peer_enable_rc={rc} NB={NB} BLK={BLK}")

src = torch.randint(0, 255, (NB, BLK), dtype=torch.uint8, device=S)
dst = torch.zeros((NB, BLK), dtype=torch.uint8, device=D)
torch.cuda.synchronize(S); torch.cuda.synchronize(D)
# CI_STREAM_MODE: normal = mooncake (hipStreamCreateWithFlags NonBlocking, normal priority ->
# shares CLR's GPU_MAX_HW_QUEUES normal-priority HW queues with torch streams);
# high = hipStreamCreateWithPriority(NonBlocking, greatest) -> CLR High queue pool;
# cumask = hipExtStreamCreateWithCUMask(all CUs) -> dedicated HW queue (bypasses the pool).
MODE = os.environ.get("CI_STREAM_MODE", "normal")
lo, hi = ctypes.c_int(), ctypes.c_int()
ck("prange", hip.hipDeviceGetStreamPriorityRange(ctypes.byref(lo), ctypes.byref(hi)))
ncu = torch.cuda.get_device_properties(S).multi_processor_count
log(f"stream_mode={MODE} nstreams={NSTREAMS} prio_least={lo.value} prio_greatest={hi.value} ncu={ncu} "
    f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '<unset>')} "
    f"HSA_ENABLE_PEER_SDMA={os.environ.get('HSA_ENABLE_PEER_SDMA', '<unset>')}")
# Production order: torch (sglang) streams exist before mooncake lazily creates its pool.
# The first torch.cuda.Stream() initialises torch's per-device pools (32 normal + 32 high
# priority streams), which take CLR HW queues first. CI_TORCH_FIRST=0 restores the old order.
TORCH_FIRST = os.environ.get("CI_TORCH_FIRST", "1") == "1"
torch_bg_stream = torch.cuda.Stream(device=S) if TORCH_FIRST else None
log(f"torch_first={TORCH_FIRST}")
streams = []
for _ in range(NSTREAMS):
    s = ctypes.c_void_p()
    if MODE == "high":
        ck("streamcreateprio", hip.hipStreamCreateWithPriority(ctypes.byref(s), 1, hi))
    elif MODE == "cumask":
        words = (ncu + 31) // 32
        mask = (ctypes.c_uint32 * words)(*([0xFFFFFFFF] * words))
        if ncu % 32:
            mask[words - 1] = (1 << (ncu % 32)) - 1
        ck("streamcreatecumask", hip.hipExtStreamCreateWithCUMask(ctypes.byref(s), ctypes.c_uint32(words), mask))
    else:
        ck("streamcreate", hip.hipStreamCreateWithFlags(ctypes.byref(s), 1))
    streams.append(s)
events = []
for _ in range(NB):
    e = ctypes.c_void_p(); ck("eventcreate", hip.hipEventCreateWithFlags(ctypes.byref(e), 2)); events.append(e)


def do_copy():
    ck("setdev", hip.hipSetDevice(0))
    t0 = time.perf_counter()
    for i in range(NB):
        st = streams[i % NSTREAMS]
        ck("memcpy", hip.hipMemcpyAsync(ctypes.c_void_p(dst[i].data_ptr()), ctypes.c_void_p(src[i].data_ptr()),
                                        ctypes.c_size_t(BLK), 4, st))
        ck("record", hip.hipEventRecord(events[i], st))
    for i in range(NB):
        ck("evsync", hip.hipEventSynchronize(events[i]))
    return time.perf_counter() - t0


@triton.jit
def _spin(out, iters):
    acc = tl.program_id(0).to(tl.float32)
    for i in range(iters):
        acc = acc * 1.0000001 + 1.0
    tl.store(out + tl.program_id(0), acc)


stop = threading.Event()
hostcopy_bytes = [0]


def bg(kind):
    torch.cuda.set_device(S)
    a = torch.randn(GEMM_N, GEMM_N, device=S, dtype=torch.bfloat16)
    b = torch.randn(GEMM_N, GEMM_N, device=S, dtype=torch.bfloat16)
    out = torch.zeros(1 << 16, device=S)
    if kind == "gemm_null":
        st = torch.cuda.default_stream(S)
    else:
        st = torch_bg_stream if torch_bg_stream is not None else torch.cuda.Stream(device=S)
    if kind == "hostcopy":
        # HiCache write-through / load-back stand-in: pinned host <-> src GPU, D2H and H2D on
        # two torch streams at once, HOST_MB chunks
        n = int(os.environ.get("CI_HOST_MB", "512")) << 20
        h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        g1 = torch.empty(n, dtype=torch.uint8, device=S); g2 = torch.empty(n, dtype=torch.uint8, device=S)
        st2 = torch.cuda.Stream(device=S)
        ready.set()
        while not stop.is_set():
            with torch.cuda.stream(st):
                h.copy_(g1, non_blocking=True)
            with torch.cuda.stream(st2):
                g2.copy_(h, non_blocking=True)
            st.synchronize(); st2.synchronize()
            hostcopy_bytes[0] += 2 * n
        return
    # calibrate spin iterations to ~SPIN_MS on an idle GPU
    iters = 1
    if kind == "spin_stream":
        iters = 1 << 16
        with torch.cuda.stream(st):
            for _ in range(8):
                t0 = time.perf_counter(); _spin[(1 << 16,)](out, iters); torch.cuda.synchronize(S)
                dt = (time.perf_counter() - t0) * 1e3
                iters = max(1, int(iters * SPIN_MS / max(dt, 0.01)))
    ready.set()
    with torch.cuda.stream(st):
        while not stop.is_set():
            if kind.startswith("gemm"):
                for _ in range(8):
                    torch.matmul(a, b)
            else:
                _spin[(1 << 16,)](out, iters)
            st.synchronize()


results = []
for kind in os.environ.get("CI_LOADS", "idle,gemm_null,gemm_stream,spin_stream,hostcopy").split(","):
    th = None
    ready = threading.Event()
    stop.clear()
    if kind != "idle":
        th = threading.Thread(target=bg, args=(kind,), daemon=True); th.start(); ready.wait(120)
        time.sleep(0.5)
    do_copy()  # warm
    hb0, ht0 = hostcopy_bytes[0], time.perf_counter()
    walls = []
    for _ in range(REPS):
        dst.zero_(); torch.cuda.synchronize(D)
        walls.append(do_copy())
    host_GiB_s = round((hostcopy_bytes[0] - hb0) / 2**30 / (time.perf_counter() - ht0), 2)
    ok = bool(torch.equal(dst, src.to(D)))
    stop.set()
    if th: th.join(60)
    gib = NB * BLK / 2**30
    r = dict(load=kind, mode=MODE, nstreams=NSTREAMS, hwq=os.environ.get("GPU_MAX_HW_QUEUES", "4"),
             peer_sdma=os.environ.get("HSA_ENABLE_PEER_SDMA", "<unset>"), sdma=os.environ.get("HSA_ENABLE_SDMA", "<unset>"), GiB=round(gib, 3),
             walls=[round(w, 4) for w in walls], wall_min=round(min(walls), 4),
             GiB_s_best=round(gib / min(walls), 2), GiB_s_worst=round(gib / max(walls), 2), byte_exact=ok, bg_host_GiB_s=host_GiB_s)
    results.append(r)
    log("RESULT " + json.dumps(r))


# ---- KV checksum cost (built-in --disaggregation-enable-kv-checksum). Production calls
# adler32_strided_checksum(...) -> .item() on the scheduler thread, on the thread's current
# stream (no set_stream in scheduler/prefill/decode: torch default stream), once per request
# at the final send (prefill) and once per prebuilt request (decode, before it joins the
# batch). Layout: CK_LAYERS buffers (default 78 = blocks/call in e2e) of PAGE_BYTES pages
# (36,864 B = 64 tokens x 576 B, the 1-page block size seen in e2e). DSA indexer state
# buffers are NOT included (adds a little). Loads: idle, and a long spin kernel loop on a
# torch stream (= forward running) -> shows CU contention and whether the default-stream
# .item() waits for the in-flight forward. Runs in the SDMA-unset pass only.
if os.environ.get("HSA_ENABLE_SDMA") is None and os.environ.get("CI_CKSUM", "1") == "1":
    try:
        del src, dst
        torch.cuda.empty_cache()
        from sglang.kernels.ops.memory.adler32 import adler32_strided_checksum
        L = int(os.environ.get("CK_LAYERS", "78"))
        PAGE_BYTES = int(os.environ.get("CK_PAGE_BYTES", "36864"))
        TOKS = [int(x) for x in os.environ.get("CK_TOKENS", "16384,65536,131072").split(",")]
        PT = 64
        maxp = max(TOKS) // PT
        bufs = [torch.randint(0, 255, (maxp, PAGE_BYTES), dtype=torch.uint8, device=S) for _ in range(L)]
        ptrs = [b.data_ptr() for b in bufs]
        torch.cuda.set_device(S)
        t0 = time.perf_counter()
        adler32_strided_checksum(ptrs, [PAGE_BYTES] * L, [torch.arange(4, device=S)] * L)
        log(f"RESULT {json.dumps(dict(arm='kv_checksum_jit_first_call', s=round(time.perf_counter() - t0, 3)))}")

        def ck(ntok):
            idx = torch.randperm(maxp, device=S)[: ntok // PT].contiguous()  # scattered pages
            torch.cuda.synchronize(S)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                adler32_strided_checksum(ptrs, [PAGE_BYTES] * L, [idx] * L)
                ts.append(time.perf_counter() - t0)
            return ts

        for kind in ("idle", "spin_stream"):
            th = None
            ready = threading.Event(); stop.clear()
            if kind != "idle":
                th = threading.Thread(target=bg, args=(kind,), daemon=True); th.start(); ready.wait(120)
                time.sleep(0.5)
            for ntok in TOKS:
                ts = ck(ntok)
                gib = ntok // PT * PAGE_BYTES * L / 2**30
                r = dict(arm="kv_checksum", load=kind, tokens=ntok, layers=L, GiB=round(gib, 3),
                         ms_min=round(min(ts) * 1e3, 2), ms_max=round(max(ts) * 1e3, 2),
                         GiB_s=round(gib / min(ts), 1), spin_ms=SPIN_MS if kind != "idle" else 0)
                log("RESULT " + json.dumps(r))
            stop.set()
            if th: th.join(60)
    except Exception as e:
        log(f"RESULT {json.dumps(dict(arm='kv_checksum', error=repr(e)[:300]))}")

# HIP IPC limits on crsuse2-m2m-276 — probing the same-node KV-transfer failure

Role: `hip-ipc`. Hypothesis under test (from the leader): `hipIpcGetMemHandle` is
unreliable on 276 and is the shared primitive behind the Round-8 RCCL failure
(`hipIpcGetMemHandle invalid argument`) and the Round-9 segfault (all 4
`sglang::schedul` die ~1 s into `run_event_loop`). All tests first-hand on **276**
via `spur exec 165913` + a `yihou-hipipc` container (image
`infera-sglang:v0519-yihou-0917-nextnfix-hicache`). GPUs verified idle before and
after; probe container removed; no files deleted.

Evidence tiers: **[hip]** direct `libamdhip64.so` ctypes call; **[torch]** torch's
battle-tested CUDA-IPC path; **[mc]** mooncake `TransferEngine`.

---

## Headline

1. **HIP IPC is NOT broken on 276 and has NO size threshold.** `hipIpcGetMemHandle`
   on a clean `hipMalloc` base returns rc=0 from 4 MiB to **200 GiB** (incl. the
   real 117 GiB KV size). torch cross-process CUDA IPC works. The hypothesis that
   a raw IPC size limit causes the failures is **not supported**.
2. **`MC_USE_HIP_IPC=0` (the intended fix) is the one that CRASHES.** Mooncake
   `register_memory` of a GPU buffer **segfaults at every size (4 MiB → 117 GiB)**
   under `MC_USE_HIP_IPC=0` (fabric/VMM mode), while the **default (IPC) mode
   registers cleanly at every size**. Do not deploy `MC_USE_HIP_IPC=0`.
3. **The isolated default-mode HIP path is healthy** — register 117 GiB, and a
   cross-GPU (GPU0↔GPU4, cross-NUMA) transfer, both succeed. So Round-9's
   default-mode segfault is **not** reproduced by single/2-process probes; its
   cause lives in the full-stack context (8 procs + RDMA/HIP dual registration +
   RCCL + scheduler warmup) I could not isolate before the leader's deployment
   went live on 276.

## Q1 — Does `hipIpcGetMemHandle` / IPC work at all on 276? YES

- **[hip]** export side: `hipMalloc(4 MiB)` then `hipIpcGetMemHandle` → `rc=0`.
- **[torch]** cross-process, spawn a child, hand it a `cuda:0` tensor through an
  IPC-backed `mp.Queue`: the child received it and computed the correct sum
  (`499500`). torch CUDA IPC uses `hipIpcGetMemHandle`+`hipIpcOpenMemHandle`
  internally, so **both** primitives work cross-process on 276.
- **Retraction:** my first `[hip]` probe reported `hipIpcOpenMemHandle` failing
  `17 (invalid device pointer)` even same-device 0→0. That was a **ctypes
  by-value 64-byte handle-struct marshalling bug in my test**, not a 276 fault —
  the `[torch]` result proves `hipIpcOpenMemHandle` works. Disregard that number.

## Q2 — Size threshold on `hipIpcGetMemHandle`? NONE

**[hip]** fresh process, full 287 GiB free, single `hipMalloc` base, then export:

| size | 4 MiB | 1 | 16 | 32 | 64 | 100 | 110 | **117** | 130 | 200 GiB |
|------|-------|---|----|----|----|-----|-----|---------|-----|---------|
| `hipIpcGetMemHandle` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 |

All rc=0 — no ceiling, including the real 117 GiB KV-pool size and beyond. Pointer
variants also export fine: base+offset, torch default `data_ptr()`, and torch
`expandable_segments:True` (tested at 256 MiB) — all rc=0. So neither size nor a
non-base/allocator pointer reproduces the RCCL "invalid argument" in isolation.

## Q3 — `MC_USE_HIP_IPC=0` (fabric) vs default (IPC) — DECISIVE, and a reversal

**[mc]** allocate a GPU buffer, init `TransferEngine` (HIP transport installed),
`register_memory(ptr, size)`:

| `MC_USE_HIP_IPC` | 4 MiB | 1 GiB | 16 GiB | 117 GiB |
|---|---|---|---|---|
| **default (IPC)** | rc=0 | — | — | **rc=0 OK** |
| **=0 (fabric/VMM)** | **SEGV(139)** | **SEGV** | **SEGV** | **SEGV** |

`MC_USE_HIP_IPC=0` core-dumps inside `register_memory` at **every** size,
including 4 MiB — a fundamental incompatibility, not size-related. Strong
inference: fabric mode drives the VMM shareable-handle path
(`hipMemExportToShareableHandle`, needs `hipMemCreate`/`hipMemMap` memory), but the
KV pool is plain `hipMalloc`/torch memory, so the export crashes. **The default
(IPC) mode is the one that works** — it registered the full 117 GiB cleanly.

**Consequence:** `MC_USE_HIP_IPC=0` is not a viable fix; it will segfault at
registration immediately (and could masquerade as the Round-9 crash). Keep
`MC_USE_HIP_IPC` at its default.

## Cross-GPU HIP transfer (the real KV path) — works in default mode

**[mc]** 2 processes, same host, SRV registers a 2 GiB buffer on **GPU4**, CLI on
**GPU0** does `transfer_sync_read` (cross-NUMA, HIP transport, default IPC):
`transfer_rc=0`, data verified (`all 0xEE`), no crash. So the isolated
register→cross-GPU-open→transfer path over HIP IPC is healthy on 276.

## Why Round 9 is still not explained (scope of these probes)

Every default-mode probe here **succeeds**. What I did **not** reproduce, and where
the Round-9 default-mode segfault likely lives:
- **8 concurrent processes** (4 prefill GPU0-3 + 4 decode GPU4-7), each registering
  ~117 GB and cross-opening peers. (Q4 — not run; the leader's live deployment
  took 276 before I got to it, and running it would collide with their GPUs.)
- **Dual RDMA+HIP registration with dmabuf.** The real legs have ionic HCAs, so
  `register_memory` also does the RDMA dmabuf export (`MOONCAKE_DISABLE_HIP_DMABUF=0`).
  My probes ran HIP-only (0 HCAs — ionic's userspace provider is ABI-incompatible
  in a bare container; the real container mounts host `libionic.so` via
  `HOST_RDMA_LIB`). I attempted a 117 GiB mlx5(rdma+hip) registration but it OOM'd
  on leaked VRAM (below) and then the deployment went live, so this is **untested**
  — it remains a candidate for the Round-9 crash.
- **RCCL / scheduler warmup interaction.**

## Operational note — fabric-mode crash leaks VRAM

The `MC_USE_HIP_IPC=0` SIGSEGVs left **~263 GiB orphaned on GPU0** with **no KFD
PIDs** — it did **not** free on process death, and persisted through `docker rm`.
It **self-reclaimed ~20 s** after the container was removed. So a crashed HIP
process temporarily strands its VRAM; budget a short settle before reusing the
GPU. (All 8 GPUs verified back at the ~284 MiB idle baseline at hand-off.)

## Source analysis (GPU-free) — the registered pointers are torch bases, NOT VMM

Question: are the pointers sglang hands mooncake `register_memory` allocation
bases of IPC-capable memory, or VMM/pool pointers that `hipIpcGetMemHandle`
rejects? Read from the installed source in the image.

Registration path: `conn.py:325 register_buffer_to_engine()` →
`_registerable_regions()` (conn.py:301, dedups exact `(ptr,len)`) → registers
`kv_args.kv_data_ptrs`/`aux`/`state`. Those pointers come from the pool's
`get_contiguous_buf_infos()`.

For GLM-5.2 (`architectures: ["GlmMoeDsaForCausalLM"]`, a DeepSeek-sparse/MLA
model) the pool is `deepseek_v4_memory_pool.py`. Its `get_contiguous_buf_infos`
(:862) registers `buf.data_ptr()` for each **per-layer `torch.zeros` KV buffer**
(`self.kv_buffer = [torch.zeros(...) per layer]`, allocation at :139/:208/:378/
:600; pointer list at :645, :273). So the registered pointers are **per-layer
torch-allocated KV buffer `data_ptr()`s — regular hipMalloc-backed (torch caching
allocator) memory, one pointer per layer buffer, not slices of a single giant
unified arena.**

**The one path that WOULD produce non-IPC-capable pointers — CUDA-VMM
post-capture backing (`KvVmmBufferOwner`, memory_pool.py:2364, whose `data_ptr()`
is a VMM VA that `hipIpcGetMemHandle` rejects) — is DISABLED here on two
independent grounds** (`is_post_capture_kv_active` → `post_capture_kv_sizing_planned`,
overrides.py:1799):
1. `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)` (environ.py:579),
   default off, and the harness/image never set it.
2. `if mla_enabled: return False` — GLM-5.2 is MLA/DSA, so it short-circuits off
   regardless. (`is_deepseek_v4(...) → return False` would also apply.)

So `post_capture_active=False`, the pool takes the regular `torch.zeros` path, and
the registered pointers are torch allocation bases. This is consistent with the
[hip] result that `hipIpcGetMemHandle` on a torch `data_ptr()` **and** on a
base+offset both return rc=0. **The "pointer is a VMM / non-IPC-capable / pool
offset" prime suspect is REFUTED for this deployment.**

Residual, low confidence (source alone can't settle): torch's caching allocator
may place several per-layer KV buffers inside one large segment, so their
`data_ptr()`s are distinct offsets into the **same** underlying hipMalloc block.
`hipIpcGetMemHandle` still succeeds on each (returns the containing block's
handle), but mooncake would then register multiple regions that resolve to the
same IPC block at different offsets, and `_registerable_regions` only dedups
*exact* `(ptr,len)` matches, not overlaps. Whether overlapping IPC registrations
misbehave in the HIP transport is untested — a possible, unconfirmed lead, not a
conclusion.

## Bottom line for the leader

- Keep `MC_USE_HIP_IPC` **default** (IPC). **Do not set `=0`** — it segfaults on
  any GPU registration.
- The raw IPC primitive and the isolated mooncake HIP path are healthy at 117 GiB
  on 276, so the Round-9 default-mode segfault is a **full-stack** effect, not a
  raw-IPC limit.
- The registered pointers are per-layer torch KV-buffer bases (not VMM, not a
  unified-pool offset); VMM post-capture backing is off. The pointer-type suspect
  is refuted.
- Prime remaining suspects: the 8-process dual RDMA+HIP dmabuf registration at
  117 GiB (concurrency/aggregate), an RCCL/warmup interaction, or (low confidence)
  overlapping IPC registrations from torch-segment-shared layer buffers.

---

## Addendum (post-instrumented-run source work, GPU-free)

### `Found 1 HCAs` — RESOLVED, not a defect
The leader's `GLOG_v=2` run showed `topology.cpp:127 Device ionic_3 port 1 is
available → Found 1 HCAs` for DP rank 3: the `RDMA_DEVICE` JSON maps each local
rank to its own ionic device, so **one HCA per rank is correct**. The earlier
"confirm N HCAs" flag from §9 is cleared.

### Allocator-pointer question — FULLY CLOSED (torch bases, all custom paths off)
Beyond §"Source analysis" above: the KV buffers are allocated in
`deepseek_v4_memory_pool.py::_create_buffers` as **one `torch.zeros(...)` per
layer** via `create_buffer` (:137) — separate per-layer allocations, not one
sliced tensor. Allocation goes through `torch.cuda.use_mem_pool(self.custom_mem_pool)`
**only if a custom pool is set**, else `nullcontext()` (default torch caching
allocator). `enable_custom_mem_pool = SGLANG_MOONCAKE_CUSTOM_MEM_POOL is not None`
(`mem_cache/utils.py:101`), and that env `EnvStr(None)` is **unset by our harness**
(the `SGLANG_MOONCAKE_CUSTOM_MEM_POOL:"True"` hits in the repo are all cached
**gb200 / NVIDIA-MNNVL** reference recipes under `.cache/`, not our engine.sh/config
— the custom pool exists "only for MNNVL/Barex PD disaggregation"). So all three
special paths are OFF: no VMM post-capture, no custom mem pool. The registered
pointers are per-layer torch `data_ptr()`s of GB-scale allocations (dedicated
hipMalloc segments → bases), and `hipIpcGetMemHandle` works on them. **The
pointer-type suspect is refuted; nothing here explains RCCL's `invalid argument`
(which the leader also localized to rounds 7/8 only, likely a separate,
GPU-layout-specific issue, not this deployment's KV registration).**

### Where the segfault lands — scheduler.py:1894, native HIP stream handle
The crash line (`run_event_loop`, scheduler.py:1868) is:
```
1893  while (
1894      self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream
1895      and _redraws < 64
```
i.e. accessing the **native HIP stream handle** (`torch.Stream.cuda_stream`) of
`schedule_stream`/`forward_stream` in the overlap-scheduler **stream-alias redraw**.
This block runs only when `self.enable_overlap` (default ON —
`not disable_overlap_schedule`, scheduler.py:486) or `pp_size>1`.

Native-code candidates in the window from `Tree cache initialized`
(`mem_cache/registry.py:290`) to this line, in order:
- `init_all_cuda_graphs()` (scheduler.py:1123) — **HIP graph capture** (the leader
  reports this completes);
- `device_module.Stream(priority=0)` (1127, and again at 1897) —
  `hipStreamCreateWithPriority`;
- `with device_module.stream(forward_stream): model_runner.prewarm_sampling()`
  (1135) — native warmup kernels;
- `run_event_loop`: `triton_load_watch.install()` (Triton internals hook), then the
  redraw's `.cuda_stream` handle access (**1894, the crash**) and
  `device_module.StreamContext`.

**Reading:** a segfault *at a bare `.cuda_stream` property read* is almost never a
bug in that line — the handle access is trivial. It is the signature of an
**already-corrupted HIP context/stream state** (from a prior async HIP error,
graph capture, or driver fault) surfacing at the next native touch. That fits the
leader's non-determinism (Round-9 both legs SIGSEGV; Round-11 decode SIGSEGV but
prefill SIGQUIT exit -3) — a poisoned GPU/driver state manifests differently per
process. **It is not mooncake** (its last activity was 3.6 min earlier), consistent
with the exoneration. Prime aim next: what corrupts the HIP context between graph
capture and the event loop — candidates are the HIP graph capture path on this
ROCm build, the overlap dual-stream (schedule_stream vs forward_stream) setup, or a
lower-level driver/hardware fault (recall the fabric-mode crash also left a
driver-level VRAM leak, §"Operational note" — this box's HIP runtime shows other
crash-state fragility).

### Scope correction — TWO findings, not one (my earlier "unified theory" was wrong)

I earlier proposed collapsing every failure into the RCCL start race, with
`scheduler.py:1894` as its only downstream surface, and concluding 276 is fine.
**That is refuted by rounds 12/13 and I retract it** (same reflex as the ctypes /
`"rdma,hip"` retractions earlier in this session). The honest split:

1. **RCCL communicator start RACE — real, same-node-specific, FIXED.** Two 4-GPU
   RCCL communicators (the prefill leg and the decode leg) initialising
   concurrently on one host trip over each other in intra-node P2P setup
   (`rccl p2p.cc:256 hipIpcGetMemHandle failed : invalid argument`). Start-order
   dependent (rounds 7/8 and 15 have identical GPU assignment yet a *different*
   leg fails), impossible cross-node (the two inits are on different machines),
   and resolved by a same-node start gate (round 16: wait for leg *i*'s `/health`
   before starting *i+1*). Scope: rounds 7/8/15, and **plausibly** the
   concurrent-start segfaults of rounds 9/11 via the mechanism below (the failed
   IPC poisons the context; the other leg dies at 1894). This is the only failure
   the start gate addresses.

2. **Solo-leg 276 segfault — NOT the race, UNEXPLAINED.** Rounds 12 and 13 were
   **single-leg** decode runs on 276 (one `engine.sh decode` after `stop.sh`;
   etcd + one engine, nothing else), and segfaulted on all four ranks anyway;
   round 14 was the identical single-leg run on 137 and **served**. With one leg
   there is **no second communicator to race with**, so the start race cannot
   reach it. I looked for a race path to a solo leg and found none: the leg's own
   4 DP ranks do init RCCL among themselves, but that intra-leg init is *identical*
   on 137 and 276, so it cannot be the 137-serves / 276-segfaults discriminator.
   The `scheduler.py:1894` finding is the **mechanism** here too (a corrupted HIP
   context surfacing at the next bare native stream-handle read), but the **cause**
   of that corruption on a solo 276 leg is **not identified**. Because the software
   stacks are byte-identical (kernel/amdgpu/firmware/rocm all equal, 137==276),
   the differing variable is **machine STATE, not version**. Untested candidates:
   residual KFD/driver state on 276 from the prior tenant's vLLM that was killed at
   Round 1; the still-running co-tenant `yzhou_model`; 276's k8s/flannel/spur host
   state. (My own observation that a HIP crash on 276 stranded 263 GiB of VRAM for
   ~20 s is consistent with a box that mishandles HIP crash state — a symptom, not
   a demonstrated cause.)

**Correction to my "Bottom line" above:** the claim that 276 "would come up clean
under the start gate" and that 137 "works because of the gate, not because it is a
healthier box" is **not supported** — round 14 (137 solo, no gate) served, and
rounds 12/13 (276 solo, no race) did not. The gate fixes finding 1; finding 2 is
open. The decisive future test (only if the user authorises 276 time) is a
**gated two-leg run on 276**: under this two-finding reading it is predicted to
still fail at 1894 (finding 2 is node-state, not the race), which makes it a real
test with a real prediction on each side.

# Mooncake HipTransport: IPC writes to sub-allocated buffers land at the allocation base

Status (2026-10-10): **known upstream, fix open and approved, not merged.**
Our image (`infera-sglang:v0521-yihou-1009`, mooncake built at `faae8dd4`) and Mooncake `main`
(`bb6d1126`, 2026-10-10T06:18Z) both carry the bug. No separate PR from us is needed; the
action item is to carry the upstream fix in our image (see "What we should do").

## Summary

`HipTransport::registerLocalMemory` (IPC branch) publishes `desc.addr = addr` together with
`hipIpcGetMemHandle(addr)`. A HIP IPC handle refers to the **whole allocation** that contains
`addr`, and `hipIpcOpenMemHandle` on the peer returns a mapping that starts at the
**allocation base**. `relocateSharedMemoryAddress` then computes
`dest - entry.addr + shm_addr`, i.e. it treats the allocation base as the registered buffer's
start. For a buffer that begins `N = addr - allocation_base` bytes into its allocation, every
transfer lands `N` bytes too low. `transfer_sync_write` returns 0 and nothing is logged:
**silent corruption** of whatever precedes the buffer in the same allocation, and the buffer's
own tail is never written. Reads return the wrong bytes the same way.

PyTorch's caching allocator routinely places tensors inside larger segments, so any framework
that registers individual tensors can be affected. The upstream PR (#4340) reports exactly
this in SGLang PD on MI355X: "PyTorch placed DSA index-K buffers and a MoE weight-scale tensor
in one allocation. KV transfers overwrote the weights." (second-hand; our own production check
is the YIHOU-REGBASE log, below).

Code (mooncake `faae8dd4`, `mooncake-transfer-engine/src/transport/hip_transport/hip_transport.cpp`):
- `:687-700` IPC registration: `hipIpcGetMemHandle(&handle, addr)`, `desc.addr = (uint64_t)addr`.
- `:128-138` `openIPCHandle`: `hipIpcOpenMemHandle(shm_addr, handle, hipIpcMemLazyEnablePeerAccess)`.
- `:799-806` relocation: `dest_addr = dest_addr - entry.addr + ((uint64_t)shm_addr)`.
- The fabric branch (`:723-747`) already resolves `hipMemGetAddressRange` -> `real_addr`; the
  IPC branch does not.
- NVLink: `nvlink_transport.cpp:1003-1011` (main) has the identical IPC pattern. The
  intra-node NVLink transport (#1622, merged 2026-03-12) and TENT NVLink (#1831, merged
  2026-04-08) were fixed earlier by registering at the allocation base; `HipTransport` and
  `NvlinkTransport`'s `cudaIpc*` branch were missed.

## First-hand evidence (crsuse2-m2m-136, MI355X, ROCm 7.2.0, image v0521)

1. HIP primitive (`diag/ipc_xcontainer2.yihou.py`, `diag/ipc_multiopen.yihou.py`): exporter in
   one container, importer in another (`--network host --ipc host`), both IPC modes
   (default dma-buf and `HSA_ENABLE_IPC_MODE_LEGACY=1`). For an interior pointer (a slice at
   +37 MiB+4 KiB of a 256 MiB tensor; a 4 MiB tensor at +4 MiB inside a 20 MiB caching-allocator
   segment; per-layer 16 MiB views of one tensor), `hipIpcGetMemHandle` and
   `hipIpcOpenMemHandle` return 0, byte 0 of the mapping is the allocation base, and the
   exported data is at `opened + (ptr - hipMemGetAddressRange(ptr).base)`. A raw
   `hipMalloc(256 KiB)` (ROCr fragment allocator) exported from its own base is fine. Every
   open of the same allocation gets its own whole-allocation mapping (no error on re-open).
2. Real TransferEngine (`diag/te_pair.yihou.py`, `diag/run_te_pair.yihou.sh`): 'decode'
   container (`HIP_VISIBLE_DEVICES=4,5,6,7`) registers 8 per-layer `torch.zeros(64 MiB)`
   plus one 64 MiB view at +4198400 of a 128 MiB tensor via `batch_register_memory`;
   'prefill' container (`0,1,2,3`) does `transfer_sync_write` per region. Engine-equivalent
   env: `MC_TE_FILTERS=<non-existent>` -> "Found 0 HCAs", "HIP transport installed".
   Result, both IPC modes: all 8 standalone regions correct; the interior region's write
   returned rc=0 but landed at holder bytes `0 .. 67108863` instead of `4198400 ..` -
   exactly `(ptr - base)` too low.
   Logs: `136:/mnt/m2m_nobackup/yihou_v0521dbg/tepair/20261010-060426/{legacy,default}/`.

## Minimal repro (upstream style, Python, one node, two processes)

```python
# exporter / destination (process A)
import json, torch
from mooncake.engine import TransferEngine
torch.cuda.set_device(0)
te = TransferEngine(); te.initialize(HOST, "P2PHANDSHAKE", "rdma", "")
holder = torch.full((128 << 20,), 0x11, dtype=torch.uint8, device="cuda")
off = (4 << 20) + 4096
buf = holder[off:off + (64 << 20)]; buf.zero_()          # interior of one allocation
te.register_memory(buf.data_ptr(), buf.numel())
publish(session=f"{HOST}:{te.get_rpc_port()}", ptr=buf.data_ptr(), n=buf.numel())
wait_for_writer(); torch.cuda.synchronize()
assert (buf == 0x5A).all()                               # FAILS
# holder[:64 MiB] now holds 0x5A instead

# writer / source (process B, other GPU or same GPU)
src = torch.full((64 << 20,), 0x5A, dtype=torch.uint8, device="cuda")
te.register_memory(src.data_ptr(), src.numel())
assert te.transfer_sync_write(session, src.data_ptr(), ptr, n) == 0   # passes
```

Run the two processes in separate containers or with different `HIP_VISIBLE_DEVICES`; a
transfer to the process's own segment uses `LOCAL_SEGMENT_ID` and bypasses relocation.

## Fix

Upstream PR **kvcache-ai/Mooncake#4340** ("[TE] Fix HIP and NVLink IPC relocation for
suballocated buffers", head `e8062249`; approved by Lzy17 2026-10-06, re-tested on MI355X /
ROCm 7.2.0 on 2026-10-08, author asked for merge). It implements the fix we would propose:

- registration resolves `hipMemGetAddressRange(addr)` (NVLink: `cuPointerGetAttribute`
  `RANGE_START_ADDR/RANGE_SIZE`, which does not need the allocation's context current), exports
  the handle from the **allocation base**, and keeps `desc.addr/desc.length` = the registered
  buffer (so lookup and `unregisterLocalMemory(addr)` are unchanged);
- a nonzero offset travels in `shm_name` as handle + host-order `uint64_t` (72 bytes); offset 0
  keeps the old 64-byte payload (wire-compatible with old peers);
- `OpenedShmEntry` stores the offset; both the first-open and the cached relocation paths add it;
- an old importer rejects the 72-byte payload ("Mismatched HIP data transfer method") instead of
  writing to the wrong address; an old exporter still omits the offset, so **both peers need the fix**;
- a range extending past its allocation is skipped with a warning (other transports may register it).

The HIP hunk is saved verbatim as `diag/mooncake-pr4340-e8062249.hip.yihou.diff` (full PR:
`diag/mooncake-pr4340-e8062249.full.yihou.diff`). Core of it:

```diff
-        if (!checkHip(hipIpcGetMemHandle(&handle, addr),
+        void* alloc_base = nullptr; uint64_t ipc_offset = 0;
+        const int allocation_rc = findIpcAllocation(addr, length, &alloc_base, &ipc_offset);
+        if (allocation_rc != 0) return allocation_rc < 0 ? -1 : 0;
+        if (!checkHip(hipIpcGetMemHandle(&handle, alloc_base),
 ...
-        desc.shm_name = serializeBinaryData(&handle, sizeof(hipIpcMemHandle_t));
+        desc.shm_name = serializeIpcHandle(handle, ipc_offset);
 ...
-            dest_addr = dest_addr - entry.addr + ((uint64_t)shm_addr);
+            dest_addr = dest_addr - entry.addr + ((uint64_t)opened.shm_addr) + opened.offset;
```

Alternative upstream approach, **#3548** (open since 2026-08-20, fixes issue #3546): register
the whole allocation (`desc.addr = base`, dedupe already-exported bases, unregister by base),
mirroring #1622/#1831. It changes the published buffer boundaries and unregister semantics;
#4340 is the narrower change and is the one with review + MI355X test results.

## Related but separate

- **#4488** (open, stacked on #4340) + **ROCm/rocm-systems#12978** (open): `hipIpcOpenMemHandle`
  looks up the exporter's process-local owner-device ordinal in the importer
  (`g_devices[owners_device_id]`, then `enableP2P`), so it returns `hipErrorInvalidValue` when
  the exporter's ordinal is out of the importer's range (e.g. TP2 prefill <- TP4 decode ranks 2,3),
  and may pick the wrong physical GPU when in range. Our legs each see exactly 4 GPUs
  (engine.sh `HIP_VISIBLE_DEVICES=$gpus`), so ordinals stay in range; our cross-container test
  with exporter ordinal 3 (physical 7) -> importer ordinals 0-3 passed. Not shown to be the
  cause of the 102 `201 invalid device context`; noted as a candidate only.

## What we should do

1. Production check (first-hand): mount `harness/mounts-v0521/src/conn.yihou-regbase.py` and grep
   `YIHOU-REGBASE ... summary n= interior=`. Any `interior>0` on the decode leg (the side whose
   registrations prefill writes into) means live KV transfers are being misplaced today.
2. Fix in our image: rebuild mooncake as `faae8dd4` + #4340 (`e8062249`, the HIP hunk applies
   to the files we build; check the `transfer_engine_impl.cpp` unregister hunk too) on **both**
   legs - the wire format requires both peers to be fixed. Re-run `diag/run_te_pair.yihou.sh`:
   the interior region must come back `correct=True`.
3. Interim mitigation if (1) shows interior registrations and a rebuild is not possible:
   none is clean on the sglang side (the caching allocator decides placement). Allocating the
   KV / DSA-indexer pools in a dedicated `torch.cuda.MemPool` or before other allocations only
   makes base-ness likely, not guaranteed. Prefer the mooncake fix.

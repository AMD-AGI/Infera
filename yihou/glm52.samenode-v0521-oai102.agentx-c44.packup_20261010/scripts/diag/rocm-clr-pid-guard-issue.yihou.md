<!-- DRAFT ONLY - not posted. Target: ROCm/rocm-systems (projects/clr, hipamd). -->

# [Issue]: `hipIpcOpenMemHandle` returns `hipErrorInvalidContext` for a valid handle from another container whose process has the same PID

## Summary

`hipIpcOpenMemHandle` rejects a handle as "opened by the same process" by comparing the PID
stored in the handle with the importer's PID. PIDs are only unique within a PID namespace. When
the exporter and importer run in different containers (separate PID namespaces) on the same
node, two unrelated processes can have the same PID number, and a valid cross-process handle
is rejected with `hipErrorInvalidContext` (201, "invalid device context").

This happens with both IPC modes (dma-buf default and `HSA_ENABLE_IPC_MODE_LEGACY=1`).

## Where

`projects/clr/hipamd/src/hip_memory.cpp` (tag `rocm-7.2.0`):

```cpp
// hipIpcGetMemHandle
ihandle->owners_process_id = amd::Os::getProcessId();                 // :3355
...
// hipIpcOpenMemHandle
if (ihandle->owners_process_id == amd::Os::getProcessId()) {          // :3381
  HIP_RETURN(hipErrorInvalidContext);                                 // :3382
}
```

This is the only `hipErrorInvalidContext` return in `hipIpcOpenMemHandle`. On `develop` the
same check is present (as of 2026-10-10).

## Reproducer

Two containers on one node, one GPU each, both started so that the test process is PID 1
(`docker run --entrypoint python3 ...`, default private PID namespace), `--ipc host`:

- exporter: `hipMalloc(64 MiB)`, `hipMemset`, `hipIpcGetMemHandle` -> writes the 64-byte handle to a
  shared file, keeps the allocation alive;
- importer: reads the handle, `hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess)`,
  `hipMemcpy` DtoH, verifies, writes back.

Results (MI355X / gfx950, ROCm 7.2.0, hip-runtime-amd 7.2.26015.70200-43, hsa-rocr 1.18.0.70200-43):

| IPC mode | PID namespaces | exporter PID | importer PID | `hipIpcOpenMemHandle` |
|---|---|---|---|---|
| legacy | private | 1 | 1 | 201 `hipErrorInvalidContext` |
| legacy | `--pid host` | 2455784 | 2455979 | 0, data correct both ways |
| default (dma-buf) | private | 1 | 1 | 201 `hipErrorInvalidContext` |
| default (dma-buf) | `--pid host` | 2457230 | 2457529 | 0, data correct both ways |

Script: a ~60-line ctypes test (no PyTorch) — attach `ipc_xcontainer.yihou.py` + `run_pidguard.yihou.sh`.

## Impact

Prefill/decode disaggregation frameworks (e.g. SGLang + Mooncake HIP transport) commonly run
the prefill and decode engines in separate containers on the same node with the same image and
start-up sequence, so their worker processes frequently get identical PIDs. We saw KV transfer
fail at the first request with `hipIpcOpenMemHandle failed (Error code: 201 - invalid device
context)` on exactly the rank pair whose PIDs coincided (prefill DP0 pid 973, decode DP0 pid
973); other ranks worked. Whether it triggers depends on how many processes/threads each
container spawned before forking its workers, so it appears or disappears with unrelated image
changes.

Related, same root cause (PID used as a node-wide identity): in dma-buf IPC mode ROCr names its
per-process fd-broker abstract socket `"\0hsa<getpid()>"` (`rocr-runtime/.../runtime.cpp`
`Runtime::IPCCreate`), which collides across containers sharing a network namespace
(`--network host`) and makes `hipIpcGetMemHandle` fail with `hipErrorInvalidValue`.

## Workaround

Run the communicating containers with `--pid host` (or otherwise ensure node-unique PIDs).

## Suggested fix

Do not use the bare PID as a process identity for cross-process IPC handles. Options:
- store and compare a per-process unique token (e.g. random 64-bit generated at runtime init),
  or the pair (PID-namespace inode from `/proc/self/ns/pid`, PID);
- or detect "same process" via the existing `amd::MemObjMap::FindIpcHandleMemObj` / allocation
  lookup instead of PID equality.

The same applies to ROCr's `"\0hsa<pid>"` socket name (include the PID-namespace inode or a
random token, and communicate it in the handle).

## Environment

- 8x AMD Instinct MI355X (gfx950), Ubuntu 22.04 containers, host kernel 6.8.0-107-generic,
  amdgpu 6.14.14
- ROCm 7.2.0 (rocm-core 7.2.0.70200-43), HIP 7.2.26015, ROCr 1.18
- Docker, `--network host --ipc host`, default PID namespace unless stated

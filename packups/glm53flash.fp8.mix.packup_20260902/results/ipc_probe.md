# HIP IPC across disjoint `HIP_VISIBLE_DEVICES` — measured, PASSES

Run 2026-09-02 ~11:46 UTC on `smci355-...-n05-29`, image
`infera/engine-sglang:glm53-c821c425`, gfx950 / MI355X, container ROCm 7.2.

## The question this answers

`examples/sglang_1p1d_glm5.3/cluster.singlenode.sh` gives the two PD legs
**disjoint** `HIP_VISIBLE_DEVICES` (prefill `0,1,2,3`, decode `4,5,6,7`). Each
process therefore sees four devices renumbered 0-3 and **cannot name, let alone
see, the other leg's physical GPUs**. `setupP2PAccess()` only iterates visible
devices, so peer access is enabled within a leg and never between them.

On this pin the same-host KV handoff is **HIP IPC over XGMI**, not loopback
RDMA. So the whole single-node PD topology rests on one unknown:

> Can `hipIpcOpenMemHandle` import a handle exported by a GPU the importing
> process cannot see?

Two attempts to answer it from source failed (ROCm header suggestive but silent
on this case; mooncake's HIP tests are all single-process/single-device, and
`grep -rn HIP_VISIBLE_DEVICES` over that repo returns nothing). Hence a
measurement.

## Answer: YES — and the bytes match, in the same container and across two containers

### Run 1 — two processes, one container, `--ipc=host`

Exporter `HIP_VISIBLE_DEVICES=4,5`, importer `HIP_VISIBLE_DEVICES=6,7`:

```
handle exported from device index: 0
IMPORT OK, bytes= 1048576
READ BACK: [7, 3, 9, 1, 4, 1, 5, 9]
MATCH
```

### Run 2 — two separate CONTAINERS, which is the shape PD actually runs

Same disjoint split; importer is a second container started with `--ipc=host`:

```
CROSS-CONTAINER IMPORT OK, bytes= 1048576
READ BACK: [7, 3, 9, 1, 4, 1, 5, 9]
MATCH
```

## Why the byte pattern matters more than the return code

A bare "import succeeded" would **not** have proved this. The handle records
device index `0`, and the importer's own ordinal `0` is a *different physical
GPU* — so a successful call could plausibly have mapped the importer's own local
memory. Writing `7,3,9,1,4,1,5,9` into the exporter's buffer and reading it back
through the imported mapping is what rules that out.

**If you repeat this, check the bytes, not the return code.**

## Consequence

`cluster.singlenode.sh`'s disjoint `HIP_VISIBLE_DEVICES` split **does not need to
change**. The queued fallback — give both legs all 8 GPUs and split with
`--base-gpu-id` so each process can see its peer's cards — is **not required**.

## Two gotchas

1. **`torch.cuda.cudart()` does not expose the IPC entry points in this build.**
   The obvious script fails immediately:
   ```
   AttributeError: module 'torch._C._cudart' has no attribute 'cudaIpcGetMemHandle'
   ```
   Use PyTorch's storage IPC path instead — `untyped_storage()._share_cuda_()` /
   `torch.UntypedStorage._new_shared_cuda(*info)` — which is what actually
   carries HIP IPC handles here.
2. **The importer must initialise its HIP context before importing.** A bare
   `torch.zeros(1, device='cuda:0')` first; without it the import path has no
   context to attach the mapping to.

## Scope — strong evidence, not proof

- Split was **physical cards 4,5 vs 6,7** of this host, not the literal
  **0-3 vs 4-7** the kit uses. Same node, same XGMI fabric, same
  disjoint-visibility property — but not the identical pairing.
- 1 MiB buffer, one handle. This shows the **mechanism** works. It says nothing
  about transfer throughput, and nothing about mooncake's own registration path,
  which wraps `hipIpc*` rather than calling it the way this probe does.
- Not run under load, and not run with a real KV pool.

## Scripts, verbatim

Exporter (`HIP_VISIBLE_DEVICES=4,5`, holds 120 s):

```python
import torch, pickle, time
t = torch.full((1<<20,), 0, dtype=torch.uint8, device="cuda:0")
t[:8] = torch.tensor([7,3,9,1,4,1,5,9], dtype=torch.uint8, device="cuda:0")
torch.cuda.synchronize()
info = t.untyped_storage()._share_cuda_()
pickle.dump(info, open("/dev/shm/ipc_probe.pkl","wb"))
print("EXPORTED dev_idx=", info[0], " pattern 7,3,9,1,4,1,5,9", flush=True)
time.sleep(120)
```

Importer (`HIP_VISIBLE_DEVICES=6,7`):

```python
import torch, pickle
torch.zeros(1, device="cuda:0")            # init HIP context FIRST
info = pickle.load(open("/dev/shm/ipc_probe.pkl","rb"))
print("handle exported from device index:", info[0])
try:
    st = torch.UntypedStorage._new_shared_cuda(*info)
    v = torch.tensor([], dtype=torch.uint8, device="cuda:0").set_(st)[:8].tolist()
    print("IMPORT OK, bytes=", st.size())
    print("READ BACK:", v)
    print("MATCH" if v==[7,3,9,1,4,1,5,9] else "MISMATCH -- mapping is NOT the exporter memory")
except Exception as e:
    print("IMPORT FAILED:", type(e).__name__, e)
```

Container for run 2 was started with
`--network=host --ipc=host --device=/dev/kfd --device=/dev/dri` on the same
image, and removed immediately afterwards by exact name.

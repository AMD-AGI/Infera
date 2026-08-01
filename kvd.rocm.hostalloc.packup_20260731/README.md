# kvd / hicache GPU fault on ROCm — root cause, fix, and read-back proof

**Ran:** 2026-07-31 (single day)
**Author:** yihou
**Status:** **PASS** — bug root-caused from first-hand measurement, fixed, and the
full L3 round-trip (write *and* read) proven on spur.

## Goal

SGLang's hierarchical KV cache (hicache / infera-kvd L3) aborted the prefill engine
with a GPU memory-access fault on MI355X (gfx950) whenever kvd was enabled at
long-context scale. The goal was to make **prefill-side kvd run with zero faults**,
understand *why* it faulted, and then prove kvd actually **serves** cached pages
rather than merely storing them.

Constraint from the operator: decode-side kvd stays OFF, MTP stays OFF.

**Success criteria**

1. Prefill leg runs with `--infera-kvd-socket` (hicache ON) and serves a full
   correctness suite with **0** `Memory access fault by GPU node-2`.
2. kvd read-back attributable: `gets_total` and `hits_total` climb while
   `sets_total` stays **flat**, on an engine whose in-GPU radix cache is empty.

## Result

| # | Criterion | Target | Actual | Verdict |
|---|---|---|---|---|
| 1 | GPU faults, prefill kvd ON | 0 | **0** (sequential suite, then 67 min under concurrent load) | PASS |
| 1 | Scheduler exceptions | 0 | **0** both legs | PASS |
| 2 | `gets_total` climbs | > 0 | **0 → 12,942** | PASS |
| 2 | `hits_total` climbs | > 0 | **0 → 12,942** | PASS |
| 2 | `sets_total` flat | no change | **12,942 → 12,942** | PASS |
| 2 | server-side corroboration | — | `cached=120000/120047` at all 5 depths | PASS |

Hit rate on replay: **12,942/12,942 = 100 %**, zero misses, zero re-stores.

## Root cause

On ROCm, `hipHostRegister` maps host pages at a device virtual address that
**differs** from the host VA. SGLang's hicache stores raw **host** `data_ptr()`s in a
device-side pointer table that a GPU kernel dereferences:

    pool_host/common.py     ALLOC_MEMORY_FUNCS -> alloc_with_host_register()
                              buffer = alloc_mmap(...)          # anonymous mmap
                              cudaHostRegister(buffer.data_ptr(), n, 0)

    memory_pool_host.py     DSAIndexerPoolHost.init_kv_buffer
                              self.index_k_data_ptrs = torch.tensor(
                                  [x.data_ptr() for x in ...],  # HOST VAs
                                  dtype=torch.uint64, device=<gpu>)
                            -> transfer_kv_all_layer_mla(dst_layers=..., ...)

On CUDA this is fine — `cudaHostRegister` maps at the *same* address, so a host VA is
directly dereferenceable from a kernel. On ROCm it is not, and gfx950 reports
`xnack-`, so there is no page-migration path to paper over it. The kernel is handed an
address that is not mapped on the device and the process aborts:

    Memory access fault by GPU node-2 (Agent handle: ...) on address <host VA>.
    Reason: Unknown.

**The fault address equals the host pointer exactly** — that is the fingerprint.

Measured directly (`scripts/probe_hostreg.py`), MI355X / gfx950 / ROCm 7.2.0, 8 MiB:

    [pin_memory]             host=0x7d7c2f600000  devPtr=0x7d7c2f600000  same=True
    [mmap + hipHostRegister] host=0x7d3bee790000  devPtr=0x7d3bede00000  same=False
    [+hipHostRegisterMapped] host=0x7d3bed600000  devPtr=0x7d3becc00000  same=False
    [+Portable|Mapped]       host=0x7d3bec400000  devPtr=0x7d3a97600000  same=False
    [MAP_PRIVATE]            host=0x7d3a96e00000  devPtr=0x7d3a96400000  same=False
    gcnArch=gfx950:sramecc+:xnack-      amdgpu.noretry=-1

No flag combination makes `hipHostRegister` produce a coincident pointer.

## The fix

`patches/patch_hicache_rocm_host_alloc.py` — register ROCm in `ALLOC_MEMORY_FUNCS` to
use `alloc_with_pin_memory` (torch `pin_memory=True`, i.e. `hipHostMalloc`), which
returns memory whose device pointer **is** the host pointer. This is exactly what the
existing `"npu"` and `"musa"` entries already do, for the same reason.

Smallest change that makes the existing pointer-table design correct on ROCm. The
alternative — translating every host pointer through `hipHostGetDevicePointer` before
building the tables — is the right upstream shape but touches ~10 call sites.

**Upstream status:** no sglang PR or issue exists for this (`gh search`, 2026-07-31).
Not fixed upstream.

## Why this was never seen on vultr

Not a fabric difference. The sanctioned kit `better/08`'s own limitations section
records that every request there was sequential and ≤ ~6,200 tokens against a 16 GB
host pool (573 MB resident). The device→host write-back path is only entered once
hicache actually backs pages up; spur's 120K-token prompts entered it immediately.
**The bug is workload-specific, not cluster-specific** — vultr would fault too at this
scale.

## How to reproduce

See `REPRODUCE.md`. TL;DR: run `scripts/probe_hostreg.py` (30 s, no kernel launch, no
fault) to see the pointer mismatch, then `scripts/micro_writeback.py` (~30 s) to make
the real kernel fault and un-fault by flipping one allocator.

## Folder map

- `REPRODUCE.md` — step-by-step, from a clean node to both proofs
- `environment.md` — exact HW/SW, pinned image digests, git SHAs
- `scripts/` — micro-repro, pointer probe, engine teardown, replay harness
- `patches/` — the fix, with full rationale in its docstring
- `results/` — kvd counters before/after, replay log
- `notes.md` — refuted hypotheses, gotchas, what is still unexplained
- `notes/working_process.md` — the debug loop round by round
- `notes/readback_result.md` — the restart-and-replay write-up

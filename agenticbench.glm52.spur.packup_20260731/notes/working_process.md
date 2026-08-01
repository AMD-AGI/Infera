# kvd-on-spur GPU fault — debug loop

**Target (pass/fail).** Prefill leg runs with `--infera-kvd-socket` (hicache ON) and
serves the full correctness suite — 4 short factual + 5-depth needle at ~120K tokens —
with **0** `Memory access fault by GPU node-2` in `prefill.log`. Decode leg keeps
kvd OFF (operator instruction). MTP stays OFF.

**Constraint.** One e2e boot is ~8 min + ~2 min needle. Round trip ≈ 15 min. So each
round must test as many *non-interfering* hypotheses as the run allows.

---

## Round 1 — evidence gathering (no GPU spent)

Directory: `round1_evidence/`. Sources: archived round-1 logs, sglang source in the
running container, the two sanctioned kvd kits, PR #56 / #58 via `gh`.

### What the logs actually say

| fact | evidence |
|---|---|
| Decode faulted on a **64-token** decode batch | `decode.strings.txt:3003` `Decode batch, #running-req: 1, #token: 64` immediately precedes the fault |
| Prefill faulted at the **last chunk** of a 13-chunk prefill | `prefill.strings.txt:1546` `#pending-token: 5359` (the tail chunk) then fault on the next line |
| Both legs ran `hicache_mem_layout='layer_first'`, `hicache_io_backend='kernel'`, `write_policy='write_through'` | `server_args=...` line, prefill log |
| The hybrid DSA pool was attached on every rank | `Attached hybrid DSA pool stack to HiRadixCache: pools=KV + INDEXER, transfer_layer_num=78` ×8 |
| hicache threads were alive at abort | decode faulthandler shows `cache_controller.py:1192 backup_thread_func`, `:972 prefetch_io_aux_func`, `:1028 prefetch_thread_func` |
| kvd never received a byte | `statctl` counters all zero on both nodes |
| The gloo `Connection closed by peer` tracebacks are **downstream** | they appear *after* the fault line, on the ranks that did not fault |

So the fault is inside sglang's GPU-side hicache transfer, **before** anything reaches
kvd. It is **not** long-context-specific: decode died on 64 tokens.

### Differential vs. the known-good reference (vultr, kit `better/07`,`08`,`09`)

Identical: sglang `0.5.15.post1`, ROCm 7.2.0, gfx950/MI355X, `page_size=64`,
`kv-cache-dtype fp8_e4m3`, `chunked_prefill_size=8192`/rank, `write_through`,
`layer_first`, `hicache_storage_backend='dynamic'`, and the same
`Attached hybrid DSA pool stack ... transfer_layer_num=78` line. **Zero faults there.**

Different — the complete list:

| knob | vultr (no fault) | spur (fault) |
|---|---|---|
| `--context-length` | 32768 | **131072** |
| `--hicache-size` | 16 GB | **32 GB** |
| host KV pool | 356,160 tok | **712,256 tok** |
| DSA indexer host alloc | 3.67 GB | **7.33 GB** |
| **workload size** | ≤ ~6,200 tok, **sequential** | **120,000 tok** |
| fabric / transport | ionic, GID 1, dmabuf OFF | mlx5_0, GID 3, dmabuf ON |
| PD legs with kvd | both | both (round 1) |

The largest kvd workload ever run in the sanctioned kits is `better/08`'s ~6,200-token
prefix (kit's own words: "*Anything at concurrency* — every request was sequential",
"*573 MB against a 16 GB host* — nothing about eviction or capacity pressure"). Spur
pushed **~20× that prompt length**. The regime is genuinely untested, so "kvd works on
gfx950" does not transfer to this workload without re-proof.

### Source reading — the write-back path under test

`memory_pool_host.py:3269 DSAIndexerPoolHost` (layer_first + kernel):

* `backup_from_device_all_layer` → `transfer_kv_all_layer_mla(src_layers=self.index_k_device_ptrs, dst_layers=self.index_k_data_ptrs, ...)`.
  Both are `torch.uint64` **pointer tables** on device. The kernel dereferences raw host
  pointers from GPU — the exact shape of failure that produces
  `Memory access fault ... Reason: Unknown` at a page-aligned address.
* `_get_indexer_page_indices` requires `host_indices.numel() % page_size == 0` and maps
  token indices → page indices by `// page_size`.

A page-count discrepancy exists in the allocation arithmetic:

```
HostKVCache (pool_host/base.py:109)   page_num        = size // page_size + 1
DSAIndexerPoolHost (:3309)            indexer_page_num = (size + page_size + 1) // page_size
                                      buffer rows      = self.page_num   (from anchor_host)
```

For spur's 712,256-token host pool: `page_num = 11130`, `indexer_page_num = 11131`
— **off by one**. The buffer is allocated with `page_num` rows, but `indexer_page_num`
is the value used as the page-count bound elsewhere. Note this identical off-by-one
holds for the vultr sizes too (5566 vs 5567), so it cannot be the *whole* story — it is
a necessary-but-not-sufficient condition that only turns fatal once the workload
actually reaches the last page. That is consistent with vultr (≤6,200 tok, never near
the end of an 356K-token pool) vs spur (120K tok prompts).

### PR #56 / #58 status in the image under test

* PR #56 `patch_mooncake_early_send_wait_event.py` — **applied and verified in bytecode**
  on both nodes. It addresses mooncake early-send, a *different* bug (silent partial KV,
  no crash). Not a candidate for this fault.
* PR #58's three DSA patches — **absent** from this image (`grep` for every marker
  returns 0 files). Confirmed: `dp_padded_rows`, `DP_PADDED`, `post_kernel_rows`,
  `dsa_backend_dp_sync`, `draft_cuda_graph_dp_vote` → 0.
  PR #58 patch 1 (`dsa_indexer_hip_dp_padded_rows.diff`) fixes *HIP/aiter paged-MQA
  sizing its output from DP-padded rows while `lengths` is sized to real rows*. That is
  a **DSA indexer** row-count bug on **HIP** under **DP-attention** — the same subsystem
  and the same class of defect as this fault. It is the single strongest candidate.
  Patch 4 is MTP-only (`draft`), irrelevant here. Patch 2b is page-table-under-MTP,
  irrelevant; patch 2a is a DP host-sync, plausibly relevant.

### Hypotheses to test (ranked)

| # | hypothesis | test | interferes with |
|---|---|---|---|
| H1 | PR #58 patch 1 (DSA indexer DP-padded rows on HIP) is required whenever the DSA indexer host pool is driven; its absence faults | apply patch 1 (+2a) in-container, rerun | — |
| H2 | `hicache_io_backend=kernel` write-back kernel is the faulting code; `direct` avoids it | `--hicache-io-backend direct` | H1 (different code path) |
| H3 | fault needs the workload to reach the last host page; a smaller `--hicache-size` or shorter ctx moves the boundary | vary `--hicache-size` | H1/H2 |
| H4 | `write_through` on every chunk is the trigger; `write_through_selective` writes far less | `--hicache-write-policy write_through_selective` | H1/H2 |

H1 and H2 touch different code (patch vs. backend selection) but both change the
write-back path, so they must not be varied in the same run if we want attribution.
H3/H4 are config-only and cheap.

**Round 2 plan:** prefill-only kvd (per operator instruction) + H1 applied, everything
else identical to the faulting round. That is the minimum change from a known-faulting
state, so a pass attributes cleanly to H1.

---

## Round 2 — micro-repro of the write-back kernel (ROOT CAUSE FOUND)

Directory: `round2_kernel_micro/`. **No model boot.** The faulting code needs only a
device indexer buffer, a host buffer, and two pointer tables, so it was reproduced
standalone in ~30 s instead of a ~15 min e2e round. This is what made a 6-way sweep
affordable.

`micro_writeback.py` rebuilds exactly what `DSAIndexerPoolHost` +
`DSATokenToKVPool` construct (page_size 64, 78 layers, page_stride 8448, the spur
run's own 3,380,992 device / 712,256 host token counts) and calls the same kernel,
`transfer_kv_all_layer_mla`.

### First shot reproduced the fault

```
device: tokens=3380992 pages=52829   host: anchor_page_num=11130 indexer_page_num=11131
  [head] dst[0]=0 dst[-1]=63 n=64
Memory access fault by GPU node-2 (Agent handle: 0x646a...) on address 0x7a2041914000.
```

It faulted on host pages **0–63**. That **refutes H3** and the off-by-one page-count
theory outright: the fault does not need the tail of the pool, or long context, or
concurrency. It also refutes H1 (PR #58 patch 1) and H4 as *causes* — no indexer
attention code and no write policy is involved in this probe at all.

### Allocation-mode sweep — the discriminator

Same kernel, same shapes, only the host allocation changed (`--host-alloc`):

| host allocation | result |
|---|---|
| `mmap(MAP_SHARED\|MAP_POPULATE)` + `cudaHostRegister(0)` ← **what sglang does** | **fault** @ host VA |
| `mmap` + `cudaHostRegister(hipHostRegisterMapped)` | **fault** |
| `mmap` + `cudaHostRegister(Portable\|Mapped)` | **fault** |
| `mmap(MAP_PRIVATE)` + `cudaHostRegister(0)` | **fault** |
| `mmap` no `MAP_POPULATE` + `cudaHostRegister(0)` | **fault** |
| `torch.zeros(..., pin_memory=True)` | **OK** |

### Mechanism, measured directly (`probe_hostreg.py`)

`hipHostGetDevicePointer` vs the host pointer, MI355X / gfx950 / ROCm 7.2.0:

```
[pin_memory]             host=0x7d7c2f600000  devPtr=0x7d7c2f600000  same=True
[mmap + hipHostRegister] host=0x7d3bee790000  devPtr=0x7d3bede00000  same=False
[+Mapped]                host=0x7d3bed600000  devPtr=0x7d3becc00000  same=False
[+Portable|Mapped]       host=0x7d3bec400000  devPtr=0x7d3a97600000  same=False
[MAP_PRIVATE]            host=0x7d3a96e00000  devPtr=0x7d3a96400000  same=False
gcnArch=gfx950:sramecc+:xnack-      amdgpu.noretry=-1
```

**Root cause.** On ROCm, `hipHostRegister` maps the pages at a **different device
address** than the host VA. SGLang's hicache stores raw **host** `data_ptr()`s in a
device-side pointer table that a GPU kernel dereferences
(`DSAIndexerPoolHost.init_kv_buffer` → `transfer_kv_all_layer_mla`), so the kernel
dereferences an address that is not mapped on the device. On CUDA the two addresses
coincide, which is why the design works there and nowhere in upstream flags it.
`gfx950` is `xnack-`, so there is no page-migration fallback. The fault address equals
the host pointer exactly — the fingerprint.

`hipHostMalloc` (torch `pin_memory=True`) returns memory whose device pointer **is** the
host pointer, which is what the pointer-table design requires — the same reason `"npu"`
and `"musa"` are already routed to `alloc_with_pin_memory` in `ALLOC_MEMORY_FUNCS`.

### Why vultr never hit it

Not a fabric difference. `better/08`'s own limitations section says every request was
sequential and ≤ ~6,200 tokens against a 16 GB host pool — 573 MB resident, "*nothing
about eviction or capacity pressure*". The device→host write-back path is only entered
once hicache actually backs pages up. Spur's 120K-token prompts entered it immediately.
The bug is **not** spur-specific; it is workload-specific, and vultr never reached it.

### Fix

`patches/patch_hicache_rocm_host_alloc.py` — route `ALLOC_MEMORY_FUNCS` to
`alloc_with_pin_memory` when `is_hip()`. Adds the `is_hip` import (`common.py` did not
have one) and a module-level `GLM52_ROCM_HOST_ALLOC = "applied"` literal so the patch is
provable in **bytecode**, not just source.

Validation of the fix at full production scale, before touching the engine:

```
case=sweep host_alloc=pin_memory layers=78  host 7.33 GB
  [head] dst 0..63                     OK
  [tail@anchor] dst 11066..11129       OK
  [last_indexer_page] dst 11130        OK
ALL OK -- no fault
```

Applied on **both** nodes; `compile OK`, `marker= applied  cuda-> alloc_with_pin_memory`,
`bytecode marker count: 1` on each.

Trade-off recorded: this bypasses `alloc_mmap`'s hugepage path on ROCm
(`SGLANG_HUGEPAGE_SIZE`, unset here).

Upstream: `gh search` finds no sglang PR or issue for this. Not fixed upstream.

**Status of the original hypotheses:** H1/H3/H4 refuted as causes; H2 (`io_backend`)
untested and now unnecessary — the bug is in the host allocation, common to both
backends.

---

## Round 3 — e2e validation of the fix (TARGET MET)

Directory: `round3_e2e_kvd_on/`. Config: **prefill kvd=1, decode kvd=0** (operator
instruction), kv-aware ON, MTP OFF, ctx=131072, router :8190. Only the ROCm host-alloc
patch differs from the round-1 faulting state — a clean single-variable change.

### Target: 0 `Memory access fault` with prefill kvd ON

| gate | prefill | decode |
|---|---|---|
| `Memory access fault` | **0** | **0** |
| `Scheduler hit an exception` | **0** | **0** |
| `ready to roll` | 1 | 1 |
| `infera-kvd adapter connected` | **8** | 0 (kvd off by design) |
| `Attached hybrid DSA pool stack` | 8 | — |
| `Errno 98` after ready | 0 | — |
| `disaggregation_decode_enable_radix_cache=True` | — | 1 |
| host pools allocated | 32.00 GB KV + 7.33 GB DSA indexer ×8 | — |

**The fault is gone.** Round 1 killed a leg on a 64-token decode batch and again at the
tail of a 13-chunk prefill; the same 120K-token needle workload now runs to completion
with `faults=0` on both legs. Target met.

### kvd actually stored data — first time on spur

```
entries        12942
host_bytes     22870688256   (22.87 GB)
long_bytes     22870688256
sets_total     12942
gets_total     0     hits 0   misses 0   evictions 0
```

Round 1 had **every counter at zero** — the GPU died before a byte reached kvd. The
write path now completes end to end. `gets_total=0` is expected and **not** a defect
here: a single sequential pass over one shared prefix is served by the in-GPU radix
cache, so L3 is written but never read back. Proving reads requires restart-and-replay
(`better/09`'s method), which is a separate experiment.

### Correctness — target met on the fault; needle is SEPARATE and still open

```
short factual: 4/4
needle:        3/5   (FAIL at depth 5% and 95%)
```

This is **not** a kvd regression: the same suite with kvd OFF on both legs scored
**4/5** earlier today, failing at a *different* depth (25%). kvd-on 3/5 vs kvd-off 4/5
is one sample each and the failing depth moves between runs.

**A first reading of the transcripts as a `max_tokens` problem was wrong.** A follow-up
probe (`needle_diag.py`) re-ran the two failing depths at `max_tokens` 256 and 1024:

| depth | max_tokens | exact | finish | 7-digit runs found |
|---|---|---|---|---|
| 5% | 256 | False | length | **none** |
| 5% | 1024 | False | length | **none** |
| 95% | 256 | **True** | length | `5385227`, 3538227, 5385382 |
| 95% | 1024 | **True** | length | `5385227`, 5385385, 5385522 |

Two things that a bigger budget does not explain:

1. **Depth 5% never produces a 7-digit run at all**, at either budget — the needle is
   not retrieved, not merely truncated. Quadrupling the budget changes nothing.
2. **Depth 95% returns the exact needle here but FAILED in the suite run**, same prompt,
   same `temperature=0`. The difference between the two is the prefix-cache state (the
   suite issues 5 needle requests sharing a filler prefix; this probe issued 2). So the
   output is not stable with respect to cache state.

Every response ends `finish=length` with degenerate repetition (`</think></think>...`,
`5385227
5385227
...`), i.e. the model does not terminate at 120K context.

**This is left explicitly open.** It is outside this loop's target (the GPU fault) and
is not caused by kvd, but it is *not* the benign truncation artefact I first called it,
and the needle score should not be quoted as a clean pass. Whether the cause is
long-context degeneration in GLM-5.2 itself, the chat template, or a genuine
cache-state-dependent KV defect needs its own experiment — a kvd-off/cache-cold A/B on
the same two depths would separate them.

### Attribution

Single variable changed vs. the faulting round: `ALLOC_MEMORY_FUNCS` → `pin_memory` on
HIP. Mechanism measured directly (`devPtr != hostPtr` under `hipHostRegister`), fix
validated standalone at production scale before the engine run, then confirmed e2e.
The fix is understood, not lucky.

---

# Round 4 — kvd read-back (Track 2 goal)

**Goal:** `gets_total > 0` and `hits_total > 0` with `sets_total` flat, on an empty
GPU radix cache. **Met.**

Restart-and-replay: engine killed (all 8 GPUs polled to VRAM 0 %), kvd daemon kept
alive, byte-identical prompts replayed.

    gets 0 -> 12,942   hits 0 -> 12,942   sets 12,942 -> 12,942 (FLAT)

Server independently reported `cached=120000` of `prompt_tok=120047` at every depth.
Two independent sources agree. Full write-up: `notes/kvd_readback_proof.md`.

Checked first, because CLAUDE.md records a config where L3 is written and never read:
`cache_controller.py:467` computes `prefetch_capacity_limit = 0.5 * mem_pool_host.size`,
non-zero at `--hicache-size 32`. The read path was structurally open.

# Round 5 — needle resolved, and a hypothesis refuted

Needle went **5/5** at ctx=262144 with prefill kvd ON, under the *same* forced
`temperature=0.0` that the leading hypothesis blamed.

That **refutes the sampling hypothesis**. `needle_sampling.py` (arm A temp=0 vs arm B
temp=1.0/top_p=0.95) was written but never needed to run — the confounder was
eliminated by a result, not by the planned A/B.

What actually correlates is **KV cache state**: every passing depth shows
`cached=120000`, i.e. the prefix was served from the warm store rather than
re-prefilled. The NOTE's own second-order observation had already pointed there and the
sampling hypothesis distracted from it.

**Left open deliberately:** *why* re-prefilling an identical prefix degrades retrieval
at some depths is not established. Asserting a mechanism from three runs would repeat a
mistake this repo has already paid for.

# Round 6 — Case A, first attempt: ABORTED (my error, not the deployment's)

`new_session_rate = 0.145`, scaled **linearly** from the probe. Aborted at t=1195 s:
in-flight pinned at the 48 cap (25 of the last 120 ticks at cap, mean 44.2), live
sessions climbed monotonically 40 -> 57. Backpressure was setting the load, so the run
was not the configured workload.

Server was healthy throughout — 0 GPU faults, 0 scheduler exceptions both legs. Purely
an offered-load error.

**The lesson:** the linear model predicted ~26 in-flight at 0.145; measurement gave
44-48. Service time grows with load. Interpolating between an unsaturated and a
saturated point is invalid. Preserved at `bench/caseA_full_ABORTED_saturated/`.

# Round 7 — Case A, rerun: DELIVERED

Rate anchored on the known-stable measured point (0.10 -> in-flight 10-27) with a
modest step to **0.110**. Result: in-flight p50 28 / p99 42 / max 46 — never pinned.

4007 s, 2919 requests, cache hit **0.8900** vs ideal 0.8899 (efficiency 1.0002, zero
eviction), **0 GPU faults**, **0 scheduler exceptions** with kvd ON throughout.

96 errors, all of them the client's hardcoded `aiohttp.ClientTimeout(total=240)`
(`agent_throughput.py:928`) — verified: 96 "timed out", 0 HTTP errors, 0 other
exceptions. Reported as measured, not adjusted.

**One tooling bug found and fixed:** `summary.json` / `metrics.jsonl` / `metadata.json`
are all written inside `if dashboard_mode and benchmark_name and data_dir:`
(`agent_throughput.py:1674`). Without `--dashboard-mode` the run prints to stdout and
persists nothing structured. The probe run lost its artifacts to this; `run_bench.sh`
now passes the flag.

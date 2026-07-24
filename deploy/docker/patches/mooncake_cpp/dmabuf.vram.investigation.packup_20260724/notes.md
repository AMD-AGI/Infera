# notes — the arc, the wrong turns, the instrument lessons

Reusable lesson: **an experiment's readings are only worth the instrument's
validation. Every wrong conclusion below came from trusting an un-calibrated
gauge (or a mis-aligned timestamp) instead of the malloc/registration ground
truth.**

## Background: why the dmabuf path at all (peermem vs dmabuf)

Two ways to register GPU memory for RDMA GPU-direct:
- **`ibv_reg_mr`** requires a **peermem kernel module** (`ib_peer_mem` /
  amdgpu-peermem). Without peermem, plain `ibv_reg_mr` **cannot register GPU
  memory**.
- **`ibv_reg_dmabuf_mr`** exports the allocation as a dma-buf fd — the path used
  when peermem is absent, and the modern/preferred one.

This investigation is about `ibv_reg_dmabuf_mr` because that is the path in play.

## The core finding (the CORRECT statement)

**No ODP on the NIC ⇒ `ibv_reg_dmabuf_mr` ALWAYS pins the GPU memory ⇒ KFD
double-counts the externally-pinned memory ⇒ reported available GPU memory
shrinks by ~pool-size, even though NO new physical VRAM is allocated.**

- The pin is unavoidable because ionic has **no ODP** (`*_odp_caps` all zero);
  the ionic provider is forced down `ib_umem_dmabuf_get_pinned` (the *pinned*
  umem variant — confirmed in kernel `ionic_reg_user_mr_dmabuf`).
- KFD counts the pool once as the original allocation, then a **second time**
  when it is pinned via the external dma-buf import → `hipMemGetInfo` free drops
  by ~pool-size at registration.
- Physically there is **no duplication** (TTM buddy unmoved) — it is a KFD
  **accounting** double-count, but a REAL functional loss: the double-counted
  memory cannot be handed out while the MR lives, so later allocs OOM.
- Reversible / leak-free: dereg removes pin + double-count; free returns.

## Experiment arc (code + result, in order) — all reproducible

Each iteration ran in the `dmabuf_probe` container on chi2798, fresh kill/run
per round, provider matched to host. Source evolved in
`src/dmabuf_verdict_mvp.cpp` (final = reordered version).

1. **OOM-drive (bare vs dmabuf):** fill pool, register, then keep mallocing
   until failure. Result: bare reg → next allocs fine; dmabuf reg → next alloc
   OOMs earlier by ~pool-size. First hard signal that dmabuf reg debits
   available memory. (ground truth: malloc fail.)
2. **Leak / 2× / pin discrimination (5-phase, both modes):** P0..P4 =
   baseline/alloc/register/dereg/free. dmabuf: hip_free drops ~pool at P2,
   returns at P3 dereg, baseline at P4. bare: no drop. → reversible, no leak.
3. **exp1 (100G+reg, then try malloc 150G) & exp2 (20× reg/dereg cycle):**
   ground truth = malloc success/fail. dmabuf exp1 step3 (while-registered)
   FAILS, step4 (after-dereg) SUCCEEDS → pin is the cause and is reversible.
   exp2 20 cycles → hip_free at baseline, final 150G succeeds → NO leak.
4. **Granularity sweep 1×100G / 100×1G / 1000×100M:** identical behavior; 1:1
   budget debit; no fd exhaustion at 1000 dma-buf fds. Real KV granularity
   (~100 MB) changes nothing.
5. **VRAM-vs-GTT with validated physical instrument (TTM buddy, handshake
   synced):** at register, hip_free drops by pool but **TTM physical unchanged**
   and GTT unchanged → the shrink is accounting, not a physical copy.
6. **DECISIVE — allocate-first-then-register (reordered):** occupy 150G first
   (~36G physical left), then register 100G. Registration **SUCCEEDS 100/100,
   TTM unmoved, hip_free 37→0.** Proves: reg needs no physical memory (no 2×
   physical) yet KFD double-counts the pin so available collapses below what
   physically remains. This is the clean proof of the core finding.
7. **4-layer no-pin diagnostic (nopin_diag.sh):** Layer-1 ODP = RED (all zero),
   Layers 2/3/4 GREEN → pin is forced by the NIC, not by config/driver/rdma.

## Wrong turn #1 — cherry-picking measurement sources

First verdict claimed "legit pin, physical 1×, harmless" from sysfs
`mem_info_vram_used` "not changing" at register. But that SAME sysfs source
**failed its own control** (a known-true `hipFree` didn't move it for ~3 s).
Citing a gauge's "no change" while it can't see a definite free = invalid. →
rule now in project CLAUDE.md: *observations report objective fact; never bend a
reading to fit a conclusion; validate each instrument against a known-true
control before trusting its "no change".* Also: I wrongly softened the finding
to "harmless" — it is NOT harmless; the double-count is a real available-memory
loss.

## Wrong turn #2 — believing a broken gauge over the malloc truth

Flipped to "physical 1×, no problem" from hip_free+sysfs while the malloc
ground truth (drive-to-OOM) showed reg made a pool's worth of memory
unallocatable. When a gauge and the malloc result conflict, the gauge is wrong.

## Wrong turn #3 — the timestamp mis-alignment (worst)

Added the host TTM buddy allocator (`amdgpu_vram_mm` line `total:/free:`) as a
real physical instrument (passed both controls). But the async sampler aligned a
P2-labelled sample to a TTM reading actually taken during the following
`malloc 150G` probe (free momentarily 186→36 during that alloc). From that
mis-aligned point I concluded "physical doubling 186→36 at register" — the
opposite of truth. Fixed with a **file-handshake** (MVP writes `sig/PHASE`,
blocks until host writes `sig/GO`) so each TTM sample binds to the exact phase
and P2 is sampled BEFORE any malloc probe. → never align logs by wall-clock
guess; bind sample to phase.

## Root cause — the 4-layer picture (nopin_diag.sh)

"No-pin" (dynamic attach + move-notifier) needs 4 layers all green:
| layer | check | chi2798 | verdict |
|-------|-------|---------|---------|
| 1 NIC ODP | ionic `*_odp_caps` | **all zero** | **RED — forces pin** |
| 2 kernel | CONFIG_DMABUF_MOVE_NOTIFY | =y (+symbol) | green |
| 3 amdgpu | 6.16.13, PeerDirect ok | new enough | green |
| 4 rdma-core | `ibv_reg_dmabuf_mr` symbol | present | green |

Layer 1 is the short pole: no ODP ⇒ forced `ib_umem_dmabuf_get_pinned` ⇒ pin ⇒
KFD double-count ⇒ available shrinks. Everything else supports no-pin, but the
NIC can't, so pin (and the double-count) is unavoidable on this hardware.

## Instrument reliability table (hard-won)

| instrument | tracks | reliable? | caveat |
|------------|--------|-----------|--------|
| hipMalloc/reg success | GROUND TRUTH | YES | trust unconditionally |
| hip_free (hipMemGetInfo) | KFD/HIP available (incl. double-count) | YES | matched malloc every phase |
| TTM buddy free (debugfs) | physical VRAM | YES | host-only; settle lag on free; BIND to phase |
| sysfs mem_info_vram/gtt_used | amdgpu BO counter | PARTIAL | blind to dma-buf pin; ~3s down-edge lag |
| rocm-smi --showmeminfo vram | (in-container) | NO | pid-ns global, didn't move on malloc |
| dmesg kfd/pin/reserve | kernel printk | NO (here) | pin emits no printk; silence ≠ absence |

## Dead ends worth remembering
- **dmesg for KFD accounting: silent.** The double-count does not print. That
  run produced nothing usable — don't repeat expecting reg-time KFD logs.
- **rocm-smi in a container: don't.** Use amdgpu sysfs/debugfs from the host.
- **Granularity is irrelevant** (1×100G ≡ 100×1G ≡ 1000×100M).

## Still open (separate issue)
Cross-node dma-buf RDMA **transfer** access-violation (task #23) — a
transport-layer problem, not the accounting question this pack-up closes.

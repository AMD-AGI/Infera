# ionic dmabuf GPU-VRAM RDMA: the "2× VRAM" is a measurement artifact (disproof)

**Ran:** 2026-07-23
**Author:** yihou (via Claude Code)
**Status:** ✅ RESOLVED — the reported dmabuf "2× VRAM shadow" **does not exist**; it is a `hipMemGetInfo` accounting artifact. `ibv_reg_dmabuf_mr` on new ionic is true zero-copy P2P (+0 real VRAM at registration).

## Goal

A colleague's cross-cluster probe (`original_probe_from_colleague/`) reported that
registering GPU VRAM for RDMA via `ibv_reg_dmabuf_mr` **doubles VRAM** (a 2 GiB
buffer costs 4 GiB) on AMD Pensando **ionic** NICs, and that plain `ibv_reg_mr`
**fails** (errno 14) on their old ionic stack. We needed to (a) verify on our
newer ionic stack, and (b) determine whether the 2× is real duplication or a
measurement error — driver bugs this blatant don't ship, so the hypothesis was
"measurement/usage artifact".

**Success criterion:** either make dmabuf register at 1× (no doubling), OR prove
with independent evidence that the 2× is not real device-VRAM duplication.

## Result

| Meter | hipMalloc(64 GiB) | **ibv_reg_dmabuf_mr(64 GiB)** |
|---|---|---|
| `hipMemGetInfo` free (what the probe used) | −64 | −64 again → looks like **2× (128 GiB)** |
| **amdgpu `mem_info_vram_used`** (driver's real VRAM) | **+64** | **+0** ← no second allocation |

The driver's own VRAM bookkeeping is flat across registration. If 2× were real, a
64 GiB buffer would consume 128 GiB true VRAM; it stays at 64. **Verdict: the 2×
is a `hipMemGetInfo` artifact (it reflects the P2PDMA/BAR mapping as "used free"),
not a physical copy.** dma-buf works exactly as designed — zero-copy P2P.

Bonus verdict on our stack (new ionic 26.03): `bare ibv_reg_mr(VRAM) = OK`,
`dmabuf = OK, 1×`. The old stack's `bare = FAIL_14` was the colleague's *real*
finding; the "dmabuf 2×" alongside it was the same artifact.

## How to reproduce

See `REPRODUCE.md`. TL;DR: run `scripts/run_hold2.sh` (or `run_hold_big.sh`) on an
**idle** GPU node — it registers a dmabuf MR and prints, per phase, both
`hipMemGetInfo` free AND the amdgpu sysfs true-VRAM; watch the true meter stay
flat at registration.

## Folder map

- `REPRODUCE.md` — step-by-step, copy-pasteable
- `environment.md` — exact ionic/kernel/GPU/libionic versions the result came from
- `original_probe_from_colleague/` — the colleague's probe **verbatim** (README + probe + run.sh)
- `scripts/` — our debug harness (v1/v2/PCIE export test, access sweep, per-buffer scaling, the hold-probe with true-VRAM sampling, kernel-config check)
- `results/` — captured outputs (the 4 GiB and 64 GiB three-meter runs; the VERDICT block)
- `notes.md` — the full debug narrative: every hypothesis raised and ruled out, the meter trap, and the source-backed mechanism

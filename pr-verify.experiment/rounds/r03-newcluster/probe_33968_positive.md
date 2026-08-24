# r03 — #33968 reproduces here. This is the positive control that was missing.

Measured 2026-08-24 on `crsuse2-m2m-237` (job 58799), inside
`infera-local:sglang-prverify-20260824`, via
`pr-verify.experiment/scripts/probe_host_devptr.py`.

## Result

```
device: AMD Instinct MI355X gcn=gfx950:sramecc+:xnack-
torch:  2.9.1+rocm7.2.0.git7e1940d4 hip=7.2.26015-fc0010cf6a

  [pin_memory]                host=0x747a08800000  devPtr=0x747a08800000  same=True
  [mmap + hipHostRegister]    host=0x747a08000000  devPtr=0x747a03600000  same=False
  [  + hipHostRegisterMapped] host=0x747a02e00000  devPtr=0x747a02400000  same=False
  [  + Portable|Mapped]       host=0x747a08000000  devPtr=0x747a03600000  same=False

VERDICT: host VA != device pointer for the three hipHostRegister strategies.
         The pointer-table design is UNSAFE with those strategies here.
```

**Deterministic**: three consecutive runs, all three `same=False` lines every
time, at different VAs each run (so it is not one stale mapping).

## Why this matters

`plan.md` step 3 said: *"On a candidate machine, run the probe first. If it does
not print `same=False`, that machine cannot validate this PR."* It prints
`same=False`. **This machine can validate #33968** — the first one that can.

n06-33 measured `same=True` for all four strategies at every size from 8 MiB to
7.33 GB, so it was a negative control like gfx942, and the PR's evidence stayed
historical. That is no longer the situation.

This is exactly the fault the PR describes: sglang's host pools hand `data_ptr()`
of host buffers to GPU kernels through a device-side pointer table, and stock
dispatch sends a `torch.device('cuda:0')` key to `alloc_with_host_register`
(confirmed independently by `validate_A.py`). When the registered device pointer
differs from the host VA, that table holds addresses the GPU cannot use.

## The n06-33 hypothesis is disproved

`context.md` and `plan.md` both name **amdgpu 6.14.14** as "the only known
uncontrolled variable left" behind n06-33's failure to reproduce.

**This cluster runs amdgpu 6.14.14 too** — and it reproduces. So the driver
version is *not* what separates a reproducing machine from a non-reproducing one.
That lead is closed; do not carry it forward.

What differs between the two boxes, measured:

| | n06-33 | crsuse2-m2m-237 |
|---|---|---|
| GPU | MI355X gfx950 | MI355X gfx950 (identical arch string) |
| amdgpu | 6.14.14 | **6.14.14** |
| host ROCm | 7.2.0 | 7.0.1 |
| container torch | `2.9.1+rocm7.2.0` | `2.9.1+rocm7.2.0.git7e1940d4` |
| container hip | `7.2.26015` | `7.2.26015-fc0010cf6a` |
| probe result | `same=True` x4 | **`same=False` x3** |

The container's torch/hip build hashes differ, and the host ROCm differs (7.0.1
vs 7.2.0) — note the *host* runtime is what backs `hipHostRegister`, and 7.0.1 vs
7.2.0 is the larger of the two deltas. `HSA_XNACK` is unset on both; this host
exposes no IOMMU groups.

**No mechanism is claimed.** Two variables moved at once (host ROCm, container
build), so this does not isolate a cause. What is established is narrower and
sufficient for the PR: *the fault is real and reproducible on gfx950*, which is
what #33968 needed and did not have.

The measurement that would isolate it: run this same probe against the same
container image on a host with ROCm 7.2.0. Not required to land the PR.

## What remains for #33968

The probe is the *first* half. `plan.md` step 3 also requires the HiCache
write-back repro on top of it: **stock must fault at the host VA, patched must
not.** That is now unblocked and worth doing — it was not possible on n06-33.

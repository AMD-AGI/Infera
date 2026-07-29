# R6 — single-node mix is unregressed by the fix

Date 2026-07-29 18:55. Node 11428 (m2m-029), single-node **mix** server (not PD),
DPA8 + MTP(steps=3, topk=1), CUDA graphs on, fix `GLM52_BUG2B_UNIFORM` applied.

## Why this test is required

The fix adds a collective (`all_reduce` over the TP group) to `draft()`, which is a
**shared** code path — mix uses it too. Mix already passed before the fix, so the only
question here is whether the fix breaks it. A new collective on a hot path is exactly the
kind of change that can deadlock a *different* configuration.

## Result

| test | result |
|---|---|
| boot + warmup | ✅ `ready to roll` |
| 4 × 24-token sequential | ✅ **4/4**, 0.39–0.64 s, `acc_len` 1.60–3.43 |
| conc=128 × 512 tokens | ✅ **128/128**, 12.8 s, all 8 dp ranks, `acc_len` mean 2.98 (2.35–4.00) |

132/132. No hang, no crash, spec-dec active throughout.

The added vote is harmless on mix for the reason the source predicts: in mix all ranks
compute `dsa_topk_indices` locally from the same code, so `_needs_eager_local` already
agrees across ranks and the all-reduce simply returns what each rank already had.

## Note on the boot failure that preceded this run

The first mix launch attempt died at weight load with
`size of tensor a (3072) must match the size of tensor b (6144)`. That is the documented
nextn `eh_proj` trap, **not** a consequence of this fix: the nextn patch had not actually
been applied on this node, because an ad-hoc `grep -q eh_proj` idempotency check returned
a false positive (`eh_proj` occurs elsewhere in the file). See PITFALLS P2. After applying
the patch properly and verifying with the exact post-patch text, the server booted clean.

## Cumulative totals for the fix

| arm | requests |
|---|---|
| PD, fix + instrumentation (R3) | 927/927 |
| PD, fix only, clean build (R5) | 457/457 |
| Mix, fix (R6) | 132/132 |
| **total** | **1516/1516** |
| PD, fix reverted (R4 control) | **0/4 — deadlock** |

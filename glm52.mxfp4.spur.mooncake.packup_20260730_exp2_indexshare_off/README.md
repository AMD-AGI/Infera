# Exp 2 — IndexShare off, patches 2 and 4 absent (llying's configuration)

**Ran:** 2026-07-30 (single day), AMD spur cluster `crsuse2-m2m`, 2 × MI355X nodes.
**Author:** yihou
**Status:** **PASS** — all criteria met. The deadlock our patches 2 and 4 exist to fix
**does not occur** in this configuration.

## Goal

Our patch set fixes a PD + DP-attention + MTP deadlock by making the draft graph/eager
decision uniform across the TP group (patch 4) and by reconciling page-table rows
(patch 2). A separate AMD run by **llying** (MI325X, GLM-5.2-FP8, branch
`llying/dev/glm5p2_fp8_exp`) reports a working PD + DPA + MTP deployment **without either
patch**, by instead turning the GLM-5.2 MTP *IndexShare* feature off:

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

This arm tests whether that alone is sufficient **on our platform and our baseline**. It
matters because IndexShare is the *source* of the rank-divergence: on the PD decode leg,
the guard term `dsa_topk_indices is None` is seeded from RDMA-shipped per-request payloads
(`eagle_disaggregation.py:54-59`), so it is a function of which requests each rank happens
to hold. Remove the seed and the term should stop diverging — no vote needed.

**What this arm runs:** patch 1 (padded rows) + patch 3 (nextn `eh_proj`) **only**.
Patches 2a, 2b and 4 are deliberately **absent** — their absence is the hypothesis under
test. `apply_arm.sh` asserts they are absent from the bytecode, so this is enforced rather
than assumed.

**Also required, and easy to miss:** MTP must be enabled on the **prefill** leg too. Our
other arms run MTP on the decode leg only. Without prefill MTP the prefill worker never
runs `_draft_extend_for_prefill`, never fills `req.output_dsa_topk_indices`, and never
registers the draft KV pool for RDMA — so the IndexShare seed could never reach the decode
leg in the first place, and "IndexShare off" would be untested rather than tested.

**Success criteria** (same as all three arms):

1. 4-prompt sequential probe → 4/4, with `spec_accept_length > 1`;
2. conc=32 × 512 tokens → 32/32, no hang, no `KVTransferError`.

## Result

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| 4-prompt probe | 4/4 | **4/4**, `acc_len` 2.00–3.43 | ✅ |
| conc=32 × 512 | 32/32 | **32/32** (run 1) | ✅ |
| conc=32 × 512, repeat | — (stability check we added) | **32/32** (run 2) | ✅ |
| conc=64 × 512 | — (headroom check we added) | **64/64** | ✅ |
| hangs / `KVTransferError` | 0 | **0** | ✅ |
| Traceback in either server log | 0 | **0** | ✅ |
| patches 2 & 4 absent from bytecode | asserted | **confirmed** by `apply_arm.sh` anti-markers | ✅ |

From the raw jsonl in `results/`:

| Run | ok | full 512 tok | acc_len mean | acc_len min | acc_len max |
|---|---|---|---|---|---|
| `stress_c32.jsonl` | 32/32 | 30/32 | 2.98 | 2.45 | 3.97 |
| `stress_c32_r2.jsonl` | 32/32 | 30/32 | 3.01 | 2.51 | 3.94 |
| `stress_c64.jsonl` | 64/64 | 60/64 | 2.98 | 1.95 | 4.00 |

All eight DP ranks served traffic in every run.

Configuration was verified live in the decode server's own startup banner rather than
trusted from the launcher — see `logs/decode.log`:

```
json_model_override_args='{"index_share_for_mtp_iteration":false}'
speculative_algorithm='EAGLE', speculative_num_steps=3
```

and the same `speculative_algorithm='EAGLE'` appears in `logs/prefill.log`, confirming
prefill MTP was genuinely on.

## Comparison with Exp 1 (same day, same cluster, same image)

| | Exp 1 (full patch set) | Exp 2 (IndexShare off, no patch 2/4) |
|---|---|---|
| conc=32 | 32/32, 32/32 | 32/32, 32/32 |
| conc=64 | 64/64 | 64/64 |
| acc_len mean | 2.80–2.85 | **2.98–3.01** |

Exp 2's acceptance length is **~5 % higher**. Two honest caveats before reading anything
into that:

- These are two different node pairs, and only ~96 requests per arm. We did **not** run a
  paired test, and no confidence interval was computed.
- The arms differ in **two** variables, not one: IndexShare, *and* prefill MTP. A higher
  accept length could plausibly come from the prefill leg now producing draft state.
  This kit does not separate them.

So: consistent with "turning IndexShare off costs nothing here", which is what llying
report (3.78/4 with it on and off) and what a reproducer on #32209 reports (3.239 vs
3.24). It is **not** evidence that IndexShare-off is faster.

## What this arm does NOT establish

- It does **not** show patches 2 and 4 are unnecessary. It shows one *alternative*
  configuration that avoids the same deadlock. Which to prefer depends on whether
  IndexShare is wanted.
- The workaround has a known **expiry**: upstream PR **#31477**
  (`[Spec][PD] Enable fused TopK for GLM-5.2 MTP IndexShare`) exists specifically to make
  IndexShare useful under PD. Today its consumer is disabled by
  `should_use_dsa_fused_topk`, which is *why* switching it off is free. When #31477 lands,
  the override starts costing (~3 % TPOT, per llying — not measured by us). As of
  2026-07-30 #31477 is open with `reviewDecision = REVIEW_REQUIRED` (**no approval**); its
  timing is unknown.
- conc=128 was **not** run. Criterion here was conc=32; 64 was added as headroom.
- No deadlock was *provoked* in this arm — we did not run a negative control that removes
  the IndexShare override to show the hang returns on **this** node pair. The bug-2b kit
  has that control for the patch-4 arm; this arm has no equivalent.

## Folder map

- `REPRODUCE.md` — cold-start reproduction, ordered and copy-pasteable
- `environment.md` — exact hardware, fabric, image and commit the numbers came from
- `notes.md` — the mechanism, why prefill MTP is required, gotchas
- `patches/` — the two patches applied (1 and 3)
- `scripts/` — every script that ran, copied verbatim
- `results/` — raw per-request jsonl
- `logs/` — full prefill / decode / router logs, uncompressed

# Notes — traps, wrong turns, and the corrections that changed conclusions

Written as what / why / how / context. The wrong turns are kept deliberately: a
packup that reads as though the third arm was the plan all along teaches nothing.

---

## 1. The first comparison was uninterpretable, and three arms were needed

**What.** The first arm compared GLM-5.3-MXFP4 at **TP4, DPA off, MTP off, kvd
off** against a GLM-5.2 baseline at **TP8 with DPA + MTP + kvd all on**. It gave
**0.58–0.83×** and looked like a regression.

**Why it was wrong.** Four axes moved at once. No ratio taken against it can be
attributed to the model.

**How it was fixed.** A matched TP8 arm — same topology, same DP-attention, same
EAGLE MTP 3/1/4, same admission limits — gave **0.89–1.11×**. The TP4 arm was
then kept as the **control that measures what those four confounds were worth**,
not discarded as a failed attempt.

**Context.** The cleanest single line is TPOT at conc 1: **14.43 ms at
TP4/no-MTP → 8.63 ms matched.** That supports the MTP attribution but does not
isolate it, because TP4→TP8 moved with it. **No run in this campaign varies MTP
alone. That hypothesis is open and must not be presented as settled.**

All four differences were read off the engine's own resolved args, never assumed:
`tp_size=4`, `enable_dp_attention=False`, `speculative_algorithm=None`,
`enable_hierarchical_cache=False`, and zero `kvd adapter connected` lines.

---

## 2. Acceptance length: 2.90 vs 3.12, and the 3.60 figure is a p99 artifact

**What.** The baseline's widely-quoted acceptance median is **3.60** over 37,878
lines. Ours is **2.90**. Quoted against each other that reads as a 1.21× decode
deficit.

**Why it is wrong.** The two numbers cover different windows. The baseline's p99
arm alone is **113.4 of its 148.8 measured minutes**, so the whole-sweep figure
is dominated by an arm we did not run.

**How.** Scope both sides to the p50+p90 windows:

| window | n | p10 | median | p90 | at 4.00 |
|---|---:|---:|---:|---:|---:|
| baseline p50+p90 | 7,170 | 2.62 | **3.12** | 3.73 | 1.9 % |
| **ours, matched** | 8,019 | 2.41 | **2.90** | 3.49 | **1.0 %** |
| baseline p99 only | 30,321 | 3.02 | 3.69 | 4.00 | **10.7 %** |

**Parity.** And our at-4.00 rate is half theirs.

**Context.** A median **at** 4.00 is a failure signal, not a win — it means the
draft is predicting a repetition loop perfectly. The baseline's at-4.00 rate
jumping **1.9 % → 10.7 %** between its short and long arms is independent
evidence that its long-output arms degenerate the way ours do. **INFERRED** — the
direct 10-gram check is impossible, that packup saved no `generated_texts`.

---

## 3. Work-per-request: the caveat that reversed

**What.** An earlier draft warned that our conc-24 points were flattered by a
higher cache-hit rate.

**Why it was wrong.** That reasoning came from client-side hit rates on the *TP4*
arm. Measured engine-side on the *matched* arm, `cached/(new+cached)` over
prefill batches:

| arm | baseline | ours |
|---|---:|---:|
| p50 | 0.4434 | **0.3293** |
| p90 | 0.4219 | **0.4682** |

**Our matched p50 did 11 points MORE prefill work.** The ratios are conservative,
not flattered. p90 is matched within 4.6 points.

**Context — the method trap.** The engine log has **no arm delimiters**, so any
window-scoped statistic needs an external boundary. A first pass used a window
that swallowed an unrelated workload's traffic and produced a p90 cached fraction
of **0.80**; the correct value is **0.4682**. The boundary that fixed it came
from the sweep's own per-run output-file mtimes. **Any statistic scoped by a
window you inferred is suspect until you can name what bounded it.**

---

## 4. `--dataset-name random` does not build independent prompts

**What.** Both this recipe and the baseline README claimed the cache-report
column is meaningless on `--dataset-name random` because there is "no shared
prefix by construction". **That is false.**

**Why.** `benchmark/serving.py:1974` seeds `random` from `--seed` (**default
42**) *before* `benchmark/datasets/random.py:112` shuffles the ShareGPT corpus,
then `:116-119` takes the first `num_prompts`. With `--num-prompts = 10 × conc`,
**each arm's prompt list is a strict prefix of the next arm's.** Separately,
`:130-134` reaches a long ISL by **repeating one conversation and truncating**,
so the same index at isl 7400 and isl 15500 shares a literal 7400-token prefix —
arms contaminate each other across ISLs too.

**How it was confirmed.** Predicted hit rates 12.5 / 50.0 / 66.7 % at conc
8/16/24; measured **12.38 / 49.51 / 66.05 %**. And independently: the baseline's
own per-run rates are **9.95 / 12.38 / 49.51 %** at conc 1/8/16 and ours are
**9.95 / 12.38 / 49.51 %** — different model, different cluster, 27 days apart.

**Context.** Two consequences outlive this packup:
- **"Throughput scales N× from conc 1 to conc 24" is not a scaling measurement**
  on either side. Part of that slope is a rising hit rate.
- **Do not fix this by changing `--num-prompts = 10 × conc`.** Matching the
  baseline outranks removing the confound, and both sides are confounded in the
  same direction. The fix for whoever re-runs *both* sides is a different
  `--seed` per arm.

---

## 5. Operational traps that each cost real time

**Weights path.** `/apps/data/models` is a symlink onto a separate NFS mount.
Bind the **realpath**. Binding the symlink's parent gives an empty directory and
fails minutes later as `Unrecognized processing class`.

**`--showmemuse`'s `VRAM%` is not occupancy.** It does not fall when memory is
released and read **76 % on empty cards**. Use `rocm-smi --showmeminfo vram` for
any occupancy decision.

**A 200 is not liveness.** The router answers `/health` and `/v1/models` from its
own registry with a dead engine behind it. A liveness check must make the engine
**generate a token, with a timeout**.

**Per-rank values read as clamps.** At dp8 the engine prints
`max_running_requests=32` and `chunked_prefill_size=8192` while the globals are
256 and 65536 — a division by `dp_size`. The baseline's log shows both values
too. Nothing is being silently capped.

**The mxfp4 silent-dequant guard is mandatory** (`big_smoke.sh` block 6b). With
`--moe-runner-backend triton`, or with aiter failing to bind, the MXFP4
checkpoint is dequantised to BF16 GEMMs: the server starts, answers correctly,
and is several times slower with nothing in any log saying so. Only the
`fused_moe` dispatch line names the packed dtype. The block is a **verdict**, not
an echo — it FAILs when `float4_e2m1fn_x2` is absent.

**`--disable-shared-experts-fusion` on big-mxfp4 is insurance, not a fix.**
`glm4_moe.py:1174`'s fusion gate special-cases only `w4afp8` and would fuse under
`quark`. But this checkpoint's shared experts are themselves MXFP4 (76 `.weight`
/ 75 `.weight_scale`, the gap being the BF16 MTP layer 78, not loaded at MTP=0),
so the mixed-precision precondition is absent. It stays on because upstream
#25261 shows this class failing **silently with wrong output** when shapes line
up. The fp8 arm ran with fusion **enabled** and was fine — that is the control
that makes it a quantization-mismatch story rather than "fusion is broken on
gfx950".

---

## 6. Caveats that travel with every number

- **Untuned MoE kernel.** `no tuned FlyDSL config` × 24 — every figure is a
  floor, not a ceiling.
- **Greedy-verified output on MTP arms.** EAGLE verify on ROCm takes
  `torch.argmax` (`eagle_utils.py:726`, `_is_hip` in an `or` with
  `is_all_greedy`), so `--temperature 1.0 --top-p 0.95` never reach token
  selection there. Three open upstream PRs: #31214 / #32922 / #37134.
- **`e_score_correction_bias` is downcast to bf16** under `aiter`+`quark`,
  collapsing 238 distinct fp32 values to 8. It is expert-*selection* bias, so it
  perturbs routing rather than weights. Upstream #37133. **No accuracy delta has
  been measured** — this affects all MXFP4 numbers on this project.
- **`ttft_p99` is not a latency percentile on these arms.** It measures a
  discrete whole-wave stall — see the PD packup's `notes.md` and
  `workspace/results/ttft_wave_stall.md`. Report `ttft_p50` and that event
  separately.
- **p99 fixlen (isl 23500 / osl 17000) was NOT RUN, by decision** — unmatched on
  four axes at the time, hence uninterpretable regardless of measurement quality,
  and the sibling packup measures 96–100 % of requests looping at that shape.
  Recorded as a scope decision, not an incomplete arm.

---

## 7. Attribution

- **matched TP8 arm** (`results/matched_tp8.csv`) — run by the **team lead**,
  not by this author. Reported here because the alignment claim rests on it.
- **TP4 control** and **MIX TP8 features-off isolator** — run by this author.
- **Smoke logs** — this author.
- No operator has independently certified any of these numbers.

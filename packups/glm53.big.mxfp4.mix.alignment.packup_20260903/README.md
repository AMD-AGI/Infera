# GLM-5.3-MXFP4 (big) — MIX fixed-length alignment against GLM-5.2

**Ran:** 2026-09-02 (all arms), packed 2026-09-03.
**Node:** `smci355-ccs-aus-n01-33`, 8×MI355X (gfx950).
**Status: PASS.** 0.89–1.11× the GLM-5.2-MXFP4 baseline over 8 matched points.

## Goal

The mission requires GLM-5.3 serving performance to be **"roughly aligned with
GLM-5.2"**. This packup establishes that, and — equally important — establishes
*what makes the comparison valid*, because the first two attempts at it produced
numbers that could not be attributed.

**Spec:** the GLM-5.3-series integration task, `mission.md`, and its distilled
form in the repo at `.claude/CLAUDE.md` on branch `yihou.dev.glm53.expr`
(commit `46e79746`). Not copied in — it is a living file that covers the whole
series, not just this experiment.

**Success criterion:** no numeric bar was set. The bar is completeness and
trustworthiness: matched configuration on both sides, every difference measured
rather than assumed, and every caveat that moves a number stated with it.

## Result

Matched: TP8, DP-attention dp8, EAGLE MTP 3/1/4, `--max-running-requests 256`,
prefill delayer, `--enable-cache-report`, `--random-range-ratio 1.0`,
`--temperature 1.0 --top-p 0.95`, `--num-prompts = 10 × conc`, via the router.

| arm | conc | ours tok/s | GLM-5.2 baseline | **ratio** | ours TPOT | base TPOT |
|---|---:|---:|---:|---:|---:|---:|
| p50 | 1 | 76.26 | 82.56 | 0.92 | **8.63** | 9.13 |
| p50 | 8 | 417.63 | 395.58 | **1.06** | **13.12** | 13.55 |
| p50 | 16 | 606.33 | 679.82 | 0.89 | 19.53 | 16.56 |
| p50 | 24 | 825.07 | 746.75 | **1.10** | 22.48 | 21.70 |
| p90 | 1 | 111.07 | 112.18 | 0.99 | 8.64 | 8.14 |
| p90 | 8 | 624.79 | 606.55 | **1.03** | **11.81** | 11.87 |
| p90 | 16 | 995.35 | 1020.36 | 0.98 | 14.96 | 14.03 |
| p90 | 24 | 1346.45 | 1331.73 | **1.01** | 16.63 | 16.52 |

**TTFT is better on our side at 7 of 8 points**, by up to 40 % (p90 conc 8:
2853 ms vs 4523 ms). Do not collapse TTFT and TPOT into one throughput ratio —
they move in opposite directions and a single number hides it.

**Every figure is a floor.** The FP4 MoE kernel logs `no tuned FlyDSL config`
24× and falls back to an untuned 2-stage default.

**Output quality, measured not assumed** (`results/repetition_tp4_control.md`,
n=980): at **osl 320 the output is clean — 0/490 looping**, so the p50 rows above
are throughput of coherent text. At **osl 3300 54.7 % of requests loop**, so the
**p90 rows are an upper bound** — a valid measurement of the engine generating
repetitive text, not a quality result. The baseline's long-output arms could not
be checked the same way (its packup saved no `generated_texts`), so the p90
*ratio* may still be fair; the p90 *absolute* is not a quality claim.

## The three arms, and why three were needed

| arm | who ran it | what it establishes |
|---|---|---|
| **matched TP8** | team lead | the alignment result above |
| **TP4 control** | this author | that the earlier 0.58–0.83 spread was *configuration*, not model |
| **MIX TP8 features-off** | this author | isolates DP-attention/MTP from topology; also the PD reference |

The first attempt compared a TP4, DPA-off, MTP-off, kvd-off arm against a TP8
baseline with all three on, got **0.58–0.83**, and it was uninterpretable. See
`notes.md` §1.

## Read these before quoting any number

Three corrections that each changed a conclusion. Full detail in `notes.md`.

1. **Acceptance is 2.90 vs 3.12 like-for-like**, not 2.90 vs 3.60. The 3.60
   figure is a **p99-arm artifact** — that arm is 113.4 of the baseline's 148.8
   measured minutes. Quoting it reads as a 1.21× decode deficit nobody measured.
2. **Our matched p50 did 11 points MORE prefill work than the baseline**
   (cached fraction 0.3293 vs 0.4434), so the ratios above are **conservative**.
   An earlier version of this claim pointed the other way and was wrong.
3. **`--dataset-name random` does not build independent prompts.** Higher
   concurrency mechanically buys a higher cache-hit rate, so the conc-1→24 slope
   is partly that, not scaling. Predicted 12.5/50.0/66.7 %, measured
   12.38/49.51/66.05 %, and **the baseline's own hit rates are identical to ours
   to two decimals**.

## Folder map

- `REPRODUCE.md` — ordered, copy-pasteable reproduction
- `environment.md` — exact HW/SW, image digest, git SHA
- `scripts/` — the four scripts that ran, verbatim
- `results/` — the CSVs and the two smoke logs
- `notes.md` — the wrong turns, the corrections, and the durable traps

## Evidence is carried here, not pointed at

**This packup deliberately carries heavy evidence** (18 MB). A packup that points
at a scratch directory is a packup whose evidence nobody has promised to keep.

| `logs/` | compressed | what |
|---|---:|---|
| `tp4_control_jsonl.tar.gz` | 16 M | TP4 control per-request JSONL — `ttfts[]`, `cached_tokens[]`, `generated_texts`. The raw basis for the cache-hit fraction, the acceptance distributions and the wave-stall analysis |
| `mix8_isolator_artifacts.tar.gz` | 1.6 M | isolator JSONL + engine log |
| `console_logs.tar.gz` | 40 K | every bring-up, smoke and sweep console log |

```bash
tar xzf logs/tp4_control_jsonl.tar.gz -C <dest>
```

Originals remain untouched at
`/apps/yihou/glm53.series.workspace_20260901/bigmodel/`.

## Cross-cutting analyses — `analysis/`

Copied in rather than referenced, because a path into scratch dangles the moment
that directory is cleaned. Both are duplicated in the PD packup; a packup must be
readable alone.

- `ttft_wave_stall.md` — `ttft_p99` on these arms is **not a latency percentile**.
  It measures a discrete whole-wave stall. Includes the 2026-09-03 update
  identifying **decode-side DP-attention** as the axis for the conc-1 tail
  (211 ms p99 with it off, 10,197 ms with it on; hip moved it 0.6 %) —
  **and the unresolved tension**: the MIX arm showing the same signature had DPA
  *off*, which argues against DPA being the whole story.
- `METHOD_the_check_that_lies.md` — the eleven-instance table of observables that
  return the same value regardless of the world. Several instances in this
  packup's `notes.md` are entries in it.

## Related

`../glm53.big.mxfp4.pd.packup_20260903/` — the PD-disaggregation work. It uses
this packup's `mix_tp8_featoff_isolator.csv` as its aggregated reference.

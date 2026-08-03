# Analysis — headline and verdict against the mission

Four files:

| file | what |
|---|---|
| [`sli_percentiles.md`](sli_percentiles.md) | every SLI ladder, recomputed from 2,811 raw samples; SLA verdict |
| [`yaml_vs_measured.md`](yaml_vs_measured.md) | every workload knob vs. its measurement, with the conversion rule |
| [`mtp_comparison.md`](mtp_comparison.md) | MTP on vs off, and what is / is not attributable |
| this file | the coherence check: does the whole picture hang together? |

---

## The mission's question, answered

> Run Case A against kvaware + kvd + MTP + PD + DPA, all on at once, and report it.

**Done.** All five features proven on by a check that would go red if absent
(table in `../README.md`), and the workload ran its full 4,007 s window with
**zero engine faults on either leg**.

## Against the Case A specification

| spec target | measured | verdict |
|---|---|---|
| input p50 / p90 / p99 = 74K / 155K / 235K | 73.6K / 152.3K / 225.7K | **PASS** (≤4 %) |
| output p50 / p90 / p99 = 320 / 3.3K / 17K | 299 / 2,791 / 9,688 | p50/p90 **PASS**; p99 **MISS −43 %** |
| cache hit 88–90 % | **88.82 %** | **PASS** |
| turns p50 / p90 / p99 = 3 / 20 / 103 | not emitted; 79 retirements, lifetimes ≤3,575 s | **window-censored** |
| inter-turn delay p50 / p90 / p99 = 4 / 31 / 240 s | 4.2 / 27.3 / 158.7 | p50/p90 **PASS**; p99 censored |
| acceptance 56 % @ 5 draft tokens | **68.4 % @ 4** (acc len 2.736) | **EXCEEDS** |
| `success_rate ≥ 0.97` | **0.9757** | **PASS** |
| `ttft_p90_ms ≤ 30,000` | **18,877** (sustain) | **PASS**, 1.59× |
| `e2e_p50_ms ≤ 4,500` | not measured; back-solves to ≈12.0 s | **MISS ~2.7×**, ungated |

## Does it cohere? Four independent cross-checks

A benchmark can pass every threshold and still be measuring the wrong thing.
These are the checks that would catch that:

**1. The load was the configured load, not backpressure.**
In-flight peaked at **44/48** and never pinned; sessions peaked at 54/128. Had
in-flight pinned, the percentiles would describe the cap rather than the
deployment — which is exactly how the earlier spur attempt lost its window.

**2. Two independent paths agree on TTFT.**
Recomputed from raw samples: p50/p90/p99 = 6,659 / 19,238 / 37,736 ms.
`summary.json`: 6,658.5 / 19,238.0 / 37,742.0. Agreement to 0.02 %.

**3. The MTP speedup is arithmetically consistent with its own acceptance.**
Engine acceptance 2.736 → theoretical ceiling 2.74×; realized TPOT ratio 1.74×,
i.e. 64 % of ceiling, the remainder being draft-forward + verify cost. A ratio
*above* acceptance would indicate a measurement error; a ratio near 1.0 would
mean speculation running but not accepted. Neither is the case. The vultr
sibling independently measured 2.11× at acceptance 2.80.

**4. Cache hit is not merely non-zero — it matches its own ideal.**
88.82 % actual against 88.99 % ideal = **99.81 % efficiency**, 0.19 % eviction.
The sustain distribution is flat from p5 to max (88.9–89.0 %). This is what a
correctly-nesting prefix looks like; a broken radix path shows a wide left tail.

## The two imperfections, and their single shared cause

Both remaining misses trace to **one hardcoded client constant**, not to the
server:

`aiohttp.ClientTimeout(total=240)` at `agent_throughput.py:929`.

- **All 39 errors are it.** The prefill leg returned **2,904/2,904 HTTP 200**;
  there are zero server-returned failures and zero engine faults.
- **The `output_tokens.p99` shortfall is it.** At TPOT p50 17.9 ms a
  17,000-token generation needs **304 s of decode alone**, before TTFT. The
  profile asks for a tail the client cannot wait for. The largest generation that
  *did* complete was 20,434 tokens (366 s) — it survived only by running below
  p50 TPOT.

A profile specifying a 17K-token p99 alongside a 240 s client timeout is
internally inconsistent. Raising the constant is the fix; nothing about the
deployment changes.

## What this run does NOT establish

Stated plainly rather than buried:

1. **No MTP-off arm on this image.** The 1.74× is a cross-run comparison whose
   reference also differs in image, `--context-length` and prefill GMU. The
   acceptance arithmetic is what makes it credible. The clean ablation costs one
   67-minute window and is the highest-value follow-up.
2. **kvd is proven correct, not exercised.** 27,099 sets against **0 gets**
   during the run. Every request nested in a prefix the in-GPU radix cache
   already held, so L3 was written and never read. 0 misses ever — correct, but
   this workload does not stress tiering. The restart-and-replay proof lives in
   the Phase-1 kit.
3. **TTFT regressed vs the MTP-off reference and MTP is not why.** MTP is
   decode-only. The causes are ~2× the concurrency and the −10 % KV pool from
   GMU 0.80. **Both runs used `--context-length 262144`** and saw the same input
   distribution — an earlier draft claimed the reference clamped ~16 % of its
   inputs at 131,072, which is retracted. The regression is therefore explained
   in direction but not fully quantified.
4. **The `accept len: 4.00` tail (3.0 % of batches) was not chased down.**
   Against a 2.736 mean, 0 retractions and a 97.6 % success rate, it reads as
   small-batch saturation rather than a repetition loop — but that is an
   inference, not a measurement.
5. **Turn-count and inter-turn-delay p99 are window-censored.** A 103-turn
   session at a 14.1 s mean inter-arrival cannot complete inside 3,600 s. Not a
   deployment property.

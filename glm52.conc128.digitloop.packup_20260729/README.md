# GLM-5.2-MXFP4 — conc=128 "digit loop" on DPA + mooncake PD — **REPRODUCED**

**Ran:** 2026-07-29 · **Nodes:** prefill `chi2867` (10.2.122.44) + decode `chi2879` (10.2.122.10),
8× MI355X gfx950 · **Engine:** sglang 0.5.15.post1 ·
**Image:** `infera/engine-sglang:pd-unified-waitevent` (854ebf70 mooncake wait_event fix **already in**)
**Model:** `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`

User report: *under conc=128 stress on the DPA + mooncake-PD path, some requests' outputs
degenerate into a "digit loop".* Reproduced, and localised to a **stop/EOS failure**, not KV
corruption.

## Headline

| topology | DPA | arm | conc | n | duration | CLEAN | DIGIT_LOOP | CORRUPT | TAIL_REP | TRUNC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PD mooncake | 1 | baseline | 1 | 32 | 109.5 s | **32** | 0 | 0 | 0 | 0 |
| PD mooncake | 1 | replay of A's 6 failures, ×4 | 1 | 24 | 69.1 s | **24** | 0 | 0 | 0 | 0 |
| **PD mooncake** | 1 | **stress A** | **128** | 512 | 34.4 s | 506 | **4** | **1** | 1 | 0 |
| **PD mooncake** | 1 | **stress B** (fresh salt) | **128** | 512 | 44.7 s | 502 | **4** | **3** | 2 | 1 |
| single-node mix | 0 | baseline | 1 | 32 | 83.5 s | **32** | 0 | 0 | 0 | 0 |
| **single-node mix** | 0 | **stress A** | **128** | 512 | 60.7 s | **512** | **0** | **0** | **0** | 0 |
| **single-node mix** | 0 | **stress B** (fresh salt) | **128** | 512 | 63.9 s | **511** | **0** | **0** | 0 | 1\* |
| **single-node mix** | **1** | **stress A** | **128** | 512 | 76.4 s | **512** | **0** | **0** | **0** | 0 |
| **single-node mix** | **1** | **stress B** (fresh salt) | **128** | 512 | 86.3 s | **511** | **0** | **0** | 0 | 1\* |

> **PD @ conc=128: ~1.2–2 % of requests degenerate. Same prompts at conc=1: 0 %.
> Same prompts on single-node at conc=128, with DPA both OFF and ON: 0 % across 2048 requests.**

\* every single-node `finish=length` is `idx=422`, `TRUNCATED` — the same idx also went TRUNCATED
on PD run B. That prompt genuinely reasons past 1024 tokens: no loop, no repeated digits. A probe
artifact that shows up in every arm, not the bug.

Workload: ISL=1024 / OSL=1024, temp=0, needle-in-log prompt with a checkable 5-digit answer, sent
through the router. This is exp07's shape — exp07 reported *512/512 PASS* at conc=128, because it
only measured completion. **A digit loop completes normally**; it is invisible to `bench_serving`.

## It is NOT the KV race, and NOT prompt-dependent

- The 854ebf70 wait_event fix is present in this image
  (`disaggregation/mooncake/conn.py:1233 wait_event.synchronize()`), and ISL=1024 is a
  **single prefill chunk** — the multi-chunk path that bug lives on is never entered.
- **Replaying the 6 failing prompts at conc=1 gives 24/24 CLEAN.** Prompt content is a pure
  function of `idx`+`salt`, so those were byte-identical to the ones that failed under load.
- Run B used a fresh salt (every prompt novel, no prefix-cache reuse) and failed at the same rate.

## The signature — unusually clean

**Every bad output, and only the bad outputs, has `finish_reason=length`** (completion_tokens =
1024, the cap). All 1008 good outputs are `finish=stop`, median **155** tokens, `</think>` ≤ 1.

**All bad outputs contain the correct needle** (15/16). The model reasons correctly, retrieves the
right number, emits a coherent first paragraph — and then **fails to stop**, spinning on `</think>`
and the answer digits until it burns the token cap. `</think>` repeats **27–839×** per bad output.

```
idx=84  'The user is asking for the calibration constant ... "SECRET-84" ... is exactly 45203.
         The other records are maintenance logs for corridors, unrelated to the gyroscope.</think>
         The log co...'   -> tail: '333333333333333333333333333333333333333333333333333...'

idx=121 'Looking through the records, I see Record SECRET-121: "the calibration constant for the
         orbital gyroscope is exactly 68209." ...'
         -> tail: '...68209682096820968209682096820968209</think>68209682096820968209...'
```

**KV is correct. The stop/EOS decision is what fails.** This is exactly the `TAIL_REPEAT` mode the
long-context packup (`glm52.longctx.packup_20260729`) observed *after* the KV fix and explicitly
did not chase — here it is, amplified by concurrency, and running long enough that the loop drifts
into digit salad, i.e. the user's "数字循环".

## Where the failures land — it tracks the concurrent decode batch

| run | bad idx | in the first 128 |
|---|---|---|
| A | 71, 84, 101, 112, 119, 121 | **6/6** |
| B | 64, 67, 76, 100, 104, 106, 116, 120, 124, **422** | **9/10** |

15/16 land in `idx < 128` — the only window where all 128 requests are genuinely in flight
together (after that, clients refill one at a time and the effective batch is decode-limited).
The trigger scales with the **concurrent decode batch size**, not with prompt content or position
in the stream. Note `--cuda-graph-max-bs 128` — batches above the capture limit replay eager,
which is the obvious first suspect and is the top item in "next steps".

## Config (exp07 1k/1k capacity, patched image)

| knob | prefill | decode |
|---|---|---|
| context-length | 32768 | 32768 |
| chunked-prefill-size | 65536 (→ 8192/rank under DPA) | same |
| max-running-requests | 2048 (→ 256/rank) | same |
| cuda-graph-max-bs | 128 | 128 |
| mem-fraction-static | 0.88 | 0.85 |
| DP | `--dp-size 8 --enable-dp-attention --ep-size 8` (symmetric) | same |
| transport | mooncake RDMA, 8× ionic, dmabuf disabled | same |

## Verdict classifier (and the false positive we fixed)

`scripts/stress_capture.py` captures **every** output and classifies:
`DIGIT_LOOP` / `CORRUPT_REASONING` / `TAIL_REPEAT` / `WRONG` / `TRUNCATED` / `CLEAN`.

First pass flagged `idx=460` as DIGIT_LOOP — a legitimate 1484-char chain-of-thought that quoted
its answer 12×. Rule (c) now also requires the repeated literal to occupy **>25 % of the output**.
A 7-case regression suite passes; the fix was applied retroactively to the stored results as
`verdict_v2` (the headline table is post-fix). If you extend the detector, re-run that suite —
distinguishing a digit loop from a legitimately numeric answer is the whole difficulty here.

## Control arms — single-node, DPA off **and** on: does NOT reproduce

Same patched image, same container, same GLM DSA env, same TP8 / ctx 32768 / **per-rank** chunk
8192 (= the PD run's 65536 ÷ dp8, so the compute shape matches) / `cuda-graph-max-bs 128` /
max-running 2048 / gmu 0.88. Client hits the server directly (no router — it is not a PD leg).
Same prompts, same salts, same classifier.

Two arms, run as a one-variable split:

| arm | removed vs the failing PD config |
|---|---|
| `up_single_nodpa.sh` | disaggregation **and** DP-attention |
| `single_dpa1.sh` | disaggregation only (DPA=1, `SGLANG_DP_USE_GATHERV=1`, dp8+ep8) |

**2048 single-node requests at conc=128 → 0 digit loops, in both arms.**
**1024 PD requests, same prompts, same conc → 16 `finish=length`, 15 of them degenerate loops.**

Consequences:

- **The trigger requires PD disaggregation.** Plain GLM-5.2 decode at batch 128 is fine, with or
  without DP-attention.
- **`cuda-graph-max-bs=128` is exonerated** — identical on every arm, and all single-node arms are
  clean. (It was the leading suspect after round 1.)
- **DP-attention alone is exonerated.** The DPA=1 single-node arm is as clean as DPA=0.

## Next steps (cheapest-first)

1. **PD with DPA=0 at conc=128** — the remaining split: is plain PD at high concurrency enough, or
   does it need PD × DPA together?
2. If PD+DPA=0 also reproduces, the suspect narrows to the **decode leg's sampling/stop path for
   transferred requests**. `</think>` repeating 27–839× while the reasoning and the retrieved
   needle are correct points at the **EOS/stop check**, not at the logits.
3. Orthogonal, cheap: re-run the PD arm with a larger `--cuda-graph-max-bs` to see whether the
   rate moves at all — now a secondary question rather than the main hypothesis.

## Folder map

- `REPRODUCE.md` — exact steps
- `notes.md` — working process, hypotheses, the two infra gotchas that cost a relaunch
- `scripts/up_conc128.sh` — relaunch both PD legs at exp07 1k/1k capacity on the patched image
- `scripts/up_single_nodpa.sh` — control arm: single-node colocated, DPA=0, no disaggregation
- `scripts/single_dpa1.sh` — control arm: single-node colocated, **DPA=1**, no disaggregation
- `scripts/stress_capture.py` — the capture+classify client (supports `IDX=`/`REP=` replay)
- `scripts/run_arms.sh` — arm driver
- `results/cap_{base,c128,c128b,replay1}.json` — PD arms, **every** output, with `verdict_v2`
- `results/sn_{base,c128,c128b}.json` — single-node DPA=0 control arms
- `results/sd_{c128,c128b}.json` — single-node DPA=1 control arms · `summary.csv` — all 9 arms
- `logs/cap_*.log`, `logs/sn_*.log`, `logs/sd_*.log` — client logs
- `logs/pd_{prefill,decode}_*_c128.trimmed.log`, `logs/single_{nodpa,dpa1}_30000.trimmed.log` — servers

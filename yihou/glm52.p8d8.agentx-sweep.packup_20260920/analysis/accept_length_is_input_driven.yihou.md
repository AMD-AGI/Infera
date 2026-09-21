# Why acceptance read 4.69 here and ~2.6 before — it is the input

Measured 2026-09-18 13:20-13:35 UTC on the live P8D8 probe deployment
(137 prefill / 136 decode, image `v0519-yihou-0917-nextnfix-hicache`,
simulation OFF, custom all-reduce ON).

## The question

`sglang:spec_accept_length` read **4.31-4.88, mean 4.69** on a 16-request
`temperature=0` probe. But real-acceptance AgentX runs measured **~2.57**
(`agentx.c72.v0519.packup_20260918`, same P8D8 shape) and **2.59 mean**
(`glm52-agentx-c40-realacc.packup_20260918`, another session). Natural variation
driven by the input, or a configuration difference?

## The experiment

Single variable. **Same service, same engine process, same configuration, same
`max_tokens=512`, same 8-way concurrency — only the prompt class changes.** Ordered
most-predictable first. Script: `scripts/acc_by_input.yihou.sh`; raw data on
`crsuse2-m2m-137` at `/mnt/m2m_nobackup/yihou_p8p4/acc_by_input/`.

| class | what the prompt asks for | mean accept_length | min | max | accept_rate |
|---|---|---|---|---|---|
| `memorized` | the first 10 primes | **4.456** | 3.525 | 5.725 | 0.691 |
| `codeedit` | add a try/except to a 2-line function | 3.569 | 3.025 | 3.850 | 0.514 |
| `adversarial` | 40 random 6-char tokens | 3.263 | 2.600 | 3.950 | 0.453 |
| `boilerplate` | memoized `fibonacci` with type hints | 3.188 | 2.850 | 4.200 | 0.438 |
| `freeform` | invent an original metaphor, no cliché | **2.506** | 2.125 | 2.800 | 0.301 |

## Conclusion — the workload explains it; configuration need not be invoked

**Input alone moves acceptance from 2.51 to 4.46 — a 1.78x spread — with the
configuration held exactly constant.** The whole disputed gap fits inside the range a
single unchanged deployment produces when you only change what you ask it.

And the endpoints land where the mechanism predicts:

- The **4.69 that prompted the question came from the easiest possible case**: "list
  the first 10 prime numbers" is a memorised sequence, so a one-layer EAGLE draft
  predicts nearly every token. That number was never representative of anything.
- The **`freeform` class lands at 2.506**, which sits inside the historical AgentX
  band of **2.57-2.59 mean (2.15-3.08 per rank)**. Agentic Claude-Code traces are novel prose, novel code and
  novel tool arguments — much closer to `freeform` than to a memorised list.

So the earlier AgentX figures were not depressed by a misconfiguration, and today's
4.69 is not evidence of an improvement. **Both are correct measurements of different
workloads.**

## Two things this does NOT establish

- **It does not prove configuration is identical** across the three runs. That axis
  was checked separately (`accept_length_config_diff.yihou.md`); this experiment only
  shows configuration is **not needed** to explain the gap, which is a weaker and more
  honest claim than "configuration is the same".
- **It does not make 2.51 the predicted AgentX number for this stack.** The classes
  here are short single-turn prompts at 8-way concurrency; AgentX is multi-turn with
  ISL into the hundreds of thousands and a much larger decode batch. The right value
  for this stack under AgentX is whatever the AgentX run measures.

## The practical consequence, which is the part that matters

The plan was to run the two screening points with `DECODE_SIMULATE_ACC_LEN=3.61`.
Against a realistic agentic workload the real acceptance on this stack is **~2.5-2.6**,
not 4.69 — so **simulation at 3.61 does not penalise this stack, it flatters it**, by
roughly 1.4x on committed tokens per verify step.

This inverts the reading I gave earlier off the 4.69 probe, and the correction matters
because it changes which option is the conservative one:

- Simulation ON is the **optimistic** setting, and it is what the reference c32/c40
  numbers used — so it remains the like-for-like choice against them.
- Simulation OFF measures what the stack actually delivers.

Neither is wrong; they answer different questions. What would be wrong is quoting a
simulated number as though it were measured, or — as I nearly did — recommending a
switch on the strength of a probe whose prompt was a memorised list.

## Method notes worth keeping

- `spec_accept_length` is a **gauge holding a recent mean**, not a lifetime average.
  It drifts as traffic changes and an **idle rank keeps reporting its last value**
  rather than decaying to zero. Read it under the load you mean to characterise, and
  correlate with `spec_verify_calls_total` to know whether a sample is fresh.
  `scripts/mtp_sampler.yihou.sh` logs a time series for exactly this reason.
- Output length matters too: the same prompt at `max_tokens=256` (all reasoning, no
  final answer) gave ~4.05, and at 2048 (reasoning plus answer) gave ~4.69.

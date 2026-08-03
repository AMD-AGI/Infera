# The needle "failures" are sampling excursions, not KV corruption

**Cost:** two probe rounds (~12 min of GPU), no engine restart. Recorded because
the failure shape is *exactly* the documented mooncake early-send signature, so
a future reader will suspect the same thing and should not have to re-derive
this.

## What was seen

First correctness run, MTP on, GLM-5.2's own sampling (temp 1.0 / top_p 0.95):

| depth | result | finish | cached | compl | `</think>` | tail |
|---|---|---|---|---|---|---|
| 5 % | **FAIL** | `length` | None | 2048 | **1118** | `6159</think>61</think>…` (want `6159362`) |
| 25 % | OK | stop | 5,952 | 184 | 1 | `3331179` |
| 50 % | OK | stop | 29,952 | 141 | 1 | `5271814` |
| 75 % | OK | stop | 59,968 | 270 | 1 | `8251068` |
| 95 % | OK | stop | 89,984 | 220 | 1 | `5385227` |

The failure emitted `6159` — the **first four digits of the wanted 7** — then
looped `</think>`. That is the mooncake early-send fingerprint verbatim: partial
digits, then a `</think>` storm, at an early depth, on the one fully-cold
request. The patch README's own example is
`want=2183762 got='2183</think>2183</think>218…'`.

## Why the first run could not settle it

`cached_tokens` climbed monotonically with depth — None, 5,952, 29,952, 59,968,
89,984 — because each probe reused the previous one's prefix. Depth 5 % was the
**only** fully cold prefill. So "early depth" and "cold cache" were perfectly
confounded, and either could explain the result.

## Round 1 — the 2×2 (`needle_2x2.py`)

Cross cold/warm with early/mid depth. It refuted **both** simple stories:

| cell | result | cached | finish | `</think>` |
|---|---|---|---|---|
| SAME filler, d=0.05 | **OK** | 120,000 | stop | 1 |
| SAME filler, d=0.50 | **FAIL** | 120,000 | `length` | 0 (filler regurgitation) |
| FRESH filler, d=0.05 | **FAIL** | 0 | `length` | 873 |
| FRESH filler, d=0.50 | **OK** | 5,952 | stop | 1 |

- The **same prompt flipped in both directions** across runs (d=0.05 FAIL→OK,
  d=0.50 OK→FAIL).
- Cache warmth does not predict the outcome: a **warmer** request
  (`cached=120000`) failed while a **colder** one (`cached=5952`) passed.
- FRESH d=0.05 emitted `447190` — **six of seven digits** of `4471903` — before
  looping. Retrieval was working; generation was not.

Every failure shares one property: `finish=length`, i.e. it ran to the 2048 cap.

## Round 2 — determinism and filler (`needle_determinism.py`)

**Arm A: the identical prompt, six times, against a fully warm cache.**

| # | result | finish | cached | compl | `</think>` |
|---|---|---|---|---|---|
| 0 | FAIL | `length` | 120,000 | 2048 | 0 |
| 1 | **OK** | stop | 120,000 | 90 | 1 |
| 2 | **OK** | stop | 120,000 | 85 | 1 |
| 3 | FAIL | `length` | 120,000 | 2048 | 0 |
| 4 | FAIL | `length` | 120,000 | 2048 | 0 |
| 5 | **OK** | stop | 120,000 | 322 | 1 |

**3/6, with prompt and cache state both fixed.** This is the decisive result.
KV corruption is a deterministic function of (prompt, KV state): corrupt pages
produce the same wrong answer every time. A 50/50 split across six identical
requests cannot come from corrupted memory — it can only come from sampling.
At temperature 1.0 a reasoning model can walk into a repetition attractor, and
once in it the model simply never emits a stop token and runs to the cap.

**Arm B: same length and depth, 2,900-word vocabulary instead of 14 words** —
2/3, with the one failure again at d=0.05, `finish=length`, 673 `</think>`.
Widening the vocabulary reduces but does not eliminate it, so the repetitive
filler is an aggravator rather than the sole cause.

## Corroboration from the engine, at zero cost

`accept len` on the decode leg is **bimodal**, and the two modes line up with
the two outcomes:

- healthy requests: **1.75 – 2.80**
- looping requests: **3.85 – 4.00** (the maximum, `--speculative-num-draft-tokens 4`)

`accept len: 4.00` means the draft model predicted every token correctly — which
is exactly what happens when the output is a loop. It is a **symptom of the
loop, not evidence MTP is healthy**, and it is the same trap the merged-branch
kit documented.

Meanwhile, across every one of these rounds: `Memory access fault` **0**,
`Scheduler hit an exception` **0**, `Traceback` **0** on both legs.

## Why this cannot affect any benchmark number

The load generator does not exercise this failure mode at all:

- Its filler is **uniformly random ASCII** — `_ASCII_CHARS = ascii_letters +
  digits + " "*10 + ".,!?-"*2`, sampled i.i.d. per character
  (`agent_throughput.py:60`, `make_filler_seeded`). It is not repetitive
  natural-language text, so there is no repetition attractor to fall into.
- It sends **`ignore_eos: True`** (`agent_throughput.py:813,910`), so *every*
  request generates exactly `max_tokens` **by construction**. "Ran to the cap"
  is the normal, intended behaviour there, not a failure.
- It **grades nothing**. Correctness is answered by `correctness.py`, never by
  the bench.

The same applies to `bench_serving --dataset-name random`.

So the needle behaviour is a property of *this probe's* repetitive filler under
non-greedy sampling, and it is out of the path of every number this experiment
reports.

## Verdict

**Not a defect in the deployment.** Retrieval is correct — the needle is found
and emitted; the failures are degenerate *generation* after retrieval, they are
stochastic under a fixed prompt and fixed cache, and they are absent from the
benchmark's own prompt distribution.

Per the task spec ("do not spend much time on the needle test"), this is closed
here rather than pursued further. Two things are **not** claimed: that the
underlying tendency to loop is fully characterised, or that a mooncake defect is
impossible in some other regime. What is established is that these particular
failures are not one.

**Provenance of the three tables above.** These probes were run interactively and
their stdout was **not redirected to a file**, so the tables are transcribed from
the run rather than backed by a log in `logs/`. Recorded as a gap rather than
papered over. All three are cheap to re-run and self-contained:

    docker exec agbench_mtp python3 .../correctness.py         http://$PIP:8190
    docker exec agbench_mtp python3 .../needle_2x2.py          http://$PIP:8190
    docker exec agbench_mtp python3 .../needle_determinism.py  http://$PIP:8190

The **engine-side** corroboration *is* backed by a log: the bimodal `accept len`
distribution is in `logs/g1_decode.log.gz`, and the zero fault/traceback counts
are in both leg logs.

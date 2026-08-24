# r03 — #33970: the stock arm reproduces the defect, on the chunk boundary

2-node 1P1D on `crsuse2-m2m-237` (prefill, TP8) + `crsuse2-m2m-106` (decode, TP8 +
DP-attention + MTP), GLM-5.2-FP8, mooncake over `mlx5_0` (mode B), router
round-robin. Probe: `scripts/needle.py`, 200 000-token prompt, 7 depths.

## Stock arm result — POSITIVE CONTROL ESTABLISHED

```
prompt_tokens=200000  chunk_size=131072  -> ~2 chunks

 depth  chunk  final   ok  want / got
  0.10      0  False False  9209400 / '910.</think>9</think>9</think>9</think>...'
  0.25      0  False False  8315778 / '831</think>831</think>8</think>8</think>...'
  0.40      0  False False  9991752 / ''
  0.55      0  False True   3488397 / '3488397'
  0.70      1   True True   6301579 / '6301579'
  0.85      1   True True   7868992 / '7868992'
  0.95      1   True True   3924757 / '3924757'

SCORE: 4/7
  non-final-chunk needles: 1/4
  final-chunk needles    : 3/3
```

**The failures are not scattered — they partition exactly on the chunk boundary.**
Every failure is in chunk 0 (non-final); every needle in chunk 1 (final) is
retrieved correctly. That is the signature the PR predicts and it is the reason
this configuration is a valid positive control:

> non-final chunks are handed to the mooncake transfer worker, which reads device
> memory outside the CUDA stream, while the forward that writes those pages may
> still be running. The final chunk is always correct because it goes through the
> sampling path, which has a real `copy_done.synchronize()`.

The **shape** of the failures corroborates it further. These are not plausible
wrong answers — `'910.</think>9</think>9</think>...'` is degenerate repetition, and
one depth returned empty. That is a model decoding from half-written KV, not a
model that read the haystack and answered incorrectly.

Note depth 0.55 lands in chunk 0 and still succeeded. Corruption is a race, so it
is expected to be probabilistic rather than total — 1/4 surviving is consistent
with a race window, and is why the probe reports a per-chunk score rather than a
single pass/fail.

## Controls that make this readable

- **Baseline health.** Before the probe, a short prompt through the same router
  answered correctly (17+25 -> 42). So the deployment is not simply broken; only
  long, multi-chunk prompts fail.
- **Built-in final-chunk control.** 3/3 on final-chunk needles, in the same run,
  same model, same transport. Whatever breaks chunk 0 does not break chunk 1.
- **Single-chunk control.** A 20 000-token prompt (1 chunk, therefore final)
  retrieved its needle correctly: `SCORE: 1/1`.
- **Arm guard.** `arm.sh` counted `wait_event` = 0 in `mooncake/conn.py` on **both**
  legs before launch, and refuses to proceed on a mismatch. This run is stock.

## Two probe corrections this round (both would have corrupted the result)

**1. Resolved chunk size, not the launch argument** (trap 7). `needle.py` now takes
`--resolved-chunk-from <leg log>` and parses `chunked_prefill_size=` off the leg's
own `server_args` line. Measured here: prefill (DPA=0) resolves 131072 as passed,
but decode (DPA=1, dp8) resolves **16384**. Using the command-line value would
mislabel which needles are non-final — and which needles are non-final *is* the
control.

**2. Score `reasoning_content`, not just `content`.** GLM-5.2 is a reasoning model:
with thinking on, the answer appears in `reasoning_content` while `content` stays
empty until it finishes. Scoring `content` alone reported misses for needles the
model plainly retrieved. That failure mode is **arm-independent** — it would
depress stock and patched alike and make the A/B meaningless. The filler carries
no digits, so matching the secret anywhere in the response is unambiguous.

## Rig defects found and fixed on the way here

Recorded because each cost a bring-up and none is a defect in the PR:

- **Orphaned engine workers held the GPUs.** `pgrep -f infera.engine.sglang` matches
  the launcher, but it execs `sglang.launch_server`, which forks one TP worker per
  GPU — and those hold the VRAM. Killing only the launcher left three orphans at
  ~224 GB each, so the next legs died with `HIP out of memory ... 96.00 MiB is free`
  minutes into the run, nowhere near the real cause. `relegs.sh` now kills every
  `python3` in the container and waits for the KFD process table to empty.
- **A health check answered off the dying engine.** The first restart reported
  "serving after 10s" for something whose cold start is ~4 minutes: the old engine
  was still answering `/health`. `relegs.sh` now waits for `/health` to STOP
  answering before launching. Same class of stale-state trap `common.sh` warns
  about for log greps.
- **The kit cannot start a round-robin router.** `common.sh:91` passes
  `--router-tokenizer-path` only on the kv-aware branch, but
  `infera/server/args.py:140` declares it `required=True` unconditionally. Worked
  around with `scripts/router.sh`; reported, not fixed here.
- **Wrong GLM-5.2 copy.** `mlx-community__GLM-5.2-mxfp4` is an MLX-format
  conversion whose `quantization_config` has `mode` but no `quant_method`, so
  sglang fails with `Unknown quantization method: ` while listing `mxfp4` as valid.
  `zai-org__GLM-5.2-FP8` is the loadable one.

## Patched arm result — the defect is gone

Same probe, same 200 000-token prompt, same seeds (so the same seven secrets),
same transport, same two nodes. The only variable is the three files.

```
 depth  chunk  final   ok  want / got
  0.10      0  False True   9209400 / '9209400'
  0.25      0  False True   8315778 / '8315778'
  0.40      0  False True   9991752 / '9991752'
  0.55      0  False True   3488397 / '3488397'
  0.70      1   True True   6301579 / '6301579'
  0.85      1   True True   7868992 / '7868992'
  0.95      1   True True   3924757 / '3924757'

SCORE: 7/7
  non-final-chunk needles: 4/4
  final-chunk needles    : 3/3
```

## The A/B

| arm | non-final chunk | final chunk (control) | total |
|---|---|---|---|
| **stock** | **1/4** | 3/3 | 4/7 |
| **patched** | **4/4** | 3/3 | **7/7** |

The three needles stock lost — `9209400`, `8315778`, `9991752` — are retrieved
exactly on the patched arm. The final-chunk control is 3/3 on **both** arms, which
is what makes this readable: the patch changed outcomes precisely where the defect
is claimed to live, and nowhere else. A patch that simply made the deployment
"better" would have moved the control too.

## Independent corroboration of the failure signature

infera's own patch header (`deploy/docker/patches/sglang_disagg/
patch_mooncake_early_send_wait_event.py`) records the fault it was written for:

```
want=2183762  got='2183</think>2183</think>218</think>218</think> the'
```

Measured here, on different hardware, a different cluster and a different
checkpoint, before reading that file:

```
want=8315778  got='831</think>831</think>8</think>8</think>8</think>...'
```

Same shape: the leading digits of the needle, then degenerate `</think>`
repetition. Two independent reproductions of the same signature.

## Barrier semantics, checked separately

`validate_B.py` on the patched tree: **RESULT: PASS** — `TransferKVChunk` carries
the event and defaults to None, `send()` forwards it on both the last-chunk and
non-last-chunk arms and clears it afterwards, the wait actually blocks, and a read
issued after the barrier observes the writes it was recorded on.

That script states in its own header that two things were **not** covered because
the previous session had only one node: *"that the corruption is actually gone
end-to-end, and the synchronize()'s cost on prefill throughput"*. This round closes
the first; the cost is measured below.

## What this configuration does and does not establish

**Does.** The corruption is real on a 2-node PD pair over cross-node RDMA, it lands
on the chunk boundary the PR predicts, and the patch removes it. This is the
configuration the PR actually targets — not a single-node loopback stand-in.

**Does not.** It says nothing about mooncake over the ionic rails (the container's
libibverbs rejects that driver ABI, see `mooncake_mvp.md`), and the numbers are
taken on a 200 Gb/s `mlx5_0` link, not the 8x400 Gb/s rails. Since a slower link
*widens* the race window, this is the conservative direction for reproducing the
defect, but it does mean the throughput figures are link-specific.

`plan.md` step 1e told the next session to write a scope disclaimer into the PR
about single-node loopback being unable to establish cross-node behaviour. **That
disclaimer is no longer needed** — drop it rather than carry it over.

## The `synchronize()` cost — `plan.md` step 1d

The patch adds a `synchronize()` that blocks the mooncake transfer worker, trading
transfer overlap for correctness. The PR's own note flags it and `plan.md` calls it
"the first thing a reviewer will ask". It was unmeasured. Now it is not.

`sglang.bench_serving` through the router, ISL 32768 / OSL 256, concurrency 8,
24 requests, same rig, arms swapped by `relegs.sh` with the guard verified each time:

| metric | stock | patched | delta |
|---|---|---|---|
| request throughput (req/s) | 0.4497 | 0.4478 | **-0.42%** |
| output throughput (tok/s) | 115.12 | 114.64 | **-0.42%** |
| mean TTFT (ms) | 11928.6 | 11997.1 | **+0.57%** |
| mean TPOT (ms) | 13.254 | 13.082 | -1.30% (better) |
| duration (s) | 53.37 | 53.59 | +0.42% |
| completed | 24/24 | 24/24 | — |

**No measurable regression.** Every delta is within ±1.3% and they do not point the
same way — TPOT is slightly *better* on the patched arm, which a real cost would not
produce. This is run-to-run noise, not a throughput price.

That is a plausible result rather than a surprising one: the barrier waits on an
event recorded on work the prefill forward has already queued, so by the time the
transfer worker dequeues a chunk the event is usually already complete. The wait is
mostly free; what it buys is that the rare case where it is *not* complete no longer
corrupts the KV.

**Caveats on these numbers, which belong in the PR:**
- One run per arm, n=24 each. Enough to exclude a large regression; not enough to
  resolve a sub-1% effect. The claim is "no measurable cost at this scale", not
  "provably zero".
- Taken on a 200 Gb/s `mlx5_0` link (mode B), not the 8x400 Gb/s ionic rails. A
  faster transport shortens the transfer, which could make the barrier a larger
  share of it. Unmeasured here.
- Concurrency 8. The barrier serialises per transfer worker, so heavier concurrency
  is where a cost would most plausibly appear.

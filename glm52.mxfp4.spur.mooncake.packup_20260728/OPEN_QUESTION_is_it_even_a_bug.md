# [ANSWERED: NO] Was the "degenerate output" actually a bug?

> **Resolved 2026-07-29: it was not.** With the chat template applied and
> `temperature=1.0, top_p=0.95`, both MXFP4 and FP8 give **0/128 degenerate at
> concurrency 128**. Full result:
> `../glm52.mxfp4.spur.mooncake.packup_20260729_degenerate_output/`.

## Original text (the question, as posed)

Date 2026-07-29. Raised by the user, and it exposes a real hole in the
reasoning chain built over the preceding several hours.

## The two claims I had been treating as evidence

1. **"Output collapses into a repeating loop."**
2. **"The same prompt at `temperature=0` gives different answers."**

Neither is sound on its own.

### (1) Looping is what greedy decoding does

Degenerate repetition under greedy/low-temperature decoding is a documented
property of neural LMs (Holtzman et al. 2019, *The Curious Case of Neural Text
Degeneration*). These prompts are raw base-LM completions —

```
"Explain quantum computing in detail, part 31."
```

— sent as `text` with **no chat template**, so a numbered-list continuation
that eventually sticks is a plausible completion, not obviously a defect.

**I never measured the baseline**: what fraction of these prompts loop on a
*healthy* engine at `temperature=0`? Without that number, "1.5–2.6 % loop" is
uninterpretable. Every rate reported in `RESULT_nomtp_control.md` and
`RESULT_pd_vs_mix_control.md` is subject to this.

### (2) Nondeterminism at temperature=0 may be expected

sglang ships:

```
enable_deterministic_inference: "Enable deterministic inference mode with
                                 batch invariant ops."     default False
```

The existence of that flag is an admission that **without** it, batch
composition changes reduction order and can flip an argmax. All our servers
run with it `False`.

So "3 runs of the same prompt gave 3 different texts" — which I presented as a
hard defect — is plausibly documented behaviour, not a bug.

## What would actually discriminate

Neither *looping* nor *varying* is a defect by itself. What cannot be
explained away is a **load-dependent** difference: the model's own greedy
behaviour must not depend on how many unrelated requests share its batch.

So: hold prompts, server, and sampling params fixed; vary **only** concurrency.

```
conc=1    -> intrinsic greedy loop rate (the missing baseline)
conc=8    -> mild batching
conc=128  -> the regime where failures were observed
```

- flat rate  => looping is the model's nature; there is no engine bug here and
  the whole line of investigation was chasing a non-defect
- rising rate => batching is injecting corruption

The same sweep also measures, per prompt, whether the *text* is identical
across concurrency levels — quantifying how badly batch-invariance is broken,
separately from whether the output is degenerate.

Tool: `dpa_mtp_fix/patches/scripts_20260729/baseline_greedy.py`.

## Second gap found while investigating this

The degeneracy predicate examined the **whole string** (unique-char count, a
long single-char run). A 512-token output that is 200 tokens of good prose
followed by 300 tokens of `1.1.1.` has plenty of unique characters and was
scored **coherent**.

Re-checking the stored tails of requests previously classed coherent:

```
coherent by the old predicate : 501
  of those, tail is looping   :  11  (2.2%)
```

**So every previously reported rate is a lower bound.** The cases labelled
"degenerate" were only the ones that broke at token 1.

`probe_onset.py` replaces it: stores full text and binary-searches the offset
where the tail becomes periodic. Onsets observed with DPA off:

```
onset=0    '1.3.2.1.3.2.1.3.2...'                       loops from the start
onset=158  '...two-mode geometric cnot gate.'  -> '. the. the. the.'
onset=170  "...shor's 4-qubit algorithm [11]." -> '. the. the. the.'
onset=356  '...historical setting of quantum mechanics.' -> '1.1.1.1. the. quantum.'
```

One failure mode with a variable onset, not two distinct phenomena.

## Status of the earlier conclusions

The *comparative* results are less affected than the absolute ones, since all
arms were measured with the same (biased low) predicate:

- MTP vs no-MTP, PD vs mix, DPA on vs off — all showed the same rate, and that
  equality does not depend on the predicate's calibration.
- What is **not** established is that any of those rates is abnormal.

Retraction already recorded separately: the "1.04 % vs 2.86 %" MTP comparison
was confounded by `--disable-custom-all-reduce` sitting inside the MTP block
of `pd_leg_spur.sh`.

## Infrastructure note

`hold_node.sh` ran `sleep 36000` (10 h) under `-t 12:00:00`. The script exited
on its own and jobs 9005/9006 went **COMPLETED mid-experiment** at exactly
10:00:27 and 10:00:56 — killing the FP8 run that was mid-boot. Fixed to
`sleep infinity` so only the scheduler wall ends a job.

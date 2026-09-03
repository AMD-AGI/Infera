# Notes — PD single-node: the defects, the transport, and one trap four times

---

## 1. Six wrapper defects. One design error, six faces, none of them errored.

**What.** `bash cluster.singlenode.sh up` could not work as shipped, for six
independent reasons.

**Why.** The GLM-5.2 kit's interface is **strictly per-leg** —
`PREFILL_DPA`/`DECODE_DPA`, `PREFILL_MTP`/`DECODE_MTP`, `PREFILL_KVD`/`DECODE_KVD`
— and `engine/up.sh` forwards exactly the names it knows, because `on()` runs a
fresh remote shell and nothing else reaches `leg.sh`. The GLM-5.3 wrappers
invented **single knobs** (`MTP`, `DPA`, `GPUS`, `EXTRA_ENGINE_ARGS`) that the
kit reads under no name at all.

**What each silently did:**

| knob | actual effect |
|---|---|
| `INFERA_IMAGE` vs `IMAGE` | `up` died at its first `require_env`, before anything started |
| KV ports | both legs bound 5557/8801 in one netns; second leg died at bind |
| `MTP=0` | decode leg launched at **`mtp=1`** (`up.sh` default `DECODE_MTP:-1`) |
| `GPUS` | **both legs onto GPUs 0-3** — measured 263.8 GB on 0-3, 0.3 GB on 4-7 |
| `EXTRA_ENGINE_ARGS` | `--disable-shared-experts-fusion` never reached the engine |
| `DPA=1` | read by nobody — but `up.sh` defaults are 0/1, so **silently correct by luck** |

**Context — why this is worse than a crash.** Two of the six (`MTP`,
`EXTRA_ENGINE_ARGS`) would have produced a **plausible benchmark number against a
configuration the wrapper file claims is different**. The `GPUS` one is worse
still: the prefill leg died with

```
ValueError: Loaded weights leave no GPU memory for the KV cache under
--mem-fraction-static=0.7. Raise --mem-fraction-static above 0.773
```

**That number is arithmetically correct and diagnostically wrong.** It is derived
from the memory free at that instant, and the reason there is none is that the
other leg is on the same cards. Following the engine's own advice would have
produced a *working* deployment on the wrong topology, with every subsequent
number meaningless and nothing logged. Meanwhile the decode leg came up healthy
and registered in etcd — so a router would have found one worker and served: a
"PD deployment" that is quietly a single aggregated leg.

**The `DPA` case is the subtlest.** A knob that appears to work while doing
nothing is worse than one that visibly fails: the next person setting `DPA=0` for
a single-variable round gets dp8 anyway and does not know.

**Fixes:** `patches/4493e33` (image name, KV ports), `patches/1b5ea46` (`GPUS`,
`EXTRA_ENGINE_ARGS`, per-leg `MTP`/`DPA`), `patches/f6ee2da` (preflight-derived
transport values), `patches/b2b1a08` (the OOM note). Each preserves the two-node
path byte-identically — verified by running the real `up.sh` under `SSH_CMD=echo`
so every per-leg command is printed rather than executed.

**Static greps under-detect this class.** `up.sh` contains the literal string
`MTP=${PREFILL_MTP:-0}`, so a word-boundary search for `MTP` matches an
occurrence that is being *assigned to*, not read. The reliable test is to launch
and read the resolved-args line.

**The GLM-5.2 wrappers have zero dead exports** — all 28 are read. The defect is
ours, not inherited, and nothing here casts doubt on the GLM-5.2 baseline.

---

## 2. The HIP transport: two dead names, one live name whose effect is invisible

**Installation is not runtime-gated.** `transfer_engine_impl.cpp:402-414` is an
unconditional `#ifdef USE_HIP` block. The image is built `USE_HIP=ON`, so hip
installs on every init and `HIP transport installed for intra-node GPU P2P`
appears 4× per leg **always**.

**Selection is gated**, at `multi_transport.cpp:489`:

```cpp
if (p == "hip")  return std::getenv("MC_DISABLE_HIP") ? 0 : 4;
if (p == "rdma") return 2;
```

`MC_DISABLE_HIP` demotes hip 4 → 0, so rdma wins for the device KV pool, which is
registered under both.

**The four names a reproducer would try, in order:**

| name | status |
|---|---|
| `MC_DISABLE_HIP_TRANSPORT` | **absent from the binary** (0 exact matches) — `leg.sh` set it since it was written; it never did anything |
| `MC_ENABLE_HIP_TRANSPORT` | **absent** — `leg.sh` unset it; also a no-op |
| `MC_DISABLE_HIP` | **live**, gates selection not installation |
| `MC_USE_HIP_IPC` | live, but **inverts its name** — `supportFabricMem()` enables *fabric memory* when set to `0`; it does not disable hip |

**The trap: the obvious check can never flip.** `HIP transport installed` is an
install-time log; `MC_DISABLE_HIP` gates selection. Different stages. It reads
4/4 in both states forever. **A correct hip-off deployment was brought up and
discarded unmeasured on the strength of its non-flip**, and from that non-flip a
false structural claim — "the A/B is impossible, capability-vs-choice is
unanswerable" — was drawn and nearly shipped.

**Verification that does work:** `MC_DISABLE_HIP=1` present in
`/proc/<pid>/environ` on both legs, plus the `:489` source read. Env-present +
source-consumes is sufficient; the throughput differential is the measurement.

**hip is single-node-only by design.** `selectTransport` calls
`isHipReachableTarget()` and skips hip buffers for any cross-host target, with the
in-source rationale *"without requiring the operator to set `MC_DISABLE_HIP`"*.
**Two-node PD never uses hip regardless of any setting**, so every hip question
here is a single-node question.

**A source tree shipped inside an image is not evidence about that image's
binaries.** The `.so` under test was compiled at mooncake `faae8dd4`; the tree
left in the image is `01d1eb2a`, **a month older**. A source read from the
convenient tree produced a confident wrong conclusion. Recovery needs no network:
the build commit is usually still in the image's own object store —
`git cat-file -t <sha>` then `git show <sha>:<path>`.

---

## 3. One trap, four instances: an observable that cannot move

Each of these returns the same value whether or not the world changed.

1. **`--showmemuse`'s `VRAM%`** — read 76 % on empty cards; does not fall when
   memory is released.
2. **`base_gpu_id`** — an index into each leg's *visible* set, so it reads 0 on
   both legs whether the GPU split is broken or correct. It was proposed as
   "cheaper and less ambiguous than reading VRAM"; it is neither.
3. **`HIP transport installed`** — install-time log for a selection-time
   variable. §2 above.
4. **`loopcheck.py` fed the wrong format** — see §4. **This one is the worst,
   because the wrong answer was the reassuring one.**

**The rule this yields:** *when a check does not move, rule out that the check is
blind before concluding the world did not move.* Three of the four produced not
merely a missed detection but a **false positive conclusion** drawn from the
non-movement.

---

## 4. The repetition arm, and a scorer that reports "clean" when misfed

**What.** At `isl 15500 / osl 3300`, conc 1, on a PD deployment with **MTP off**:
**60 % of generations loop** (6/10), worst 10-gram repeated **905×**, unique-word
ratio 0.070.

**Why it matters.** Both halves of the known mitigation were **verified in
effect**, not assumed:
- **chat template — on.** `--backend sglang-oai-chat` posts `messages` to
  `/v1/chat/completions` (`serving.py:961`, `:433`), which applies it
  server-side. This is a property of the endpoint, not a flag we set.
- **`temperature 1.0 / top_p 0.95` — genuinely reaching token selection**,
  because MTP is off on this arm so the ROCm argmax branch at
  `eagle_utils.py:726` never ran.

**So the arm eliminates both leading candidates at once.** Chat template is not
the cause; the ROCm argmax path is not necessary for it either. Every previously
degenerate arm ran MTP **on**, which is why that path looked like the explanation.

**The A/B that was cancelled before running.** `--apply-chat-template` is
**parsed and never read** in `benchmark/serving.py` — one occurrence, an
assignment at `:2072` that fires only for the image/mmmu datasets. Running it
would have given two identical conditions and a zero differential.

**Corroborated at 49× the sample size, after this arm was run.** The TP4 MIX
control — same condition, MTP off, and it includes the osl 3300 shape — was
scored over **980 generations** (see the alignment packup's
`results/repetition_tp4_control.md`):

| condition | osl 3300 looping | n |
|---|---:|---:|
| MTP **on** (earlier MIX arms) | 54 % | 80 |
| MTP **off** (this arm A) | 60 % | 10 |
| MTP **off** (TP4 control) | **54.7 %** | **490** |
| MTP off, **osl 320** (TP4 control) | **0.0 %** | **490** |

**Turning MTP off does not change the looping rate**, so the ROCm silent-greedy
fallback in EAGLE verify cannot be the cause — the rate is identical when that
path never executes. Arm A's n=10 conclusion holds at n=490. The clean/degenerate
split by output length is also confirmed at 490 samples per band.

**Remaining candidate, INFERRED and untested:** the prompts are themselves
repetitive by construction — `datasets/random.py:130-134` reaches a long ISL by
**repeating one ShareGPT conversation ~50× and truncating**. A model continuing a
highly repetitive prompt with repetitive output may not be a defect. **The test is
one run at osl 3300 with a naturally long prompt.**

**The scorer trap.** `loopcheck.py` expects **JSON Lines with a `text` key** (the
Case A capture format). `bench_serving --output-details` writes **one JSON object
with a `generated_texts` array**. Fed directly it parses the whole file as a
single record, scores `r.get("text","")` on an empty string, and prints:

```
1 generations ... looping = 0 (0.0%) ... worst=x0
```

**A clean bill of health from a parser that never saw the text.** Caught only
because `n=1, osl=0` contradicted a run known to have 10 completions of
1761–3174 words. Convert first — one record per generation with `text` and
`completion_tokens` — so the metric itself is unchanged and comparable.
`results/repetition_armA_generations.jsonl` is already in the converted form.

---

## 5. Reading a differential against a *measured* noise floor

The hip A/B's 23 % is meaningful and its 3 % is not, and that distinction is only
available because an arm was spent measuring variance.

Same configuration on two nodes: **44.07 → 44.52 (1.01×)** and **261.57 → 274.12
(1.05×)**. So run-to-run plus node/build variance is **~5 %**. The conc-8
differential of 23 % is 4–5× that; the conc-1 differential of 3 % is inside it
and is called noise, not a small effect.

That arm looked like overhead when it was proposed. It also retroactively
validated the cross-node points in the DPA-off curve and confirmed the rebuilt
image reproduces the original binary behaviourally, not just structurally.

---

## 6. Three arms for one attributable number

The PD/MIX comparison took three arms and **the first two were individually
plausible**:

| comparison | says | problem |
|---|---|---|
| PD (DPA on) vs MIX TP4 | 1.18× at conc 24 | PD has twice the GPUs |
| PD (DPA on) vs MIX TP8 | 1.15× at conc 24 | PD has DPA, MIX does not |
| **PD (DPA off) vs MIX TP8 (DPA off)** | **0.87× at conc 24** | attributable |

**That is what isolating a variable costs on a stack whose failures are silent.**
The third arm was the correction, not the plan.

And then the curve inverted the reading again: 0.87× at conc 24 became
**1.10 / 1.26 / 0.87 / 0.87** once conc 1/8/16 existed. **A single concurrency
point does not characterise a topology** — that happened three times in this
campaign.

---

## 7. Caveats that travel with every number here

- **Untuned MoE kernel** — `no tuned FlyDSL config` × 24. Floors, not ceilings.
- **The conc-1→24 slope is partly a rising cache-hit rate**, not scaling.
  `--dataset-name random` draws seeded ShareGPT and each arm's prompt list is a
  strict prefix of the next. See the alignment packup's `notes.md` §4.
- **`ttft_p99` is not a latency percentile on these arms.** It measures a
  discrete **whole-wave stall** — at PD conc 8, requests 8-15 (exactly the second
  wave) stalled ~17.9 s with a 0.10 s spread while nothing else exceeded 3.9 s.
  Present on aggregated MIX too, so **not caused by disaggregation**, but PD's
  version is ~3× longer and ~19× tighter. Mechanism **not established**. Full
  analysis: `workspace/results/ttft_wave_stall.md`.
- **Long-output shapes are not covered by the hip A/B.** KV volume per handoff
  grows with ISL and the balance could flip back. The p90 extension was designed
  and **not run** — packup was prioritised over further experiments.
- **The DPA-off curve mixes nodes**: conc 24 on `n01-33`, conc 1/8/16 on
  `n01-21`, MIX reference on `n01-33`. Justified by the measured 1.01×/1.05× node
  delta, but it is a mix and it is stated rather than hidden.

## 8. Attribution

All arms in this packup were run by this author. **No operator has independently
certified any of these numbers.** The `patches/` were authored by this author and
reviewed by the team lead, who authorised each engine-script edit; `f6ee2da` and
`667b02f` carry retractions of earlier claims made by this author.

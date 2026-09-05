
### T78 — the phase-⑤ door is *which operator the chain runs on*, not a defect in `apply.py`

> **DIAGNOSED 2026-09-05 20:4x by m4, read-only, no GPU. The title below and the axis it
> names — "real vs replayed" — are BOTH WRONG, and are kept rather than deleted so the
> next reader sees which side they failed on.**
>
> **The discriminator is the operator.** `apply.py` behaved correctly in every arm.
>
> | artefact | operator | top-level defs | apply |
> |---|---|---|---|
> | `p9`, `217d` (real stage 4) | `attention_chunk_gated_delta_rule` | `run` + one private helper | **refused** |
> | `192227` | `sampler_vocab_softmax` | all 9 of `sampler.py`'s public names | **passes** |
>
> The refused artefacts are a **benchmark-harness reference implementation** — they define
> `run(...)` for the measurement harness and **none** of the names of the file they would
> overlay. `surface_regressions` reported exactly that, naming all 7 dropped names.
> **A harness cannot be an overlay.** And the workset says so itself, in its own README:
> *"the baseline is the composite, not the single function, and that is a stated
> limitation"* — `chunk_gated_delta_rule_fwd_h` **has no reference anywhere in that tree.**
>
> **The "succeeds on a replayed one" premise does not hold either.** Replayed `133147`
> also failed — differently: manifest missing `operator_id`/`image`/`apply_mode`, and
> `files[0]` installing `fla/wy_fast.py` against a workset declaring `sampler.py`. **A
> pairing mismatch.** So "replayed passes" is true only of runs whose replayed artefact
> happens to be the `sampler_vocab_softmax` one. **The leader's four-run table below
> confounds the operator with the mock/real axis; the same four rows are consistent with
> the operator explanation and it was never tested against them.**
>
> **Left unverified, and stated as such:** whether a *real campaign* product would pass is
> untested. Bug 29 records that the 217 rescue's real-campaign product for this same
> operator was a three-part composite whose `def`s live inside a string literal, so the
> `ast` read saw every stock name as deleted — **the leader's guessed mechanism, but on
> the real-forge branch, not on what failed today. That tree is gone and it cannot be
> re-checked.** So this operator may fail on **both** branches, for two unrelated reasons.
>
> **Consequence, and it is a decision not a fix:** no stage-4 run repairs this (rule 18 is
> not in tension). The cheap move is to run the chain on an operator whose workset
> reference is engine-shaped — **but m3's real ranking picks the operator and no `--var`
> reaches that choice**, the same family trait as every mock adapter being bound to its
> corpus operator. **Open question for the user, not a bug anyone can fix in `apply.py`.**

**[SUPERSEDED — original framing, kept for the record.]** This is the phase-⑤ blocker and
it currently has no owner. Recorded 2026-09-05 by the
leader from a comparison of four runs, not from a diagnosis — **the observation is
first-hand, the mechanism is unknown, and nobody should treat the two as the same.**

| run | `mock_stages` | stage 4 | `apply_patch` |
|---|---|---|---|
| `p9` (093) | `none` | **real** | **FAILED** — 22 s, wrote nothing to its grant |
| m3's `217d` (217) | `none` | **real** | **FAILED** — same stopping point |
| m5's `keep12` (237) | `m1,m2,m3,m4` | replayed | **succeeded**, chain completed 17/17 |
| m5's `keep13` (237) | `m1,m2,m3,m4` | replayed | reached `integrate_and_verify` |

**Two independent full-real chains, two nodes, same stop. Two mock-1-4 chains, same node,
straight through.** The variable that tracks it is whether `kernel_optimization` was
produced by a live `optimize_kernel` or replayed from the corpus.

**What is measured about the failure** (m2, from `p9`):

```
agent: '${m5_agent:-runner}'      program body, no transcript, no zone logs
18:10:58 -> 18:11:20              22 seconds
did:     generated a 456-line patch from m4's optimized_kernel.py,
         staged trees/0/{a,stock,tree}
wrote:   nothing to .../v0/content -- "the agent's grant and it is empty"
event:   escalated   -> 2a5b4e8, program body, no agent to instruct
```

**It did the substantive work and then did not write its handoff.** Not resource, not
timing, not contention — **deterministic, in 22 seconds, with no GPU involved.**

**Candidate directions, none investigated:**

- the real `kernel_optimization` differs from the replayed one in a field `apply.py`
  reads — **note that the leader's builder produces the replayed one, and it has already
  been wrong three times today** (`packup_noop_` naming, missing `environment.yaml`,
  absolute host paths);
- `optimized_kernel.py` from a live `optimize_kernel` is `build_candidate.py`'s
  **three-part composite** (see bug 29), while the corpus's is whatever the corpus holds
  — and `apply.py`'s `surface_regressions` is `ast`-based, so a composite's defs live
  inside a string literal and read as *dropped*;
- something about `trees/0/{a,stock,tree}` staging that only a real artefact triggers.

**The second is the one I would look at first** and it is a five-minute read, not an
experiment: `apply.py:117-121` versus what a live stage 4 emits.

**Why this is worth a numbered item rather than a message:** it is the only thing standing
between us and a completed full-real chain. Stages 1–4 are sealed and passing on real
artefacts (`p9`, twelve tasks). **`apply_patch` is the first door of stage 5 and it is
shut for exactly the runs that matter.**

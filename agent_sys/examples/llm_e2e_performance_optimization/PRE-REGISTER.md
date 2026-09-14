# PRE-REGISTER — the chain tail (m3 / m4 / m5 / packup)

**Written 2026-09-06T06:39:45Z** (read from `date -u`, not written from memory)
on `smci355-ccs-aus-n04-25`, repo HEAD `f027073d` (committed 2026-09-06T06:35:03Z).
Author: m35. Zero results from this cluster had arrived when this was written —
that is the point of the file.

**This file is frozen.** When a result arrives it is graded by looking a row up
here, not by negotiating with the row. If a bar turns out to be wrong, that is a
separate conversation and it is settled by **appending a dated correction at the
bottom**, never by editing a row. A baseline that gets quietly improved destroys
the reason it exists — and the rule binds in both directions: it forbids
loosening a bar because the result was bad *and* tightening one because the
result was good.

---

## 0. What is being pre-registered

Every validator that fires on a kind this owner is responsible for:

| kind | validators |
|---|---|
| `operator_workset` (m3) | `check_environment`, `check_workset_shape`, `check_workset_runs` |
| `kernel_optimization` (m4) | `check_environment`, `check_optimization_shape`, `check_speedup_substantiated` |
| `patch_overlay` (m5) | `check_command_parses`, `check_environment`, `check_overlay_applies` |
| `stock.measurement` (m5) | `check_command_parses`, `check_environment`, `check_acceptance`, `check_bench_report`, `check_measurement_order` |
| `patched.measurement` (m5) | `check_command_parses`, `check_environment`, `check_acceptance`, `check_bench_report`, `check_patch_live` |
| `integration_report` (m5) | `check_environment`, `check_no_regression` |
| `e2e_packup` (packup) | `check_environment`, `check_packup_shape` |

Read off `steps/m3_analysis.yaml` / `m4_kernel_opt.yaml` / `m5_integration.yaml`
at HEAD `f027073d` by parsing every `- module: handoff` block, not by copying a
document.

---

## 1. The current state, as of this timestamp

**Source: `e2e-flow/WHAT-GREEN-ESTABLISHES.md` at HEAD `f027073d`, §"调用/拒绝"
table (lines 45–65) and §"2026-09-06 01:35" / "03:32" entries.** That document
was written about **the first cluster**. It is cited here as history, not as a
statement about this one.

**On THIS cluster (`smci355-ccs-aus-n04-25`), the count for every row below is
zero invocations and zero refusals.** Nothing in the tail has run here yet.

First-cluster history, which is the only history that exists:

| validator | invocations | refusals | what a PASS was worth there |
|---|---|---|---|
| `check_environment` | 375 | 3 | load-bearing; refused a real `integration_report` once (2026-09-06 01:35) |
| `check_workset_shape` | — | — | load-bearing (m3 owner's record) |
| `check_workset_runs` | — | — | load-bearing |
| `check_optimization_shape` | 15 | 5 | load-bearing |
| `check_speedup_substantiated` | 15 | 7 | load-bearing; aborts rather than swapping denominators |
| `check_overlay_applies` | 6 | **0** | **has never refused anything** |
| `check_acceptance` | 12 | **0** | **has never refused anything** |
| `check_bench_report` | 12 | **0** | **has never refused anything** |
| `check_measurement_order` | 12 | 2 | load-bearing |
| `check_patch_live` | 6 | **0** | **has never refused anything** |
| `check_no_regression` | 6 | 4 | load-bearing; **it is the one that has stopped the chain every time** |
| `check_packup_shape` | **2** | **0** | **weakest evidence in the package** — see §2 |

**All five never-refused validators belong to m5.** The last stage of the chain
has the thinnest evidence surface in the package, and it is the stage this round
has to get through.

### 1a. `packup` — the first data point in existence

- **The `packup` task's real body (`assets/packup.task/packup.py`) has never run
  inside a graph, anywhere.** Every first-cluster run stopped at a refused
  `integration_report` before `packup` was dispatched.
- `check_packup_shape`'s **two** invocations were both against a **mock**
  `e2e_packup` synthesised by `assets/lib/mock_m5.sh packup`. Per
  `assets/packup.task/readme.md:53-59`, the mock's source is a 47-file kit
  produced **out of band** by running `packup.py` by hand over nine sealed
  handoffs — it is not sealed and it was never produced by the graph.
- Therefore: **the first in-graph `e2e_packup` this round produces is the first
  one that has ever existed**, and both its validators will be seeing a real
  artefact of this kind for the first time.

**Consequence stated before the result: a first `check_packup_shape` refusal is
NOT prima-facie evidence of a defect in `packup.py`.** It is equally likely to be
a bar that has only ever been measured against a hand-made kit. Grading rule in
§3.7 says how to tell the two apart, and it was written now.

---

## 2. What each outcome would mean

The bars below were chosen before any result. Each row names the file to open.

### 2.1 `check_optimization_shape` on `kernel_optimization`

- **PASS** → the m4 artefact carries the four documents, the apply block and an
  `integration_point`. Given this round's m4 is **degraded by design**, a pass
  establishes *shape only* and specifically **does not** establish that a
  campaign ran.
- **REFUSE** → open the reported `found: [...]` list first. This validator
  enumerates what it actually saw, so **a refusal from it cannot be a zero-file
  zone** — that self-immunity is the reason to trust its refusal without the
  materials check. Still run the materials check before charging any *other*
  validator's refusal to a producer.

### 2.2 `check_speedup_substantiated` on `kernel_optimization`

- **PASS with `speedup: 1.0` and a self-declared degradation** → the intended
  outcome this round. An empty change measured against itself yields 1.0 and
  that is a **correct value, not a guess**.
- **PASS with a speedup > 1 from a degraded artefact** → treat as a **defect in
  the artefact**, not a success. Nothing this round is entitled to claim a
  speedup, and the schema
  (`assets/schemas/kernel_optimization.schema.json:739`) forbids a `mock: true`
  document carrying one at all.
- **REFUSE** → expected if the premise (gpu_arch etc.) disagrees, which it should
  not, because m1–m4 share one container on one node.

### 2.3 `check_overlay_applies` on `patch_overlay` — 0 refusals in 6

- **PASS** → establishes the overlay plan parses and each file's before/after
  hashes are recorded. Given six invocations and no refusal ever, **a pass here
  is the weakest kind of pass in this stage**: nobody has demonstrated it can
  say no.
- **REFUSE** → this would be its **first refusal anywhere** and is a result worth
  recording on its own, independent of the chain.

### 2.4 `check_acceptance`, `check_bench_report`, `check_patch_live` — 0 refusals each

Same reading as 2.3. **A pass from any of these three establishes that the
validator was invoked. It does not establish that it saw anything** — a pass
carries no reason, so the only retrospective check on it is the zone's file
count from `bash assets/lib/refusal_saw_something.sh <run dir>`.

`check_patch_live` is the exception worth stating: it re-hashes the patched file
**inside the running container** and matches declared markers, so its pass has
content the other two do not — *provided* the manifest declared a
`runtime_marker`. If the degraded artefact declares none, the second layer is
absent and the pass is thinner than it looks. **Check the manifest for
`runtime_marker` before quoting a `check_patch_live` pass.**

### 2.5 `check_measurement_order` — 2 refusals in 12

- **PASS on both arms** → the two arms did not overlap in time and followed the
  declared sequence. This is a real check with a real refusal history.

### 2.6 `check_no_regression` on `integration_report` — the gate that has ended every round

This is the single most likely place the round ends, and the bar is fixed now.

- **PASS** → the chain reaches `packup`. This has never happened.
- **REFUSE on `stock_vs_m2`** → the stock arm did not reproduce m2's bench within
  `stock_vs_m2_tolerance` (default **0.10**, `steps/m5_integration.yaml:412`).
  **The tolerance is not to be widened to obtain a pass.** First-cluster history:
  −26.1 % and −29.3 % with m2 replayed; **−11.2 % on `time_to_first_token` with
  m2 real**, against measured noise floors of 4.4 % / 0.76 % / 2.7 % at
  `bench_rounds=3` — i.e. the instrument is about twice as tight as the bar, so
  the residual is not noise. **If this cluster produces a number inside 10 %,
  that is a finding about the cluster and it is the most valuable single result
  of the round.**
- **REFUSE on a patched-vs-stock metric** → read `results/integration_report.json`
  `bars`; the verdict is re-readable against the bar it was decided against.
- **Sibling consequence, stated in advance:** a refusal here marks
  `integration_report` invalid and the framework marks the task's whole output
  set invalid with it. **`stock.measurement` and `patched.measurement` will then
  appear invalid despite passing.** Count verdicts, not handoff states — a ledger
  that counts states reports three refusals where there is one.

### 2.7 `check_environment` on any of these kinds

- **PASS** → an `environment.yaml` exists at the kind's expected path and its
  three guarded fields agree. It covers **3 of 28 fields** and an absent value is
  read as agreement, so a pass **does not** establish the record is right.
- **REFUSE on `e2e_packup`** → the likely mechanism is `env_render.py --inherit`
  failing, i.e. m1's record did not arrive — an upstream fact, not a packup fact.

### 2.8 `check_packup_shape` on `e2e_packup` — the first real data point

Bars, from `steps/m5_integration.yaml:426-452`, recorded here so the first result
is a lookup:

| requirement | bar |
|---|---|
| `require_files` | `README.md`, `REPRODUCE.md`, `environment.md`, `notes.md` |
| `min_content_lines` | README 20, REPRODUCE 15, environment 12, notes 8 |
| `min_command_lines` (REPRODUCE.md) | 8 |
| `require_dirs` | `results/`, `logs/`, `scripts/`, each ≥ 1 non-empty file (`st_size > 2`), counted recursively |
| `min_result_files` | 4 non-empty files under `results/` |
| placeholders | any of `[more information needed]`, `<...>`, `tbd`, `todo` in a required file is a refusal |

- **PASS** → the first `e2e_packup` ever produced in a graph satisfies the
  layout. It establishes the flow completed. It establishes nothing about the
  numbers inside the kit.
- **REFUSE naming a required file or a floor** → open the named file. This
  validator reports counts, so its refusals are checkable.
- **REFUSE with `THIS VALIDATOR DID NOT RUN: <ExcType>`** → a crash, not a
  refusal (`check.py:184-188`). Grade it as **no data**, not as a producer
  defect.

### 2.9 The zero-file zone, pre-committed

Roughly **one zone in eleven to thirteen** on real runs is handed zero files;
the mechanism is unknown and a controlled sample of 80 staged handoffs produced
zero empties, so **that rate is itself an estimate whose denominator was never
counted**. Rule fixed now:

> **No refusal in this round is attributed to a producer until
> `bash assets/lib/refusal_saw_something.sh <run dir>` has been run** (`bash`,
> not `sh`). A refusal from a zero-file zone is recorded as **no data**.

---

## 3. Grading rules fixed in advance

1. A **PASS** is graded on whether the validator has ever refused anything
   (§1 table), not on the fact of the pass.
2. A **REFUSE** is graded only after the materials check.
3. `speedup: 1.0` from the degraded m4 artefact is **the expected correct value**
   and is not to be read as a failure to optimise.
4. A degraded artefact that does not **say in itself** that it is degraded is a
   defect regardless of every verdict on it.
5. A number measured on the first cluster is **never** carried into an artefact
   produced on this one.
6. The launch line is recorded beside the run, because it cannot be recovered
   from the artefact (`${var:-default}` stays unrendered in the staged package).
7. **Distinguishing a `packup.py` defect from a bar that only ever saw a
   hand-made kit:** the refusal names a file and a count. Open the file. If the
   file is present and the count in the message matches what is on disk, it is a
   producer defect. If the message's count disagrees with what is on disk, it is
   an instrument defect and is reported as such — that exact failure mode
   (`iterdir` vs `rglob`) has already happened once in this validator.

---

## 4. Corrections

*(Append only. Each entry dated from a real `date -u` read. Nothing above this
line is edited.)*

### 2026-09-06T06:52:56Z — addition to §2.8: a `.MISSING` stub clears the `results/` floor

Not a change to a bar; a condition on how a **pass** is read. Nothing above is edited.

`packup.py:145-148` writes `<stage>.MISSING` for each upstream artefact it looked
for and did not find. **I measured that file at 67 bytes** — the leader's message
said 61; the number does not change the conclusion, and the measured one is 67.
`check_packup_shape` counts a file as real at `st_size > 2`, so:

> **Six `.MISSING` stubs satisfy `min_result_files: 4`.** The six labels are
> `m1.deploy_kit`, `m2.bench`, `m2.kernel_table`, `m3.workset`, `m4.verification`,
> `m4.forge_result` (`packup.py:132-139`).

**Therefore a `check_packup_shape` PASS does not establish that `results/` holds
four real artefacts.** It is "a pass carries no reason" in its most concrete
form. Fixed reading, before the first result:

> On any `check_packup_shape` pass, run `ls results/*.MISSING` in the kit and
> report the count beside the verdict. A pass with a non-zero `.MISSING` count
> is reported as **"passed with N upstream artefacts absent"**, never as
> "passed".

### 2026-09-06T06:56:18Z — addition to §2.6: `stock_vs_m2` has THREE outcomes, and the tolerance knob is inert

Read first-hand at `check_no_regression.validator/check.py:415-478` and
`compare.py:122-179,308-311,695-703`. §2.6 above described two outcomes; there
are three, and the third is not a refusal. Nothing above is edited.

| block state | validator | line |
|---|---|---|
| block absent | **refuse** | `check.py:422-429` |
| `ok: null` **and** `unavailable_because` empty | **refuse** — "silence here reads as a pass" | `check.py:432-437` |
| `ok: null` **and** `unavailable_because` non-empty | **PASS, printed loudly, returns `[]`** | `check.py:438-441` |
| `ok` set, every `abs(rel_delta) <= tolerance` | pass | `check.py:476` |
| `ok` set, any breach | refuse | `check.py:464-471` |

Two consequences fixed now, before any number arrives:

1. **`abs()` — both directions.** A stock arm *faster* than m2's bench breaches
   exactly as a slower one does, deliberately: `check.py:459-461` says checking
   one direction "would pass exactly the case where the later stage got a
   quieter node."
2. **`--var stock_vs_m2_tolerance` is inert, and inert in the way that hides
   itself.** The validator prefers the **producer's** value —
   `tolerance = block.get("tolerance")` at `check.py:445`, falling back to the
   `--var` only when it is `None`. The producer's value comes from
   `compare.py:310-311`, `os.environ.get("E2E_STOCK_VS_M2_TOLERANCE", 0.10)`,
   and a case-insensitive sweep of `steps/` and `assets/` finds
   **`E2E_STOCK_VS_M2_TOLERANCE` set nowhere** and the canonical invocation at
   `integrate_and_verify.task/readme.md:284-292` **not passing the flag**. So
   the producer always writes `0.10`, the validator always uses the producer's
   `0.10`, and the `--var`'s own default is also `0.10`.
   > **The wrong answer and the right answer are the same number**, which is why
   > nothing has ever noticed. Anyone raising `--var stock_vs_m2_tolerance` would
   > see no change and could conclude the measurement is stubborn.
   And the fallback at `check.py:446` looks unreachable in practice: the three
   producer paths that omit `tolerance` (`compare.py:142,148,156`) are all
   `ok: null` paths, which return before line 445.

   **This does not license widening the bar.** It records that the bar is not
   where an operator would look for it, so that a future decision to change it is
   made in the producer with the change visible, rather than by a `--var` that
   silently does nothing.

### 2026-09-06T06:57:17Z — addition to §2.4: what `check_acceptance` and `check_bench_report` CAN be made to establish

Both are `strength: strong` and both have **0 refusals in 12**. Read first-hand,
each has one arm that has never been exercised, and in both cases a `--var`
decides whether this round exercises it.

**`check_acceptance` — the ad-hoc arm.** `min_adhoc_cases: '${adhoc_cases:-3}'`
(`steps/m5_integration.yaml:306`), and the producer side is wired from the **same
`--var`** at `:161` (`E2E_ADHOC_CASES`) — I checked, and unlike
`stock_vs_m2_tolerance` this one is genuinely linked. The package's own probe
already records the open question at `lib/probe_validators.py:131`: *"a green row
here is not an exercised arm — `adhoc_cases=0` is the ..."*. Every first-cluster
run used `m5_agent=runner`, where no agent exists to generate cases.

> **Fixed before the result: `--var adhoc_cases` must NOT be lowered to 0 on a
> real m5. If it is, `check_acceptance`'s pass is recorded as "passed with the
> ad-hoc arm switched off" and this round adds nothing to its 0-refusal record.**
> At `3` with the real `e2e_integrator` agent, this is **the first time the
> ad-hoc arm has ever evaluated anything**, and that is worth recording whichever
> way it lands.

The two rules it exists for are the "produces a NUMBER rather than an error"
class, per its own docstring: a needle run that silently sent a 3000-token prompt
(guarded by `needle_min_token_ratio: 0.95`, read back from `usage.prompt_tokens`)
and an eval whose html holds fewer scored rows than its index claims
(`min_scored_per_eval: 20`). **A pass therefore does establish those two did not
happen** — that is more than "the validator was invoked", and it is the strongest
thing available from this one.

**`check_bench_report` — `expect_rounds: '${bench_rounds:-1}'`.** At the default
`1` the round-for-round arm comparison compares one round, and the first
cluster's measured noise floors were taken at `bench_rounds=3`. A pass at
`bench_rounds=1` establishes traffic was sent (`min_requests`, via
`integration_min_requests`, default 50) and the four `require_metrics` are
present. It does **not** establish anything about dispersion. Recorded so a pass
at 1 is not later quoted as if it were a pass at 3.

Note in passing, already correct and worth not re-deriving: `integration_min_requests`
is deliberately **not** `min_requests` — the two bars measuring the same quantity
in m2 and m5 were split on 2026-09-04 because one `--var` was moving two owners'
grading bars.

### 2026-09-06T06:52:56Z — addition: `check_patch_live`'s second layer is conditional on the payload printing

`check_patch_live.validator/check.py:172-227` matches the declared
`runtime_marker` regexes **in the engine log**, not in the file. A marker that is
a comment or a module-level constant buys nothing. Recorded now so a pass is not
over-read later: **quote a `check_patch_live` pass only after confirming the
manifest declares a `runtime_marker` and that the payload's markers are
`print()` statements.** If the manifest declares none, the check degrades to the
static mount proof and says so, and `--var require_runtime_marker=false` is the
switch that permits that.

### 2026-09-06T07:03:09Z — addition: `check_speedup_substantiated` MEASURES, and it constrains the m4 payload against m5

Read first-hand at `check_speedup_substantiated.validator/check.py:595-700,1030-1048`
and `build_workset.task/harness/_common.py:283-294`. This is not a static check:
`cost: gpu_hours`, and it re-runs the workset's performance entrypoint as
`run_performance.sh --impl <the optimised kernel>`. The harness `exec`s that file
in a bare namespace and raises *"the Definition's `<label>` defines no `run`"*
if it has no top-level `run`.

> **Two validators pull the same file in opposite directions:**
>
> | validator | requires the payload to be |
> |---|---|
> | `apply_patch/apply.py:828` (m5) | the **stock engine module** — it refuses any dropped public definition |
> | `check_speedup_substantiated` (m4) | **harness-shaped** — a top-level `run(**inputs)`, exec'd |
>
> **The only file satisfying both is the stock module plus an appended `run`
> delegating to the operator's public symbol.**

Consequences fixed before the artefact is built:

1. **A `call_site_fragment` operator is unusable for a second, independent
   reason** — there is no module-level symbol for `run` to delegate to. P7 of
   `RUNG5-CHECKLIST.md` already stops on it; this is why it is a hard filter and
   not a preference.
2. **The delegation's signature is an open risk that only m3's Definition can
   close.** The delegation is written `run(*args, **kwargs)` so it restates no
   signature it does not own, but the public symbol's parameters must still
   accept the Definition's declared `inputs` keys. If they do not, the failure
   arrives as an empty measurement, not as a signature error.
   **Grading rule: an empty or zero-shape measurement from
   `check_speedup_substantiated` is recorded as "the delegation did not match
   the Definition's inputs" until proven otherwise — NOT as a producer defect
   and NOT as evidence about the kernel.**
3. **`no claim` does not skip the measurement.** The short-circuit at
   `check.py:1037-1040` is reached *after* the candidate has been re-measured;
   it only skips the ratio. So a degraded artefact that claims nothing is still
   measured, and still needs a working `--impl`.

### 2026-09-06T07:11:51Z — additions: how to READ a pass, and three pre-registered explanations

**A. Two validators on my kinds now need their pass read rather than counted.**
Naming the pattern, because it has appeared twice and will appear again:

| validator | a pass alone does not establish | read this instead |
|---|---|---|
| `check_packup_shape` | that `results/` holds four real artefacts | `ls results/*.MISSING` — six 67-byte stubs clear the floor of 4 |
| `check_no_regression` | that the stock-vs-m2 comparison happened | the printed line — `ok: null` + a non-empty `unavailable_because` returns `[]` and passes |

> **The pattern: a validator that can pass by declining is a validator whose
> verdict is not its finding.** Report the printed line beside the verdict for
> both, always.

**B. The noise floors this file leans on are BORROWED.** §2.6 quotes 4.4 % /
0.76 % / 2.7 % at `bench_rounds=3`. **Those were measured on the first cluster.**
They are used here because there is nothing else, and they stay labelled as
borrowed until this cluster measures its own. Leader has set `bench_rounds=3`
for both m2 and m5 so the two stages grade against the same thing.
**Do not quote them as this cluster's noise floor.**

**C. Unknown-and-unreachable, kept as such.** Whether the first cluster's sealed
stage-4 kernel file preserved the engine module's public surface is **a property
of a file that does not exist on this cluster**. It is not "probably fine" and it
is not "unlikely"; it is **unreachable from here**, and it stays written that way.
The thing that replaced it as evidence is different and checkable: on every
first-cluster run the `call_site_fragment` + `overlay_files` stop
(`apply.py:808`) was armed and fired only `if dropped`, so **the only payload
shape that has ever passed `apply` anywhere is stock-module-shaped.**

**D. Pre-registered explanation for a thin or empty m3 worklist.** If `identify`
produces an `operator_identity` with few or no resolved operators, the first
hypothesis is **not** a producer defect:

- `identify.py:696` builds `repos` from `E2E_SGLANG_SRC` / `E2E_AITER_SRC`,
  **both unwired**, so `repos == []`;
- `min_resolve_ratio` defaults to `0.0`, so it **will not refuse for having
  resolved nothing**;
- `E2E_MAGPIE_ROOT` turns a missing value into `/nonexistent` rather than
  aborting.

Together those mean `identify` emits an artefact regardless. **A thin worklist is
graded as "the source trees were never wired up" until that is excluded.**
(Found by m2; recorded here so it is a lookup rather than a rediscovery.)

### 2026-09-06T07:17:24Z — addition: `check_workset_shape` / `check_workset_runs`, and the answer to "is P7 the only guard"

**Yes. P7 is the only thing standing between us and an unsatisfiable apply.**
`check_workset_shape.validator/check.py:234-287` (`_check_integration`) checks
the `substitution` / `public_symbol` pair for **internal consistency** and
**admits `call_site_fragment` by design**:

```python
if kind is None:                    return          # silent — absent field passes
if kind == "call_site_fragment":
    if symbol: problems.append(...)                 # refuses only the INCONSISTENT pair
    return                                          # -> a fragment with public_symbol: null PASSES
```

So **two admissible workset shapes are unusable for the degraded m4 route** and
neither is a refusal:

1. `substitution: call_site_fragment` with `public_symbol: null` — exactly what
   `build_workset.task/mock_adapt.py:654-655` writes, i.e. the first cluster's
   own workset was admissible and still could not feed a passing apply;
2. `substitution` absent entirely — the validator returns silently.

**Nothing downstream refuses either until `apply.py:808`, after a bring-up.**
That is what P9's `m3_extract.py` exists to catch on the login node.

**The positive half, which de-risks `--delegate-to`.** For `module_symbol` the
same function is genuinely strong:

| condition | outcome |
|---|---|
| `public_symbol` empty | refuse |
| `module_symbols` null | refuse — "UNVERIFIED rather than wrong" |
| `public_symbol not in module_symbols` | refuse, and it **lists what the image does define** |

> **So a workset that passes `check_workset_shape` with `substitution:
> module_symbol` has had its `public_symbol` confirmed present at module level
> in the target file**, against the symbol list `identify` recorded in the same
> read that produced `base_sha256`.

**And that is a genuinely independent second route to the fact my tool checks.**
`mk_reverse_payload.py`'s N2 guard reads the **live image** now; the validator
compares against a **list recorded at identify time**. Different sources,
different moments. **A disagreement between them is not a tie — it means the
recorded list is stale relative to the image**, and the live read wins.
That refusal message also enumerates what it found, so it is self-immune to the
zero-file-zone failure, same property as `check_optimization_shape`'s `found: [...]`.

**`check_workset_runs` is `cost: gpu_hours` too** — one shape (`reverify_shapes`
default 1), `min_groups: 5`, `min_iters_per_group: 10`. Cheaper than
`check_speedup_substantiated` but not free, and it runs at m3, before anything
of mine. Added to the budget line in RUNG5-CHECKLIST P10.

**Pre-registered reading of a `max_rsd` failure (`--var workset_max_rsd`, default
0.10).** Its own docstring: *"A node too busy to give a stable measurement fails
`max_rsd` … Neither is a defect in the artefact and both are correct verdicts."*

> **A `check_workset_runs` refusal naming `max_rsd` is recorded as "the node was
> not quiet", NOT as a defect in m3's workset** — unless `lines.sh` and
> `rocm-smi` at that moment show the node was quiet. **Establish the node's state
> at the time of the refusal before attributing it.** This is the same "one
> machine in two states" cause that `check_no_regression`'s residual is still
> open on, arriving two stages earlier.

### 2026-09-06T07:43:46Z — CORRECTION TO THE GRADING TABLE: there is a third outcome, and §3 did not have a row for it

Everything above grades **pass** and **refuse**. A validator can also **crash
without writing a verdict**, which produces neither — and that is what happened
to this round's first real validation, so this is not hypothetical.

```
07:29:59Z  check_deploy_kit: exited 1 and wrote no verdict.json; nothing was decided.
           lib/schema.py:169  from jsonschema import Draft202012Validator
           ImportError: cannot import name 'Draft202012Validator'
```

**Grading rule, fixed now:**

| outcome | what it establishes | how to tell |
|---|---|---|
| PASS | the validator was invoked; **not** that it saw anything | `verdict.json` exists, `true` |
| REFUSE | a judgement about the artefact — **after** the materials check | `verdict.json` exists, `false` |
| **CRASHED** | **NOTHING. Not about the artefact, not about the producer.** | **no `verdict.json` at all**; the reason is in the escalation's `attributes.detail` |

> **A crash is not a refusal, and the difference is invisible in every summary
> that counts handoff states.** The handoff is recorded `invalid`, which reads
> identically to a refusal. `check_packup_shape.validator/check.py:184-188`
> already knows this — it catches `Exception` and writes
> `"THIS VALIDATOR DID NOT RUN: <type>: <msg>"` into its reasons, *"because
> verdict.json cannot express the difference (todo.md T29)"*. **The seven other
> validators that call `schema_lib.validate` do not have that guard**, which is
> why this one crashed silently instead of saying so.

**And this is the sharpest limit yet on what my own materials check buys.**
At 07:36 I reported the zone green: **1 zone, 37 files, none empty**, and that
was *correct*. The run died at validation anyway, because **"was the validator
shown something" and "did the validator survive long enough to look" are
different questions, and only the first has a tool.** My check answered its
question and its question was not the one that decided the outcome.

> **Therefore: a green materials check is a precondition for attributing a
> refusal, never evidence that validation happened.** Before reading any board,
> check `verdict.json` **count** against the number of validators the kinds
> declare. A missing verdict is the third outcome and it is silent.

### 2026-09-06T11:45:15Z — MET: Finish Standard 3, for the first time this round

`mission.verify.e2e.md` §Finish Standard 3:

> *"Every refusal is attributed **after** running the materials check below, so
> no refusal is charged to a producer that was handed an empty directory."*

**Satisfied on 2026-09-06 at run `20260906T093443-ae2c38`**, which is the first
run on this cluster to produce a refusal at all:

```
check_deploy_kit refused the sealed kit at 10:19:31, naming scripts/env.sh:172
bash assets/lib/refusal_saw_something.sh <run>  ->  1 zone, 29 files, v1, none empty
verdicts on handoff 0d66e5f1:  {false} {true} {true}
```

**The zone held 29 files, so the `false` is a judgement about the artefact and
not a zero-file zone. The refusal is attributable to the producer**, and the
producer defect was independently established: the kit assigned a second,
differently-spelled name for a port that already had one
(`DK_ROUTER_PORT=$(( … ))` unconditional, where `: "${DK_PORT_ROUTER:=…}"` is
overridable), which two earlier kits had got right.

**Why this entry exists rather than a note in a message:** every prior use of the
materials check in this round was on a **pass**, where it is the weaker half —
it can only qualify a green that carries no reason of its own. **This is the
first time it has been run on a refusal, which is the case the Finish Standard
is written about.**

> **A Finish Standard we have actually met. There are few enough of those to be
> precise about which.** Standards 1, 2 and 4 remain unmet: no run has reached
> `packup`, module 4 has produced no artefact from a real campaign, and claim
> discipline is per-claim rather than a standard that can be marked done.

### 2026-09-06T12:22:04Z — addition to item D: a THIRD pre-registered explanation for a thin worklist

Item D already names two: `identify` runs with `repos == []` (`E2E_SGLANG_SRC` /
`E2E_AITER_SRC` unwired) and `min_resolve_ratio` defaulting to `0.0` so it will
not refuse for resolving nothing. **A third arrives with run `041f89`'s launch
line, and this one is deliberate rather than a defect:**

> **`--var stack_window_s=0`.** `shared.yaml:173` defaults it to 3;
> `m2_profiling.yaml:131` documents 0 as the way to declare *"no stack window
> was asked for"*, which is the fix for run `fdb0bd`'s two `_on` refusals.
> **The cost is that no launcher frames are captured, so m3 loses the
> launcher-attribution signal.**

**Grading rule, fixed before the workset arrives:**

- A `kernel_worklist` that is thin **or whose operators lack launcher
  attribution** is graded first against these three, **in this order**:
  1. `stack_window_s=0` — expected, declared, and visible in the launch line;
  2. `repos == []` — `identify` had no source tree to resolve against;
  3. the workload's own prefix-hit profile.
- **None of the three is a producer defect**, and a refusal or a thin result is
  charged to m3 only after all three are excluded.

**And it lands on my gate.** `m3_extract.py` reads
`edit_target.entry_function` for the `first_call` marker — which is exactly the
field launcher attribution feeds. **If it comes back empty, that is explanation
1, not a workset defect**, and the consequence is that `check_patch_live` loses
its second layer (see the `runtime_marker` addition above). The extractor
already stops on an empty `entry_function`; **that stop should be read as "no
launcher frames were captured", not as "m3 failed".**

### 2026-09-06T12:27:20Z — CORRECTION to §2.9: the zero-file-zone rate is not a rate, and our own data disagrees with the replacement

§2.9 above says *"roughly one zone in eleven to thirteen"* and cites a controlled
sample of 80 that found zero. **The other cluster has overturned the rate**, and
their finding is better than the number it replaces:

> Seven real runs, **exactly one empty zone each, always module 2's closure** —
> surviving m2 replayed vs real, m5 replayed vs real, both halves of a node, and
> different validator subsets. **Exactly one per run is inconsistent with an
> independent per-zone probability**, which would give some runs zero and some
> two. It read as a ~1-in-13 rate for a night **because every report gave a
> count instead of an identity.**

**Their method note is the durable part and it supersedes mine:** resolve the
zone's identity from its own `inputs.json` → handoff id → kind. Two indirect
methods (directory nesting; mapping the zone name's task id through the store)
disagreed with each other on one zone while the kind was never in doubt.

**And our own runs do not reproduce it.** Measured 2026-09-06T12:27:20Z across all six trees
on this cluster: **7 zones total, ZERO empty.** But the denominator matters more
than the total:

| run | zones | empty | can it test the claim? |
|---|---|---|---|
| 15c264, 3e8a03, 6ded23, ae2c38 | 1 each | 0 | **No** — only `deploy_and_prove`; m2 never validated |
| **fdb0bd** | **3** | **0** | **Yes** |
| 041f89 | 0 | 0 | No — not yet at validation |

**So the testable sample is ONE, not six.** Identity-resolved on `fdb0bd` by
their own method:

```
files=30  closure=deploy_and_prove        kind=deploy_kit
files=17  closure=run_profiling_mode_off  kind=profiling_mode_off.bench_result
files=39  closure=run_profiling_mode_on   kind=kernel_table
```

> **Both module-2 closures were validated and neither was empty.** That is one
> genuine counterexample to "exactly one per run, always module 2" — **not five**,
> and I would rather say one than let six trees look like a sample.

**Grading rule, unchanged in force but re-founded:** still run
`refusal_saw_something.sh` before attributing any refusal. What changes is what
a zero means — **it is no longer a lottery ticket, it is a claim about a
specific closure**, and the check should report which. **Not established
(their words, and I keep the limit): that only module 2's closure can be
affected.** Every run in their sample had module 2 in the graph and upstream of
live work.

---

## APPEND 2026-09-06T15:04:31Z — two of the four thin-worklist explanations are gone, BEFORE any result

Never backfilled; the original four stay above as written.

Four explanations were pre-registered for a thin m3 worklist. **Two have since
been removed by evidence that arrived after the registration and before any m3
result exists.** Recording now so the remaining set cannot be chosen after the
fact.

| # | explanation | status |
|---|---|---|
| 1 | Magpie has never completed a real scan here | **DEAD.** `check_kernel_table` PASSED on `298750`, which it cannot do without a kernel table. Found by m2 from the verdict side; I had leaned on this one. |
| 2 | `identify` resolution stuck at a lower level | **NO LONGER INDEPENDENT.** `identify.py:9` — level 1 reads the `launcher` block, and a successful stack capture is what writes it. So this is now *downstream of tonight's `CAPTURE_OK` test*, not a separate cause. If the capture succeeds this explanation cannot also be true. |
| 3 | `min_resolve_ratio: 0.0` refuses nothing | stands |
| 4 | the operator set itself is small | stands |

**Why this matters more than the arithmetic:** with four live explanations a thin
worklist explains itself. With two, and one of those two contingent on a test
that runs tonight, **a thin worklist after a successful capture is a finding
rather than a shrug.** That is the whole point of having written them down.

**What I am NOT doing:** not adding a fifth to keep the count up, and not
softening 3 or 4. If the remaining two turn out to be insufficient, that is a
result and it gets reported as one.

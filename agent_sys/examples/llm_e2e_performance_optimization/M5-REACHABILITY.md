# What is the shortest real path to module 5, and what blocks it today?

**Written 2026-09-06, m35.** A code and artefact read on the held node
`smci355-ccs-aus-n04-25`. **No GPU touched; chain 6 was live throughout.**

Module 5 has never run for real on **either** cluster, and the other cluster's
`packup` passed with 3/4/5 replayed — `check_no_regression` accepting a report
whose comparison blocks self-declare `unavailable_because: mock`, i.e. **it
passed by having nothing to judge.** So nothing anywhere has yet answered what
m5 actually needs. This is that answer.

---

## 1. Inputs — m5 declares four, and **one has never existed on this host**

`steps/m5_integration.yaml`, task `m5_integration`:
`inputs: [kernel_optimization, operator_workset, profiling_evidence, deploy_kit]`

| input kind | on this host today | where |
|---|---|---|
| **`deploy_kit`** | **YES — five real instances** | `runs/{15c264,3e8a03,6ded23,ae2c38,fdb0bd,041f89}/handoffs/<id>/v*/content/` |
| **`profiling_evidence`** | **NO — never produced, anywhere** | see below |
| **`operator_workset`** | **NO** | m3 has never run |
| **`kernel_optimization`** | **NO** | m4 has never run |

### `profiling_evidence` is the one worth naming, because its parts exist and it does not

The best run (`fdb0bd`) produced **four** module-2 kinds:

```
b35dc7d3   profiling_mode_off.bench_result
e1f4f2c2   profiling_mode_on.bench_result
88274848   profiling_mode_on.profile_result
a56d7e08   kernel_table
04172a90   (EMPTY SLOT — allocated, never written)
d2b75308   (EMPTY SLOT — allocated, never written)
```

**`profiling_evidence` is the merged kind**, produced by the `merge_profiling_evidence`
closure — and from that run's own `store/task`:

```
deploy_and_prove          succeeded
run_profiling_mode_off    succeeded
run_profiling_mode_on     output_validating      <- refused here; the run stopped
merge_profiling_evidence  waiting_handoff        <- NEVER RAN
m3_analysis               waiting_handoff
m4_kernel_opt             waiting_handoff
m5_integration            waiting_handoff
```

> **So m5 is not one stage away. From the best state reached today it is four
> task-completions away**, and the first of them is a merge that has never
> executed on this cluster. **The raw materials exist; the input does not.**

**Two empty handoff slots** are allocated and never written — that is what a
`waiting_handoff` downstream looks like on disk, and it is worth knowing so
nobody reads an allocated slot as a produced artefact.

---

## 2. What a real module 5 costs the host — price it before launching

### It takes the node. This is a property of the code, not of scheduling.

| fact | where |
|---|---|
| `mix_up.sh:81` calls `reset_gpus.sh` before **every** bring-up | `assets/serve/mix_up.sh` |
| that sweep is **node-wide**: it kills every KFD-holding process whose command name matches `^(python3?\|pt_main_thread\|ray\|sglang.*)$` | `reset_gpus.sh:25,37` |
| it protects **only** schedulers and daemons — not a co-tenant, not another of our lines, not m1's engine | `reset_gpus.sh:27` |
| the `sudo` fallback **works on this host** (`sudo -n true` → rc 0, measured), so it reaches root processes inside other containers | measured 2026-09-06 |
| arms are pinned to `0..TP-1` by `GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"`, **set nowhere**, so **no `--var` moves them** | `assets/serve/mix_worker.sh:26` |
| two coordination ports are **literals** — `kv-events:5557`, `kv-snapshot:8801` — so **two m5 stages can never share a host** | `mix_up.sh:70` |

> **A run with a real module 5 owns the node for the duration.** Not a preference;
> `--var` cannot make it otherwise.

### It runs ten steps twice

`assets/integrate_and_verify.task/readme.md` §STEPS: per arm — serve, smoke,
needle, probe, lm_eval, bench — then teardown, then the same again for the
patched arm, then compare. **`check_measurement_order` refuses the pair if the
arms overlap in time**, so the two arms are strictly serial by construction.

**The dominant cost is measured and recorded in the package** (`m5_integration.yaml:114`):

> *"`probe` costs 2062 s and 2001 s on the two sealed arms, **two thirds of an
> arm**, and `--max-tokens` is what sizes it."*

So **~34 minutes of probe per arm** at defaults, before smoke, needle, two evals
at 100 examples each, and `bench_rounds` replay rounds. **A default-configured
real m5 is well over an hour**, and that is before m4's own
`check_speedup_substantiated` ceiling of 60 minutes upstream of it.

---

## 3. The blockers, ranked, each with the one thing that settles it

### A. Missing artefacts — fixable by running an upstream stage, no code change

| # | blocker | settled by |
|---|---|---|
| A1 | `run_profiling_mode_on` must pass validation. On `fdb0bd` it **refused twice** (no stack window captured while one was asked for). Fixed in later launches by `--var stack_window_s=0`. | the next run reaching `_on` validation with 0 refusals |
| A2 | `merge_profiling_evidence` must run and produce `profiling_evidence`. **Never executed on this cluster.** | `find <run>/handoffs -name README.md \| xargs head -1 \| grep profiling_evidence` |
| A3 | `m3_analysis` must produce `operator_workset` | `python3 m3_extract.py <workset>` — exit 0 |
| A4 | `m4_kernel_opt` must produce `kernel_optimization` | see §4 and the route question in `README.md` |

**A2 is the one nobody has been tracking.** Every conversation today treated
"reach m3" as the next milestone; **the merge sits between m2 and m3 and has
never run.** I cannot say whether it is cheap or has its own failure mode,
because **it has never executed** — that is the honest state, and the
measurement that would answer it is simply a run that gets past `_on` validation.

### B. Host facts — all satisfied today, none is currently blocking

| fact | state |
|---|---|
| model | `/apps/data/models/Qwen3-32B` present |
| image | `infera/engine-sglang:qwen3-local-20260906` present, used by five bring-ups |
| cards | 8 × MI355X; m5 needs `0..TP-1` = 0-3 at `tp=4` |
| ports | `port_router/worker/etcd` are `--var`s; **5557 and 8801 are not** |
| foreign tenants | `rc_26_7_902`, `xiaoming-dev` — both map `/dev/kfd`, **both hold no GPU**; if either takes one before m5, the sweep would kill it |

### C. Code — one open item, and it is not blocking m5's *start*

`check_no_regression`'s `stock_vs_m2` compares the stock arm against m2's bench.
**It is the only validator that has ever refused anything in this package**, and
on the first cluster the residual was **−11.2 % on `time_to_first_token`** with
m2 real, against a measured noise floor of 4.4 % — *"one machine in two states"*,
and **no launch variable addresses it.** It blocks a *passing* m5, not a
*running* one.

---

## 4. Can module 5 be degraded the way module 4 was? **No — and that is the useful answer.**

The user authorised a self-declaring degraded m4. **The same move does not exist
for m5**, for a reason that is structural rather than a missing feature.

### The mock path is unreachable here, twice over

```
mock_m5.sh:26    E2E_MOCK_ROOT defaults to /shared_nfs/yihou/agent_sys/cheat_for_mock
mock_m5.sh:48-51 declines with 3 unless E2E_MOCK_STAGES names all|m5|stage5-integration
mock_m5.sh:332   packup branch: exit 1 when $E2E_PACKUP_MOCK/content/items/codes is absent
```

- `/shared_nfs/.../stage5-integration` — **absent** (checked; `/shared_nfs` is empty)
- `/data/yihou/cheat_for_mock/stage5-integration` — **absent**
- `E2E_PACKUP_MOCK` is declared in **no** steps yaml, so no `--var` reaches it

**And `--var m5_agent=runner` does not help**: it swaps the closure onto the
program agent, whose `entry.sh` runs *the same* `mock_m5.sh`, hitting the same
absent corpus.

### Why m4's trick has no m5 analogue

**m4's degradation works because a no-op optimisation is a real thing you can
measure.** The reverse payload is the engine's own module plus log lines; you
measure it and get `1.0`, which is a *correct value*.

**m5's output IS the measurement.** There is no "no-op measurement" — a
fabricated `stock.measurement` would have to fabricate:

| what | and it would have to satisfy |
|---|---|
| AIPerf exports | `check_bench_report` validates them against **m2's own `bench_result` schema** |
| eval html + index | `check_acceptance` recounts `"Correct Answer"` in the html and requires it to **equal** the index's scored count |
| needle results | prompt length read back from the server's `usage.prompt_tokens`, ratio ≥ 0.95 |
| container hashes | `check_patch_live` compares `env/container_hashes.tsv` against the plan |
| engine-log markers | `check_patch_live` matches the declared `runtime_marker` regexes in the recorded log |
| step timestamps | `check_measurement_order` refuses arms that overlap in time |

> **Correction to my own first draft, made before this file was committed.** I
> wrote that `check_patch_live` *"re-hashes the patched file inside the running
> container"* and called it the check that closes the door. **That is wrong.**
> `check_patch_live.validator/check.py:115-150` reads
> **`env/docker_mounts.json` and `env/container_hashes.tsv`** — files the
> **task** wrote during the live deployment (`m5_integration.yaml:598-602` says
> so explicitly: *"the patch-live evidence is collected by this task, not by its
> validator"*). **The validator runs no docker.** A fabricated
> `container_hashes.tsv` would satisfy it.
>
> I took the yaml's prose for the validator's behaviour and did not open the
> validator until the claim was about to be published. **The load-bearing
> sentence was the one I had not checked.**

### So the true obstacle is a RULE, not a mechanism — and that is worth being precise about

**No single validator makes a fabricated module 5 impossible.** Every check
above reads a file that the producing task wrote. A sufficiently determined
forgery — AIPerf exports passing m2's schema, an eval html whose
`"Correct Answer"` count matches its index, a `container_hashes.tsv`, a
`steps.json` with non-overlapping timestamps — would pass.

**What forbids it is that all of those are measurements, and this project does
not hand-write measurements.** `mission.verify.e2e.md` §Reference material:
*"Do not hand-write the missing parts — a replayed artefact that claims
measurements nobody took is worse than an absent one."*

> **m4's degradation is honest because the measurement is real:** a no-op change
> measured against itself genuinely yields `1.0`. **m5's output IS the
> measurement**, so there is nothing left to measure honestly once you remove
> the deployment. **The difference is not that m5 is better guarded — it is that
> m4 had a real cheap experiment available and m5 does not.**

**Enforcement is procedural, not mechanical.** If that is uncomfortable, the
mechanical fix would be for the validators to re-derive at least one number
themselves rather than read it — which is what `check_no_regression` already
does (it recomputes pooled means and the noise floor from raw per-round figures
rather than trusting the report's `verdict`). **`check_no_regression` is
therefore the hardest of them to forge, and it is also the only one that has
ever refused anything.** Those two facts are probably the same fact.

### What IS available: shrink it, do not fake it

All `--var`s, all reduce real work rather than simulate it:

```
--var eval_names=gsm8k          one eval instead of two   (P11: pre-authorised, with its cost)
--var eval_examples=<n>         default 100
--var eval_max_tokens=<n>       default 2048 — this is what sizes the 34-minute probe
--var bench_rounds=1            default 1; 3 is what makes dispersion interpretable
--var adhoc_cases=0             default 3 — but 0 switches off the never-exercised arm
--var needle_frontier_tokens=0  already the default
```

> **Conclusion, stated at the strength the evidence supports: module 5 is the
> one stage that cannot be *honestly* degraded.** Not "cannot be faked" — the
> validators read files a producer wrote, so a forgery would pass. What is
> unavailable is an honest cheap version: m4 had a real no-op experiment to
> measure, and m5 has none, because m5's output *is* the measurement.
>
> **So it can be made cheaper and it must be made shorter, but it needs a real
> two-arm bring-up on a node it owns** — and the thing standing between us and a
> fabricated alternative is the rule against hand-writing measurements, not a
> mechanism.

---

## 5. The shortest real path, stated as a sequence

1. a run reaches `run_profiling_mode_on` validation with **0 refusals**
   *(the `stack_window_s=0` fix is in; unverified past validation as of writing)*
2. `merge_profiling_evidence` runs → **`profiling_evidence` exists for the first time**
3. `m3_analysis` → `operator_workset`; **run `m3_extract.py` immediately** — its
   `baseline` classification decides m4's route
4. `m4_kernel_opt` → `kernel_optimization`, degraded, self-declaring
   *(budget the 60-minute `check_speedup_substantiated` ceiling)*
5. **before m5:** node exclusivity confirmed, foreign containers holding no GPU,
   `RUNG5-CHECKLIST` P0–P12
6. `m5_integration`: two serial arms, >1 h at defaults, then `packup`

**Steps 1–2 have never been completed on this cluster. Step 6 has never been
completed on any cluster.**

---

## 6. What I could not establish

- **Whether `merge_profiling_evidence` is cheap or has its own failure mode.**
  It has never executed. **The measurement is a run that gets past `_on`
  validation** — nothing cheaper will do, and no code read substitutes, because
  the question is about a task's behaviour rather than its text.
- **Whether m5's two arms fit a 16-hour hold alongside m1–m4.** The per-step
  costs I quote (`probe` 2062 s / 2001 s) are **the first cluster's, recorded in
  the package**, on a different model. They are borrowed and labelled as such.
- **Whether `stock_vs_m2` can pass here.** It has refused on every real
  comparison ever made. Nothing in this study bears on it.

---

## AMENDMENT 2026-09-06T15:04:58Z — the `compare.py` / `check_environment` barrier is DOWNGRADED, not removed

**Flagged by m2, who deliberately did not edit this file so I would check rather
than inherit. That is the right instinct and it is the failure mode this file
has paid for before.**

**The old finding, as written above:** `compare.py` never writes an environment
record, so `integration_report` could never satisfy `check_environment` by
construction, and mock hid it for a whole ladder.

**That is no longer true.** `assets/compare.py:813`:

```python
if args.environment:
    src = Path(args.environment)
    ...
    env_render.write(env_render.inherit(src, [], warnings_env), out, "structured_text")
```

**But it is conditional on a flag, and the else-branch is a note rather than a
failure**, deliberately (`:826-830`): a hard exit here ends a `kind: ai` stage
with all three outputs empty and nothing to retry into, so *"a refusable report
beats no report."*

```
compare: NOTE --environment was not supplied, so this report carries no
environment record and `check_environment` will refuse it.
```

**So the barrier moved from STRUCTURAL to CONDITIONAL:**

| | before | now |
|---|---|---|
| can the body satisfy `check_environment`? | never | yes, **if** STEP 10 passes `--environment` |
| what happens if it does not? | silent absence | a note on stderr, in a turn the agent reads |

`assets/integrate_and_verify.task/readme.md:271,293` spells the flag and `:296`
says *"`--environment` is not optional in practice."* **So the instruction
exists; nothing enforces it.**

**Class:** the same unenforced-dependency shape as `produced_by.commit =
'unknown'` — a knowable value, a working mechanism, a field waiting for someone
to remember. **The one thing better here:** the note lands on stderr inside a
step the agent runs and reads, so it reaches the one actor who can add the flag,
in the same turn. A validator's output would not.

**Consequence for m5's cost: one barrier lighter, and the residual risk is an
omitted flag rather than an impossible artefact.** The rest of this file's
reachability arithmetic is unchanged.

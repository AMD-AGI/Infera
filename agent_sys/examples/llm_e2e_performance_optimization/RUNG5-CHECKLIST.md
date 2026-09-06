# Rung-5 preconditions — the run that reaches `packup`

Written 2026-09-06T06:52:56Z on `smci355-ccs-aus-n04-25`, against repo HEAD
`f027073d`. Owner: m35. **Every line here is a stop, not a caution.** Each names
a command whose output decides, because a precondition that asks you to remember
something is the kind that decays.

> ## THIS IS THE ONLY FILE YOU NEED TO READ BEFORE LAUNCHING
>
> Folded 2026-09-06T13:45:10Z. Everything else in this workspace is supporting
> evidence for a step below. **If you read one thing and then write a launch
> line, read this one** — that is the failure mode the whole day was spent
> paying for.
>
> **The next round's first command**, with the reasons inline at P-1:
>
> ```
> --var trace_end_ms=120000     # NOT 60000 — see P-1, it blocked four stages
> --var stack_window_s=<unset>  # leave it; 0 costs identify resolution level 1
> ```
>
> Deeper background, none of it required: `ON-ARM-REFUSAL.md`,
> `M5-REACHABILITY.md`, `PREFLIGHT-VERDICT-298750.md`,
> `PREFLIGHT-HANDOVER-298750.md`, `README.md`.

---

## P-1 — `--var trace_end_ms=120000`, and the arithmetic is here so you need not look it up

**This is the first command of the next round.** The default that six launches
used, `60000`, is **below a floor nothing enforces**, and it cost four stages.

```
the m2 _on arm cuts TWO captures in sequence INSIDE ONE load, and the second
one requires the load to still be running (assets/load/capture.sh:130):

    measurement   warmup_s 60 + window_s 10        = 70 s
    stack window  0          + stack_window_s 3    =  3 s
                                                     ------
    floor         trace_end_ms must EXCEED           73 s + setup
```

**Below the floor:** the stack capture starts after the load has ended,
`capture_stacks.log` reads `ABORT: no aiperf load in flight`, `replay.sh` treats
that as non-fatal by design, and then **`check_trace_coverage` and
`check_kernel_table` both refuse — for something that is not the producer's
fault.** Measured on `fdb0bd`: load ended 12:14:58, measurement window ran to
12:15:19, stack capture aborted. Those two refusals blocked
`merge_profiling_evidence`, so `profiling_evidence` was never produced and
m3/m4/m5 never started. **One override, four stages.**

**`120000` is `assets/load/aiperf_replay.sh:38`'s own fallback**, not a number
anyone invented; `shared.yaml:149`'s default is `180000`. Both clear the floor.

> **UNTESTED.** The whole test is one grep after the next run:
> `grep CAPTURE_OK <run>/…/load.profiling_mode_on/capture_stacks.log`.

**The alternative is worse:** `--var stack_window_s=0` also makes both refusals
pass, changes **nothing** about the artefact, and permanently gives up
`identify` resolution **level 1** — pushing m3 onto Magpie, which has never
completed a real scan here, while `min_resolve_ratio: 0.0` refuses nothing.

---

## P-1b — m2 and m5 must receive the SAME `trace_end_ms`

One `--var`, **two declaration sites with different defaults**:
`shared.yaml:149` = `180000` (m2's path) and `m5_integration.yaml:126` = `60000`.
Passing it once feeds both, which is what every launch has done. **The hazard is
raising it for m2 alone.**

> `check_no_regression`'s `stock_vs_m2` compares m5's stock arm against m2's
> bench. The replay window selects **which slice** of the Mooncake trace runs;
> the trace carries `hash_ids`, so prefix hit rate — and with it TTFT and
> throughput — follows the slice. **A different window is a different workload,
> and the comparison then refuses for something that looks like a regression and
> is not.**

---

## P-0b — the next rung is `merge_profiling_evidence`, NOT m3

**"Reach m3" is the wrong milestone and everyone used it all day, including me.**

```
run_profiling_mode_on     -> the four component kinds
merge_profiling_evidence  -> profiling_evidence      <- HAS NEVER EXECUTED
m3_analysis               -> operator_workset
```

`profiling_evidence` is one of m5's four declared inputs and **has never existed
on this cluster.** Its components were produced on `fdb0bd`; the merge sat at
`waiting_handoff`. **It is not known whether the merge is cheap or has its own
failure mode, because it has never run** — and no code read substitutes, since
the question is about behaviour.

**Check it explicitly rather than assuming m3 is next:**

```sh
find <run>/handoffs -name README.md | xargs head -1 | grep profiling_evidence
```

---

## P-0c — preflight verdicts land in THREE different places

**Looking only under `runs/` will make you conclude a bring-up recorded
nothing.** Measured on `298750`, which wrote three, none where the last one was:

| bring-up | where its `preflight.json` landed |
|---|---|
| m1's deploy | inside the **sealed kit** — `handoffs/<id>/v*/content/items/codes/<packup>/results/` |
| `check_deploy_serves` | **`validate_work_root`** — `/data/yihou/e2e_flow/validate/serves-<id>/` |
| an m2 arm | **`work_root/<arm>`** — `/data/yihou/e2e_flow/pmoff/` |

```sh
find /data/yihou/e2e_flow /data/yihou/agent_sys_runroot/runs/<id> \
     -name 'preflight.json' -printf '%T+  %p\n' | grep -v /package/ | sort
```

**Before reporting that anything went unrecorded, search both roots.**

---

## P0 — the package copy is not stale

```sh
grep -m1 source_commit <generated-tree>/GENERATED.txt
git -C <repo> rev-parse HEAD
```

**STOP unless they are equal.** A tree that was not regenerated tests the code
you had, not the code you have, and the run says nothing about which. This has
already been reported once as "the fix did not work" when the fix was never in
the tree.

---

## P1 — is there a live chain on this host? (this one can veto; P2 cannot)

```sh
sh <package>/assets/lib/lines.sh
```

**STOP if any line is live and is not this one.** Ask this **before** reading the
cards: a chain sitting in a CPU stage holds no GPU and will want the cards back,
so an idle card reading does not clear it. `lines.sh` reads declarations, not
occupancy — that is why P2 exists and why neither replaces the other.

---

## P2 — the cards, and the containers, on the host itself

```sh
rocm-smi --showpids
rocm-smi --showmemuse
docker ps -q | wc -l
docker ps --format '{{.Names}}\t{{.Label "infera_e2e_run"}}\t{{.Status}}'
```

Not `docker ps -a | head -N` — a `head` cuts running containers off the bottom
and the surviving output looks complete. Count with `docker ps -q | wc -l`.

> ### STOP CONDITION, and it is the one that owns the node
>
> **If any container we did not create is holding a GPU: STOP and report to the
> leader. Do not launch and let `reset_gpus.sh` decide.**

Why this is a stop and not a caution — measured, not reasoned:

| fact | where |
|---|---|
| the sweep runs **on the host**, so `rocm-smi --showpids` gives it host-namespace pids and other containers' processes are reachable | `assets/serve/mix_up.sh:82` calls `bash "$SCRIPTS/reset_gpus.sh"` |
| it `kill -9`s anything whose `comm` matches `^(python3?\|pt_main_thread\|ray\|sglang.*)$` | `reset_gpus.sh:25` |
| it protects **only** `slurmstepd\|slurmd\|slurmctld\|kubelet\|containerd\|dockerd\|systemd` — not a co-tenant, not another of our lines, not m1's engine | `reset_gpus.sh:27` |
| the sudo fallback works here: **`sudo -n true` → rc 0, measured 2026-09-06T06:35Z on this host** | so root-owned processes inside other containers are not out of reach |
| the symmetric hazard: >2 % VRAM held by an *unrecognised* process makes it `exit 1` after 180 s and `mix_up.sh` aborts | `reset_gpus.sh:55-69` |

Do **not** plan around the observation that `^python3?$` fails to match
`python3.12`. It is true and it is not a guard.

**TP is 4, not 8** (leader, corrected 2026-09-06T07:11:31Z). m1 launched
`--var tp=4 --var gpu_devices=0,1,2,3`. So m5's arms, pinned to `0..TP-1` by
`GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"`, land on **cards 0-3**, not 0-7.
**This does not make cards 4-7 available.** `reset_gpus.sh` sweeps the whole
node whatever TP is, so a real m5 still owns the host and the four idle cards
are not a place to put anything.

**Ownership is a label, an auto-remove flag and a process list — never a name
prefix.** `yihou_` is shared and `m3_` has been read as ownership and been wrong
four times. Measured here 2026-09-06T06:35Z: `rc_26_7_902` and `xiaoming-dev`
both carry `/dev/kfd` and `/dev/dri`, both `AutoRemove=false`, labels only
`org.opencontainers.image.version:24.04` — **no `infera_e2e_run`, so not ours**,
and at that moment holding no KFD handle.

---

## P2b — while m3's `check_workset_runs` is measuring, nothing else of ours touches a GPU on this host

Same class as the m5 STOP above, and for the same reason: **our own scheduling
can manufacture a refusal that reads as a defect in someone's artefact.**

`check_workset_runs` is `cost: gpu_hours`, it lands at **m3**, and `max_rsd`
defaults to **0.10**. Its own docstring: *"A node too busy to give a stable
measurement fails `max_rsd` … Neither is a defect in the artefact and both are
correct verdicts."* So anything else of ours on this node during that window can
fail m3's workset for a reason that has nothing to do with m3.

- **Before m3's measurement starts:** announce it, and no other GPU work runs.
- **If it refuses naming `max_rsd`:** record it as **"the node was not quiet"**,
  not as a defect in the workset — unless `lines.sh` and `rocm-smi` **at that
  moment** say the node was quiet. Establish the node's state at the time of the
  refusal before attributing it.

This is the same "one machine in two states" cause that `check_no_regression`'s
11.2 % residual is still open on, arriving two stages earlier. **Second time
today the answer is "this host runs one GPU thing at a time."**

---

## P2c — a guard whose discriminator is written by the thing it guards

Recorded 2026-09-06T10:02:02Z, from m2's card-preflight fix.

The new preflight waits and re-reads **only when the GPU holder is one of ours**,
and the ownership label it reads is written by the **deployer agent** at
`deploy.sh:131-133` **inside the kit** — **not by any package asset.**

> **So the ownership branch works only while the agent keeps emitting that
> label. A `kind: ai` producer's habit is not a mechanism**, and nothing in the
> package fails if the habit lapses — the guard just stops finding a label.

**The saving grace is the direction it fails in, and that is m2's design rather
than luck:** an unidentifiable holder **aborts immediately** rather than falling
through to the wait. So a lapsed label costs a false abort, not a silent
overwrite. **Check that clause survives any future edit to the preflight** — it
is the half that makes the rest safe.

**Same shape as my own `reset_gpus.sh` guard** (`reset_gpus.guard.patch`), which
reads `infera_e2e_run` — but that one is written by `mix_up.sh`, a **package
asset**, so it is a mechanism rather than a habit. **Worth stating the
difference when either is discussed: same field, different durability.**

---

## P3 — `mock_stages` must exclude m5, and there is no smoke test for what follows

`assets/packup.task/entry.sh:11-14` calls `mock_m5.sh packup` first.
`assets/lib/mock_m5.sh:48-51` declines with **3** only when m5 is absent from
`E2E_MOCK_STAGES`, whose default is `all` (`steps/m5_integration.yaml:163`).
Otherwise control reaches `mock_m5.sh:332`, which needs
`${E2E_PACKUP_MOCK:-/shared_nfs/.../packup-out-of-band}/content/items/codes` and
**`exit 1`s** when it is absent. `E2E_PACKUP_MOCK` is declared in no steps yaml,
so no `--var` reaches it, and `/shared_nfs` is empty on this cluster.

> **A mocked m5 dies at `packup` with `exit 1` before `packup.py` runs.**
> `mock_stages` must exclude m5. There is no mock rehearsal of packup available
> here: **the first `e2e_packup` is both the first real one and the first one
> that has ever existed.**

---

## P4 — the redact prefixes, because this one fires last and costs everything

`packup.py:528-532` runs `redact.py` under `check=True`, and `e2e_packup` is
`is_end: true`. A non-zero here means **the flow's only export is never written,
after the whole chain has run.** The prefix table it passes is only
`TASK_PACKAGE, ZONE, TMPDIR, HOME, MODEL_MOUNT (= parent of E2E_MODEL_PATH),
WORK_ROOT (= E2E_WORK_ROOT), MOCK_ROOT` — the **smallest** of the seven redact
tables in the package.

```
--var work_root=/data/yihou/e2e_flow
--var scratch_root=/data/yihou/e2e_flow/kfo        # MUST be under work_root
```

**Why `scratch_root` and not just `work_root`:** `KFO_SCRATCH_ROOT` has its own
independent default (`steps/m4_kernel_opt.yaml:365`,
`${scratch_root:-/mnt/m2m_nobackup/yihou/e2e_flow/kfo}`), and **`SCRATCH_ROOT` is
not in packup's prefix list.** m4's paths reach packup raw: `packup.py:180`
`copytree`s `kernel_optimization/items/codes` **wholesale**, and
`optimize_kernel.task` calls `redact.py` **nowhere** — so `WORK_ROOT` is the only
prefix that can cover m4's scratch paths, and it only does so if `scratch_root`
is underneath it. `packup.py:500-510`'s own comment names "m4's scratch work
root" as one of the roots that made this fail before.

**m2's materials are NOT exposed** (checked, since it was asked): m2's producers
redact with their own tables — `load/replay.sh:455-462` carries
`TRACE_DIR=$(dirname "$E2E_AIPERF_TRACE")` and `MODEL_MOUNT`, and
`accept/measure.sh:474-486` carries `GSM8K_DIR` and `TRACE_DIR` — and both
**abort loudly at their own stage** if a path cannot be placed. So m2's paths
arrive at packup already as `@TRACE_DIR@`, and an already-placeholdered path is
not a raw path. The failure, if it comes, surfaces at m2, which is the right
place.

**Residual risk, stated rather than assumed away:** any root recorded by m1 or m3
in a `.py/.sh/.json/.jsonl` is also uncovered — neither `build_workset.task` nor
`optimize_kernel.task` calls `redact.py`. `packup.py` copies only a `.yaml` from
the workset (`m3.workset`), and `.yaml` is not in `REFUSE_SUFFIXES`
(`redact.py:56` = `{.py,.sh,.json,.jsonl}`), so m3 is incidentally safe. m4 is
not, and P4's `scratch_root` rule is the whole mitigation.

---

## P5 — the mocked/real variable table, read as a table

`expect_ranks`, `adhoc_cases`, `bench_rounds` at minimum. **Audit every row
against this launch's `mock_stages`, not the rows you remember**, and note these
three checked rows so nobody re-derives them:

| `--var` | reaches | status |
|---|---|---|
| `adhoc_cases` | producer `E2E_ADHOC_CASES` (`:161`) **and** `check_acceptance.min_adhoc_cases` (`:306`) | **properly linked.** Do **not** set it to 0 on a real m5 — that switches off the one arm of `check_acceptance` that has never been exercised. |
| `bench_rounds` | `check_bench_report.expect_rounds` | live. At the default `1` a pass says nothing about dispersion; the first cluster's noise floors were measured at `3`. |
| `stock_vs_m2_tolerance` | `check_no_regression` args **only** | **INERT.** The validator prefers the producer's `tolerance`, which comes from `E2E_STOCK_VS_M2_TOLERANCE` — set nowhere — so the producer always writes `0.10`. Changing this `--var` changes nothing; the identical default is why nobody noticed. | — three
launches were lost fixing them one at a time, and reading the table as a table
found the remaining two in one pass. A real value carried into a mocked stage
produces a refusal that reads exactly like a producer defect.

---

## P6 — record the launch line beside the run

The staged package keeps `${var:-default}` unrendered, so which `--var` a run
used **cannot be read back from the artefact**. Write the exact command line into
the run directory before starting it.

---

## P7 — the operator must be `module_symbol`

The moment a real `operator_workset` exists, read its
`operators[].integration.substitution`.

```sh
grep -n "substitution\|public_symbol\|apply_mode\|build_step" \
     <workset>/items/codes/workset.yaml
```

**STOP and report if it is `call_site_fragment`.** `apply_patch.task/apply.py:808-827`
hard-stops that combined with `apply_mode: overlay_files`, and
`assets/schemas/workset.schema.json`'s `apply_mode` enum has exactly **one**
value — so the pair is unsatisfiable and no seed, no `--var` and no re-cut
payload can reach a legal combination. Also stop on a non-empty `build_step`
(`apply.py:395-401`): that operator needs the image rebuilt and cannot be an
overlay.

---

## P8 — `${PIPESTATUS[0]}` is EMPTY in this session's shell

Measured 2026-09-06 on this host: `$0` is `/bin/zsh`, and after `false | head -1`

```
PIPESTATUS[0]=''      pipestatus[1]='1'
```

zsh arrays are 1-indexed and the lowercase name is the live one. **So the
standing advice "use `PIPESTATUS`" yields an empty string here** — an unset
variable in a comparison, i.e. a failure read as a success, which is the same
shape as every other tool in the notes that answers a different question than
the one asked. In an interactive/`zsh` context use `${pipestatus[1]}`; inside a
`bash -c` or a `#!/usr/bin/env bash` script `${PIPESTATUS[0]}` is correct.
Better where possible: do not pipe the command whose status you need.

---

## P9 — the moment an `operator_workset` exists, run ONE command

> **This is not a wrapper around four greps.** `check_workset_shape` **admits**
> the shape that cannot work: `_check_integration` (`check.py:234-287`) checks
> the `substitution` / `public_symbol` pair only for **internal consistency**, so
> `call_site_fragment` + `public_symbol: null` passes, and an **absent**
> `substitution` returns silently. Nothing downstream refuses either until
> `apply.py:808` — **which is after a bring-up.**
> **This command is the only thing that moves that failure to the login node.**


```sh
python3 /data/yihou/e2e_verify_20260906/m35/m3_extract.py <the operator_workset content dir>
```

The content dir is whatever holds `items/codes/workset.yaml` — a staged input
(`$AGENT_SYS_INPUT_OPERATOR_WORKSET`) or a finished m3's output slot. **Exit 1
means STOP and report; exit 0 prints the `mk_reverse_payload.py` line to paste.**

It reads the four facts the degraded m4 artefact needs, plus the one risk that
cannot be closed before m3 exists:

| field | why |
|---|---|
| `integration.substitution` | **STOP on `call_site_fragment`** — `apply.py:808` hard-stops it with `overlay_files`, the `apply_mode` enum has one value so the pair is unsatisfiable, and there is then no module-level symbol for `run` to delegate to |
| `integration.build_step` | **STOP if non-empty** — `apply.py:395` refuses; needs the image rebuilt |
| `integration.public_symbol` | → `--delegate-to` |
| `integration.target_files` | → `--container-path` (SGLANG_ROOT here is `/sgl-workspace/sglang/python/sglang`) |
| `edit_target.entry_function` | → `--function`, the `first_call` marker |
| the Definition's `inputs` keys | the signature risk: the delegation is `run(*args, **kwargs)` and restates nothing, but the public symbol's parameters must accept these keys. **A mismatch surfaces as an EMPTY measurement, not an error** — PRE-REGISTER has the pre-registered reading. |

**Four facts have to agree and two are stop conditions, which is why this is one
command and not four greps** — reading them one at a time is how you get three of
four and launch.

Known-answer tested 2026-09-06T07:15:37Z, three cases, output pasted into the report:

```
module_symbol operator      -> rc 0, prints the ready-to-run command
call_site_fragment operator -> rc 1, "UNUSABLE for the degraded m4 route"
missing workset             -> rc 1, names the path it looked for (errors, not empty)
mixed workset (one of each) -> rc 1 overall, and the usable one STILL prints its command
```

The last case was a defect found by testing rather than by reading: the stop list
was accumulated across operators, so a stop on the first suppressed the command
line for the second — and a real workset carries five.

---

## P10 — the m4 validator costs GPU time, and the ceiling is 60 minutes

Recorded 2026-09-06T07:13:23Z for the time budget, so it is not discovered when it fires.

`check_speedup_substantiated` is `cost: gpu_hours` and it is **not** a static
check: it re-runs the workset's performance entrypoint on the card, twice —
once for the seed (`check.py:926`) and once for the candidate (`check.py:970`) —
over at least `min_shapes_measured: 3` shapes.

```
steps/m4_kernel_opt.yaml   timeout_seconds: 1800      per entrypoint run
                           min_shapes_measured: 3
check.py:766               timeout = _num(args.get("timeout_seconds"), 1800)
```

> **Worst case 2 x 1800 s = 60 minutes of wall clock before it gives up, on the
> GPU.**

**And it is not the only GPU-cost validator upstream of it.**
`check_workset_runs` (m3) is `cost: gpu_hours` as well — one shape
(`reverify_shapes` default 1), `min_groups: 5`, `min_iters_per_group: 10`.
Cheaper, but it lands before anything of m4's and it must be in the budget. That is a hard ceiling, not an estimate; the typical cost is two
> harness runs over three shapes and is much less. Budget the ceiling.

**A degraded m4 does not buy its way out of this.** The `no claim is made, so
there is no ratio to substantiate` short-circuit (`check.py:1037-1040`) is
reached **after** the candidate has been re-measured. It skips the ratio, not
the measurement.

---

## P11 — PRE-AUTHORISED CONTINGENCY: trimming `eval_names` to one eval

**Decided 2026-09-06T07:37:39Z, before the clock was binding, so that using it later is a
lookup and not an argument under pressure.** Leader's ruling: **do NOT trim
today.** At 07:36Z against a 14:00Z hold there were 6 h 24 m left and the largest
single item is 60 minutes, so the clock was not binding and the trim was not
bought.

### The trigger, stated in advance

> **Trim only if the remaining hold is under ~2 h and a full attempt does not
> otherwise fit.** Ten minutes per attempt is what it buys.

### What a trim costs — nothing on `check_acceptance`

Verified in code, not assumed (`check_acceptance.validator/check.py:164-204`):

- `evals_ok` is a **per-eval loop**. No cross-eval check, no minimum number of
  evals, no comparison between them.
- `min_scored_per_eval: 20` — per row. **Preserved on one eval.**
- the html-vs-index cross-check (`counted = html.count("Correct Answer")` vs the
  index's `scored`) — per row. **Preserved.** This is one of the two
  "produces a NUMBER rather than an error" failures a pass rules out.
- `needle_ok` reads `needle.json` and the `usage.prompt_tokens` ratio. **It never
  touches evals.** That is the other one.
- `check_no_regression:112` iterates `report.correctness.evals`, whatever is
  present. `compare.py:433` guards the pair with
  `if "gsm8k" in evals and "mixed_prefix_gsm8k" in evals`, so
  `prefix_reuse_delta` is **absent, not an abort**. The schema requires
  `correctness: ['smoke','needle','evals']` and **does not require**
  `prefix_reuse_delta`.

> **A pass from `check_acceptance` on one eval establishes exactly what a pass on
> two establishes.**

### What it does cost, in the schema's own words

`assets/schemas/integration_report.schema.json:328`:

> *"gsm8k against mixed_prefix_gsm8k, per arm. The pair is deliberate: the
> difference measures whether prefix reuse changes the answer, **which is a
> failure a kernel patch can cause and which no other check here can see.**"*

**Dropping the second eval silently removes the only check in the package for a
class of failure a kernel patch can cause.** Nothing refuses; the field is
simply gone.

### The condition that makes the trim defensible, and it must be re-checked

**Only trim while m4's artefact is a no-op by construction** — the reverse
payload adds log lines and changes no arithmetic, so the failure class that
check exists for cannot occur. `payload_record.json` carries
`"degraded": true` and `"expected_speedup": 1.0`, and
`mk_reverse_payload.py`'s surface check proves nothing was dropped.

> **If a later round ships a real optimisation, the second eval goes back, and
> this section is why.** Leader's reason for not taking the trim today is worth
> keeping verbatim: *"'Our patch cannot cause that failure' is an argument I
> would rather not need to be right about when I am not paying for the
> alternative."*

**How to trim, if the trigger fires:** `--var eval_names=gsm8k`. Record in the
run notes that `prefix_reuse_delta` is absent by decision, not by defect.

---

## P12 — DEFERRED WITH A TRIGGER: port the `THIS VALIDATOR DID NOT RUN` guard

**Decided 2026-09-06T07:49:10Z. Recorded, not declined, and deliberately NOT done now.**

`check_packup_shape.validator/check.py:184-188` catches `Exception` and writes
`"THIS VALIDATOR DID NOT RUN: <type>: <msg>"` into its reasons, *"because
verdict.json cannot express the difference (todo.md T29)"*. **The other seven
callers of `schema_lib.validate` do not have that guard**, which is why
2026-09-06's crash was silent rather than self-reporting.

It converts the third outcome from *no verdict* into *a verdict that says it did
not run* — and the second one is **countable**, which is the whole difference.

**Why not now** (leader's reasoning, kept so this is a decision and not a
deferral):

1. **It does not unblock anything.** The environment fix removes the trigger; the
   guard only makes a recurrence visible.
2. **It touches eight validator bodies and the package is staged per task at
   launch** — an untested package change arriving exactly when we finally have a
   run to spend. That is "a stale copy silently fails", inverted.
3. It is valuable, hence recorded rather than dropped.

> **Reopen trigger — port it the moment EITHER:**
> **(a)** a second crash-without-verdict is seen from any validator, **or**
> **(b)** the chain has reached `packup` once and hold time remains.
> **(b) is the good case:** harden after the round's goal is met, not instead of
> it.

**The eight to port to:** `check_deploy_kit`, `check_environment`,
`check_bench_report`, `check_bench_result`, `check_kernel_table`,
`check_no_regression`, `check_optimization_shape`, `check_workset_shape`.
(The last one carries its own `from jsonschema import …` at `:667` rather than
calling the shared lib — it is in the list for the same reason and by a
different route.)

---

## After the run — before attributing any refusal

```sh
bash <package>/assets/lib/refusal_saw_something.sh <run dir>
```

**`bash`, not `sh`** — it uses `set -o pipefail` and process substitution, and
`dash -n` cannot fail on either. A refusal from a zero-file zone is recorded as
**no data**, not as a producer defect. A PASS carries no reason at all, so the
zone's file count is the only retrospective check on it.

And count **verdicts, not handoff states**: one refusal on `integration_report`
marks the task's whole output set invalid, so two passing siblings appear
invalid with it.

---

## What the 2026-09-06 hold established, and what it did not

Written 2026-09-06T13:45:41Z, at the end of it, deliberately not the encouraging version.

**Established:**

- **Stage 1 is reproducible and seals inside a live chain.** Three separate runs,
  3/3 green each, on kits three different agents produced — and on run `298750`
  it sealed with `deploy_and_prove` reaching `output_validating` and the `_off`
  arm coming up behind it. **The cleanest single result: `check_deploy_kit`
  flipped false → true on an **instruction change alone**, package, image and
  host unchanged — a controlled comparison with one variable, rarer than a green
  board.**
- **The `_on` blocker is diagnosed**, to an **unenforced dependency between
  launch variables** rather than to a code defect: `trace_end_ms=60000` sits
  below a floor of `warmup_s + window_s + stack_window_s`, and **that override
  was ours, chosen to save time.** The repair is one `--var` (P-1) and **it is
  untested.**
- **A refusal was attributed properly for the first time this round** —
  `mission.verify.e2e.md` Finish Standard 3 — because the materials check ran
  before the attribution and the zone held 39 files.
- **The preflight instrumentation works**: three bring-ups, three verdicts, all
  three instruments in the declared order, the label query returning a real
  answer against two real foreign containers, and KFD recorded as non-deciding
  **in the artefact** rather than left as a bare zero.

**NOT established:**

- **Module 5 has never run for real, on either cluster.** It is **four
  task-completions** away from the best state reached, and one of those is a
  merge that has never executed.
- **`packup` has never been reached with module 5 real, anywhere.** The other
  cluster reached it with 3/4/5 replayed and `check_no_regression` passing on a
  report whose comparison blocks self-declare `unavailable_because: mock` — **it
  passed by having nothing to judge.**
- **The ambiguous-holder branch of the card preflight is untested after three
  chances** — idle node, idle node, and a handover where the predecessor had
  already settled. It cannot be manufactured honestly; it needs a foreign tenant
  to take a card.
- **`stock_vs_m2` has refused every real comparison ever made**, and nothing this
  hold produced bears on whether it can pass here.
- **The `trace_end_ms` repair, the `reset_gpus.sh` ownership guard, and the
  `THIS VALIDATOR DID NOT RUN` port are all written and none is applied.** Each
  says in its own header why, and each names the one observation that would let
  it land.

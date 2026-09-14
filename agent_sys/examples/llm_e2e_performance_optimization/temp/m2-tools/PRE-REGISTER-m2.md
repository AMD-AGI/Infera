# PRE-REGISTRATION — m2's validators, written before any result exists

**Written 2026-09-06T06:59:38Z** (`NOW=$(date -u +%FT%TZ)`, read once and
referenced — not typed beside a reading). Cluster
`smci355-ccs-aus-n04-25`, slurm 29184. m1's bring-up is live; **no m2 kind has
been produced on this cluster and no m2 validator has been invoked here.**

**Why this exists:** the pressure to reinterpret a bar arrives *with* the first
data point, not before it. Everything below is fixed now so that when a verdict
lands, each validator is one lookup rather than one negotiation.

**Append-only.** Corrections go at the bottom, dated, with what changed and what
did not. **Nothing here gets backfilled when results arrive** — a baseline whose
credibility rests on never being edited is worthless the moment it is edited;
one whose edits are visible stays usable.

---

## 0. The precondition on EVERY entry below — the zero-file zone

Before attributing **any** verdict on this page to the artefact it names:

```sh
bash agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/assets/lib/refusal_saw_something.sh <run dir>
```

`bash`, not `sh` — the header says `sh` and line 39 uses `set -o pipefail`,
which dash rejects.

The framework can hand a validation zone a **zero-file** `materials/`. When it
does:

- **a refusal says nothing about the artefact.** It is a well-formed report
  naming missing files that exist in the sealed content.
- **a pass says even less.** A pass carries no reasons at all, so no
  read-the-report heuristic can be applied to it. **The zone's file count is the
  only retrospective check on a pass.**

The discriminator that survives, and the reason the note-lines below matter:

> **A refusal that quotes a number computed from file contents proves the
> validator read the file. A refusal that only says "X is missing" does not.**

This is why the `(note)` lines in §1–§6 are listed as *evidence*, not as
decoration: `check_bench_result` printing `721 request record(s), 0 errored` and
`check_trace_coverage` printing `419218 GPU kernel events` cannot come from an
empty directory. Their absence beside a verdict is itself the signal.

---

## 1. `check_trace_coverage` — on `profiling_mode_on.profile_result`

**Reads:** `items/env/trace_manifest.json`, the `.trace.json.gz` files under
`items/result/traces/`, and (when `verify_ranks=1`, the default) it
**re-parses the traces from scratch** rather than trusting the manifest.

### `expect_ranks` — this run's value is **4**, and all three candidate values differ

**This is the clearest instance the round has produced of the mocked/real
variable class, and it is worth writing down because the three values are all
plausible and all present:**

| value | where it comes from | correct here? |
|---|---|---|
| **8** | the declared default, in **both** `m2_profiling.yaml:119` and the validator itself (`check.py:72` `args.get("expect_ranks", 8)`) | **no** |
| **2** | the **mock** value, and what m1's recorded `LAUNCH-LINE.txt` carries | **no** |
| **4** | this host's actual TP — m1 launched `--var tp=4 --var gpu_devices=0,1,2,3`, cards 0-3 at 3 % VRAM and 4-7 at 0 % | **yes** |

`m2_profiling.yaml:114-118` states the rule: *"a mock run passes
`--var expect_ranks=2`; a real run leaves it alone"* — and "leaves it alone" is
**only safe at TP 8**, which this is not. So it must be passed, as **4**.

**Pre-registered prediction, so that the verdict is a lookup:**

> If the bring-up took four cards, `check_trace_coverage` **passes** and prints
> `(note) 4 rank(s), <N> GPU kernel events`.
>
> If it refuses with **`expected 4 rank(s), the manifest lists 2`** the fault is
> my launch line, not m2's capture.
> If it refuses with **`expected 8 rank(s), the manifest lists 4`** I failed to
> pass the variable at all and took the default.
> **Neither is a producer defect, and neither is a reason to touch the bar.**

Note the first cluster hit `expected 4 rank(s), the manifest lists 2` — the same
sentence with the numbers swapped — and it went unseen for hours because
validation was off. **Here validation is on, so this fires in seconds.**

**Other refusals and what they would mean:**

| refusal | meaning |
|---|---|
| `<n> trace file(s) on disk against <m> in the manifest` | the manifest describes a capture the directory does not contain — producer defect |
| `the stack window carries <n> rank file(s), expected <m>` (`check.py:248`) | the **stack** window, governed by `stack_ranks` (default 2), not `expect_ranks`. A separate variable; do not "fix" it by changing `expect_ranks` |

**A pass establishes:** four readable per-rank traces exist, and — because
`verify_ranks=1` — a **re-parse from the trace bytes agrees with the manifest**.
That is two genuinely different methods (a count and a recomputation), which is
the strongest evidence shape available here.
**A pass does not establish** that the captured window contains the work we
care about; a window opened on an idle scheduler is a valid trace of nothing,
and only the `min_gpu_kernels_per_rank` floor (default 1000) speaks to that.

---

## 2. `check_bench_result` — on `profiling_mode_off.bench_result` AND `profiling_mode_on.bench_result` (×2)

**Reads:** `items/result/…` AIPerf exports, `summary.json`, `items/logs/`, and
validates the export against a named JSON schema.

**Refusals and meanings:**

| refusal | meaning |
|---|---|
| `items/<x>/<y> is missing` / `is empty` | the load produced nothing — bring-up or AIPerf failure, not a grading question |
| `the export does not validate against the '<name>' schema` | AIPerf wrote a shape this package does not know — **suspect an AIPerf version drift before suspecting the producer** |
| `the replay sent <n> request(s), want at least <m>` | `min_requests`, default **50**. **Counted, not assumed: my trace carries 437 records with `timestamp < 60000`**, so at `trace_end_ms=60000` this floor is cleared by ~9×. **If it binds anyway, the load did not reach the router** — that is a deployment fact, not a floor to lower |
| `the run was cancelled — the export is a partial window` | a timeout or a hand-stop, not a defect |
| `the load ran with streaming off, so this export carries no TTFT` | `--streaming` was dropped from the aiperf invocation |

**Evidence of a real read:** `(note) <total> request record(s), <errored>
errored (<rate>)`. **A `check_bench_result` pass with no such note beside it
should be treated as unproven** until the zone's file count is checked.

**A pass establishes** an AIPerf export exists, validates against the schema,
carries at least `min_requests` requests, and was streamed.
**It does not establish** that the numbers are comparable to anything: it grades
the export's shape, not its magnitude, and nothing here reads the workload.

---

## 3. `check_command_parses` — on the three bench/profile kinds and on `profiling_evidence`

**Reads:** `items/command` (and `items/script`), and parses it **with the shell
its own shebang names** — not with `sh`. The distinction is load-bearing and
documented at `check.py:53-63`: an unterminated quote is tolerated by dash and
rejected by bash, so a script can be broken under the shell it declares and fine
under the shell that runs it.

**A refusal means** the recorded command is not a runnable script — which is
about reproducibility, not about whether the measurement happened.
**A pass establishes** that one recorded script parses. It is a thin end-to-end
claim and should not be quoted as evidence that anything ran.

*History, not a claim about here:* this is the validator whose bar was
questioned on the first cluster as "too thin for end-to-end", and the criterion
was deliberately **not** raised while looking at the first data point. Same rule
applies now.

---

## 4. `check_kernel_table` — on `profiling_mode_on.kernel_table`

**Reads:** `items/text.json`, `items/schema`, `items/table.csv`.

| refusal | meaning |
|---|---|
| `items/text.json is missing — the CSV is the export but this is the …` | the structured half was not written |
| `items/schema is not byte-identical to the package's …` | the handoff carries a schema that has drifted from the package's copy |
| `table.csv has no column '<c>' (header: …)` | column drift in the ranking export |
| `table.csv has <n> data row(s), floor is <min_rows>` | `kernel_table_min_rows`, default **20** (`steps/common.yaml`) |
| `the '% Total' column sums to <x>, want <lo>..<hi>` | the shares do not add up — an arithmetic self-check on the producer's own output |

**The `% Total` rule is the one worth watching**, because it is the only rule
here that could not be satisfied by copying files: it recomputes a sum from the
table's own contents.

**Pre-registered link to the workload, so it is a lookup and not a rediscovery
(the leader's reopen condition, restated here where it fires):**

> My trace has a **theoretical prefix hit rate of 0.81**, so prefill is largely
> radix-served and the profile is decode-dominated. **If `check_kernel_table`
> refuses on `min_rows`, or m3's `kernel_worklist` comes out thin, the workload
> shape is the FIRST thing to re-examine** — a decode-dominated window may not
> surface 20 distinct kernels. This is **not** a prediction that it will; it is a
> pointer so nobody re-derives the connection at the moment it matters.
> Regenerating is one command
> (`tools/gen_conversation_trace.py --sessions-per-s … --turns-max …`).

---

## 5. `check_profiling_evidence` — on the merged `profiling_evidence`

**Reads:** `items/env/parts.json` and the per-part directories under
`items/result/<name>/`.

Its refusals are almost all about **the merge's bookkeeping**, not the
measurements:

| refusal | meaning |
|---|---|
| `items/result/<name>/ is missing` / `holds no files` | a declared part did not arrive |
| `items/env/parts.json is missing — it is what says which handoff each part …` | the merge did not record its own provenance |
| `parts.json does not account for <missing>` | the merge dropped a part silently |
| `parts.json claims part(s) <extra> that this kind does not declare` | the merge invented one |
| `parts.json lists '<n>' twice` / `row '<n>' has no '<field>'` | malformed bookkeeping |

**A pass establishes** that four parts arrived, are non-empty, and are each
accounted for by name. **It does not establish anything about their contents** —
those were graded upstream by §1, §2 and §4, and this validator does not re-open
them.

⚠ **This is the entry most exposed to the zero-file-zone failure**, because
almost every one of its refusals is of the form *"X is missing"* — precisely the
shape that an empty `materials/` produces and that cannot be distinguished from
a real absence by reading the report. **For this validator the file-count check
in §0 is not optional.**

---

## 6. `check_environment` — on all five m2 kinds (not in the assigned list, included because it fires)

Grades the `environment.yaml` each kind carries and cross-checks fields
**across every handoff in the phase**, refusing when two of them describe
different machines.

**Its known refusal shape is pure absence** — `no environment.yaml at any of
[...]` — which by §0's discriminator is **exactly the kind that cannot be
distinguished from a zero-file zone by reading it.** Any
`check_environment` refusal on m2 gets the file-count check before it is
attributed to anything.

*History, not a claim about here:* on the first cluster this validator could
**never** pass for `integration_report`, because `compare.py` never wrote the
record at all — and that was invisible for the whole ladder because every rung's
stage 5 was mocked and the mock rendered the file itself. That is a fact about
m5, recorded here only because it is the reason a `check_environment` pass is
worth reading carefully rather than assuming.

---

## 7. Current state — dated, and deliberately not to be updated in place

**As of 2026-09-06T06:59:38Z, on this cluster:**

> **Every validator in §1–§6 has been invoked ZERO times on this cluster.**
> Zero passes, zero refusals, zero verdicts. No m2 kind exists here.

First-cluster history, **marked as history and not transferable**: m2's four
kinds sealed `valid` in three separate mock runs and 14 of the 21 verdicts in
run `072849` were m2's — **but those were mock runs against a sealed corpus that
does not exist on this cluster**, so they establish nothing about what will
happen here beyond "these validators have run at all".

The one first-cluster event that *is* directly relevant is the
`check_trace_coverage` refusal quoted in §1, because it is the same validator,
the same variable, and the same failure available to me today.

---

## Corrections — append only, newest last

### 2026-09-06T07:16:42Z — m2 captures ONCE and nothing grades its spread

Asked by the leader before any result: *what controls the number of bench rounds
m2 actually captures, and does anything grade m2's dispersion at all?*

**Plain answer: there is no rounds control for m2, and nothing grades its
dispersion.** Three reads, all first-hand:

1. `assets/load/line.sh:392` — `E2E_LOAD_ROUND="$MODE"`. **The "round" is the
   mode name (`off` / `on`), a label, not a repetition counter.**
2. **No repeat loop exists anywhere in m2's path.** `grep -nE 'for .*seq 1|for
   r in|ROUNDS' assets/load/line.sh assets/analyze/scan.sh` returns **nothing**
   (rc=1). Compare `assets/accept/measure.sh`, m5's path, which does have
   `for r in $(seq 1 "$ROUNDS")`.
3. **`check_bench_result` contains no dispersion vocabulary at all** — zero hits
   for `stddev|std_dev|spread|dispersion|variance|rsd|round`. It grades one
   export's shape: files present, schema valid, request count above
   `min_requests`, streaming on. **m5's `check_bench_report` is round-aware by
   contrast** — it iterates `round_dir` and its header opens *"Every replay
   round produced a complete AIPerf export"*.

**`--var bench_rounds=3` does not change this**: it is declared only in
`m5_integration.yaml:364`, so m2 never reads it. m2 will capture exactly one
load per mode however it is set.

### What this limits, stated before the number lands

The noise floors this round leans on (4.4 % / 0.76 % / 2.7 %) were measured
across **three** rounds. `check_no_regression`'s `stock_vs_m2` compares m5's
stock arm against m2's bench — and per m35, on m5's side it reads
`.../r1/profile_export_aiperf.json` and `sorted(glob(...))[0]` on m2's, so
**the comparison consumes one round from each side regardless of
`bench_rounds`.**

> **So the floors describe the spread of a measurement process observed over
> three repetitions, while the comparison that uses them consumes a single
> observation from each side. And on m2's side there is nothing to repeat:
> one capture exists, and no validator would notice if it were an outlier.**

**Consequence for reading a `stock_vs_m2` residual:** a residual inside the
floor is consistent with "no regression" **and equally consistent with one
unlucky capture on either side**, because nothing here can tell those apart.
That is a limit on what our `check_no_regression` result can mean. It is **not**
a reason to widen anything — `compare.py:300-305` argues against that
explicitly — and it is **not** a defect in m2, which was never designed to
repeat. It is a property of the comparison, recorded now so that it cannot be
discovered in the direction that flatters whatever number arrives.

**Nothing above changes §1-§7.** No bar moved, no prediction rewritten, no
count edited; this is an added limitation, not a revision.


### 2026-09-06T07:34:07Z — reading rule: m4's operator is READ from the artefact, never assumed

**Chain scope, not m2 scope** — recorded here because this is the document that
gets opened when results land, and the rule has to be present at the moment the
artefact is read rather than in the file that explains the launch.

`workset_operator` **cannot be supplied in advance in a single-run chain**: its
correct value is the operator name **m3 produces inside the same run**. Omitted,
`optimize_kernel.task/steps/10_read_inputs.py:32` takes
`--operator=os.environ.get("E2E_WORKSET_OPERATOR") or None` and the code picks.
That is the documented silent-first-match hazard, and here **absence is not a
lapse — it is the only option the topology allows.**

> **Rule: when m4's `kernel_optimization` lands, the operator it worked on is
> read out of the artefact. It is never inferred from the worklist order, never
> from what m3 "should" have ranked first, and never from a launch variable —
> because no launch variable could have carried it.**

If m3 yields more than one operator, **which one m4 optimised was chosen by
nobody**, and the artefact is the only record of the choice. A report that names
the operator without having opened the artefact is asserting the thing this rule
exists to prevent.

*(m35's P7 filter approaches the same seam from the other side.)*

### 2026-09-06T08:42:27Z — the pre-registered instrument was right and my improvisation was misleading

**This document caught its own author, and that is the strongest justification for
it the round has produced.**

§0 names one tool: `bash assets/lib/refusal_saw_something.sh <run dir>`. When
the first two verdicts landed I ran it **and** an ad-hoc check, on the number
that decides whether a pass means anything:

```
refusal_saw_something.sh                        ->  1 zone, files: 30, none empty
find $zone/materials -type f | wc -l            ->  0 files
```

**The `0` was mine and it was wrong.** These zones carry no `materials/`
*directory* — they carry a `materials.json` **file** naming a shared materials
root one level up. My `find` was handed a path that does not exist and **failed
to zero**. Resolved by reading `materials.json` and counting at the path it
names: **30 files**, agreeing with the tool.

Three things worth keeping:

1. **It is bug record entry 4b, which I wrote this morning, reproduced by me two
   hours later** — on the one check whose whole purpose is to stop a green being
   over-credited.
2. **It failed toward the expensive conclusion.** The improvisation argued
   *"the green is worthless, the zone was empty"*. A false alarm that demands
   the costly action is not the safe direction to fail in.
3. **Naming the tool is what saved it.** Had §0 said "check the zone is not
   empty", I would have written that `find` and believed it. The rule
   generalises: **a pre-registration should name the instrument, not describe
   the intent** — an intent can be satisfied by an instrument that answers a
   different question, and this one did.

*Corollary for the rest of this run: when the pre-registered tool and an
improvised check disagree, the improvisation is the one to debug first.*

### 2026-09-06T12:20:25Z — `stack_window_s=0`: a THIRD pre-registered explanation for a thin worklist

Written **before m3 runs**, so that if the `kernel_worklist` comes out thin it is a
lookup and not an investigation.

Run `20260906T113811-fdb0bd` sealed `run_profiling_mode_off` and then had `_on`
refused by two validators, both quoting numbers computed from files (39-file
materials, not an empty zone):

```
check_trace_coverage  PROBLEM: items/result/stacks_manifest.json is missing - the round
                      was asked for a stack window and this handoff carries none.
                      Set --var stack_window_s=0 to say that is intended
check_kernel_table    PROBLEM: no launcher frames were resolved ... this round wanted at
                      least 10 in the head. Set --var stack_window_s=0 and
                      min_launchers_in_top_n to 0
```

Both are the **mocked/real variable class** again: `stack_window_s` defaults to `3`,
which is right for a round that wants stack attribution and wrong for one that does
not. The launch line now carries `--var stack_window_s=0 --var
kernel_table_min_launchers=0`.

> **Consequence, registered in advance: with no stack window there are no launcher
> frames, so m3 loses the launcher-attribution signal entirely.**

**That is now the THIRD pre-registered explanation for a thin or empty
`kernel_worklist`**, and they are independent:

| # | explanation | registered |
|---|---|---|
| 1 | workload prefix hit rate — decode-dominated profile may not surface 20 distinct kernels | materials/README §5 |
| 2 | `identify` resolves nothing — `E2E_SGLANG_SRC`/`E2E_AITER_SRC` unwired, `min_resolve_ratio=0.0` so it will not refuse | bug record entry 1 |
| 3 | **no launcher frames, by our own declaration** | here |

**If the worklist is thin, check 3 first** — it is the one we caused deliberately today
and the only one whose cause is a flag on the launch line.

### This is a declaration of intent, not a weakened bar

We are **not** choosing a thinner profile to make a validator pass. `mission.md` asks
for 跑通即可 — reachability, not a real optimisation — and these two validators were
refusing precisely because the round *had not said so*. The flag is the package's own
way of saying it: `shared.yaml:171` documents that `min_launchers_in_top_n` must be `0`
whenever `stack_window_s` is, so the pair moves together by design.

**One naming trap recorded, because it would have cost a full run:**
`min_launchers_in_top_n` is the **validator args field** (`common.yaml:162`), fed by
`${kernel_table_min_launchers:-10}`. The refusal text names the args field. Passing
`--var min_launchers_in_top_n=0` is accepted **silently** by `show` — unrecognised
`--var` names are listed as supplied and ignored — so it would have loaded clean,
launched clean, and refused again for the same reason.

### 2026-09-06T14:09:01Z — Path A, no waivers: outcomes fixed before the result exists

Run `20260906T140831-026b96`, pid 3849649. **`--var trace_end_ms=120000`, and
NONE of `stack_window_s` / `stack_ranks` / `kernel_table_min_launchers`** —
verified absent from `/proc/3849649/cmdline`.

**Why no waivers:** each of those three tells a validator not to look rather than
giving it something to look at. A successful stack capture is what *produces*
launcher frames, so with `trace_end_ms=120000` the default
`min_launchers_in_top_n: 10` becomes answerable **on evidence for the first
time**. Keeping the waiver would have meant the one thing this launch is for
could not be observed through the check most sensitive to it.

**Current board, written down so it cannot be quietly improved or quietly
excused: 10 of 11 true** (run `298750`). This run risks the
`check_kernel_table` pass, deliberately — that pass came from a waiver we
granted ourselves.

**Pre-named test, reported before any verdict:**
`grep CAPTURE_OK <run>/…/capture_stacks.log`

| outcome | pre-registered meaning |
|---|---|
| `CAPTURE_OK` **and** `check_kernel_table` true | Path A works end to end; both of today's declarations were unnecessary. Strongest single result of the effort. |
| `CAPTURE_OK` **and** `check_kernel_table` false | The capture works; attribution does not reach 10 of the top 25. **A real finding about attribution**, asked on evidence for the first time. **Not a producer defect until read.** |
| no `CAPTURE_OK` | `trace_end_ms=120000` is insufficient. m35's caveat becomes live: the floor arithmetic compares *requested* durations and ignores inter-capture setup; ~31 s of slack was needed on the measured run. |
| `check_trace_coverage` still false | **Read its args.** If it names `expect_stack_ranks` again, the manifest is still absent — the capture did not produce what the validator wants, which is different from the capture not running. |

**None of these outcomes is a reason to restore a waiver in the same breath.**
If `check_kernel_table` refuses, restoring `kernel_table_min_launchers=0` in a
*second* launch tells us exactly what the flag buys — that is a measurement.
Restoring it to recover a green is not.

### 2026-09-06T14:15:21Z — CORRECTION: the entry above names a run I was not driving

**The 14:09:01Z entry says the pre-registration applies to
`20260906T140831-026b96`. That is m35's run, not mine.**

```
3849649 -> runs/20260906T140819-2f9956   container=yihou_e2e_chain    jobid=29184   MINE
3856205 -> runs/20260906T140831-026b96   container=yihou_e2e_chain2   jobid=29313   m35's
```

Mine was killed by the leader at 14:10:35 — it lacked `--timeout 21600` and
carried `jobid=29184`, a hold that ended at 14:00. **Nothing had brought up, so
nothing was lost but five minutes.**

**How I got the wrong run: I read `agent_sys_runroot/latest`.** That symlink
points at the *newest* run, and m35's was created **sixteen seconds after**
mine. I then "confirmed" it with `find … -newermt '3 minutes ago'` — **which
selects on recency as well.** Two checks, one failure mode, so they agreed and
told me nothing. *That is ownership-by-adjacency, the class I have catalogued
four times today, committed against my own run.*

> **The discriminator that works: the orchestrator's own `/proc/<pid>/cmdline`,
> or an agent's `readlink /proc/<pid>/cwd`.** Both name a run directory the
> process is actually in. `latest` names whatever started last, which is a fact
> about the clock and not about ownership.

### What the pre-registered table applies to

**The four outcomes stand — they are about Path A, not about who launched it —
but they apply to `20260906T140831-026b96`, and one row must be read
differently:**

> **`check_kernel_table` on that run carries `--var kernel_table_min_launchers=0`.**
> m35 kept the waiver deliberately, so that its result is comparable with the
> earlier run. **Whatever it returns, it is a waiver result and not attributable
> to evidence.** The row that reads *"attribution does not reach 10 of the top
> 25 — a real finding, asked on evidence for the first time"* **does not apply
> to this run.** That question is still open and needs a launch without the flag.

`CAPTURE_OK` is unaffected: it is a producer outcome and `trace_end_ms=120000`
is the only variable bearing on it. **The main question of Path A is answerable
on this run; the secondary one is not.**

**And the clause from the entry above stands unchanged and now matters more:**
restoring a waiver in a *second* launch measures what the flag buys; restoring
it to recover a green does not. **The reverse also holds — a waiver already in
place does not become evidence because the run around it succeeded.**

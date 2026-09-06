# Reproduce, debug and accept the llm e2e opt chain on a second cluster

### Scope

`agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/` — one task package
whose graph runs module 1 → 5 in a **single `agent-sys run`**. No `agent_sys` core
changes. The first round was driven on one cluster; this round is about whether the
result is a property of the package or a property of that cluster.

**Nothing in this document names a host, a job id, a scheduler, or a shared
filesystem path.** Every locality the first round relied on is listed in §Local as a
thing you must supply, not a thing you can inherit.

---

## Where the first round got to

**Six real chain runs. Best board: seventeen validators invoked, sixteen pass, one
refusal.** Read this table as the starting point to reproduce, not as a claim to
trust.

| module | real or replayed in the best run | its validators |
|---|---|---|
| 1 deploy | **real** | `check_deploy_kit`, `check_environment` — pass |
| 2 profiling | **real** | `check_profiling_evidence`, `check_bench_result` ×2, `check_command_parses`, `check_trace_coverage`, `check_kernel_table` — pass |
| 3 analysis | replayed | `check_worklist_shape`, `check_identity_resolved` — pass |
| 4 kernel_opt | replayed | `check_optimization_shape`, `check_overlay_applies` — pass |
| 5 integration | **real** | `check_acceptance`, `check_bench_report`, `check_measurement_order`, `check_patch_live` — pass |
| 5 integration | **real** | `check_environment` pass, **`check_no_regression` FAIL** |

**Three caveats that belong inside the number, not under it:**

1. **One of the seventeen is `check_nothing`** — a stub that returns true, reads
   nothing and is `strength: weak`. It passes by construction. The honest count is
   **fifteen real validators passing, one stub, one refusal.**
2. **Modules 3 and 4 were replayed**, so their passes are against a sealed corpus,
   not against artefacts a live stage produced in that run.
3. **`packup` was never reached in any run.** The one refusal marks
   `integration_report` invalid, and the framework marks the task's whole output set
   invalid with it, so two passing siblings go down too. **A ledger that counts
   handoff states rather than verdicts will report three refusals where there is
   one.**

### What is established, and how it was checked

- **A real `integration_report` carrying a real `items/env/environment.yaml` passes
  `check_environment`.** Four earlier rounds failed to settle this for four
  unrelated reasons. Checked to the refuting standard: the deciding zone held **149
  files**, and the record inside names the node and container of that run.
- **`check_measurement_order` passes on both arms.** It had never been exercised.
- **A real module 2 removes two thirds of the `check_no_regression` gap.** With
  module 2 replayed the stock arm missed the recorded bench by **-26.1 %** and
  **-29.3 %** on two independent nights; with module 2 real, on the same machine in
  the same hour, `inter_token_latency` came inside the 10 % bar and only
  `time_to_first_token` remained, at **-11.2 %**.
- **The residual is not noise.** Measured floors at `bench_rounds=3`:
  `ttft_ms` **4.4 %**, `inter_token_latency_ms` **0.76 %**, `request_latency_ms`
  **2.7 %** — the instrument is about twice as tight as the bar.

### What is NOT established

- **Module 4 has never completed a real campaign.** Every run replayed it. One
  attempt spent 113 minutes still in preparation with no source file changed.
  **Running it through is a first-class goal of this round** and is the main reason
  a cluster with longer holds is worth the move.
- **No chain has completed all five stages.** `packup` is unreached.
- **`check_no_regression` under clean comparability.** The surviving explanation for
  the 11.2 % is *one machine in two states* — module 2 runs a profiled capture and
  two bring-ups sit between it and module 5's stock arm. **No launch variable
  addresses this; it is a measurement-design question.**
- **A real module 3 cannot currently feed a passing `apply_patch`.** Checked across
  all five operators in a real module-3 workset: **every baseline is
  harness-shaped** — it defines `run(...)` for the measurement harness and none of
  the `public_symbol` it declares. A harness cannot be an overlay, so
  `forge_mock=1` plus a real module 3 are jointly incompatible with a passing apply,
  **for every operator, structurally.** The only route is a real campaign emitting an
  engine-shaped module — which loops back to module 4.

---

## Brief

1. **Reproduce.** Stand the package up on the new cluster and reach the same board:
   modules 1, 2, 5 real, 3 and 4 replayed, fifteen real validators passing.
   **A different result here is the most valuable outcome of this round** — it means
   the first was cluster-shaped.
2. **Run module 4 for real, end to end**, and carry its output into module 5. This is
   the one stage the first round never completed.
3. **Complete all five stages in one run, reaching `packup`.**
4. **Debug what breaks**, and record it. The defect list below is what to expect, not
   what exists.

## Category

experiment & hardening

## Goal

### Finish Standard

1. One `agent-sys run` walks modules 1 → 5 and reaches `packup`.
2. Module 4 produces an artefact from a real campaign, and module 5 consumes it.
3. Every refusal is attributed **after** running the materials check below, so no
   refusal is charged to a producer that was handed an empty directory.
4. Every claim names a file to open and a condition that fails. **Read the artefact,
   not the exit code.**

### Task breakdown

Rung by rung, one stage promoted from replayed to real at a time, so a failure is
attributable. The first round's order was 1 → 2 → 5 → (3, 4). **Module 4 is the
expensive one; put it last and reuse its output once obtained.**

---

## Known defects and traps — expect these, do not rediscover them

### Framework, not ours

1. **One validation zone per run is handed ZERO files, and it is always the same
   one.** **Seven real runs, seven results of exactly one empty zone**, and in the six
   resolved so far the zone belongs to **module 2's closure** (`m2_profiling`, the
   task carrying `deploy_kit` + `profiling_evidence`). **It survives module 2 replayed
   vs real, module 5 replayed vs real, both halves of a node, and different validator
   subsets.** It was mistaken for a ~1-in-13 *rate* for a whole night because every
   report gave a count; **exactly one per run is inconsistent with an independent
   per-zone probability**, which would give some runs zero and some two.
   **Established: deterministic for that closure. NOT established: that only that
   closure can be affected** — every run in the sample has module 2 in the graph and
   upstream of live work. **The cheap test nobody has run is a graph without it.**
   **Untested shape hypothesis:** that closure carries **two kinds** and the only
   stage in the graph that gathers from four producers sits immediately upstream of
   it. **Nobody has read the staging code.**
   Consequences in increasing severity: a false refusal that reads like a producer
   defect; **a silent pass on nothing**; and — measured once — **the death of a
   healthy run**, when a spurious refusal on a replayed upstream stage escalated to a
   sink that cannot answer and a 900 s timer ended the run while module 5 was writing
   its report.
   **Mechanism unknown and deliberately left open.** A controlled sample of 80 staged
   handoffs on a mock loop produced **zero** empties while reproducing the *shape*
   (`v0` empty, `v1` populated) at 30 % and resolving it correctly every time — so
   "an empty `v0` confuses staging" is refuted. **The version number is noise; the
   file count is the discriminator.** **Caution on that 0-in-80:** those runs all
   stalled early and may never have staged the same zone set, so it and the real-path
   number were probably never the same population. **Do not treat them as a
   contradiction requiring explanation until the zone counts are compared.**
   **The zone directory is named `validation.<TASK-ID>.<phase>.<hash>` — the owning
   task is IN the name, and the store maps it to a closure. Resolve the identity, do
   not report a count.**
   **Before attributing any refusal:**
   ```sh
   bash assets/lib/refusal_saw_something.sh <run dir>
   ```
   **`bash`, not `sh`** — the script uses `set -o pipefail` and process substitution.
   It reports per-zone file counts. **A refusal from a zero-file zone says nothing
   about the artefact.**
   **A PASS establishes that the validator was invoked, not that it saw anything** —
   a pass carries no reason, so the file count is the only retrospective check on it.
   **Consequence on the first cluster: every verdict that project recorded for the
   validator on that kind is void — both of its refusals and all of its passes.**
   Expect the same to be true of whichever kind it lands on for you.

2. **An escalation with no receiver leaves a task at `running` and the log looking
   healthy.** Two flavours, and the event type tells them apart without opening
   anything else: a **program** body that exits non-zero has no agent to instruct; an
   **ai** body whose `mainloop` already returned has nowhere to deliver. Either way
   the phase line still says `running` until a stall timer fires. **Observed cost: 15
   minutes of hold spent after the body had already exited, with the reason sitting
   in the event's `attributes.detail` the whole time.**

3. **A sibling failure invalidates passing handoffs.** See the ledger note above.

### Ours, already paid for

4. **The launch line is not recoverable from the artefact.** The staged package keeps
   `${var:-default}` unrendered, so which `--var` a run used cannot be read back.
   Four separate incidents trace to this. **Record your launch line beside the run.**

5. **Variables that have one value when a stage is mocked and another when it is
   real.** At minimum `expect_ranks` (the replayed corpus may be a different TP than
   your bring-up), `adhoc_cases`, `bench_rounds`. **Audit the whole table against
   your `mock_stages` every launch, not the rows you remember** — three launches were
   lost fixing them one at a time.

6. **Port bands.** The variable that moves the whole band is `E2E_KIT_PORT_BASE`; the
   per-service one moves a single port and leaves the rest of the band behind. On a
   shared host a collision produces a body `exit 1` with **no escalation recipient**,
   so it reads as a stall for the full timer. Treat a per-service port as claiming a
   **range**, and verify after bring-up by reading the kit's deployment record rather
   than trusting the launch line.

7. **A stale package copy fails silently.** The package is generated into a tree; a
   copy that is *not* regenerated goes on testing the code you had, not the code you
   have. One round refused for exactly this and would have been reported as "the fix
   did not work". **Make it a precondition: abort unless the generated tree's
   recorded source commit equals HEAD.**

8. **Module 5's arms are pinned to the first TP devices.** The device list is
   defaulted in the worker script and set nowhere, so no `--var` moves it. Two module
   5 stages also cannot share a host: two coordination ports are literals.
   **Consequence: any run whose `mock_stages` omits module 5 needs the host to
   itself.**

9. **Module 5 brings up its arms through a node-wide GPU reset** that kills every
   process holding a GPU whose command name matches the engine's, protecting only the
   scheduler's own daemon. **It does not protect a co-tenant, another line, or you.**
   If the host is shared, either do not run a real module 5 on it, or stop before
   that stage — **and treat stopping as a safety requirement rather than a scheduling
   convenience.**

10. **A guard that fires only on disagreement treats an absent value as agreement.**
    The environment consistency check covers 3 of 28 fields for this reason, and
    extending it field-by-field buys false comfort about the rest. **If you touch it,
    the decision is whether the absent case becomes loud for all fields.**

---

## Reference material

You need three things and **none of them should be taken from the first cluster's
filesystem**:

1. **The package**, from this repository, at the branch head.
2. **A container image** with the serving engine and the model, plus the model
   weights. Everything the package needs from the image is discovered at run time and
   recorded in the environment handoff; **the image digest is a launch variable, and
   it must be the digest actually present on your hosts** — a recorded digest from
   elsewhere is a provenance signal, not a configuration.
3. **A corpus of sealed handoffs, if you intend to replay any stage.** The first
   round's corpus was produced by running the five modules separately and sealing
   their outputs. **Two honest options on a new cluster:** produce your own by
   running each stage alone first, or carry the old one across and accept that every
   replayed stage's `check_environment` then describes a machine that is not yours.
   **Do not hand-write the missing parts** — a replayed artefact that claims
   measurements nobody took is worse than an absent one.

**`RUN-PLAN.md` and `WHAT-GREEN-ESTABLISHES.md` inside the package are the two
documents to read before the first launch.** The first is the canonical launch block
and its numbered preconditions; the second records which validators have ever refused
anything and what a pass from each is worth. **Both contain first-cluster specifics —
read them for mechanism, not for values.**

---

## Rules

**MUST**: the whole round follows these.

### General

1. **Use an agent team.** The leader runs a polling loop, records a problem on first
   sighting and intervenes on the second if the owner has not resolved it. **An owner
   with nothing assigned is the leader's failure, not theirs.**
2. **A checkpoint writer appends progress to a summary file on a fixed interval** —
   estimated completion, elapsed and projected time, reliability; current state;
   code problems fixed and unfixed; non-code problems; open questions; new commits
   with one line each.
3. **Research → gather → analyse → plan → workspace → work.** All temporary activity
   inside a dedicated scratch directory; the repository receives only the package and
   the notes.
4. **Work in English. Report to the user in Chinese.**
5. **Every commit is signed off.** CI rejects a commit without the trailer.

### Method — these were the expensive lessons, not stylistic preferences

6. **Pre-register.** Before a result arrives, write down what each outcome would
   mean and what the current state is. The first round's best single decision was
   fixing the grading criteria and the current distribution *before* six results
   landed, which turned each one into a lookup instead of a negotiation.
7. **Read the artefact that says what a task is doing, not the one that says it was
   scheduled.** A phase line answers "what was dispatched"; a transcript or the
   event's `attributes.detail` answers "what is it doing and why did it stop". **A
   run was destroyed thirty seconds before sealing because the cheaper artefact was
   read instead.**
8. **Before any irreversible action against a run, open that artefact.** Killing an
   orchestrator does **not** kill its agents, and an agent will rebuild a container
   about a minute after you stop it. **Order: agents first, then containers, then
   verify.**
9. **A tool that returns a well-formed answer on bad input is the dangerous kind.**
   `find` failing to zero, an empty variable expanding to a match-everything glob, a
   syntax check that cannot fail on an invalid option, a name-resolution question
   asked of a parser. **Prefer tools that error on bad input over tools that return
   empty, and before trusting a new instrument, run it on a case whose answer you
   already know.**
10. **Never let a computed value reach a destructive command without checking it is
    non-empty.** One unchecked variable killed four agents belonging to three people.
11. **Do not infer ownership from a name.** Container prefixes are shared. Ownership
    is a label, an auto-remove flag and a process list — answer it where the question
    is asked, not from memory.
12. **When relaying someone else's finding with a stronger scope than they claimed,
    first run the one reading that would falsify the stronger version.**
13. **Check self-corrections that make a factual claim.** A retraction is a claim
    pointing the other way and can be wrong in the same manner as the original.

### Local — supply these, do not inherit them

14. **Compute is reckoned as time remaining, not elapsed.** Before launching, compute
    whether the run fits in what is left. The first round lost rounds to this twice
    and had holds cancelled without warning three times.
15. **Deletion:** do not delete anything whose path does not contain your own
    identifier or a temporary directory — on hosts and inside mounted directories
    alike. Stop containers gracefully; never force-remove. Never run a cleaning
    operation against a shared run root.
16. **Do not request machines.** Query and use what is held; if there is not enough,
    say so.
17. **A foreign GPU workload on a host you hold:** stop it only if it stays stopped.
    **A job that respawns is not something you are enforcing against — it is a fight
    you are losing quietly.** Report it and leave it.
18. **Before a bring-up, read the cards on the host itself** — a declaration of which
    devices a line wants is not an observation of which are occupied, and the two
    were measured disagreeing. **Ask two questions in this order: is there a live
    chain on this host, and are the cards busy. The first can veto; the second
    cannot, because a chain in a CPU stage holds nothing and will want the cards
    back.**

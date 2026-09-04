# Promoting the flow from mock to real, one stage at a time

Mission Brief item 3. Written before the mock run went green, so that the order
is decided by argument rather than by whatever is convenient when it does.

**The rule the whole plan rests on: promote one stage per run.** A run with two
stages newly real cannot attribute a failure to either. That costs more runs and
it is the only thing that makes a failure mean something.

## The ladder

| rung | `--var mock_stages=` | what becomes real | what it proves that the rung below did not |
|---|---|---|---|
| 0 | `all` | nothing | the graph, the seals, 21 validators, the handoff wiring |
| 1 | `m2,m3,m4,m5` | **m1** | a real bring-up; `check_deploy_serves` against an engine rather than the stub |
| 2 | `m3,m4,m5` | **+m2** | a real profile and a real bench, and that m3's readers read what m2 actually writes |
| 3 | `m4,m5` | **+m3** | `build_workset`'s **AI** step, and `check_workset_runs` re-measuring on the real workset |
| 4 | `m5` | **+m4** | KernelForge, and the M4.3.5 reversal against a real baseline |
| 5 | `none` | **+m5** | the two arms, and `check_no_regression` on numbers nobody chose |

Each rung is a separate `agent-sys run`. Nothing skips.

### The vars that change with the rung, and they are not only the agent ones

The table above says which stage becomes real. **That is not the same as which
`--var`s change**, and the gap would have cost a GPU run — caught by m2, who had
been telling everyone to pass the wrong one.

| var | rungs 0–1 | rung 2 onward |
|---|---|---|
| `m<N>_agent=runner` | one per still-mocked stage | **removed** for the promoted stage |
| `expect_ranks` | **2** | **omit it** (defaults to 8), or track `--var tp` |
| `adhoc_cases` | **0** | omit from rung 5 (`todo.md` T12) |
| `image` | **the sealed kit's**, not the node's | the real bring-up's |
| `transport_env` | **required on every rung** — `SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR` | same |

**`image` and `transport_env` are on this table because each cost a run, and
they fail in opposite ways.**

`image` is a fact about **the artefact being graded**, exactly like
`expect_ranks` — m2's framing, and it is the one that makes this list
predictable rather than a list of remembered mistakes. Passing a tag that exists
on the node instead of the one the sealed kit renders makes `check_deploy_kit`
refuse: `environment.md` is a rendering of the record and the two must not
disagree. **The validator is right and the refusal is loud**, which is the good
version of this mistake.

`transport_env` is the bad version. **A validator declares no agent, so the
package's `env` block never reaches it** — `check_deploy_serves`'s own header
says so, and records a previous run lost to it. With `transport_env` unset,
`spur` has no `SPUR_CONTROLLER_ADDR`, `deploy.sh` dies with *"failed to connect
to controller"*, and the validator refuses **in one second**. It looks exactly
like a deployment failure. It cost three rung-0 runs and two wrong attributions
— to GPU contention with a live deployment, then to a missing `local` branch in
`remote.sh` — because the diagnostic that names the cause goes to **stdout, and
nothing in a run keeps a validator's stdout** (`temp/bugs/2026-09-03-a-validators-
stdout-is-not-kept-anywhere.md`).

The correct incantation was written in a comment in `steps/m1_deploy.yaml:128`
the whole time. **A parameter documented next to its declaration is not
documented to the person composing a command line**, which is what this table is
for.

**`expect_ranks` is a fact about the artefact, but which artefact depends on the
node — so the table's "omit it" advice is wrong on a partly-occupied host.**
m1's, 2026-09-04, from measuring `crsuse2-m2m-249`: GPUs 0–3 are held by another
tenant (~300 GB each, no docker container behind them), leaving four cards. So
m1 brings up at `tp=4`, and at **rung 2** — where the trace comes from *that*
bring-up rather than from the sealed TP-2 capture — `expect_ranks` must be **4**.
Omitting it defaults to 8 and `check_trace_coverage` refuses a perfectly good
four-rank capture **after a full bring-up and a three-minute load**.

Rung 1 is unaffected: m2 is still replaying the sealed TP-2 artefact there, so
`expect_ranks=2` stays. The var belongs to m2; **the node is what changes it**,
which is why it is recorded here rather than left to be rediscovered.

**There is no `--var` that names a GPU *set*, only a count.** `E2E_TP` gives the
number and the index is left to the agent's `rocm-smi` read at step 1. On a
partly-occupied host that is the largest behavioural risk at rung 1 — an agent
taking the default devices takes 0–3 and OOMs against a co-tenant. m1 routed
"take only 4–7" through `E2E_INSTRUCTION`, which is the declared channel for a
site fact and the right refusal to change the package mid-rung. It is recorded
in `todo.md` as a gap rather than a solution: **a site fact carried in prose is a
site fact nothing validates.**

`expect_ranks` is the one to watch: it is a fact about **the artefact being
graded**, not about the run. The sealed capture is TP-2, so rungs 0 and 1 need
`2`; at rung 2 the trace comes from the real bring-up, and passing `2` makes
`check_trace_coverage` refuse a perfectly good capture **after a full bring-up
and a three-minute load**. It fails loudly, so it costs a run rather than a
wrong number — which is the good version of this mistake and still a run.

***Corrected 2026-09-04 by m2, the var's owner: this paragraph said "a real TP-8
bring-up" and the sentence above it, added the same day, said 4.* The document
contradicted itself about the one number it exists to get right, and both
spellings were written as facts. Neither is: **the rung-2 value is whatever
`tp_size` the rung-1 kit recorded**, because m1 sizes the bring-up from the
cards that were free — `env.sh:230 _pick_gpus` takes every free card — and that
is a property of the node on the day. 249 left four free and gave `tp_size: 4`;
235 was measured 8/8 free. **Do not carry a number between rungs or between
nodes; read it.** The command below does.*

***Corrected again 2026-09-04 by m1, whose finding the sentence above cites.
`env.sh:230 _pick_gpus` is real, and it is a property of **one kit**, not of this
stage.*** It was in the kit the 06:24 run's agent wrote. **The 07:16 run's agent
wrote a different kit with no `_pick_gpus` at all** — `env.sh:62` is a literal
`: "${E2E_KIT_GPU_DEVICES:=0,1,2,3}"`. Two agents, two kits, **two different
device policies, and nothing in the package validates either**. So the device
policy is not something this document can state as a fact about m1; it is
whatever that day's agent wrote, and the only way to know is to open the kit.

The rest of m2's correction stands and is strengthened by this: **do not carry a
number between rungs or between nodes — read it from the kit and the record.**
That now applies to *which* cards as well as how many.

**And it was the literal `0,1,2,3` that caused the 07:29 incident**, not a land
grab: the agent probed the node at transcript record 92, then hardcoded the
default anyway, and `deploy.sh` preflights **ports and container names and not
cards** (`deploy.sh:61`: *"preflight ok: ports free, names free"*). It bound four
cards a co-tenant was loading a model onto. See `todo.md` **T27**, item 4.

## The rung-2 launch line, as a whole command

**Written out rather than left as a delta, because every launch-line failure
today came from someone assembling a command from the table above** — and the
table is a diff, which is the one shape that cannot be pasted. Owner: m2, since
the var that changes is `expect_ranks`.

**First read the number from the artefact.** `expect_ranks` describes the
capture m2 is about to take, which is sized by the deployment rung 1 recorded:

```sh
RUN=<the rung-1 run directory under /home/yihou/agent_sys_runroot/runs/>
grep -E '^  (tp_size|node|image):' \
  "$(find "$RUN" -path '*items/codes/environment.yaml' | head -1)"
```

Then the run, with `tp_size`'s value as `expect_ranks` and `image` as `image`:

```sh
python3 -m agent_sys.cli.main run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /home/yihou/agent_sys_runroot \
  --var jobid=<the hold> --var node=<the node> --var node_ip=<its IP> \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=<the image the rung-1 kit records> \
  --var mock_stages=m3,m4,m5 \
  --var m3_agent=runner --var m4_agent=runner --var m5_agent=runner \
  --var expect_ranks=<the kit's tp_size> \
  --var adhoc_cases=0 \
  --var transport_env=SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR
```

**Four things about it that are easy to get wrong and each cost a run once:**

- **`m2_agent=runner` is absent, and that absence is the promotion.** A rung is
  promoted by *removing* a var, so a rung-2 command that still carries it is a
  rung-1 command that looks like a rung-2 command and will report a clean mock.
- **`m3/m4/m5_agent=runner` all stay.** m3 is `kind: ai`; drop its var and a
  model gets called at rung 2.
- **`transport_env` on every rung**, and it must expand — `$SPUR_CONTROLLER_ADDR`
  is set in the login shell (`http://crs-m2m-cpu-spur-005…:6817`). Unset, the
  refusal arrives in one second and reads exactly like a deployment failure.
- **`--demo-root`**, since `/shared_nfs` is mounted `ro` on the login node.

**Verified to load, not merely written**: this exact form, with rung 1's own
values substituted, returns `6 tasks in the graph; nothing was dispatched` from
`agent-sys show` — which is the whole of the check a launch line can be given
before it costs a node. A `${NAME}` with no default is a load-time fault naming
the file, the line and the variable, so `show` is the difference between finding
a missing var in under a second and finding it after a bring-up.

## Rung 0 cannot complete on the login node, and that is by design

Measured 2026-09-03: the mock graph runs cleanly through `deploy_kit`, all four
m2 kinds, `profiling_evidence`, `kernel_worklist` and `operator_identity` — and
then stops at **`build_workset`**, on a login node that has no `torch`.

**It is not a defect.** m3 wrote the reason into the mock's own header:

> The one thing this cannot supply is `evidence/`, because evidence is a
> **measurement**: the caller runs the entrypoints afterwards, which is the same
> thing STEP 7 and STEP 8 do. **On a host without torch that step fails and the
> mock is correctly incomplete.**

A mock that *fabricated* `evidence/` would be exactly what MOCK-MAP forbids, and
it would defeat `check_workset_runs`, whose whole job is that the numbers were
measured on this hardware. So the mock stops where measurement begins, which is
the correct place for it to stop.

**Consequence for the ladder, and it changes what "rung 0 green" can mean:**

- **There is no host with torch, and I asserted there was without checking.**
  Measured after writing it: `spur exec 106253 python3 -c "import torch"` →
  `ModuleNotFoundError`. The node's *host* environment has no torch; only the
  **containers** do — m3's real STEP 7/8 run the harness inside
  `rocm/sgl-dev:v0.5.18-rocm720-mi35x`, which is how they got 142.5 dB and
  0.5–1.2% rsd. `spur exec` hands you the host, not a container.

  *(My fifth assumption-stated-as-fact today. The others: a `## STEPS` grep,
  a `cli/stream.py` pointer, an `entry.sh` read by shape, and a `$ref`
  claim inherited without checking. Same remedy each time: **look before
  asserting**.)*

- **The fix was m4's principle, and it has been implemented — this paragraph
  used to say it had not.** `build_workset`'s mock now runs its entrypoints
  **in a container, where the real path measures**: `a94ce98`, 2026-09-03.
  `assets/build_workset.task/entry.sh:92` calls `measure_in_container.sh`
  immediately after `mock_adapt.py`, and the same commit dropped the host-side
  `torch` probe that the paragraph above was diagnosing.

  It is exercised, not merely written: **rung 1's refusal at 05:08:56 came from
  inside `measure_in_container.sh`** — the mock reached the container path and
  stopped on the visibility guard.

  *Left visible rather than silently edited, because the failure is this
  document's own recurring one. The replaced text stated the **plan**, was
  written before the commit that carried it out, and was then read for a day as
  a description of the current state — by me, and I routed it to m3 as work
  they had not done. **A plan and a status in the same tense are
  indistinguishable to every later reader.** Same genus as §4.3 arriving in the
  section that records §4.3, and as the `writes_in_place` correction above it.
  Caught by m3 checking the tree before doing the work rather than after.*
- **"Mock e2e green" therefore never meant "no hardware".** It meant "no model
  call, no bring-up, no campaign". Stage 3 onward still needs a card, because
  three of this package's validators are `cost: gpu_hours` and two of them grade
  measurements rather than shapes.

Say what rung 0 covered in those terms, not as "the mock passed".

## Before rung 1, and again before every rung

1. **`m2`'s interpreter sweep** — `python3 assets/lib/interpreter_sweep.py`. About a minute, no node. **Treat a clean result as a gate, not a formality**: all four bugs in that class were introduced by bodies written *after* the previous sweep, so the sweep is only worth its cost when it is re-run.

   Read the last three sections: **validators that died without writing a verdict** (a validator that refuses is healthy; one that dies without a verdict is read by the phase as broken rather than as a refused handoff), and the two "still ambiguous" lists, which say whether the run graded anything on a partial fixture.

   *In the repo since 2026-09-04. It lived in scratch until then, and `RUN-PLAN` pointed at it there — **a documented gate backed by a file only one person had**, which is worse than an undocumented tool because the document implies the check exists. Scratch copies under `ws_handoff_refine/m2/` and `~/ws_handoff_refine_m2/` are frozen; this path is the one that runs. Writes its zones to `$E2E_SWEEP_SCRATCH` (default `~/ws_handoff_refine_m2/sweep`), never beside itself.*

2. **The node, before you spend the hold** — `assets/lib/nodeprobe.sh <node>`, seconds, no hold needed. Free cards, m1's anchor in a local base, disk on **both** filesystems, and whether `spur-authz` accepts this flow's mounts. **Re-probe at the moment of taking, not from a message**: measured 2026-09-04, a node went from seven free cards to 0/8 at 97 % in twenty minutes. `--auto` sweeps every idle/mix node and prints why each one failed; `report.py <rows.jsonl>` re-reads a sweep without re-running it.

   *Measured 2026-09-04, and it changes how a probe result is read: **a cancelled
   Slurm job does not reclaim its GPUs.** Job `109192` was cancelled at 06:46:49
   while four containers of ours were serving on node 006; fifteen minutes later
   all four were still `Up`, the engine still answered `/health` with 200, and
   **all eight cards read 74–76%**. Containers talk to the **host** docker
   daemon, so they are not in the job's cgroup and nothing tears them down when
   the hold ends.*

   *So a card at 90% is not evidence that anyone is still working, and a node
   that looks fully occupied may be carrying **corpses of cancelled jobs**. The
   probe cannot tell the two apart, and neither can `squeue`. This cuts both
   ways: it is why a node can look busy and be free, and it is why **your own
   cleanup is not optional** — an abandoned bring-up keeps four cards out of
   circulation for the rest of the reservation, for everyone.*

   *`sinfo` cannot answer this. GRES accounting is not configured on this cluster — there is no `GresUsed` field and `sinfo -o "%G"` prints `?` — and the co-tenants allocate through the host docker daemon, which never speaks to Slurm. `idle`/`mix` is a statement about CPUs.*

3. **`agent-sys show`** — under a second.
4. **The node's state, before and after**: `docker ps` and the port band. Every identifier this package binds carries a run tag; **check the tag before killing anything.** Measured 2026-09-03: a validator's teardown crashed and warned that ports might be held, and the ports that were held belonged to *a different owner's run in flight*. Killing them would have destroyed live work.

### How to stop a run, in order — because killing the orchestrator does not stop it

Measured 2026-09-04 (`todo.md` **T26**). The orchestrator was killed at 06:47;
its `deploy_and_prove` agent **kept working for 44 more minutes**, brought up two
further deployments, and took every free card on the node. A run is three things
and stopping it means stopping all three, outermost first:

1. **the orchestrator** — the `agent-sys` process you launched;
2. **the agents it dispatched** — *these are not in its process tree.* Find them
   by **cwd**, not by name:

   ```sh
   ls -l /proc/*/cwd 2>/dev/null | grep agent_sys_runroot
   ```

   **Do not grep for `agent_sys.cli.main`.** Two people did, both got an empty
   result, and both reported "no agent is alive" while one was. A dispatched
   agent is a `claude` binary under `~/.local/share/claude/versions/`, which that
   pattern cannot match — *a cwd is a property of the process, a command line is
   a property of how it was invoked* (m4). An empty grep and no agent look
   identical, which is the same failure as `mock.sh` not distinguishing *unset*
   from *not listed*;
3. **the containers, by label** — `infera_e2e_kit_run` / `infera_e2e_run` carry
   the run tag. They talk to the **host** daemon, so they are outside the job's
   cgroup and survive both the agent and the Slurm allocation (item 2 above).
   `docker stop` then `docker rm`, never `-f`, and **check the label rather than
   the name**: three ownership errors were made today by reasoning from names.

**Confirm rather than assume at each step**, and confirm the *whole* thing at the
end: no process with a run-tree cwd, no container carrying the run's label, and
the cards back at 0%.

## The three things a real run can do that a mock cannot

**1. Call a model.** Four leaves are `kind: ai`. `--var m<N>_agent=runner` is what keeps them off that path, and **the default is the real agent** — so a rung is promoted by *removing* a var, and forgetting one is how a model gets called by accident. It has already happened once: the first full mock sat at `deploy_and_prove: running` while an AI deployment agent prepared to bring a model up for real on the node in `--var`.

**The first real run of any `kind: ai` closure in this package happens with the leader watching.** Not distrust: an AI agent with a live node and a container is the one thing here that can change state nobody asked for.

**2. Hold cluster resources.** Containers, ports, GPU memory. Bodies must call `assets/lib/reclaim.sh` in a `finally` (CONTRACT §5.0) and must never `docker rm -f` a name they did not create. Both held nodes carry other tenants' work.

**3. Produce a number somebody will believe.** Which is the point, and the reason for the rest of it.

## What each rung must not be allowed to mean

- **Rung 1 green does not mean m1 is proven.** It means the engine answered eleven probes and one load cleared the floors — on one model, on one node, once.
- **Rung 3 green does not mean the workset is right.** It means
  `check_workset_runs` re-measured a shape and agreed with the record.

  *Corrected 2026-09-03: this note went on to say `writes_in_place` was
  stub-validated and that `apply_patch` hedged accordingly. Both stopped being
  true an hour after it was written — and I knew, and said so in the same
  message the document went on contradicting. **Caught by m3 reading this file
  against something they happened to know**, which is §4.3's own class arriving
  in the document that records §4.3, by its author, within the hour. Left
  visible rather than silently edited.*

  `writes_in_place` is validated on MI355X, and the result strengthens the rung
  rather than merely un-hedging it: of four numerically-correct implementations
  the unsubstitutable one scored **151.9 dB against the baseline's 142.4**.
  **The artefact that makes it wrong is the same one that makes it score
  higher** — it refuses to write the caller's buffer. So a gate reading only
  output quality does not merely fail to notice that implementation; it
  **prefers** it.
- **Rung 5 green does not mean the optimisation is good.** `check_no_regression` recomputes from raw numbers and the bars stay at 5% / 10%. If the two arms disagree by more than that, the finding may still be about the node rather than the patch — that is `todo.md` **T7**, the comparability gate, and it is unbuilt. **Do not widen the bars.**

## "Mock e2e green" is a file and a condition, and the run's exit code is 5

**Read this before reading an exit code from a mock run.** `agent-sys run` on
the mock exits **5**, every time, and that is the correct output. Anyone who
stops at the exit code will read the deliverable as failing. It is not.

**Why it cannot be otherwise.** The corpus's `integration_report` carries a
*refused* verdict — `cheat_for_mock/README.md` warned about it from the start —
and `check_no_regression` does not take the report's word for it: it
**recomputes** from the raw numbers and reaches `REJECTED` independently. So one
handoff seals `invalid`, and `main.py`'s completion rule (*every task
`SUCCEEDED`, no handoff `INVALID`*) gives 5.

Three ways to make that a 0, and all three are worse than the 5:

- **Swap the fixture for a passing one.** There isn't one. Because the validator
  recomputes, a passing fixture means changing numbers nobody chose — and the
  corpus's entire value is that nobody chose them.
- **Widen the bar.** Tried once, and `DELIVERY-NOTE-FROM-LEADER.md` is explicit
  that it was the wrong answer. **Do not widen the bars.**
- **Declare it an expected failure** via `cli/expectations.py`. The framework
  really does have this (pytest's `xfail(strict=True)`), and it is a trap:
  declaring **one** promise switches off completion checking for **all fifteen
  handoffs**, and the sealed verdict has no reason field, so the promise would
  also accept a *real* regression next week wearing this refusal's clothes.
  Measured and written up in
  `temp/bugs/2026-09-04-declaring-one-expected-failure-disables-completion-checking-for-the-whole-run.md`.

**So the exit code does not carry the claim, and CLAUDE.md principle 1 says what
does:** *read the artefact, not the exit code; every acceptance claim names a
file to open and a condition that fails.*

### The claim

> One run produces **15 handoffs** and **43 verdicts**, of which **42 are true**;
> the single false verdict is **`check_no_regression` on `integration_report`**;
> and that validator's **`validator_report.txt` carries exactly four `PROBLEM:`
> lines** — the producer's `max_throughput_regression=35%` against the
> validator's 5% bar, the producer's `max_latency_regression=30%` against its
> 10% bar, four metrics regressed, and six metrics whose noise floor exceeds
> their bar so the run cannot judge the patch.

### How to check it

```sh
python3 assets/lib/accept_mock.py                    # newest run under ~/agent_sys_runroot
python3 assets/lib/accept_mock.py --run <run-dir>
```

**0** the claim holds — this is the deliverable's green. **1** it does not, and
every difference is printed. **2** the script could not tell (no run, no report),
which is deliberately not 1: *cannot judge* and *judged and refused* are
different facts, which is the same distinction the validators themselves draw.

**It is stricter than the run, not softer.** Nothing is inverted, relaxed or
skipped: the verdict is still false and the run still exits 5. It fails in three
directions the exit code cannot distinguish — a **different** refusal from the
same validator on the same kind, a **second** refusal anywhere, and the expected
refusal **disappearing** (43 true is not the claim either; an artefact that
stopped refusing is a finding).

Its ability to fire is not assumed. Six controls on copies of a real run —
reword one `PROBLEM`, add one, flip the refusal to a pass, add a second refusal,
delete the report, and an untouched copy as the null — are recorded in the
commit that added it.

## The one measurement to take at rung 5 that nobody has

`todo.md` T7 wants the within-arm round-to-round spread on a **quiet** node. Both holds carry other tenants today. If a quiet window appears, take it: it is the number that decides whether 5% / 10% are right, and the previous round widened them to 35% / 30% on a cross-instance artefact for want of it.

---

## Standalone verification — m4 (`optimize_kernel`)

The user's redirection, 2026-09-04: *"e2e串通也可以通过先单独运行每个模块保证单独通(也可以并行)"*
— verify each module on its own rather than only through the ladder. This is m4's.
**m4 is the stage that most needs it: it is the only one that has never executed
in any graph, on any rung.**

> **Run from a directory outside the package, and point `--out` outside it too.**
> `apply.py` left `e2e-flow/stage/` (33 K, the extracted stock file) inside the
> deliverable on 2026-09-04. **`agent-sys run` stages the working tree, so every
> later run would have copied that scratch into every zone**, and the brief says
> this directory receives only the package and `todo.md`.
>
> **The scratch comes from `Path.cwd()`** (`apply.py:499`, `stage = Path.cwd() /
> "stage"`), **not from `--package`** — I first recorded it as `--package` and
> that was wrong, found by reading the line rather than by it happening twice.
> The distinction changes the remedy: no flag moves it, so **`cd` somewhere
> disposable before invoking**, and passing `--package` at the working tree is
> both correct and harmless.
>
> Nothing in the CLI hints at it, which is the part that will catch the next
> person: the invocation names `--out` and `--package` and silently uses a third
> location neither of them mentions. **Check `git status --porcelain -- <package>`
> after running any body by hand**, whatever you believe about where its scratch
> goes.
>
> **And never nest a default in a value you pass by hand.** `${a:-${b:-x}}` does
> not resolve: the renderer stops at the first `}`, so a *string* consumer
> receives the literal `x}` — or the whole unexpanded expression when nothing is
> set. Measured by m2 and confirmed by the leader; **nothing in the package uses
> one, so there is no exposure today.**
>
> It belongs in this section rather than only in theirs because **a hand-driven
> body is where someone invents a value on the spot**, and the values this stage
> binds are container names, ports and paths — all string consumers. There is no
> `float()` to throw, so **the loudness of the case we found is a property of the
> consumer, not of the defect**: the failure here would be a container called
> `mybox}` on a shared host, created successfully, doing real work under a name
> nobody will recognise. That is the failure `run_in_container.sh` refuses an
> empty `HIP_VISIBLE_DEVICES` to prevent, arriving through the one door it does
> not watch.
>
> **Related, and the reason a hand-driven run is worth its awkwardness at all:**
> drive `--package` from the **working tree, never a run's staged copy**. A run
> stages the tree *at launch*, so its copy is a snapshot of a moment — the branch
> you are testing may have landed after it, and **a missing branch reads exactly
> like a branch that failed to fire.** Confirm the code is in the file you are
> invoking before you invoke it (`grep -c '<the new line>' <pkg>/…`), which is
> what m5 handed over instead of an instruction and what made the `apply.py` run
> trustworthy.

Written in a shape m1, m2, m3 and m5 can copy: **the command, the input, the
cost, the smallest honest proof.**

### 1. The command, in full

Not a delta against another table. Every launch-line failure on 2026-09-04 came
from someone assembling a command from a diff.

```sh
agent-sys run --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /home/yihou/agent_sys_runroot \
  --var jobid=<JOBID> --var node=<NODE> --var node_ip=<NODE_IP> \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=<THE KIT'S IMAGE, not the node's> \
  --var transport_env=SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR \
  --var mock_stages=all \
  --var m1_agent=runner --var m2_agent=runner --var m3_agent=runner --var m5_agent=runner \
  --var workset_operator=sampler_vocab_softmax \
  --var gpu=<A FREE CARD> \
  --var scratch_root=/mnt/m2m_nobackup/yihou/e2e_flow/kfo \
  --var forge_mock=1
```

Four of these are m4-specific and each has a reason that has already cost
something:

- **`workset_operator`** — a workset with more than one operator is a run-time
  refusal, not a coin flip. m3's real worksets carry one today, so it is
  optional *now* and mandatory the moment they carry two.
- **`gpu`** — no default, and the refusal is deliberate (CONTRACT §5.2). Cards
  are shared; `HIP_VISIBLE_DEVICES` empty stops the run rather than picking
  card 0 next to a co-tenant.
- **`scratch_root`** — must be node-local. On this cluster's NFS every ROCm
  kernel launch segfaults *after* the copies and the first round.
- **`forge_mock=1`** — see §4. This is what makes the run minutes instead of
  hours, and it is the whole reason a standalone m4 is cheap.

**`m4_agent` is deliberately absent.** Leaving it out keeps the real
`kind: ai` agent, which is the point of verifying m4 rather than its mock.

### 2. The input, and the seam — tested 2026-09-04

m4 takes `operator_workset` **and** `deploy_kit`. m3 has produced two real
measured worksets; both were driven through m4's STEP 1 and STEP 2 on the login
node, no GPU:

| workset | STEP 1 | why |
|---|---|---|
| `/home/yihou/m3_verify/` (006) | **refused** | `the workset carries no evidence.performance_report` — the `evidence` block is absent from `workset.yaml` although `evidence/{correctness,performance}.json` exist on disk |
| `/home/yihou/m3_verify234/` (234) | **passes** | `ok: operator sampler_vocab_softmax, 3 performance shape(s), 3 correctness shape(s), baseline from evidence/performance.json` |

**So m3's current output is consumable by m4 as-is.** Everything m4 previously
refused a workset for — `integration`, `noise_floor`, `apparatus`,
`ground_truth.dtypes`, and three *performance-role* shapes rather than one — is
present in the 234 artefact. The A2 scaffold change landed and this is the
evidence.

The `m3_verify` gap is one line in `workset.yaml` and m3 fixed it between the
two runs; recorded because **the files existing on disk is not the same as the
workset declaring them**, and only the declaration is read.

**The pairing constraint, which is the actual seam and is easy to get wrong:**
the workset's `ground_truth.environment` and the `deploy_kit`'s record must
agree on every abort-listed field. Pairing m3's 234 workset with a `deploy_kit`
from a different bring-up produced, correctly:

```
ABORT: fixed.tp_size: workset 4, this run 8
```

m3 measured at `tp=4` because four cards were free. **A standalone m4 run needs
a `deploy_kit` minted for the same bring-up the workset was measured against**,
or it aborts before spending anything — which is the gate working, and is also
§4's "can it disagree" half obtained for free.

### 3. What it costs — and "hours" is true of one step only

| step | needs | time |
|---|---|---|
| 1 `10_read_inputs` | nothing | < 1 s |
| 2 `20_premise_gate` | nothing | < 1 s |
| 3 `30_run_forge` | GPU + the container | **`forge_mock=1`: ~1 s. A real campaign: ≥ 1 hour, enforced** |
| 4 `40_correctness` | GPU + the container | ~90 s, most of it container start and torch import |
| 5 `50_performance` | GPU + the container | ~90 s, same |
| 6 `60_write_handoff` | the container, for `base_sha256` | seconds |
| 7 `70_selfcheck` | nothing | seconds |

**The "hours" in the bug file is `KFO_MAX_HOURS` and belongs to STEP 3 alone.**
It is not a description of the stage. And the floor is *enforced* — `cli.py:46`
`MIN_MAX_HOURS` rejects anything below 1.0 — so **there is no such thing as a
five-minute real campaign**; the choice is a campaign or a mock, with nothing
in between.

**Everything except STEP 3 is about three minutes on one card**, and most of
that is torch importing. That is the same correction rung 0 got today: a stage
treated as a node-hour problem needed one card for seconds.

### 4. The smallest honest proof

m3's standard — *does it agree, and can it disagree* — applied here:

**Can it disagree: already demonstrated, on real artefacts.** The `tp_size 4 vs 8`
abort above is m4 refusing a premise that does not hold, on two artefacts
neither of which was built to make it fail. That is the half that matters,
because a stage that only ever produces a number proves nothing.

**Does it agree: `--var forge_mock=1` with a matched workset/kit pair.** This
exercises everything except the optimiser:

- STEPs 1–2 **for real** against m3's workset and m1's record;
- STEP 3's wiring, with the seed copied behind a banner and a `forge_result.json`
  of nulls — *a mock is not a small campaign*;
- STEPs 4–5 **for real**: the workset's own entrypoints, in the container, on a
  card, measuring the seed against itself. The ratio is ~1.0 **and that is the
  correct answer** — it proves the measurement path end to end without a claim;
- STEP 6 **refusing to write a claim** under a mocked forge — the schema
  forbids it, so a green run here is a run that produced a handoff *with no
  speedup in it*;
- STEP 7, `check_optimization_shape` against the real output.

**What it does not prove, stated so nobody reads more into a green run:**
KernelForge has still never run, `check_speedup_substantiated`'s re-measurement
has never graded a real optimised kernel, and no number this produces is a
speedup. **A campaign is the only thing that proves the campaign.**

### 4a. What the first real run found — 2026-09-04, `crsuse2-m2m-217`

**The middle row is NOT green.** STEPs 1–3 passed, STEP 4 is blocked, 5 and 6
unreached, campaign untouched. Recorded as a partial run rather than a pass,
because the three defects below are the return on it and each would otherwise
have surfaced inside an expensive rung.

**What did work, and it is the first time for both:** STEP 2's premise **held**
on a third machine — a 234-measured workset against a 006 kit on a 217 run,
which is `node` deliberately not being an abort field, vindicated. And m3's
harness accepted m4's `--environment` flag and made the two-document
comparison, warning on `image_id` and carrying it forward.

#### A near-miss, not a fixed bug: `KFO_MOCK` did not cross the boundary

`run_in_container.sh` forwarded an **enumerated** list of environment names.
`KFO_MOCK` was not on it, so `--var forge_mock=1` never reached the container
and `30_run_forge.sh` took the **real campaign branch**.

**It died on git's dubious-ownership check rather than running for an hour.**
That is luck, not design. Had the copy succeeded it would have consumed a node
we hold under **preemptible burst** and produced nothing anyone asked for.

Recorded as a near-miss because the fix — forwarding `AGENT_SYS_*`, `KFO_*` and
`E2E_*` by **prefix** — removes the class rather than the instance. The same
list also dropped `AGENT_SYS_INPUT_<KIND>`, and *the kind is part of the
variable name*, so any fixed list has to duplicate the kind list from
`steps/m4_kernel_opt.yaml`. **A prefix has no list to fall behind.**

**The general form, for anyone writing a boundary crossing: the wrapper IS the
boundary. Anything it does not carry does not arrive — and it arrives as a
fault in whatever runs on the far side**, which is where nobody is looking.

#### The blocker: `--impl` has an unstated contract

```
the Definition's 'candidate:sampler_vocab_softmax' defines no `run`
error: the correctness entrypoint exited 1
```

`harness/_common.py:289` execs the candidate and **requires a top-level `run`
callable**. m4's mock seed is `edit_target.source_file` — the engine's stock
`srt/layers/sampler.py`, an sglang module with no `run`. The two sides disagree
about what `--impl PATH` *is*.

**This bites the real campaign, not just the mock:** whatever KernelForge emits
must define `run` for m3's harness to accept it, and **nothing states that
anywhere**. A campaign is an hour minimum, so discovering it after one is the
most expensive version of this mistake available.

**Leader's ruling, 2026-09-04, and the order is the substance:**

1. **m3 declares the candidate contract in the workset** — they own the harness
   and the schema, so the authority belongs there. Same family as the flag
   spelling m3 made data: one authority, two readers, except here the authority
   did not exist.
2. **Then m4 adapts its seed to satisfy it.**

**Not the other way round.** A seed changed first would be fitted to m3's
*current behaviour*, which is precisely the undeclared thing this is meant to
stop relying on.

### 5. Useful at every size

One card, three minutes, and the pieces detach:

- **no card at all** — STEPs 1, 2 and 7 run on the login node. That is the
  workset seam and the premise gate, which is where both real defects found in
  this stage so far have been.
- **one card, ~3 min** — add STEPs 4, 5, 6 with `forge_mock=1`. This is the
  recommended standalone run.
- **one card, ≥ 1 h** — drop `forge_mock` and it is rung 4.

The middle row is the one to run whenever a card is free for five minutes.

---

# Module 5 standalone — the smallest run that proves the wiring

Written for the user's *"先单独运行每个模块保证单独通"*. **The answer to "is
there a smaller thing?" is yes, and it needed a fix first.**

## 1. The command

```sh
python3 -m agent_sys.cli.main run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /home/yihou/agent_sys_runroot \
  --var jobid=<the hold> --var node=<the node> --var node_ip=<MEASURED, see below> \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=<a servable image ON THAT NODE> \
  --var transport_env=SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR \
  --var mock_stages=m1,m2,m3,m4 \
  --var m1_agent=runner --var m3_agent=runner --var m4_agent=runner \
  --var expect_ranks=2 \
  --var container=yihou_m5_solo --var port_router=8151 \
  --var port_worker=8152 --var port_etcd=8153 \
  --var work_root=/mnt/m2m_nobackup/yihou/m5_solo \
  --var aiperf_trace=/shared_nfs/yihou/agent_sys/debugging/profiling/conversation_trace.jsonl \
  --var gsm8k_data=<a local GSM8K jsonl> \
  --var eval_max_tokens=1024 --var eval_examples=5 \
  --var needle_tokens=2000 --var trace_end_ms=30000 \
  --var bench_rounds=1 --var adhoc_cases=1
```

**`m5_agent` is deliberately absent.** m5 runs with its real `kind: ai` agent,
because that agent following the STEPS readme *is* the module. `--var
m5_agent=runner` swaps in the mock and proves nothing about the real path.

**`node_ip` is measured, never derived.** `spur exec <job> hostname -I`. There is
no pattern: `-061` → `10.245.159.129`, `-031` → `10.245.144.239`, `-006` →
`10.245.151.128`. Deriving it cost a bring-up.

## 2. What it consumes, and where a real one comes from

| input | source in this run | a real one |
|---|---|---|
| `deploy_kit` | m1 mock — the sealed `stage1-deploy` kit | rung 1 |
| `profiling_evidence` | m2 mock, merged from four sealed parts | rung 2 |
| `operator_workset` | m3 mock + adaptation (C) | rung 3 |
| `kernel_optimization` | m4 mock + adaptation (G) | rung 4 |

**Two of these are known-defective as inputs and both are m4's**, found by
driving `apply_patch`'s real path on 2026-09-04: the manifest declares
`@SGLANG_ROOT@/python/sglang/srt/layers/sampler.py`, which repeats the root and
names a path no image has; and its `base_sha256` was cut against
`rocm/sgl-dev:v0.5.18-…`, so it will not match whatever image this run serves.
**Both surface as a clean refusal with a named cause** (`03693af`), not a
traceback — but they stop the run at `apply_patch` until m4 re-cuts.

## 3. Resource and duration

**One node, 8 cards, two containers brought up in sequence** — m5 is the
designed exception to the one-container rule (CONTRACT §5).

Measured per arm, from the sealed arms' own `env/steps.json`:

| step | stock | patched | reducible? |
|---|---|---|---|
| serve | 480 s | 240 s | no — bring-up is irreducible |
| smoke | 271 s | 220 s | no |
| needle | 200 s | 194 s | yes, `needle_tokens` |
| **probe** | **2062 s** | **2001 s** | **no — see the correction below** |
| lm_eval | 23 s | 428 s | yes, `eval_examples` |
| bench_r1 | 44 s | 161 s | yes, `trace_end_ms` |
| **total** | **3080 s** | **3244 s** | **~105 min for the pair** |

**`probe` is two thirds of an arm, and until today its cost knob could not be
set.** `measure.sh:149,157` reads `E2E_EVAL_MAX_TOKENS` and `E2E_EVAL_THREADS`
and **nothing declared either**, so the `:-2048` fallback always won. Now
declared on `e2e_integrator`; `runner` should mirror them.

### Corrected 2026-09-04 by measuring it: probe is not the problem, and 256 breaks it

The two paragraphs this replaces said `probe` was two thirds of an arm and that
`eval_max_tokens` was the knob that would shrink it. **Measured on
crsuse2-m2m-047, an idle node, against the same deployment:**

| `--max-tokens` | probe | verdict |
|---|---|---|
| 256 | 28 s | **FAIL** |
| 512 | 34 s | pass |
| 1024 | 38 s | pass |
| **2048** — the sealed run's own budget | **37 s** | pass |

**2062 s → 37 s at the identical budget.** The knob barely moves the cost; **the
2062 s was the contended chassis**, the same one that read 193 tok/s beside an
idle neighbour and 47 beside a busy one. So the ~105-minute two-arm total is a
property of a busy machine and not of this module, and it should not be quoted
as module 5's cost.

**And `eval_max_tokens=256`, which the command above used to say, is wrong.**
It fails probe, and not on the deployment:

```
multiply  expected 391 -> "23"   completion_tokens [256,256,256]  finish_reason "length"
natalia   serial 72,72,72  but batched 48,48,72,72,48,48,72,48
```

The model was still reasoning when the budget ran out and the extractor took a
number out of the middle of the working; the batched instability is the same
cause. **The floor is between 256 and 512**, and since 512 costs 34 s against
2048's 37 s there is no reason to run below **1024**, which is what the command
now passes.

**The whole reduced suite ran in 102 s** on that node — smoke 9, needle 4, probe
28, lm_eval 14, bench_r1 47 — against baselines of 271/200/2062/23/44. **The
bring-up is the only large cost left**, and it is irreducible.

**What is still an estimate:** a *full-scale* arm on an idle node. needle at
31 000 tokens and lm_eval at 100 examples have not been measured here, so no
full-scale total is quoted. One extrapolation per section is the limit, and this
section already spent its correction on the last one.

## 4. The smallest honest proof

**What a reduced run proves:** two bring-ups on one node in one session, a
teardown between them, the overlay mounted on the second, the patch-live
evidence collected from inside the running container, both arms measured in the
same order, the comparison computed, and the packup assembled — every edge and
every one of the seven validators, against artefacts this package produced.

**What it does not prove, and cannot:** that the optimisation is good. **This is
structural rather than a promise.** At reduced scale `n` is small, so the noise
floor `1.96·√2·rsd/√n` rises above the bar, and `check_no_regression` returns
**`uninterpretable`** — *this run cannot resolve a difference at this bar* —
instead of a verdict. The gate refuses to let a short run claim a speedup, and
the report carries the floor either way. Measured on the sealed arms at n=60:
floors of 12.8%, 17.9% and 20.3% against a 10% bar.

So the pass condition is **not** "accepted". It is:

- `apply_patch` produces `patch_overlay` and `check_overlay_applies` passes;
- both arms seal, and `check_measurement_order` confirms they were disjoint in
  time **and on the same machine** (`f6131f7`);
- `check_patch_live` passes on the patched arm — the mounted bytes were the ones
  that ran, proven by re-hashing inside the container;
- `check_no_regression` returns a verdict **or** `uninterpretable` **with the
  floor stated** — both are passes for this purpose, and a silent `same` at
  reduced scale would be the failure;
- `packup` writes `e2e_packup` and `check_packup_shape` passes.

**Can it disagree?** Yes, and that is the half worth stating: the same run with
`--var eval_max_tokens=2048 --var trace_end_ms=120000` raises `n`, drops the
floor below the bar, and the same gate then returns a real verdict. A proof that
cannot fail is not a proof, and the difference between the two is one `--var`.

## 5. If it still does not fit

**It may not.** ~105 min at full scale against holds that died at 28 min, 1 h 21
and 5 h today, with no fitted pattern. The reduced run removes `probe` and
`lm_eval` as the dominant terms and leaves **two bring-ups, ~12 min of
irreducible `serve` plus ~8 min of `smoke`**, which is the floor for anything
that measures two arms at all.

**Below that floor there is no honest module-5 test**, because a single arm does
not exercise the sequencing, the mount, or the patch-live evidence — which are
the three things that have never run. If even ~25 minutes cannot be held, the
finding is that **Brief item 3 is not achievable in this environment**, and that
is a fact about the allocation rather than about this stage.

## Standalone verification — m1 (`deploy_and_prove`)

The fifth of five, in m4's four headings. **m1 is the opposite of m4's case: it
is the stage that has run most**, and the one whose standalone is cheapest to
*set up* and dearest to *execute* — because the thing it verifies is a real
model bring-up and there is no honest way to make that fast.

### 1. The command, in full

```sh
python3 -m agent_sys.cli.main run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /home/yihou/agent_sys_runroot \
  --var jobid=<JOBID> --var node=<NODE> --var node_ip=<NODE_IP> \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=<AN IMAGE THAT CONTAINS infera> \
  --var gpu_devices=0,1,2,3 --var tp=4 \
  --var parser_args="--reasoning-parser qwen3" \
  --var transport=spur --var transport_env="SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR" \
  --var mock_stages=m2,m3,m4,m5 \
  --var m2_agent=runner --var m3_agent=runner --var m4_agent=runner --var m5_agent=runner \
  --var expect_ranks=2 --var adhoc_cases=0
```

**`m1_agent` is deliberately absent**, for m4's reason: leaving it out keeps the
real `kind: ai` agent, which is the point.

**`mock_stages` is `m2,m3,m4,m5` and NOT `all`, and this is the one place m1's
command cannot be copied from m4's.** m4's readme never mentions `mock.sh`, so
`mock_stages=all` leaves their ai agent unaffected — their mock lives in
`entry.sh`, which a `kind: ai` closure does not run. **`deploy_and_prove`'s
readme calls `mock.sh` itself, at STEP 0** (`readme.md:71`, 3 references).
So with `mock_stages=all` the real agent dutifully mocks and stops, and the run
looks like a pass having deployed nothing. Measured across the four ai readmes:
`deploy_and_prove` 3, `optimize_kernel` 0, `build_workset` 0,
`integrate_and_verify` 0. **m1 is the exception; copying m4's line silently
verifies nothing.**

Three more with reasons that have each cost something:

- **`image` must contain `infera`.** Not "a recent sglang" — the kit runs
  `python -m infera.engine.sglang` and `python -m infera.server`, so a plain
  sglang image cannot serve it at any version. Measured on three of them:
  `ModuleNotFoundError: No module named 'infera'`. `nodeprobe.sh`'s **READY**
  means *a base carrying the build anchor is present*, **not that a servable
  image exists** — 006 and 217 happened to carry one, 235 did not.
- **`gpu_devices`** — optional, and passing it is what exercises the branch that
  has never run: a named set means *use exactly these and stop if one is not
  free*, rather than choose. Omit it (`none`) to exercise the discovery path
  instead. Both are worth doing; they are different tests.
- **`transport_env`** — a validator declares no agent, so the package's `env`
  block never reaches `check_deploy_serves`. Without this it refuses **in one
  second** with `failed to connect to controller`, which reads exactly like a
  deployment failure and has cost several runs.

### 2. What it consumes — nothing, and that is the point

**m1 is the only stage whose standalone needs no other module's artefact.**
`deploy_and_prove` declares `inputs: []`; `deploy_kit` is the flow's root
handoff. Every other standalone has had to source a real upstream input — m4
needed one of m3's worksets *paired with the deploy_kit from the same bring-up*,
m5 needs the chain below it.

So the setup cost is a node, a model on disk and an image. There is no "where do
I get a real one" question, because m1 **is** where the real one comes from: the
`deploy_kit` this run seals is exactly what m2, m3, m4 and m5's standalones
consume.

### 3. What it costs — measured, and the expensive part is irreducible

Measured on the 2026-09-04 rung-1 run (249, TP-1 bring-up, all three validators
green):

| step | needs | time |
|---|---|---|
| package load, `show` | nothing | < 1 s |
| image build, **if the node has no infera image** | disk + network | **~4 min** (base local); a base pull is ~120 GB |
| `deploy_and_prove` — the AI agent | node + cards | **40 min 33 s** |
| `check_environment` | nothing | seconds |
| `check_deploy_kit` | nothing | seconds — pure static, `cost: seconds` |
| `check_deploy_serves` | node + cards | a **second** full bring-up, 11 probes, a 1k/1k conc-16 **180 s** load, teardown |

**`deploy_and_prove`'s 40 minutes is an AI agent's session, not a weight load** —
worth stating because the sealed kit documents its own bring-up at 186 s and
anyone sizing a budget from that number is out by a factor of thirteen.

**Two knobs shorten it and neither is free.** `deploy_load_seconds` (default 180)
and the rest of the load shape are overridable *"so a wiring run costs seconds"*
— but the bring-up dominates, so shortening the load saves ~2 minutes of ~50 and
weakens the one validator that grades sustained behaviour. **Not recommended:
unlike m4's `forge_mock=1`, there is no cheap mode here that leaves the proof
intact**, because m1's expensive step *is* the thing being proven.

### 4. The smallest honest proof — does it agree, and can it disagree

**Does it agree.** Three validators, in cost order, and all three graded a real
engine on 2026-09-04:

```
check_environment    PASS   completeness / strong
check_deploy_kit     PASS   completeness / strong
check_deploy_serves  PASS   usability / strong
```

The third is the one that matters: it takes **the kit the agent just wrote**,
redeploys it under a *different* run tag, port band and work root, probes it,
loads it and tears it down. A kit that only works for its author fails there.

**Can it disagree — yes, and each of these has fired on a real artefact:**

| refusal | what produced it |
|---|---|
| `environment.md does not render fixed.image` | a wrong `--var` reaching a real record |
| `nothing reads ${E2E_KIT_GPU_DEVICES:=…}` | a kit sealed before the parameter existed — fixed in the adapter, not by an exemption |
| `fixed.gpu_devices has 9 entries but fixed.gpu_count is 8` | the record-internal invariant, on a planted fault |
| `no router-side reading of "disagg_mode"` | one component agreeing with itself |
| the model served under a filesystem path | `--served-model-name` omitted |

`gate.sh` keeps this honest: it runs `check_deploy_kit` against the real sealed
kit (**must pass**) and against the same kit with **twelve planted faults**
(**each must be reported**). Four minutes, no node. **Run it before committing
anything to the layout** — a schema tightening that refused the sealed kit
reached a live run on 2026-09-04 because the author verified the yaml parsed and
never ran the gate. *Structure is not behaviour.*

**What a green standalone does NOT prove**, and this is the half worth keeping:

- **one model, one node, one shape, once.** Not that the flow is parameterised:
  rung 1 on 249 passed all three validators while **four `--var`s were inert**,
  because a `kind: ai` agent received no `E2E_*` at all and the agent recovered
  the values from the sealed kit's defaults. Right answer, wrong mechanism.
- **it says nothing about whether m1's record is consumable downstream**, because
  at this rung m2–m5 replay sealed handoffs and read their own records. The
  one-authority-two-readers property is first tested at rung 2.
- **`check_deploy_serves` grades shape, not answer.** Its completion probe
  asserts `finish_reason: stop`, non-empty `content`, and a model id that is not
  a path — so a deployment that returns 526 characters of chain-of-thought where
  the answer should be **passes**. Measured, `todo.md` T21/T28.

---

## Standalone verification — m2 (`run_profiling_mode_off`, `run_profiling_mode_on`, `merge_profiling_evidence`)

The user's redirection, 2026-09-04: verify each module on its own, in parallel,
rather than only through the serial ladder. This is m2's, in the shape m4
(`5caea8a`) and m5 (`fbb73c1`) set.

**What is already known about m2 without spending a card**: its four kinds have
sealed `valid` in three separate mock runs, `merge_profiling_evidence` included,
and every validator that grades them passed — 14 of the 21 verdicts in run
`072849` were m2's. **What has never happened is `line.sh` bringing a service up
on a card.** That is the whole of what this section buys.

### 1. The command, in full

Not a delta against the vars table above. **The table is a diff, and a diff is
the one shape nobody can paste** — every launch-line failure on 2026-09-04 came
from someone assembling a command out of one.

```sh
agent-sys run --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /home/yihou/agent_sys_runroot \
  --var jobid=<JOBID> --var node=<NODE> --var node_ip=<NODE_IP> \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=<THE KIT'S IMAGE, not the node's> \
  --var transport_env=SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR \
  --var mock_stages=m1,m3,m4,m5 \
  --var m1_agent=runner --var m3_agent=runner \
  --var m4_agent=runner --var m5_agent=runner \
  --var expect_ranks=<THE KIT'S tp_size> \
  --var gpu_devices=<THE CARDS THAT ARE FREE, e.g. 4,5,6,7> \
  --var adhoc_cases=0
```

**m2 is real by *absence* from `mock_stages`** — there is no `m2_agent` to drop,
because m2's three leaves are already `agent: runner`. The four `m<N>_agent=runner`
vars stay: they keep m1, m3, m4 and m5 off their AI paths.

Two are m2's and each has cost something:

- **`expect_ranks` — read it out of the kit, never carry it.** It is a fact
  about *the artefact being graded*. `check_trace_coverage` is `strength: strong`,
  declares `${expect_ranks:-8}`, and is deliberately **not** derived from
  `${tp}`. This document held both `4` and `8` as facts on the same day; neither
  was one. m1 sizes the bring-up from the cards that were free, so the value is a
  property of the node on the day:

  ```sh
  grep -E '^  (tp_size|node|image):' \
    "$(find "$RUN" -path '*items/codes/environment.yaml' | head -1)"
  ```

- **`gpu_devices`** — new on 2026-09-04 (`03e3bae`). Until then nothing told the
  kit which cards to use and it defaulted to a hardcoded `0,1,2,3`, so both
  bring-ups took 0–3 whatever was free. Omit it and `line.sh` falls back to the
  set the kit records having **taken**, which is the right default; name it when
  the node is partly occupied.

### 2. What it consumes, and where a real one comes from

**One input: m1's `deploy_kit`. This is the one genuine dependency in the
parallel plan, and it is narrower than it looks.**

**m2 does not need m1's standalone to have run.** It needs *a* real kit whose
`fixed.node` is the node m2 is pointed at — `line.sh` aborts on a mismatch,
deliberately, because otherwise every number would be filed under the wrong
environment. Two ways to have one:

1. **`--var mock_stages=m1,…` replays the sealed kit**, which is a real kit: the
   2026-09-02 deployment scripts plus MOCK-MAP (I)'s shim, and m1 verified that
   shim makes the untouched sealed kit pass `check_deploy_kit`. The mocked
   producer renders `environment.yaml` from *this* run's vars, so the node
   matches by construction. **This is the parallel path and it needs nothing
   from m1.**
2. **A kit already on disk** from any earlier m1 run on the same node.

**Verified, not assumed** — every path `line.sh` reads was checked against the
real kit run `062414` produced on 006: `items/codes/environment.yaml` readable;
exactly one packup directory under `items/codes` carrying `scripts/`;
`deploy.sh`, `wait_ready.sh` and `teardown.sh` all present; all six `fixed.*`
fields (`node`, `image`, `image_id`, `model_name`, `served_model_name`,
`tp_size`) populated; and **all six runtime-contract variables honoured**,
including `E2E_KIT_ROUTER_EXTRA_ARGS`, whose absence would make every capture
produce nothing while reporting success.

**The path that is easy to get wrong**: the staged kit must be visible *from the
node*, because `line.sh` runs `deploy.sh` there by absolute path. That works
only because `--demo-root` is under `/home`, which is NFS and mounted rw on the
nodes. A run root on `/tmp` or local scratch fails here, and until `2cdc403` it
failed while blaming the filesystem for a transport error.

### 3. What it needs, and for how long

**One node, both cards-halves free if possible, for two sequential bring-ups.**

| | measured? | value |
|---|---|---|
| load window per line | **yes** — sealed `items/env/load.json`, `trace_window_ms: [0, 180000]` | **180 s** |
| concurrency / workers | **yes**, same file | 32 / 8 |
| lines | by construction | **2** — `profiling_mode_off`, then `profiling_mode_on` |
| bring-up + teardown per line | **no — never measured for m2** | m2's real path has never run; m1's real bring-ups are the only data and they are m1's to quote |

So the honest statement is **2 × (bring-up + 180 s + teardown), with only the
180 s measured**. I will not put a total here: an unmeasured number in a table of
measured ones is the thing this document keeps getting caught by.

**The two lines are sequential and must be** — each brings its own service up
and tears it down (M2.5), and the profiled line runs with CUDA graphs off, which
measured ~8× slower on the sealed pair. They use different port bands
(`PORT_OFFSET=10`) and different run tags, so a leftover from the first cannot
be mistaken for the second.

### 4. The smallest honest proof — does it agree, and can it disagree

**Agrees:** all four kinds seal `valid` and `check_profiling_evidence` passes
over four parts that arrived separately. That last clause is the load-bearing
one — MOCK-MAP (H) refuses a stand-in for `profiling_evidence` precisely so that
`require_same_environment` is exercised on genuine separateness rather than on a
hand-shaped artefact.

**Can disagree, and these are the ones to watch because each fails loudly on a
real card and cannot fail in a mock:**

- **`check_trace_coverage`** — one readable trace per rank, ≥1000 GPU kernels
  each, span within `max_span_ratio` of the window. A profiler window opened on
  an idling scheduler loop is a valid trace of nothing, and this is the check
  that says so. It is also the one `expect_ranks` decides.
- **`check_bench_result`** — the profiler-detached line is the only throughput
  in this flow worth quoting; the profiled line is *not* a control for it.
- **`check_kernel_table`** — shared with m3 (M3.5), so a disagreement here is a
  cross-stage one and not m2's alone.

**The negative control that makes the positive mean something**: point the run
at a node whose kit records a different `fixed.node` and `line.sh` must abort
before bringing anything up. If it proceeds, the whole agreement above is about
an environment nobody recorded.

---

# The control experiment for module 5, and its predictions written first

**Two overlays, neither an optimisation.** `check_patch_live` and six other
validators have never spoken, because the patched arm has never come up: m4's
seed drops twelve public names and the engine dies at import, and the real
optimisation is blocked on M5.1.1 (this operator has no installable symbol —
m3's `4d5a6e6`). **Making something slower needs no installable symbol**, so a
control experiment is available today even though an optimisation is not.

| overlay | what it is | expected verdict |
|---|---|---|
| **null** | the stock `sampler.py` plus a marker constant | `same`, or `uninterpretable` on a metric whose floor exceeds its bar |
| **degraded** | the stock `sampler.py` plus a real 2 ms cost in `Sampler.forward` | **`REGRESSED`** on throughput and inter-token latency |

**A null overlay alone would be half a control.** *Known-no-effect in, no effect
out* is a negative control; a gate validated only against it has never been
shown to **detect** anything on a real deployment. The tamper batteries show the
logic detects, but they are offline over hand-edited reports and cannot show a
real difference surviving bring-up, mounting, measurement and reduction.

**And the null overlay defeats `require_difference` by construction**, which is
the one thing a later reader would never reconstruct. That check exists to catch
*"a patch that applies cleanly and changes nothing"* and it is implemented as a
**hash** comparison — a marker changes the hash and nothing else. It is
satisfied here by a marker rather than by a change. **Said here because the
artefact must carry it, not a message.**

## The numbers this is predicted against

Measured on crsuse2-m2m-047, idle, stock arm, 123 requests:

```
inter_token_latency_ms  avg 10.02  std 0.66   -> per-request rsd 6.6%
ttft_ms                 avg 140.45 std 250.94 -> rsd 179%, one 1764 ms outlier, p50 87.58
output_token_throughput_tps 448.82
```

## Predictions, before the run

1. **ITL can be resolved and TTFT cannot.** Noise floor `1.96·√2·rsd/√n` at
   n=123: **ITL ≈ 1.7%**, comfortably under its 10% bar; **TTFT ≈ 45%**, far over
   it. So **TTFT is predicted `uninterpretable` on both arms** and ITL is
   predicted decisive on both. If TTFT comes back with a verdict instead, its
   dispersion did not repeat — which is a fact about the deployment, not a bug.
2. **Null arm: `same` on throughput and ITL.** Identical semantics, so the only
   difference is noise, and the floor says noise is ~1.7% against a 10% bar.
3. **Degraded arm: `REGRESSED` on both.** 2 ms added per decode step against a
   10.02 ms ITL is **+20%**, over the 10% latency bar; throughput follows at
   about **−17%**, over the 5% bar.

**If the degraded arm comes back `same`, that is a finding about the gate and
not about the overlay** — and the only reason that distinction is available
afterwards is that this paragraph was written before the run.

## The run this pre-registration belongs to — ADDED AFTERWARDS

**This subsection was written after the run, and says so.** The predictions above
were committed at **09:38:06** (`d71d765`) and nothing in the repository recorded
*when the experiment started*, so the ordering the pre-registration depends on
was true and **not auditable**. Found by checkpoint. Added here rather than
silently, because a timestamp inserted without a note is worth less than none.

Everything below is recoverable without asking me: git records the commit time,
and the overlay roots are directory names on the node with the generation time
inside each `mounts.json`.

```
09:38:06   predictions committed                     d71d765
09:39:37   null overlay generated                    overlay/20260904_093936
09:39:38   degraded (2 ms) overlay generated         overlay/20260904_093937
10:03:32   stock control arm, first step
10:23:59   null arm, first step
10:44:50   degraded (20 ms) overlay generated        overlay/20260904_104449
10:51:01   degraded (20 ms) arm, first step
```

Hold: job **109496**, node **crsuse2-m2m-047**. **Every element of the experiment
postdates the commit**, the earliest by 91 seconds.

**One honest gap in the chain.** The numbers the predictions are *calibrated*
against — ITL 10.02 ms, std 0.66, 123 requests — came from an **exploratory**
stock arm run earlier the same morning, before the predictions. That ordering is
the right one (measure, then predict, then run), but **that arm's artefacts were
overwritten by the control run's stock arm**, so the calibration source is quoted
here rather than recoverable. The control run's own stock arm measured ITL
10.14 ms, std 0.67 — consistent, and it is the one the comparisons used.

## What a green pair does and does not license

**Does:** the plumbing works end to end — two bring-ups, the mount, the
patch-live evidence, the ordering, the comparison, the packup — and the
instrument responds correctly to *both* no-effect and a real effect.

**Does not:** anything about an optimisation. **No optimisation is claimed, none
was installed, and a green null run must never be quoted as "module 5 works".**
It is "the plumbing works and the gate does not hallucinate a difference".

### 5. Executed 2026-09-04 — what it proved, and the two corrections it forced

Run twice on `crsuse2-m2m-006` (job `109260`, eight cards at 0 %, read at bind
time rather than carried). **Both corrections are to the four sections above,
written an hour earlier by me.**

**Correction 1 — "the command, in full" was not in full.** The first run died in
11.7 s on `aiperf_replay.sh:38: AIPERF_TRACE: parameter null or not set`. That is
**not a defect**: `shared.yaml:129` declares `E2E_AIPERF_TRACE: '${aiperf_trace:-}'`
with no default *on purpose*, so an operator who supplies no trace gets a loud
failure instead of a silent fallback to somebody's debugging path. The guard
worked exactly as designed and my command omitted the var — **the same "a diff is
the one shape nobody can paste" failure this section was written to prevent,
committed in the section that prevents it.** The command above now carries:

```sh
--var aiperf_trace=/shared_nfs/yihou/agent_sys/debugging/profiling/conversation_trace.jsonl
```

**Correction 2 — a mocked m1 deploys a *stub*, so this cannot verify the
profiled line.** §2 said `--var mock_stages=m1,…` gives m2 "a real kit" and that
m2 is therefore parallel-safe. The kit *is* real as an artefact; what its
`deploy.sh` brings up is not. Measured — `deployment.json` on the node reads
`"container": "stub_yihou_e2e_flow_pmoff"`, and `deploy.log` says **`ready after
1s`**, which no engine loading 27 B of weights across four cards can do.

So the honest split:

| | mocked m1 (parallel) | needs a real kit |
|---|---|---|
| kit read, scripts located, three scripts present | **proved** | |
| `line.sh` lifecycle: deploy → wait_ready → load → teardown → reclaim | **proved, both lines** | |
| `profiling_mode_off.bench_result` sealed **valid**, `check_bench_result` + `check_command_parses` + `check_environment` all PASS | **proved** | |
| aiperf replay against a live endpoint, real artefacts, `AIPERF_OK` | **proved** | |
| the profiler capture, `profile_result`, `kernel_table` | — | **yes** |
| any number worth quoting | — | **yes** |

**m2 is parallel-safe for its wiring and not for its measurement.** That is a
narrower claim than §2 made and it is the true one.

**The capture's refusal is a negative control I did not plan and would not have
thought to write.** Against the stub, `capture.log` reads:

```
===== 1/6 preflight =====
  ABORT: /mnt/…/pmon/profiles is not mounted rw in stub_yihou_e2e_flow_pmon
```

It named the mount, the host side and the container, and refused before opening
a window — so *"the profiler produced nothing"* arrived as a stated reason rather
than as an empty trace that `check_trace_coverage` would have had to catch three
tasks later.

**Nothing leaked.** Both lines' `teardown.log` recorded `processes_stopped: 2`
and `reclaim.log` ran; the node afterwards had no container of mine and all
eight cards at 0 %. An `aiperf_serves-*` container running there at the time was
**not mine** — its mounts name run `094614-790b14` and it was created four
minutes after my run ended, which is the reading that settled it. The name alone
would have said the opposite.

**Cost, now measured**, replacing the "no total" above for the mocked path:
`09:32:08 → 09:42:29`, **10m21s** for deploy_kit + both lines against a stub, of
which the clean line's load is the sealed 180 s window. **A real-engine run will
be longer by two model loads and is still unmeasured.**

---

# The rung-4 launch line, as a whole command

**Rung 4 is `--var mock_stages=m5`: m1 through m4 real in one run.** So like
rung 5 it does **not** consume the rung below's artefacts — m3 runs live and
produces its own workset. What it takes from rung 3 is **launch-line values**,
read by a person.

## 1. Read these from rung 3's run before typing anything

```sh
RUN=<the rung-3 run directory under /home/yihou/agent_sys_runroot/runs/>
grep -E '^  (tp_size|node|image|image_id|model_path|model_name):' \
  "$(find "$RUN" -path '*items/codes/environment.yaml' | head -1)"
```

**Read, not carried:** `image`, `tp_size`, `model_name`, `model_path`. They are
m1's to mint and rung 4 mints them again — reading rung 3's is how the two runs
stay comparable, not how they are wired.

**Read from the workset, not chosen:** `impl_flag` and `report_flag`. m3 pins
the entrypoint flag spelling as data (`entrypoints.*.flags`) and both my
producer and `check_speedup_substantiated` take it from there. **Passing them on
the command line is how the two ends get edited apart** — the failure is silent
and produces a ratio of 1.000.

**Left empty unless the workset carries more than one operator:**
`workset_operator`. `pick_operator` refuses ambiguity by name rather than
guessing, so an empty value is correct while there is one operator and a
refusal the moment there are two.

## 2. The command

```sh
python3 -m agent_sys.cli.main run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /home/yihou/agent_sys_runroot \
  --var jobid=<the hold> --var node=<the node> --var node_ip=<MEASURED> \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=<the image rung 3 recorded> \
  --var tp=<the kit's tp_size> \
  --var mock_stages=m5 \
  --var gpu=<a card the deployment holds> --var measure_gpu=<the same card> \
  --var work_root=/mnt/m2m_nobackup/yihou/e2e_flow \
  --var scratch_root=/mnt/m2m_nobackup/yihou/e2e_flow/kfo \
  --var container=yihou_e2e_flow \
  --var forge_max_hours=3.0 \
  --var transport_env=SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR
```

**Verified to load, not merely written**: this exact form with representative
values returns `6 tasks in the graph; nothing was dispatched` from `show`,
2026-09-04.

**`gpu` and `measure_gpu` are the same card and both are named.** One arg drives
both spellings inside `check_speedup_substantiated`; on the launch line they are
two vars and nothing checks they agree. **And the card must be one the
deployment was given** — `run_in_container.sh` refuses a request outside the
container's `HIP_VISIBLE_DEVICES`, because the pin is an env default and
`docker exec -e` would otherwise override it onto a co-tenant's card and return
a number.

**`forge_max_hours` at or below 2.0 is a declared smoke test**, not a short
campaign: `30_run_forge.sh` writes `degraded` and says the analysis is
static-only. A rung-4 result at 2.0 is not a rung-4 result.

## 3. What has no source yet — and it is blocking

**`forge-loop` is not installed anywhere this run can reach.** Measured
2026-09-04:

```
node host   command -v forge-loop kernel-agents   ->  rc=1, nothing
image       docker run … command -v forge-loop     ->  NOT_IN_IMAGE
```

The workset's one-liner is `exec forge-loop --invocation-spec-file … --driver …`
(`operators/<op>/run_forge.sh`, generated by m3), and `30_run_forge.sh` runs it
with `sh "./$ONELINE"` **on whichever host the body runs on** — it does not go
through `run_in_container.sh`. So the campaign has no executable on either side
of the boundary.

**`kernelforge_repo` looks like the answer and is not.**
`steps/m4_kernel_opt.yaml:358` declares `KFO_KERNELFORGE_REPO:
'${kernelforge_repo:-}'` and **nothing reads it** — the only other mention is a
row in `optimize_kernel.task/readme.md` calling it *"forge's own knobs"*. That
is T44's shape: a var emitted for a consumer that does not exist. It may well be
where the answer *belongs* — point it at a checkout and put its `bin` on
`PATH` — but today it is a declaration, not a mechanism.

**So rung 4 cannot run, and the gap is an installation rather than a defect.**
Nothing in the package is wrong; the campaign's binary simply is not on this
cluster. **This is the value to establish before a node is held**, because every
other rung-4 unknown is cheap by comparison and this one is not answerable from
inside the package at all.

**What is NOT blocking, so it is not confused with the above:** `forge_model`
(`Claude-Sonnet-5[1m]`), `forge_snr_threshold` (30.0) and `forge_max_hours` all
have working defaults and are policy rather than plumbing.

# The rung-5 launch line, as a whole command

**Owner: m5.** Written now, while rung 1 is in flight, because the rungs are
serial and the alternative is improvising it at the moment a node frees. The
rung-2 section above is the shape this follows.

**Rung 5 is `--var mock_stages=none`: every stage real, in one run.** So unlike
rung 2, it does not consume rung 4's *artefacts* — it produces its own. What it
takes from rung 4 is the **launch-line values**, and the distinction matters
because those are read by a person, not by the graph.

## 1. Read these from rung 4's run before typing anything

```sh
RUN=<the rung-4 run directory under /home/yihou/agent_sys_runroot/runs/>
grep -E '^  (tp_size|node|image|image_id|model_path):' \
  "$(find "$RUN" -path '*items/codes/environment.yaml' | head -1)"
```

`image` and `tp` should match what rung 4 ran on, **not because the graph checks
it, but because rung 5's whole output is a comparison** — two arms measured
against each other and the stock arm against m2's numbers
(`stock_vs_m2_tolerance`, default 0.10). Change the deployment between rungs and
the comparison is still computed, still passes its bars, and means less than it
appears to.

## 2. The command

```sh
python3 -m agent_sys.cli.main run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /home/yihou/agent_sys_runroot \
  --var jobid=<the hold> --var node=<the node> --var node_ip=<MEASURED> \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=<the image rung 4 recorded> \
  --var tp=<the kit's tp_size> \
  --var mock_stages=none \
  --var measure_gpu=<free GPUs on the node, see below> \
  --var bench_rounds=3 \
  --var work_root=/mnt/m2m_nobackup/yihou/e2e_flow \
  --var container=yihou_e2e_flow \
  --var transport_env=SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR
```

**Verified to load, not merely written**: this exact form, with the placeholders
filled by representative values, returns `6 tasks in the graph; nothing was
dispatched` from `agent-sys show`, 2026-09-04.

**Every `m<N>_agent=runner` is gone, and that absence is the whole rung.** Five
of them, not one. A rung-5 command carrying any is a lower rung wearing rung 5's
name, and it will report a clean mock for the stage it silently kept.

**`node_ip` is measured, never derived** — `spur exec <job> hostname -I`. There
is no pattern: `-061` → `10.245.159.129`, `-031` → `10.245.144.239`, `-006` →
`10.245.151.128`. Deriving it cost a bring-up.

## 3. What `show` cannot catch, which is most of what will go wrong here

`show` catches one class only: a `${NAME}` with no default and no value. Every
item below loads clean and fails later, so this list is the check.

| var | rungs 0–4 | rung 5 | if it is wrong |
|---|---|---|---|
| `adhoc_cases` | `0` (or `1`) | **omit — default 3** | `check_acceptance`'s ad-hoc arm passes on zero cases |
| `m<N>_agent` | `runner` | **all five absent** | that stage runs mocked and reports green |
| `transport_env` | required | required | refuses in 1 s, looking exactly like a deployment failure |
| `integration_min_requests` | n/a | **omit — default 50** | see below |
| `expect_ranks` | `2` | omit (defaults 8) or track `tp` | m2's, listed so it is not carried in |

**`transport_env` is not a package variable at all** — it is consumed by the
runner, so it appears in no yaml and `show` is structurally unable to see it
missing. That is why it cost three rung-0 runs.

## 4. Three values that are m5's, and each is a finding rather than a choice

### `adhoc_cases` — rung 5 is the first run that may not lower it

Every launch line to date passes it below the floor: three pass `0` and one
passes `1`. **At `0` the producer generates no cases and the validator's floor is
`0`** — both are `'${adhoc_cases:-3}'` (`E2E_ADHOC_CASES` and
`min_adhoc_cases`), so the arm grades nothing and passes.

That was **forced, not lazy**: the cases are invented by m5's `kind: ai` agent,
and at rungs 0–4 `m5_agent=runner` means there is no agent to invent them. Rung 5
is the first rung where the knob *can* be honoured, which makes it the first
where lowering it is a choice.

**So if rung 5 also lowers it, `check_acceptance`'s ad-hoc arm will have been
read-but-never-exercised for the entire effort** — the same shape as a bar that
is printed and never applied. Omit the var.

### `require_runtime_marker` — rung 5 may not be runnable at all yet

The default is `true` (mine, flipped 2026-09-04 on measured evidence: a
hash-perfect overlay that never executed measured identical to stock). With it
true, an overlay declaring no `runtime_marker` is refused by `check_patch_live`.

`mock_adapt` writes a marker only when the workset declares a `public_symbol`,
and **a `call_site_fragment` operator has none** — so for the current workset
there is no legal path through that check. This is M5.1.1 reaching the same wall
from a second direction, and it is **with the user**.

**Say this before the node is held, not on it.** If M5.1.1 is unresolved when a
rung-5 window opens, the options are: resolve it; or run with `--var
require_runtime_marker=false` and **record in the run that the patched arm's
numbers cannot distinguish *executed* from *mounted and never entered*** — which
is precisely the distinction that validator exists to draw, so it is a
documented downgrade and not a workaround.

### `integration_min_requests` — new as of `47db1fc`, and rung 5 is where it bites

It was `min_requests`, shared with m2's `check_bench_result`. Split because the
override was shared while the reason to override is stage-local. **Rung 5 is the
first run where getting it wrong is expensive**: `--var min_requests=0` no longer
touches stage 5, and `--var integration_min_requests=0` no longer touches
stage 2. Omit both; the default is 50 on each side.

## 5. Values with no recorded source, which is a finding now and a block later

Named because a value rung 4 does not record is one somebody invents on the node.

- **`measure_gpu`** (default 4) — how many cards each arm takes. **Nothing in any
  sealed artefact records how many GPUs were free**, and the two rung-5 arms need
  their own cards beside whatever the host already carries. Measured on 217 the
  hard way: GPUs 0–3 held by another tenant's non-docker processes. **Read it
  from the node at hold time and write it into the run's notes**; do not carry a
  number between rungs or between nodes.
- **`bench_rounds`** (default 1) — the command above says `3`, and that is a
  judgement, not a measurement. One round cannot separate a real change from
  run-to-run spread; the noise floor is `1.96·√2·rsd/√n`, so `n` is the only term
  the launch line controls. **Three is the smallest `n` that makes the floor
  reportable at all**, and `check_no_regression` will mark the comparison
  `uninterpretable` rather than pass it if the floor exceeds the bar. If a
  rung-5 window is short, cutting rounds is the first thing to cut and the
  `uninterpretable` verdict is the honest consequence.
- **`measure_container`** (default empty) — the two arms name themselves from it.
  No source, and it is an identifier bound on a shared host, so leaving it
  defaulted on a busy node is how two runs collide.

## 6. What rung 5 proves that rung 4 did not, stated so a green cannot overclaim

Rung 5 is the only rung where **`integrate_and_verify` runs**, and with it the
two-arm bring-up, `check_acceptance` on generated cases, and
`check_no_regression` on numbers nobody chose.

Three things will still be untested after a green rung 5, and they should be
named in whatever reports it: `apply.py`'s **generic dropped-names refusal** and
its **no-op gate** (gates 8 and 9, never reached — a run that gets past
`apply_patch` is a run where neither fired), and every refusal path of the seven
m5 validators that a passing run by construction does not take.

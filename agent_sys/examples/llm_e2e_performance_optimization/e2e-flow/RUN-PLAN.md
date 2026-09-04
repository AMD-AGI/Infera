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

1. **`m2`'s interpreter sweep** — `/home/yihou/ws_handoff_refine_m2/interpreter_sweep.py`. About a minute. **Treat a clean result as a gate, not a formality**: all four bugs in that class were introduced by bodies written *after* the previous sweep, so the sweep is only worth its cost when it is re-run.

   *Moved off `/shared_nfs` on 2026-09-04 — that export is mounted `ro` on the login node and `rw` on a held node, same volume, two mounts. The copy at `ws_handoff_refine/m2/` is **frozen** and predates the input wiring; editing it changes no result.*

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

## The one measurement to take at rung 5 that nobody has

`todo.md` T7 wants the within-arm round-to-round spread on a **quiet** node. Both holds carry other tenants today. If a quiet window appears, take it: it is the number that decides whether 5% / 10% are right, and the previous round widened them to 35% / 30% on a cross-instance artefact for want of it.

---

## Standalone verification — m4 (`optimize_kernel`)

The user's redirection, 2026-09-04: *"e2e串通也可以通过先单独运行每个模块保证单独通(也可以并行)"*
— verify each module on its own rather than only through the ladder. This is m4's.
**m4 is the stage that most needs it: it is the only one that has never executed
in any graph, on any rung.**

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
  --var eval_max_tokens=256 --var eval_examples=5 \
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
| **probe** | **2062 s** | **2001 s** | **yes, `eval_max_tokens` — and this is the whole problem** |
| lm_eval | 23 s | 428 s | yes, `eval_examples` |
| bench_r1 | 44 s | 161 s | yes, `trace_end_ms` |
| **total** | **3080 s** | **3244 s** | **~105 min for the pair** |

**`probe` is two thirds of an arm, and until today its cost knob could not be
set.** `measure.sh:149,157` reads `E2E_EVAL_MAX_TOKENS` and `E2E_EVAL_THREADS`
and **nothing declared either**, so the `:-2048` fallback always won. Now
declared on `e2e_integrator`; `runner` should mirror them.

**The reduced total is an estimate and is labelled as one.** `serve`, `smoke`
and `bench` are measured; the effect of `eval_max_tokens=256` on `probe` is an
extrapolation nobody has taken. **Do not quote a reduced wall time as measured
until one run has produced it** — that is the mistake this package keeps
cataloguing, and predicting it here would be a fourth entry.

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

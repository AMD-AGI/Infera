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
`2`; at rung 2 the trace comes from a **real TP-8 bring-up**, and passing `2`
makes `check_trace_coverage` refuse a perfectly good eight-rank capture **after
a full bring-up and a three-minute load**. It fails loudly, so it costs a run
rather than a wrong number — which is the good version of this mistake and still
a run.

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

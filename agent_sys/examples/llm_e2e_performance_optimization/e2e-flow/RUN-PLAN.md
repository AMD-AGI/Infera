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

`expect_ranks` is the one to watch: it is a fact about **the artefact being
graded**, not about the run. The sealed capture is TP-2, so rungs 0 and 1 need
`2`; at rung 2 the trace comes from a **real TP-8 bring-up**, and passing `2`
makes `check_trace_coverage` refuse a perfectly good eight-rank capture **after
a full bring-up and a three-minute load**. It fails loudly, so it costs a run
rather than a wrong number — which is the good version of this mistake and still
a run.

## Before rung 1, and again before every rung

1. **`m2`'s interpreter sweep** — `/shared_nfs/yihou/agent_sys/ws_handoff_refine/m2/interpreter_sweep.py`. About a minute. **Treat a clean result as a gate, not a formality**: all four bugs in that class were introduced by bodies written *after* the previous sweep, so the sweep is only worth its cost when it is re-run.
2. **`agent-sys show`** — under a second.
3. **The node's state, before and after**: `docker ps` and the port band. Every identifier this package binds carries a run tag; **check the tag before killing anything.** Measured 2026-09-03: a validator's teardown crashed and warned that ports might be held, and the ports that were held belonged to *a different owner's run in flight*. Killing them would have destroyed live work.

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

# `run_profiling_mode_off`

Measure the deployment with the profiler detached and decode CUDA graphs **on**.

**These are the numbers that mean something.** Stage 5's stock arm has to
reproduce them (M5.1.3.1), and they are the only throughput in this flow worth
quoting. The other line runs with graphs off and measured eight times slower on
the reference pair — 15.65 ms mean inter-token latency against 124.98 ms — which
is the intent of that line and not a regression in it.

**This task brings its own service up and tears it down** — M2.5 forbids
splitting bring-up from use across two agents: *"agent A 去把服务部署好，agent B
去使用：这是不被允许的"*. So there is no `serve_*` task in this package and no
`deployment_*` handoff.

**And it carries no deployment recipe of its own** — M2.3/M2.4: *"module 1 的
output 已经包含了如何部署的全量信息"*. The bring-up is `scripts/deploy.sh` out of
the `deploy_kit` handoff, the readiness criterion is that kit's
`wait_ready.sh`, and the teardown is its `teardown.sh`. This task supplies the
configuration, the load and the evidence, and nothing about how to start an
engine.

## Inputs and outputs

| | |
|---|---|
| in | `deploy_kit` — the environment record and the deployment scripts |
| out | `profiling_mode_off.bench_result` |
| graded by | `check_environment`, `check_bench_result` |

## STEPS

Executed in order by `entry.sh` → `../load/line.sh`. Each step's acceptance
criterion is what the next step is allowed to assume; a step that does not meet
it aborts, and the trap in step 3 runs.

1. **Mock, if this stage is mocked.**
   `bash "$PKG/assets/lib/mock.sh" stage2-profiling profiling_mode_off.bench_result:aiperf_baseline`
   *Accept:* exit 0 and the output slot holds the 16 sealed files. *Exit 3 means
   this stage is not mocked* — fall through to step 2. Any other non-zero is a
   failure.

2. **Locate the kit and agree with its environment record.** Exactly one packup
   directory with a `scripts/` under `items/codes/`; `deploy.sh`,
   `wait_ready.sh` and `teardown.sh` all present and visible on the node.
   *Accept:* one directory — zero means the producer wrote its kit somewhere the
   content type does not put it, two means a consumer has to guess which one
   worked — and `fixed.node` in `items/codes/environment.yaml` equal to
   `$E2E_NODE`. A node mismatch aborts: this line would otherwise measure a
   machine the kit does not describe, and every number would be filed under the
   wrong environment.

3. **Set the kit's runtime contract.** `E2E_KIT_RUN_TAG` (this line's own tag),
   `E2E_KIT_PORT_BASE`, `E2E_KIT_WORK_ROOT`, and both engine seams **empty** —
   which is what makes this the clean arm: a caller that sets neither gets the
   kit's own bring-up byte for byte.
   *Accept:* nothing to check yet; the tag is verified against the handshake in
   step 6. `check_deploy_kit` has already refused any kit that does not read
   these, so the dependency is a gate on m1's output rather than a hope.

4. **Register teardown before bring-up, not after.**
   *Accept:* the trap is installed. It is installed first because a line that
   fails between bring-up and load must not leave a TP-8 engine holding every
   GPU on a node four other owners share. It is the kit's own `teardown.sh`
   scoped by this line's run tag, so it removes what this invocation created and
   never something a co-tenant owns (CONTRACT §5.2).

   *No preflight of our own.* The kit owns that; a second opinion about whether
   the checkpoint is readable is a second thing to keep in step with the
   deployment it describes.

5. **`deploy.sh`**, then **`wait_ready.sh`**.
   *Accept:* two separate criteria, on purpose. `deploy.sh` returning 0 says the
   launch commands were accepted; `wait_ready.sh` returning 0 says the service
   answered. A readiness wait that exits 0 when it gave up turns "the model
   never loaded" into "the benchmark measured nothing", and the second is
   discovered three stages later. Cold start is dominated by the checkpoint load
   off shared storage.

6. **Read the handshake**, `${E2E_KIT_WORK_ROOT}/deployment.json`, through the
   transport — it is on the node's local disk.
   *Accept:* `endpoint`, `container` and `run_tag` all present, and `run_tag`
   equal to what step 3 asked for. A mismatch means either a stale
   `deployment.json` is being read or the kit ignored the tag — and teardown is
   scoped by that tag, so a wrong one is a leaked deployment. `endpoint` is the
   **product** endpoint, the router and not the engine's own port.

7. **Replay the trace**, `E2E_CAPTURE=0`, and assemble the handoff.
   *Accept:* `AIPERF_OK` in the replay log, and `items/result/summary.json`
   reporting at least `min_requests` requests. AIPerf exits 0 over an empty
   window, so the exit code is not the criterion — the request count is.

8. **Write the environment record** into the output, inherited from the kit with
   this line's runtime substituted.
   *Accept:* `env_render.py` prints the path it wrote. It validates before it
   writes, so nothing is produced if the record would be malformed. Inherited
   and never rebuilt: m1 is the sole producer, and a stage that re-derived the
   record could differ from m1's with nothing to notice.

9. **Tear down** (the trap from step 4).
   *Accept:* `teardown.sh` exits 0. Failure here does not fail the task — the
   evidence is already sealed — but it is logged, because a leaked deployment is
   the next round's port collision.

## If you are an AI agent running this task

**This section exists because a `kind: ai` task never runs `entry.sh`.** The
program body below it enforces guards that you now enforce yourself, and reads
values from the environment that no environment can tell you about
(`agent/runner.py:801` — *"an env var cannot instruct an agent"*). Declaring a
variable makes it reachable; only this page makes it used.

### Three values the shell reads and you would not know to look for

**1. The load is a replay of a recorded conversation, not a synthetic sweep.**
There is a trace file — a JSONL of `{timestamp, input_length, output_length,
hash_ids}` records — and the load sends requests *at the timestamps it records*,
with `ignore_eos` so each response runs to the recorded output length rather
than to the model's own stopping. That is what makes two lines comparable: same
arrivals, same lengths, so a difference between them is the engine and not the
load.

Without it there is nothing to send and the load aborts (`aiperf_replay.sh`
requires it). **This is not hypothetical**: on 2026-09-05 a bring-up succeeded
completely — worker serving, six verification probes green, a real completion —
and the run then died at the load because no trace had been passed. The
operator supplies the path; your job is to make sure it reaches the load step,
and to stop with a clear message if it has not, rather than loading something
else. The default is empty **on purpose**, so that a missing trace is loud.

**2. Which GPUs to use is a precedence, not a default.** Three sources, in this
order, and the order is the point:

  1. an explicit operator choice for this run;
  2. failing that, **the set the deployment kit records having actually taken** —
     the kit knows which cards it bound;
  3. failing that, the kit's own built-in default.

Taking the default when the kit recorded something is the failure mode: it was a
hardcoded `0,1,2,3` for a while, so every bring-up took the same four cards
whatever was free, on a node five people share. **Read the kit's record before
falling back**, and never assume the default is a description of the node.

**3. A kit may be a replay rather than a fresh bring-up, and the numbers must
say so.** The deployment record carries a provenance field naming the source it
was replayed from, when it was replayed. A number measured against a replayed
kit is not comparable to the deployment that kit originally described, and a
reader meets the number long after any message that qualified it. **Carry the
provenance into the output** rather than into a log line.

### Guards the program enforced and you must now enforce yourself

`assets/load/line.sh` holds **ten refusals with their own diagnostic**, plus
seven required-variable checks. **Three paragraphs do not restore ten guards**,
and this page does not pretend otherwise — the list is here so you know what
stops being true the moment this task is promoted:

| what it refuses | why it matters |
|---|---|
| the kit carries no environment record | nothing downstream can say which machine produced the numbers |
| the kit's scripts are missing | a kit that cannot deploy is not a kit |
| **the kit was taken on a different node than this run targets** | the numbers would describe a machine nobody used |
| `deploy.sh` wrote no handshake | the bring-up did not report where it is |
| the handshake is unusable, or its run tag is not this line's | teardown is scoped by that tag; a wrong one leaks a deployment |
| **the endpoint carries no port to load against** | benching a port nothing listens on looks like a dead engine |
| the router does not answer | a load against nothing produces a clean, empty, wrong result |

**Bold the two that are silent when they fail.** A wrong node and a wrong port
both produce plausible output — a full set of numbers, from the wrong machine or
from nowhere — where the others produce an obvious absence. If you enforce only
two of these, enforce those.

## Watch out

- **`min_requests` is 50, not 100.** Measured on this trace: 346 requests in a
  120 s window, 166 in 60 s, ~80 in 30 s. A bar of 100 once failed a perfectly
  working shortened replay, invalidated both of that task's handoffs and stopped
  the graph. The bar must sit below any legitimately shortened window, or it
  stops being "did anything get sent" and becomes a second, undeclared
  constraint on window length.
- The trace saturates a single-node MIX deployment, so the latency percentiles
  describe a queue and not the model. They are a load, not an SLA.
- `ignore_eos` is sent, so output length follows the trace rather than the
  model's own stopping.

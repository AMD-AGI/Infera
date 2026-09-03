# `run_profiling_mode_off`

Measure the deployment with the profiler detached and decode CUDA graphs **on**.

**These are the numbers that mean something.** Stage 5's stock arm has to
reproduce them (M5.1.3.1), and they are the only throughput in this flow worth
quoting. The other line runs with graphs off and measured eight times slower on
the reference pair — 15.65 ms mean inter-token latency against 124.98 ms — which
is the intent of that line and not a regression in it.

**This task brings its own service up and tears it down.** There is no
`serve_*` task in this package and no `deployment_*` handoff, because
M2.3/M2.4/M2.5 forbid splitting bring-up from use across two agents:
*"agent A 去把服务部署好，agent B 去使用：这是不被允许的"*. What it brings up is
described by m1's `deploy_kit`, which it reads rather than re-deriving.

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

2. **Read m1's environment record and agree with it.**
   `items/codes/environment.yaml` inside `$AGENT_SYS_INPUT_DEPLOY_KIT`.
   *Accept:* the file parses, and `fixed.node` equals `$E2E_NODE`. A mismatch
   aborts: this line would otherwise measure a machine the kit does not
   describe, and every number would be filed under the wrong environment.

3. **Register teardown before bring-up, not after.**
   *Accept:* the trap is installed. It is installed first because a line that
   fails between bring-up and load must not leave a TP-8 engine holding every
   GPU on a node four other owners share. It removes only the container this
   line created — never a name it did not create (CONTRACT §5.2).

4. **Check the checkpoint is readable on the node.**
   `test -r "$E2E_MODEL_PATH/config.json"`, on the node.
   *Accept:* exit 0. This costs a second and saves a cold-start's worth of
   waiting for a failure that says something less specific.

5. **Bring the engine up**, `CUDA_GRAPH=1 PROFILE=0`, in a container named for
   this line and on this line's ports.
   *Accept:* `MIX_UP_OK` in the bring-up log and the router answering `/health`.
   Cold start is dominated by the checkpoint load off shared storage. A worker
   process that dies is reported with the last 40 lines of its own log rather
   than waited out.

6. **Replay the trace**, `E2E_CAPTURE=0`, and assemble the handoff.
   *Accept:* `AIPERF_OK` in the replay log, and `items/result/summary.json`
   reporting at least `min_requests` requests. AIPerf exits 0 over an empty
   window, so the exit code is not the criterion — the request count is.

7. **Write the environment record** into the output, inherited from the kit with
   this line's runtime substituted.
   *Accept:* `env_render.py` prints the path it wrote. It validates before it
   writes, so nothing is produced if the record would be malformed.

8. **Tear down** (the trap from step 3).
   *Accept:* the container and its etcd are gone. Failure here does not fail the
   task — the evidence is already sealed — but it is logged, because a leaked
   container is the next round's port collision.

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

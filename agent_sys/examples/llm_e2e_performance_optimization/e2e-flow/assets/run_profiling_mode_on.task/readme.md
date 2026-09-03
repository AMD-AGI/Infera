# `run_profiling_mode_on`

Capture a profile of the deployment under a known load, with decode CUDA graphs
**off** and the profiling control plane **on**, and rank the kernels in it.

**Graphs off is the point, not a second variable.** With graphs on the profiler
records one graph launch instead of the kernels inside it, so a graphs-on
capture cannot attribute time to a kernel — which is the only thing this line
exists to produce. The cost is that its throughput is not a control for
anything; that number comes from `run_profiling_mode_off`.

**Three outputs from one task, and they are not separable.** The profiler window
has to fall *inside* the load window, and the ranking reads the trace this task
just wrote. As sibling tasks agent_sys would schedule them concurrently with
nothing to synchronise them, so lining them up would need a rendezvous file — an
edge the graph cannot see and cannot report on.

Like the other line, it brings its own service up and tears it down (M2.5), and
like the other line it carries **no deployment recipe of its own** (M2.3/M2.4):
the bring-up is `scripts/deploy.sh` out of the `deploy_kit` handoff. What makes
this line different from the clean one is three values passed into that kit and
nothing else — which is the property that keeps the two arms comparable.

| seam | this line | the clean line |
|---|---|---|
| `E2E_KIT_ENGINE_EXTRA_ARGS` | `--disable-cuda-graph` | empty |
| `E2E_KIT_ENGINE_EXTRA_ENV` | `SGLANG_TORCH_PROFILER_DIR=…` | empty |
| `E2E_KIT_ROUTER_EXTRA_ARGS` | `--enable-profiling` | empty |

The engine seams reach the **worker** process, and `--enable-profiling` is a
**router** flag, which is why it is a third seam rather than folded into the
first. The profiler directory must also be somewhere the *engine container* can
write: SGLang writes the trace to a path the container sees, and without that
docker creates the directory in the container layer and the capture reports
success while the host sees nothing.

## Inputs and outputs

| | |
|---|---|
| in | `deploy_kit` |
| out | `profiling_mode_on.bench_result`, `profiling_mode_on.profile_result`, `profiling_mode_on.kernel_table` |
| graded by | `check_environment`, `check_bench_result`, `check_trace_coverage`, `check_kernel_table` |

## STEPS

Executed in order by `entry.sh` → `../load/line.sh`.

1. **Mock, if this stage is mocked.** Three kinds, and `kernel_table` is
   **reshaped after it is copied**: the sealed sample is a `reproducible`
   handoff and this kind is `structured_text`, so `../lib/m2_reshape.py` moves
   the record to `items/text.json`, Magpie's export to `items/table.csv`, the
   schema to `items/schema`, and rewrites the README with the sections the
   content type requires. *Accept:* exit 0 and four items in the reshaped
   output. Exit 3 means this stage is not mocked — fall through.

2. **Locate the kit and agree with its environment record.** Same as the other
   line: one packup directory with `deploy.sh`, `wait_ready.sh` and
   `teardown.sh`, and `fixed.node` equal to `$E2E_NODE`.

3. **Set the kit's runtime contract**, with the three seams in the table above
   set rather than empty. This is the only place the two lines differ.

4. **Register teardown before bring-up.** Same reason, and it matters more here:
   this line runs longer.

5. **`deploy.sh`**, then **`wait_ready.sh`**, then **read the handshake**.
   *Accept:* as the other line — `deploy.sh` accepted the launch, the service
   answered, and `deployment.json` carries `endpoint`, `container` and a
   `run_tag` equal to what step 3 asked for.

6. **Probe the profiling control plane, and abort if it is off.**
   POST the admin profile start route with a role that cannot exist.
   *Accept:* **400**. The gate is checked *before* the role is validated, so 400
   means profiling is on, 403 means it is not, and neither disturbs a running
   profile. **A 403 aborts**, because every capture below would then produce
   nothing and report success — there is no error the caller would see. Any
   other status is a warning and the capture's own `CAPTURE_OK` becomes the
   criterion.

7. **Replay the trace and cut two windows inside it**, `E2E_CAPTURE=1`.
   - the **measurement window**: `E2E_WINDOW_S` seconds after `E2E_WARMUP_S` of
     warm-up, `with_stack` **off**, one file per rank.
     *Accept:* `CAPTURE_OK`, and the per-rank manifest showing GPU kernels above
     the floor on every rank.
   - the **stack window**: `E2E_STACK_WINDOW_S` seconds, `with_stack` **on**,
     `E2E_STACK_RANKS` rank files kept. *Accept:* `CAPTURE_OK` — **and failure
     here is not fatal.** The ranking is complete without it and the launcher
     block is an enrichment. Whether a round missing it is acceptable is
     `check_trace_coverage`'s call, which is where "did we get what we needed"
     belongs.
   - the load itself: *accept:* `AIPERF_OK` and at least `min_requests`
     requests.

8. **Rank the kernels.** Magpie over the measurement window's traces, on the
   node, into a staging directory; then reshaped into the `structured_text`
   layout. *Accept:* `gap_analysis.csv` non-empty, `text.json` validating
   against `assets/schemas/kernel_table.schema.json`, and the shares summing to
   a whole run. Minutes, not seconds: every event in every rank is parsed.

9. **Write the environment record** into all three outputs, inherited from the
   kit with this line's runtime substituted.

10. **Tear down** (the trap from step 4).

## Watch out

- **The two windows are not interchangeable and must not be aggregated.** The
  measurement window is what every number about this round comes from. The stack
  window is short, was taken afterwards, and holds only some of the ranks; it
  exists so the ranking can name the Python frame that launched each kernel.
- **`with_stack` costs 13x the bytes**, measured: the same workload profiled
  with and without stacks came to 2,996,700 against 228,553 bytes, from 9,565
  `python_function` events against none. Kernel counts and total kernel time
  were identical across the pair, so stacks do not change *what* is measured —
  but at ~60 MB per rank for the measurement window, stacks on would be about a
  gigabyte per rank. That is why the split exists, and why
  `check_trace_coverage` fails a measurement window that carries stacks.
- **`record_shapes` is on in both windows**, and Magpie's `Input Shapes` column
  exists only because of it. Without it every roofline in stage 3 is impossible,
  and the table looks entirely normal.
- Set `--var stack_window_s=0` to skip the stack window deliberately. Then
  `check_kernel_table`'s `min_launchers_in_top_n` must be 0 too, or the round is
  being graded for something it was told not to take.

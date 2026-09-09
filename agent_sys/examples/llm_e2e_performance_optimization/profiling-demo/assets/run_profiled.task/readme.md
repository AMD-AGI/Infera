# run_profiled

Replay the same trace against the graphs-off deployment and cut a torch-profiler
window out of the middle of the load. Two handoffs: the report and the traces.

## Why this is one task and not two

The profiler window has to fall inside the load window. As two sibling tasks
agent_sys would schedule them concurrently with nothing to synchronise them, so
lining them up would need a rendezvous file on disk — a dependency the graph
cannot see and cannot report on. One task that starts the load, waits for the
engine to report an actual batch, and then cuts the window is the honest shape
for a coupling this tight.

## The window does not open on a timer

A cold AIPerf synthesises every prompt in the window before it sends anything,
which can take minutes. Starting a warm-up clock when the container starts would
put the profiler window in the middle of that, capturing an idle scheduler loop
and producing eight perfectly well-formed traces of nothing.

So the sequence is: start the load, poll the engine log until it reports a
prefill or decode batch, *then* start the warm-up clock, then open the window.

## Three details in the profile request that are not obvious

Each is documented at its site in `assets/load/capture.sh`, and each was learned
the expensive way in the reference kit:

- `with_stack` must be explicit either way. SGLang defaults it to true, which
  adds millions of `python_function` events. Measured on
  smci355-ccs-aus-n04-29 on 2026-09-01, profiling one workload twice in the
  engine image: 2,996,700 bytes of trace against 228,553, so **13.1× the
  uncompressed bytes and 16.5× gzipped**, from 9,565 `python_function` events
  against none. The kernel count and total kernel time were identical across the
  pair, so stacks cost bytes and change nothing that is measured.

  This task therefore cuts **two** windows. The measurement window runs with
  `with_stack: false`; a second window of `stack_window_s` seconds runs with it
  on, and only `stack_ranks` of its rank files are carried. That second window is
  what lets `kernel_scan` name the Python frame behind each ranked kernel — the
  only evidence that answers "which file do I edit" for a symbol like
  `main_kernel`, which is what TileLang names every kernel it generates.

  Two windows rather than one because at 60.5 MB per rank for 15 s, stacks across
  the measurement window would be roughly 1 GB per rank and 8 GB for the round,
  while resolution needs a few launches per kernel and nothing more.
- `record_shapes: true` is what populates Magpie's `Input Shapes` column. Without
  it the CSV looks entirely normal and no roofline can be built from it.
- `output_dir` must exist before the request. SGLang does not create it, and the
  export fails inside the profiler callback long after `start` answered 200 — so
  the symptom is an empty directory at the end and no error anywhere.

The stop request usually returns `ReadTimeout`. That is the router's own 30-second
read timeout, not a failed stop; the engine keeps writing. What decides is the
flush check, which polls the directory's byte count until it stops moving.

## What comes out

Measured: eight ranks, 60.5 MB each, 462 MB total for a 15-second window —
four times the reference kit's per-rank size, because a MIX deployment puts
prefill and decode kernels in one trace and this load saturates the engine. Each
rank held about 181,000 GPU kernel events over a 16.5-second span.

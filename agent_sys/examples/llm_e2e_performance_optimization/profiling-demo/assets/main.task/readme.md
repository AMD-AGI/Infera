# main

A non-leaf. Its work is its subgraph, so it runs nothing itself and declares no
agent.

The finished graph is six leaves in two rounds plus two analysis steps:

```
serve_baseline -> run_baseline -> serve_profiled -> run_profiled -> kernel_scan -> packup
```

The two rounds exist because CUDA graphs are a start-up flag and cannot be
toggled on a live engine. Graphs on gives a throughput number worth quoting;
graphs off gives a profiler trace in which individual kernels are attributable,
because with graphs on the profiler sees one launch instead of the kernels inside
it. One deployment cannot be both, so there are two, and `serve_profiled` depends
on `run_baseline` rather than on `serve_baseline` — that edge says "the baseline
has been measured, it is safe to tear it down".

Today only `serve_baseline` is wired. See DESIGN.md section 10 for why the graph
is being filled in one leaf at a time rather than all at once.

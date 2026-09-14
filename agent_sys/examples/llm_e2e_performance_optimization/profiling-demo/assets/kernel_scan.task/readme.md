# kernel_scan

Rank every GPU kernel in the captured window by the CUDA time it owns, with
Magpie. This table is what the operator-selection stage of the wider pipeline
consumes.

## Why it still reaches the cluster

It reads a handoff and writes a handoff; it needs no GPU. It runs Magpie on the
compute node anyway, because Magpie's dependencies are installed there and the
login node's interpreter is not guaranteed to have them. Both hosts mount the
same NFS home, so the staged input and the handoff output are reachable from
either side.

## The one change from the reference invocation

`run_megapie_kernel_analyze.sh` writes into the trace directory. That works only
when the caller owns it — and a directory the engine container's root created on
a bind mount is not owned by the user running the analysis. So the output
directory is a separate argument here. This is the failure that stopped the
manual walk-through with `PermissionError` after the traces were already captured.

## What it costs

Measured 3 minutes 43 seconds for eight ranks of a 15-second window. Every event
in every rank is parsed, so the cost scales with the window and the load, not
with the file count.

## What comes out

Four files under `items/result/`, and they are four because they answer different
questions:

| file | what it is for |
|---|---|
| `gap_analysis/gap_analysis.csv` | Magpie's own export, unmodified. The artefact. |
| `top_kernels.json` | the head, so `check_kernel_table` can judge whether the table has one |
| `text.json` | **every** row, under the next stage's field names, with launcher frames merged in |
| `launchers.json` | the launcher resolution in full, including what it could not place and why |

`text.json` carries every row rather than the head because the next stage
classifies every kernel into a bucket before it sorts. Collectives held 79% of
GPU time in the sample profile, so handing it a top-25 would have discarded most
of the routable candidates along with the noise.

Measured on this deployment: 158 rows, with `main_kernel` at 15.21%,
`fused_moe_kernel` at 13.55% and AITER's `cross_device_reduce_2stage` at 12.97% —
the top three accounting for 41.7% of 154 seconds of aggregate self CUDA time.

That third one is worth noticing: an all-reduce holding an eighth of the engine's
GPU time is a TP-8 communication cost, not a kernel to optimise in isolation.

## Naming the Python frame behind each kernel

`assets/analyze/launchers.py` reads the stack window that `run_profiled` cut and
resolves each ranked kernel to the Python call site that launched it. The
algorithm is Hyperloom's `_trace_launcher_resolver`, reimplemented rather than
imported because Hyperloom is not on this cluster's compute nodes: pair a
`kernel` event to a `cuda_runtime` launch on `args.correlation`, take the
`python_function` frames whose span covers that timestamp, walk inward to the
first frame that is neither a JIT dispatch layer nor a decorator, and require
several probes to agree before reporting anything.

Two departures from Hyperloom's version, both deliberate:

- **`tilelang/jit/` and `tilelang/engine/` are added to the skipped-frame list.**
  Without them `main_kernel` resolves to TileLang's own dispatcher, which is
  neither editable nor specific to any one of the kernels it compiled. With them
  the walk continues out to the sglang backend that called it, which is the
  framework-level entry point the next stage's design asks for.
- **TraceLens elision handling is left out.** Hyperloom matches a symbol
  truncated with a trailing `...`; Magpie does not truncate, so prefix matching
  here would only create a way to bind a wrong kernel to a right-looking frame.

The resolution is not fatal when it comes up empty. A ranking with no launcher
block is the state this package shipped in before the stack window existed, and
the consumer falls back to searching for the symbol name. What the round must not
do is make the difference invisible, which is why `launchers.json` is written
either way and records the reason when the answer is "there were no stack
traces".

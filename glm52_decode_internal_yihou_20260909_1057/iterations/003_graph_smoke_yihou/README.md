# Iteration003: graph and warmup smoke

## Hypothesis
The successful scheduler-free eager MTP path also supports pinned target/draft/draft-extension graphs and warmup followed by exact-ISL reset.

## Changes from iteration002
Enable CUDA Graph at bs16; warmup3, measured maxsteps4,OSL32. Add CPU-tested state progression/layout/terminal guards and actual graph execution counters. Same weights/routing/physical KV initialization/optimization stack. Full command in command.txt; frozen source hashes/snapshots included.

## Pass criteria
Exit0; all rank accounting consistent; each measured step's worker length equals prior+actual accept count; bootstrap finite; actual target/draft/draft-extension graph executes; no claim of full mission completion from this short smoke.

## Logs
runtime.log is written by tee inside the container for live visibility; console.log mirrors Spur output on completion. Permit up to roughly30min startup if JIT/graph logs show progress.

## Result
PASS exit0. All8 rank results identical.4 measured iterations,256 useful tokens,context1024->1040. Actual execute counters target=4,draft=4,draft_extend=4; all state/layout/finite assertions pass. Complete=false as expected. Warmup3 excluded; measured0.094634s at short context is a smoke sample, not target-workload performance.

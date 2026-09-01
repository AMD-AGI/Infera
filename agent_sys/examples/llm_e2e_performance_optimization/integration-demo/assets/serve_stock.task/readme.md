# serve_stock

Bring GLM-5.3-Flash up in MIX mode with nothing mounted over it. This is the
baseline arm.

## Why the baseline is measured here rather than looked up

Three candidates, and two do not survive contact with the evidence.

`profiling-demo`'s recorded `aiperf_baseline` is not comparable: that package
replayed the same trace against the same configuration twice and measured 631
output tok/s with 25.9 s mean TTFT cold, against 1004 tok/s and 484 ms with the
deployment reused. A Mooncake trace carries `hash_ids`, AIPerf expands each into
a real token block, and prefix hit rate then decides how much prefill there is to
do. A number from another session is not a baseline.

A published reference score does not work either, and the 1P1D kit's README says
why: an absolute eval score has no external baseline, and the comparison worth
making is against a prior run or between `gsm8k` and its shared-prefix variant.

So: the same node, the same session, the same trace, unpatched.

## It reads the mount plan and applies none of it

Deliberate. The deployment record then says which mounts were withheld, and a
reader can tell this arm apart from a deployment that never had a plan at all.

## CUDA graphs are on

`profiling-demo` turned them off in one round because a profiler sees a graph
launch instead of the kernels inside it. This package captures no profile, so
there is no reason to measure a configuration nobody would deploy.

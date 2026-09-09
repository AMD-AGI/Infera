# run_baseline

Replay the first two minutes of a Mooncake production trace against the
graphs-on deployment, with no profiler attached, and hand back the report.

This is the round whose throughput is worth quoting. The profiled round runs the
same load against an engine with decode CUDA graphs off, several times slower by
construction.

## Why a trace replay and not a synthetic sweep

A random-prompt sweep builds every prompt independently, so it has no shared
prefix by construction and cannot measure prefix reuse at all. A Mooncake trace
carries `hash_ids`, AIPerf expands each one into a real token block, and the
radix cache and kv-aware routing are then genuinely exercised. Requests go out at
the timestamps the trace recorded rather than as fast as a client can send.

## What it costs, and what the numbers mean

Measured: 346 requests in the two-minute window, 1.71 requests/second, mean input
sequence length 14,241 tokens.

Effective concurrency reaches the 256 ceiling and stays there, so **this trace
saturates a single-node MIX deployment** and the latency percentiles describe a
queue rather than the model. That is fine for a profiling load — the engine is
never idle — and it is stated in the handoff's `watchout` so nobody quotes a
p99 from it as a latency figure.

## AIPerf runs in its own container

The engine image ships Python 3.10 and AIPerf needs 3.11 or newer. Four of its
container settings fail quietly if dropped, and each is documented at its site in
`assets/load/aiperf_replay.sh`: the uid mapping, the two dataset mmap paths, a
`PYTHONPATH` that keeps the image's own entries, and the offline flags with the
`sitecustomize` shim that lets a local tokenizer directory be loaded as one.

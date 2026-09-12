# Disaggregated serving

```{admonition} One-pager
:class: tip
**What:** give prefill and decode their own pools, parallel shapes and replica
counts, and charge the KV handoff between them. **Why:** to size the two sides
independently before building them. **Cost:** this is an **analytical pool
model** — it returns means and capacity, not disaggregated tail latency.
```

For what disaggregation is and when it pays off in a real deployment, see
[PD disaggregation](../features/pd_disaggregation.md). This page is about
projecting it.

## Project a PD topology

```bash
inferasim inference ... --disaggregate \
  --prefill-tp 4 --prefill-ep 4 --prefill-replicas 1 \
  --decode-tp 8  --decode-ep 8  --decode-replicas 2 \
  --kv-transfer-bw-gbps 400
```

Each pool gets its own parallel shape, which is the point: prefill is
compute-bound and decode is bandwidth-bound, so the shape that suits one rarely
suits the other. `--kv-transfer-bw-gbps` prices the handoff, so a topology that
moves a large cache over a thin fabric is charged for it rather than getting the
split for free.

## What the model does

The projector divides the resolved concurrency across each pool, prices the
**limiting replica** in each, and caps the steady-state request rate at the
slower pool. Requests are split without dropping a remainder, so the load you
asked for is the load that is priced.

`--decode-admission-steps` matters most on this path. With prefill off the
critical path, what remains visible in TTFT is how long a finished prefill waits
to join a decode batch — and that wait is the thing disaggregation trades the
interference tax for.

Speaking of which: the number worth reading before you disaggregate anything is
**TPOT pollution** on the colocated
[projection report](projection_runs.md#reading-the-report). That is the
continuous-batching interference you would be buying your way out of. If it is
small, the KV transfer is unlikely to pay for itself.

## What the model does not do

```{admonition} This is not an event-driven PDD simulation
:class: warning
The discrete-event simulator models colocated unified-batch engines and fleets
of those engines. It does **not** run independent prefill/decode event queues,
transfer-contention events, or role-specific schedulers. Consequently the
disaggregated path returns analytical means, capacity and throughput — not PDD
tail distributions.
```

So these questions are out of scope here and need real hardware, or an
architecture-level simulator built for role-specific scheduling:

- What is the p99 TTFT when the KV transfer link is congested?
- How does a prefill-side queue back up into decode admission?
- What happens at the moment a decode replica is added or drained?

Use the [benchmarking path](../reference/benchmarking.md) for those. Use this
path to decide which two or three topologies are worth benchmarking.

## Next steps

- Compare against the colocated projection: [Projection runs](projection_runs.md).
- Search over pool shapes: [Sweeps and tuning](sweeps.md).
- The full list of exclusions: [Boundaries and verification](boundaries.md).

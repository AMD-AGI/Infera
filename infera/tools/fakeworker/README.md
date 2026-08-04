# Fake worker

A worker that joins the fleet and serves tokens, with no GPU and no weights.

Everything above the engine — discovery, routing, failover, the circuit breaker,
drain, autoscaling — is testable without a model. What blocked that was simply
that there was no way to *be* a worker without loading one: `tests/e2e/harness`
starts real engines in containers, so the cheapest fleet anyone could build cost
a GPU and a multi-minute weight load per member.

```bash
infera-fake-worker --model-name my-model --port 9101 \
  --discovery-backend etcd --etcd-endpoint http://127.0.0.1:2379
```

## Why it can be trusted

It registers through the **real** registration clients with a real
`EngineConfig`, so the payload is built by `build_worker_payload` — the same
function every engine uses — and parsed back by `worker_info_from_json`, the
same function every discovery backend uses. If that contract changes, this
changes with it or the test suite fails. Only what happens *after* a request
arrives is faked.

## The knobs that matter

Most flags are obvious. These three exist because they make otherwise expensive
problems reproducible on a laptop:

| Flag | Why |
|---|---|
| `--startup-delay-s` | Simulates the 5–15 minute weight load. `/health` stays 503 until it elapses. This is the single biggest reason naive autoscaling overshoots — an unready replica still counts in the fleet but consumes 0% of the metric, so the loop keeps asking for more. Reproducing it here costs nothing; reproducing it on real hardware costs a GPU-hour per iteration. |
| `--max-concurrency` | Gives requests somewhere to queue. `num_requests_waiting` is the metric the entire industry autoscales on, and without a queue it is identically zero — so a scaling test against a fake fleet would pass without testing anything. |
| `--fail-first N` | Refuse the first N requests with 503, then recover. Drives the router's circuit breaker through open → half-open → closed without killing a process. |

`/metrics` exposes engine-native names (`vllm:num_requests_waiting`,
`sglang:num_queue_reqs`, …) chosen by `--engine`, so a scaling rule written
against fakes transfers to a real fleet unchanged.

> The vLLM names are from its published metrics documentation. **The SGLang
> names are second-hand from a research pass and have not been checked against a
> live SGLang.** Verify before depending on them.

## Limits — read these before drawing conclusions

**No KV transfer is simulated.** A `--disagg-mode prefill` / `decode` fake takes
part in pool membership and routing, and that is all. No KV moves between the
legs. Useful for testing that P and D pools exist, are discovered, and are
routed to independently; **useless for testing Mooncake, bootstrap handshakes,
or anything about the transfer itself.** Do not conclude that PD "works" because
a fake fleet answered.

**`--kv` synthesizes a tokenizer canary from the model name.** All fakes for one
model agree, which is what they need to do — `CanaryVerifier` silently drops a
worker whose canary differs from the first-registered one, so disagreeing fakes
would produce a fleet that is half the expected size for no visible reason. But
that synthetic canary will never match a **real** worker's, so do not mix fakes
and real workers under one model name. Without `--kv` there is no canary at all
and the fakes join anything.

**It answers instantly by construction.** `--ttft-ms` and `--itl-ms` are a
latency model, not a performance model: TTFT scales linearly with prompt length
and nothing contends for memory. Do not use it to predict real throughput.

## Gotchas found while building this

- **`--advertise-host` must be routable from the router.** It defaults to
  `$POD_IP`. `0.0.0.0` registers a URL no peer can reach.
- **The server requires `--router-tokenizer-path` even for `round-robin`**, and
  resolves it eagerly. Any existing directory satisfies it, which is enough to
  bring a router up against fakes.
- **Registration alone is not enough** — `heartbeat_loop()` has to be running or
  the etcd lease (30 s) expires and the worker vanishes from the pool about half
  a minute after it appears. This looks exactly like a discovery bug and is not
  one. The real worker entrypoint starts it too; so does this.

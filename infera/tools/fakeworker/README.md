# Fake worker

A worker that joins the fleet and serves tokens, with no GPU and no weights.

Everything above the engine — discovery, routing, failover, the circuit breaker,
drain, autoscaling — is testable without a model. What blocked that was simply
that there was no way to *be* a worker without loading one: `tests/e2e/harness`
starts real engines in containers, so the cheapest fleet anyone could build cost
a GPU and a multi-minute weight load per member.

```bash
export INFERA_ALLOW_FAKE_WORKER=1
infera-fake-worker --model-name my-model --port 9101 \
  --discovery-backend etcd --etcd-endpoint http://127.0.0.1:2379
```

`INFERA_ALLOW_FAKE_WORKER` is required, and the tool refuses to start without
it. It ships in the same package as the server and registers through the real
registration clients, so it can join a fleet and advertise any address it likes
— which the router will then dial, sending it real prompts. That needs no
privilege a worker does not already have, but it should be a deliberate act
rather than something a stray command does by default.

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

## PD and DP attention

Both work, and both are verified end to end against a real router with no GPU.

**PD.** `--disagg-mode prefill|decode` advertises the same `disagg_meta` a real
worker does — `{"protocol": ..., "params": {"bootstrap_addr": ...}}`, prefill
only — with `--pd-protocol` constrained to the router's own protocol registry so
a typo dies at argparse rather than at the first request.

**DP attention.** Two deployment shapes exist and they behave differently, which
is worth knowing before concluding anything is broken:

| Shape | Registration | What the router does |
|---|---|---|
| per-rank endpoints | `--dp-rank R --dp-size N` | Nothing. The address already selects the rank, so no header is pinned and no room alignment happens. `dp_rank` on the `RouteTarget` stays `None`. |
| rank-multiplexed | `--dp-size N`, **no** `--dp-rank` | `expand_targets` fans one worker into N targets; the router pins `X-Data-Parallel-Rank`, aligns `bootstrap_room % dp_size == dp_rank`, and injects `disagg_prefill_dp_rank` for the decode leg. |

`is_rank_multiplexed()` is `dp_size > 1 and dp_rank is None` — so registering a
rank makes the worker an endpoint and opts *out* of router-side DP routing. That
is correct, not a bug, but the first time you see it the DP path looks dead.

### `GET /debug/routing`

The reason PD and DP are testable at all. It reports what the *router* decided —
per-rank request counts keyed on the header it sent, and the handoff fields it
injected on the last request:

```json
{"dp_rank": null, "dp_size": 4,
 "requests_by_dp_rank": {"0": 2, "1": 2, "2": 2, "3": 2},
 "last_handoff": {"bootstrap_host": "127.0.0.1", "bootstrap_port": 18430,
                  "bootstrap_room": 7442254485466660987,
                  "disagg_prefill_dp_rank": 3}}
```

None of that is observable with a real engine: a malformed handoff does not
raise, it hangs on KVPoll until a ~300 s timeout, and the failure surfaces
nowhere near the router that caused it. Here you can assert on it directly —
e.g. that `bootstrap_room % dp_size == disagg_prefill_dp_rank` holds on every
request, which is the invariant SGLang's `follow_bootstrap_room` balancer
enforces with a `KVTransferError`.

## NATS transport

`--request-transport nats` routes requests through a broker instead of having
the router dial this worker. It uses the **real** `NatsRequestServer`, which
proxies to this process's own HTTP surface exactly as it proxies to a real
engine's — so the transport under test is the production one, not a stand-in.

```bash
INFERA_ALLOW_FAKE_WORKER=1 infera-fake-worker --model-name m --port 9101 \
  --request-transport nats --nats-server nats://127.0.0.1:4222 \
  --discovery-backend etcd --etcd-endpoint http://127.0.0.1:2379
```

Shutdown then goes through the real NATS drain — unsubscribe first, then wait on
the in-flight set infera actually holds:

```
deregistered worker 127.0.0.1:19951 (lease 7587883597149818 revoked)
draining 1 in-flight NATS request(s), up to 60s
```

```{note}
**The fake's HTTP drain is not representative of a real worker's.** This process
serves the requests itself, so it knows its own in-flight count exactly. A real
worker on HTTP transport does not: the router talks straight to the engine, so
infera has to poll the engine's `/metrics` and wait out a settle window because
those gauges lag. Comparing the fake's HTTP drain against its NATS drain
therefore measures nothing — both are exact. The difference only shows up with a
real engine.
```

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
- **A bind failure used to still register.** uvicorn logs `address already in
  use` and gives up, but the process keeps running — so a port collision
  produced a worker that was in the pool and served nothing. Now the socket must
  be bound before registration, and a collision exits 3. Worth remembering
  because the symptom was a router error about the *worker* returning garbage,
  which pointed nowhere near the real cause.
- **Registration alone is not enough** — `heartbeat_loop()` has to be running or
  the etcd lease (30 s) expires and the worker vanishes from the pool about half
  a minute after it appears. This looks exactly like a discovery bug and is not
  one. The real worker entrypoint starts it too; so does this.

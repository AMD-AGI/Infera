# Scaling a fleet

Adding and removing workers while traffic is flowing. Every number on this page
was measured on the hardware described in [Measurements](#measurements) — none
of it is projected.

## How it works

There is no scaling controller. Workers **self-register** into discovery when
they are ready and **deregister** when they shut down, and the router routes to
whatever is registered at that instant. Scaling is therefore just starting and
stopping worker processes; nothing has to be told about it.

```
worker ready ──► register (etcd lease / Pod annotation) ──► router's watch fires
                                                            ──► receives traffic
leaving ──► stop being routed to ──► drain in-flight ──► deregister ──► exit
```

What triggers "stop being routed to" depends on the backend: under Kubernetes
the orchestrator marks the Pod before the process is even signalled, elsewhere
removing the record is what does it. See [Who says the worker is
leaving](#who-says-the-worker-is-leaving), and
[Graceful shutdown](graceful_shutdown.md) for the feature as a whole.

That shape is why scale-up and scale-down have very different costs. Scale-up is
bounded by **model load**, which is minutes. Scale-down is bounded by the
**longest in-flight generation**, which is seconds — and the router stops
choosing the worker in milliseconds, long before it stops serving.

## Scaling up

Start another worker with the same `--model-name` and the same discovery
settings. It joins when it is ready, and not before: registration happens after
the engine has loaded weights, so a worker in the pool is always a worker that
can serve.

```bash
infera-worker ... --port 20002 --etcd-endpoint http://etcd:2379
```

On Kubernetes, raise `replicas` on the worker service in the `InferaDeployment`.

**Budget minutes, not seconds.** Measured cold start for an 8B model on one
MI355X was **140 s** from `docker run` to appearing in `/v1/workers`, almost all
of it weight loading. Anything that reacts to load by starting a worker has to
tolerate that delay — a rule that scales up when a queue is deep will still be
scaling up long after the queue drained.

The corollary matters more than it looks: for a burst shorter than the cold
start, **adding workers cannot help**. Either keep headroom, or shift traffic
between roles that are already running (see
[PD disaggregation](pd_disaggregation.md)).

## Scaling down

Send `SIGTERM`. Do not `SIGKILL`, and do not simply delete the Pod without a
grace period.

The worker then, in this order:

1. **Stops being routed to.** The worker removes its registration, which is
   what stops new work arriving. Under Kubernetes a Pod being *deleted* has
   already left routing well before this — the registry acts on its
   deletionTimestamp, before the process is even signalled — so this is only
   cleanup there (see [Who says the worker is
   leaving](#who-says-the-worker-is-leaving)).
2. **Drains.** On the NATS transport infera tracks in-flight requests directly.
   On HTTP the router talks straight to the engine, so infera asks the engine
   instead, polling its `/metrics` until running, queued, and PD-handoff queues
   all reach zero. Bounded by `--drain-timeout` (default 30 s).
3. **Stops the engine.**

Requests already in flight run to completion. Requests that arrive during the
drain go to other workers.

**Two different timings, easily conflated.** A worker stops *receiving* new
requests within a second of the shutdown starting — the router's watch picking
up either the `deletionTimestamp` or the record's removal — and it is the number
that decides whether traffic is still being sent somewhere that is about to die. How long the *process* then lives is
a separate and much larger number, set by the longest generation it was already
serving. Measured: under a second to stop receiving, while a 40-second
generation ran to completion afterwards.

Watching `/v1/workers` measures neither. The record now goes when the drain
*starts*, not when it ends, so its disappearance marks the beginning of the
in-flight work rather than the end of it — a worker finishing a long generation
is absent from that list for all of it. A Pod being deleted shows as `draining`
only for the window between its deletion being requested and the process being
signalled, which is the preStop hook's 15 s.

```{note}
`--drain-timeout` is a **ceiling, not a delay** — a worker with nothing in flight
exits in about six seconds regardless. Set it above your p99 generation time.
Anything still running when it expires is cut, with a warning naming the count.
```

### The transport decides how well this works

Draining is only as good as the router's view of what is in flight, and that
differs by transport — not by implementation quality, but by where the
information lives.

| | who knows what is in flight | drain |
|---|---|---|
| **NATS** (`--request-transport nats`) | infera — it owns the request path and holds the in-flight set | exact, no polling |
| **HTTP** (default in the recipes) | only the engine — the router dials it directly and never sees the request | poll the engine's `/metrics`, behind a settle window |

Measured with a GPU-free stand-in worker, same generation:

- **NATS, one in-flight generation**: the log reads `draining 1 in-flight NATS
  request(s)` — it knows the count — the 300-chunk generation completed in full,
  and the worker deregistered **21.3 s** later, which is just the remaining
  generation time with no overhead.
- **NATS, nothing in flight**: leaving rotation to exit in **3 ms**.
- **HTTP with a real engine, nothing in flight**: at least the **6 s** settle
  window, because a single zero reading cannot be told apart from a gauge that
  has not refreshed yet.

So NATS costs a broker and buys a drain that is exact rather than inferred. It
also buys request cancellation the HTTP path does not have — a timeout or client
disconnect publishes to `infera.cancel.<worker>` and the worker tears down the
engine connection, instead of leaving it generating.

### Admission control

Setting `INFERA_NATS_REQ_MAX_PENDING` (or `--nats-req-max-pending`) above zero
on **both** the server and the workers makes the request path JetStream-backed:
a WorkQueue stream with one durable consumer per worker. The router reads that
consumer's backlog before dispatching and refuses a worker over the limit.

This matters for scaling because it covers the window scaling cannot: a burst
shorter than a 140 s cold start cannot be answered by adding workers, so the
choice is between queueing behind a saturated worker and steering away from it.

**Look at the distribution, not the status codes.** A refusal raises the same
retryable failure as any other pre-first-byte error, so the request fails over
to a freer worker and the client sees `200`. Only when every worker is over the
limit and retries are exhausted does a `429` reach the client. Measured with one
deliberately saturated worker (concurrency 1) and one fast one, limit 3:

| | saturated worker | fast worker |
|---|---|---|
| 20 requests under backlog | **+0** | **+20** |
| round-robin without the throttle | +10 | +10 |

The worker's consumer showed `num_ack_pending = 10` against a limit of 3 at the
time — the ack happens after the request is fully proxied, precisely so the
backlog gauge reflects genuinely in-flight work rather than mere delivery.

```{note}
The check is per dispatch, so it steers *new* requests. Requests already
dispatched are unaffected, and a simultaneous burst is all admitted — every
admission check runs before any of them has built backlog.
```

```{note}
The Rust router does not implement the NATS transport (`lib.rs`: "Configs
outside this set (NATS transport, ...) are served by the Python backend"), so
the Rust data plane and the NATS drain are currently an either/or.
```

### Why in-flight work is visible at all

The engine's own gauges are the only source of truth on the HTTP path, and they
have three properties worth knowing:

- **SGLang serves `/metrics` only with `--enable-metrics`.** Without it the
  endpoint 404s and the drain has nothing to read. The worker entrypoint injects
  the flag, so this is handled — but a hand-rolled deployment that bypasses it
  will silently lose the drain.
- **The gauges lag.** Measured on SGLang: `num_running_reqs` stayed at 12 for
  5–15 s after the last response completed. The drain therefore requires the
  count to read zero continuously for a settle window before believing it,
  which also protects against a request accepted moments before `SIGTERM` that
  has not been counted yet.
- **PD handoff queues count as in-flight.** A prefill worker can show no running
  and no queued requests while KV transfers are still outstanding. Stopping it
  there strands the decode workers waiting on that KV, so
  `num_prefill_bootstrap_queue_reqs`, `num_prefill_inflight_queue_reqs`,
  `num_decode_prealloc_queue_reqs` and `num_decode_transfer_queue_reqs` are
  included in the count.

If the in-flight count cannot be read at all — an unknown engine, a renamed
series, a dead HTTP server — the worker logs a warning naming the metric it
looked for and shuts down **without** draining rather than blocking. A rolling
update that stalls on a parse failure is worse than one that cuts a request, and
a silent full-timeout wait would be indistinguishable from a genuinely busy
worker.

### On Kubernetes

The recipes deploy with `discoveryBackend: kubernetes` and
`--request-transport http`, so the shutdown path differs from a bare
etcd deployment in two ways — and gains one stage.

**Discovery is a Pod annotation, not an etcd lease.** Registering writes
`infera.amd.com/worker-info` on the worker's own Pod; deregistering clears it.
The registry additionally marks a Pod `DRAINING` the moment it carries a
`deletionTimestamp`, without waiting for the container to exit. That matters
because a terminating Pod keeps `phase: Running` — without the check it would
stay a routing candidate for the whole `preStop` delay, turning a hook meant to
make shutdown graceful into extra seconds of accepting work about to be killed.

The mark, rather than an outright removal, is what keeps the two timings above
distinguishable on this backend too: the worker leaves routing immediately and
its record stays until it clears its own annotation at the end of the drain, so
`/v1/workers` shows a rollout in progress instead of a worker that vanished.

**There is a `preStop` delay before `SIGTERM`.** The operator injects
`sleep 15`, so the full sequence is:

```
deletion requested ──► deletionTimestamp set ──► registry drops the worker
                   ──► preStop sleep 15 (still serving what it has)
                   ──► SIGTERM ──► drain ──► deregister ──► engine.stop()
                   ──► [kubelet SIGKILL at terminationGracePeriodSeconds]
```

### Who says the worker is leaving

The two discovery backends learn this in different ways, and only one of them
needs the worker to say anything. The difference is not an inconsistency to be
smoothed over — it is what each backend can actually observe.

**Kubernetes: the orchestrator says so.** A condemned Pod carries
`deletionTimestamp` from the moment deletion is requested, which is before the
`preStop` hook runs and therefore before the process is signalled at all. The
registry reads it and drops the worker from routing immediately — measured at
under 100 ms against the 15 s `preStop` delay. The worker announcing the same
thing later would add nothing: routing has already stopped, and the terminating
check returns before the annotation is even parsed.

So on this backend the annotation carries **identity only** — worker id, URL,
model, engine, role, KV endpoints — all of it fixed for the life of the
process. That is deliberate. The heartbeat re-asserts the annotation to
self-heal, rebuilding it from config; if state lived there too, a refresh
landing mid-drain would overwrite it with a payload that omits the status,
which parses as `ACTIVE`, and the worker would be handed new work it is about
to refuse.

**Everywhere else: removing the record says so.** On etcd there is no
orchestrator at all — a record is either present with an unexpired lease or it
is gone, with no third state to put it in. The same is true on Kubernetes
whenever the Pod is *not* being deleted: a liveness probe restarting the
container, a node shutting down gracefully, someone killing the process. No
deletionTimestamp is set, so the registry reads the annotation as usual and the
worker stays routable until it clears it.

So every shutdown deregisters *first* and drains after. In-flight generations
are finished either way; the cost is that the worker is absent from
`/v1/workers` while it drains rather than shown as draining. The head start —
leaving routing before the process is signalled at all — is what deleting a Pod
buys, and only that.

```{warning}
`discoveryBackend: etcd` **is not supported for in-cluster deployments** and the
operator refuses it. The combination keeps the `preStop` delay while losing the
early notice it exists to provide: the server no longer watches Pods, so nothing
reads the `deletionTimestamp`, and the only signal left arrives after `SIGTERM`
— that is, after the delay has already elapsed. For its whole duration the
router keeps handing new work to a Pod that is already condemned. Use the
default `kubernetes` backend in Kubernetes; external etcd is for deployments
outside it.
```

### Worst case, and the budget

Every stage is individually bounded:

| Stage | Bound | Set by |
|---|---|---|
| `preStop` | 15 s | operator |
| drain | `--drain-timeout` (default 30 s) | flag |
| deregister | 10 s | registration HTTP client timeout |
| `engine.stop()` | 30 s | `SIGTERM` to the engine's process group, then `SIGKILL` |
| **total** | **≈95 s at defaults** | |

`terminationGracePeriodSeconds` has to cover that whole sum, because the kubelet
`SIGKILL`s the moment it expires — mid-drain if that is where things are. The
operator now **derives** it as `preStop + --drain-timeout + 50 s` of teardown
headroom, with a 120 s floor, reading the flag from `ServiceSpec.Args` or from
the container directly when an `extraPodSpec` template supplies it.

```{warning}
This used to be a fixed 120 s with a comment saying it "must exceed preStop +
the worker `--drain-timeout`" — and nothing parsed that flag, so the invariant
was documented and unenforced. Raising `--drain-timeout` for long generations
(the only reason anyone raises it) pushed shutdown past the grace and turned the
drain back into a kill. Measured on a live cluster before the change: a worker
declaring `--drain-timeout 300` still received `terminationGracePeriodSeconds:
120`, i.e. 365 s of budget granted 120.
```

Measured on a live k3s cluster: `kubectl delete pod` took the worker out of
routing in **87–93 ms**, against the 15 000 ms `preStop` delay. That gap is the
whole point of reading `deletionTimestamp` — the alternatives (the `DELETE`
event, or `phase` leaving `Running`) only fire once the container has already
exited, so without it the router would keep assigning work for the entire
`preStop` window and then have it killed.

Two runs, both with a Pod deleted while holding in-flight work:

- **Real SGLang Qwen3-8B** deployed by the operator (`InferaDeployment`, two
  workers, one MI355X each, Kubernetes discovery, HTTP transport): four
  concurrent 2500-token generations in flight, **4/4 completed with HTTP 200**
  and full-length output (6.5–13.3 kB), replacement Pod registered before the
  drain finished.
- **GPU-free stand-in workers**, same path: a 300-chunk generation completed
  in full across the drain.

```{note}
`spec.services.<name>.resources` is **ignored when `extraPodSpec` is set** — the
template is passed through verbatim, so the GPU request has to live on your own
container. A worker that omits it schedules, starts, and then fails with "No
accelerator available".
```

If you set the grace period yourself it is respected as long as it is **larger**
than the derived value; it is only ever raised, never lowered.

## PD and DP

Prefill and decode register into separate pools and are selected per request, so
they scale **independently** — add prefill for longer inputs, decode for more
concurrent users. Two constraints:

- **Neither pool can go to zero.** PD dispatch fails closed when either side is
  empty — `minReplicas: 0` on either is an outage, not an idle saving. The 503
  names the empty pool (`has 1 decode worker(s) but no prefill worker`), so the
  cause is visible without reading the fleet.
- **A DP worker's shape decides who picks the rank.** A worker registering
  `dp_size > 1` with **no** `dp_rank` is rank-multiplexed: the router fans it
  out into one target per rank and pins `X-Data-Parallel-Rank`. A worker that
  registers its own `dp_rank` is a plain endpoint and opts out — its address
  already selects the rank. Both are valid; only the first involves the router.

## Across machines

Nothing about scaling changes when workers live on different hosts — discovery
is already the coordination point, so a worker on another machine joins the same
way. Two things do change, and both are configuration rather than mechanism:

**`--advertise-host` must be the node's routable address.** It is the URL peers
dial, and the single-node habit of leaving it at `127.0.0.1` registers an
address that resolves to the wrong machine everywhere else. The failure is
quiet in the worst way: the router *lists* the worker and cannot reach it, so it
looks like a broken worker rather than a misconfiguration. On Kubernetes, take
it from the downward API (`POD_IP`).

**Discovery must be reachable from every node.** An etcd bound only to loopback,
or advertising a loopback client URL, works perfectly on the node running it and
is invisible from the others.

Checking both before deploying costs nothing:

```bash
# from each worker node
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://<etcd-host>:2379/v3/kv/range -d '{"key":"Lw=="}'
# from the router node, once a worker has registered
curl -s http://<router>:8000/v1/workers | jq -r '.workers[].url'   # must be dialable
```

Measured on two nodes (chi2800 / chi2866, one MI355X each, workers advertising
their own IPs, etcd and router on the first node): both workers registered with
distinct addresses, 12 requests distributed 7/7 across the machines, and a
`SIGTERM` to the **remote** worker drained cleanly — its three in-flight
3000-token generations all completed (13.7–14.4 k characters), its record
disappeared after 30 s, and 100 requests flowing through the router during the
whole transition saw **0 failures**. (As above, the record surviving 30 s is the
generations finishing, not 30 s of continuing to receive work.)

```{warning}
This covers workers on separate machines. It does **not** cover a single worker
*spanning* machines (`numberOfNodes > 1`, LeaderWorkerSet) or PD over RDMA
between nodes — neither has been exercised here. Note also that on this cluster
`rdma/hca` is not advertised as an allocatable resource, so a PD deployment
would need host networking and direct device access rather than a device plugin.
```

## Measurements

SGLang 0.5.15 and vLLM 0.1.dev19253, Qwen3-8B, one MI355X per instance, HTTP
transport, etcd discovery, real router.

| | |
|---|---|
| Cold start (`docker run` → in `/v1/workers`) | **140 s** |
| Scale-down: `SIGTERM` → stops receiving new requests | **< 1 s** |
| Scale-down: `SIGTERM` → record gone from `/v1/workers` | **30–38 s** |
| Router reaction to a worker's record being deleted | **15 ms** |
| Drain settle window | 6 s |

Two runs, both with traffic flowing throughout:

**Drain under load.** Six concurrent 4000-token generations in flight at
`SIGTERM`. Both engines: **6/6 completed with HTTP 200** and full-length output
(15–19 k characters). SGLang 22 s, vLLM 19 s from signal to last response.

**Scale up then down.** Two instances, continuous traffic, a third added and
then one removed. **260 requests, 0 failures**, including in the 5-second
windows around each transition. The removed instance left rotation, drained the
one generation it was holding (`engine idle for 6s, 1 request(s) completed`),
and only then exited.

```{note}
This run predates the change that made the shutdown order backend-specific, so
its logs show the worker announcing `DRAINING` before draining. On etcd the two
steps are now the other way round — deregister, then drain — which is what stops
new work arriving on a backend where nothing else can. The request counts are
unaffected: both orderings stop new work before waiting on in-flight work.
```

**PD scaling, measured.** A 1P1D fake fleet grown to 2P2D and shrunk back under
continuous traffic: **200 requests, 0 failures**, both pools scaling
independently and the drained workers finishing their in-flight work. Taking the
last prefill away then returns 503 naming the empty pool.

**Scaling over the API, measured.** A PD deployment at 1 prefill / 2 decode
resized to 3 / 4 by one `POST /v1/admin/scale`. The CR carried both new counts
immediately and the Pods reached them **within 35 s**, an operator resync apart.
Reading back during that window distinguished the two: `replicas` already 3 and
4 while `current_replicas` still read 1 and 2, which is what tells "not scaled"
from "still scaling".

**And with a real engine.** One vLLM worker (Qwen3-8B, one MI300X) scaled to two
over the API, then back. The added worker loaded weights, registered, and
**served 4 of the 8 requests** sent after it joined — the count that matters,
since a Pod that exists is not the same as a worker that answers. Scaling back
down left the survivor serving normally.

Refusals were checked against the cluster rather than only in unit tests, since
what matters is that a rejected request leaves nothing behind: scaling a pool to
zero, naming a service that does not exist, and a negative count were each
refused with the CR unchanged. A server started without `--enable-scaling-api`
answered 403.

The RBAC the operator generates was read back from the cluster: `get` and
`patch` on `inferadeployments`, restricted by `resourceNames` to the one
deployment, alongside the unchanged Pod grant.

```{warning}
**Not measured:** multi-node workers, TP > 1, PD scaling with a *real* engine
(the runs above used GPU-free stand-ins, so no KV moved), and scale-down during
an active KV transfer. The PD handoff queues are counted in the drain, but that
path has not been exercised on hardware.

The API was measured both ways: with stand-ins for the PD run, since the path
from request to Pod count does not involve inference, and with a real engine for
the single-pool run, which is where "the added worker serves" was checked. What
neither shows is a *scaled-down* worker draining real work — that is the drain
measurement above.
```

## Scaling a deployment

Edit the service's `replicas` in the `InferaDeployment`. That is the write that
counts — every supported path ends there:

```bash
kubectl patch inferadeployment qwen --type=merge \
  -p '{"spec":{"services":{"decode":{"replicas":5}}}}'
```

Or over the server's admin API, for an orchestration layer that would otherwise
need cluster credentials and knowledge of the CR's shape:

```bash
infera-server --enable-scaling-api   # off by default

curl -X POST http://router:8000/v1/admin/scale \
  -H 'Content-Type: application/json' \
  -d '{"services": {"prefill": 4, "decode": 8}}'
```

Both pools move in one write, so a rebalance cannot half-apply. `GET` on the
same path reports each pool's requested size alongside what the cluster has
actually got, which is how a caller tells "not scaled" from "still scaling".

The API refuses to take a pool to zero: an empty pool stops serving, and in a PD
deployment an empty prefill or decode pool fails *every* request rather than
only the ones that would have landed there. Retiring a pool is a `kubectl` edit,
where the intent is unambiguous.

```{note}
The server finds the deployment to scale from its own Pod labels, so this needs
a deployment the operator created, and the RBAC that ships with it. A cluster
running an operator from before this feature has a server that cannot write:
re-apply the operator manifests to get the grant.
```

For a multi-node service the count is **groups**, not pods: `replicas: 5` with
`numberOfNodes: 3` is fifteen pods and five servable instances, since only
node-rank 0 of each group registers.

Pods removed by a scale-down drain first — the operator injects the `preStop`
delay and a grace period sized from `--drain-timeout`, so the sequence is the
same one `kubectl delete pod` follows.

```{warning}
Do **not** scale the generated `Deployment` or `LeaderWorkerSet` directly. Both
carry a real `/scale` subresource, so the write succeeds and nothing reports an
error — and then the next reconcile reverts it, because this reconciler assigns
the whole child `.Spec` on every pass. Measured: a `kubectl scale` to 3 went
back to 1 in under 3 seconds. The only symptom is a replica count that keeps
snapping back.
```

## Autoscaling

Infera ships no autoscaler. `/v1/admin/scale` gives an external one somewhere to
push a decision to, but it is an entry point rather than a control loop: nothing
in Infera watches load and decides.

An `InferaDeployment` cannot carry a Kubernetes `/scale` subresource, and that
is a property of its shape rather than an omission: `spec.services` is a map
with user-chosen keys, while the scale subresource requires `specReplicasPath`
to be a *static* dot-notation JSONPath, and a CRD may declare only one. A single
path could name one service — hardcoding `decode`, say — which leaves every
other pool, and in a PD deployment specifically the prefill pool, with no handle
at all. The admin API sidesteps that by naming the service in the request
instead of the path.

Pointing an autoscaler at the generated workload does not work either, for the
reason in the warning above: those objects are derived state and are rewritten
every pass.

Two harder problems sit behind the plumbing anyway:

- **A 140-second cold start sits inside a control loop that ticks every 15
  seconds.** A burst shorter than the cold start cannot be answered by adding
  workers at all.
- **Nothing in Kubernetes lets a scaler choose *which* replica to remove**, so
  the one holding the warmest KV cache is as likely to go as any other. Upstream
  has declined to fix this (k8s#123541, closed as not planned).

The signals worth scaling on (`vllm:num_requests_waiting`,
`sglang:num_queue_reqs`, KV utilisation) are exposed by the engines and already
read by the drain path, but nothing polls them continuously yet.

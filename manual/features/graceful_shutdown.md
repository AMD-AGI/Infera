# Graceful shutdown

```{admonition} One-pager
:class: tip
**What:** a worker being removed stops receiving new requests immediately, then
finishes the generations it already accepted before the process exits.
**Why:** without it, every request in flight on that worker is severed —
a rolling upgrade or a scale-down turns into a burst of client errors.
**Requires:** a Kubernetes deployment using the default `kubernetes` discovery
backend. See [Outside Kubernetes](#outside-kubernetes) for what other
deployments get instead.
```

```{important}
The full behaviour described here — a worker leaving rotation *before* it is
signalled, and staying visible while it finishes — is supported **only on
Kubernetes with the default `kubernetes` discovery backend**. That is not an
implementation gap: it depends on the orchestrator knowing a Pod is condemned,
which nothing outside Kubernetes can tell the router.
```

## What it prevents

A generation can run for tens of seconds. If a worker is stopped while holding
one, the client gets a truncated stream or a connection reset — and there is no
retry that helps, because the tokens already sent cannot be un-sent.

That makes ordinary operations expensive. Rolling out a new image, scaling down
after a burst, draining a node for maintenance: each replaces workers that are
very likely mid-generation.

Graceful shutdown separates two things that would otherwise happen at once:

- **Stop receiving.** The worker leaves the routing candidate list. New requests
  go elsewhere from that moment.
- **Stop serving.** The worker keeps working on what it already accepted, and
  only then exits.

The gap between them is the drain.

## The sequence on Kubernetes

```
kubectl delete pod / scale down / rolling update
  │
  ├─► Kubernetes marks the Pod as condemned          ← under 100 ms
  │   the router drops it from routing here
  │
  ├─► preStop delay (15 s), still serving what it has
  │
  ├─► SIGTERM
  │   drain: wait for in-flight generations to finish
  │   (bounded by --drain-timeout, default 30 s)
  │
  ├─► deregister, stop the engine
  │
  └─► [SIGKILL if the grace period expires]
```

The important part is the first step. Kubernetes marks a Pod the instant its
deletion is requested — before the `preStop` hook runs, and therefore before the
worker process is signalled at all. The router watches for that mark, so it
stops choosing the worker in well under a second, while the worker itself does
not learn it is leaving for another 15 seconds.

Without that, the `preStop` delay would work against you: it is meant to give
the router time to react, but if the router only finds out at `SIGTERM`, the
delay is simply 15 more seconds of accepting work that is about to be drained.

**The worker stays visible while it drains.** Its record is removed at the end,
not the beginning, so `/v1/workers` reports it as draining rather than having it
disappear. A worker that vanishes looks exactly like one that crashed; this way
an operator can see a rollout progressing and how far along it is.

## What you need to configure

Nothing, if you deploy through the operator — it injects the `preStop` delay and
sizes `terminationGracePeriodSeconds` to cover the whole sequence.

For a hand-written manifest, two things matter:

- **A `preStop` delay.** Without it `SIGTERM` arrives immediately and the drain
  starts before the router has necessarily reacted.
- **A `terminationGracePeriodSeconds` that covers the whole sequence**, which is
  the `preStop` delay plus `--drain-timeout` plus teardown. Set it too low and
  the kubelet sends `SIGKILL` partway through the drain — turning a graceful
  shutdown back into an abrupt one, which is the failure this feature exists to
  avoid. Raising `--drain-timeout` for long generations without raising the
  grace period is the usual way to hit this.

```{warning}
`discoveryBackend: etcd` is **not supported for in-cluster deployments**, and
the operator refuses it. The combination keeps the `preStop` delay while losing
the early notice that delay exists to provide: the router no longer watches
Pods, so nothing sees the Pod being condemned, and the only remaining signal
arrives after `SIGTERM` — once the delay has already elapsed. For its whole
duration the router keeps handing new work to a Pod on its way out, which is
worse than either backend on its own.
```

## Outside Kubernetes

Deployments on bare metal or under a container runtime use an external etcd for
discovery, and there is no orchestrator to say a worker is leaving. A record is
either present or absent; nothing observes that a process is on its way out.

Shutdown there still drains, but in the other order: the worker removes its
registration first — which is what stops new requests arriving — and then waits
for its in-flight generations. In-flight work is still finished rather than cut.
What is lost is the two properties that depend on the orchestrator:

- **No early notice.** Routing stops when the process is signalled, not before.
- **No visible draining.** The worker disappears from `/v1/workers` for the
  duration of the drain rather than being reported as finishing.

If you are running in Kubernetes, use the default backend and you get both.

# Graceful shutdown

```{admonition} One-pager
:class: tip
**What:** a worker being removed stops receiving new requests immediately, then
finishes the generations it already accepted before the process exits.
**Why:** a severed generation cannot be retried — the tokens already streamed
cannot be un-sent — so without this, every rolling upgrade or scale-down
produces a burst of client errors. **Requires:** Kubernetes, with the default
`kubernetes` discovery backend.
```

```{important}
Supported **only on Kubernetes with the default `kubernetes` discovery
backend**. This is not an implementation gap: it relies on the orchestrator
knowing a Pod is being removed, which nothing outside Kubernetes can tell the
router. `discoveryBackend: etcd` is rejected by the operator for in-cluster
deployments.
```

## What happens

Removing a worker — a rolling update, a scale-down, draining a node — separates
two things that would otherwise happen at once:

1. **It stops receiving.** Kubernetes marks the Pod the moment its removal is
   requested, which is *before* the worker process is signalled. The router sees
   that mark and stops choosing the worker within milliseconds, so new requests
   go elsewhere while it is still running and long before it is told to stop.
2. **It keeps serving.** The worker finishes the generations it already
   accepted, bounded by `--drain-timeout`, and only then exits.

The early mark is what makes this different from simply stopping a process.
The `preStop` delay that follows is not spent waiting for the router to notice
— that already happened — but letting work in progress finish before the
process is signalled at all.

Deploying through the operator needs no configuration: it injects the `preStop`
delay and sizes the termination grace period to cover the whole sequence. For
hand-written manifests and the per-stage timings, see
[Scaling a fleet](scaling.md).

## When the Pod is not being deleted

A worker can also be stopped without its Pod going anywhere — a liveness probe
failing and restarting the container, a node being shut down gracefully, someone
killing the process. There is no deletion, so there is no early mark, and the
router has no way to know until the worker says so.

On those paths the worker removes its own registration as its first act on
`SIGTERM`, which stops new requests arriving, and drains after. In-flight
generations still finish. What is missing is the head start: from the moment the
decision is made to the moment the process is signalled, the router is still
sending work, because nothing has told it otherwise.

## Elsewhere

Deployments outside Kubernetes use an external etcd for discovery, where a
worker record is simply present or absent and nothing observes that a process is
leaving. Shutdown behaves as in the section above — deregister, then drain —
with in-flight work finished either way. The early notice is the part that needs
Kubernetes.

In both cases the worker is absent from `/v1/workers` while it drains rather
than shown as draining, since removing the record is what stops new work
arriving.

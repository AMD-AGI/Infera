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
   go elsewhere while it is still running.
2. **It keeps serving.** The worker finishes the generations it already
   accepted, bounded by `--drain-timeout`, and only then deregisters and exits.

Because the record is removed at the end rather than the beginning, the worker
stays visible in `/v1/workers` while it drains — an operator can see a rollout
progressing instead of workers appearing to crash.

Deploying through the operator needs no configuration: it injects the `preStop`
delay and sizes the termination grace period to cover the whole sequence. For
hand-written manifests and the per-stage timings, see
[Scaling a fleet](scaling.md).

## Elsewhere

Deployments outside Kubernetes use an external etcd for discovery, where a
worker record is simply present or absent and nothing observes that a process is
leaving. Shutdown there still finishes in-flight work, but in the other order:
the worker deregisters first — which is what stops new requests arriving — and
drains after. In-flight generations are not cut; what is unavailable is the
early notice and the visible draining above.

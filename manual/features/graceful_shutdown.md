# Graceful shutdown

```{admonition} One-pager
:class: tip
**What:** a worker being removed stops receiving new requests immediately, then
finishes the generations it already accepted before the process exits.
**Why:** a severed generation cannot be retried — the tokens already streamed
cannot be un-sent — so without this, every rolling upgrade or scale-down
produces a burst of client errors. **Requires:** nothing, to finish in-flight
work; Kubernetes with the default `kubernetes` discovery backend for the advance
notice below.
```

Removing a worker — a rolling update, a scale-down, draining a node — separates
two things that would otherwise happen at once:

1. **It stops receiving.** The router stops choosing the worker as soon as its
   removal is requested, which is before the worker itself is told to stop, so
   new requests go elsewhere while it is still running.
2. **It keeps serving.** The worker finishes the generations it already
   accepted, bounded by `--drain-timeout`, and only then exits.

Deploying through the operator needs no configuration: it injects the `preStop`
delay and sizes the termination grace period to cover the whole sequence, from
the drain timeout you set. For hand-written manifests and the per-stage timings,
see [Scaling a fleet](scaling.md).

```{important}
Finishing in-flight work happens on every backend. What needs **Kubernetes with
the default `kubernetes` discovery backend** is the *advance* notice — the
router learning a worker is leaving before the process is signalled. That is not
an implementation gap: it relies on the orchestrator knowing a Pod is being
removed, which nothing outside Kubernetes can tell the router.
`discoveryBackend: etcd` is rejected by the operator for in-cluster deployments.
```

## Request migration

`--drain-timeout` is how long one generation is worth waiting for. Most finish
well inside it; a long one may not, and something has to happen at the deadline.
By default it is cut, and the client reads that as a failure.

Request migration is the alternative: the generation is finished on a different
worker, so the client reads one uninterrupted stream and never learns a worker
changed underneath it. It covers the opposite case too — a worker that crashes,
is evicted, or drops off the network, where there was no advance notice and no
drain window to run out of.

It is off by default:

```bash
infera-server --migration-limit 1   # or $INFERA_MIGRATION_LIMIT=1
```

The limit is how many times one generation may be moved. `1` covers a worker
dying without letting a request wander the fleet during a broader outage.

Migration does not shorten the drain: a generation is moved only after it has
had the window it was promised, so a Pod exits at the same moment either way.
What changes is whether the client sees an error.

```{important}
The continuation is **not byte-identical** to what the original worker would
have produced: sampling state does not move with the request. Output stays
coherent and the seam is not visible to a reader, but a caller who needs
reproducible output for a fixed seed should leave this off.
```

### What it applies to

Streaming requests on mixed (non-PD) workers using the NATS request transport.
Everything else keeps the behaviour it had: PD streams and HTTP-transport
workers end with an error as before, and non-streaming requests are already
covered by the ordinary failover in `--request-max-retries`, which re-runs them
cleanly rather than stitching one together.

Accuracy depends on the engine. Where it reports the token ids it sampled — vLLM
on both endpoints, SGLang on completions — the continuation resumes from exactly
those; otherwise the decoded text is carried instead, which reads the same but
may not tokenize identically. The choice is automatic, and nothing fails because
the ids were unavailable.

A request can also stop being migratable partway through, and then ends with an
error rather than an approximation: streams the router cannot parse, and, on the
text path, tool calls or reasoning content, which do not appear in the text the
client receives.

### Observing it

`infera_migrations_total{reason}` counts generations moved, separating a rollout
doing its job (`worker_draining`) from a worker dying (`stream_broken`).
`infera_migrations_failed_total{reason}` counts the ones that could not be; a
rising `no_candidate` usually means the model has too few replicas for a
migration to have anywhere to go.

## Where the advance notice is missing

In-flight work is always finished. The head start is what varies, and two cases
do not get it:

**A worker stopped without its Pod being removed** — a liveness probe restarting
the container, a node shutting down, someone killing the process. Nothing
observes a deletion that never happens, so the router keeps sending work until
the worker itself drops out of the pool.

**Deployments outside Kubernetes**, which discover through an external etcd
where a worker record is simply present or absent, with nothing to observe that
a process is leaving.

In both cases a draining worker is absent from `/v1/workers` rather than shown
as draining. A Pod being deleted does show as `draining`, in the window between
its removal being requested and the drain beginning.

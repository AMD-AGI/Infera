## Problem and root cause

Auxiliary KV-metrics publication runs in the scheduler loop. A missing or slow PUSH-socket receiver can exhaust its queue; the default blocking `sock_send` then stalls request progress. This remains present on main `8a455f1`.

Our archived GLM-5.2 run `c44a` hung at this call (py-spy, October 9, ledger R001); `c44b` advanced past it with nonblocking metrics. Later stack throughput is not evidence for this patch, and the historical throughput runs used simulated MTP acceptance.

## Fix

Send auxiliary KV metrics with `zmq.NOBLOCK`, catch only queue-unavailable `zmq.Again`, and warn once per publisher. Keep KV-event publishing on its existing separate path. Unexpected socket errors still propagate. A healthy receiver continues to receive the serialized metrics after draining a full queue.

## Validation

Registered CPU tests use real ZeroMQ sockets for an unbound receiver, full high-water mark and recovery, plus closed-socket and non-EAGAIN controls. These tests do not require GPUs or models.

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/managers/test_kv_metrics_backpressure.py
pre-commit run --hook-stage pre-commit --files python/sglang/srt/managers/scheduler_components/kv_events_publisher.py test/registered/unit/managers/test_kv_metrics_backpressure.py
```

Local result: **4 passed**; all applicable pre-commit hooks passed.

## Draft blockers

- Maintainer agreement that auxiliary KV metrics are best-effort and may be dropped under backpressure.
- Upstream CPU CI and review. No GPU validation is needed for the socket contract; this is not a throughput claim.

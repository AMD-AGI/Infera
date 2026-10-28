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

## Review scope and CI

This change deliberately treats auxiliary KV metrics as best-effort telemetry: a full queue drops the current sample and warns once, so telemetry backpressure cannot stop request progress. Durable KV-event publication remains on its existing separate path. Please review this policy and the bounded logging behavior.

Author-side validation is complete for this CPU/socket-only change at `82176e70284c8d67b41de74b8e8005d148e57c8b`: the four real-ZMQ regression cases and applicable local pre-commit hooks pass, and upstream lint/title checks pass. No GPU or model execution is needed to validate this socket contract; no throughput improvement is claimed.

Ready for review. Upstream CPU test execution still requires the maintainer-controlled `run-ci` label; the existing gated/skipped test workflows are not claimed as passing. Maintainer review and required upstream checks remain merge requirements.

<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #40492223128](https://github.com/sgl-project/sglang/actions/runs/40492223128)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #40492222991](https://github.com/sgl-project/sglang/actions/runs/40492222991)<!-- slot:pr-test-extra:end -->
Latest PR Test (AMD ROCm 10): <!-- slot:pr-test-amd:start -->:x: [Run #40492222914](https://github.com/sgl-project/sglang/actions/runs/40492222914)<!-- slot:pr-test-amd:end -->
<!-- pr-states:end -->
Marked ready for review at unchanged head `82176e7`. The four registered real-ZMQ CPU regressions pass, including missing/full receivers, recovery and non-EAGAIN errors; upstream lint/title also pass. The body now distinguishes the explicit best-effort telemetry policy from merge requirements.

Could a maintainer enable `run-ci` for the registered CPU test and review the drop-on-backpressure policy? No GPU/model execution is required for this socket-only change. Existing gated/skipped tests are not claimed as passing.

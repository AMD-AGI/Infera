A PD request currently keeps Prefill's routing load charged while Decode continues generating, even after Prefill has completed. This adds opt-in independent Prefill accounting through `INFERA_PD_PREFILL_GUARD_RELEASE=completion` (native Rust CLI: `--pd-prefill-guard-release completion`). The default remains `decode`.

The existing accounting entry moves to the Prefill response/drain without firing duplicate start/finish hooks. It is released after the full HTTP body or NATS terminal frame, or when that drain fails or is cancelled. Decode retains its own entry. HTTP streaming/unary and NATS paths preserve main's existing disconnect, failed-stream and drain-timeout abort handling. This does not change the scoring formula, physical KV ownership, or the Python router.

Validation:
- Full ordinary Router suite: 289 unit, 36 HTTP functional, 4 ZMQ and 14 render-probe tests pass; broker-dependent tests are explicitly ignored by this command.
- New lifecycle cases cover default versus completion mode, response headers versus body EOF, HTTP 200/500 bodies, stalled-task cancellation, and Prefill completion while unary Decode headers are still pending.
- New real-broker test passes for NATS data followed by both successful Done and Error terminal frames.
- Four existing real-broker tests were also run: three pass; `the_bucket_bootstraps_a_cold_start_without_clobbering_a_live_view` fails identically on clean main `ff75ec65` (0 cached blocks versus expected 3 after an empty snapshot). This PR changes neither that test nor KV snapshot handling.
- `cargo clippy --all-targets -- -D warnings`, formatting and diff checks pass.

Motivating development-build measurements found lower Prefill queueing in two same-node comparisons at different KV capacities. Those historical results are not a fresh GPU performance validation of this extraction against current main.

This is an independent lifecycle PR, without the experimental R2/R3/R4 scoring, session-affinity implementation, or benchmark archives. It preserves main's current abort machinery; future metrics/session changes touching these paths should retain both lifecycles when rebased.

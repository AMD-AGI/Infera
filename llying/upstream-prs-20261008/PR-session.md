Add explicit, opt-in session affinity to the Rust router. `INFERA_SESSION_AFFINITY=off|prefill|both` consumes `X-Dynamo-Session-ID` without changing model input. Prefill and Decode keep independent worker/rank bindings; their rank numbers need not match. `both` also supports aggregated workers. Defaults and requests without a session header keep normal routing.

Bindings are scoped by model/session/role. Per-session locking makes simultaneous first requests choose one target. Active leases prevent idle expiry; TTL defaults to 3600 seconds and the map is bounded at 65,536 entries. Changed/unavailable targets and upstream failures invalidate bindings, with generation checks protecting replacements from late callbacks. At capacity, new entries fall back to ordinary routing. State is local to one Router process and is lost on restart.

Pinned selection uses the original `RouteTarget`, preserving rank-specific cache lookup and load accounting. The implementation is independent of R1, experimental scoring and Decode radix: it does not change load-release policy or enable an engine cache. Existing main stream completion/abort handling is retained. Session counters have role labels, not session-ID labels.

Validation:
- 294 unit tests, 39 HTTP functional tests, 4 ZMQ tests and 14 render probes pass using an isolated build directory; broker-dependent cases are reported as ignored by the ordinary suite.
- New coverage includes TTL/active leases, concurrent first picks, worker replacement, capacity, stale failure callbacks, original rank accounting, per-role header routing, unchanged model messages, aggregated routing, and invalidation after a truncated Decode stream.
- A real-NATS integration test passes for both unary and streaming 4xx responses, verifying the next request selects a new binding. Terminal NATS error/status handling also has unit coverage.
- Four existing broker tests were checked: three pass; the existing empty-snapshot preservation test fails identically on clean main `ff75ec65` (0 versus expected 3 blocks). No KV snapshot code is changed here.
- `cargo clippy --all-targets -- -D warnings`, formatting and diff checks pass.

Historical GLM-5.2 experiments motivate this capability: P affinity reduced session migration/misses; D affinity under radix greatly increased local prefix reuse. These are not a guarantee of throughput improvement for every workload, nor fresh GPU validation of this extraction.

This is cross-turn session affinity, not same-number P/D rank affinity. It has no dependency on closed PR #161. This PR and #185 both touch guard ownership paths and should be rebased together carefully when one merges; each is independently based on main. General serving-metrics expansion remains deferred to #183.

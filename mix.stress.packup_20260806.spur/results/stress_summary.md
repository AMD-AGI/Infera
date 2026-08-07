# Task 3 — agentic stress (GLM-5.2 mix, Case-A fix.yaml, closed-loop)

Config: initial_sessions=8, max_inflight=16, max_sessions=24 (mission), new_session_rate=0.20
(re-solved off run1 to drive toward the in-flight cap; init_sessions kept at 8 per mission).
ramp 400s + sustain 3600s (honest window, operator). infera driver agent.agent_throughput
against router :8100. NOT the customer/AgentX bench (rule 5).

## Sustain-phase result (the number)
| metric | value |
|---|---|
| requests completed | 2092 (over 3600s) |
| success rate (whole run) | 98.2% (31 err / 2364) |
| offered QPS | 0.58 req/s |
| cache-hit rate | 89.0% (Case-A target 88-90%) |
| live sessions (steady) | ~23-24 (at session cap) |
| in-flight (steady) | ~15-16 (at in-flight cap => backpressure binding) |
| TTFT p50 / p90 | 1460 / 3610 ms |
| TPOT p50 / p90 | 19.0 / 32.6 ms |
| input TPM | 2,988,618 |
| uncached TPM / GPU | 41,330 |
| gen (visible) TPM / GPU | ~3,381 |

Interpretation: the in-flight cap (16) binds in steady state, so this is a genuine
saturation point — the server holds ~24 live sessions with 89% prefix reuse and 98%
success. TPOT p50 19ms and cache-hit 89% agree with the reference kit numbers.
Note run1 (rate 0.05, archived under results/stress_run1_calib) under-loaded to liveN~4;
rate raised to 0.20 for this stress run.

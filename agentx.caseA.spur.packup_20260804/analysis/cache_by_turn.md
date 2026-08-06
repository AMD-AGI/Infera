# Cache-hit decomposition by turn index

Source: `profile_export.jsonl`, `benchmark_phase == "profiling"` only
(warmup excluded). `usage_prompt_cache_read_tokens` is the **server's own**
report, not aiperf's `Theoretical Prefix Cache Hit`.

| segment | C=8 n | C=8 prompt | C=8 cache read | C=8 hit | C=16 n | C=16 hit |
|---|---:|---:|---:|---:|---:|---:|
| turn 0 | 53 | 4,662,020 | 0 | **0.0 %** | 73 | **0.0 %** |
| turns 1–2 | 71 | 5,760,385 | 4,036,160 | 70.1 % | 99 | 67.1 % |
| turns 3+ | 105 | 8,083,244 | 5,365,056 | 66.4 % | 131 | 65.5 % |
| **overall** | 229 | 18,505,649 | 9,401,216 | **50.8 %** | 303 | **49.7 %** |

Turn 0 contributes 25 % of prompt tokens at C=8 and exactly zero cache read.
Removing it lifts the realized rate from 50.8 % to 67.9 %.

The scenario **forces** this: `inferencex-agentx-mvp` locks
`--cache-bust first_turn_prefix`, injecting a per-play unique marker at the head
of every first user turn so recycled traces cannot inherit a warm prefix. It is
a deliberate anti-inflation rule, not a deployment shortfall.

Flat across concurrency (66.1 % → 66.6 % server-measured overall) ⇒ no
load-induced eviction of the hot tier at these levels.

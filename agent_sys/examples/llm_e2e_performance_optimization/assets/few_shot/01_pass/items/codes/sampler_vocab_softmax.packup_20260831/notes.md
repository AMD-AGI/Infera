# Notes — what bit, and what is still open

**`--max-hours` at or below 2.0 silently ruins the campaign.**
`kernel_agents/cli.py:1391` gates hardware profiling on `max_hours > 2.0`
(strictly), and the implementer turn cap drops 500 → 100 at the same threshold.
Nothing warns. A first attempt at `1.0` produced a report that looked fine and
had static-only analysis behind it.

**A non-clean `CLAUDE_CONFIG_DIR` crashes the backend probe.** With plugins or
MCP servers configured, `claude --print --output-format json` returns a message
array rather than an object and KernelForge does `payload.get("result")` on it
(`forge_llm/agent_backends/claude.py:333`). Point it at an empty directory.

**Container root cannot write an NFS `$HOME`.** `root_squash` maps root to
nobody. Three writes failed and only one was loud; the Triton cache and the
experience KB both failed in silence. `TRITON_CACHE_DIR` and
`KNOWLEDGE_LOCAL_ROOT` must point at local disk.

**`kernel-agents list` looks in the wrong place by default** —
`<project_root>/experiments`, not `<workspace>/forge_experiments`. Pass `--dir`
or conclude, wrongly, that the campaign produced nothing.

**The agent recorded its own failed attempt, and that is the most useful thing
in the tree.** An earlier iteration folded the combine step into normalize, so
every program redid the row reduction; it regressed at low batch because the
redundant work grows as `batch × segments × next_pow2(segments)` — about 8 MB of
extra traffic at B1. The final kernel folds *conditionally*, above 256 segments.
`results/candidates_index.jsonl` carries the reverted iterations and their
reasons.

**Open: the optimized kernel's measurement spread.** 19–21% round to round at B1
and B8, against 1.7–2.2% for the baseline, measured as medians of five fresh
processes. The cause is **not identified**. Candidates not tested: launch jitter
across three kernels versus one, Triton JIT cache state, interference from other
tenants. The practical consequence is that a single measurement of this kernel
is not evidence — an earlier single sample of 21.67 µs was written up as a
regression before repeats showed the median was 18.9 µs.

**Open: the day-to-day gap.** Forge measured 2.8328×; a re-measurement the next
day on the same host measured 2.6123×. Inside the spread above, but not
explained.

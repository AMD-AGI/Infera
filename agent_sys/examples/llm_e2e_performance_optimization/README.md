# `llm_e2e_performance_optimization`

The end-to-end LLM performance-optimisation flow, as an `agent_sys` task
package. **This directory is not itself a task package** — it is the container.
The package is [`e2e-flow/`](e2e-flow/), and `--package` points at that folder,
never at this one.

`e2e-flow` runs five stages in **one** `agent-sys run`, because a handoff only
travels inside one run's graph:

| stage | task | what it does |
|---|---|---|
| 1 · deploy | `m1_deploy` | a plain instruction in, a proven deployment and a runnable delivery kit out |
| 2 · profiling | `m2_profiling` | replay a trace against the live deployment, cut a profiler window, rank the kernels |
| 3 · analysis | `m3_analysis` | turn that profile into a ranked operator list and a KernelForge workset per operator |
| 4 · kernel optimisation | `m4_kernel_opt` | hand one workset to KernelForge, hand back the optimised kernel and its evidence |
| 5 · integration | `m5_integration` | put that kernel in front of the real service and decide whether it broke or slowed anything |
| 6 · regression | — | not built yet |

Read, in this order: [`e2e-flow/README.md`](e2e-flow/README.md),
[`e2e-flow/CONTRACT.md`](e2e-flow/CONTRACT.md) — the fifteen-kind cross-module
contract — and [`todo.md`](todo.md), which carries everything the flow defers.

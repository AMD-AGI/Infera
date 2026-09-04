# `llm_e2e_performance_optimization`

The six-stage end-to-end LLM performance-optimisation flow, as `agent_sys` task
packages. **This directory is not itself a task package** — it is the container.
Each stage is a self-contained package in its own folder, and `--package` points
at the folder, never at this one.

The series book is `temp/mission.md`; `temp/bugs/` records `agent_sys` bugs found
while driving these, and `temp/claude_code_skill_used_by_human/` carries the
packup skill the deliverables are shaped by.

| stage | folder | what it does |
|---|---|---|
| 1 · e2e 运行 | [`deploy-demo/`](deploy-demo/) | a plain instruction in, a proven deployment and a runnable delivery kit out |
| 2 · profiling | [`profiling-demo/`](profiling-demo/) | replay a Mooncake trace against a live deployment, cut a profiler window, rank the kernels |
| 3 · analysis + workset | [`analyze-demo/`](analyze-demo/) | turn that profile into a ranked operator list and a KernelForge workset per operator |
| 4 · kernel optimization | [`kernel-opt-demo/`](kernel-opt-demo/) | hand one workset to KernelForge, hand back the optimised kernel and its evidence |
| 5 · integration | [`integration-demo/`](integration-demo/) | put that kernel in front of the real service and decide whether it broke or slowed anything |
| 6 · regression | — | not built yet |

Each folder's `README.md` is how to run that stage and what it delivers; where a
folder also carries `DESIGN.md`, that is the reviewed design behind it.

A sixth package, `../llm_e2e_perf_opt_debug_workset/`, is not a stage: it is a
one-leaf harness that checks a candidate workset against stage 4's contract in
seconds, before committing to a campaign.

# agent_sys examples — a ladder

Each rung adds one thing to the one before it. Start at the top.

| level | package | what it shows | needs |
|---|---|---|---|
| 1 | [`01_hello_world`](01_hello_world/) | a program task and an AI task under one subgraph; a validator that checks the agent instead of trusting it; running on the host and in docker | the `claude` CLI |
| 2 | [`02a_sglang_serve_qwen36`](02a_sglang_serve_qwen36/) | an AI agent deploys Qwen3.6 on sglang and hands back a runnable, proven deploy kit | 1 GPU, model weights, an infera+sglang image |
| 2 | [`02b_profiling_qwen36`](02b_profiling_qwen36/) | the same deployment, then the load replayed twice — profiler off (online numbers) and on (torch trace, kernel table) | as 2a, plus the aiperf image |
| 3 | [`03_kernelforge_one_kernel`](03_kernelforge_one_kernel/) | optimise one kernel with KernelForge and prove the speed-up by re-measuring it | as 2a, plus a KernelForge checkout |
| 4 | [`llm_e2e_performance_optimization`](llm_e2e_performance_optimization/) | the full five-stage flow: deploy, profile, analyse, optimise, integrate | a GPU node and the stages above |

Levels 2 and 3 do not copy level 4: their `steps/` and `assets/<object>` are links
into `llm_e2e_performance_optimization/e2e-flow`, and each writes only its own
`main.yaml`, root readme and README. A run stages a copy of whatever a link
names, so a linked object behaves exactly as if the package held it.

Packages that exist to be run as end-to-end tests rather than read live in
[`../tests/e2e_test/`](../tests/e2e_test/); a demo that is also a test is linked
from there.

## Before any of them

From the repository root:

```bash
pip install -e "agent_sys[claude]"
PYTHONPATH=$PWD/agent_sys python3 -m cli.main show --package agent_sys/examples/01_hello_world
```

`show` loads and type-checks a package and dispatches nothing; run it after
every edit. `run` is the same command with the work.

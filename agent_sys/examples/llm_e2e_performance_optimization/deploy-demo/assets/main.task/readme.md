# main — the LLM end-to-end performance-optimisation flow

This is a **non-leaf**. It runs nothing itself; its work is its subgraph, and
this file exists because `closure` spec §2.6 requires a body and a non-leaf's
body is a readme.

## The flow this package will hold

The series book (`../temp/mission.md`) splits end-to-end optimisation into six
stages:

1. **e2e 运行** — a plain instruction in, a proven deployment and a runnable
   delivery kit out. **This is the only stage declared today.**
2. **profiling** — run the bench tools against the deployment and hand back the
   standard profile artefacts: the JSON directory, the command-line analysis,
   and each tool's own standard output.
3. **analysis** — from the profile, a list of operators worth optimising, each
   with a complete workset: environment, sglang integration points, the operator
   itself, a naive PyTorch reference, runnable correctness and performance
   tests, and the profile figure it was cut from.
4. **kernel optimisation** — the workset into a nested kernel-forge agent, out
   comes a correctness-checked, measured, summarised kernel.
5. **integration** — the optimised kernel back into the engine, with end-to-end
   correctness, an eval, and a performance regression.
6. **regression** — the standing bar the next change is measured against.

Each stage arrives as another entry in `main.yaml`'s subgraph with its own step
file, and each stage boundary becomes one handoff kind. Stage 2 will consume
`deploy_kit`; that is why stage 1's output is a *kit* rather than a running
server.

## Why stage 1 is the one that exists first

Every later stage needs a deployment it can point at, and needs it to be the
*same* deployment each time. Standardising the deploy step is therefore not the
easy part of the flow taken first — it is the part the other five rest on. A
profiling stage that runs against a hand-built server measures that server, not
the model.

## What this stage is not

- **Not a benchmark.** No throughput, latency or accuracy claim is made or
  checked here. That is stage 2's, and stage 2 does not exist yet.
- **Not a claim about the engine.** The subject is the *deployment procedure*
  and its written form. Whether the engine is fast, or correct, is measured
  further along.
- **Not a sandbox claim.** This package is run with `agent_sys`'s permission
  management **off**, and its banner says so. A pass here is not evidence about
  isolation.

# `e2e-flow` — the five stages, in one graph

Deploy → profile → analyse → optimise one kernel → integrate, as **one
`agent-sys run`**.

The five stages also exist as five separate packages next door
(`../{deploy,profiling,analyze,kernel-opt,integration}-demo/`), each driven to a
real cluster run on 2026-09-02. They are not a flow: **a handoff only travels
inside one run's graph**, so five packages are five runs and nothing chains.
This package is the join the repo-root `mission.md` asks for. The five demos are
kept, untouched, as reference and as a fallback.

## Status — 2026-09-03

**Phase 0 complete: the contract is frozen and the graph loads.** The bodies are
skeletons; every validator's `check.py` exits 1 on purpose, so nothing here can
report a pass it has not earned.

```
17 closures · 15 handoff kinds · 20 validators
```

## Read these, in this order

| file | what it settles |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | **the frozen cross-module contract** — the fifteen kinds, the naming rule, the environment rule, the schema rule, and what each module deletes |
| [`MOCK-MAP.md`](MOCK-MAP.md) | which sealed handoff stands in for which kind, and the six adaptations that are real work rather than a copy |
| [`assets/schemas/README.md`](assets/schemas/README.md) | who writes which schema, and against which real artefact |
| [`../todo.md`](../todo.md) | everything the mission deferred, with what would settle it |

## Run it

Six variables carry no default, because they are facts about one allocation on
one cluster and a default would be one machine's answer shipped as everyone's.

```sh
python3 -m agent_sys.cli.main show \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --var jobid=106250 --var node=crsuse2-m2m-061 --var node_ip=10.245.159.129 \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=infera/engine-sglang:gfx950-local
```

`show` loads and type-checks every yaml, derives the edge set from the handoff
wiring, checks it against every `froms`, and dispatches nothing — **in under a
second.** It is the loop; run it after every edit. `run --dry-run` is the next
rung, `run` with `--var mock_stages=all` the one after.

Promote one stage at a time out of mock — `--var mock_stages=m2,m3,m4,m5`, then
`m3,m4,m5`, and so on — so that a failure is attributable to the stage that was
just promoted.

## The shape

```
main
├── m1_deploy          deploy_and_prove                    → deploy_kit
├── m2_profiling       run_profiling_mode_off  ┐
│                      run_profiling_mode_on   ├─ parallel → profiling_evidence
│                      merge_profiling_evidence┘
├── m3_analysis        rank → identify → build_workset     → operator_workset
├── m4_kernel_opt      optimize_kernel                     → kernel_optimization
└── m5_integration     apply_patch → integrate_and_verify → packup
                                                           → e2e_packup  (is_end)
```

`deploy_kit` reaches all four later stages: it carries the environment record,
and m3 and m4 have to *run* things — a workset's tests, then KernelForge — in
the container m1 brought up.

## Three things a reader will otherwise assume wrong

1. **There is no `serve_*` task anywhere.** Bring-up and use may not be split
   across agents (M2.5, M5.2), so a task that needs a service brings it up in
   its own STEPS and tears it down. That deleted five tasks and eight handoff
   kinds relative to the demos.
2. **`items_schema` is not the schema layer.** For a file item it validates the
   *filename string*; the contents are never read
   (`handoff/content.py:184-197`). The schema layer mission G2 asks for is
   `assets/schemas/`, loaded from both sides by `assets/lib/schema.py`.
3. **The 5% / 10% regression bars are measured and must not be widened.** The
   within-arm round-to-round spread on a steady node is ~2%. A previous round
   widened them to 35% / 30% in response to a *cross-instance* artefact; the
   missing control is a comparability gate at bring-up, not a looser bar
   (`../todo.md` T7).

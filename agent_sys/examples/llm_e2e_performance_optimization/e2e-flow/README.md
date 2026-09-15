# `e2e-flow` — the five stages, in one graph

Deploy → profile → analyse → optimise one kernel → integrate, as **one
`agent-sys run`**.

The five stages began as five separate task packages, one per stage. They were
not a flow: **a handoff only travels inside one run's graph**, so five packages
are five runs and nothing chains. This package is the join.

```
17 closures (11 leaves + 6 non-leaves) · 15 handoff kinds · 21 validators
```

## Read these, in this order

| file | what it settles |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | **the cross-module contract** — the fifteen kinds, the naming rule, the environment rule and the schema rule |
| [`DESIGN.md`](DESIGN.md) | the design: handoff kinds, validators, the task split per stage, and the two debugging tools |
| [`MOCK-MAP.md`](MOCK-MAP.md) | which sealed handoff stands in for which kind, and the six adaptations that are real work rather than a copy |
| [`assets/schemas/README.md`](assets/schemas/README.md) | who writes which schema, and against which real artefact |
| [`../todo.md`](../todo.md) | everything the mission deferred, with what would settle it |

## Run it

Ten variables carry no default, because they are facts about one allocation on
one cluster and a default would be one machine's answer shipped as everyone's.

```sh
python3 -m agent_sys.cli.main show \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --var jobid=1 --var node=n --var node_ip=0.0.0.0 \
  --var model_name=m --var model_path=/p --var image=i
```

`show` loads and type-checks every yaml, derives the edge set from the handoff
wiring, checks it against every `froms`, and dispatches nothing — **in under a
second.** It is the loop; run it after every edit. `run --dry-run` is the next
step, `run` with `--var mock_stages=all` the one after.

Promote one stage at a time out of mock — `--var mock_stages=m2,m3,m4,m5`, then
`m3,m4,m5`, and so on — so that a failure is attributable to the stage that was
just promoted.

### The variables a real run has to supply

Derived from the package itself, not from a launch record:

```sh
# every `${name}` with no default anywhere in main.yaml / shared.yaml / steps/
grep -rhoE '\$\{[a-z_0-9]+\}' main.yaml shared.yaml steps/*.yaml | tr -d '${}' | sort -u
# every `${name:-}` — has a default, and the default is empty
grep -rhoE '\$\{[a-z_0-9]+:-\}' main.yaml shared.yaml steps/*.yaml | sed -E 's/.\{(.*):-.$/\1/' | sort -u
```

| variable | what it is |
|---|---|
| `jobid`, `node`, `node_ip` | the allocation and the host the engine is brought up on. `_agree_or_die` refuses rather than guessing when one of these disagrees with the sealed environment record |
| `model_name`, `model_path` | the served name and the weights directory on that host |
| `image` | the engine image. **The sealed kit's when a stage is replayed, the node's when it is real** — these are different values and the difference has cost a run |
| `tp`, `expect_ranks` | tensor parallelism, and the rank count a capture must contain. **`expect_ranks` must equal the deployment's `tp`** — its default does not track `tp`, and a mismatch makes `check_trace_coverage` refuse a correct capture |
| `gpu_devices`, `measure_gpu` | the cards the engine takes, and the card measurements are taken on |
| `mock_stages` | which stages replay a sealed handoff instead of running. `none`, `all`, or a comma list such as `m2,m3,m4,m5` |
| `work_root`, `scratch_root` | **must be local disk.** A root-squashed network home makes the engine fail to write its logs *silently* |
| `container` | the container name. An identifier bound on a shared host is a parameter, never a constant |
| `transport`, `transport_env` | how a task reaches the work host. Both are needed and they are not the same variable: `transport` names the mechanism, `transport_env` carries what that mechanism needs |
| `bench_rounds`, `adhoc_cases`, `aiperf_trace`, `gsm8k_data` | the measurement inputs. **`adhoc_cases` is conditioned on whether stage 5 is real, not on how far the chain has been promoted** |

**Audit the whole table against your `mock_stages` every launch, not the rows you
remember.** Several of these take one value when a stage is replayed and another
when it is real, and a real value carried into a replayed stage produces a
refusal that reads exactly like a producer defect.

**A run does not record which `--var` it was given.** The staged package keeps
`${var:-default}` unrendered, so the launch line cannot be read back out of the
artefacts — write it down beside the run.

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
   (`handoff/content.py`). The schema layer mission G2 asks for is
   `assets/schemas/`, loaded from both sides by `assets/lib/schema.py`.
3. **The 5% / 10% regression bars are measured and must not be widened.** The
   within-arm round-to-round spread on a steady node is ~2%. A previous round
   widened them to 35% / 30% in response to a *cross-instance* artefact; the
   missing control is a comparability gate at bring-up, not a looser bar
   (`../todo.md` T7).


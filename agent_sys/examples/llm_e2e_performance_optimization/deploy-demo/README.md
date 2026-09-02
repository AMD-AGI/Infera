# `llm_e2e_performance_optimization` — stage 1: `e2e_deploy_standardized`

The task package for the LLM end-to-end performance-optimisation flow. The flow
has six stages (`assets/main.task/readme.md` lists them); **this package
declares the first**, and declares it in the shape the other five will attach
to.

Stage 1 is *e2e 运行*: a plain instruction in, a proven deployment and a
**runnable delivery kit** out.

## Running it

Four variables have no default, because the model, the weights, the image and
the operator's skill directory are facts about a site rather than about this
package:

```bash
agent-sys run --package agent_sys/examples/llm_e2e_performance_optimization/deploy-demo \
  --var model_name=Qwen/Qwen3.6-27B \
  --var model_path=/shared_nfs/yihou/models/Qwen3.6-27B \
  --var image=infera/engine-sglang:gfx950-local \
  --var packup_skill=$HOME/.claude/skills/experiment-result-packup \
  --demo-root /shared_nfs/yihou/agent_sys_debug/runroot
```

Omit any of the four and the load fails naming the file, the line and the
variable. The optional ones are `etcd_image`, `work_root` (default `/var/tmp`),
`deploy_mode` (default `mix`), `tp_size` (default `1`), `instruction`, and
`reproduce_timeout_seconds` (default 5400 — pass something small when testing
the wiring rather than the workload).

`--demo-root` is the one knob that relocates runs, handoffs, playground and
workspace together; on a cluster where the login node and the GPU node share
only a network filesystem, it is what puts them somewhere both can see.

## The shape

```
main                            non-leaf, no agent, one entry
  └── e2e_deploy_standardized   leaf, agent: ai (Claude Code SDK), is_end
        outputs: deploy_kit                 content_type: code
          check_deploy_kit           program, seconds,    completeness / strong
          check_deploy_reproduces    ai,      gpu_hours,  usability    / weak
```

One leaf, one handoff kind, two validators — deliberately
`examples/single_real_task`'s shape, because that shape was exercised end to end
before this package existed.

## What this package adds over `single_real_task`

`single_real_task` proved **one** deployment: Qwen3.6-27B, mix mode, gfx942. It
names the model in its agent brief, in its validator and in its task name. This
package makes the same work **standardised**:

1. **The model is a parameter.** Nothing in `main.yaml` or `steps/deploy.yaml`
   names a model, an architecture or a port. The same closure runs Qwen3.6-27B
   and GLM-5.3-Flash, and only `--var` changes.
2. **The instruction is an input.** `$E2E_INSTRUCTION` is the stage's actual
   argument; the brief is written around it rather than around one bring-up.
3. **A kit must be machine-readable.** `results/` must carry at least two
   non-empty `.json` files, because stage 2 (profiling) consumes results and
   consumes files rather than paragraphs.
4. **A kit must say what success looks like.** `REPRODUCE.md` must carry an
   `Expected output` section — it is the only criterion the reproduction check
   has.
5. **A kit must name the three facts a reproduction fails without**: the GPU
   architecture, the image, the model.
6. **A kit must not register the model under a filesystem path.**
7. **A kit must evidence the deployment mode from two independent components** —
   the worker's own log line and the router's worker listing. The mode is
   selected by *omitting* a flag, so the launch command is not evidence of it.
8. **A kit must carry a real answer**, not only a health check: one file with
   `"finish_reason": "stop"` and a non-empty `"content"`.
9. **The exemplars are data.** `assets/e2e_deploy_standardized.task/examples/`
   holds sanitised kits from runs that passed, one per model. A run that passes
   is how the next run gets a better example — and what the exemplar
   demonstrates becomes rules 7 and 8 rather than staying prose.

## Evidence that the shape check is not vacuous

Both controls were run before any agent run, against kits nobody wrote for this
validator (2026-09-01):

| kit | verdict | what it shows |
|---|---|---|
| `glm53flash.mix.packup_20260830` — a human-written kit | **FAIL**, 5 faults | unfilled `<your-token>` / `<branch>` placeholders, no `Expected output` section, 0 `.json` files in `results/`. Each fault is specific and each is real |
| the `b6` kit from `single_real_task`'s accepted run — which **passed** that package's shape check | **FAIL**, 6 faults, all of them the same rule | every evidence file shows the model served as `/data/<user>/…/Qwen3.6-27B`. The kit satisfies every inherited rule; only the new served-name rule bites, and it bites on exactly the trap that run recorded in its own notes |

The second row is the one worth reading: a validator that passed everything it
was pointed at would be worth nothing, and this one refuses a kit that its
predecessor accepted, for a reason written down before the check existed.

When rules 7 and 8 were added from the exemplars, the same two controls were
re-run and a third added (2026-09-01):

| kit | verdict |
|---|---|
| the Qwen kit from the run that passed both validators | **PASS** |
| the GLM kit from the run whose shape check passed | **PASS** |
| a copy of the Qwen kit with `chat_completion.json`, `verification.json` and `worker_mode_line.txt` deleted | **FAIL**, on exactly the two new rules and nothing else |

## What it does not do

Stated because an example that overclaims is worse than none.

- **No `resources` block.** The leaf legitimately wants GPUs, but `cli/build.py`
  is the only reader of that field and the CLI composition root declares no
  pools, so writing one would name a pool that does not exist. Nothing here
  reserves a GPU; the agent's brief tells it to tear its server down for exactly
  that reason.
- **`repos: [infera, sglang]` is declared and read by nothing.** It is where the
  schema says such a fact goes; `env_mgr` does not act on it today.
- **No throughput, latency or accuracy claim.** Those are stage 2's, and stage 2
  does not exist yet.
- **Nothing about isolation.** This package runs with permission management
  **off**; the run banner says so. A pass here is not evidence about the
  sandbox.
- **Stages 2–6 are not here.** `main.yaml`'s subgraph has one entry.

## Cost, and why it is not in the test suite

This package needs real hardware, real weights on disk, and credentials for the
agent backend. Nothing imports it, nothing loads it during `pytest`, and it is
data in the tree like every other example.

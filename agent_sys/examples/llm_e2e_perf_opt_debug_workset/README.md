# `llm_e2e_perf_opt_debug_workset` — the single-task debug main

The series task book asks each stage for *"单独调试用的 single task main"*. This
is stage 4's.

**One leaf, no agent, no GPU, no credentials, seconds to run.** It publishes one
workset and validates it. That is all it does, and that is the point: it
exercises the `workset` contract — the program body, the grant, the `code`
content type, the shape validator — without dispatching the Opus agent that
costs money and the KernelForge campaign that costs hours.

## Running it

```bash
E=agent_sys/examples
docker exec -u "$(id -u):$(id -g)" <container> bash -lc "
agent-sys run --package $E/llm_e2e_perf_opt_debug_workset \
  --demo-root /tmp/yihou/<fresh> --timeout 300 \
  --var real_package=\$PWD/$E/llm_e2e_performance_optimization/kernel-opt-demo \
  --var workset_dir=\$PWD/$E/llm_e2e_performance_optimization/kernel-opt-demo/assets/worksets/sampler_vocab_softmax"
```

Measured 2026-09-01: full green, `check_workset_shape: PASS`, under 30 s.

**The use it is actually for**: point `--var workset_dir=` at a workset somebody
just produced — by hand, or by stage 3 when it exists — and find out whether it
satisfies the contract *before* handing it to a three-hour campaign. A workset
missing `measure_baseline.py`, or whose `driver.py` never prints `case_ms:`,
fails here in seconds instead of after an hour of a campaign measuring nothing.

## Why it is a sibling directory

`spec_loader.YamlPackage` scans **every `*.yaml` under the package root except
`assets/`**, so a second `main.yaml` inside the real package would be loaded on
every ordinary run and collide with the real one. `examples/demo-broken/` is a
sibling for the same reason.

There is no CLI flag for "run one leaf" — `cli/main.py`'s parser has no
`--task`/`--only`/`--from` — so a second package is the supported route. The
alternative, and the one the real package also offers, is `--var mock=1`: same
graph, no campaign. Use `mock=1` to debug the *agent* and this package to debug
the *workset*.

## The bodies are shims, not copies

`entry.sh` in both asset folders execs the **real** package's body via
`$KFO_REAL_PACKAGE`. Only the two files the assets convention requires — a
readme and an entry — are duplicated, and both are four lines.

**Symlinking the whole asset directory was tried first and does not work.**
`spec_loader/assets.py:105` walks `assets_root.rglob("*")`, and `rglob` does not
descend into symlinked directories, so the bodies were never found and the load
failed with *"declares neither a body nor members"*. Recorded here so nobody
spends the same twenty minutes.

`check.py` resolves its `zone` helper from its own `__file__`, so running it out
of the real package picks up the real `assets/lib/zone.py` too — no second copy
of that either.

## What can still drift

`steps/debug.yaml` restates the validator's `args` — the required file list, the
driver tokens, the case floor. **Those are duplicated and nothing enforces that
they match** `llm_e2e_performance_optimization/kernel-opt-demo/steps/kernel_optimization.yaml`.
If they diverge, a workset that passes here starts failing there and this
harness becomes a liar. It is the one real maintenance cost of the split, it is
not solved, and it is written here rather than left to be discovered.

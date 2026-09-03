# Mock map — which sealed handoff stands in for which kind

Mission Brief item 2: *"通过 mock ai agent，结合上轮单独运行的真实 handoff，构造
符合 validator 的。快速跑通 e2e 流程"*. The point of the exercise is that **the
validators are unchanged**. A mock that writes something a validator accepts but
a real producer would never emit proves nothing, so nothing here synthesises —
`assets/lib/mock.sh` copies bytes a real run on this cluster produced.

Source: `/shared_nfs/yihou/agent_sys/cheat_for_mock/`. **Read its `README.md`
first**; it documents four things that mislead.

## The map

| kind | mock source | adaptation needed |
|---|---|---|
| `deploy_kit` | `stage1-deploy/deploy_kit` | **environment.yaml** (A) |
| `profiling_mode_off.bench_result` | `stage2-profiling/aiperf_baseline` | (A) |
| `profiling_mode_on.bench_result` | `stage2-profiling/aiperf_profiled` | (A) |
| `profiling_mode_on.profile_result` | `stage2-profiling/torch_trace` | (A) · **360 MB** |
| `profiling_mode_on.kernel_table` | `stage2-profiling/kernel_table` | (A) · the **real** 124-kernel one |
| `profiling_evidence` | `stage2-profiling/profile_packup` | (A) · **content type differs** (B) |
| `kernel_worklist` | `stage3-analyze/kernel_worklist` | (A) |
| `operator_identity` | `stage3-analyze/operator_identity` | (A) |
| `operator_workset` | `stage3-analyze/operator_workset` | (A) · **merge with stage4/workset** (C) |
| `kernel_optimization` | `stage4-kernel-opt/kernel_optimization` | (A) |
| `patch_overlay` | `stage5-integration/patch_overlay` | (A) |
| `stock.measurement` | `stage5-integration/{deployment,acceptance,bench}_stock` | (A) · **three into one** (D) |
| `patched.measurement` | `stage5-integration/{deployment,acceptance,bench}_patched` | (A) · (D) |
| `integration_report` | `stage5-integration/integration_report` | (A) · **carries a refused verdict** (E) |
| `e2e_packup` | — none sealed — | **(F)** |

## The six adaptations, and why each is real work rather than a copy

### (A) `environment.yaml` does not exist in any sealed handoff

The fields do — as `items/env/deployment.json` and `context.json` — but the
document `assets/schemas/environment.schema.json` describes does not, and it
**deliberately requires three fields those records lack**: `gpu_arch`,
`gpu_count` and `image_id`. Verified: feeding the raw `context.json` to the
schema returns eight problems.

That is the schema doing its job, not an oversight. A mock therefore **renders**
an `environment.yaml` from the sealed record plus the run's own `--var`s. One
renderer, `assets/lib/env_render.py`, used by every mock and by every real
producer — **owner: leader**, since it is the one piece all fifteen kinds share.

### (B) `profiling_evidence` is `reproducible`; `profile_packup` is `code`

`check_items` rejects a top-level item the content type never declared, so
`items/codes/…` cannot be laid into a kind whose required items are `result` and
`env`. The mock must reshape: packup's `results/` becomes `result/`, its
`scripts/` becomes `script`.

### (C) `operator_workset` is the merged kind; neither source is it yet

`stage3-analyze/operator_workset` (27 files, two operators) and
`stage4-kernel-opt/workset` (10 files, one operator, `sampler_vocab_softmax`)
are the two halves M3.7 merges. Neither alone satisfies
`check_workset_shape`'s `min_shapes: 3` **and** `require_entrypoints:
[correctness, performance]`.

**This is m3's first deliverable and m4 is blocked on it** — the two owners
agree the merged shape before either writes a body.

### (D) Six evidence kinds became two

`stock.measurement` folds `deployment_stock` + `acceptance_stock` +
`bench_stock`; likewise patched. The union is mechanical (`env/` merges,
`result/` gains subdirectories) but the `env/steps.json` that
`check_measurement_order` reads has to be **written**, since no sealed artefact
carries the merged step order.

### (E) `integration_report`'s sealed verdict is `false`

```
check_no_regression   result: false   strength: strong
```

It is a **sound sample of a refused report and a misleading one if taken for a
passing example.** Because `check_no_regression` is `strong`, mocking it
verbatim will correctly stop the graph — which is *also* a useful test, once,
of the refusal path.

Two mock modes, and the run should exercise both:
- `--var mock_report=refused` — verbatim. Expect the graph to stop at m5. This
  is the only cheap end-to-end test of a strong refusal we have.
- `--var mock_report=accepted` — the same report with the two arms' numbers
  taken from the **stock control** measured under matched load (patched 475.7 ms
  vs stock 470.3 ms mean ITL, 1.1% apart), which is what a non-regressing run on
  this cluster actually looks like.

### (F) `e2e_packup` has no sealed source

`integration_packup` was never sealed — the graph stopped at (E) before it was
dispatched. An unsealed 47-file packup exists at
`/shared_nfs/yihou/agent_sys/debugging/integration/packup-out-of-band/`,
produced by `integration`'s own unmodified `packup.py` over the nine sealed
handoffs and graded PASS against the real `check_packup_shape` body.

Use it as the mock source, and note in the run's record that it is **not
sealed** and its provenance is `PRODUCED-BY-DEPLOY.md` in that directory.

## Digests

Every digest under `cheat_for_mock/` is invalid — a past `chmod -R 777` changed
every file's executable bit, and the tree digest is git-shaped so it records
exactly that. **Content is intact**;
`/shared_nfs/yihou/agent_sys/temp/leader/repair_modes.py` restores the modes by
searching candidate executable-sets for the one that reproduces the manifest's
own digest, which is proof rather than a guess.

Nothing on the consuming path verifies a digest anyway
(`env_mgr.fs.layout.copy_out` is a plain `copytree` — `todo.md` T9), so this
does not block the mock run. **It does mean the mock run cannot be used to test
digest verification.**

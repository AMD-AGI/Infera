# Handoff analysis — every kind in the five stages, and the two worksets compared

Written 2026-09-03. Two questions were asked and are answered in order:

1. **What does each stage's handoff look like, and what validates it?** (part I)
2. **The two `workset`-shaped handoffs, in detail.** (part II)

Sources: the packages' own `steps/*.yaml` for the declarations, and the 25 real
sealed handoffs under `/shared_nfs/yihou/agent_sys/cheat_for_mock/` for the
measured shapes. Where the two disagree, the measured shape is what is written.

---

# Part I — the handoff inventory

| stage | package | kinds | validators | terminal artefact |
|---|---|---|---|---|
| 1 e2e deploy | `deploy-demo` | 1 | 2 | `deploy_kit` |
| 2 profiling | `profiling-demo` | 7 | 5 | `profile_packup` |
| 3 analysis + workset | `analyze-demo` | 6 | 6 | `analyze_packup` |
| 4 kernel optimisation | `kernel-opt-demo` | 2 | 3 | `kernel_optimization` |
| 5 integration | `integration-demo` | 10 | 8 | `integration_packup` |

## Stage 1 — `deploy-demo`

| handoff | content_type / shape | validators |
|---|---|---|
| **`deploy_kit`**<br>one packup-shaped runnable kit | `code`, `items/codes/<model>.packup_YYYYMMDD/`<br>• `README.md` / `REPRODUCE.md` / `environment.md` / `notes.md`<br>• `scripts/` `results/` `logs/` `patches/`<br>• measured: 38 files, 226K | **`check_deploy_kit`** — completeness / **strong** / static / seconds<br>• content-line floors on the four documents (5/8/8/3)<br>• `REPRODUCE.md` ≥5 commands **and** an `Expected output` section<br>• `results/` holds ≥2 `.json`<br>• the served model name must not be a filesystem path (a measured fault)<br>• `environment.md` must name GPU arch, image, model<br>• **mode read-back from two independent components**: the worker's own log line and the router's worker listing<br>• **completion evidence**: one file carrying both `finish_reason: stop` and non-empty `content`<br>**`check_deploy_reproduces`** — usability / **weak** / dynamic / gpu_hours<br>• a fresh Claude Code session follows `REPRODUCE.md` and actually brings the model up<br>• `timeout_seconds` default 5400; `claude_cli` must come from a site variable |

## Stage 2 — `profiling-demo`

| handoff | content_type / shape | validators |
|---|---|---|
| **`deployment_baseline`**<br>graphs-on deployment | `reproducible`, the five standard items<br>• `result/`: `health.txt` `models.json` `smoke.txt` `workers.json`<br>• `env/`: `deployment.json` `engine_argv.txt` `gpu.txt` `image.txt` `rocm.txt` `router_cmd.txt`<br>• `logs/`: `mix_up` `router.tail` `worker.tail` (gz) | **`check_service_live`** — completeness / strong / seconds<br>• `expect_workers: 1`, `expect_disagg_mode: mixed`<br>• the arithmetic answer must be `391`<br>• engine log free of `memory access fault` / `HIP error` / `CUDA error` / `Traceback` |
| **`deployment_profiled`**<br>graphs-off + profiler control plane | as above | `check_service_live` |
| **`aiperf_baseline`**<br>Mooncake replay, no profiler | `reproducible`<br>• `result/`: `profile_export.jsonl.gz` `profile_export_aiperf.{csv,json}` `server_metrics_export.csv` `summary.json`<br>• `logs/`: `aiperf` `replay` `replay_console` (gz) | **`check_aiperf_report`** — completeness / strong / seconds<br>• required metrics `request_count` `output_token_throughput_tps` `ttft_ms` `input/output_sequence_length`<br>• `min_requests: 1`; error rate ≤ `${max_error_rate:-0.05}` |
| **`aiperf_profiled`**<br>same replay under the profiler | as above | `check_aiperf_report` |
| **`torch_trace`**<br>one measurement window per rank + a short with_stack window | `reproducible`, **360M — the whole cheat directory's size is this one item**<br>• `result/traces/` `result/stacks/` + two manifests | **`check_trace_coverage`** — completeness / strong / **minutes**<br>• ranks = `${tp:-8}`, stack ranks = `${stack_ranks:-2}`<br>• ≥1000 GPU kernels per rank; ≥10000 Python frames per stack rank<br>• window ≥1s, span ratio ≤4.0 |
| **`kernel_table`**<br>Magpie gap-analysis ranking | `reproducible`<br>• `result/`: `gap_analysis` `launchers.json` `text.json` `top_kernels.json`<br>• measured: **124 kernels** | **`check_kernel_table`** — usability / strong / seconds<br>• columns `Name` `Calls` `Self CUDA total (us)` `Avg time (us)` `% Total` `Input Shapes`<br>• ≥20 rows; `% Total` sums into [80.0, 100.5]<br>• ≥5 launchers in the top 25; head share ≥50% |
| **`profile_packup`**<br>terminal artefact | `code`, packup layout under `items/codes/` | **`check_packup_shape`** — completeness / strong / seconds<br>• required dirs `scripts` `results` `logs`; document floors 10/15/10/5<br>• `results/` ≥4 files; `REPRODUCE.md` ≥5 commands |

## Stage 3 — `analyze-demo`

| handoff | content_type / shape | validators |
|---|---|---|
| **`kernel_table`**<br>entry seed | `structured_text`, `items/{text.json, gap_analysis/gap_analysis.csv}`<br>• measured: a **synthetic seed, 34 rows** | **`check_kernel_table`** — usability / strong / seconds<br>• ≥20 rows; `% Total` sums into [50.0, 100.5]<br>(same validator name as stage 2, looser rules) |
| **`kernel_worklist`**<br>classified and ranked candidates | `structured_text`, `items/{text.json, schema, worklist.csv}` | **`check_worklist_shape`** — completeness / strong / seconds<br>• every kernel classified, `unknown` ratio ≤0.3<br>• every exclusion carries a reason; selection ordered and ≥1 |
| **`operator_identity`**<br>which repo/file/entry function each operator is | `structured_text`, `items/{text.json, schema, resolution_log.txt}` | **`check_identity_resolved`** — **trustworthiness** / strong / **dynamic** / minutes<br>• each selected operator carries a checked classification<br>• resolved paths must exist where claimed<br>• `require_actionable: true` |
| **`operator_workset`**<br>per-operator material for KernelForge | `reproducible` (six items: `result/env/script/code/logs/watchout`)<br>• `code/<operator_id>/` one directory per operator<br>• measured: 2 operators, 27 files | **`check_workset_shape`** — completeness / strong / seconds<br>• 10 required files incl. `invocation_spec.json` `forge_task.yaml` `program.md` `run_forge.sh` `reference/naive_torch.py` `scripts/{task_runner,forge_driver}.py` `tests/cases.json` `provenance.json`<br>• the driver must mention `--bench-mode` `--profile-run` `SNR`<br>• ≥3 cases; the two exported formats must agree |
| **`workset_evidence`**<br>measured correctness and performance | `structured_text`, `items/{text.json, schema, logs/<operator>.txt}` | **`check_workset_runs`** — **trustworthiness** / strong / **gpu_hours**<br>• every driver actually ran on the target GPU and met the SNR gate<br>• ≥5 groups × ≥10 iters; RSD ≤0.1; pass ratio ≥0.5 |
| **`analyze_packup`**<br>terminal artefact | `reproducible`, **items_schema declares 10 items** (`result/env/command/code/logs/watchout` + `results/` `REPRODUCE.md` `environment.md` `notes.md`)<br>— this is the real defect fixed this round: it declared six under `additionalProperties: false` while `packup.py` wrote ten | **`check_analyze_packup_shape`** — completeness / strong / seconds<br>• required `README.md` `REPRODUCE.md` `environment.md` `notes.md` + `results/` `logs/`<br>• ≥400 bytes each<br>• forbidden placeholders: `TBD` / `to be filled in` / `TODO: write` / `lorem ipsum` |

## Stage 4 — `kernel-opt-demo`

| handoff | content_type / shape | validators |
|---|---|---|
| **`workset`**<br>one operator's testable material | `code`, `items/codes/<operator_id>/`<br>• measured: one operator `sampler_vocab_softmax`, 10 files | **`check_workset_shape`** — completeness / strong / seconds<br>• required `README.md` `environment.md` `integration.md` `program.md` `baseline_measurement.md` `kernel/{driver,measure_baseline}.py`<br>• driver must contain `case_ms:` `case_snr:` `SNR:` `--bench-mode`<br>• ≥3 correctness cases<br>• **the baseline is cross-checked against a profile number** — the check the gfx942-baseline-in-a-gfx950-run fault slipped past |
| **`kernel_optimization`**<br>optimised kernel + evidence + report | `code`, `items/codes/<operator>.packup_YYYYMMDD/`<br>• measured: 21 files, 73K | **`check_optimization_shape`** — completeness / strong / seconds<br>• seven required evidence files: `results/{forge_result.json,optimization_report.md,optimized_kernel.py,verification.json}` + `scripts/kernel/{driver,graph_harness,measure_baseline}.py`<br>• document floors 5/8/8/3<br>**`check_speedup_substantiated`** — trustworthiness / **weak** / dynamic / minutes<br>• re-measures the claimed speedup on this machine<br>• `rounds:5` × `iters:30`, tolerance 0.15, noise floor 1.05<br>• **checks the ratio, never the denominator** — the standing example of an internal-consistency validator that cannot catch a wrong premise |

## Stage 5 — `integration-demo`

| handoff | content_type / shape | validators |
|---|---|---|
| **`kernel_patch`**<br>a patch set against a named image | `code`, `items/{codes/{manifest.json,patches/,notes},watchout}` | **`check_patch_shape`** — completeness / strong / seconds<br>• required fields `schema_version` `operator_id` `logical_operator` `image` `apply_mode` `files`<br>• `apply_mode` restricted to `overlay_files`<br>• paths must fall under one of four container roots (`/sgl-workspace/sglang/python/sglang` etc.) |
| **`patch_overlay`**<br>patch set → bind-mount plan | `reproducible`<br>• `result/{mounts.json, files/, patches/}`, `env/overlay.json` | **`check_overlay_applies`** — completeness / strong / seconds<br>• each mount's patched file must **differ** from stock<br>• hashes match what was recorded<br>• `compile_python: true` — it actually compiles |
| **`deployment_stock`**<br>the control arm, nothing mounted | `reproducible`; over stage 2 it adds `env/container_hashes.tsv` `env/docker_mounts.json` `env/marker_hits.tsv` | **`check_service_live`** — completeness / strong / seconds<br>• as stage 2, **plus** `expect_cuda_graph: true` |
| **`deployment_patched`**<br>the experimental arm | as above | `check_service_live` +<br>**`check_patch_live`** — **trustworthiness** / strong / seconds<br>• the running container's file hashes equal the mount plan's<br>• all mounts present<br>• the patch's declared markers **actually fired** |
| **`acceptance_stock` / `acceptance_patched`**<br>smoke + needle + lm-eval | `reproducible`<br>• `result/{smoke.json, needle.json, probe.json, lm_eval/}`<br>• four gz logs | **`check_acceptance`** — completeness / strong / minutes<br>• smoke must cover `arithmetic` `long_generation` `workers` `engine_log`<br>• needle: ≥1 depth retrieved, token ratio ≥0.95<br>• ≥20 scored samples per eval |
| **`bench_stock` / `bench_patched`**<br>two Mooncake replays each, cold and warm | `reproducible`, `result/r1/` (measured: only r1) | **`check_bench_report`** — completeness / strong / seconds<br>• rounds = `${bench_rounds:-2}`<br>• ≥`${min_requests:-50}` requests per round, error rate ≤5%<br>• metric set as `check_aiperf_report` |
| **`integration_report`**<br>the two arms side by side + the argument | `structured_text`, `items/{text.json, schema, report.md}` | **`check_no_regression`** — usability / **strong** / seconds<br>• **recomputes** every comparison from the raw numbers and must agree with the stated verdict<br>• throughput regression ≤`${max_throughput_regression:-0.05}`, TTFT ≤`${max_ttft_regression:-0.10}`<br>• eval confidence 0.95; both `stock` and `patched` arms required<br>⚠ the cheat sample carries `result: false` — the refusal was correct, the two arms were measured under different node load |
| **`integration_packup`**<br>terminal artefact | `code`, `items/{codes,watchout}` | **`check_packup_shape`** — completeness / strong / seconds<br>• stricter than stage 2: floors 20/15/12/8, ≥8 commands<br>⚠ absent from the cheat set — the graph stopped at the refusal above, so the packup was produced out of band |

## Three seams that will bite when the flow is stitched

1. **`kernel_table` is one name over two types.** Stage 2 seals it as
   `reproducible` (five items); stage 3 consumes a `structured_text`
   (`text.json` + CSV). Stage 2's artefact **cannot** be fed to stage 3's
   consumer as-is. Today only the per-stage roots keep them from colliding.
2. **"workset" spans two kinds.** Stage 3's `operator_workset` is
   `reproducible` and multi-operator; stage 4's `workset` is `code` and
   single-operator, and their required-file lists differ (10 items including
   `forge_task.yaml` vs 7 including `integration.md`). 3 → 4 needs an explicit
   conversion. Part II is the detailed comparison.

   **2026-09-05 — the merge happened in the definitions and not in the replay
   path, and three people walked into it in one hour.** M3.7 merged the two into
   a single `operator_workset` kind in `e2e-flow`. But
   `assets/build_workset.task/mock_adapt.py:69` still sources the replay from

   ```
   MOCK_ROOT / "stage4-kernel-opt/workset/content/items/codes/sampler_vocab_softmax"
   ```

   — **stage 4's copy, not `stage3-analyze/operator_workset`.** So a corpus
   graft into the stage-3 directory has no effect on what a mocked stage 3
   actually replays, and the task reports `succeeded` either way, because the
   mock did its job faithfully on the artefact it was pointed at.

   How it was found: the leader grafted a real 11:19:40 `operator_workset` into
   `stage3-analyze/`, m5's run then handed `apply_patch` a 54-file
   `sampler_vocab_softmax` scaffold with `target_files:
   ['torch/functional.py', 'python/sglang/srt/layers/sampler.py']`, and the
   corpus and the stage-4 artefact agreed with each other while the thing in
   between had replaced one of them. **`E2E_MOCK_ROOT` is honoured** — the root
   was right and the stage directory was wrong, which is why every check of the
   root came back clean.

   **Both directories exist in the corpus, so the wrong one is chosen by name
   and nothing complains.** Anyone grafting stage-3 material must either put it
   under `stage4-kernel-opt/workset/` or change that line; the seam is not
   closed until they are one path.
3. **`check_service_live` and `check_packup_shape` are each two validators
   sharing a name.** The two `check_service_live` differ by
   `expect_cuda_graph`; the two `check_packup_shape` differ by roughly 2× in
   their line floors. Resolution is per-file so nothing is actually wrong — but
   a reader assumes one validator where there are two.

Standing gap, recorded rather than fixed: `check_service_live` proves a
deployment is **live**, not that it is **comparable** to the other arm's. The
−21% false regression in stage 5 is what that gap costs.

---

# Part II — the two worksets, compared

## 0. What they are, and the structural comparison

| | **`operator_workset`** (stage 3) | **`workset`** (stage 4) |
|---|---|---|
| kind / content_type | `reproducible` (six items) | `code` (one item, `codes`) |
| where it comes from | produced for real by `build_workset`, upstream `kernel_worklist` + `operator_identity` | **package data shipped in the repo**; `publish_workset` only publishes it |
| operators covered | **many** — measured: `moe_gemm_mfma_moe1_silu_mul`, `moe_gemm_mfma_moe2_afp4_wfp4` | **one** — `sampler_vocab_softmax` |
| target | gfx950 / MI355X / AITER MoE fp4 | gfx942 / MI300X / sglang sampler softmax |
| consumer it is written for | KernelForge (`forge-loop`) | a general kernel-optimisation agent |

```
operator_workset/items/                      workset/items/
├── env          handoff-level environment    └── codes/
├── result       what was produced                └── sampler_vocab_softmax/
├── script       run every operator                   ├── README.md
├── watchout     what a consumer must not assume      ├── program.md
└── code/                                             ├── integration.md            *
    └── <operator_id>/                                ├── baseline_measurement.md   *
        ├── README.md                                 ├── environment.md            *
        ├── program.md                                └── kernel/
        ├── invocation_spec.json          *               ├── sampler_softmax_kernel.py *
        ├── forge_task.yaml               *               ├── driver.py
        ├── provenance.json               *               ├── measure_baseline.py
        ├── run_forge.sh                                  └── graph_harness.py      *
        ├── reference/naive_torch.py
        ├── scripts/{forge_driver,standalone_driver,task_runner}.py
        └── tests/cases.json              *
```
`*` = the other one has no equivalent.

**Which defines it better is not one question.**

| dimension | winner | why |
|---|---|---|
| machine-readability | **stage 3** | `invocation_spec.json` (schema_version 2) + `forge_task.yaml` + `provenance.json` + `tests/cases.json`. Stage 4 carries **no JSON and no YAML at all**; its cases are a `_CASES` tuple inside `driver.py` |
| scaling to many operators | **stage 3** | `code/<operator_id>/` is naturally plural; stage 4's shape is one directory |
| traceability back to the profile | **stage 3** | `provenance.json` carries `calls` / `self_us` / `avg_us` / `pct_total` / `input_shapes`, plus `why_selected` and `excluded_reason` |
| cross-checked double writing | **stage 3** | `check_workset_shape` re-derives task_id, primary shape, case count and source files from `invocation_spec.json` **and** from `forge_task.yaml` and compares them, so editing one without the other is caught |
| environment completeness | **stage 4** | `environment.md` gives OS, kernel, CPU, RAM, GPU SKU, driver version, CU count, partition mode, image digest, Python, PyTorch, Triton, ROCm, rocprofv3. Stage 3 gives gfx950 + an image tag; the ROCm version is only implied by the tag `rocm720` |
| integration point | **stage 4** | `integration.md` names `sampler.py:183` with the commit SHA and three load-bearing facts: the write is in-place, `dim=-1` is the contiguous dimension, and the result feeds `torch.multinomial` so rows must sum to 1 |
| credibility of the baseline | **stage 4** | `baseline_measurement.md` has the raw 5×30 table **and** a cross-check against the live trace: 55.40 µs measured standalone vs 55.59 µs in the trace — 0.3% |
| whether anything was actually run | **stage 4** | stage 3's own `items/env` says: *"No GPU, no ROCm runtime and no torch were reachable from the build step… until `verify_workset` has run, no number in this handoff is a measured number."* |

**In one line: stage 3 has the better schema, stage 4 has the more substantial
content.**

## 1. Against the mission's nine requirements

Mission §1.3.2 lists nine things a workset must carry.

| # | requirement | stage 3 `operator_workset` | stage 4 `workset` |
|---|---|---|---|
| 1 | environment: image / GPU / rocm / other software versions | **partial** — image + gfx950 + AITER commit; no explicit rocm/torch/triton versions, only "the image carries them" | **met** — every version explicit, plus the image digest |
| 2 | sglang integration-point reference and explanation | **not met** — `invocation_spec` points at a kernel source file inside **AITER**, not at an sglang integration point | **met** — `integration.md` |
| 3 | the operator source itself, carved out | **not met** — `items/env` states plainly *"nothing in this handoff is a copy of kernel source"*; only paths and line numbers | **met** — `sampler_softmax_kernel.py` |
| 4 | a PyTorch naive implementation | **met, by reference** — `reference/naive_torch.py` imports `aiter.fused_moe.torch_moe_stage1` and argues the case: a reference written by reading the kernel tends to agree with the kernel's bugs | **met** — `driver.py:_reference`, a float64 oracle |
| 5 | one-command correctness and performance runs | **met** — `run_forge.sh` + `standalone_driver.py`; `items/script` runs every operator | **met** — `driver.py`, three modes |
| 6 | ≥3 correctness cases | **met** — `tests/cases.json`, validator `min_cases: 3` | **met** — B1/B8/B32, validator `min_correctness_cases: 3` |
| 7 | 5 weighted rounds, each averaging ≥10 iterations | **met, but in a different handoff** — split into `workset_evidence`, gated by `check_workset_runs` (`min_groups:5`, `min_iters_per_group:10`) | **met, in band** — `measure_baseline.py`, 5 rounds × 30 iters, **each round a fresh process**, median rather than mean (one outlier had already read 21.67 µs as a regression where the median was 18.9 µs) |
| 8 | the operator's performance as captured in the profile | **met** — `provenance.json` in full | **met** — `program.md`'s ranking table plus the 0.3% cross-check |
| 9 | an operator definition format (cf. SIKL) | **partial** — a custom schema exists, but it is not SIKL-shaped | **not met** — no machine-readable definition at all |

**Score: stage 3 = 4.5 / 9, stage 4 = 7 / 9.**

Read requirement by requirement, **stage 4 complies better**. The gap is
requirements 2 and 3, which the mission states bluntly, and where stage 3 chose
"give the coordinates, not the copy". That choice is defensible — the source
lives in the image and forge-loop edits it in its own git worktree — but it is
**not what the mission asked for**. Conversely, requirement 9 is the only one
naming an external standard, and only stage 3 comes near it.

## 2. Similarity to flashinfer-bench

The FlashInfer Trace Schema, from the project's own docs
(`docs/flashinfer-trace/{definition,workload}.md`):

- **Definition** — `name` / `op_type` / `axes` (each `{type: var|const, value}`)
  / `inputs` / `outputs` (`{shape, dtype}`) / `reference` (source as a string) /
  `tags` / `constraints` / `description`
- **Workload** — JSONL, one Trace object per line:
  `{definition, workload: {uuid, axes, inputs}, solution, evaluation}`, the last
  two `null` when standalone. Workloads sharing a definition live in one JSONL.

| | similarity | why |
|---|---|---|
| stage 3 `operator_workset` | **medium** | same idea — a declarative JSON describing one operator plus a referenced reference implementation; `invocation_spec.json` has `schema_version` / `logical_operator` / `description`, and `tests/cases.json` is semantically a workload set. But **none of the field names, the var/const axis model, the tensor specs, or the inlined-source `reference` correspond**. What stage 3 records — which repo, which file, which line the operator lives at — has no counterpart in a Definition at all |
| stage 4 `workset` | **low** | no declarative schema anywhere. The only correspondence is that `driver.py:_reference` plays Definition's `reference` and `_CASES` plays a workload's axis binding — but both are Python constants, not data |

**Stage 3 is closer, and neither is flashinfer-shaped.** Stage 3 is another
dialect of the same family; stage 4 does not speak the language.

## 3. Similarity to `rank0/`

`rank0/` **is** the flashinfer-bench trace schema, essentially verbatim:

```
rank0/definitions/{gemm,moe}/<name>.json   Definition
rank0/workloads/{gemm,moe}/<name>.jsonl    Workload, one Trace per line:
                                           {definition, workload:{axes,inputs,uuid},
                                            solution:null, evaluation:null}
```

`definitions/gemm/aiter_gemm_a16w16_nt_n128_k6144.json` carries `name` /
`op_type` / `axes` (`m` var, `n`/`k` const) / `inputs` / `outputs` /
`reference` / `tags` / `description` — matching the published Definition, with
**one field the upstream schema does not have: `baseline`**, inlining
`aiter.tuned_gemm.tgemm.mm` as the performance ground truth.

| | similarity to `rank0` | detail |
|---|---|---|
| stage 3 `operator_workset` | **medium-low** | same per-operator directories, same JSON declaration, same separation of a naive reference from a tuned baseline. **And it points at the same operators** — `rank0/definitions/moe/aiter_fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128.json` and stage 3's `moe_gemm_mfma_moe1_silu_mul` are the same AITER fused-MoE path. But stage 3 records shapes as a flat string plus a prose `dims_note`, where `rank0` uses named axes and a per-tensor `{shape: ["num_tokens","model_dim"], dtype: "float4_e2m1"}`. **The same thing, once as prose and once as schema** |
| stage 4 `workset` | **low** | one line in `README.md` — `Signature | sampler_softmax(logits: Tensor[B, V] fp32, out: ...)`. `B` and `V` are exactly `rank0`'s var/const axes, in no machine-readable form |

**Stage 3 is closer, and closer in the useful sense**: it already has
`invocation_spec.json`. Promoting its shape information from the `input_shapes`
string plus `dims_note` into `axes` / `inputs` / `outputs`, and inlining
`reference/naive_torch.py` into a `reference` field, lands it on `rank0`'s
shape. Stage 4 would first have to invent a schema.

## Summary

| question | answer |
|---|---|
| better and more complete definition | **split** — schema and traceability to stage 3; environment, integration point and measured baseline to stage 4 |
| complies better with the mission's nine | **stage 4** (7/9 vs 4.5/9) |
| closer to flashinfer-bench | **stage 3** (medium vs low) |
| closer to `rank0` | **stage 3** (medium-low vs low, with a clear conversion path) |

**One thing outside the questions but visible from the answers:** the mission's
requirement 9 names **SIKL**, while `rank0` uses the **flashinfer-bench** trace
schema. They are not the same format. If `rank0` is the target to converge on,
that mission line needs updating — flagged, not acted on.

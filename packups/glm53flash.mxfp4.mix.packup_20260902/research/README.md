# Research artifacts

- **`GLM-5.3-Flash-MXFP4.README.md`** — the vendor (OneNexus) model card, as
  shipped in the checkpoint. It is the source of the recipe every run here
  started from, and it is also the source of the one open question: its
  published launch command carries **no** `--disable-shared-experts-fusion` yet
  reports a successful 4×MI350 validation. See `../results/root_cause.md`
  ("Still open") for the check that would settle it.

- **`p1_test.py`** — probe P1 from `../PLAN.md` §4: call `should_ignore_layer`
  with the real `FusedMoE` prefix (`model.layers.3.mlp.experts`) against the
  checkpoint's `exclude` list, with and without `fused_mapping`. It was written
  to settle whether per-expert exclude entries can ever match at the granularity
  the loader asks at. It became moot once the ladder found the actual cause; the
  exclude-list hypothesis it was probing is the wrong turn recorded in
  `../notes.md` §2(a).

- **`aiter-image-delta/`** — probe P3: the aiter FP4 commit lists inside the two
  candidate base images (`rocm720` = ours, `rocm724` = the vendor's). The two do
  differ, which is why the ladder's rung 1 existed. Rung 1 then **passed**, so
  the base image is not a variable for this result. The larger `fused_moe.py` /
  `gemm_op_a4w4.py` copies from both images are left in the workspace at
  `/apps/yihou/glm53.series.workspace_20260901/research/flash-debug/` and are
  not duplicated here.

The broader background research — InferenceX scrapes, the GLM-5.2 comparison
material, the big-model card — stays in the workspace at
`/apps/yihou/glm53.series.workspace_20260901/research/glm53-research/`. None of
it is needed to reproduce this result.

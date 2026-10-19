# main — optimise one kernel with KernelForge

This task runs nothing itself. `publish_inputs` publishes a deploy kit and a
one-operator workset from `assets/fixture/`; `m4_kernel_opt` (e2e-flow's stage 4)
hands the workset's operator to KernelForge, runs the workset's own tests on the
result, and publishes `kernel_optimization` — the candidate kernel, its
correctness verdict and its measured speed-up.

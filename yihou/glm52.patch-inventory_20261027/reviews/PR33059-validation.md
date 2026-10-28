# PR #33059 validation

Run ID: `pr33059-cpu-20261027`.
Base: `449617bc94a5d4ae23ad62d0da2f842e711ca983`.
Head: `d0062f8bce0c86de998ff36b8c106e2ab4bc6ca5`.

Python 3.13.13; torch 2.14.1+cpu; torchvision 0.29.1+cpu; transformers 5.19.0; compressed-tensors 0.18.0; apache-tvm-ffi 0.1.11. Only dependencies needed for these CPU imports were installed; this is not a complete GPU serving environment.

- Patched method: 12 tests passed (`runtime/pr33059-test.log`).
- Unpatched main with identical tests: 10 expected failures, 2 unpadded controls passed (`runtime/pr33059-baseline.log`). Source was restored in a finally block.
- All applicable repository lint hooks passed (`runtime/pr33059-lint.log`).
- Full diff and PR body: `PR33059-rebased.diff`, `PR33059-pr-body.md`.

The regression runs the real indexer dispatch method with CPU tensors and stubbed Aiter/top-k boundaries. It checks sliced row values, metadata dimensions, restored -1 padding, dtype/shape, and empty-query launch avoidance. It is not GPU numerical or distributed-model validation.

Current Aiter source `7559e3411beecb04d2f64ac488d6990a3b03534e` divides by the query count during SplitKV scheduling. This source review motivated the explicit empty-query return; no hardware reproduction of that boundary is claimed.

Outstanding reviewer request: MI300 and MI355 DeepSeek-V3.2 DP-attention accuracy plus the stage-c 8-GPU suite and complete upstream CI. Historical v0.5.15 deployment reports do not satisfy it. Related #32175 changes CP metadata padding and needs contract coordination if combined; it is not the same narrow current-main caller change. #34455 was closed unmerged.

Final CI snapshot (2026-10-27 08:29 UTC): upstream lint passed (run 40473090386); title check passed after renaming (run 40473491609). Existing run-ci started base/AMD CI. Intel SPR job 122084833207 failed during Docker build with no space left on device, before test execution. Other suites remain in progress. All unfulfilled local GPU requests were released; no local GPU/model test ran.

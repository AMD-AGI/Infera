## Description

Add opt-in `MC_HIP_STREAM_PRIORITY=high` for HIP transfer streams. A transfer stream at the normal priority can wait behind compute work; high priority requests the device's greatest supported stream priority while preserving nonblocking semantics. Unset/`default` retains today's behavior, and invalid values warn and use the default. This is a scheduling hint, not a guarantee of preemption.

Rebased the yihou M02 patch onto Mooncake main `72e267f4` on 2026-11-02. The original HIP validation below was performed at `13df6a68`; the stream-pool implementation and its test source are unchanged by this rebase. Historical isolated peer-copy experiments R039/R043 reported a roughly 200 ms to 11 ms stall reduction behind long kernels. The prepared priority image was never used for an end-to-end serving run, and that historical number is not a measurement of this new revision. No simulated-acceptance or combined-stack throughput is claimed.

#4953 changes HIP asynchronous batch submission; it does not add this stream-priority selection. #4340 is an independent IPC correctness fix owned by its author, tracked separately.

## Suggested Merge Priority

- [x] P3 — Optional performance enhancement requiring workload qualification.

## Module

- [x] Transfer Engine (`mooncake-transfer-engine`)

## Type of Change

- [x] Performance improvement
- [x] Documentation update

## How Has This Been Tested?

On an allocated MI355X with ROCm 7.2:

```bash
cmake -S . -B /data/yihou/glm52-upstream-20261027/mooncake-build \
  -DUSE_HIP=ON -DUSE_CUDA=OFF -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
  -DBUILD_UNIT_TESTS=ON -DBUILD_EXAMPLES=OFF
cmake --build /data/yihou/glm52-upstream-20261027/mooncake-build --target hip_stream_pool_test -j 8
ctest --test-dir /data/yihou/glm52-upstream-20261027/mooncake-build \
  -R hip_stream_pool_ --output-on-failure
```

**4/4 registered CTest cases pass**: default/high/invalid/unset, each with priority/flags/round-robin/byte-copy and zero-count tests. Direct compilation with `hipcc -std=c++17 -Wall -Wextra -Werror` also passes. With the unmodified main stream-pool implementation, the high-priority test fails (observed priority 0, expected -1), while the zero-count control passes.

All applicable PR-scoped pre-commit hooks pass with clang-format 20. `make -C docs html` succeeds and the new environment option appears in the generated page. Initial full configure required initializing the existing `extern/pybind11` submodule; configure/build/CTest then succeeded.

- [x] Unit tests pass
- [ ] End-to-end peer-copy/serving tests under concurrent compute

## Checklist

- [x] Code/diff reviewed by the preparing agent
- [ ] Human submitter's line-by-line review completed
- [x] C/C++ formatting and PR-scoped pre-commit pass
- [x] Documentation updated and built
- [x] Regression tests added

## AI Assistance Disclosure

- [x] AI tools were used: Codex assisted with rebasing, tests, validation and this PR text. No assistant or co-author commit trailers were added.

## Follow-up, 2026-11-02

The old Python 3.10 wheel failure was `ModuleNotFoundError: mooncake._structured_ranges`, introduced in the base by #3954 and fixed by merged #5030. This branch now includes that fix. The packaging source regression `python -m pytest -q python/tests/packaging/test_project.py::test_structured_object_storage_has_one_authoritative_source` passes (1 test), and all applicable PR-scoped pre-commit hooks pass. Upstream CI is rerunning; a complete wheel rebuild was not performed locally.

The current #4953 revision (`35c488e9`) still acquires streams through `stream_pool_.getNextStream(device_id)` and does not change the stream pool's priority creation. Source review finds no competing priority implementation; combined runtime qualification is not claimed. #4953 is coordination work, not an established mandatory merge dependency for this opt-in.

## Draft blockers

1. Human submitter review required by this repository's `AGENTS.md` remains pending. The preparing agent cannot certify that review on the submitter's behalf.
2. Maintainer CI and review after this rebase. Copy/compute-interference measurements are workload qualification before enabling the option for a serving workload; they are not a blanket prerequisite for reviewing this default-preserving option. Historical microbenchmark claims remain historical and no production speedup is claimed.


New-head CI update: the CUDA CTest build now fails before tests in `on_cuda_transport_guard_test` because `mooncake_te_build_config.h` is not on its include path ([job](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125092088899)). The target/test/header are unchanged from main in this PR. Existing #5144 fixes this distinct base regression; it is open and tracked as the current build blocker. This is separate from the old wheel issue fixed by #5030.

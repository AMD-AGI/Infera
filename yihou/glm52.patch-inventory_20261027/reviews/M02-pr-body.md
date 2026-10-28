## Description

Add opt-in `MC_HIP_STREAM_PRIORITY=high` for HIP transfer streams. A transfer stream at the normal priority can wait behind compute work; high priority requests the device's greatest supported stream priority while preserving nonblocking semantics. Unset/`default` retains today's behavior, and invalid values warn and use the default. This is a scheduling hint, not a guarantee of preemption.

Rebased the yihou M02 patch onto Mooncake main `81b3694b1bcde8af454457f835734f1306420fb3`. Historical isolated peer-copy experiments R039/R043 reported a roughly 200 ms to 11 ms stall reduction behind long kernels. The prepared priority image was never used for an end-to-end serving run, and that historical number is not a measurement of this new revision. No simulated-acceptance or combined-stack throughput is claimed.

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

## Draft blockers

1. Human submitter review required by this repository's `AGENTS.md` is still pending; this draft does not claim that review occurred.
2. Fresh peer-transfer latency plus compute interference/throughput measurements on the final revision, under representative load; verify cross-device behavior and default-off control. The isolated tests confirm configuration and copy correctness, not a production speedup.
3. Maintainer CI/review and coordination with the stream/event lifecycle changes in #4953.

The current SGLang image build can report that a valid Mooncake library lacks dma-buf support. Under `set -o pipefail`, `grep -q` exits as soon as it finds the symbol; a large `printf` producer can then receive SIGPIPE, making the successful capability check fail. The `ldd | grep -qi` check has the same early-close pattern.

Use Bash substring checks on the captured symbol and linked-library output instead. Preserve the case-insensitive library check and all missing-capability failures. This targets the current v0.5.22 build script; it does not restore old v0.5.21 recipes or change example configurations.

Regression: execute only the real script's artifact-check section with controlled `nm`, `ldd`, and `strings` output, including a capability match followed by 2.3 MB of output. On unpatched main: **3 failed, 1 passed** (even the valid artifact is rejected). With this change: **4 passed**; absent symbol/library/routing markers still fail with the correct message. No historical serving throughput is used as evidence.

```bash
python -m pytest -q tests/unit/deploy/test_mooncake_artifact_checks.py
bash -n deploy/docker/scripts/build_mooncake_sglang.sh
pre-commit run --files deploy/docker/scripts/build_mooncake_sglang.sh tests/unit/deploy/test_mooncake_artifact_checks.py
```

All applicable local lint and syntax checks passed.

Draft blocker: the complete current image build has not been rerun. The isolated regression confirms artifact-check behavior; image-build CI and maintainer review are still needed.

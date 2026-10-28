The current SGLang image build can reject a valid Mooncake library as missing dma-buf support. Under `set -o pipefail`, `grep -q` exits after finding the symbol; a large `printf` producer can then receive SIGPIPE, making the successful capability check fail. The `ldd | grep -qi` check has the same early-close pattern.

Use Bash substring checks on the captured symbol and linked-library output. Preserve case-insensitive library matching and rejection of missing capabilities. The production change is limited to post-build artifact inspection in the current v0.5.22 script; it changes no compilation flags, dependency pins or runtime behavior.

## Validation

**11 CPU regression cases pass; unpatched main fails 9 and passes the 2 missing-symbol controls.** Shell syntax and all applicable pre-commit hooks pass.

- Four cases feed large controlled command output through the production checks, including a valid capability followed by 2.3 MB of output.
- Seven cases compile ELF fixtures with 60,000 exported padding symbols and execute the production artifact-check block with real `nm`, `ldd`, and `strings`, without mocking those commands. The valid artifact passes, including case-insensitive HSA-library matching. Missing dma-buf symbol, HSA library, routing marker, destination-rail marker, timeout marker, and optional batch-registration marker each fail with the expected diagnostic.
- The ELF fixtures are CPU test libraries, not Mooncake or a functioning HSA runtime. These tests exercise artifact inspection without accessing a GPU. The ELF cases require Linux, a C compiler and binutils, and explicitly skip when those tools are unavailable.

```bash
python -m pytest -q tests/unit/deploy/test_mooncake_artifact_checks.py
bash -n deploy/docker/scripts/build_mooncake_sglang.sh
pre-commit run --files deploy/docker/scripts/build_mooncake_sglang.sh tests/unit/deploy/test_mooncake_artifact_checks.py
```

## Review scope and integration limits

Ready for review based on reproducible before/after failures and coverage of the affected artifact checks using real binary-inspection commands. The complete image has **not** been rebuilt, and no Mooncake runtime, GPU or serving-throughput result is claimed. Full-image integration remains unverified; the focused shell fix can be reviewed independently of that broader build.

The original implementation revision passed all executed upstream checks (DCO, lint, unit, agent_sys, rust, unit-torch-cpu and documentation; engine/e2e were skipped). The added ELF regression is being submitted to the same CI. Required upstream checks and maintainer review remain merge requirements.

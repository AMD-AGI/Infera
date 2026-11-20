#!/usr/bin/env bash
# The U17 PR's new test run against main's code (wt/main2 synced into the nightly image, tests/ from the PR tree), to
# show it fails without the fix: both cases of test_fused_shared_expert_rows_owned_once_after_ep_dispatch fail on main
# (no localize_fused_shared_expert_ids / aiter_topK_ep_shared_cols there).
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u17-aiter-fused-shared-ep-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main2 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly || exit 1
b=$(wt main2)
w=$(wt u17-aiter-fused-shared-ep-main)
mkdir -p "$NL/pip"
docker rm -f up-nb-u17 > /dev/null 2>&1
# shellcheck disable=SC2046
docker run --rm --name up-nb-u17 $(gpu_args) -v "$b:/src:ro" -v "$w/tests:/work/tests:ro" \
    -v "$w/pyproject.toml:/work/pyproject.toml:ro" -v "$U/checks:/checks:ro" -v "$OUT:/out" \
    -v "$NL/pip:/root/.cache/pip" -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_ADDOPTS="-p no:cacheprovider" \
    -w /work --entrypoint bash "$IMG_NIGHTLY" -c "
pip install -q $PIP_TEST > /out/pip-u17.log 2>&1
python3 /checks/sync_vllm.py /src > /dev/null
python3 -m pytest -v -rfEs --junitxml=/out/u17.xml \
    tests/kernels/moe/test_rocm_aiter_topk.py::test_fused_shared_expert_rows_owned_once_after_ep_dispatch \
    > /out/u17.out 2>&1
tail -25 /out/u17.out"
python3 - "$OUT" <<'EOF' | tee "$OUT/verdict.txt"
import sys
import xml.etree.ElementTree as ET
out = sys.argv[1]
want = {
    "test_fused_shared_expert_rows_owned_once_after_ep_dispatch[2]",
    "test_fused_shared_expert_rows_owned_once_after_ep_dispatch[4]",
}
try:
    root = ET.parse(f"{out}/u17.xml").getroot()
except Exception as e:
    print(f"u17: no junit xml ({e})")
    sys.exit(1)
failed, passed = set(), set()
for tc in root.iter("testcase"):
    bad = tc.find("failure") is not None or tc.find("error") is not None
    (failed if bad else passed).add(tc.get("name"))
print(f"u17: failed on main {sorted(failed)}; passed on main {sorted(passed)}")
ok = failed == want and not passed
print("verdict:", "PASS (new test fails on main as expected)" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
exit "${PIPESTATUS[0]}"

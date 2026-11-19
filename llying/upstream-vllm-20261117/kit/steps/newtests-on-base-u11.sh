#!/usr/bin/env bash
# The U11 PR's new tests run against main's code (wt/main2 synced into the nightly image, tests/ from the PR tree), to
# show they fail without the fix. Expected to fail on main: test_kv_cache_configs_record_largest_worker_bytes_per_block
# (main has no max_worker_bytes_per_block) and test_worker_slots_sized_by_largest_pipeline_stage[24-24] (main ignores
# the engine-wide value); the [None-16] and [8-16] cases pass on main too.
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u11-offload-pp-worker-bytes-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main2 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly || exit 1
b=$(wt main2)
w=$(wt u11-offload-pp-worker-bytes-main)
mkdir -p "$NL/pip"
docker rm -f up-nb-u11 > /dev/null 2>&1
# shellcheck disable=SC2046
docker run --rm --name up-nb-u11 $(gpu_args) -v "$b:/src:ro" -v "$w/tests:/work/tests:ro" \
    -v "$w/pyproject.toml:/work/pyproject.toml:ro" -v "$U/checks:/checks:ro" -v "$OUT:/out" \
    -v "$NL/pip:/root/.cache/pip" -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_ADDOPTS="-p no:cacheprovider" \
    -w /work --entrypoint bash "$IMG_NIGHTLY" -c "
pip install -q $PIP_TEST > /out/pip-u11.log 2>&1
python3 /checks/sync_vllm.py /src > /dev/null
python3 -m pytest -v -rfEs --junitxml=/out/u11.xml \
    tests/v1/core/test_kv_cache_utils.py::test_kv_cache_configs_record_largest_worker_bytes_per_block \
    tests/v1/kv_connector/unit/offloading_connector/test_config.py::test_worker_slots_sized_by_largest_pipeline_stage \
    > /out/u11.out 2>&1
tail -25 /out/u11.out"
python3 - "$OUT" <<'EOF' | tee "$OUT/verdict.txt"
import sys
import xml.etree.ElementTree as ET
out = sys.argv[1]
want = {
    "test_kv_cache_configs_record_largest_worker_bytes_per_block",
    "test_worker_slots_sized_by_largest_pipeline_stage[24-24]",
}
try:
    root = ET.parse(f"{out}/u11.xml").getroot()
except Exception as e:
    print(f"u11: no junit xml ({e})")
    sys.exit(1)
failed, passed = set(), set()
for tc in root.iter("testcase"):
    bad = tc.find("failure") is not None or tc.find("error") is not None
    (failed if bad else passed).add(tc.get("name"))
print(f"u11: failed on main {sorted(failed)}; passed on main {sorted(passed)}")
ok = failed == want and len(passed) == 2
print("verdict:", "PASS (new tests fail on main exactly where expected)" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
exit "${PIPESTATUS[0]}"

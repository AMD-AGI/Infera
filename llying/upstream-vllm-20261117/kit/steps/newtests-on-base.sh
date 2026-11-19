#!/usr/bin/env bash
# The U13 / U19 PRs' new tests run against main's code (wt/main2 synced into the nightly image, tests/ from the PR
# tree), to show they fail without the fix (pr-checklist 2.1). Expected: U13 test_aiter_ag_rs_gate[NONE-True-None-True]
# (FULL capture) and [FULL-False-None-True] (main also requires a uniform batch descriptor, which the PR drops) fail,
# its other cases pass; U19 test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[2|4] fail.
# The step passes when exactly those fail; per-branch output in $OUT/<branch>.out.
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u13-aiter-agrs-full-capture-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u19-rocm-pin-worker-slots-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main2 rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly || exit 1
b=$(wt main2)
mkdir -p "$NL/pip"
run_on_base() {  # NAME PR_BRANCH PIP_EXTRA PYTEST_ARGS...
    local name=$1 w
    w=$(wt "$2")
    local extra=$3
    shift 3
    docker rm -f "up-nb-$name" > /dev/null 2>&1
    # shellcheck disable=SC2046
    docker run --rm --name "up-nb-$name" $(gpu_args) -v "$b:/src:ro" -v "$w/tests:/work/tests:ro" \
        -v "$w/pyproject.toml:/work/pyproject.toml:ro" -v "$U/checks:/checks:ro" -v "$OUT:/out" \
        -v "$NL/pip:/root/.cache/pip" -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_ADDOPTS="-p no:cacheprovider" \
        -w /work --entrypoint bash "$IMG_NIGHTLY" -c "
pip install -q $PIP_TEST $extra > /out/pip-$name.log 2>&1
python3 /checks/sync_vllm.py /src > /dev/null
python3 -m pytest -v -rfEs --junitxml=/out/$name.xml $* > /out/$name.out 2>&1
tail -25 /out/$name.out"
}
run_on_base u13 u13-aiter-agrs-full-capture-main ray tests/distributed/test_comm_ops.py -k aiter_ag_rs_gate
t=tests/v1/kv_offload/cpu/test_shared_offload_region.py
run_on_base u19 u19-rocm-pin-worker-slots-main "" "$t::test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved" \
    "$t::test_pin_mmap_region_selects_by_savings_not_row_count"
python3 - "$OUT" <<'EOF' | tee "$OUT/verdict.txt"
import sys
import xml.etree.ElementTree as ET
out = sys.argv[1]
expected = {
    "u13": {"test_aiter_ag_rs_gate[NONE-True-None-True]", "test_aiter_ag_rs_gate[FULL-False-None-True]"},
    "u19": {
        "test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[2]",
        "test_pin_mmap_region_rocm_registers_own_slots_whenever_a_page_is_saved[4]",
    },
}
ok = True
for name, want in expected.items():
    try:
        root = ET.parse(f"{out}/{name}.xml").getroot()
    except Exception as e:
        print(f"{name}: no junit xml ({e})")
        ok = False
        continue
    failed, passed = set(), set()
    for tc in root.iter("testcase"):
        bad = tc.find("failure") is not None or tc.find("error") is not None
        (failed if bad else passed).add(tc.get("name"))
    print(f"{name}: failed on main {sorted(failed)}; passed on main {len(passed)}")
    ok &= failed == want and len(passed) > 0
print("verdict:", "PASS (new tests fail on main exactly where expected)" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
exit "${PIPESTATUS[0]}"

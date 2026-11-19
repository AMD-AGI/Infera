#!/usr/bin/env bash
# Infera Python CI for every PR branch and origin/main (as .github/workflows/ci.yml: Python 3.10, the dev / gaie /
# projection extras, pytest -m "not slow and not integration"). The dependencies come from main's pyproject; each
# tree runs from its own directory (PYTHONPATH), and a branch fails only on tests that main passes.
# key: for d in /home/liyingli/bench_agentx/upstream-20261119/iwt/*; do git -C $d rev-parse HEAD; done
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready py310 || exit 1
names="main vllm-mooncake-serial router-pool-idle server-nofile kv-aware-prepare vllm-dp-engine-lb vllm-034-arg-parser"
mounts=""
for n in $names; do mounts+="-v $U/iwt/$n:/src/$n:ro "; done
mkdir -p "$NL/pip"
docker rm -f up-infera-py > /dev/null 2>&1
# shellcheck disable=SC2086
docker run --rm --name up-infera-py --network host $mounts -v "$OUT:/out" -v "$NL/pip:/root/.cache/pip" \
    -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 -e PYTHONDONTWRITEBYTECODE=1 --entrypoint bash python:3.10-slim-bookworm -c "
set -u
apt-get update -qq > /dev/null && apt-get install -y -qq git > /dev/null 2>&1
cp -r /src/main /tmp/main && pip install -q '/tmp/main[dev,gaie,projection]' > /out/pip.log 2>&1 && pip uninstall -y -q amd-infera
for n in $names; do
    echo \"== \$n\"
    cd /src/\$n && PYTHONPATH=/src/\$n python -m pytest -p no:cacheprovider -m 'not slow and not integration' -q \
        --junitxml=/out/\$n.xml 2>&1 | tail -4
done" 2>&1 | tee "$OUT/run.txt"
rc=0
for n in $names; do
    [[ $n == main ]] && continue
    echo "== $n vs main"
    python3 "$U/checks/junit_diff.py" "$OUT/main.xml" "$OUT/$n.xml" | tail -6 || rc=1
done | tee "$OUT/verdict.txt"
! grep -q "verdict: FAIL\|missing or empty" "$OUT/verdict.txt"

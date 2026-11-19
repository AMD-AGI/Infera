#!/usr/bin/env bash
# vLLM's pre-commit hooks (ruff, typos, mypy, SPDX, torch.cuda calls, test tethering, ...) on the files of the U13 / U19
# PR commits, CPU only (plan GPUS 0). Each tree is copied into the container and committed to a scratch git repo, so
# the host's .git is never written by root. Hook environments are cached under $NL/precommit.
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u13-aiter-agrs-full-capture-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u19-rocm-pin-worker-slots-main rev-parse HEAD
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly || exit 1
mkdir -p "$NL/precommit" "$NL/pip"
rc=0
for br in u13-aiter-agrs-full-capture-main u19-rocm-pin-worker-slots-main; do
    w=$(wt "$br")
    files=$(git -C "$w" diff --name-only HEAD~1 HEAD | tr '\n' ' ')
    echo "== $br $(git -C "$w" rev-parse --short=9 HEAD): $files"
    docker rm -f "up-precommit-$br" > /dev/null 2>&1
    # shellcheck disable=SC2046
    docker run --rm --name "up-precommit-$br" $(gpu_args) -v "$w:/src:ro" -v "$OUT:/out" \
        -v "$NL/precommit:/root/.cache/pre-commit" -v "$NL/pip:/root/.cache/pip" --entrypoint bash "$IMG_NIGHTLY" -c "
command -v git > /dev/null || { apt-get update -qq && apt-get install -y -qq git > /dev/null; }
pip install -q pre-commit > /out/pip-precommit-$br.log 2>&1
mkdir /tmp/t && cp -a /src/. /tmp/t && rm -f /tmp/t/.git && cd /tmp/t
git config --global --add safe.directory '*'
git init -q && git add -A && git -c user.email=ci@local -c user.name=ci commit -q -m tree
pre-commit run --show-diff-on-failure --files $files > /out/precommit-$br.out 2>&1; r=\$?
cat /out/precommit-$br.out; exit \$r" 2>&1 | tail -80
    r=${PIPESTATUS[0]}
    echo "== $br pre-commit rc=$r" | tee -a "$OUT/verdict.txt"
    (( r == 0 )) || rc=1
done
exit $rc

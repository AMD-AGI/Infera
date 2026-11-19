#!/usr/bin/env bash
# Infera's pre-commit hooks (trailing whitespace, EOF, YAML, large files, merge markers, ruff format, ruff check) on the
# files each Infera PR branch changes vs origin/main, CPU only. cargo-fmt is skipped (needs a Rust toolchain; the Rust
# CI step infera-rust already runs cargo fmt --check on the Rust branches), gofmt has no Go files here, and
# check-author-email skips itself under CI=1 (all PR commits are authored as liyingli@amd.com). Each tree is copied
# into a scratch git repo in the container, so the host's .git is never written by root.
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/infera rev-parse origin/main pr/kv-aware-prepare pr/router-pool-idle pr/server-nofile pr/vllm-034-arg-parser pr/vllm-dp-engine-lb pr/vllm-mooncake-serial
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly || exit 1
mkdir -p "$NL/precommit-infera" "$NL/pip"
: > "$OUT/verdict.txt"
rc=0
for br in kv-aware-prepare router-pool-idle server-nofile vllm-034-arg-parser vllm-dp-engine-lb vllm-mooncake-serial; do
    w=$(iwt "$br")
    files=$(git -C "$w" diff --name-only --diff-filter=AM origin/main HEAD | tr '\n' ' ')
    echo "== $br $(git -C "$w" rev-parse --short=9 HEAD): $files"
    docker rm -f "up-iprecommit-$br" > /dev/null 2>&1
    # shellcheck disable=SC2046
    docker run --rm --name "up-iprecommit-$br" $(gpu_args) -v "$w:/src:ro" -v "$OUT:/out" -e CI=1 \
        -v "$NL/precommit-infera:/root/.cache/pre-commit" -v "$NL/pip:/root/.cache/pip" --entrypoint bash "$IMG_NIGHTLY" -c "
command -v git > /dev/null || { apt-get update -qq && apt-get install -y -qq git > /dev/null; }
pip install -q pre-commit > /out/pip-iprecommit-$br.log 2>&1
mkdir /tmp/t && cp -a /src/. /tmp/t && rm -f /tmp/t/.git && cd /tmp/t
git config --global --add safe.directory '*'
git init -q && git add -A && git -c user.email=liyingli@amd.com -c user.name=liyingli commit -q -m tree
SKIP=cargo-fmt,gofmt pre-commit run --show-diff-on-failure --files $files > /out/precommit-$br.out 2>&1; r=\$?
cat /out/precommit-$br.out; exit \$r" 2>&1 | tail -60
    r=${PIPESTATUS[0]}
    echo "== $br pre-commit rc=$r" | tee -a "$OUT/verdict.txt"
    (( r == 0 )) || rc=1
done
exit $rc

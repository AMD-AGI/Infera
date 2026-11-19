#!/usr/bin/env bash
# Infera Rust CI (as .github/workflows/ci.yml job rust: cargo fmt --check, clippy --locked --all-targets -D warnings,
# cargo test --locked) for origin/main and the PR branches that touch rust/. A cargo fmt difference is saved as a
# patch (fmt-<branch>.diff) to apply on the login node. Cargo home and target dir on the node's NVMe, shared.
# key: for d in main vllm-mooncake-serial router-pool-idle; do git -C /home/liyingli/bench_agentx/upstream-20261119/iwt/$d rev-parse HEAD; done
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready rust || exit 1
names="main vllm-mooncake-serial router-pool-idle"
mounts=""
for n in $names; do mounts+="-v $U/iwt/$n:/src/$n:ro "; done
mkdir -p "$NL/cargo" "$NL/cargo-target"
docker rm -f up-infera-rust > /dev/null 2>&1
# shellcheck disable=SC2086
docker run --rm --name up-infera-rust --network host $mounts -v "$OUT:/out" -v "$NL/cargo:/cargo-home" \
    -v "$NL/cargo-target:/target" -e CARGO_TARGET_DIR=/target --entrypoint bash up-rust:1 -c "
export CARGO_HOME=/cargo-home PATH=/usr/local/cargo/bin:\$PATH RUSTUP_HOME=/usr/local/rustup
rustup component add rustfmt clippy > /dev/null 2>&1
rustc --version; cargo clippy --version
for n in $names; do
    rm -rf /tmp/w && cp -r /src/\$n/rust /tmp/w && cd /tmp/w
    echo \"== \$n fmt\"; cargo fmt --all -- --check > /out/fmt-\$n.diff 2>&1 && echo fmt ok || echo FMT DIFF
    echo \"== \$n clippy\"; cargo clippy --locked --all-targets -j 32 -- -D warnings > /out/clippy-\$n.log 2>&1 \
        && echo clippy ok || { echo CLIPPY FAIL; tail -30 /out/clippy-\$n.log; }
    echo \"== \$n test\"; cargo test --locked -j 32 > /out/test-\$n.log 2>&1; echo \"test rc=\$?\"
    grep -E '^test result|FAILED|panicked' /out/test-\$n.log | head -12
    for t in \$(grep -oE '^test [^ ]+ \.\.\. FAILED' /out/test-\$n.log | awk '{print \$2}'); do
        cargo test --locked -j 32 -- --exact \$t --test-threads=1 > /dev/null 2>&1 && echo \"FLAKY \$t (passes alone)\" >> /out/flaky-\$n.txt
    done
done" 2>&1 | tee "$OUT/run.txt"
# Pass if every PR branch has clean clippy, fmt, and no test failure that main does not also have.
fail=0
for n in vllm-mooncake-serial router-pool-idle; do
    grep -A1 "== $n fmt" "$OUT/run.txt" | grep -q "fmt ok" || { echo "$n: cargo fmt differs (see fmt-$n.diff)"; fail=1; }
    grep -A1 "== $n clippy" "$OUT/run.txt" | grep -q "clippy ok" || { echo "$n: clippy failed"; fail=1; }
    for t in $(grep -oE '^test [^ ]+ \.\.\. FAILED' "$OUT/test-$n.log" | awk '{print $2}'); do
        grep -q "FLAKY $t " "$OUT/flaky-$n.txt" 2> /dev/null && { echo "$n: flaky $t (passes alone)"; continue; }
        grep -qE "^test $t \.\.\. FAILED" "$OUT/test-main.log" || { echo "$n: NEW test failure $t"; fail=1; }
    done
done | tee "$OUT/verdict.txt"
! grep -q "differs\|failed\|NEW" "$OUT/verdict.txt"

#!/usr/bin/env bash
# Apply both infera fixes from this packup onto a clean checkout, tests included.
#
# Why a script: `git diff` only captures TRACKED files, so the two patch files
# carry the source changes but NOT the new test files (test_net_port_block.py is
# brand new and untracked). Applying the .patch files alone leaves you with
# fixes and no regression coverage. This installs both halves.
#
# Usage:  bash patches/apply_all.sh /path/to/infera/repo
set -euo pipefail
REPO="${1:?usage: apply_all.sh <path-to-infera-repo>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO"
echo "== repo: $(pwd)  branch: $(git rev-parse --abbrev-ref HEAD)  head: $(git rev-parse --short HEAD)"

echo "== checking both patches apply cleanly"
git apply --check "$HERE/0001-free_tcp_port_block-randomise-scan-start.patch"
git apply --check "$HERE/0002-storage_classify-bind-mount-subpath.patch"

echo "== applying source fixes"
git apply "$HERE/0001-free_tcp_port_block-randomise-scan-start.patch"   # infera/common/net.py
git apply "$HERE/0002-storage_classify-bind-mount-subpath.patch"        # infera/kvd/storage_classify.py

echo "== installing tests"
# 0001's test is a NEW file (untracked upstream) -> copy, don't patch.
mkdir -p tests/unit/common tests/unit/kvd
cp "$HERE/test_net_port_block.py"    tests/unit/common/test_net_port_block.py
# 0002's test file already exists upstream; the packed copy is the full post-fix
# version (2 cases added). Overwrite rather than patch so it applies regardless
# of unrelated upstream drift in that file.
cp "$HERE/test_storage_classify.py"  tests/unit/kvd/test_storage_classify.py

echo "== running the affected suites"
python3 -m pytest tests/unit/common/test_net_port_block.py \
                  tests/unit/kvd/test_storage_classify.py -q

echo
echo "== done. expected: 51 passed (4 port-block + 47 storage-classify)"
echo "   review with: git diff --stat && git status --short"

#!/usr/bin/env bash
# The three suites group E touches, run with -v ON PURPOSE.
#
# `-v` is not decoration here: the engine image has no pytest-asyncio, where
# @pytest.mark.asyncio is an unknown mark and the coroutine is collected, never
# awaited, and reported as PASSING. The only way to see that a test really ran
# is to read it in the verbose list. See notes.md §3.
#
# Run patch_and_test.sh first: it unpacks liying_rest.tgz, which carries
# test_ready_timeout.py and the patched sources. This adds the two bigram
# suites, which the image's older tests/ tree predates.
set -u
docker cp /tmp/extra_tests.tgz merged_run:/tmp/ >/dev/null
docker exec merged_run bash -c '
set -u; cd /opt/infera
tar xzf /tmp/extra_tests.tgz -C /opt/infera
echo "=== new suites only (all must actually EXECUTE, not skip) ==="
python3 -m pytest tests/unit/common/test_net_port_block.py \
                  tests/engine/sglang/test_ready_timeout.py \
                  tests/unit/router/test_kv_event_bigram.py \
                  -p no:cacheprovider -v 2>&1 | tail -35
echo "pytest_rc=${PIPESTATUS[0]}"
'

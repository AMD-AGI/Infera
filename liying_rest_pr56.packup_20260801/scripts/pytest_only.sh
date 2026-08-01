#!/usr/bin/env bash
set -u
docker cp /tmp/extra_tests.tgz merged_run:/tmp/ >/dev/null
docker exec merged_run bash -c '
set -u
cd /opt/infera
tar xzf /tmp/extra_tests.tgz -C /opt/infera
echo "=== pytest: the two new suites + both bigram suites ==="
python3 -m pytest tests/unit/common/test_net_port_block.py \
                  tests/engine/sglang/test_ready_timeout.py \
                  tests/unit/router/test_kv_event_bigram.py \
                  tests/unit/router/test_kv_event_e2e.py \
                  -p no:cacheprovider -q 2>&1 | tail -30
echo "pytest_rc=${PIPESTATUS[0]}"
'

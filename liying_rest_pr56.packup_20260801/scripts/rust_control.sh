#!/usr/bin/env bash
set -u
docker cp /tmp/zmqtest.tgz merged_run:/tmp/ >/dev/null || exit 1
docker cp /tmp/_inner_rust_control.sh merged_run:/tmp/ >/dev/null || exit 1
docker exec merged_run bash -c 'tar xzf /tmp/zmqtest.tgz -C /opt/infera'
docker exec merged_run bash /tmp/_inner_rust_control.sh

#!/usr/bin/env bash
set -u
docker cp /tmp/_inner_full.sh merged_run:/tmp/ >/dev/null || exit 1
docker exec merged_run bash /tmp/_inner_full.sh

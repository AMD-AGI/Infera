#!/usr/bin/env bash
# Node-side driver for the Rust router A/B. Inner script must be a FILE:
# `docker exec` without -i has no stdin, so a heredoc is silently discarded.
set -u
docker cp /tmp/_inner_rust_ab.sh merged_run:/tmp/ >/dev/null || exit 1
docker exec merged_run bash /tmp/_inner_rust_ab.sh

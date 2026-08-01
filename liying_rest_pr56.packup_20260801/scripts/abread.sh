#!/usr/bin/env bash
# Read the round-3 A/B result out of both router logs.
#
# `tracing` colourises log field names, so `cache_hits` and its value are
# separated by ANSI escapes: a naive `grep -o 'cache_hits=[0-9]*'` finds NOTHING
# even when every pick is present in the log. Strip the escapes first.
set -u
docker exec merged_run bash -c '
for t in before after; do
  echo "############ $t ############"
  sed -r "s/\x1B\[[0-9;]*[mK]//g" /tmp/rustab/$t.log \
    | grep -o "cache_hits=[0-9]* request_blocks=[0-9]*"
done'

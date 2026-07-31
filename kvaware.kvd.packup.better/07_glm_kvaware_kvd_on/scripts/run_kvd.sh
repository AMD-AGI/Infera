#!/bin/bash
mkdir -p /tmp/kvd /tmp/kvd-long
rm -f /tmp/kvd/kvd.sock
exec python3 -m infera.kvd --socket /tmp/kvd/kvd.sock --max-bytes 16G \
  --long-path /tmp/kvd-long --long-bytes 128G --log-level INFO > /tmp/kvd.log 2>&1

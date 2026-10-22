#!/usr/bin/env bash
# Hard-stop guard: at 00:45Z 10-22 stop every yihou-dcp-pd-054-* container on 054 (docker stop only).
set -u
target=$(date -u -d "2026-10-22 00:45:00" +%s)
while (( $(date -u +%s) < target )); do sleep 30; done
for i in 1 2 3 4 5; do
  spur exec 228279 bash -c "cd /tmp && n=\$(docker ps -q --filter name=^yihou-dcp-pd-054- | wc -l); echo running=\$n; [ \$n -eq 0 ] || docker stop -t 120 \$(docker ps -q --filter name=^yihou-dcp-pd-054-)" && break
  sleep 30
done
spur exec 228279 bash -c "docker ps --format '{{.Names}}' | grep yihou-dcp-pd || echo 'no yihou-dcp-pd containers running'"
date -u

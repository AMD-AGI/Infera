#!/bin/bash
# Scope the engine-log error scan to a time window. These logs are appended across the
# whole session, so a whole-file grep mixes in earlier phases' traffic.
LOG="${1:-/tmp/glm52_mix_base.log}"; FROM="${2:-2026-08-06 11:}"
strings "$LOG" | grep -a "^\[$FROM" | grep -aiE 'error|abort|exceed|reject|invalid|Traceback|finish.*length' | tail -10 | cut -c1-230
echo "--- non-200 http in window ---"
strings "$LOG" | grep -a "^\[$FROM" | grep -aoE 'HTTP/1.1" [0-9]{3}' | sort | uniq -c

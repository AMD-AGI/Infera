#!/bin/bash
# robustly kill the native router (proc name sglang::router) + free :8100/:29000
pkill -9 -f launch_router 2>/dev/null
for p in $(ss -tlnp 2>/dev/null | grep -E ':29000|:8100' | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u); do kill -9 "$p" 2>/dev/null; done
sleep 2
ss -tlnp 2>/dev/null | grep -E ':29000|:8100' >/dev/null && echo "PORTS_STILL_HELD" || echo "PORTS_FREE"

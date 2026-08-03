#!/bin/bash
# One-shot health probe: driver line + both legs' fault counts + kvd.
D=/shared_nfs/yihou_agbench_mtp
echo "[$(date -u +%H:%M:%S)] $(tail -c 3000 $D/logs/caseA_armB.driver.log | tr '\r' '\n' | grep -E 'Sessions:' | tail -1 | sed 's/\x1b\[[0-9;]*m//g')"
pgrep -f "agent.agent_throughput" >/dev/null && echo "  driver=RUNNING" || echo "  driver=EXITED"

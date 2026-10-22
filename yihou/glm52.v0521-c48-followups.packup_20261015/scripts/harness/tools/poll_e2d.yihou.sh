#!/usr/bin/env bash
# One read-only poll of a running round: fault counters + seedguard + bench progress.
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/${1:-e2d}; f=$R/server-logs/decode-0.a1.log
echo "$(date -u +%T) SG=$(strings $f | grep -ac SEEDGUARD) bad>0=$(strings $f | grep -a SEEDGUARD | grep -ac 'bad_entries=[1-9]') MAF=$(strings $f | grep -ac 'Memory access fault')/$(strings $R/server-logs/prefill-0.log | grep -ac 'Memory access fault') kce=$(strings $f | grep -ac kernel_code_entry) TB=$(strings $f | grep -ac Traceback)/$(strings $R/server-logs/prefill-0.log | grep -ac Traceback)"
strings $f | grep -a SEEDGUARD | tail -1 | sed 's/.*YIHOU-SEEDGUARD //' | cut -c1-200
strings $R/agentx-c*/benchmark.log 2>/dev/null | grep -a "progress\|srv " | tail -1 | cut -c1-170
tail -1 $R/status.tsv | cut -c1-140

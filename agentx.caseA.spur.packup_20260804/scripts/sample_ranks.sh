#!/bin/bash
# Persist per-DP-rank prefill load at a fixed interval, for the whole run.
# One JSON line per tick -> results/rank_samples.jsonl
#
# Sources, per tick:
#   * prefill leg log  -> cumulative "Prefill batch" line count per DP rank,
#                         and the token-usage / running-req of each rank's last line
#   * router /metrics  -> per-worker picks + active_blocks (kv-aware bookkeeping)
# Cumulative counters are differenced offline; the raw values are kept as-is so
# nothing is lost to a bad delta.
set -u
W=/shared_nfs/yihou_agentx_caseA
JOB="${JOB:-35748}"
CTR="${CTR:-agx_caseA}"
LOG="${LOG:-$W/logs/prefill_dpa8.log}"
OUT="${OUT:-$W/results/rank_samples.jsonl}"
INT="${INT:-15}"
RURL="${RURL:-http://10.245.145.242:8190}"

while true; do
  TS=$(date -u +%FT%TZ)
  # per-rank cumulative prefill-batch counts, straight off the leg log
  COUNTS=$(strings "$LOG" 2>/dev/null | grep -aoE "DP[0-9]+ TP[0-9]+ EP[0-9]+\] Prefill batch" \
           | grep -oE "^DP[0-9]+" | sort | uniq -c \
           | awk '{printf "\"%s\":%s,", $2, $1}' | sed 's/,$//')
  # last-seen token usage per rank
  USAGE=$(strings "$LOG" 2>/dev/null | grep -aE "DP[0-9]+ TP[0-9]+ EP[0-9]+\] (Prefill|Decode) batch" \
           | grep -oE "DP[0-9]+.*token usage: [0-9.]+" \
           | awk '{for(i=1;i<=NF;i++){if($i ~ /^DP/){r=$i}; if($i=="usage:"){u=$(i+1)}} last[r]=u} END{for(r in last) printf "\"%s\":%s,", r, last[r]}' \
           | sed 's/,$//')
  printf '{"t":"%s","prefill_batches":{%s},"token_usage":{%s}}\n' "$TS" "$COUNTS" "$USAGE" >> "$OUT"
  sleep "$INT"
done

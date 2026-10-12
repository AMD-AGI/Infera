#!/usr/bin/env bash
# Kineto (torch profiler) CPU+GPU trace of the DECODE engine, 8 forward steps, via SGLang's native
# /start_profile (io_struct.py:2257-2283; http_server.py:1164), at profiling minute M of a chained round.
# Leader-approved for e2j R1 (fresh, reference run). Also writes a py-spy-free context snapshot (decode log tail).
# usage: kineto_on_chain.yihou.sh <chain_log> <tag e.g. e2jR1> [minute=12] [decode_url=http://10.245.147.130:29357]
set -uo pipefail
CHAIN=${1:?chain log}; TAG=${2:?tag}; MIN=${3:-12}; DEC=${4:-http://10.245.147.130:29357}
OUT=/shared_nfs/yihou_glm52/v0521dbg_20261010/diag/decode_cost.yihou/kineto.$TAG.yihou
mkdir -p "$OUT" && chmod 777 "$OUT"
for i in $(seq 1 1440); do l=$(grep -a "$TAG: " "$CHAIN" 2>/dev/null | tail -1); [[ -n "$l" ]] && break; sleep 10; done
echo "$(date -u +%T) $l"; case "$l" in *"profiling started"*) ;; *) echo "no profiling start; exit"; exit 1;; esac
t0=$(date -u -d "$(echo "$l" | grep -oE '^[0-9T:-]+Z')" +%s); at=$(( t0 + MIN * 60 )); now=$(date -u +%s)
(( at > now )) && sleep $(( at - now ))
echo "$(date -u +%T.%N) start_profile"
curl -s -m 30 -X POST "$DEC/start_profile" -H 'Content-Type: application/json' -d "{
  \"output_dir\": \"$OUT\", \"num_steps\": 8, \"activities\": [\"CPU\", \"GPU\"],
  \"with_stack\": false, \"record_shapes\": false, \"profile_prefix\": \"$TAG\"}" | tee "$OUT/start_profile.json"; echo
date -u +%T.%N > "$OUT/t_start"
sleep 120   # num_steps auto-stops; 4 ranks export traces
ls -la "$OUT"

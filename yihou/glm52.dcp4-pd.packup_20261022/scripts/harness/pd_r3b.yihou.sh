#!/usr/bin/env bash
# r3 on the LIVE r2 stack (prefill/router/etcd untouched): decode restarted with custom AR ON, NCCL debug off.
#  phase 1: decode a2 = sim OFF -> census/audit/smoke -> T4a (math + passkey); gate on math/8K/34K PASS
#  phase 2: decode a3 = sim 3.61 -> census/audit -> flush decode+prefill -> fresh AgentX C40 3600 s (tag r3)
# Login host: setsid nohup bash pd_r3.yihou.sh > runs/pd_r3.log 2>&1 < /dev/null &
set -uo pipefail
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness; JOB=228279; round=pdBeperf-r2; RD=$PD/runs/$round
export PD_JOB=$JOB PD_CONC=40
BASE="PREFILL_HICACHE_SIZE=450 DECODE_CUSTOM_AR=1 DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO"
ts(){ date -u +%FT%TZ; }
nx(){ spur exec "$JOB" bash -c "cd /tmp && $1"; }
pfx=yihou-dcp-pd-054-$round
client_running(){ local i o
  for i in $(seq 1 10); do
    o=$(nx "docker ps --format '{{.Names}}' | grep -c '^$pfx-agentx-client-'; echo RC=\$?" 2>/dev/null)
    [[ "$o" == *RC=* ]] && { [[ "${o%%$'\n'*}" -gt 0 ]] && return 0 || return 1; }
    sleep 20; done; return 0; }
wait_client(){ local i; for i in $(seq 1 150); do client_running || return 0; sleep 60; done; return 1; }
restart_decode(){ local att=$1
  echo "$(ts) stop decode (docker stop, container kept)"
  nx "docker stop -t 180 $pfx-decode-0" >/dev/null 2>&1 || echo "$(ts) docker stop exec returned non-zero (attach?); verifying"
  local i st=""; for i in $(seq 1 20); do
    st=$(nx "docker inspect -f '{{.State.Running}}' $pfx-decode-0" 2>/dev/null); [[ "$st" == false ]] && break; sleep 15; done
  [[ "$st" == false ]] || { echo "$(ts) decode still running/unknown ($st)"; return 1; }
  sleep 20
  (cd "$H" && source env.$round-054.yihou.sh && bash run.v2.yihou.sh retry-decode "$att") || return 1
  (cd "$H" && source env.$round-054.yihou.sh && for m in census audit; do bash run.v2.yihou.sh $m > "$RD/$m.$att.txt" 2>&1; echo "$m rc=$?"; done)
}
echo "$(ts) wait for any running client (prof load)"; wait_client
# (v2, 18:47Z: tolerant decode stop -- spur exec attach errors on docker stop)
# ---- phase 1
export PD_EXTRA="$BASE DECODE_SIMULATE_ACC_LEN="
restart_decode a2 || { echo "$(ts) STOP: decode a2 failed"; exit 1; }
(cd "$H" && source env.$round-054.yihou.sh && bash run.v2.yihou.sh smoke > "$RD/smoke.a2.txt" 2>&1; echo "smoke rc=$?")
(cd "$H" && source env.$round-054.yihou.sh && bash t4.yihou.sh a)
f=""; for i in $(seq 1 60); do f=$(ls -t "$RD"/t4/check.*.txt 2>/dev/null | head -1); [[ -n "$f" && $(grep -c "^passkey_96k" "$f") -ge 1 ]] && break; sleep 30; done
echo "$(ts) T4a result $f:"; cat "$f"
for k in math_nothink passkey_8k passkey_34k; do grep -q "^$k PASS" "$f" || { echo "$(ts) STOP: T4a $k not PASS"; exit 1; }; done
# ---- phase 2
export PD_EXTRA="$BASE"
restart_decode a3 || { echo "$(ts) STOP: decode a3 failed"; exit 1; }
for port in 29357 29101; do for i in $(seq 1 60); do
  c=$(nx "curl -s -o /dev/null -w '%{http_code}' -X POST http://10.245.144.125:$port/flush_cache" 2>/dev/null)
  echo "$(ts) flush :$port -> $c"; [[ "$c" == 200 ]] && break; sleep 10; done; done
echo "$(ts) bench r3 (fresh decode, C40 3600 s)"
(cd "$H" && source env.$round-054.yihou.sh && BENCH_TAG=r3 bash run.v2.yihou.sh bench); echo "$(ts) bench driver rc=$? (attach errors are expected; polling the client)"
sleep 120; wait_client
ls "$RD"/agentx-c40-r3/agentx_conc40.json >/dev/null 2>&1 && echo "$(ts) r3 JSON present" || echo "$(ts) r3: no JSON"
echo "$(ts) R3 DONE"

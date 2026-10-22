#!/usr/bin/env bash
# Full-restart sequence for one node (no decode-only restarts: r3 showed errors after one):
#   G) launch variant Be  (sim OFF) round <TAG>g -> T4a gate (math, passkey 8K, 34K PASS) -> stop
#   P) launch variant Beperf (sim 3.61) round <TAG> -> AgentX A1 (fresh) -> flush -> A2 (warm)
# Login host: PD_EXTRA="..." setsid nohup bash pd_seq.yihou.sh NK JOB TAG [CONC=40] > runs/pd_seq.<TAG>.log 2>&1 < /dev/null &
set -uo pipefail
NK=$1; JOB=$2; TAG=$3; CONC=${4:-40}
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness
export PD_JOB=$JOB PD_CONC=$CONC
IP=$(awk -F'\t' '$1=="decode"{print $3}' "$H/topology.yihou.$NK.tsv")
ts(){ date -u +%FT%TZ; }
nx(){ spur exec "$JOB" bash -c "cd /tmp && $1"; }
nclients(){ local pfx=$1 i o; for i in $(seq 1 10); do
    o=$(nx "docker ps --format '{{.Names}}' | grep -c '^$pfx-agentx-client-'; echo RC=\$?" 2>/dev/null)
    [[ "$o" == *RC=* ]] && { echo "${o%%$'\n'*}"; return; }; sleep 20; done; echo unknown; }
bench(){ local round=$1 t=$2 pfx=yihou-dcp-pd-$NK-$1 try n i
  for try in 1 2 3 4; do
    local tt=$t; [[ $try -gt 1 ]] && tt=$t-t$try
    echo "$(ts) bench $round $tt try $try"
    (cd "$H" && source env.$round-$NK.yihou.sh && BENCH_TAG=$tt bash run.v2.yihou.sh bench); echo "$(ts) driver rc=$? (attach lost is expected)"
    ls $PD/runs/$round/agentx-c$CONC-$tt/agentx_conc*.json >/dev/null 2>&1 && { echo "$(ts) driver stayed attached; JSON present"; break; }
    sleep 60; n=$(nclients $pfx); echo "$(ts) clients running: $n"; [[ "$n" != 0 ]] && break
  done
  for i in $(seq 1 150); do n=$(nclients $pfx); [[ "$n" == 0 ]] && break; sleep 60; done
  echo "$(ts) client done: $(ls $PD/runs/$round/agentx-c$CONC-$t*/agentx_conc*.json 2>/dev/null | tr '\n' ' ')"; }
flush(){ local port c i; for port in 29357 29101; do for i in $(seq 1 60); do
    c=$(nx "curl -s -o /dev/null -w '%{http_code}' -X POST http://$IP:$port/flush_cache" 2>/dev/null)
    echo "$(ts) flush :$port -> $c"; [[ "$c" == 200 ]] && break; sleep 10; done; [[ "$c" == 200 ]] || return 1; done; }
stop_round(){ local round=$1 i st
  (cd "$H" && source env.$round-$NK.yihou.sh && bash run.v2.yihou.sh stop) > "$PD/runs/$round/stop.txt" 2>&1
  for i in $(seq 1 10); do
    st=$(nx "docker ps -q --filter name=^yihou-dcp-pd-$NK-$round- | wc -l" 2>/dev/null)
    [[ "$st" == 0 ]] && { echo "$(ts) $round stopped"; return 0; }
    nx "docker stop -t 120 \$(docker ps -q --filter name=^yihou-dcp-pd-$NK-$round-)" >/dev/null 2>&1; sleep 20
  done; echo "$(ts) WARN: $round containers may still run"; return 1; }
# ---- G  (SKIP_GATE=1 skips it, e.g. a perf re-run after a gate that already passed on the same code)
if [[ "${SKIP_GATE:-0}" != 1 ]]; then
g=pdBe-${TAG}g
bash "$H/launch_pd.yihou.sh" "$NK" "$JOB" Be "${TAG}g" || { echo "$(ts) STOP: gate launch failed"; exit 1; }
f=""; for i in $(seq 1 60); do f=$(ls -t "$PD/runs/$g"/t4/check.*.txt 2>/dev/null | head -1); [[ -n "$f" ]] && grep -q "^passkey_96k" "$f" && break; sleep 30; done
echo "$(ts) gate T4a ($f):"; cat "$f"
for k in math_nothink passkey_8k passkey_34k; do grep -q "^$k PASS" "$f" || { echo "$(ts) STOP: gate $k not PASS (stack $g left up for debugging)"; exit 1; }; done
stop_round "$g"
fi
# ---- P
p=pdBeperf-$TAG
[[ -n "${OLD_ROUND:-}" ]] && stop_round "$OLD_ROUND"
bash "$H/launch_pd.yihou.sh" "$NK" "$JOB" Beperf "$TAG" || { echo "$(ts) STOP: perf launch failed"; exit 1; }
bench "$p" A1
flush || { echo "$(ts) STOP: flush failed"; exit 1; }
bench "$p" A2
echo "$(ts) SEQ DONE $p (stack left up)"

#!/usr/bin/env bash
# Purpose: run the whole P8D8 sweep end to end, unattended:
#   gate on free GPUs -> launch the deployment -> wait for health -> run every
#   concurrency point in sequence against that one deployment.
# Usage: ./run_full_sweep.yihou.sh [CONC ...]     (default: 80 112 144 192 256)
# Artifacts: $SWEEP/c<NNN>/ per point, $SWEEP/driver.log, $LOGD/orchestrator.log
#
# Why one deployment for every point: a teardown costs ~33 minutes of GPU-memory
# release (measured 2026-09-18), and relaunching per point would also add "a
# different engine process" as a variable to a sweep that is supposed to vary
# only concurrency.
#
# Why the GPU gate: relaunching into an incomplete release jams startup. See
# wait_gpus_free.yihou.sh.
set -uo pipefail

WS="${WS:-/home/yihou/dev/git/infera.glm52.view/glm52.p8d8-vs-p4d4_20260918}"
BENCH="${BENCH:-/home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd}"
SWEEP="${SWEEP:-/mnt/m2m_nobackup/yihou_p8p4/sweep}"
LOGD="${LOGD:-/mnt/m2m_nobackup/yihou_p8p4/logs}"
LAUNCH_OUT="${LAUNCH_OUT:-/mnt/m2m_nobackup/yihou_p8p4/launch/sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG="$WS/scripts/config.yihou.p8d8.sh"
TOPO="$WS/scripts/topology.yihou.tsv"
PNODE="${PNODE:-crsuse2-m2m-137}"
DNODE="${DNODE:-crsuse2-m2m-136}"
POINTS=("$@"); [[ ${#POINTS[@]} -gt 0 ]] || POINTS=(80 112 144 192 256)

mkdir -p "$SWEEP" "$LOGD"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGD/orchestrator.log"; }

say "=== orchestrator start: points ${POINTS[*]} ==="

# 1. GPUs must be genuinely free on BOTH nodes before launching.
say "gate: waiting for all GPUs free"
TIMEOUT_S=3600 INTERVAL_S=30 bash "$WS/scripts/wait_gpus_free.yihou.sh" "$PNODE" "$DNODE" \
    >> "$LOGD/orchestrator.log" 2>&1
if [[ $? -ne 0 ]]; then say "GATE FAILED -- not launching"; exit 1; fi
say "gate passed"

# 2. No stale containers may hold the names.
stale=$(ssh -o BatchMode=yes "$PNODE" 'docker ps -a --format "{{.Names}}" | grep -c p8d8' 2>/dev/null || echo 0)
stale2=$(ssh -o BatchMode=yes "$DNODE" 'docker ps -a --format "{{.Names}}" | grep -c p8d8' 2>/dev/null || echo 0)
if (( stale > 0 || stale2 > 0 )); then
    say "removing $stale + $stale2 stale p8d8 containers (all names contain yihou)"
    ssh -o BatchMode=yes "$PNODE" 'docker ps -a --format "{{.Names}}" | grep p8d8 | xargs -r docker rm -f' >/dev/null 2>&1
    ssh -o BatchMode=yes "$DNODE" 'docker ps -a --format "{{.Names}}" | grep p8d8 | xargs -r docker rm -f' >/dev/null 2>&1
fi

# 3. Launch.
say "launching deployment -> $LAUNCH_OUT"
ssh -o BatchMode=yes "$PNODE" "cd $BENCH && ./launch.sh CONFIG=$CONFIG TOPOLOGY=$TOPO OUT_DIR=$LAUNCH_OUT" \
    > "$LOGD/launch-orchestrated.log" 2>&1
rc=$?
say "launch.sh exited $rc"

# 4. Health, with patience: a cold start has taken up to ~22 min on this model.
say "waiting for all three endpoints"
deadline=$(( $(date +%s) + 2400 ))
while :; do
    p=$(ssh -o BatchMode=yes "$PNODE" 'curl -s -m 5 -o /dev/null -w "%{http_code}" http://10.245.153.247:29001/health' 2>/dev/null)
    d=$(ssh -o BatchMode=yes "$PNODE" 'curl -s -m 5 -o /dev/null -w "%{http_code}" http://10.245.154.168:29002/health' 2>/dev/null)
    r=$(ssh -o BatchMode=yes "$PNODE" 'curl -s -m 5 -o /dev/null -w "%{http_code}" http://10.245.153.247:28000/health' 2>/dev/null)
    say "  health prefill=$p decode=$d router=$r"
    [[ "$p" == 200 && "$d" == 200 && "$r" == 200 ]] && break
    if (( $(date +%s) > deadline )); then say "HEALTH TIMEOUT -- aborting"; exit 1; fi
    sleep 30
done
say "deployment healthy"

# 5. Confirm the mitigation actually reached the decode engine, rather than
#    assuming the config was honoured. A setting written into a config is not a
#    setting the engine received -- agentx_env.py caught exactly that class of
#    drift once already today.
#
#    NOTE the mitigation CHANGED on 2026-09-19: `SGLANG_DSA_FUSE_TOPK=0` was
#    withdrawn (it stopped the crash but garbled the output) and replaced by
#    `index_share_for_mtp_iteration=false`, which keeps the fused indexer.
#    This check was left pointing at the old variable and would have aborted the
#    launch it was meant to protect.
idx=$(ssh -o BatchMode=yes "$DNODE" \
    'docker inspect glm52-pd-yihou-p8d8-decode-0 --format "{{join .Config.Cmd \" \"}}" | grep -c "index_share_for_mtp_iteration"; true' 2>/dev/null | tail -1)
fuse_off=$(ssh -o BatchMode=yes "$DNODE" \
    'docker inspect glm52-pd-yihou-p8d8-decode-0 --format "{{join .Config.Env \"\n\"}}" | grep -c "SGLANG_DSA_FUSE_TOPK=0"; true' 2>/dev/null | tail -1)
say "decode: index_share override present=$idx (expect 1), DSA_FUSE_TOPK=0 present=$fuse_off (expect 0)"
if [[ "$idx" != 1 ]]; then say "MITIGATION MISSING -- aborting before wasting hours"; exit 1; fi
if [[ "$fuse_off" != 0 ]]; then say "WITHDRAWN WORKAROUND STILL SET -- aborting"; exit 1; fi

# 5b. TEXT COHERENCE GATE -- the lesson of 2026-09-19.
#     The previous configuration ran two full 3,600 s points, reported zero rail
#     faults and a perfectly healthy-looking acceptance gauge of 3.5-3.8, and was
#     emitting `1!au!au!au!...` the whole time. Under simulated acceptance the
#     gauge is FORCED and carries no correctness information; the text does.
#     Garbling IS visible in the text even with simulation on -- that is how it
#     was found -- so this gate works without a separate simulation-off launch.
#     This configuration (IndexShare=false + custom all-reduce ON) is untested
#     for correctness by anyone, so it must not go to a five-hour sweep unchecked.
say "text coherence gate: 3 probes, expecting the first ten primes in order"
bad=0
for i in 1 2 3; do
    txt=$(ssh -o BatchMode=yes "$PNODE" "curl -s -m 90 http://10.245.153.247:28000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"glm5.2-mxfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"List the first 10 prime numbers, separated by commas.\"}],\"temperature\":0,\"max_tokens\":256}'" 2>/dev/null \
        | python3 -c 'import json,sys
try:
    m=json.load(sys.stdin)["choices"][0]["message"]
    print(((m.get("content") or "")+" "+(m.get("reasoning_content") or "")).replace(chr(10)," ")[:400])
except Exception as e:
    print("PARSE_ERROR")' )
    ok=$(printf '%s' "$txt" | python3 -c '
import sys
t=sys.stdin.read(); pos=-1
for p in ["2","3","5","7","11","13","17","19","23","29"]:
    j=t.find(p,pos+1)
    if j<0: print("NO"); raise SystemExit
    pos=j
print("YES")')
    say "  probe $i: $ok  ${txt:0:90}"
    [[ "$ok" == YES ]] || bad=$((bad+1))
done
if (( bad > 0 )); then
    if [[ -n "${ALLOW_GARBLED:-}" ]]; then
        say "GARBLED ($bad/3) but ALLOW_GARBLED set -- proceeding. Simulated acceptance"
        say "  waives correctness by construction; the reference c32/c40 kit does the same."
    else
        say "GARBLED OUTPUT ($bad/3 probes failed the content check) -- ABORTING before the sweep"
        exit 1
    fi
fi
say "text coherence gate passed"

# 6. Run every point.
say "starting sweep driver for points ${POINTS[*]}"
ssh -o BatchMode=yes "$PNODE" \
    "SRVLOG=$LAUNCH_OUT/server-logs/prefill-0.log bash $WS/scripts/sweep_driver.yihou.sh ${POINTS[*]}" \
    >> "$LOGD/orchestrator.log" 2>&1
say "=== orchestrator finished (driver exit $?) ==="

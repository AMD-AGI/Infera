#!/usr/bin/env bash
# Queue step: one AgentX point on the running kit deployment, done the way scripts/tune_round.sh does it on ours:
# stop the router -> /flush_cache the prefill (device and host tiers) -> start the router for this variant ->
# the kit's engine/agentx/agentx.sh -> copy its output back -> flush check, engine pace, decode step fit, and
# results/<VARIANT>/c<CONC>/ for scripts/tune_report.py.
# ROUTER=image runs the image's infera-router (PR 210 as built). ROUTER=exp mounts pr210/router-src's build
# (PR 210 + --decode-request-weight) and sets INFERA_DECODE_REQUEST_WEIGHT=WEIGHT. The router command line is
# engine/up.sh's, unchanged.
# Usage: kit_round.sh RUN=<id> VARIANT=<name> CONC=<n> [ROUTER=image|exp] [WEIGHT=0] [DURATION=3600] [WARMUP_PER_LANE=10]
set -uo pipefail
for a in "$@"; do export "$a"; done
: "${RUN:?}" "${VARIANT:?}" "${CONC:?}"
ROUTER="${ROUTER:-image}" WEIGHT="${WEIGHT:-0}" DURATION="${DURATION:-3600}" WARMUP_PER_LANE="${WARMUP_PER_LANE:-10}"
source "$(dirname "$0")/kit_env.sh"
OUT="$PR_RUNS/$RUN/$VARIANT"; LOGS="$PR_RUNS/$RUN/logs"; WD=/scratch/liyingli_2p1d/pr210/work
mkdir -p "$OUT"
exec > >(tee -a "$OUT/round.log") 2>&1
echo "ROUTER=$ROUTER WEIGHT=$WEIGHT CONC=$CONC DURATION=$DURATION WARMUP_PER_LANE=$WARMUP_PER_LANE" > "$OUT/round.env"

# --- router down, prefill flushed (tune_round.sh: /flush_cache answers 400 while requests are in flight)
ssh "$P_NODE" "docker rm -f $CTR-router >/dev/null 2>&1; true"
code=000
for i in $(seq 1 30); do
    code="$(curl -s -o "$OUT/flush.body" -w '%{http_code}' --max-time 120 -X POST "$PREFILL_URL/flush_cache")"
    [[ "$code" == 200 ]] && break
    sleep 10
done
[[ "$code" == 200 ]] || die "flush: prefill still $code"
flushed_at="$(date -u +%Y-%m-%dT%H:%M:%S)"
echo "$flushed_at" > "$OUT/flushed_at"
say "flushed prefill at $flushed_at"

# --- router for this variant
rargs=(--host 0.0.0.0 --port 28300 --router-backend rust
    --discovery-backend etcd --etcd-endpoint "$P_IP:24701"
    --request-transport http --kv-event-transport zmq --router-tokenizer-path /perf_apps/data/models/GLM-5.2-MXFP4
    --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2
    --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on)
dargs=(-d --name "$CTR-router" --network host -v /perf_apps/data/models/GLM-5.2-MXFP4:/perf_apps/data/models/GLM-5.2-MXFP4:ro)
case "$ROUTER" in
    image) ;;
    exp) [[ -x "$EXP_ROUTER_BIN" ]] || die "no experimental router at $EXP_ROUTER_BIN"
         dargs+=(-v "$EXP_ROUTER_BIN:/opt/infera-router-exp:ro" -e INFERA_ROUTER_BIN=/opt/infera-router-exp
                 -e "INFERA_DECODE_REQUEST_WEIGHT=$WEIGHT") ;;
    *) die "ROUTER must be image or exp" ;;
esac
ssh "$P_NODE" "docker run $(q "${dargs[@]}") infera-sglang:pr210-aab4cc36 python3 -m infera.server $(q "${rargs[@]}") >/dev/null" ||
    die "router start"
for i in $(seq 1 60); do
    modes="$(curl -sf -m 10 "$ROUTER_URL/v1/workers" | python3 -c 'import json,sys; print(*sorted(w["disagg_mode"] for w in json.load(sys.stdin)["workers"]))' 2>/dev/null)"
    [[ "$modes" == "decode prefill" ]] && break
    sleep 10
done
[[ "$modes" == "decode prefill" ]] || die "router lists '$modes'"
echo "=== $(date -u +%FT%TZ) router for $VARIANT (ROUTER=$ROUTER WEIGHT=$WEIGHT)" >> "$LOGS/router.log"
follow "$CTR-router" "$P_NODE" "$LOGS/router.log"
ssh "$P_NODE" "docker exec $CTR-router ps -eo args | grep -m1 infera-router" > "$OUT/router.cmdline"
if [[ "$ROUTER" == exp ]]; then
    grep -q /opt/infera-router-exp "$OUT/router.cmdline" || die "router is not the experimental binary: $(cat "$OUT/router.cmdline")"
    got="$(ssh "$P_NODE" "docker exec $CTR-router printenv INFERA_DECODE_REQUEST_WEIGHT")"
    [[ "$got" == "$WEIGHT" ]] || die "router has INFERA_DECODE_REQUEST_WEIGHT='$got', want $WEIGHT"
fi
curl -sf -m 10 "$ROUTER_URL/v1/workers" > "$OUT/workers.json"
say "router up ($(tr -s ' ' < "$OUT/router.cmdline" | cut -c1-160))"

# --- AgentX (the kit writes runs/<RUN>/agentx; keep one directory per variant)
started="$(date -u +%FT%TZ)"; echo "$started" > "$OUT/agentx.started"
ssh "$P_NODE" "d=$WD/runs/$RUN; [[ ! -e \$d/agentx ]] || mv \$d/agentx \$d/agentx.moved-\$(date +%H%M%S)"
kit_on "$P_NODE" "CONC=$CONC DURATION=$DURATION WARMUP_PER_LANE=$WARMUP_PER_LANE bash engine/agentx/agentx.sh" > "$OUT/agentx.console.log" 2>&1
rc=$?
date -u +%FT%TZ > "$OUT/agentx.ended"
ssh "$P_NODE" "mv $WD/runs/$RUN/agentx $WD/runs/$RUN/agentx-$VARIANT && tar -C $WD/runs/$RUN --exclude=tmp -cf - agentx-$VARIANT" |
    tar -C "$OUT" -xf - || say "copy back failed"
say "agentx rc=$rc"

# --- analysis
agx="$OUT/agentx-$VARIANT"
python3 "$R/scripts/pr210/kit_window.py" "$agx/runner.log" "$started" > "$OUT/window" || die "no profiling window"
read -r w0 w1 < "$OUT/window"
python3 "$R/scripts/flush_check.py" "$LOGS" "$flushed_at" > "$OUT/flush_check.txt" 2>&1
python3 "$R/scripts/engine_pace.py" "$LOGS" decode-0.log "$w0" "$w1" > "$OUT/engine_pace.txt" 2>&1
python3 "$R/scripts/decode_step_fit.py" "$LOGS/decode-0.log" "$w0" "$w1" > "$OUT/decode_fit.txt" 2>&1
res="$R/results/$VARIANT/c$CONC"; mkdir -p "$res"
cp "$agx/agentx_conc$CONC.json" "$res/" 2>/dev/null
find "$agx" -name profile_export_aiperf.json -exec cp {} "$res/" \; -quit
cp "$OUT/engine_pace.txt" "$OUT/decode_fit.txt" "$OUT/flush_check.txt" "$OUT/round.env" "$res/"
(cd "$R" && python3 scripts/tune_report.py "$VARIANT/$CONC") > "$OUT/report.md" 2>&1
grep -m1 -A3 "^| 轮次" "$OUT/report.md" | tail -n 1
(( rc == 0 )) || die "agentx.sh rc=$rc"
say "done $VARIANT"

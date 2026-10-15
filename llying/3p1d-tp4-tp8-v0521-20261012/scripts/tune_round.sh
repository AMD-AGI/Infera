#!/usr/bin/env bash
# One tuning round on the live stack, engines untouched:
#   memory snapshot -> stop router -> /flush_cache the 3 prefills -> router with this
#   variant's INFERA_* switches (verified from its startup log) -> AgentX at CONC
#   -> tune_post.sh (census, flush check, balance / pace analyses, router metrics, publish).
# The client always sends X-Dynamo-Session-ID; the router ignores it unless
# INFERA_SESSION_AFFINITY is on. Decode is not flushed: it keeps no prefix cache, and
# its warm state is part of what the rounds hold constant.
# Usage: tune_round.sh VARIANT=<name> CONC=<n> [ROUTER_ENV="INFERA_X=v ..."] [ROUND=r2]
# Output: runtime/runs/<ROUND>/tune/<VARIANT>-c<CONC>/, published to results/<VARIANT>/c<CONC>/.
set -uo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${VARIANT:?}" "${CONC:?}"
[[ "$VARIANT" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || { echo "bad VARIANT '$VARIANT'" >&2; exit 2; }
[[ "$CONC" =~ ^[1-9][0-9]*$ ]] || { echo "bad CONC '$CONC'" >&2; exit 2; }
export ROUND="${ROUND:-r2}" ROUTER_ENV="${ROUTER_ENV:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set -a; source "$R/config/config.3p1d.sh"; set +a
LOGS="$RUNS/launch/server-logs"
DECODE_LOG="${DECODE_LOG:-decode-0.after-realacc.log}"
name="$VARIANT-c$CONC"
OUT="$RUNS/tune/$name"
RES="$R/results/$VARIANT/c$CONC"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
status() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "$1" "${2:-}" | tee -a "$RUNS/tune/status.tsv"; }
die() { status "$1" "FAIL ${2:-}"; exit 1; }
noansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"
router="$CONTAINER_PREFIX-router"
[[ "$router" == *llying* ]] || exit 2

mkdir -p "$RUNS/tune"
[[ ! -e "$OUT" && ! -e "$RES" ]] || { echo "$OUT or $RES already exists" >&2; exit 2; }
bash "$HERE/wait_home_space.sh" 20 10800 || die pre "/home has no room"
mkdir -p "$OUT"
printf 'VARIANT=%s\nCONC=%s\nROUND=%s\nROUTER_ENV=%s\nDECODE_LOG=%s\n' \
    "$VARIANT" "$CONC" "$ROUND" "$ROUTER_ENV" "$DECODE_LOG" > "$OUT/round.env"
status start "ROUTER_ENV='$ROUTER_ENV'"

snapshot() {  # per-GPU VRAM and host memory on every node; before a flush this is the
              # previous round's high-water mark (the flush empties the allocator cache)
    local node
    for node in $(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | cut -f4 | sort -u); do
        echo "== $node"
        on "$node" bash -c 'rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2; free -g | head -2'
    done > "$OUT/mem.$1.txt" 2>&1
}

# --- preconditions
# /home is a shared volume; it filled up on 2026-10-13 10:42 and every log follower and
# this pipeline died on write errors. A round writes ~0.5 GB here; wait for headroom.
for i in $(seq 1 60); do
    free_gb="$(df -B1G --output=avail "$RUNS" | tail -1 | tr -d ' ')"
    (( free_gb >= 20 )) && break
    (( i % 10 == 1 )) && status pre "waiting: only ${free_gb} GB free on $(df --output=target "$RUNS" | tail -1)"
    (( i < 60 )) || die pre "only ${free_gb} GB free after 30 min"
    sleep 30
done
curl -fsS --max-time 10 "$router_url/health" >/dev/null || die pre "router unhealthy"
clients="$(on "$CONTROL_NODE" docker ps --format '{{.Names}}' | grep -c -- "^$CONTAINER_PREFIX-agentx-client-" || true)"
[[ "$clients" == 0 ]] || die pre "$clients AgentX client container(s) still running"
curl -fsS --max-time 10 "$router_url/v1/workers" > "$OUT/workers.before.json" || die pre "workers"
mapfile -t prefill_urls < <(python3 -c '
import json,sys
p=json.load(open(sys.argv[1])); w=p if isinstance(p,list) else (p.get("workers") or p.get("data") or [])
print("\n".join(sorted(x["url"] for x in w if x.get("disagg_mode")=="prefill")))' "$OUT/workers.before.json")
want_prefill="$(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | awk -F'\t' '$3=="prefill"' | wc -l)"
(( ${#prefill_urls[@]} == want_prefill )) || die pre "expected $want_prefill prefill workers, got ${#prefill_urls[@]}"
snapshot before

# --- retire router, flush prefills
tag="$name-$(date -u +%H%M%S)"
on "$CONTROL_NODE" docker stop --time 30 "$router" >/dev/null || die router "stop"
on "$CONTROL_NODE" docker rename "$router" "$router-$tag" || die router "rename"
for url in "${prefill_urls[@]}"; do
    code=000
    for i in $(seq 1 90); do
        code="$(curl -s -o "$OUT/flush.body" -w '%{http_code}' --max-time 120 -X POST "$url/flush_cache")"
        [[ "$code" == 200 ]] && break
        sleep 10
    done
    printf '%s %s try=%s code=%s %s\n' "$(date -u +%FT%TZ)" "$url" "$i" "$code" \
        "$(tr '\n' ' ' < "$OUT/flush.body" | cut -c1-120)" >> "$OUT/flush.txt"
    [[ "$code" == 200 ]] || die flush "$url still $code after $i tries"
done
flushed_at="$(date -u +%Y-%m-%dT%H:%M:%S)"
echo "$flushed_at" > "$OUT/flushed_at"
status flush "ok ${#prefill_urls[@]} prefills at $flushed_at"

# --- router with this variant's switches
bash "$HERE/restart_router.sh" "ROUND=$ROUND" AFFINITY=false "TAG=unused-$tag" "ROUTER_ENV=$ROUTER_ENV" \
    > "$OUT/router.restart.log" 2>&1 || die router "restart rc=$? (see router.restart.log)"
on "$CONTROL_NODE" docker logs "$router" 2>&1 | noansi |
    grep -E 'router experiment controls|streaming guard|session affinity configuration' \
    > "$OUT/router.controls.txt"
want_guard=false; want_aff=Off; want_r2=Off
[[ " $ROUTER_ENV " == *" INFERA_PD_PREFILL_GUARD_RELEASE=completion "* ]] && want_guard=true
[[ " $ROUTER_ENV " == *" INFERA_SESSION_AFFINITY=prefill "* ]] && want_aff=Prefill
[[ " $ROUTER_ENV " == *" INFERA_SESSION_AFFINITY=both "* ]] && want_aff=Both
[[ " $ROUTER_ENV " == *" INFERA_R2_DECODE_DEMAND=on "* ]] && want_r2=On
grep -q "prefill_guard_completion: $want_guard, decode: $want_r2," "$OUT/router.controls.txt" ||
    die router "experiment controls != guard=$want_guard r2=$want_r2"
grep -q "mode=$want_aff " "$OUT/router.controls.txt" || die router "session affinity != $want_aff"
status router "ok guard=$want_guard affinity=$want_aff r2=$want_r2"

# --- AgentX
start="$(date -u +%Y-%m-%dT%H:%M:%S)"
echo "$start" > "$OUT/bench.start"
( sleep 300; python3 "$HERE/flush_check.py" "$LOGS" "$flushed_at" > "$OUT/flush_check.early.txt" 2>&1 ) &
status bench "start C$CONC"
# Through a pipe: agentx_bench.sh runs under set -e, and a failed write to a full /home on its own
# stdout would kill it mid-run; tee only warns and keeps draining.
bash "$BENCH_DIR/agentx_bench.sh" "CONFIG=$R/config/config.3p1d.sh" "ROUND=$ROUND" "CONC=$CONC" \
    "OUT_DIR=$OUT/agentx-$name" AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true 2>&1 |
    tee --output-error=warn "$OUT/agentx.log" > /dev/null
rc=${PIPESTATUS[0]}
bash "$HERE/wait_home_space.sh" 5 7200 || true
exec bash "$HERE/tune_post.sh" "VARIANT=$VARIANT" "CONC=$CONC" "ROUND=$ROUND" "AGENTX_RC=$rc"

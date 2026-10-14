#!/usr/bin/env bash
# Finish a tuning round whose host-side driver died while the AgentX client container kept
# running (2026-10-13 10:42: /home full): wait for the client container to exit, take its full
# log and output from the control node (as agentx_bench.sh would have), then tune_post.sh.
# Usage: tune_finish.sh VARIANT=<name> CONC=<n> [ROUND=r2]
set -uo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${VARIANT:?}" "${CONC:?}"
export ROUND="${ROUND:-r2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set -a; source "$R/config/config.3p1d.sh"; set +a
name="$VARIANT-c$CONC"
OUT="$RUNS/tune/$name"
dst="$OUT/agentx-$name"
client="$(cat "$dst/client.container")"
client_out="$NODE_RW_ROOT/runs/$YIHOU_ROUND/agentx-$name"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
say() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "finish" "$*" | tee -a "$RUNS/tune/status.tsv"; }
[[ "$client" == "$CONTAINER_PREFIX-agentx-client-"* ]] || { echo "unexpected client name '$client'" >&2; exit 2; }
say "waiting for $client to exit"
for i in $(seq 1 720); do
    running="$(on "$CONTROL_NODE" docker inspect -f '{{.State.Running}}' "$client" 2>/dev/null)"
    [[ "$running" == false ]] && break
    (( i < 720 )) || { say "client still running after 6 h"; exit 1; }
    sleep 30
done
code="$(on "$CONTROL_NODE" docker inspect -f '{{.State.ExitCode}}' "$client")"
echo "$code" > "$dst/client.container-exit-code"
on "$CONTROL_NODE" docker logs "$client" > "$dst/runner.log" 2>&1
on "$CONTROL_NODE" bash -c "tar -C $(printf '%q' "$client_out") --exclude=./aiperf_artifacts/server_metrics_export.json -cf - . | base64 -w0" |
    base64 -d | tar -C "$dst" -xf - || say "copy-back of $client_out failed"
result="$dst/agentx_conc$CONC.json"
if [[ -s "$result" ]]; then
    echo "AgentX passed: $result (finished by tune_finish.sh, client exit $code)" >> "$OUT/agentx.log"
else
    echo "AgentX result is missing: $result (client exit $code)" >> "$OUT/agentx.log"
fi
say "client exit $code, output copied back"
exec bash "$HERE/tune_post.sh" "VARIANT=$VARIANT" "CONC=$CONC" "ROUND=$ROUND" "AGENTX_RC=$code"

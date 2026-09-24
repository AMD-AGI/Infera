#!/usr/bin/env bash
# One completion and one streamed chat through the router, then the Mooncake
# transport lines of both engines. Usage: scripts/smoke.sh
source "$(dirname "$0")/common.sh" "$@"
set +e  # diagnostics: report every check even when one fails
url="http://$CONTROL_IP:$ROUTER_PORT"

curl -s "$url/v1/workers"; echo
curl -s "$url/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",
    \"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"; echo
curl -sN "$url/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Reply with the number only.\"}],
    \"max_tokens\":64,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true},
    \"chat_template_kwargs\":{\"enable_thinking\":false}}"; echo
for target in "$PREFILL_NODE:prefill" "$DECODE_NODE:decode"; do
    echo "== ${target#*:}"
    on "${target%%:*}" docker logs "$PREFIX-${target#*:}" 2>&1 |
        grep -E "TransferEngine initialized|Auto-selecting RDMA|matched rails" | sort -u | head -12
done

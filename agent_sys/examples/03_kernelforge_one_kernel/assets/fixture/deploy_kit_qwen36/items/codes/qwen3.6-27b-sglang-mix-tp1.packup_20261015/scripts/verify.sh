#!/usr/bin/env bash
# verify.sh <out-dir>
# Read the deployment back out of the RUNNING system (not the launch command)
# and prove it answers through the router. Writes into <out-dir>:
#   router_workers.json   GET  router /v1/workers     (router's reading of the mode)
#   router_models.json    GET  router /v1/models      (served name, not a path)
#   worker_mode_line.txt  the worker's own `worker ready:` line (its reading)
#   chat_completion.json  POST router /v1/chat/completions
#   engine_server_info.json  GET engine /get_server_info (argv read back)
#   verification.json     every criterion, each true/false, with the values
# Exits 0 only if every criterion holds. Expected mode defaults to `mixed`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
OUT="${1:?usage: verify.sh <out-dir>}"
: "${E2E_KIT_EXPECT_MODE:=mixed}"
mkdir -p "$OUT"

curl -s -m 20 "$KIT_ROUTER_URL/health" -o /dev/null -w '%{http_code}' > "$OUT/.router_health_code"
curl -s -m 20 "$KIT_ROUTER_URL/v1/workers" > "$OUT/router_workers.json"
curl -s -m 20 "$KIT_ROUTER_URL/v1/models"  > "$OUT/router_models.json"
grep -m1 'worker ready:' "$KIT_LOG_DIR/worker.log" > "$OUT/worker_mode_line.txt" || true
curl -s -m 60 "$KIT_ENGINE_URL/get_server_info" > "$OUT/engine_server_info.json" || true
curl -s -m 600 "$KIT_ROUTER_URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$E2E_KIT_SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer with one word.\"}],\"max_tokens\":2048,\"temperature\":0}" \
  > "$OUT/chat_completion.json"

OUT="$OUT" SERVED="$E2E_KIT_SERVED_NAME" MODE="$E2E_KIT_EXPECT_MODE" ROUTER="$KIT_ROUTER_URL" \
TAG="$E2E_KIT_RUN_TAG" GRAPH_MAX_BS="$E2E_KIT_CUDA_GRAPH_MAX_BS" python3 - <<'PY'
import json, os, re, sys, datetime
o = os.environ["OUT"]; served = os.environ["SERVED"]; mode = os.environ["MODE"]
def load(name):
    try:
        return json.load(open(os.path.join(o, name)))
    except Exception as exc:
        return {"_unreadable": str(exc)}
workers = load("router_workers.json"); models = load("router_models.json")
chat = load("chat_completion.json"); info = load("engine_server_info.json")
line = open(os.path.join(o, "worker_mode_line.txt")).read().strip()
ws = workers.get("workers", []) if isinstance(workers, dict) else workers
active = [w for w in ws if isinstance(w, dict) and w.get("status") == "active"]
ids = [m.get("id") for m in models.get("data", [])] if isinstance(models, dict) else []
m = re.search(r"disagg=DisaggMode\.(\w+)", line)
ch = (chat.get("choices") or [{}])[0] if isinstance(chat, dict) else {}
msg = ch.get("message") or {}
crit = {
    "router_health_200": open(os.path.join(o, ".router_health_code")).read().strip() == "200",
    "worker_active": len(active) >= 1,
    "router_reports_mode": bool(active) and all(w.get("disagg_mode") == mode for w in active),
    "worker_reports_mode": bool(m) and m.group(1).lower() == mode,
    "models_equal_served_name": bool(ids) and all(i == served for i in ids),
    "no_model_id_is_a_path": all(not str(i).startswith("/") for i in ids),
    "finish_reason_stop": ch.get("finish_reason") == "stop",
    "content_nonempty": bool((msg.get("content") or "").strip()),
    "response_model_is_served_name": chat.get("model") == served if isinstance(chat, dict) else False,
}
v = {
    "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_tag": os.environ["TAG"], "router": os.environ["ROUTER"],
    "expected_mode": mode, "served_model_name": served,
    "router_disagg_modes": [w.get("disagg_mode") for w in ws if isinstance(w, dict)],
    "worker_ready_line_mode": m.group(0) if m else None,
    "model_ids": ids,
    "finish_reason": ch.get("finish_reason"),
    "content": msg.get("content"),
    "reasoning_chars": len(msg.get("reasoning_content") or ""),
    "usage": chat.get("usage") if isinstance(chat, dict) else None,
    "engine_cuda_graph_max_bs_decode": info.get("cuda_graph_max_bs_decode") if isinstance(info, dict) else None,
    "engine_cuda_graph_backend_decode": info.get("cuda_graph_backend_decode") if isinstance(info, dict) else None,
    "engine_mem_fraction_static": info.get("mem_fraction_static") if isinstance(info, dict) else None,
    "engine_context_length": info.get("context_length") if isinstance(info, dict) else None,
    "engine_max_running_requests": info.get("max_running_requests") if isinstance(info, dict) else None,
    "engine_reasoning_parser": info.get("reasoning_parser") if isinstance(info, dict) else None,
    "kit_cuda_graph_max_bs": int(os.environ["GRAPH_MAX_BS"]),
    "criteria": crit,
    "all_pass": all(crit.values()),
}
json.dump(v, open(os.path.join(o, "verification.json"), "w"), indent=2)
print(json.dumps(v, indent=2))
sys.exit(0 if v["all_pass"] else 1)
PY
rc=$?
rm -f "$OUT/.router_health_code"
exit $rc

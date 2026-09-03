#!/usr/bin/env python3
"""A stand-in for the router+engine, shaped like the real one's answers.

Exists to exercise `check_deploy_serves`'s own logic on a node with no free GPU
and no engine image. It answers the eleven probes in `probes.yaml` the way the
sealed kit's recorded evidence says the real deployment answered them --
`results/router_workers.json`, `router_models.json`, `chat_completion.json`.
It is NOT a model and proves nothing about one.
"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("STUB_MODEL", "Qwen/Qwen3.6-27B")
CTX = int(os.environ.get("STUB_CTX", "32768"))
ROLE = os.environ.get("STUB_ROLE", "router")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        raw = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            return self._send(200, json.dumps({"status": "ok", "active_workers": 1}))
        if p == "/v1/workers":
            return self._send(200, json.dumps({"workers": [{
                "worker_id": "127.0.0.1:9", "url": "http://127.0.0.1:9",
                "model_name": MODEL, "engine": "sglang", "status": "active",
                "disagg_mode": "mixed", "disagg_meta": {}}]}))
        if p == "/v1/models":
            return self._send(200, json.dumps({"object": "list", "data": [
                {"id": MODEL, "object": "model", "owned_by": "infera"}]}))
        if p == "/get_model_info":
            return self._send(200, json.dumps({"model_path": "/models/stub", "is_generation": True}))
        if p == "/get_server_info":
            return self._send(200, json.dumps({"tp_size": 1, "max_total_num_tokens": CTX}))
        if p == "/metrics":
            return self._send(200,
                "# HELP sglang:num_queue_reqs queued\n# TYPE sglang:num_queue_reqs gauge\n"
                "sglang:num_queue_reqs 0\n", "text/plain")
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        p = self.path.split("?")[0]
        if p == "/health_generate":
            return self._send(200, json.dumps({"status": "ok"}))
        if p != "/v1/chat/completions":
            return self._send(404, json.dumps({"error": "not found"}))

        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        text = "".join(m.get("content", "") for m in req.get("messages", []))
        # Roughly four characters per token, the same estimate the validator
        # builds its oversize prompt with.
        over = len(text) / 4 > CTX

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if over:
                # The researched divergence: SGLang answers a streaming context
                # overflow with HTTP 200 carrying an error payload.
                self.wfile.write(b'data: {"error":{"code":400,"message":"context overflow"}}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                return

            # **A full OpenAI chunk, not a `delta` in a bare envelope.** AIPerf's
            # chat parser needs `id` / `object` / `created` / `model` / `index`,
            # and a first chunk carrying the assistant role; without them every
            # request is counted as an error and the load collects zero
            # successful responses — measured, 80 036 of them.
            #
            # The length comes from the request rather than being fixed, because
            # `judge_load`'s floor is a *mean output length*: a stub that always
            # answered three tokens would fail an honest load for the wrong
            # reason. `min_tokens` is what AIPerf sends alongside `ignore_eos` to
            # pin the length, so it wins over `max_tokens`.
            want = int(req.get("min_tokens") or req.get("max_tokens") or 16)
            ident = f"chatcmpl-stub-{int(time.time()*1000)}"
            created = int(time.time())

            def chunk(delta, finish=None):
                return b"data: " + json.dumps({
                    "id": ident, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }).encode() + b"\n\n"

            try:
                self.wfile.write(chunk({"role": "assistant", "content": ""}))
                for i in range(want):
                    self.wfile.write(chunk({"content": f" t{i}"}))
                    self.wfile.flush()
                self.wfile.write(chunk({}, "stop"))
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # a client that hung up mid-stream is its own business
            return

        if over:
            return self._send(400, json.dumps({"object": "error", "message": "context overflow"}))
        return self._send(200, json.dumps({"model": MODEL, "choices": [{"index": 0,
            "finish_reason": "stop", "message": {"role": "assistant", "content": "Paris"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1}}))

port = int(sys.argv[1])
print(f"stub {ROLE} listening on {port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()

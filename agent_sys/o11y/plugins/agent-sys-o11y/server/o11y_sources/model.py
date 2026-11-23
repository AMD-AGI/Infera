"""Model call via `claude -p --bare --json-schema --tools ''` (research: oss.md §3, otel.md).

Gateway notes (EXPERIMENT 2026-11-23): `haiku` resolves to Haiku 4.5 through the gateway; the result
arrives in `structured_output` of the final `result` message; `is_error` must be checked. Thinking is
disabled by default (CLAUDE_CODE_DISABLE_THINKING=1) because it only adds cost for extraction.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass

# The child gets only what authentication, transport and the user's normal telemetry export need
# (user ruling: the analyzer's own calls keep the user's export). Everything else is dropped, so nothing from
# the analyzer's environment reaches the model process.
ENV_KEEP = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TZ",
    "CLAUDE_CONFIG_DIR",
    "NODE_EXTRA_CA_CERTS",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
}
ENV_KEEP_PREFIXES = (
    "ANTHROPIC_",
    "OTEL_",
    "CLAUDE_CODE_ENABLE_TELEMETRY",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA",
    "CLAUDE_CODE_USE_",
    "AWS_",
    "GOOGLE_",
    "CLOUD_ML_",
    "VERTEX_",
)


def child_env(environ, disable_thinking: bool = True) -> dict:
    env = {k: v for k, v in environ.items() if k in ENV_KEEP or k.startswith(ENV_KEEP_PREFIXES)}
    if disable_thinking:
        env["CLAUDE_CODE_DISABLE_THINKING"] = "1"
    env["AGENT_SYS_O11Y"] = "off"  # never start nested o11y machinery
    return env


@dataclass
class LlmResult:
    ok: bool
    data: dict | None
    cost_usd: float
    duration_s: float
    model: str | None
    error: str | None = None
    usage: dict | None = None


def run(
    prompt: str,
    schema: dict,
    *,
    model: str,
    effort: str | None,
    claude_bin: str = "claude",
    timeout_s: float = 240,
    disable_thinking: bool = True,
    cwd: str | None = None,
) -> LlmResult:
    argv = [
        claude_bin,
        "-p",
        "--bare",
        "--output-format",
        "json",
        "--model",
        model,
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--no-session-persistence",
        "--tools",
        "",  # no tools at all: only the built-in StructuredOutput remains (measured 2.1.325)
        "--permission-mode",
        "dontAsk",
    ]
    if effort:
        argv += ["--effort", effort]
    env = child_env(os.environ, disable_thinking)
    t0 = time.time()
    try:
        p = subprocess.run(
            argv, input=prompt, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=cwd
        )
    except subprocess.TimeoutExpired:
        return LlmResult(False, None, 0.0, time.time() - t0, model, "timeout")
    except FileNotFoundError as e:
        return LlmResult(False, None, 0.0, time.time() - t0, model, f"claude not found: {e}")
    dur = time.time() - t0
    try:
        out = json.loads(p.stdout)
    except ValueError:
        return LlmResult(
            False,
            None,
            0.0,
            dur,
            model,
            f"rc={p.returncode} non-JSON output: {p.stdout[:200]} {p.stderr[:200]}",
        )
    msgs = out if isinstance(out, list) else [out]
    res = next(
        (m for m in reversed(msgs) if isinstance(m, dict) and m.get("type") == "result"), None
    )
    if res is None:
        return LlmResult(False, None, 0.0, dur, model, f"no result message (rc={p.returncode})")
    cost = float(res.get("total_cost_usd") or 0.0)
    mu = res.get("modelUsage") or {}
    used_model = next(iter(mu), None)
    if res.get("is_error") or res.get("subtype") != "success":
        return LlmResult(
            False,
            None,
            cost,
            dur,
            used_model,
            f"{res.get('subtype')}: {str(res.get('result'))[:300]}",
            res.get("usage"),
        )
    data = res.get("structured_output")
    if not isinstance(data, dict):
        return LlmResult(
            False, None, cost, dur, used_model, "no structured_output", res.get("usage")
        )
    return LlmResult(True, data, cost, dur, used_model, None, res.get("usage"))

#!/usr/bin/env python3
"""E7 PD correctness smoke: the same greedy prompts through the PD router and a colocated reference, then diff.

Python stdlib only (runs on the node host or inside the vLLM image, `python3 -I`). Every HTTP call has a timeout.
Called by scripts/kit/pd_smoke.sh; see that file for the full procedure.

Prompts (fixed, deterministic):
  p1-short   short factual chat question
  p2-code    medium code question
  p3-long    > 8192 prompt tokens (synthetic key/value records + one lookup question); the PD answer needs the
             transferred KV of the whole prompt
  p3-repeat  p3-long again: P now has a local prefix hit, D (prefix cache off) still pulls everything
Metric deltas (Prometheus text from P and D /metrics), names from vLLM main 7a62ca06:
  vllm:mooncake_* (V/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/stats.py:200-260; histograms filled on P only),
  vllm:external_prefix_cache_hits, vllm:prompt_tokens_cached_by_source{source=p2p} (V/vllm/v1/metrics/loggers.py:700-805).
Exit: 0 all outputs identical + transfer evidence + no failures; 1 otherwise (inspect); 2 usage / unreachable.
Note: an exact-text mismatch is NOT proof of a KV bug (batch shape and kernel numerics differ between servers); the
first-divergence offset and the long-prompt answer are reported to judge it.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    ("p1-short", "What is the capital of France? Answer in one short sentence."),
    ("p2-code", "Write a Python function that returns the n-th Fibonacci number iteratively, then explain its time "
                "complexity in two sentences."),
]


def long_prompt(n_records=900, ask=421):
    # Deterministic pseudo-random 6-char codes (LCG), ~12-14 tokens per line -> > 8192 tokens at 900 lines.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    x = 12345
    lines = []
    want = None
    for i in range(1, n_records + 1):
        code = ""
        for _ in range(6):
            x = (1103515245 * x + 12345) % (2 ** 31)
            code += alphabet[x % len(alphabet)]
        lines.append(f"Record {i:04d}: item-{i:04d} has access code {code}.")
        if i == ask:
            want = code
    text = ("Below is a list of records.\n" + "\n".join(lines) +
            f"\n\nQuestion: what is the access code of item-{ask:04d}? Reply with the code only.")
    return text, want


def http(method, url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw, time.time() - t0


def scrape(url, timeout):
    if not url:
        return None
    try:
        raw, _ = http("GET", url, timeout=timeout)
        return raw.decode(errors="replace")
    except Exception as e:  # noqa: BLE001 - report, do not abort the smoke
        print(f"WARN metrics {url}: {e}", flush=True)
        return None


LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)')


def metric_sum(text, name, label=None):
    """Sum every sample of `name` (exact) whose label set contains `label` ('k="v"'); None if absent."""
    if text is None:
        return None
    tot, seen = 0.0, False
    for ln in text.splitlines():
        m = LINE.match(ln)
        if not m or m.group(1) != name:
            continue
        if label and label not in (m.group(2) or ""):
            continue
        tot += float(m.group(3))
        seen = True
    return tot if seen else None


METRICS = [  # (role, display key, metric name, label filter)
    ("P", "mooncake_xfers", "vllm:mooncake_xfer_time_seconds_count", None),
    ("P", "mooncake_bytes", "vllm:mooncake_bytes_transferred_sum", None),
    ("P", "mooncake_failed_transfers", "vllm:mooncake_num_failed_transfers_total", None),
    ("P", "mooncake_kv_expired", "vllm:mooncake_num_kv_expired_reqs_total", None),
    ("P", "prompt_tokens", "vllm:prompt_tokens_total", None),
    ("D", "mooncake_failed_recvs", "vllm:mooncake_num_failed_recvs_total", None),
    ("D", "external_prefix_cache_hits", "vllm:external_prefix_cache_hits_total", None),
    ("D", "p2p_cached_prompt_tokens", "vllm:prompt_tokens_cached_by_source_total", 'source="p2p"'),
    ("D", "prompt_tokens", "vllm:prompt_tokens_total", None),
]


def complete(base, model, prompt, max_tokens, endpoint, timeout):
    if endpoint == "chat":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                "top_p": 1.0, "seed": 0, "max_tokens": max_tokens, "stream": False}
        url = base.rstrip("/") + "/v1/chat/completions"
    else:
        body = {"model": model, "prompt": prompt, "temperature": 0.0, "top_p": 1.0, "seed": 0,
                "max_tokens": max_tokens, "stream": False}
        url = base.rstrip("/") + "/v1/completions"
    try:
        raw, dt = http("POST", url, body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    try:
        j = json.loads(raw)
        ch = j["choices"][0]
        if endpoint == "chat":
            msg = ch.get("message") or {}
            text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\n<<content>>\n" + (msg.get("content") or "")
        else:
            text = ch.get("text") or ""
        return {"text": text, "finish_reason": ch.get("finish_reason"), "usage": j.get("usage"), "latency_s": round(dt, 3),
                "kv_transfer_params": j.get("kv_transfer_params")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"bad response ({e}): {raw[:500]!r}"}


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd", required=True, help="PD router base URL")
    ap.add_argument("--ref", required=True, help="colocated reference base URL (router or engine)")
    ap.add_argument("--metrics-p", default="")
    ap.add_argument("--metrics-d", default="")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=900, help="per request, seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-long-tokens", type=int, default=8193)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for name, url in (("pd", a.pd), ("ref", a.ref)):
        try:
            _, dt = http("GET", url.rstrip("/") + "/health", timeout=10)
            print(f"health {name} {url}: ok ({dt:.2f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR health {name} {url}: {e}", flush=True)
            return 2

    lp, want = long_prompt()
    prompts = PROMPTS + [("p3-long", lp), ("p3-repeat", lp)]
    before = {"P": scrape(a.metrics_p, 10), "D": scrape(a.metrics_d, 10)}
    rows, ok = [], True
    for pid, text in prompts:
        r_ref = complete(a.ref, a.model, text, a.max_tokens, a.endpoint, a.timeout)
        r_pd = complete(a.pd, a.model, text, a.max_tokens, a.endpoint, a.timeout)
        row = {"id": pid, "prompt_chars": len(text), "ref": r_ref, "pd": r_pd}
        if "error" in r_ref or "error" in r_pd:
            row["verdict"] = "ERROR"
            ok = False
        else:
            d = first_div(r_ref["text"], r_pd["text"])
            row["identical"] = d is None
            row["first_divergence_char"] = d
            row["prompt_tokens_pd"] = (r_pd.get("usage") or {}).get("prompt_tokens")
            row["verdict"] = "IDENTICAL" if d is None else "MISMATCH"
            ok = ok and d is None
            if pid.startswith("p3"):
                row["expected_code"] = want
                row["ref_has_code"] = want in r_ref["text"]
                row["pd_has_code"] = want in r_pd["text"]
                pt = row["prompt_tokens_pd"]
                if pt is not None and pt < a.min_long_tokens:
                    row["verdict"] += f" (WARN long prompt only {pt} tokens)"
                    ok = False
        for side in ("ref", "pd"):
            with open(os.path.join(a.out, f"{pid}.{side}.txt"), "w") as f:
                f.write((row[side].get("text") if "text" in row[side] else row[side].get("error", "")) or "")
        rows.append(row)
        print(f"{pid}: {row['verdict']}  first_div={row.get('first_divergence_char')} "
              f"pd_prompt_tokens={row.get('prompt_tokens_pd')} pd_latency={r_pd.get('latency_s')} "
              f"ref_latency={r_ref.get('latency_s')}"
              + (f" code_ref={row['ref_has_code']} code_pd={row['pd_has_code']}" if "expected_code" in row else ""),
              flush=True)
    time.sleep(2)
    after = {"P": scrape(a.metrics_p, 10), "D": scrape(a.metrics_d, 10)}
    deltas = {}
    for role, key, name, label in METRICS:
        b, e = metric_sum(before[role], name, label), metric_sum(after[role], name, label)
        deltas[f"{role}.{key}"] = None if (b is None or e is None) else e - b
    print("metric deltas:", json.dumps(deltas), flush=True)
    xfers = deltas.get("P.mooncake_xfers")
    bad = [k for k in ("P.mooncake_failed_transfers", "P.mooncake_kv_expired", "D.mooncake_failed_recvs")
           if (deltas.get(k) or 0) > 0]
    if not xfers:
        print("WARN no Mooncake transfer counted on P (metric absent or 0): KV may not have moved", flush=True)
        ok = False
    if bad:
        print("WARN failure counters moved:", bad, flush=True)
        ok = False
    with open(os.path.join(a.out, "smoke.json"), "w") as f:
        json.dump({"pd": a.pd, "ref": a.ref, "endpoint": a.endpoint, "max_tokens": a.max_tokens, "rows": rows,
                   "metric_deltas": deltas, "pass": ok}, f, indent=1)
    print(("PASS" if ok else "CHECK") + f": details {a.out}/smoke.json", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

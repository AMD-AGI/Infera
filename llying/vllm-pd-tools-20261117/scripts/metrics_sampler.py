#!/usr/bin/env python3
"""Sample vLLM /metrics of the PD engines every INTERVAL s into JSONL (one line per sample and engine URL), until
STOP_FILE exists or DURATION s pass.
  metrics_sampler.py OUT.jsonl INTERVAL DURATION URL [URL...]
Kept: every vllm: gauge, and the counters used for PD analysis (tokens, prefix / external hits, spec decode, preemptions,
mooncake transfer sums / counts), per label set (the engine label is the DP rank).
"""
import json
import os
import re
import sys
import time
import urllib.request

LINE = re.compile(r"^(vllm:[a-zA-Z0-9_:]+)(\{[^}]*\})? ([0-9.eE+-]+|NaN)$")
COUNTERS = re.compile(
    r"(prompt_tokens|generation_tokens|prefix_cache_(hits|queries)|external_prefix_cache_(hits|queries)"
    r"|spec_decode_num_(drafts|accepted_tokens)|num_preemptions|request_success|mooncake_xfer_time_seconds_(sum|count)"
    r"|mooncake_bytes_transferred_(sum|count)|kv_offload_.*_(sum|count)|request_prefill_time_seconds_(sum|count)"
    r"|request_queue_time_seconds_(sum|count)|time_to_first_token_seconds_(sum|count))(_total)?$"
)
TYPES = re.compile(r"^# TYPE (vllm:\S+) (\w+)")


def scrape(url):
    out, types = {}, {}
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=15) as r:
        for line in r.read().decode().splitlines():
            t = TYPES.match(line)
            if t:
                types[t.group(1)] = t.group(2)
                continue
            m = LINE.match(line)
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2) or "", m.group(3)
            base = re.sub(r"_(total|sum|count|bucket)$", "", name)
            if types.get(base) == "gauge" or types.get(name) == "gauge" or COUNTERS.search(name):
                eng = re.search(r'engine="([^"]+)"', labels)
                extra = re.sub(r'(engine|model_name)="[^"]*",?', "", labels).strip("{},")
                key = name + (f"[{extra}]" if extra else "")
                out.setdefault(eng.group(1) if eng else "-", {})[key] = float(val)
    return out


def main():
    out, interval, duration = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    urls = sys.argv[4:]
    stop = os.environ.get("STOP_FILE", out + ".stop")
    t_end = time.time() + duration
    with open(out, "a") as f:
        while time.time() < t_end and not os.path.exists(stop):
            t = time.time()
            for u in urls:
                try:
                    f.write(json.dumps({"t": round(t, 1), "url": u, "m": scrape(u)}) + "\n")
                except Exception as e:  # an engine restarting or busy; keep sampling
                    f.write(json.dumps({"t": round(t, 1), "url": u, "error": str(e)[:200]}) + "\n")
            f.flush()
            time.sleep(max(0.0, interval - (time.time() - t)))


if __name__ == "__main__":
    main()

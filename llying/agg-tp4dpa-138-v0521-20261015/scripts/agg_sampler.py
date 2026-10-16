#!/usr/bin/env python3
"""Every INTERVAL seconds, append one JSON line per worker with its scheduler gauges and token
counters (per DP rank, labels kept), for agg_report.py's window averages.

Usage: agg_sampler.py OUT_JSONL INTERVAL URL [URL ...]   (runs until killed)
"""

import json
import re
import sys
import time
import urllib.request

NAMES = ("num_running_reqs", "num_queue_reqs", "token_usage", "gen_throughput", "cache_hit_rate",
         "spec_accept_length", "num_used_tokens", "max_total_num_tokens", "num_retracted_reqs",
         "prompt_tokens_total", "generation_tokens_total", "cached_tokens_total", "realtime_tokens_total",
         "num_requests_total", "hicache_[a-z_]+", "evicted_tokens_total", "utilization", "fwd_occupancy",
         "kv_used_tokens", "kv_evictable_tokens", "scheduler_idle_seconds_total", "spec_accept_rate",
         "prefill_delayer_outcomes_total", "prefill_delayer_wait_seconds_(?:sum|count)",
         "prefill_delayer_wait_forward_passes_(?:bucket|sum|count)",
         "process_cpu_seconds_total", "scheduler_process_cpu_seconds_total")
LINE = re.compile(r"^sglang:(" + "|".join(NAMES) + r")(\{[^}]*\})? ([0-9.eE+-]+|NaN)$")


def scrape(url: str) -> dict:
    text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m and m.group(3) != "NaN":
            labels = re.sub(r'(model_name|engine_type|moe_ep_rank|pp_rank|tp_rank)="[^"]*",?', "", m.group(2) or "")
            out[m.group(1) + labels.replace(",}", "}").replace("{}", "")] = float(m.group(3))
    return out


def main() -> int:
    path, interval, urls = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    pending = []
    while True:
        t = time.time()
        for url in urls:
            try:
                pending.append({"t": t, "url": url, "m": scrape(url)})
            except Exception as exc:  # a scrape miss is a gap, not a reason to stop
                pending.append({"t": t, "url": url, "error": str(exc)[:200]})
        try:
            with open(path, "a") as f:
                f.writelines(json.dumps(p) + "\n" for p in pending)
            pending = []
        except OSError:
            pending = pending[-1000:]
        time.sleep(max(0.0, interval - (time.time() - t)))


if __name__ == "__main__":
    sys.exit(main())

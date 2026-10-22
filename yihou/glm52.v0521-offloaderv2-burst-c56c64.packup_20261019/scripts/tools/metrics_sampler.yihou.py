#!/usr/bin/env python3
"""Poll sglang /metrics of the prefill and decode legs every INTERVAL s and append selected per-DP gauges as TSV.

Usage: metrics_sampler.yihou.py OUT.tsv DURATION_S INTERVAL_S leg=http://ip:port [leg=http://ip:port ...]
Columns: unix_ts  leg  dp_rank  metric  value   (one row per metric per DP rank per sample)
Only the metric names in KEEP are written (counters included where useful for rate computation).
"""
import sys, time, re, urllib.request

KEEP = {
    "num_running_reqs", "num_queue_reqs", "num_decode_prealloc_queue_reqs", "num_decode_transfer_queue_reqs",
    "num_decode_host_receive_queue_reqs", "num_decode_deferred_kv_release_reqs", "kv_used_tokens",
    "kv_available_tokens", "kv_evictable_tokens", "decode_sum_seq_lens", "token_usage", "full_token_usage",
    "pending_prealloc_token_usage", "num_retracted_reqs", "num_paused_reqs", "num_prefill_bootstrap_queue_reqs",
    "num_prefill_inflight_queue_reqs", "hicache_host_used_tokens", "hicache_host_total_tokens", "cache_hit_rate",
    "gen_throughput", "num_used_tokens", "num_transfer_failed_reqs_total", "num_bootstrap_failed_reqs_total",
    "new_token_ratio", "max_total_num_tokens", "spec_accept_length", "evicted_tokens_total",
    "load_back_tokens_total", "hicache_backup_tokens_total",
}
LINE = re.compile(r'^sglang:([a-z_]+)(\{[^}]*\})?\s+(\S+)$')
DP = re.compile(r'dp_rank="(\d+)"')


def main():
    out, dur, itv = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    legs = [a.split("=", 1) for a in sys.argv[4:]]
    t_end = time.time() + dur
    with open(out, "a", buffering=1) as f:
        while time.time() < t_end:
            t0 = time.time()
            for leg, url in legs:
                try:
                    body = urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=4).read().decode("utf-8", "replace")
                except Exception as e:  # keep sampling through transient errors
                    f.write(f"{t0:.1f}\t{leg}\t-\t_error\t{type(e).__name__}\n")
                    continue
                seen = set()
                for ln in body.splitlines():
                    m = LINE.match(ln)
                    if not m or m.group(1) not in KEEP:
                        continue
                    labels = m.group(2) or ""
                    d = DP.search(labels)
                    key = (d.group(1) if d else "-", m.group(1))
                    if key in seen:  # first series per (dp, metric) only
                        continue
                    seen.add(key)
                    f.write(f"{t0:.1f}\t{leg}\t{key[0]}\t{key[1]}\t{m.group(3)}\n")
            time.sleep(max(0.0, itv - (time.time() - t0)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Snapshot vLLM counters from one or more /metrics URLs, or diff two snapshots.
  metrics_snap.py snap OUT.json URL [URL...]   sums each counter over all label sets (DP ranks / engines)
  metrics_snap.py diff A.json B.json           prints the deltas used by the gates:
      accept_len = 1 + accepted / drafts (mean tokens per verify step and request), per-position acceptance,
      prefix cache hit rate, generation / prompt tokens
"""
import json
import re
import sys
import urllib.request

KEEP = re.compile(
    r"^(vllm:(spec_decode_num_drafts|spec_decode_num_draft_tokens|spec_decode_num_accepted_tokens"
    r"|spec_decode_num_accepted_tokens_per_pos|prefix_cache_queries|prefix_cache_hits|generation_tokens"
    r"|prompt_tokens|request_success|num_preemptions)(_total)?)(\{[^}]*\})? ([0-9.eE+-]+)$"
)


def scrape(url):
    out = {}
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=20) as r:
        for line in r.read().decode().splitlines():
            m = KEEP.match(line)
            if not m:
                continue
            key = m.group(1).replace("_total", "")
            if "per_pos" in key:
                pos = re.search(r'position="(\d+)"', m.group(4) or "")
                key += f"[{pos.group(1) if pos else '?'}]"
            out[key] = out.get(key, 0.0) + float(m.group(5))
    return out


def main():
    if sys.argv[1] == "snap":
        total = {}
        for url in sys.argv[3:]:
            for k, v in scrape(url).items():
                total[k] = total.get(k, 0.0) + v
        json.dump(total, open(sys.argv[2], "w"), indent=1, sort_keys=True)
        return
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b}
    drafts = d.get("vllm:spec_decode_num_drafts", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens", 0)
    res = {
        "drafts": int(drafts),
        "accept_len": round(1 + acc / drafts, 3) if drafts else None,
        "per_pos": [round(d[k] / drafts, 3) for k in sorted(k for k in d if "per_pos" in k)] if drafts else [],
        "prefix_hit": round(d["vllm:prefix_cache_hits"] / d["vllm:prefix_cache_queries"], 4)
        if d.get("vllm:prefix_cache_queries") else None,
        "gen_tokens": int(d.get("vllm:generation_tokens", 0)),
        "prompt_tokens": int(d.get("vllm:prompt_tokens", 0)),
        "preemptions": int(d.get("vllm:num_preemptions", 0)),
    }
    print(json.dumps(res))


if __name__ == "__main__":
    main()

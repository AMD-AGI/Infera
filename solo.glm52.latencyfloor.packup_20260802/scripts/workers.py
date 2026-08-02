#!/usr/bin/env python3
"""Print the router's worker registry, one line per worker, and verify PD.

A FILE, not an inline heredoc: nested quoting through two SSH hops plus
`docker exec` mangles inline python (feedback_nested_ssh_quoting).
"""
import json
import sys

d = json.load(sys.stdin)["workers"]
for w in d:
    print(
        f"  {w['worker_id']:24s} {w['disagg_mode']:8s} {w['status']:8s} "
        f"dp_size={w['dp_size']} kv_events={w.get('kv_events_endpoint')}"
    )
modes = {w["disagg_mode"] for w in d if w["status"] == "active"}
ok = modes >= {"prefill", "decode"}
print(f"  -> {'OK' if ok else 'FAIL'} (want both prefill and decode active)")
sys.exit(0 if ok else 1)

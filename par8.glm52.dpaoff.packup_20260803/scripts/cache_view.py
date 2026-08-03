#!/usr/bin/env python3
"""kv-aware discriminator for the RUST router: per-DP-rank pick distribution.

The Rust router routes only /health, /v1/workers, /v1/models, /metrics and the
completion endpoints (rust/router/src/handlers.rs:33-38) -- there is no
/v1/admin/cache-view, and total_blocks() is unrouted. So the only per-rank
kv-aware signal is the policy log line (rust/router/src/policy.rs:314).

Reads /tmp/router.log, strips ANSI, and reports per role:
  * which DP ranks were picked, and how often  -> proves per-rank routing
  * the cache_hits distribution                 -> proves the KV view is populated

A run where every pick is #dp0 with cache_hits=0 means kv-aware is degrading to
load balancing -- which is exactly what the bigram bug looked like.

    cache_view.py [logfile] [--since-line N]
"""
import re
import sys
from collections import Counter, defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/router.log"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PICK = re.compile(
    r'pick .*?role=(?P<role>\w+).*?picked=(?P<picked>\S+).*?'
    r'cache_hits=(?P<hits>\d+).*?request_blocks=(?P<blocks>\d+).*?'
    r'active_blocks=(?P<active>\d+).*?w_overlap=(?P<w>[\d.]+)'
)

picks = defaultdict(Counter)
hits = defaultdict(list)
blocks = defaultdict(list)
weights = {}
n = 0
with open(LOG, errors="replace") as f:
    for line in f:
        m = PICK.search(ANSI.sub("", line))
        if not m:
            continue
        n += 1
        role = m.group("role")
        picks[role][m.group("picked")] += 1
        hits[role].append(int(m.group("hits")))
        blocks[role].append(int(m.group("blocks")))
        weights[role] = m.group("w")

if not n:
    print(f"  no policy pick lines in {LOG} -- drive traffic first")
    sys.exit(1)

print(f"  {n} pick decisions in {LOG}")
for role in sorted(picks):
    ranks = picks[role]
    h, b = hits[role], blocks[role]
    print(f"\n  === {role}  (w_overlap={weights[role]}) ===")
    print(f"  distinct targets picked: {len(ranks)}")
    for target, c in sorted(ranks.items()):
        print(f"    {target:34s} {c:5d} picks")
    nz = sum(1 for x in h if x > 0)
    print(f"  cache_hits: max={max(h)} mean={sum(h)/len(h):.1f} nonzero={nz}/{len(h)}")
    print(f"  request_blocks: max={max(b)} mean={sum(b)/len(b):.1f}")

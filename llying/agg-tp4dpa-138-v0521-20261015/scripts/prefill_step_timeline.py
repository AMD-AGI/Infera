#!/usr/bin/env python3
"""Prefill step timeline of an aggregated DP-attention round from the server log.

With EAGLE + DP attention (no coordination), any rank's prefill turns the step into an eager
step and the other ranks run IDLE batches. The prefill delayer holds a rank's new requests
unless every rank can prefill ("all"), but a rank in the middle of a chunked prefill proceeds
regardless. This script groups the per-rank "Prefill batch" log lines into steps (lines closer
than --gap seconds belong to one step) and reports:
  - how many ranks prefill in each step, and how many of those steps are chunk continuations;
  - steps where a non-prefilling rank had queued requests (last #queue-req it logged > 0), i.e.
    requests that could have prefilled in a step that was eager anyway;
  - per-rank prefill steps and new tokens.

Usage: prefill_step_timeline.py RESULT_DIR SERVER_LOG [--gap 0.03] [--chunk 8192]
RESULT_DIR supplies the measured window (last DURATION seconds of samples.jsonl).
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LINE = re.compile(r"^(\S+)Z \[[^\]]* DP(\d+) TP\d+\] (Prefill|Decode) batch, (.*)$")
FIELD = re.compile(r"#([a-z-]+): ([\d.]+)")


def window(result_dir: Path) -> tuple[float, float]:
    env = dict(x.split("=", 1) for x in (result_dir / "round.env").read_text().splitlines() if "=" in x)
    rows = [json.loads(x) for x in (result_dir / "samples.jsonl").read_text().splitlines() if x.strip()]
    end = [r for r in rows if "m" in r][-1]["t"]
    return end - float(env.get("DURATION") or 3600), end


def parse(log: Path, start: float, end: float):
    events = []
    for line in log.open(errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1)[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        fields = {k: float(v) for k, v in FIELD.findall(m.group(4))}
        events.append((t, int(m.group(2)), m.group(3), fields))
    events.sort(key=lambda e: e[0])
    return [e for e in events if start - 600 <= e[0] <= end], start, end


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("server_log", type=Path)
    ap.add_argument("--gap", type=float, default=0.03)
    ap.add_argument("--chunk", type=int, default=8192, help="per-rank prefill budget (full chunk)")
    args = ap.parse_args()
    start, end = window(args.result_dir)
    events, start, end = parse(args.server_log, start, end)

    queue = {}            # last #queue-req logged by each rank
    last_prefill = {}     # rank -> (step index, new tokens) of its previous prefill
    steps = []
    current = None
    for t, rank, kind, f in events:
        if kind == "Decode":
            queue[rank] = f.get("queue-req", 0)
            continue
        if current is None or t - current["t_last"] > args.gap:
            if current is not None:
                steps.append(current)
            current = {"t": t, "t_last": t, "ranks": {}, "queue_before": dict(queue)}
        current["t_last"] = t
        current["ranks"][rank] = f
        queue[rank] = f.get("queue-req", 0)
    if current is not None:
        steps.append(current)
    steps = [s for s in steps if start <= s["t"] <= end]
    ranks = sorted({r for s in steps for r in s["ranks"]} | set(queue))

    by_count = Counter()
    continuation = Counter()
    missed_steps = Counter()
    missed_slots = 0
    missed_slot_queue = []
    per_rank = defaultdict(lambda: {"steps": 0, "new": 0.0, "full": 0, "alone": 0})
    for i, s in enumerate(steps):
        n = len(s["ranks"])
        by_count[n] += 1
        cont = []
        for r, f in s["ranks"].items():
            new = f.get("new-token", 0)
            p = per_rank[r]
            p["steps"] += 1
            p["new"] += new
            p["full"] += new >= args.chunk
            p["alone"] += n == 1
            prev = last_prefill.get(r)
            cont.append(prev is not None and prev[0] == i - 1 and prev[1] >= args.chunk)
            last_prefill[r] = (i, new)
        if all(cont):
            continuation[n] += 1
        idle_with_queue = [r for r in ranks if r not in s["ranks"] and s["queue_before"].get(r, 0) > 0]
        if idle_with_queue:
            missed_steps[n] += 1
            missed_slots += len(idle_with_queue)
            missed_slot_queue += [s["queue_before"][r] for r in idle_with_queue]

    total = len(steps)
    span = end - start
    print(f"window {datetime.fromtimestamp(start, timezone.utc):%Y-%m-%d %H:%M:%S} -> "
          f"{datetime.fromtimestamp(end, timezone.utc):%H:%M:%S} UTC ({span / 60:.0f} min), "
          f"{len(ranks)} ranks, step gap {args.gap * 1000:.0f} ms")
    print(f"prefill (eager) steps: {total} ({total / span * 60:.1f}/min), "
          f"rank-prefills {sum(len(s['ranks']) for s in steps)} "
          f"(mean {sum(len(s['ranks']) for s in steps) / max(total, 1):.2f} ranks per step)")
    print("ranks prefilling per step:  " + "  ".join(
        f"{n}: {c} ({100 * c / total:.1f}%)" for n, c in sorted(by_count.items())))
    print("  of which every prefilling rank continues a full chunk from the previous step:  " + "  ".join(
        f"{n}: {continuation[n]}" for n in sorted(by_count)))
    print(f"steps where >= 1 non-prefilling rank had queued requests: {sum(missed_steps.values())} "
          f"({100 * sum(missed_steps.values()) / max(total, 1):.1f}% of eager steps); by ranks prefilling: "
          + "  ".join(f"{n}: {missed_steps[n]}" for n in sorted(by_count)))
    if missed_slots:
        q = sorted(missed_slot_queue)
        print(f"  idle rank-slots with queued requests: {missed_slots} "
              f"(queue at that time p50 {q[len(q) // 2]:.0f}, p90 {q[int(len(q) * 0.9)]:.0f})")
    single = by_count[1]
    print(f"single-rank steps: {single} ({100 * single / max(total, 1):.1f}%), "
          f"of which another rank had queued requests: {missed_steps[1]}")
    print("per rank: " + "  ".join(
        f"DP{r}: {p['steps']} steps, {p['new'] / 1e6:.2f}M new tok, full chunks {p['full']}, alone {p['alone']}"
        for r, p in sorted(per_rank.items())))
    busiest = max(p["steps"] for p in per_rank.values())
    print(f"lower bound on eager steps if every step aligned all prefilling ranks: {busiest} "
          f"(busiest rank) vs {total} now; with prefill steps balanced across ranks: "
          f"{-(-sum(p['steps'] for p in per_rank.values()) // len(per_rank))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

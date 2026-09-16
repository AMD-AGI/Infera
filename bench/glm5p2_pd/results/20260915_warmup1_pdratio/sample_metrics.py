#!/usr/bin/env python3
"""Live /metrics sampler for one P:D-ratio point.

Same gauges and the same per-DP-rank reduction as the 137/138 sweep's sampler,
generalised to an arbitrary number of instances: this campaign's whole question
is what happens per prefill worker as workers are added, so each worker has to
be sampled separately rather than folded into one "prefill" row.

Gauges are per-DP-rank, so everything is summed over labels except rates and
ratios, which are averaged over the ranks that report a nonzero value.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request

SUM_GAUGES = (
    "num_running_reqs",
    "num_queue_reqs",
    "num_decode_prealloc_queue_reqs",
    "num_decode_transfer_queue_reqs",
    "kv_used_tokens",
    "kv_available_tokens",
    "http_requests_active",
)
AVG_GAUGES = ("token_usage", "cache_hit_rate", "gen_throughput")

SAMPLE_RE = re.compile(r"^sglang:(?P<name>[a-z_]+)\{[^}]*\}\s+(?P<value>[-0-9.eE+]+)$")


def scrape(url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    sums: dict[str, float] = {}
    avgs: dict[str, list[float]] = {}
    for line in body.splitlines():
        m = SAMPLE_RE.match(line)
        if m is None:
            continue
        name, raw = m.group("name"), m.group("value")
        try:
            value = float(raw)
        except ValueError:
            continue
        if name in SUM_GAUGES:
            sums[name] = sums.get(name, 0.0) + value
        elif name in AVG_GAUGES:
            avgs.setdefault(name, []).append(value)
    for name, values in avgs.items():
        nonzero = [v for v in values if v != 0.0]
        pool = nonzero or values
        sums[name] = sum(pool) / len(pool)
    return sums


def fmt(role: str, sample: dict[str, float] | str) -> str:
    if isinstance(sample, str):
        return f"{role}=<{sample}>"

    def g(name: str) -> float:
        return sample.get(name, 0.0)

    return (
        f"{role} run={g('num_running_reqs'):.0f} q={g('num_queue_reqs'):.0f} "
        f"prealloc={g('num_decode_prealloc_queue_reqs'):.0f} "
        f"xfer={g('num_decode_transfer_queue_reqs'):.0f} "
        f"http={g('http_requests_active'):.0f} "
        f"kv={g('token_usage'):.3f} hit={g('cache_hit_rate'):.3f} "
        f"gen={g('gen_throughput'):.0f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--endpoint",
        action="append",
        required=True,
        metavar="NAME=HOST:PORT",
        help="repeatable, e.g. --endpoint p0=10.245.153.247:19001",
    )
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    urls: dict[str, str] = {}
    for item in args.endpoint:
        name, _, hostport = item.partition("=")
        if not name or not hostport:
            ap.error(f"bad --endpoint {item!r}, want NAME=HOST:PORT")
        urls[name] = f"http://{hostport}/metrics"

    with open(args.out, "a", buffering=1) as fh:
        fh.write(
            f"# sampler start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"endpoints={','.join(urls)}\n"
        )
        while True:
            parts = []
            for role, url in urls.items():
                try:
                    parts.append(fmt(role, scrape(url, args.timeout)))
                except (urllib.error.URLError, OSError, TimeoutError) as exc:
                    parts.append(fmt(role, type(exc).__name__))
            fh.write(
                f"{time.strftime('%H:%M:%S', time.gmtime())} " + " | ".join(parts) + "\n"
            )
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())

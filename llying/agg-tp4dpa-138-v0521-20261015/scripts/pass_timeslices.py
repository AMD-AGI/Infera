#!/usr/bin/env python3
"""Decode step time and prefill share of an aggregated DP-attention run, from AIPerf 1 s timeslices.

Works on any AIPerf server_metrics_export.json (ours, or the InferenceX B300 Dynamo C48 artifact).
The file stores every Prometheus series with 1 s timeslices (0.8-1.8 GB); only the series needed
here are parsed, by seeking to each metric's key and decoding that object, so the file is never
loaded as a whole.

Per 1 s slice: decode passes D (all DP ranks step together) and prefill passes P_r per rank
(prefill_none + prefill_cuda_graph). Slices where no rank prefills give the pure decode step time
1 / D at that slice's running batch; the eager (prefill) share of wall time is then
1 - N_decode * t_decode / T over the profiling window.

Usage: pass_timeslices.py SERVER_METRICS_EXPORT_JSON [--endpoint-port PORT]
(PORT: the sglang worker metrics endpoint; default: the endpoint that reports cuda_graph_passes.)
"""

import argparse
import json
import mmap
import statistics
from collections import defaultdict
from datetime import datetime, timezone


def metric_object(mm: mmap.mmap, name: str, start: int = 0) -> tuple[dict, int]:
    key = f'"{name}": '.encode()
    pos = mm.find(key, start)
    if pos < 0:
        raise KeyError(name)
    obj, end = json.JSONDecoder().raw_decode(mm[pos + len(key):pos + len(key) + 64 * 1024 * 1024].decode())
    return obj, pos + len(key) + end


def slices(series: dict) -> dict[int, float]:
    # Counter and gauge slices start at different sub-second offsets; key both by whole second.
    return {round(s["start_ns"] / 1e9): s.get("total", s.get("avg", 0.0)) for s in series["timeslices"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_json")
    ap.add_argument("--endpoint-port", default=None, help="sglang worker metrics endpoint port")
    args = ap.parse_args()
    with open(args.export_json, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        head = json.loads(mm[:4096].decode(errors="ignore").split('"phase_time_ranges"')[0].rstrip(", \n") + "}}")
        passes, _ = metric_object(mm, "sglang:cuda_graph_passes")
        running, _ = metric_object(mm, "sglang:num_running_reqs")
    if args.endpoint_port is None:
        args.endpoint_port = passes["series"][0]["endpoint_url"].rsplit(":", 1)[1].split("/")[0]
    worker = [s for s in passes["series"] if s["endpoint_url"].endswith(f":{args.endpoint_port}/metrics")]
    decode = defaultdict(dict)
    prefill = defaultdict(lambda: defaultdict(float))
    for s in worker:
        rank, mode = s["labels"]["dp_rank"], s["labels"]["mode"]
        for t, v in slices(s).items():
            if mode == "decode_cuda_graph":
                decode[t][rank] = v
            elif mode.startswith("prefill"):
                prefill[t][rank] += v
    run = defaultdict(dict)
    for s in running["series"]:
        if s["endpoint_url"].endswith(f":{args.endpoint_port}/metrics"):
            for t, v in slices(s).items():
                run[t][s["labels"]["dp_rank"]] = v
    t0, t1 = (datetime.fromisoformat(head["summary"][k]).replace(tzinfo=timezone.utc).timestamp()
              for k in ("start_time", "end_time"))
    ts = sorted(t for t in decode if decode[t] and t0 <= t < t1)
    ranks = sorted({r for t in ts for r in decode[t]})
    total_d = sum(max(decode[t].values()) for t in ts)
    total_p = {r: sum(prefill[t].get(r, 0) for t in ts) for r in ranks}
    pure = [t for t in ts if sum(prefill[t].values()) == 0 and max(decode[t].values()) > 0]
    steps = [max(decode[t].values()) for t in pure]
    t_pure = len(pure) / sum(steps)
    bs_pure = statistics.mean(statistics.mean(run[t].values()) for t in pure if run.get(t))
    bs_all = statistics.mean(statistics.mean(run[t].values()) for t in ts if run.get(t))
    span = len(ts)
    print(f"window: {span} one-second slices ({head['summary']['start_time']} -> {head['summary']['end_time']}), "
          f"{len(ranks)} DP ranks")
    print(f"decode passes {total_d:,.0f} ({total_d / span:.2f}/s); prefill passes per rank "
          + " / ".join(f"{total_p[r]:,.0f}" for r in ranks) + f" (sum {sum(total_p.values()):,.0f})")
    print(f"pure-decode slices: {len(pure)} ({100 * len(pure) / span:.1f}%), mean {statistics.mean(steps):.1f} "
          f"steps/s -> {1000 * t_pure:.1f} ms per decode step at mean running {bs_pure:.1f} / rank "
          f"(window mean running {bs_all:.1f} / rank)")
    share = total_d * t_pure / span
    print(f"decode share of wall time ~ {100 * share:.1f}%, eager (prefill) share ~ {100 * (1 - share):.1f}% "
          f"-> {(span - total_d * t_pure) / max(sum(total_p.values()), 1) * 1000:.0f} ms of eager time per rank-prefill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

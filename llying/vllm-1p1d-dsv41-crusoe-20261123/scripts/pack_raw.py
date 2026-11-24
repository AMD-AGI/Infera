#!/usr/bin/env python3
"""Pack one 1P1D run directory into a git-sized record: small artifacts verbatim, logs gzipped, and the four
large files reduced (streamed line by line, never loaded whole) to the fields the analysis uses.

  requests.csv.gz          per request (profile_export.jsonl): phase, ids, timestamps, TTFT, latency, ISL/OSL
  engine_timeseries.csv.gz per second (server_metrics_export.json): P/D running, waiting, waiting_by_reason,
                           KV usage, per rank, both phases; the router's federated copies are skipped
  timeslices_key.csv.gz    client time slices (profile_export_aiperf_timeslices.csv), key metrics avg/p50/p90
  router_picks.csv.gz      every kv-aware pick in router.log: time, role, picked target, cache_hits, blocks

usage: pack_raw.py RUN_DIR OUT_DIR
"""
import csv
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

GAUGES = {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_requests_waiting_by_reason",
          "vllm:kv_cache_usage_perc"}
ROLE = {"30000": "P", "30001": "D", "31300": "P", "31301": "D"}
SLICE_METRICS = {"Effective Total Throughput", "Effective Prefill Concurrency", "Effective Decode Concurrency",
                 "Effective Concurrency", "Total Token Throughput", "Output Token Throughput", "Request Throughput",
                 "Time to First Token", "Inter Token Latency", "Request Latency", "Input Sequence Length",
                 "Output Sequence Length", "Tokens In Flight"}
REQ_METRICS = ("time_to_first_token", "request_latency", "input_sequence_length", "output_sequence_length",
               "reasoning_token_count", "usage_prompt_tokens", "usage_completion_tokens", "decode_duration",
               "http_req_sending", "http_req_waiting")
SMALL = 2 * 1024 * 1024


def gz(src, dst):
    with open(src, "rb") as i, gzip.open(f"{dst}.gz", "wb", compresslevel=9) as o:
        shutil.copyfileobj(i, o)


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size > 200 * 1024:
        gz(src, dst)
    else:
        shutil.copy2(src, dst)


def requests(jsonl, out):
    cols = ["phase", "conversation_id", "turn_index", "agent_depth", "session_num",
            "credit_issued_ns", "request_start_ns", "request_ack_ns", "request_end_ns", "was_cancelled",
            *REQ_METRICS, "error_type"]
    with open(jsonl) as f, gzip.open(out, "wt", newline="") as o:
        w = csv.writer(o)
        w.writerow(cols)
        for line in f:
            d = json.loads(line)
            m, mt = d.get("metadata", {}), d.get("metrics") or {}
            val = lambda k: (mt.get(k) or {}).get("value") if isinstance(mt.get(k), dict) else mt.get(k)
            err = d.get("error") or {}
            w.writerow([m.get("benchmark_phase"), m.get("conversation_id"), m.get("turn_index"), m.get("agent_depth"),
                        m.get("session_num"), m.get("credit_issued_ns"),
                        m.get("request_start_ns"), m.get("request_ack_ns"), m.get("request_end_ns"),
                        m.get("was_cancelled"), *[val(k) for k in REQ_METRICS], err.get("type", "")])


def engine_timeseries(js, out):
    key = re.compile(r'^(\s*)"([^"]+)": (.*?),?$')
    phase = metric = role = None
    labels, in_slices, sl = {}, False, {}
    with open(js) as f, gzip.open(out, "wt", newline="") as o:
        w = csv.writer(o)
        w.writerow(["phase", "metric", "role", "engine", "reason", "start_ns", "avg", "min", "max"])
        for line in f:
            mm = key.match(line)
            if not mm:
                continue
            ind, k, v = len(mm.group(1)), mm.group(2), mm.group(3)
            if ind == 2:
                phase = {"metrics": "profiling", "warmup_metrics": "warmup"}.get(k)
                metric = None
            elif ind == 4 and phase:
                metric = k if k in GAUGES else None
            elif metric is None:
                continue
            elif k == "endpoint_url":
                port = re.search(r":(\d+)/metrics", v)
                role = ROLE.get(port.group(1)) if port else None
                labels, in_slices = {}, False
            elif k in ("engine", "reason") and not in_slices:
                labels[k] = v.strip('"')
            elif k == "timeslices":
                in_slices = True
            elif in_slices and role:
                if k == "start_ns":
                    sl = {"start_ns": v}
                elif k in ("avg", "min"):
                    sl[k] = v
                elif k == "max":
                    w.writerow([phase, metric, role, labels.get("engine", ""), labels.get("reason", ""),
                                sl.get("start_ns"), sl.get("avg"), sl.get("min"), v])


def timeslices(src, out):
    with open(src) as f, gzip.open(out, "wt", newline="") as o:
        r, w = csv.reader(f), csv.writer(o)
        w.writerow(next(r))
        for row in r:
            if len(row) > 5 and row[3] in SLICE_METRICS and row[5] in ("avg", "p50", "p90"):
                w.writerow(row)


def router_picks(log, out):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    pick = re.compile(r"^(\S+Z)\s.*pick policy=\"kv-aware\" role=(\w+) retention=\"\w+\" picked=(\S+) "
                      r"cache_hits=(\d+) request_blocks=(\d+) active_blocks=(\d+)")
    with open(log, "rb") as f, gzip.open(out, "wt", newline="") as o:
        w = csv.writer(o)
        w.writerow(["time", "role", "picked", "cache_hits", "request_blocks", "active_blocks"])
        for raw in f:
            m = pick.match(ansi.sub("", raw.replace(b"\x00", b"").decode(errors="replace")))
            if m:
                w.writerow(m.groups())


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(run.iterdir()):
        if p.is_file() and p.name != "router.log" and p.stat().st_size < 20 * SMALL:
            put(p, out / p.name)
        elif p.is_dir() and p.name in ("accuracy", "kvaware"):
            shutil.copytree(p, out / p.name, dirs_exist_ok=True)
    ax = next(run.glob("agentx-c*"))
    res = next(ax.glob("result/conc_*"))
    art = res / "aiperf_artifacts"
    a = out / "agentx"
    for p in [ax / "runtime.env", ax / "runner.log", res / "benchmark_command.txt", res / "benchmark.log",
              res / "workload_distribution_summary.txt", art / "server_metrics_export.csv",
              art / "profile_export_aiperf.json", art / "profile_export_aiperf.csv",
              art / "profile_export_console.txt", art / "logs" / "aiperf.log"]:
        if p.exists():
            put(p, a / p.name)
    # An aborted run leaves some of these behind; pack what exists.
    for fn, src, dst in [(requests, art / "profile_export.jsonl", a / "requests.csv.gz"),
                         (engine_timeseries, art / "server_metrics_export.json", a / "engine_timeseries.csv.gz"),
                         (timeslices, art / "profile_export_aiperf_timeslices.csv", a / "timeslices_key.csv.gz"),
                         (router_picks, run / "router.log", out / "router_picks.csv.gz")]:
        if src.exists():
            fn(src, dst)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1024:10.1f} KB  {p.relative_to(out)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect native SGLang CPU/GPU traces only after observing a full decode batch.

Run on the Docker host after prepare.py and launch.sh. The original benchmark
supplies 16 warmup requests and 128 measured 10000/500-token requests per round.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

from analyze import analyze_trace


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def request(base, endpoint, body=None, timeout=900):
    req = urllib.request.Request(base + endpoint, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        text = response.read().decode()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def check_benchmark(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError(f"Expected one fresh benchmark record in {path}")
    data = records[0]
    if (data["completed"] != 128 or len(data["errors"]) != 128 or any(data["errors"])
            or data["input_lens"] != [10000] * 128 or data["output_lens"] != [500] * 128
            or data["max_concurrency"] != 32):
        raise RuntimeError(f"Benchmark completion/length check failed: {path}")
    keys = ("completed", "duration", "output_throughput", "median_tpot_ms", "p90_tpot_ms",
            "accept_length", "concurrency", "total_output_tokens", "max_concurrency")
    return {**{key: data.get(key) for key in keys}, "errors": 0,
            "input_length": 10000, "output_length": 500}


def run_round(run, name, container, base, steps=None):
    directory = run / "rounds" / name
    directory.mkdir(exist_ok=False)
    console = (directory / "benchmark-console.log").open("w")
    lines = queue.Queue()
    monitor = subprocess.Popen(["docker", "logs", "--follow", "--tail", "0", container],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def read_logs():
        with (directory / "server-window.log").open("w") as log:
            for line in monitor.stdout:
                log.write(line)
                log.flush()
                lines.put(line)

    reader = threading.Thread(target=read_logs, daemon=True)
    reader.start()
    bench = subprocess.Popen(["bash", str(run / "scripts/bench.sh"), name, "128", "32"],
                             stdout=console, stderr=subprocess.STDOUT)
    full_samples = []
    profile_record = None
    deadline = time.monotonic() + 1800
    try:
        while bench.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Benchmark timeout: {name}")
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            match = re.search(r"Decode batch,.*#running-req: (\d+).*#retracted-req: (\d+).*cuda graph: (True|False)", line)
            if match and profile_record is None and steps is not None:
                if match.groups() == ("32", "0", "True"):
                    full_samples.append(line.strip())
                else:
                    full_samples.clear()
                if len(full_samples) >= 2:
                    body = {"output_dir": str(directory / "traces"), "num_steps": steps,
                            "activities": ["CPU", "GPU"], "with_stack": False,
                            "record_shapes": False, "profile_by_stage": False,
                            "merge_profiles": False, "profile_prefix": name}
                    profile_record = {"request": body, "requested_at": now(),
                                      "trigger_samples": full_samples[-2:]}
                    save(directory / "profile-request.json", profile_record)
                    print(f"[{now()}] {name}: full batch confirmed; profiling {steps} forwards", flush=True)
                    response = request(base, "/start_profile", body)
                    profile_record.update(response=response, response_at=now())
                    save(directory / "profile-request.json", profile_record)
                    if isinstance(response, dict) and response.get("success") is False:
                        raise RuntimeError(f"Profiler rejected request: {response}")
        if bench.returncode:
            raise RuntimeError(f"Benchmark failed ({bench.returncode}); see {directory}")
        if steps is not None:
            if profile_record is None:
                raise RuntimeError("Never observed two full graph-enabled c32 decode batches")
            # Native /start_profile with num_steps auto-stops and exports each TP rank.
            traces = list((directory / "traces").glob("*.trace.json.gz"))
            if len(traces) != 4 or {int(re.search(r"TP-(\d+)", p.name)[1]) for p in traces} != set(range(4)):
                raise RuntimeError(f"Expected one trace for each of four TP ranks: {traces}")
            coverage = []
            for path in sorted(traces):
                report, _ = analyze_trace(path)
                item = {key: report[key] for key in ("file", "rank", "kernel_count",
                    "stage_annotation_counts", "verify_batch_sizes", "graph_launch_count",
                    "graph_launches_without_kernels")}
                coverage.append(item)
            save(directory / "trace-validation.json", coverage)
            if any(item["graph_launches_without_kernels"] or item["verify_batch_sizes"] != {"32": steps}
                   for item in coverage):
                raise RuntimeError("Incomplete graph kernels or non-c32 verify steps; see trace-validation.json")
        metrics = check_benchmark(directory / "benchmark.jsonl")
        metrics["profiled"] = steps is not None
        save(directory / "metrics.json", metrics)
        save(directory / "server-info-after.json", request(base, "/server_info", timeout=60))
        print(f"[{now()}] {name}: completed 128/128; output tok/s={metrics['output_throughput']:.2f}", flush=True)
        return metrics
    finally:
        if bench.poll() is None:
            bench.terminate()  # Only this task's benchmark client, never the server.
            bench.wait(timeout=30)
        monitor.terminate()
        monitor.wait(timeout=30)
        reader.join(timeout=5)
        console.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", help="Measure original mode without incomplete graph traces")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    if args.output.exists():
        parser.error("Output directory already exists; choose a fresh destination")
    if args.baseline_only and args.skip_baseline:
        parser.error("baseline-only and skip-baseline cannot be combined")
    run = args.run_dir.resolve()
    state = json.loads((run / "state/profile-environment.json").read_text())
    container = state["container"]
    base = f"http://127.0.0.1:{state['port']}"
    deadline = time.monotonic() + args.ready_timeout
    print(f"[{now()}] Waiting for {container} at {base}", flush=True)
    while True:
        info = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
        if not info["State"]["Running"]:
            raise RuntimeError("Server exited before readiness; inspect docker logs")
        try:
            request(base, "/health", timeout=5)
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() > deadline:
                raise TimeoutError("Server readiness timeout; container retained for diagnosis")
            time.sleep(5)
    save(run / "state/server-info-before.json", request(base, "/server_info", timeout=60))
    print(f"[{now()}] Server ready", flush=True)
    metrics = {}
    if not args.skip_baseline:
        metrics["baseline_c32"] = run_round(run, "baseline_c32", container, base)
    for index in range(1, (0 if args.baseline_only else args.repeats) + 1):
        name = f"profile_{index:02d}"
        metrics[name] = run_round(run, name, container, base, args.steps)
    full_log = subprocess.check_output(["docker", "logs", container], stderr=subprocess.STDOUT)
    with gzip.open(run / "state/server-full.log.gz", "wb") as dest:
        dest.write(full_log)
    text = full_log.decode(errors="replace")
    observations = {"sampled_running_counts": dict(Counter(re.findall(r"#running-req: (\d+)", text))),
                    "flydsl_192_row_decline": "FlyDSL sparse MLA decode declined: seq 192, need 1..96" in text}
    save(run / "state/observations.json", observations)
    save(run / "state/round-metrics.json", metrics)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(run / "state", args.output / "state")
    for name in metrics:
        shutil.copytree(run / "rounds" / name, args.output / name)
    print(f"[{now()}] Evidence copied to {args.output}. Server is still running.", flush=True)


if __name__ == "__main__":
    main()

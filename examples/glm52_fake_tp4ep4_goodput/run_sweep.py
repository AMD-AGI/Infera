#!/usr/bin/env python3
"""Run unprofiled native serving benchmarks on an already-launched Docker host.

Writes full per-request timing, scheduler logs and server state under run_dir.
No profiler endpoint is called, and profiler/graph-debug overrides are rejected.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def get(base, endpoint):
    with urllib.request.urlopen(base + endpoint, timeout=30) as response:
        raw = response.read().decode()
        return json.loads(raw) if raw else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 40, 64])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", default="sweep")
    parser.add_argument("--min-requests", type=int, default=128)
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    args = parser.parse_args()
    if min(args.concurrencies) < 1:
        parser.error("concurrencies must be positive")
    if min(args.repeats, args.min_requests, args.waves) < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("Use positive counts and a shell-safe label")
    run = args.run_dir.resolve()
    state = json.loads((run / "state/environment.json").read_text())
    capacity = state["max_running_requests"]
    if max(args.concurrencies) > capacity:
        parser.error(f"Requested concurrency exceeds the prepared server capacity {capacity}")
    name = state["container"]
    base = f"http://127.0.0.1:{state['port']}"
    inspect = json.loads(subprocess.check_output(["docker", "inspect", name]))[0]
    banned = [entry.split("=", 1)[0] for entry in inspect["Config"]["Env"]
              if any(key in entry.split("=", 1)[0] for key in ("PROFIL", "ROCTRACER", "DEBUG_CLR_GRAPH", "DEBUG_HIP_GRAPH", "LD_PRELOAD"))]
    if banned:
        raise RuntimeError(f"Unexpected profiling/debug environment overrides: {banned}")
    save(run / "state/container-inspect.json", inspect)
    save(run / "state/no-profile-validation.json", {"forbidden_environment": banned,
         "profile_endpoint_calls": 0, "timestamp": now()})
    deadline = time.monotonic() + args.ready_timeout
    print(f"[{now()}] Waiting for server", flush=True)
    while True:
        try:
            get(base, "/health")
            break
        except (urllib.error.URLError, TimeoutError):
            running = subprocess.check_output(["docker", "inspect", name, "--format", "{{.State.Running}}"], text=True).strip()
            if running != "true" or time.monotonic() > deadline:
                raise RuntimeError("Server did not become ready; container retained for diagnosis")
            time.sleep(5)
    info = get(base, "/server_info")
    save(run / "state/server-info-ready.json", info)
    if info["max_running_requests"] != capacity or info["enable_profile_cuda_graph"]:
        raise RuntimeError("Unexpected server capacity or graph profiling configuration")
    if info["mem_fraction_static"] != state.get("mem_fraction_static", 0.85):
        raise RuntimeError("Runtime memory fraction differs from the prepared experiment")
    print(f"[{now()}] Ready; profiling overrides absent", flush=True)
    for repeat in range(1, args.repeats + 1):
        for conc in args.concurrencies:
            point = f"{args.label}_c{conc:02d}_r{repeat:02d}"
            directory = run / "rounds" / point
            directory.mkdir(exist_ok=False)
            num_requests = max(args.min_requests, args.waves * conc)
            num_requests = ((num_requests + conc - 1) // conc) * conc
            warmup = max(16, conc)
            start = now()
            dataset = run / "reference/synthetic_prompts.json"
            prompts = json.loads(dataset.read_text())
            # The archived file has only 160 rows. The native random tokenizer
            # path otherwise silently zips a larger requested count down to 160.
            if len(prompts) < num_requests:
                dataset = directory / "synthetic_prompts_expanded.json"
                save(dataset, [prompts[i % len(prompts)] for i in range(num_requests)])
            command = ["python3", str(run / "scripts/bench_with_details.py"),
                "--backend", "sglang", "--host", "127.0.0.1", "--port", str(state["port"]),
                "--model", "/shared_nfs/models/GLM-5.2-MXFP4",
                "--dataset-name", "random", "--dataset-path", str(dataset),
                "--tokenize-prompt", "--random-input-len", "10000", "--random-output-len", "500",
                "--random-range-ratio", "1", "--num-prompts", str(num_requests),
                "--max-concurrency", str(conc), "--warmup-requests", str(warmup),
                "--fake-prefill", "--output-details", "--output-file", str(directory / "benchmark.jsonl")]
            (directory / "benchmark-command.txt").write_text(shlex.join(command) + "\n")
            save(directory / "run.json", {"point": point, "concurrency": conc,
                "num_requests": num_requests, "warmup_requests": warmup, "started_at": start,
                "profile": False, "repeat": repeat})
            print(f"[{start}] {point}: {num_requests} measured + {warmup} warmup requests", flush=True)
            with (directory / "benchmark-console.log").open("w") as console:
                completed = subprocess.run(["docker", "exec", "-e", "PYTHONPATH=/sglang/python", name, *command],
                                           stdout=console, stderr=subprocess.STDOUT, timeout=2400)
            text = subprocess.check_output(["docker", "logs", "--since", start, name], stderr=subprocess.STDOUT).decode()
            (directory / "server-window.log").write_text(text)
            save(directory / "server-info-after.json", get(base, "/server_info"))
            if completed.returncode:
                raise RuntimeError(f"Benchmark failed: {point}; inspect benchmark-console.log")
            records = [json.loads(line) for line in (directory / "benchmark.jsonl").read_text().splitlines() if line.strip()]
            if len(records) != 1:
                raise RuntimeError(f"Expected one benchmark record: {point}")
            data = records[0]
            if (data["completed"] != num_requests or any(data["errors"]) or not all(data["successes"])
                    or data["input_lens"] != [10000] * num_requests or data["output_lens"] != [500] * num_requests):
                raise RuntimeError(f"Completion or token length validation failed: {point}")
            lines = [line for line in text.splitlines() if "Decode batch," in line]
            occupancy = Counter(int(re.search(r"#running-req: (\d+)", line)[1]) for line in lines)
            retractions = sorted({int(re.search(r"#retracted-req: (\d+)", line)[1]) for line in lines})
            if not occupancy.get(conc):
                raise RuntimeError(f"No scheduler sample at requested concurrency: {point}, {occupancy}")
            observation = {"sampled_running_requests": dict(sorted(occupancy.items())),
                "sampled_retractions": retractions, "all_samples_cuda_graph": all("cuda graph: True" in line for line in lines),
                "profiling_started_in_logs": "Profiling starts" in text, "completed_at": now()}
            save(directory / "observations.json", observation)
            if observation["profiling_started_in_logs"] or retractions != [0]:
                raise RuntimeError(f"Profiling or request retraction detected: {point}")
            print(f"[{now()}] {point}: {data['completed']}/{num_requests}; output={data['output_throughput']:.2f} tok/s; "
                  f"TPOT p50/p90={data['median_tpot_ms']:.3f}/{data['p90_tpot_ms']:.3f} ms; "
                  f"ITL p50/p90/p99={data['median_itl_ms']:.3f}/{data['p90_itl_ms']:.3f}/{data['p99_itl_ms']:.3f} ms", flush=True)
    with gzip.open(run / f"state/server-full-{args.label}.log.gz", "wb") as dest:
        dest.write(subprocess.check_output(["docker", "logs", name], stderr=subprocess.STDOUT))
    print(f"[{now()}] Requested sweep completed. Server retained for boundary confirmation.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run B4 on one crsuse2 node pair the way the aus campaign ran it.

Stages, in order (``all`` runs them unattended):
  prepare   run directory, topology, idle-node / image / port checks, AITER cache
  launch    etcd, OTLP collector, P and D (real acceptance), router
  gate      placement_answer_probe.py: known answers over every P/D rank
  switch    relaunch D with simulated acceptance 3.61, restart the router
  preflight preflight_placement.py, capacity check against B4
  measure   samplers + AgentX C80 for 3600 s, then capture logs/diagnostics
``stop`` removes this run's containers (only names under CONTAINER_PREFIX).
The aus scripts in runtime/scripts are used unmodified; this file replaces only
their Slurm-bound launcher (launch_placement.py / transition_placement.py).
Usage: run_b4.sh STAGE (the wrapper sources the config into the environment).
"""

import base64, concurrent.futures, datetime, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

E = os.environ
RUN = Path(E["RUN"])
ROOT = Path(E["TRACE_RUNTIME"])
sys.path.insert(0, str(ROOT / "scripts"))
import transition_two_node as ops  # noqa: E402

PREFIX = E["CONTAINER_PREFIX"]
# Forwarded to engine.sh, which sources the config on the worker node.
PASSTHROUGH = ("B4_PREFILL_NODE", "B4_DECODE_NODE", "B4_RUN_ID")
B4_CAPACITY = {"prefill": int(E["REFERENCE_PREFILL_TOKENS"]), "decode": int(E["REFERENCE_DECODE_TOKENS"])}


def log(message):
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def status(name):
    (RUN / "STATUS").write_text(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {name}\n")
    log(name)


def workers():
    rows = [
        dict(role="prefill", node=E["PREFILL_NODE"], data_ip=E["PREFILL_IP"], engine_port=29001,
             bootstrap_port=28998, kv_port=25557, snapshot_port=28801),
        dict(role="decode", node=E["DECODE_NODE"], data_ip=E["DECODE_IP"], engine_port=29002,
             bootstrap_port=28999, kv_port=25558, snapshot_port=28802),
    ]
    for row in rows:
        row.update(container=f"{PREFIX}-{row['role']}-0", gpu_ids=list(range(8)), tp=8, dp=8,
                   allocation_job=E["ALLOCATION_JOB_ID"])
    return rows


def resolved():
    sys.path.insert(0, str(ROOT / "scripts/bench-harness/tools"))
    from campaign_topology import load
    return load(RUN / "topology.json")


def vram_fractions(node):
    data = json.loads(ops.remote(node, ["rocm-smi", "--showmeminfo", "vram", "--json"]))
    return [int(v["VRAM Total Used Memory (B)"]) / int(v["VRAM Total Memory (B)"])
            for k, v in sorted(data.items()) if k.startswith("card")]


def prepare():
    if RUN.exists():
        raise SystemExit(f"run exists: {RUN}")
    for d in ["snapshot", "logs", "traces", "sampling", "launch/server-info", "launch/server-logs"]:
        (RUN / d).mkdir(parents=True)
    (RUN / "topology.json").write_text(json.dumps(workers(), indent=2) + "\n")
    rows = resolved()
    (RUN / "snapshot/config.sh").write_bytes(Path(E["CONFIG"]).read_bytes())
    # sample_node_runtime.py uses this known_hosts file without accept-new.
    for node in {w["node"] for w in rows}:
        subprocess.run(["ssh", "-F", "/dev/null", "-o", "UserKnownHostsFile=/tmp/agentx_known_hosts", "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR", node, "true"], check=True)
    base_cache = "fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35"
    image_key = E["EXPECTED_IMAGE_ID"].removeprefix("sha256:")
    for node in {w["node"] for w in rows}:
        image = ops.remote(node, ["docker", "image", "inspect", "--format", "{{.Id}}", E["IMAGE"]]).strip()
        assert image == E["EXPECTED_IMAGE_ID"], (node, image)
        busy = [f for f in vram_fractions(node) if f >= 0.02]
        assert not busy, f"{node}: GPUs hold VRAM {busy}"
        others = ops.remote(node, ["docker", "ps", "--format", "{{.Names}}"]).split()
        assert not [n for n in others if n.startswith(PREFIX)], f"{node}: {PREFIX} containers exist"
        # Same AITER commit (4ad99832) as the crsuse2 base image: seed its JIT cache.
        cache = f"/tmp/aiter-jit-{os.getuid()}"
        ops.remote(node, ["bash", "-c", f"[[ -d {cache}/{image_key} ]] || "
                          f"[[ ! -d {cache}/{base_cache} ]] || cp -a {cache}/{base_cache} {cache}/{image_key}"])
    ops.check_ports_free(E["PREFILL_NODE"], [22379, 22380, 4317, 28000])
    for w in rows:
        ops.check_ports_free(w["node"], [w[k] for k in ["engine_port", "bootstrap_port", "kv_port", "snapshot_port"]])
    status("PREPARED")


def registration(w):
    key = base64.b64encode(f"/infera/workers/{w['ip']}:{w['engine_port']}".encode()).decode()
    req = urllib.request.Request(f"http://{E['PREFILL_IP']}:22379/v3/kv/range", data=json.dumps({"key": key}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as f:
        kvs = json.load(f).get("kvs", [])
    return (int(kvs[0]["mod_revision"]), json.loads(base64.b64decode(kvs[0]["value"]))) if kvs else (0, None)


def launch_worker(w, simulate):
    before, _ = registration(w)
    configured = json.loads(E["RDMA_DEVICE"])
    mapping = {str(i): configured[str(g)] for i, g in enumerate(w["gpu_ids"])}
    extra = [f"{k}={E[k]}" for k in PASSTHROUGH if k in E] + [f"B4_DECODE_SIMULATE_ACC_LEN={simulate}"]
    cmd = ["bash", f"{E['BENCH_DIR']}/engine.sh", w["role"], w["instance"], w["ip"], ",".join(map(str, w["gpu_ids"])),
           w["engine_port"], w["bootstrap_port"], w["kv_port"], w["snapshot_port"], w["container"],
           f"{E['PREFILL_IP']}:22379", f"CONFIG={E['CONFIG']}",
           "WORKER_RDMA_DEVICE=" + json.dumps(mapping, separators=(",", ":")), f"DIAG_INSTANCE={w['instance']}",
           f"WORKER_ALLOCATION_JOB_ID={w['allocation_job']}",
           f"SERVER_LOG={RUN}/launch/server-logs/{w['instance']}.log", *extra]
    log(f"START {w['instance']} on {w['node']} simulate_acc_len={simulate or 'real'}")
    output = ops.remote(w["node"], cmd, timeout=300)
    with (RUN / "logs" / f"engine-{w['instance']}.log").open("a") as f:
        f.write(output)
    ops.wait_healthy(w["node"], w["container"], w["url"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        revision, value = registration(w)
        if revision > before and value and value.get("url") == w["url"]:
            assert value["dp_size"] == w["dp"] and value["disagg_mode"] == w["role"]
            log(f"READY {w['instance']}")
            return
        time.sleep(1)
    raise TimeoutError(f"{w['instance']} never registered")


def record_placement():
    """placement-resolved.json, server-info and container snapshots, as launch_placement.py writes them."""
    out = []
    for w in resolved():
        c = json.loads(ops.remote(w["node"], ["docker", "inspect", w["container"]]))[0]
        info = ops.get(w["url"] + "/get_server_info")
        assert info["tp_size"] == w["tp"] and info["dp_size"] == w["dp"] and info["chunked_prefill_size"] == 4096
        (RUN / f"launch/server-info/{w['instance']}.json").write_text(json.dumps(info, indent=2) + "\n")
        (RUN / f"snapshot/{w['instance']}-container.json").write_text(json.dumps(c, indent=2) + "\n")
        diag = next(m["Source"] for m in c["Mounts"] if m["Destination"] == "/aus-diag")
        out.append(dict(w, container_id=c["Id"], diag_dir=diag, scheduler_pids=info.get("scheduler_pids")))
    (RUN / "placement-resolved.json").write_text(json.dumps(out, indent=2) + "\n")


def start_router():
    subprocess.run([sys.executable, str(ROOT / "scripts/start_performance_router.py")], check=True)
    ops.wait_healthy(E["PREFILL_NODE"], f"{PREFIX}-router", f"http://{E['PREFILL_IP']}:28000")


def launch():
    node, ip = E["PREFILL_NODE"], E["PREFILL_IP"]
    (RUN / "snapshot/capture-start-epoch.txt").write_text(f"{int(time.time())}\n")
    label = f"infera.allocation-job={E['ALLOCATION_JOB_ID']}"
    ops.remote(node, ["docker", "run", "-d", "--init", "--name", f"{PREFIX}-etcd", "--network", "host", "--label", label,
                      "quay.io/coreos/etcd:v3.5.14", "etcd", "--advertise-client-urls", f"http://{ip}:22379",
                      "--listen-client-urls", "http://0.0.0.0:22379", "--listen-peer-urls", "http://127.0.0.1:22380",
                      "--initial-advertise-peer-urls", "http://127.0.0.1:22380",
                      "--initial-cluster", "default=http://127.0.0.1:22380"])
    ops.wait_healthy(node, f"{PREFIX}-etcd", f"http://{ip}:22379")
    ops.remote(node, ["docker", "run", "-d", "--init", "--name", f"{PREFIX}-collector", "--network", "host",
                      "--label", label, "-v", f"{ROOT}:{ROOT}", E["IMAGE"], "python3",
                      str(ROOT / "scripts/otlp_jsonl_collector.py"), "--output", str(RUN / "traces/spans.jsonl"),
                      "--ready-file", str(RUN / "traces/ready.json")])
    for _ in range(60):
        if (RUN / "traces/ready.json").exists():
            break
        time.sleep(2)
    else:
        raise TimeoutError("collector not ready")
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(launch_worker, w, "") for w in resolved()]:
            f.result()
    record_placement()
    start_router()
    status("READY_FOR_REAL_ACCEPTANCE_PROBE")


def gate():
    subprocess.run([sys.executable, str(ROOT / "scripts/placement_answer_probe.py")], check=True)
    status("PLACEMENT_GATE_PASSED")


def switch():
    ops.wait_idle()
    decode = next(w for w in resolved() if w["role"] == "decode")
    for node, name in [(E["PREFILL_NODE"], f"{PREFIX}-router"), (decode["node"], decode["container"])]:
        ops.remote(node, ["docker", "stop", "-t", "60", name])
        ops.remote(node, ["docker", "rm", name])
    for _ in range(240):
        if all(f < 0.02 for f in vram_fractions(decode["node"])):
            break
        time.sleep(10)
    else:
        raise TimeoutError("decode VRAM not released")
    launch_worker(decode, E["DECODE_SIMULATE_ACC_LEN"])
    record_placement()
    start_router()
    status("READY_FOR_PREFLIGHT")


def preflight():
    subprocess.run([sys.executable, str(ROOT / "scripts/preflight_placement.py")], check=True)
    capacity = {}
    for w in resolved():
        with urllib.request.urlopen(w["url"] + "/metrics", timeout=10) as f:
            raw = f.read().decode()
        values = [float(v) for v in re.findall(r"^sglang:max_total_num_tokens\{[^}]*\} ([0-9.eE+-]+)$", raw, re.M)]
        assert len(values) == w["dp"], (w["instance"], values)
        drift = max(abs(v - B4_CAPACITY[w["role"]]) / B4_CAPACITY[w["role"]] for v in values)
        capacity[w["instance"]] = {"tokens_per_rank": values, "b4": B4_CAPACITY[w["role"]], "max_relative_drift": drift}
    (RUN / "capacity-check.json").write_text(json.dumps(capacity, indent=2) + "\n")
    worst = max(c["max_relative_drift"] for c in capacity.values())
    limit = float(E["MAX_AUTOMATIC_CAPACITY_RELATIVE_DRIFT"])
    if worst > limit and E.get("B4_ALLOW_CAPACITY_DRIFT") != "1":
        raise SystemExit(f"KV capacity drifts {worst:.4%} from B4 (limit {limit:.2%}); see capacity-check.json")
    status("PREFLIGHT_PASSED")


def measure():
    """Same samplers, intervals and client call as run_placement_performance.py; analysis runs offline."""
    rows = json.loads((RUN / "placement-resolved.json").read_text())
    engine = [sys.executable, str(ROOT / "scripts/sample_engine_metrics.py")]
    for w in rows:
        engine += ["--endpoint", f"{w['instance']}={w['url']}/metrics"]
    engine += ["--endpoint", f"router=http://{E['PREFILL_IP']}:28000/metrics", "--output", str(RUN / "sampling/engine.jsonl"),
               "--interval", "2", "--duration", "0"]
    node = [sys.executable, str(ROOT / "scripts/sample_node_runtime.py")]
    for w in rows:
        node += ["--node", f"{w['instance']}={w['node']}"]
    node += ["--output", str(RUN / "sampling/nodes.jsonl"), "--interval", "5", "--duration", "0"]
    capture = [sys.executable, "-c", "import subprocess,time,sys\nwhile True:\n subprocess.run([sys.executable,sys.argv[1]])\n time.sleep(60)",
               str(ROOT / "scripts/capture_placement.py")]
    observers = [subprocess.Popen(cmd, stdout=(RUN / "logs" / name).open("w"), stderr=subprocess.STDOUT, start_new_session=True)
                 for cmd, name in [(engine, "engine-sampler.log"), (node, "node-sampler.log"), (capture, "capture.log")]]
    try:
        (RUN / "c80-started.txt").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
        status("BENCHMARK_C80")
        cmd = ["bash", f"{E['BENCH_DIR']}/agentx_bench.sh", f"CONFIG={E['CONFIG']}", f"TOPOLOGY={RUN / 'topology.json'}",
               "CONC=80", "DURATION=3600", f"OUT_DIR={RUN / 'c80'}"]
        with (RUN / "logs/c80.log").open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        (RUN / "c80-completed.txt").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
    finally:
        for p in observers:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(p.pid, sig)
                    p.wait(timeout=10)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
    subprocess.run([sys.executable, str(ROOT / "scripts/capture_placement.py")], check=True)
    (RUN / "final-server-info").mkdir(exist_ok=True)
    for w in rows:
        (RUN / f"final-server-info/{w['instance']}.json").write_text(json.dumps(ops.get(w["url"] + "/get_server_info"), indent=2) + "\n")
    status("MEASUREMENT_COMPLETE")


def stop():
    for node in {E["PREFILL_NODE"], E["DECODE_NODE"]}:
        names = [n for n in ops.remote(node, ["docker", "ps", "-a", "--format", "{{.Names}}"]).split() if n.startswith(PREFIX + "-")]
        if names:
            subprocess.run(ops.SSH + [node, "docker stop -t 60 " + " ".join(names) + " && docker rm " + " ".join(names)],
                           check=False)
        log(f"{node}: stopped {names}")


STAGES = {"prepare": prepare, "launch": launch, "gate": gate, "switch": switch, "preflight": preflight, "measure": measure}

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "stop":
        stop()
    elif stage == "all":
        for name, fn in STAGES.items():
            try:
                fn()
            except BaseException:
                if (RUN / "STATUS").exists():
                    status(f"FAILED_{name.upper()}_SERVICES_PRESERVED")
                raise
    else:
        STAGES[stage]()

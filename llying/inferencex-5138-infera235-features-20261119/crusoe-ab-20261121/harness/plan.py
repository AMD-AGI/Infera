#!/usr/bin/env python3
"""Launch scripts for one rendered #5138 configuration (render/out/<name>.json) on crusoe, built the way srtctl at the
InferenceX pin (2d9079c + runners/srt-slurm/patches) builds them for frontend.type: infera, plus the crusoe site layer.

From srtctl (src/srtctl): the worker command (backends/sglang.py build_worker_command, _config_to_cli_args; the Infera
launcher args of frontends/infera.py), the worker environment and its order (cli/mixins/worker_stage.py), the router
command (frontends/infera.py build_router_command, frontend_args_to_cli), etcd (services/infra.py), the setup script
run before every worker and the router, and the AgentX client environment of benchmark-multinode-tmpl.yml,
runtime_settings.sh and the recipe's benchmark.env.

Crusoe site layer, not part of the recipe: KV pools in amdgpu GEM BOs registered through dma-buf (v3 has no peer memory
and amdgpu 6.16 charges dma-buf-pinned KFD memory twice), the libionic 54.0-187 provider in place of the image's
(54.0-149 does not speak the host driver's ABI; the recipe's host libionic needs GLIBC_2.38), a node-local HF cache in
place of /var/lib/hf-hub-cache, this cluster's NIC names (ionic_0..7) for the recipe's disaggregation-ib-device (AUS's
rdma0..7), and docker instead of pyxis. Co-located workers share --ipc host, and RCCL names shared memory after pids,
which start low in every container; the cluster's docker-authz denies --pid=host, so the k-th worker on a node burns
30,000 x k pids before it starts (as the 11-16 crusoe kit did), where pyxis has one pid namespace per node.

Writes RUN/proc/<name>.sh and .env per process, RUN/manifest.tsv (name, node, kind, health URL), RUN/bench.env and
RUN/plan.json. Stdlib only: it runs on the node hosts.
"""
import argparse
import json
import re
import shlex
from pathlib import Path

SETUP_SCRIPT = "/infmax-workspace/benchmarks/multi_node/srt-slurm-recipes/configs/infera-sglang.sh"
SGLANG_UTILS = "/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/utils.py"
IMAGE_LIBIONIC = "/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71"
HOST_LIBIONIC = "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so"
ETCD_IMAGE = "quay.io/coreos/etcd:v3.5.14"
# Environment only the patched Mooncake (748af1a5 + #4540 + #4541, harness/build_mooncake.sbatch) reads; a worker whose
# environment sets any of them gets that build unpacked over the image before it starts.
PATCHED_MOONCAKE_ENV = ("MC_RDMA_WRITE_PREFER_DEST_NIC", "MC_IB_TIMEOUT")
MOONCAKE_OVERLAY = "/ix5138/mooncake-overlay.tar"
GPUS_PER_NODE = 8
ROUTER_VERSION = "0.3.3"
# The matrix's total-cpu-dram-gb / decode-cpu-dram-gb for a worker on this many GPUs: InferenceX's
# infx.matrix.generate.agentic_dram_offload_gb for runner cluster:mi355x-amds at the key's dram-utilization 0.85.
CPU_DRAM_GB = {4: 1274, 8: 2549}
# Keys build_worker_command sets itself (it pops them from the role's args).
POPPED = {"model-path", "model_path", "served-model-name", "served_model_name", "nccl-port", "nccl_port",
          "grpc-http-sidecar-port", "grpc_http_sidecar_port", "smg-http-sidecar-port", "smg_http_sidecar_port"}
ENABLE_METRICS_KEYS = {"enable-metrics", "enable_metrics"}
DOCKER_GPU = ["--device", "/dev/kfd", "--device", "/dev/dri", "--device", "/dev/infiniband",
              "--group-add", "video", "--group-add", "render", "--cap-add", "SYS_PTRACE", "--cap-add", "IPC_LOCK",
              "--security-opt", "seccomp=unconfined", "--ulimit", "memlock=-1:-1", "--ulimit", "nofile=65536:65536"]


def config_to_cli(config: dict) -> list[str]:
    """srtctl backends/sglang.py _config_to_cli_args."""
    args = []
    for key, value in sorted(config.items()):
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag}")
        elif isinstance(value, list):
            args.append(f"--{flag}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag}", str(value)])
    return args


def frontend_args_to_cli(args: dict) -> list[str]:
    """srtctl frontends/base.py frontend_args_to_cli: keys verbatim, True bare, False/None dropped."""
    out = []
    for key, value in args.items():
        if value is True:
            out.append(f"--{key}")
        elif value is not False and value is not None:
            out.extend([f"--{key}", str(value)])
    return out


def fabric(value: str, node: dict, gid_index: str) -> str:
    """The binder's '@fabric.<name>' cluster facts, for this node."""
    if not isinstance(value, str) or not value.startswith("@fabric."):
        return value
    facts = {"socket-ifname": node["ifname"], "rdma-gid-index": gid_index}
    name = value.removeprefix("@fabric.")
    if name not in facts:
        raise SystemExit(f"no crusoe value for {value}")
    return facts[name]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--label", required=True, help="container name prefix, unique per point")
    p.add_argument("--node", action="append", required=True, help="host,ip,ifname (node 0 runs etcd and the router)")
    p.add_argument("--place", action="append", required=True,
                   help="role:node:gpus, one per worker in order, e.g. prefill:0:0,1,2,3")
    p.add_argument("--port-base", type=int, default=30000)
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--site", required=True, type=Path, help="the campaign directory (gem/, ionic/, src/)")
    p.add_argument("--hf-home", required=True, help="node-local directory mounted as the recipe's /mnt/hf_home")
    p.add_argument("--gid-index", default="1")
    p.add_argument("--aiter-jit", default="", help="the image's aiter JIT directory, kept in a volume when set")
    p.add_argument("--cache-prefix", default="lyl-ix5138", help="docker volumes for compile caches, per port base")
    p.add_argument("--ib-devices", default=",".join(f"ionic_{i}" for i in range(8)),
                   help="this cluster's names for rdma0..rdma7 (AUS's) in the recipe's disaggregation-ib-device, "
                        "a list or a per-GPU JSON mapping")
    a = p.parse_args()

    cfg = json.loads(a.config.read_text())
    nodes = [dict(zip(("host", "ip", "ifname"), n.split(","))) for n in a.node]
    head = nodes[0]
    base = a.port_base
    etcd_port, etcd_peer, router_port = base + 79, base + 80, base
    etcd = f"http://{head['ip']}:{etcd_port}"
    roles = cfg["roles"]
    served = next((roles[r]["args"].get("served-model-name") for r in ("prefill", "agg", "decode")
                   if r in roles and roles[r].get("args", {}).get("served-model-name")), Path(a.model).name)
    gem_so = str(a.site / "gem/build/libhip_gem_alloc.so")
    libionic = str(a.site / "ionic/root/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-187")
    ix_root = str(a.site / "src/InferenceX/inferencex-e2e")
    site_env = {"LD_PRELOAD": gem_so, "SGLANG_AMD_GEM_ALLOCATOR": gem_so, "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "AMD_GEM",
                "MOONCAKE_DISABLE_HIP_DMABUF": "0", "MC_HIP_KFD_PIN_FIX": "0", "MC_DISABLE_HIP": "1"}
    common_mounts = [f"{a.model}:{a.model}:ro", f"{ix_root}:/infmax-workspace:ro"]
    site_mounts = [f"{a.site / 'gem/build'}:{a.site / 'gem/build'}:ro",
                   f"{a.site / 'gem/sglang/mooncake_utils.py'}:{SGLANG_UTILS}:ro",
                   f"{libionic}:{IMAGE_LIBIONIC}:ro", f"{a.hf_home}:/mnt/hf_home"]
    for mount in cfg.get("extra_mount") or []:
        src, dst = mount.split(":")[:2]
        if dst == HOST_LIBIONIC or dst == "/mnt/hf_home/hub":
            continue  # replaced by the site's libionic and HF cache above
        raise SystemExit(f"recipe extra_mount {mount} has no crusoe counterpart")
    proc = a.run_dir / "proc"
    proc.mkdir(parents=True, exist_ok=True)
    manifest, record = [], []

    def write(name, node_idx, kind, health, docker_args, image, cmd, env):
        envfile = proc / f"{name}.env"
        envfile.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
        container = f"{a.label}-{name}"
        argv = ["docker", "run", "-d", "--name", container, "--network", "host", *docker_args,
                "--env-file", str(envfile), image, *cmd]
        (proc / f"{name}.sh").write_text(f"#!/usr/bin/env bash\n# {kind} on {nodes[node_idx]['host']}\n"
                                         f"exec {shlex.join(argv)}\n")
        manifest.append(f"{name}\t{node_idx}\t{kind}\t{container}\t{health}")
        record.append({"name": name, "node": nodes[node_idx]["host"], "kind": kind, "container": container,
                       "env": env, "cmd": cmd})

    write("etcd", 0, "etcd", f"http://{head['ip']}:{etcd_port}/version", [], ETCD_IMAGE,
          ["/usr/local/bin/etcd", "--data-dir", "/tmp/etcd", "--listen-client-urls", f"http://0.0.0.0:{etcd_port}",
           "--advertise-client-urls", etcd, "--listen-peer-urls", f"http://127.0.0.1:{etcd_peer}",
           "--initial-advertise-peer-urls", f"http://127.0.0.1:{etcd_peer}",
           "--initial-cluster", f"default=http://127.0.0.1:{etcd_peer}"], {})

    endpoints = {"prefill": [], "decode": []}
    counts = {"prefill": 0, "decode": 0}
    per_node = {}
    for place in a.place:
        mode, node_idx, gpus = place.split(":")
        node_idx = int(node_idx)
        node = nodes[node_idx]
        i = counts[mode]
        counts[mode] += 1
        role = roles[mode]
        ports = base + (100 if mode == "prefill" else 500) + 20 * i
        http, nccl, bootstrap, kv_port, sys_port = ports, ports + 1, ports + 2, ports + 3, ports + 4
        args = {k: v for k, v in (role.get("args") or {}).items() if k not in POPPED}
        if "disaggregation-ib-device" in args:
            names = a.ib_devices.split(",")
            args["disaggregation-ib-device"] = re.sub(r"\brdma(\d+)\b", lambda m: names[int(m.group(1))],
                                                      str(args["disaggregation-ib-device"]))
        kv_events = role.get("kv_events")
        kv_cfg = None
        if kv_events:
            kv_cfg = {"publisher": "zmq", "topic": "kv-events", **(kv_events if isinstance(kv_events, dict) else {})}
        cmd = ["python3", "-m", "infera.engine.sglang", "--discovery-backend", "etcd", "--etcd-endpoint", etcd,
               "--advertise-host", node["ip"], "--request-transport", "http", "--kv-event-transport", "zmq",
               "--kv-events", "off"]
        if kv_cfg is None:
            cmd.append("--no-enable-kv-events")
        cmd += ["--model-path", a.model, "--served-model-name", served, "--host", "0.0.0.0", "--port", str(http),
                "--nccl-port", str(nccl)]
        if not ENABLE_METRICS_KEYS & set(args):
            cmd.append("--enable-metrics")
        cmd += ["--disaggregation-mode", mode, "--skip-server-warmup"]
        if mode == "prefill":
            cmd += ["--disaggregation-bootstrap-port", str(bootstrap)]
        if kv_cfg is not None:
            cmd += ["--kv-events-config", json.dumps({**kv_cfg, "endpoint": f"tcp://*:{kv_port}"})]
        cmd += config_to_cli(args)

        env = {"HEAD_NODE_IP": head["ip"], "INFERA_READINESS_PORT": str(sys_port), "INFERA_DEFAULT_KV_FP8": "0"}
        mode_env = {k: fabric(v, node, a.gid_index) for k, v in (role.get("env") or {}).items()}
        merged = {**(cfg.get("environment") or {}), **mode_env}
        if merged.get("SGLANG_CUDA_COREDUMP", "0").lower() not in ("1", "true"):
            mode_env.setdefault("SGLANG_CUDA_COREDUMP_BEFORE_CRASH", "0")
        mode_env.setdefault("SGLANG_PYSPY_DUMP_BEFORE_CRASH", "0")
        env.update(mode_env)
        env.update({k: fabric(v, node, a.gid_index) for k, v in (cfg.get("environment") or {}).items()})
        n_gpus = len(gpus.split(","))
        if n_gpus != role["gpus"]:
            raise SystemExit(f"{mode} {i}: placed on {n_gpus} GPUs, the recipe asks for {role['gpus']}")
        if n_gpus < GPUS_PER_NODE:
            env["ROCR_VISIBLE_DEVICES"] = gpus
        env.update(site_env)
        env["ROUTER_VERSION"] = ROUTER_VERSION
        name = f"{mode}{i}"
        k = per_node.get(node_idx, 0)
        per_node[node_idx] = k + 1
        burn = f'i=0; while [ $i -lt {30000 * k} ]; do ( : ); i=$((i + 1)); done; ' if k else ""
        overlay = ""
        volumes = [f"{a.cache_prefix}-{base}-{name}-cache:/root/.cache"]
        if any(key in env for key in PATCHED_MOONCAKE_ENV):
            tar = a.site / "mooncake/mooncake-overlay.tar"
            if not tar.is_file():
                raise SystemExit(f"{name} sets {PATCHED_MOONCAKE_ENV} but {tar} (the patched Mooncake) is missing")
            volumes.append(f"{tar}:{MOONCAKE_OVERLAY}:ro")
            overlay = f"tar -C / -xf {MOONCAKE_OVERLAY} && "
        setup = ["bash", "-c", f'{overlay}{burn}bash {SETUP_SCRIPT} && exec "$@"', "_"]
        if a.aiter_jit:
            volumes.append(f"{a.cache_prefix}-{base}-{name}-aiterjit:{a.aiter_jit}")
        mounts = [m for v in (*common_mounts, *site_mounts, *volumes) for m in ("-v", v)]
        write(name, node_idx, mode, f"http://{node['ip']}:{http}/health",
              ["--ipc", "host", "--shm-size", "32g", *DOCKER_GPU, *mounts], a.image,
              [*setup, *cmd], env)
        endpoints[mode].append(f"{node['ip']}:{http}")
    for mode, role in roles.items():
        if mode in counts and counts[mode] != role.get("workers", 1):
            raise SystemExit(f"{mode}: {counts[mode]} placed, the recipe has {role.get('workers', 1)} workers")

    fargs = dict((cfg.get("frontend") or {}).get("args") or {})
    fargs.setdefault("router-tokenizer-path", a.model)
    router_cmd = ["python3", "-m", "infera.server", "--router-backend", "rust", "--host", "0.0.0.0",
                  "--port", str(router_port), "--discovery-backend", "etcd", "--etcd-endpoint", etcd,
                  "--request-transport", "http", "--kv-event-transport", "zmq", *frontend_args_to_cli(fargs)]
    router_env = {**(cfg.get("environment") or {}), **((cfg.get("frontend") or {}).get("env") or {}),
                  "ROUTER_VERSION": ROUTER_VERSION}
    setup = ["bash", "-c", f'bash {SETUP_SCRIPT} && exec "$@"', "_"]
    write("router", 0, "router", f"http://{head['ip']}:{router_port}/v1/workers",
          [m for v in common_mounts for m in ("-v", v)], a.image, [*setup, *router_cmd], router_env)

    pre, dec = roles["prefill"]["args"], roles["decode"]["args"]
    bench = {
        "MODEL": "amd/GLM-5.2-MXFP4", "TOKENIZER": a.model, "SERVED_MODEL_NAME": served, "IMAGE": a.image,
        "MODEL_PREFIX": "glm5.2", "FRAMEWORK": "infera-sglang", "PRECISION": "fp4", "THINKING_MODE": "thinking_on",
        "SCENARIO_TYPE": "agentic-coding", "IS_AGENTIC": "1", "IS_MULTINODE": "true", "DISAGG": "true",
        "SPEC_DECODING": "mtp", "ISL": "0", "OSL": "0", "MAX_MODEL_LEN": "0",
        "KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "hicache", "KV_OFFLOAD_BACKEND_METADATA": '{"name":"hicache"}',
        "ROUTER_METADATA": json.dumps({"name": "infera-router", "version": ROUTER_VERSION}, separators=(",", ":")),
        "KV_P2P_TRANSFER": "mooncake", "TOTAL_CPU_DRAM_GB": str(CPU_DRAM_GB[roles["prefill"]["gpus"]]),
        "DECODE_CPU_DRAM_GB": str(CPU_DRAM_GB[roles["decode"]["gpus"]]),
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "POWER": "0", "REQUIRE_POWER": "0", "ENABLE_AGENTX_POWER": "0",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "1", "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
        "PREFILL_NUM_WORKERS": str(counts["prefill"]), "PREFILL_TP": str(pre.get("tp-size", 1)),
        "PREFILL_PP_SIZE": str(pre.get("pp-size", 1)), "PREFILL_EP": str(pre.get("ep-size", 1)),
        "PREFILL_DP_ATTN": str(bool(pre.get("enable-dp-attention"))).lower(),
        "DECODE_NUM_WORKERS": str(counts["decode"]), "DECODE_TP": str(dec.get("tp-size", 1)),
        "DECODE_PP_SIZE": str(dec.get("pp-size", 1)), "DECODE_EP": str(dec.get("ep-size", 1)),
        "DECODE_DP_ATTN": str(bool(dec.get("enable-dp-attention"))).lower(),
        "SRTCTL_FRONTEND_TYPE": "infera", "SRT_FRONTEND_HOST": head["ip"], "SRT_FRONTEND_PORT": str(router_port),
        "SRT_PREFILL_ENDPOINTS": ",".join(endpoints["prefill"]), "SRT_DECODE_ENDPOINTS": ",".join(endpoints["decode"]),
        **((cfg.get("benchmark") or {}).get("env") or {}),
    }
    (a.run_dir / "bench.env").write_text("".join(f"{k}={v}\n" for k, v in bench.items()))
    (a.run_dir / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    (a.run_dir / "plan.json").write_text(json.dumps({"config": str(a.config), "nodes": nodes, "procs": record,
                                                     "bench": bench}, indent=1) + "\n")


if __name__ == "__main__":
    main()

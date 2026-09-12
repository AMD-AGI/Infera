from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agentx_env = load_module("agentx_env", ROOT / "tools" / "agentx_env.py")
collect_agentx = load_module("collect_agentx", ROOT / "tools" / "collect_agentx.py")
smoke = load_module("smoke", ROOT / "eval" / "smoke.py")


class ConfigAndTopologyTests(unittest.TestCase):
    def test_local_config_and_topology_load(self):
        command = (
            "COMPONENT=test; source lib/common.sh; load_config; "
            "printf '%s %s %s\\n' \"$PREFILL_TP\" \"$DECODE_TP\" \"$(topology_count prefill)\"; "
            "topology_rows"
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "4 8 1")
        self.assertIn("prefill-0\x1fprefill\x1fcrsuse2-m2m-137", lines[1])
        self.assertIn("decode-0\x1fdecode\x1fcrsuse2-m2m-138", lines[2])

    def test_command_line_override_is_validated(self):
        result = subprocess.run(
            [
                "bash",
                "-c",
                "COMPONENT=test; source lib/common.sh; load_config PREFILL_TP=8",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TP(8) with DPA", result.stderr)

    def test_empty_command_line_override_wins(self):
        result = subprocess.run(
            [
                "bash",
                "-c",
                "COMPONENT=test; source lib/common.sh; "
                "load_config HOST_RDMA_LIB=; printf '<%s>' \"$HOST_RDMA_LIB\"",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.stdout, "<>")

    def test_every_shell_script_parses(self):
        scripts = sorted(ROOT.glob("*.sh"))
        scripts += sorted((ROOT / "eval").glob("*.sh"))
        scripts += sorted((ROOT / "lib").glob("*.sh"))
        subprocess.run(["bash", "-n", *map(str, scripts)], check=True)

    def test_image_build_layers_infera_on_pinned_rocm_base(self):
        script = (ROOT / "build_image.sh").read_text(encoding="utf-8")
        self.assertIn('$ROCM_LLM_BENCH_DIR/Dockerfile', script)
        self.assertIn('SGLANG_BASE_IMAGE=$ROCM_OPT_BASE_IMAGE', script)
        self.assertLess(
            script.index("building optimized ROCm base"),
            script.index("building Infera PD engine"),
        )
        config = (ROOT / "config.sh").read_text(encoding="utf-8")
        self.assertIn("ROCM_OPT_BASE_IMAGE=", config)
        self.assertIn("glm52-v518-c29bd17-b02ab81", config)

    def test_launch_ssh_cannot_consume_topology_rows(self):
        script = (ROOT / "launch.sh").read_text(encoding="utf-8")
        self.assertIn('"${remote_overrides[@]}" </dev/null', script)
        self.assertIn("--listen-peer-urls", script)
        self.assertIn("$ETCD_PEER_PORT", script)

    def test_preflight_supports_asymmetric_pd_gpu_counts(self):
        script = (ROOT / "preflight.sh").read_text(encoding="utf-8")
        self.assertIn("prefill_gpu_count < decode_gpu_count", script)
        self.assertNotIn("requires equal P/D GPU counts", script)


def service_fixture():
    env = {
        "CONTAINER_PREFIX": "glm52",
        "ENGINE_PORT_BASE": "29001",
        "BOOTSTRAP_PORT_BASE": "8998",
        "KV_EVENT_PORT_BASE": "5557",
        "SNAPSHOT_PORT_BASE": "8801",
        "SERVED_MODEL": "glm5.2-mxfp4",
        "MODEL": "/models/glm",
        "IMAGE": "infera:test",
        "INFERENCEX_DIR": "/repo/InferenceX",
        "AGENTX_FAILED_REQUEST_THRESHOLD": "0.10",
        "KV_P2P_TRANSFER": "mooncake",
        "PREFILL_TP": "4",
        "PREFILL_EP": "4",
        "PREFILL_DP": "1",
        "PREFILL_DPA": "0",
        "PREFILL_HICACHE": "1",
        "PREFILL_MAX_RUNNING": "32",
        "DECODE_TP": "4",
        "DECODE_EP": "4",
        "DECODE_DP": "1",
        "DECODE_DPA": "0",
        "DECODE_HICACHE": "0",
        "DECODE_MTP": "1",
        "DECODE_MAX_RUNNING": "32",
        "DECODE_SIMULATE_ACC_LEN": "3.61",
        "TOTAL_CPU_DRAM_GB": "",
    }
    topology = [
        {
            "instance": "prefill-0",
            "role": "prefill",
            "node": "node-p",
            "ip": "10.0.0.1",
            "url": "http://10.0.0.1:29001",
            "ports": [29001, 8998, 5557, 8801],
            "container": "glm52-prefill-0",
        },
        {
            "instance": "decode-0",
            "role": "decode",
            "node": "node-d",
            "ip": "10.0.0.2",
            "url": "http://10.0.0.2:29002",
            "ports": [29002, 8999, 5558, 8802],
            "container": "glm52-decode-0",
        },
    ]
    workers = {
        "workers": [
            {
                "worker_id": "runtime-d",
                "url": "http://10.0.0.2:29002",
                "disagg_mode": "decode",
            },
            {
                "worker_id": "runtime-p",
                "url": "http://10.0.0.1:29001",
                "disagg_mode": "prefill",
            },
        ]
    }
    infos = {
        name: {
            "served_model_name": "glm5.2-mxfp4",
            "tp_size": 4,
            "ep_size": 4,
            "dp_size": 1,
        }
        for name in ("prefill-0", "decode-0")
    }

    def container(role: str):
        command = [
            "python3",
            "-m",
            "infera.engine.sglang",
            "--model-path",
            "/models/glm",
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--max-running-requests",
            "32",
            "--disaggregation-transfer-backend",
            "mooncake",
        ]
        environment = []
        if role == "prefill":
            command.append("--enable-hierarchical-cache")
        else:
            command += ["--speculative-algorithm", "EAGLE"]
            environment.append("SGLANG_SIMULATE_ACC_LEN=3.61")
        return {
            "Image": f"sha256:{role}",
            "Config": {"Cmd": command, "Env": environment},
        }

    containers = {
        "prefill-0": container("prefill"),
        "decode-0": container("decode"),
    }
    hardware = {"node-p": "mi355x", "node-d": "mi355x"}
    dram = {"node-p": 3023, "node-d": 3023}
    return env, topology, workers, infos, containers, hardware, dram


class AgentXEnvironmentTests(unittest.TestCase):
    def make(self, **changes):
        env, topology, workers, infos, containers, hardware, dram = service_fixture()
        env.update(changes.pop("env", {}))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            return agentx_env.make_runtime_env(
                topology=topology,
                workers_payload=workers,
                infos=infos,
                containers=containers,
                hardware=hardware,
                dram=dram,
                env=env,
                router_url="http://10.0.0.2:8000",
                concurrency=8,
                duration=3600,
                output_dir=root / "out",
                runtime_dir=root / "cache" / "aiperf",
                hf_home=root / "cache" / "hf",
                **changes,
            )

    def test_runtime_env_comes_from_live_service(self):
        values = self.make()
        self.assertEqual(values["PREFILL_NUM_WORKERS"], "1")
        self.assertEqual(values["DECODE_NUM_WORKERS"], "1")
        self.assertEqual(values["PREFILL_TP"], "4")
        self.assertEqual(values["SPEC_DECODING"], "mtp")
        self.assertEqual(values["KV_OFFLOADING"], "dram")
        self.assertEqual(values["RUNNER_TYPE"], "mi355x")
        self.assertEqual(values["TOTAL_CPU_DRAM_GB"], "3023")
        self.assertEqual(values["MODEL"], "/models/glm")
        self.assertIn("sha256:prefill", values["IMAGE_IDS"])
        self.assertIn("http://10.0.0.1:29001/metrics", values["AIPERF_SERVER_METRICS_URLS"])

    def test_live_drift_fails_before_agentx(self):
        env, topology, workers, infos, containers, hardware, dram = service_fixture()
        command = containers["decode-0"]["Config"]["Cmd"]
        command[command.index("--max-running-requests") + 1] = "64"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(agentx_env.InspectError, "live max_running=64"):
                agentx_env.make_runtime_env(
                    topology=topology,
                    workers_payload=workers,
                    infos=infos,
                    containers=containers,
                    hardware=hardware,
                    dram=dram,
                    env=env,
                    router_url="http://10.0.0.2:8000",
                    concurrency=8,
                    duration=3600,
                    output_dir=root / "out",
                    runtime_dir=root / "cache" / "aiperf",
                    hf_home=root / "cache" / "hf",
                )


class AgentXAnalysisTests(unittest.TestCase):
    def test_multi_pd_gpu_denominator_and_topology(self):
        record = {
            "model": "/models/GLM-5.2-MXFP4",
            "hw": "mi355x",
            "framework": "sglang",
            "disagg": True,
            "prefill_num_workers": 2,
            "prefill_tp": 4,
            "prefill_ep": 4,
            "num_prefill_gpu": 8,
            "decode_num_workers": 2,
            "decode_tp": 4,
            "decode_ep": 4,
            "num_decode_gpu": 8,
            "conc": 8,
            "request_metrics": {
                "throughput": {
                    "total": {"tokens_per_second": 1600},
                    "input": {"tokens_per_second": 1440},
                    "output": {"tokens_per_second": 160},
                    "per_gpu": {"total_tput_tps": 100},
                },
                "latency": {
                    "full_response_itl": {"p50": 0.01, "p90": 0.02},
                    "ttft": {"p50": 1.5},
                },
            },
            "num_requests_successful": 10,
            "num_requests_total": 10,
            "request_accounting": {"records_error_dropped": 0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            point = Path(temporary)
            (point / "agentx_conc8.json").write_text(json.dumps(record))
            row = collect_agentx.one(point, "c8")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["Total GPUs"], 16)
        self.assertEqual(row["Token Throughput per Chip (tok/s/chip)"], 100)
        self.assertEqual(row["Topology"], "2P2D pTP4/EP4 dTP4/EP4")

    def test_bundled_reference_is_newer_than_august_snapshot(self):
        path = ROOT / "tools" / "ref" / "InferenceX_GLM-5.2_interactivity.csv"
        with path.open(encoding="utf-8") as stream:
            rows = list(
                csv.DictReader(line for line in stream if not line.startswith("#"))
            )
        self.assertGreaterEqual(len(rows), 50)
        self.assertGreater(max(row["Date"] for row in rows), "2026-08-21")


class SmokeHandler(BaseHTTPRequestHandler):
    workers = {"workers": []}

    def log_message(self, *_):
        return

    def send_json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        elif self.path == "/v1/workers":
            self.send_json(self.workers)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if payload.get("tools"):
            message = {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        }
                    }
                ]
            }
        else:
            content = payload["messages"][0]["content"]
            message = {"content": "Jupiter" if "largest planet" in content else "101"}
        self.send_json({"choices": [{"message": message}]})


class SmokeTests(unittest.TestCase):
    def test_direct_router_smoke(self):
        SmokeHandler.workers = {
            "workers": [
                {"disagg_mode": "prefill"},
                {"disagg_mode": "decode"},
                {"disagg_mode": "decode"},
            ]
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            artifacts = smoke.run(
                f"http://127.0.0.1:{server.server_address[1]}",
                "glm",
                prefill_count=1,
                decode_count=2,
                timeout=5,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(artifacts["concurrent_burst"]["requests"], 24)
        self.assertIn("tool_call", artifacts)

    def test_gsm8k_is_independent_from_agentx_environment(self):
        text = (ROOT / "eval" / "gsm8k.sh").read_text(encoding="utf-8")
        self.assertNotIn("agentx_env.py", text)
        self.assertIn("run_lm_eval", text)
        self.assertIn("SGLANG_SIMULATE_ACC_LEN", text)

    def test_engine_sets_long_http_keep_alive(self):
        text = (ROOT / "engine.sh").read_text(encoding="utf-8")
        self.assertIn("SGLANG_TIMEOUT_KEEP_ALIVE", text)
        self.assertIn("--json-model-override-args", text)


if __name__ == "__main__":
    unittest.main()

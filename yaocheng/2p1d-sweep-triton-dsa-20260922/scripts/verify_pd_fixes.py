#!/usr/bin/env python3
"""Inspect the pinned PD image without importing torch or starting an inference worker.

Host mode is invoked through verify_pd_fixes.sh; --inside-image reads source files,
compiles Python to memory, and checks the router CLI. These checks establish patch
presence, not GPU correctness or the current upstream merge status.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def inspect_sources(root):
    checks, files = {}, {}

    def require(name, relative, needles, counts=None):
        path = root / relative
        try:
            source = path.read_text()
            files[relative] = hashlib.sha256(source.encode()).hexdigest()
            # AST serialization removes comments, so a commented-out patch cannot pass.
            text = ast.unparse(ast.parse(source)) if path.suffix == ".py" else source
            if path.suffix == ".py":
                compile(source, str(path), "exec")
            missing = [needle for needle in needles if needle not in text]
            missing.extend(f"{needle} (need {count} occurrences)"
                           for needle, count in (counts or {}).items() if text.count(needle) < count)
            checks[name] = {"passed": not missing, "file": relative, "missing": missing}
        except (OSError, SyntaxError) as exc:
            checks[name] = {"passed": False, "file": relative, "error": str(exc)}

    require("dsa_padded_rows", "srt/layers/attention/dsa/dsa_indexer.py",
            ["_p1v2_rows", "_p1v2_trim", "_p1v2_clip"])
    require("dsa_no_rank_local_host_sync", "srt/layers/attention/dsa_backend.py",
            ["max_seqlen_k = self.req_to_token.shape[1]", "must not be None for DRAFT_EXTEND_V2"])
    require("dsa_page_table_rows", "srt/layers/attention/dsa_backend.py",
            ["def _glm52_match_page_table_rows"], {"_glm52_match_page_table_rows(": 3})
    for name, relative, marker in [
        ("draft_vote_batch", "srt/managers/schedule_batch.py", "force_disable_draft_cuda_graph"),
        ("draft_vote_collective", "srt/managers/scheduler_components/dp_attn.py", "can_run_draft_cuda_graph"),
        ("draft_vote_forward", "srt/model_executor/forward_batch_info.py", "can_run_dp_draft_cuda_graph"),
        ("draft_vote_scheduler", "srt/disaggregation/decode.py", "force_disable_draft_cuda_graph"),
        ("draft_vote_consumer", "srt/speculative/eagle_draft_cuda_graph_runner.py", "can_run_dp_draft_cuda_graph"),
        ("draft_vote_base", "srt/speculative/base_spec_worker.py", "requires_dp_attention_eager_forward"),
        ("draft_vote_eagle", "srt/speculative/eagle_worker_v2.py", "requires_dp_attention_eager_forward"),
    ]:
        require(name, relative, [marker])
    require("mooncake_wait_event_field", "srt/disaggregation/common/utils.py", ["wait_event"])
    require("mooncake_wait_before_read", "srt/disaggregation/mooncake/conn.py",
            ["kv_chunk.wait_event.synchronize()", "wait_event=wait_event", "_early_send_wait_event"])
    require("mooncake_overlap_event", "srt/disaggregation/prefill.py", [],
            {"req.disagg_kv_sender._early_send_wait_event = ev": 2})
    require("mooncake_failed_session_recovery", "srt/disaggregation/mooncake/conn.py",
            ["_failed_session_probe_loop", "_run_one_probe_pass", "failed_sessions.discard(session_id)",
             "SGLANG_ENABLE_FAILED_SESSION_PROBE"])
    require("pd_rejection_guard", "srt/arg_groups/speculative_hook.py",
            ["is_hip=get_platform().is_hip and cfg.disaggregation_mode != 'decode'"])
    require("responses_bootstrap_fields", "srt/entrypoints/openai/protocol.py", ["class ResponsesRequest"],
            {"bootstrap_host:": 3, "bootstrap_port:": 3, "bootstrap_room:": 3})
    require("responses_bootstrap_forwarding", "srt/entrypoints/openai/serving_responses.py",
            ["bootstrap_host=request.bootstrap_host", "bootstrap_port=request.bootstrap_port",
             "bootstrap_room=request.bootstrap_room"])
    require("hicache_copy_geometry", "kernels/jit/csrc/kvcacheio/hicache.cuh",
            ["kCopyGroupThreads = 32", "copy_lanes_per_worker", "pick_group_bytes"])
    require("hicache_copy_dispatch", "kernels/ops/kvcache/hicache.py", ["def _tiles_across_lanes"])
    require("hicache_k_only_hip", "srt/mem_cache/pool_host/mha.py", [],
            {"self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(": 2})
    require("hicache_staged_writeback_gate", "srt/mem_cache/pool_host/mla.py",
            ["self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel("])
    require("nextn_shared_experts_fusion", "srt/models/glm4_moe.py",
            ["fused_shared_experts_architecture = 'GlmMoeDsaForCausalLMNextN'"])
    return checks, files


def inside_image():
    root = Path("/sgl-workspace/sglang/python/sglang")
    checks, files = inspect_sources(root)
    cli = subprocess.run(["/usr/local/bin/infera-router", "--help"], text=True,
                         capture_output=True, timeout=30)
    checks["router_rank_affinity"] = {
        "passed": cli.returncode == 0 and "--pd-dp-rank-affinity" in cli.stdout
                  and "INFERA_PD_DP_RANK_AFFINITY" in cli.stdout,
        "returncode": cli.returncode,
    }
    versions = {}
    for package in ("sglang", "aiter", "mooncake-transfer-engine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    head = subprocess.run(["git", "-C", str(root.parents[1]), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=15)
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "source_sha256": files, "versions": versions,
            "sglang_checkout_head": head.stdout.strip() if head.returncode == 0 else None}


def host(output):
    env = os.environ
    with Path(env["TOPOLOGY"]).open() as stream:
        nodes = [row["node"] for row in csv.DictReader(stream, delimiter="\t")]
    if env.get("VERIFY_NODES"):
        nodes = env["VERIFY_NODES"].split()
    output.mkdir(parents=True, exist_ok=True)
    ssh_options = shlex.split(env["SSH_OPTS"])
    source = Path(__file__).read_text()
    results = {}
    for index, node in enumerate(dict.fromkeys(nodes)):
        try:
            inspect = subprocess.run(
                ["ssh", "-n", *ssh_options, node,
                 shlex.join(["docker", "image", "inspect", "--format", "{{.Id}}", env["IMAGE"]])],
                text=True, capture_output=True, timeout=45, check=True)
            image_id = inspect.stdout.strip()
            if env.get("EXPECTED_IMAGE_ID") and image_id != env["EXPECTED_IMAGE_ID"]:
                raise ValueError(f"image ID mismatch: {image_id}")
            command = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only",
                       "--cpus", "1", "--memory", "1g",
                       "--name", f'{env["CONTAINER_PREFIX"]}-patch-check-{os.getpid()}-{index}',
                       "--entrypoint", "python3", env["IMAGE"], "-B", "-", "--inside-image"]
            proc = subprocess.run(["ssh", *ssh_options, node, shlex.join(command)], input=source,
                                  text=True, capture_output=True, timeout=120)
            if not proc.stdout.strip():
                raise ValueError(proc.stderr.strip() or f"probe exited {proc.returncode}")
            result = json.loads(proc.stdout)
            result.update(node=node, image_id=image_id, returncode=proc.returncode)
            result["passed"] = result["passed"] and proc.returncode == 0
            if proc.stderr:
                result["stderr"] = proc.stderr
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            result = {"node": node, "passed": False, "error": detail.strip()}
        results[node] = result
        safe_name = node.replace("/", "_")
        (output / f"{safe_name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f'{"PASS" if result["passed"] else "FAIL"} {node}: {output / (safe_name + ".json")}')
    summary = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "scope": "CPU-only image/source checks; no GPU execution", "nodes": results,
               "passed": bool(results) and all(r["passed"] for r in results.values())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-image", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.inside_image:
        result = inside_image()
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    if not args.output:
        parser.error("host mode requires --output")
    return host(args.output)


if __name__ == "__main__":
    raise SystemExit(main())

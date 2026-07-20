###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""End-to-end greedy-generation comparison: Arm A (vanilla vLLM,
no kv_transfer_config) vs Arm B (InferaKvdConnector + hipFile
GPU-direct save/load path) on Kimi K2.5 MXFP4 TP=4.

Goal: verify that the chunked-fusion async-save path is bit-equivalent
to vanilla for token IDs at temperature=0. Any mismatch is a SHIP
BLOCKER — it would mean KV bytes coming back from disk differ from
what the model itself just produced.

RUN MODEL: this script runs INSIDE the jj_vllm_gptoss container
(it shells out via subprocess to launch the two vllm serve processes,
launch the kvd daemon, and HTTP-poll both endpoints). Invoke via:

    ssh <host> \\
      'docker exec -w /opt/infera jj_vllm_gptoss bash -lc \\
       "PYTHONPATH=$INFERA_ROOT python \\
        $INFERA_ROOT/bench/kvcache/correctness/compare_generation.py"'

Test plan:
  - launch kvd on /tmp/kvd-kimi-compare.sock backed by /mnt/nvme8
  - launch Arm A on GPUs 2,3,5,6 :: port 8804 (vanilla, no connector)
  - launch Arm B on GPUs 0,1,4,7 :: port 8803 (kvd + GPU_DIRECT=true)
  - send 5 prompts × 3 runs to BOTH arms, greedy, max_tokens=50, seed=0
  - decode token IDs via Kimi tokenizer, find first divergence point
  - report (prompt_idx, run_idx) → mismatch_count + first divergence

Cleanup: we record exact PIDs (kvd, both vLLM servers) and SIGTERM
+ wait + SIGKILL on exit. Never broad-pkill.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------
# Config — knobs at the top so it's easy to skim and tweak
# ----------------------------------------------------------------------

# Repo root, auto-derived from this file's location
# (bench/kvcache/correctness/compare_generation.py -> parents[3]).
# Override with INFERA_ROOT env if running from a relocated checkout.
INFERA_ROOT = os.environ.get("INFERA_ROOT", str(Path(__file__).resolve().parents[3]))
MODEL_PATH = "/PATH/TO/amd-Kimi-K2.5-MXFP4"
SERVED_NAME = "kimi-k2.5"
MAX_MODEL_LEN = 65536
KV_CACHE_BYTES = 26 << 30  # 26 GiB; below working set so kvd reads fire
GPU_MEM_UTIL = 0.9

ARM_A = {
    "name": "vanilla",
    "port": 8804,
    "gpus": "2,3,5,6",
    "log": "/tmp/vllm-compare-arm-a.log",
}
ARM_B = {
    "name": "kvd_gpu_direct",
    "port": 8803,
    "gpus": "0,1,4,7",
    "log": "/tmp/vllm-compare-arm-b.log",
}

KVD_SOCKET = "/tmp/kvd-kimi-compare.sock"
KVD_LONG_DIR = "/mnt/nvme8/kvd-compare-long"
KVD_SHORT_DIR = "/mnt/nvme8/kvd-compare-short"
KVD_LOG = "/tmp/kvd-compare.log"

PROMPT_TOKEN_TARGETS = [100, 1000, 5000, 18000, 30000]
RUNS_PER_PROMPT = 3
GEN_TOKENS = 50

# Where to dump the comparison JSON (host-visible).
RESULT_PATH = (
    Path(INFERA_ROOT) / "bench" / "correctness" / "results" / "compare_generation_latest.json"
)
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

READY_TIMEOUT_S = 1200  # Kimi K2.5 cold-load + compile takes ~10 min
POLL_INTERVAL_S = 5


# ----------------------------------------------------------------------
# Process management — local subprocess (we run INSIDE the container)
# ----------------------------------------------------------------------


def spawn(cmd: list[str], *, env: dict, log_path: str) -> subprocess.Popen:
    """Launch cmd as a process group (so we can clean up children too),
    tee stdout/stderr to log_path. Returns the Popen handle; caller
    keeps it for liveness checks and termination."""
    logf = open(log_path, "wb")
    p = subprocess.Popen(
        cmd,
        env={**os.environ, **env},
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return p


def terminate(p: subprocess.Popen | None, label: str) -> None:
    if p is None:
        return
    if p.poll() is not None:
        return
    print(f"[cleanup] SIGTERM pgid={p.pid} ({label})", flush=True)
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 30
    while time.time() < deadline:
        if p.poll() is not None:
            return
        time.sleep(0.5)
    print(f"[cleanup] SIGKILL pgid={p.pid} ({label}) — didn't exit on TERM", flush=True)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def http_post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_ok(url: str, timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def wait_for_arm(port: int, *, label: str, log_path: str, p: subprocess.Popen) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    url = f"http://127.0.0.1:{port}/v1/models"
    last_log_line = ""
    while time.time() < deadline:
        if http_get_ok(url):
            print(f"[ready] {label} responding on {url}", flush=True)
            return
        if p.poll() is not None:
            tail = _tail(log_path, 80)
            raise RuntimeError(
                f"{label} pid={p.pid} died before ready (exit={p.returncode})\n"
                f"--- log tail ---\n{tail}"
            )
        cur = _tail(log_path, 1).strip()
        if cur and cur != last_log_line:
            print(f"[wait:{label}] {cur[:160]}", flush=True)
            last_log_line = cur
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(f"{label} on {url} did not respond within {READY_TIMEOUT_S}s")


def _tail(path: str, n: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            chunk = min(size, 64 * 1024)
            fh.seek(size - chunk)
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
    except FileNotFoundError:
        return ""


# ----------------------------------------------------------------------
# Launchers
# ----------------------------------------------------------------------

COMMON_ENV = {
    "AMDGCN_USE_BUFFER_OPS": "0",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_TRITON_ROPE": "1",
    "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
    "HSA_NO_SCRATCH_RECLAIM": "1",
    "VLLM_RPC_TIMEOUT": "1800000",
    "PYTHONHASHSEED": "0",
}


def launch_kvd() -> subprocess.Popen:
    # Remove stale socket if any.
    try:
        os.unlink(KVD_SOCKET)
    except FileNotFoundError:
        pass
    os.makedirs(KVD_LONG_DIR, exist_ok=True)
    os.makedirs(KVD_SHORT_DIR, exist_ok=True)
    env = {"PYTHONPATH": INFERA_ROOT}
    cmd = [
        sys.executable,
        "-m",
        "infera.kvd",
        "--socket",
        KVD_SOCKET,
        "--max-bytes",
        str(4 << 30),
        "--long-path",
        KVD_LONG_DIR,
        "--long-bytes",
        str(20 << 30),
        "--spillover-path",
        KVD_SHORT_DIR,
        "--spillover-bytes",
        str(20 << 30),
    ]
    p = spawn(cmd, env=env, log_path=KVD_LOG)
    # Wait up to 30s for the socket to appear.
    for _ in range(60):
        if os.path.exists(KVD_SOCKET):
            print(f"[kvd] up pid={p.pid} socket={KVD_SOCKET}", flush=True)
            return p
        if p.poll() is not None:
            raise RuntimeError(f"kvd died at startup (exit={p.returncode})\n{_tail(KVD_LOG, 60)}")
        time.sleep(0.5)
    raise RuntimeError(f"kvd socket {KVD_SOCKET} didn't appear in 30s")


def launch_arm_a() -> subprocess.Popen:
    """Vanilla vLLM, no kv_transfer_config. Same model+tp+max_len as Arm B
    but no connector → uses ONLY in-VRAM prefix cache, no kvd I/O."""
    env = dict(COMMON_ENV)
    env["HIP_VISIBLE_DEVICES"] = ARM_A["gpus"]
    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--served-model-name",
        SERVED_NAME,
        "--tensor-parallel-size",
        "4",
        "--port",
        str(ARM_A["port"]),
        "--host",
        "0.0.0.0",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEM_UTIL),
        "--kv-cache-memory-bytes",
        str(KV_CACHE_BYTES),
        "--trust-remote-code",
    ]
    p = spawn(cmd, env=env, log_path=ARM_A["log"])
    print(f"[arm-a] launching pid={p.pid} on GPUs {ARM_A['gpus']} port {ARM_A['port']}", flush=True)
    return p


def launch_arm_b() -> subprocess.Popen:
    """Connector + hipFile GPU-direct save/load path."""
    env = dict(COMMON_ENV)
    env["HIP_VISIBLE_DEVICES"] = ARM_B["gpus"]
    env["PYTHONPATH"] = INFERA_ROOT
    env["INFERA_KVD_SOCKET"] = KVD_SOCKET
    env["INFERA_KVD_HIPFILE_ROOTS"] = f"long={KVD_LONG_DIR},short={KVD_SHORT_DIR}"
    env["INFERA_KVD_CHUNK_TOKENS"] = "512"
    env["INFERA_KVD_GPU_DIRECT"] = "true"
    kv_transfer = json.dumps(
        {
            "kv_connector": "InferaKvdConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "infera.engine.vllm.kvd_connector",
        }
    )
    cmd = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--served-model-name",
        SERVED_NAME,
        "--tensor-parallel-size",
        "4",
        "--port",
        str(ARM_B["port"]),
        "--host",
        "0.0.0.0",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEM_UTIL),
        "--kv-cache-memory-bytes",
        str(KV_CACHE_BYTES),
        "--trust-remote-code",
        "--kv-transfer-config",
        kv_transfer,
    ]
    p = spawn(cmd, env=env, log_path=ARM_B["log"])
    print(f"[arm-b] launching pid={p.pid} on GPUs {ARM_B['gpus']} port {ARM_B['port']}", flush=True)
    return p


# ----------------------------------------------------------------------
# Prompt construction + comparison
# ----------------------------------------------------------------------


def make_prompts(tokenizer) -> list[tuple[int, str]]:
    """Build prompts that hit roughly each target token length. Uses
    repeated lorem-style words so the tokenizer sees a deterministic
    stream and avoids any tool-call / JSON special tokens — they're
    off the critical path for KV bit-equality testing."""
    base = (
        "The quick brown fox jumps over the lazy dog near a quiet river. "
        "It paused to watch a small bird gather twigs for a nest. "
        "After a moment of rest, the fox continued along the bank. "
    )
    prompts: list[tuple[int, str]] = []
    for target in PROMPT_TOKEN_TARGETS:
        text = ""
        while True:
            text += base
            n_tok = len(tokenizer.encode(text, add_special_tokens=False))
            if n_tok >= target:
                break
            if len(text) > 10 * target * 8:  # safety belt
                break
        prompts.append((target, text))
    return prompts


def query_arm(port: int, prompt: str, seed: int) -> dict:
    """Hit /v1/completions with greedy + seed → returns full response."""
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": SERVED_NAME,
        "prompt": prompt,
        "max_tokens": GEN_TOKENS,
        "temperature": 0,
        "top_p": 1.0,
        "seed": seed,
        # logprobs=1 gives per-token IDs to compare exactly.
        "logprobs": 1,
        "stream": False,
    }
    return http_post_json(url, payload, timeout=600.0)


def extract_token_ids(resp: dict, tokenizer) -> list[int]:
    """Pull generated token ids out of a /v1/completions response.
    vLLM's logprobs object carries piece-form strings under
    `logprobs.tokens`; we re-encode them to get vocab IDs. Falls back
    to encoding the generated text when logprobs isn't populated."""
    choice = resp["choices"][0]
    text = choice.get("text", "")
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or []
    ids: list[int] = []
    for tok in tokens:
        try:
            sub = tokenizer.encode(tok, add_special_tokens=False)
            ids.extend(sub)
        except Exception:
            pass
    if not ids and text:
        ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:GEN_TOKENS]


# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------


@dataclass
class Launched:
    kvd: subprocess.Popen | None = None
    arm_a: subprocess.Popen | None = None
    arm_b: subprocess.Popen | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-launch", action="store_true", help="assume both arms already running"
    )
    parser.add_argument("--arm-a-port", type=int, default=None)
    parser.add_argument("--arm-b-port", type=int, default=None)
    args = parser.parse_args()
    if args.arm_a_port:
        ARM_A["port"] = args.arm_a_port
    if args.arm_b_port:
        ARM_B["port"] = args.arm_b_port

    launched = Launched()
    try:
        if not args.skip_launch:
            launched.kvd = launch_kvd()
            launched.arm_a = launch_arm_a()
            launched.arm_b = launch_arm_b()
            wait_for_arm(ARM_A["port"], label="arm-a", log_path=ARM_A["log"], p=launched.arm_a)
            wait_for_arm(ARM_B["port"], label="arm-b", log_path=ARM_B["log"], p=launched.arm_b)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        prompts = make_prompts(tokenizer)
        actual_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for _, p in prompts]
        print(f"[prompts] built {len(prompts)}; lengths (tokens) = {actual_lens}", flush=True)

        results: list[dict] = []
        mismatch_total = 0
        for p_idx, (target, prompt) in enumerate(prompts):
            for run_idx in range(RUNS_PER_PROMPT):
                seed = 0
                print(f"[probe] prompt={p_idx}(target={target}) run={run_idx}", flush=True)
                try:
                    resp_a = query_arm(ARM_A["port"], prompt, seed=seed)
                    resp_b = query_arm(ARM_B["port"], prompt, seed=seed)
                except Exception as exc:
                    print(f"  ! request failed: {exc}", flush=True)
                    results.append(
                        {
                            "prompt_idx": p_idx,
                            "target_tokens": target,
                            "run_idx": run_idx,
                            "error": str(exc),
                        }
                    )
                    continue
                ids_a = extract_token_ids(resp_a, tokenizer)
                ids_b = extract_token_ids(resp_b, tokenizer)

                first_div = None
                for i, (a, b) in enumerate(zip(ids_a, ids_b)):
                    if a != b:
                        first_div = i
                        break
                if first_div is None and len(ids_a) != len(ids_b):
                    first_div = min(len(ids_a), len(ids_b))

                mismatch = sum(1 for a, b in zip(ids_a, ids_b) if a != b)
                mismatch += abs(len(ids_a) - len(ids_b))
                mismatch_total += mismatch

                divergence_dump: list[dict] = []
                if first_div is not None:
                    end = min(first_div + 10, max(len(ids_a), len(ids_b)))
                    for i in range(first_div, end):
                        a = ids_a[i] if i < len(ids_a) else None
                        b = ids_b[i] if i < len(ids_b) else None
                        try:
                            a_txt = tokenizer.decode([a]) if a is not None else "<none>"
                        except Exception:
                            a_txt = "<decode-err>"
                        try:
                            b_txt = tokenizer.decode([b]) if b is not None else "<none>"
                        except Exception:
                            b_txt = "<decode-err>"
                        divergence_dump.append(
                            {
                                "offset": i,
                                "a_id": a,
                                "b_id": b,
                                "a_txt": a_txt,
                                "b_txt": b_txt,
                            }
                        )

                results.append(
                    {
                        "prompt_idx": p_idx,
                        "target_tokens": target,
                        "actual_prompt_tokens": actual_lens[p_idx],
                        "run_idx": run_idx,
                        "seed": seed,
                        "mismatch_count": mismatch,
                        "first_divergence_offset": first_div,
                        "arm_a_len": len(ids_a),
                        "arm_b_len": len(ids_b),
                        "divergence_first10": divergence_dump,
                    }
                )
                print(
                    f"  -> mismatch={mismatch} first_div={first_div} "
                    f"len_a={len(ids_a)} len_b={len(ids_b)}",
                    flush=True,
                )

        summary = {
            "prompts": PROMPT_TOKEN_TARGETS,
            "runs_per_prompt": RUNS_PER_PROMPT,
            "gen_tokens": GEN_TOKENS,
            "total_pairs": len(results),
            "total_mismatch_token_count": mismatch_total,
            "pairs_with_any_divergence": sum(1 for r in results if r.get("mismatch_count", 0) > 0),
            "results": results,
        }
        RESULT_PATH.write_text(json.dumps(summary, indent=2))
        print(f"\n[done] wrote {RESULT_PATH}", flush=True)
        print(
            f"[done] {summary['pairs_with_any_divergence']}/{summary['total_pairs']} pairs diverged; "
            f"total token mismatches = {mismatch_total}",
            flush=True,
        )
        return 0 if mismatch_total == 0 else 1
    finally:
        terminate(launched.arm_b, "arm-b vllm")
        terminate(launched.arm_a, "arm-a vllm")
        terminate(launched.kvd, "kvd")


if __name__ == "__main__":
    sys.exit(main())

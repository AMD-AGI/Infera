#!/usr/bin/env python3
"""Validate SGLang #46918 through the production absorbed-Q BMM dispatcher."""

import argparse
import hashlib
import inspect
import json
import os
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HEAD = "9763ef56bf57d03f8af4f9adc9eebb717cd0d9b1"
ARMS = {
    "default": {"max_waves": "2", "disable_hand_config": "true"},
    "capped": {"max_waves": "2", "disable_hand_config": "false"},
    "uncapped": {"max_waves": "0", "disable_hand_config": "false"},
}


def checked_path(value):
    path = Path(value).resolve()
    if "yihou" not in str(path):
        raise ValueError(f"Artifact/source path must contain yihou: {path}")
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(checkout, *arguments):
    return subprocess.check_output(["git", "-C", str(checkout), *arguments], text=True).strip()


def verify_checkout(checkout):
    actual = git(checkout, "rev-parse", "HEAD")
    if actual != HEAD:
        raise RuntimeError(f"Expected candidate {HEAD}, got {actual}")
    changed = git(checkout, "diff", "HEAD", "--name-only", "--", "python")
    if changed:
        raise RuntimeError(f"Candidate Python source has local modifications:\n{changed}")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_tensor(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(dtype=__import__("torch").uint8).numpy().tobytes()
    ).hexdigest()


def error_metrics(torch, actual, reference):
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
    if not finite:
        return {"finite": False}
    delta = actual - reference
    rms = float(reference.square().mean().sqrt())
    rmse = float(delta.square().mean().sqrt())
    return {
        "finite": True,
        "exact_equal": bool(torch.equal(actual, reference)),
        "max_absolute_error": float(delta.abs().max()),
        "mean_absolute_error": float(delta.abs().mean()),
        "rmse": rmse,
        "reference_rms": rms,
        "relative_rmse": rmse / rms if rms else None,
    }


def run_worker(args):
    # Set before importing Torch, SGLang or Triton: a compiler abort must not dump a core.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    checkout = checked_path(args.checkout)
    out = checked_path(args.out)
    verify_checkout(checkout)

    import torch
    import triton

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("An allocated ROCm GPU is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one allocated GPU before running the harness")
    props = torch.cuda.get_device_properties(0)
    architecture = str(getattr(props, "gcnArchName", ""))
    if architecture.split(":")[0] != "gfx950":
        raise RuntimeError(f"Issue reproduction requires gfx950, got {architecture}")

    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        forward_mla_rocm as module,
    )

    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(checkout / "python"):
        raise RuntimeError(f"Imported SGLang outside candidate checkout: {module_path}")
    if not module._use_aiter_gfx95:
        raise RuntimeError("Production AITER gfx95 path is disabled; refusing torch fallback")
    kernel = module.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant
    kernel_path = Path(inspect.getfile(kernel)).resolve()
    # Fingerprint the JIT functions actually imported by the wrapper, not a
    # guessed source location that might belong to a different AITER install.
    implementation_paths = set()
    for value in inspect.unwrap(kernel).__globals__.values():
        function = value
        # AITER stacks Triton Heuristics and JITFunction wrappers.
        for _ in range(8):
            if inspect.isfunction(function) or not hasattr(function, "fn"):
                break
            function = function.fn
        if inspect.isfunction(function) and function.__module__.startswith(
            "aiter.ops.triton._triton_kernels."
        ):
            implementation_paths.add(Path(inspect.getfile(function)).resolve())
    if not implementation_paths:
        raise RuntimeError(
            "Could not fingerprint the AITER wrapper's imported Triton implementation"
        )
    config = module._absorb_bmm_config(64, args.rows, 512)
    if args.arm == "default" and config is not None:
        raise RuntimeError(f"config=None arm unexpectedly selected {config}")
    if args.rows == 256 and args.arm != "default":
        expected = 4 if args.arm == "uncapped" else 2
        if config is None or config["waves_per_eu"] != expected:
            raise RuntimeError(f"Expected waves_per_eu={expected}, got {config}")
    if args.rows > 256 and config is not None:
        raise RuntimeError("Fallback boundary must select config=None")

    torch.manual_seed(args.seed)
    # CPU-generated inputs make same-seed comparisons independent of GPU RNG state.
    q_storage = torch.randn((args.rows, 64, 256), dtype=torch.bfloat16).cuda()
    q = q_storage[:, :, :192]
    weights = torch.randn((64, 192, 512), dtype=torch.bfloat16).cuda()
    weights = weights.to(torch.float8_e4m3fn)
    scales = torch.ones((64,), device="cuda", dtype=torch.float32)
    attn = SimpleNamespace(w_kc=weights, w_scale=scales)
    metadata = {
        "status": "before_first_production_call",
        "candidate_head": HEAD,
        "arm": args.arm,
        "rows": args.rows,
        "seed": args.seed,
        "config": config,
        "device": props.name,
        "architecture": architecture,
        "compute_units": props.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "module_path": str(module_path),
        "module_sha256": digest_file(module_path),
        "aiter_wrapper_path": str(kernel_path),
        "aiter_wrapper_sha256": digest_file(kernel_path),
        "aiter_implementation_files": [
            {"path": str(path), "sha256": digest_file(path)}
            for path in sorted(implementation_paths)
        ],
        "q_shape": list(q.shape),
        "q_stride": list(q.stride()),
        "q_dtype": str(q.dtype),
        "q_sha256": digest_tensor(q),
        "weight_shape": list(weights.shape),
        "weight_stride": list(weights.stride()),
        "weight_dtype": str(weights.dtype),
        "weight_sha256": digest_tensor(weights),
        "scales": "all ones, FP32",
        "environment": {
            name: os.environ.get(name)
            for name in (
                "SGLANG_USE_AITER",
                "SGLANG_ROCM_ABSORB_BMM_MAX_WAVES_PER_EU",
                "SGLANG_ROCM_ABSORB_BMM_DISABLE_HAND_CONFIG",
                "TRITON_CACHE_DIR",
                "AITER_JIT_DIR",
                "AITER_ROOT_DIR",
                "AITER_CONFIG_GEMM_BF16",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
        },
    }
    write_json(out / "result.json", metadata)
    torch.cuda.synchronize()
    started = time.monotonic()
    result = module.rocm_absorb_q_bmm(attn, q, is_capture_mode=False)
    torch.cuda.synchronize()
    metadata["first_call_wall_seconds"] = time.monotonic() - started
    if tuple(result.shape) != (64, args.rows, 512):
        raise RuntimeError(f"Unexpected output shape: {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Production output contains nonfinite values")
    if result.dtype != torch.bfloat16:
        raise RuntimeError(f"Unexpected output dtype: {result.dtype}")

    # The stored FP8 weights are dequantized by float(); scales are exactly one.
    # This diagnostic intentionally excludes AITER's internal activation FP8 quantization.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    reference = torch.bmm(q.float().transpose(0, 1), weights.float())
    torch.cuda.synchronize()
    metadata["fp32_reference_diagnostics"] = error_metrics(torch, result, reference)
    metadata["reference_limitation"] = (
        "Reference uses BF16 activations and dequantized stored FP8 weights. "
        "AITER also quantizes activations internally. These errors are diagnostic; "
        "no unvalidated numerical tolerance is used to declare correctness."
    )
    torch.save(result.detach().cpu(), out / "output.pt")
    for _ in range(args.warmup):
        module.rocm_absorb_q_bmm(attn, q, is_capture_mode=False)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            module.rocm_absorb_q_bmm(attn, q, is_capture_mode=False)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / args.iterations)
    metadata.update(
        status="completed",
        output_shape=list(result.shape),
        output_dtype=str(result.dtype),
        output_finite=True,
        steady_microseconds=samples,
        median_microseconds=statistics.median(samples),
        warmup=args.warmup,
        repeats=args.repeats,
        iterations_per_repeat=args.iterations,
        timing_scope="Production absorbed-Q dispatch, including activation quantization and allocation",
    )
    write_json(out / "result.json", metadata)
    print(json.dumps(metadata), flush=True)


def run_matrix(args):
    checkout = checked_path(args.checkout)
    out = checked_path(args.out)
    verify_checkout(checkout)
    aiter_root = os.environ.get("AITER_ROOT_DIR")
    if not aiter_root or not checked_path(aiter_root).is_dir():
        raise RuntimeError(
            "Set AITER_ROOT_DIR to a populated run-owned source directory under yihou; "
            "an empty cache directory is not a valid substitute for AITER sources"
        )
    if out.exists():
        raise RuntimeError(f"Use a fresh artifact directory; refusing to overwrite: {out}")
    out.mkdir(parents=True)
    rows = [int(value) for value in args.rows.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    summary = {
        "candidate_head": HEAD,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [],
        "paired_diagnostics": [],
        "limitations": [
            "Only a small synthetic production-kernel matrix, not model/E2E correctness",
            "FP32-reference errors include AITER activation quantization",
            "No numerical tolerance is asserted without a validated quantization reference",
            "These isolated timings do not establish a deployment throughput improvement",
        ],
    }
    for row in rows:
        for seed in seeds:
            for arm, knobs in ARMS.items():
                case = out / f"m{row}-seed{seed}-{arm}"
                case.mkdir()
                env = os.environ.copy()
                env.update(
                    SGLANG_USE_AITER="1",
                    SGLANG_ROCM_ABSORB_BMM_MAX_WAVES_PER_EU=knobs["max_waves"],
                    SGLANG_ROCM_ABSORB_BMM_DISABLE_HAND_CONFIG=knobs["disable_hand_config"],
                    PYTHONPATH=str(checkout / "python") + os.pathsep + env.get("PYTHONPATH", ""),
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                for variable, dirname in (
                    ("TRITON_CACHE_DIR", "triton-cache"),
                    ("XDG_CACHE_HOME", "xdg-cache"),
                    ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
                    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor-cache"),
                    ("TORCH_HOME", "torch-cache"),
                    ("SGLANG_CACHE_DIR", "sglang-cache"),
                    ("AITER_JIT_DIR", "aiter-jit"),
                    ("AMD_COMGR_CACHE_DIR", "comgr-cache"),
                    ("MIOPEN_USER_DB_PATH", "miopen-db"),
                    ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache"),
                    ("HF_HOME", "huggingface-cache"),
                    ("TMPDIR", "tmp"),
                ):
                    directory = case / dirname
                    directory.mkdir()
                    env[variable] = str(directory)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--checkout",
                    str(checkout),
                    "--out",
                    str(case),
                    "--rows",
                    str(row),
                    "--seed",
                    str(seed),
                    "--arm",
                    arm,
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--iterations",
                    str(args.iterations),
                ]
                print(f"Starting {case.name}", flush=True)
                started = time.monotonic()
                timed_out = False
                with (
                    (case / "stdout.log").open("w") as stdout,
                    (case / "stderr.log").open("w") as stderr,
                ):
                    try:
                        process = subprocess.Popen(
                            command,
                            env=env,
                            cwd=checkout,
                            stdout=stdout,
                            stderr=stderr,
                            start_new_session=True,
                        )
                        returncode = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        # Only the new process session created by this case is terminated.
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        returncode = None
                item = {
                    "name": case.name,
                    "rows": row,
                    "seed": seed,
                    "arm": arm,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "wall_seconds": time.monotonic() - started,
                }
                if (case / "result.json").exists():
                    item["result"] = json.loads((case / "result.json").read_text())
                summary["cases"].append(item)
                write_json(out / "summary.json", summary)
                print(
                    f"Finished {case.name}: returncode={returncode}, timeout={timed_out}",
                    flush=True,
                )

    # All comparisons use CPU tensors. Loading our own output tensors uses weights_only.
    import torch

    for row in rows:
        for seed in seeds:
            group = {
                item["arm"]: item
                for item in summary["cases"]
                if item["rows"] == row and item["seed"] == seed
            }
            baseline = group["default"]
            for arm in ("capped", "uncapped"):
                candidate = group[arm]
                if any(
                    item.get("result", {}).get("status") != "completed"
                    for item in (baseline, candidate)
                ):
                    continue
                for field in ("q_sha256", "weight_sha256"):
                    if baseline["result"][field] != candidate["result"][field]:
                        raise RuntimeError(f"Mismatched input {field} for {candidate['name']}")
                actual = torch.load(
                    out / candidate["name"] / "output.pt", map_location="cpu", weights_only=True
                )
                reference = torch.load(
                    out / baseline["name"] / "output.pt", map_location="cpu", weights_only=True
                )
                summary["paired_diagnostics"].append(
                    {
                        "rows": row,
                        "seed": seed,
                        "arm": arm,
                        "reference_arm": "default",
                        "metrics": error_metrics(torch, actual, reference),
                        "median_time_ratio_to_default": candidate["result"]["median_microseconds"]
                        / baseline["result"]["median_microseconds"],
                    }
                )
    summary["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["required_arms_completed"] = all(
        item.get("result", {}).get("status") == "completed"
        for item in summary["cases"]
        if item["arm"] != "uncapped"
    )
    write_json(out / "summary.json", summary)
    print(f"Evidence written to {out / 'summary.json'}", flush=True)
    return 0 if summary["required_arms_completed"] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("matrix", "worker"))
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", default="256,128,64,257")
    parser.add_argument("--seeds", default="104729")
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--arm", choices=tuple(ARMS), default="default")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.mode == "worker":
        args.rows = int(args.rows)
        run_worker(args)
        return 0
    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Prepare the archived fake-decode server for profiling; run on the GPU host.

The original packup is read-only input. No container is started by this script.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1] / "packups/glm52_fake_tp4ep4_10k500_c16_c32.packup_20260910-055810"
PINNED_IMAGE = "sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d"
COMMITS = {"/sglang": "402df1e1e453e1e85ec0f5ac4052d36598cc691a",
           "/aiter": "2c71811b32c8ce2e1266aedaec199df7d90f597d"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--kit", type=Path, default=KIT)
    parser.add_argument("--model", type=Path, required=True, help="Host model directory")
    parser.add_argument("--image", default=PINNED_IMAGE)
    parser.add_argument("--allow-image-mismatch", action="store_true",
                        help="Allow a different image with the SAME source commits; record deviation")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Fresh node-local cache directory")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cache from this same image/build")
    parser.add_argument("--hip-graph-mode", choices=("original", "traceable"), default="traceable",
                        help="traceable disables ROCm packet capture, while retaining CUDA Graph replay")
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=31832)
    args = parser.parse_args()
    for path in (args.model, args.cache_dir):
        if any(c in str(path) for c in '\n\r"$`\\:') or not path.is_absolute():
            parser.error("Model/cache paths must be absolute and shell-safe")
    if not (args.model / "config.json").is_file():
        parser.error("Model config.json is missing")
    if args.cache_dir.exists() and not args.reuse_cache:
        parser.error("Use a fresh cache directory")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image]))[0]
    if image["Id"] != PINNED_IMAGE and not args.allow_image_mismatch:
        parser.error("Image differs from archive; use --allow-image-mismatch to record this explicitly")
    check = "import subprocess,json; print(json.dumps({p:subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip() for p in ['/sglang','/aiter']}))"
    commits = json.loads(subprocess.check_output([
        "docker", "run", "--rm", "-w", "/", "--entrypoint", "python3", image["Id"], "-c", check]))
    if commits != COMMITS:
        parser.error(f"Source commit mismatch: {commits}")
    helper = runpy.run_path(str(args.kit / "scripts/prepare_workspace.py"))
    helper["prepare"](args.run_dir, args.name, args.port)
    run = args.run_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=args.reuse_cache)
    launch = run / "scripts/launch.sh"
    original = launch.read_text()
    text = original.replace(PINNED_IMAGE, image["Id"])
    text = text.replace('-v "$MODEL:$MODEL:ro"', f'-v "{args.model}:$MODEL:ro"')
    text = text.replace('"$W/cache/', f'"{args.cache_dir}/')
    text = text.replace('-e HIP_VISIBLE_DEVICES=0,1,2,3',
                        '-e PYTHONPATH=/sglang/python -e SGLANG_PROFILE_V2=0 '
                        '-e HIP_VISIBLE_DEVICES=0,1,2,3')
    # ROCm 7.2 packet capture can hide whole target graphs from ROCTracer.
    # This runtime boolean needs the literal "false"; "0" did not disable it
    # in the validated image. CUDA Graph remains enabled, but replay overhead
    # changes, so compare operator distributions separately from original perf.
    if args.hip_graph_mode == "traceable":
        text = text.replace('-e PYTHONNOUSERSITE=1',
                            '-e DEBUG_CLR_GRAPH_PACKET_CAPTURE=false -e PYTHONNOUSERSITE=1')
    launch.write_text(text)
    (run / "state/profile-launch.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), text.splitlines(True), fromfile="relocated-original", tofile="profile-launch")))
    model_hashes = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                    for name in ("config.json", "model.safetensors.index.json")}
    (run / "state/profile-environment.json").write_text(json.dumps({
        "archived_image": PINNED_IMAGE, "actual_image": image["Id"],
        "exact_image_reproduction": image["Id"] == PINNED_IMAGE,
        "commits": commits, "host_model": str(args.model), "model_hashes": model_hashes,
        "cache_dir": str(args.cache_dir), "container": args.name, "port": args.port,
        "hip_graph_mode": args.hip_graph_mode, "cache_reused": args.reuse_cache, "run_dir": str(run),
    }, indent=2) + "\n")
    (run / "state/image-inspect.json").write_text(json.dumps(image, indent=2) + "\n")
    print(f"Ready to launch: bash {launch} sim c32 0.85 32", flush=True)


if __name__ == "__main__":
    main()

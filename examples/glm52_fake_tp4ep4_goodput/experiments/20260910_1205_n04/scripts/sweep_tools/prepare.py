#!/usr/bin/env python3
"""Prepare an unprofiled TP4/EP4 concurrency sweep on the Docker host."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1] / "packups/glm52_fake_tp4ep4_10k500_c16_c32.packup_20260910-055810"
IMAGE = "sha256:416d51effc431e27a4ddeed5cdd8ca6012c68eeabb071f987ce99f1ba9854973"
ARCHIVED_IMAGE = "sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d"
COMMITS = {"/sglang": "402df1e1e453e1e85ec0f5ac4052d36598cc691a",
           "/aiter": "2c71811b32c8ce2e1266aedaec199df7d90f597d"}
GRAPH_BATCHES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old!r}")
    return text.replace(old, new, 1)


def client_with_details(source):
    """Preserve native client behavior; retain already-measured request timing."""
    text = replace_once(source,
        '    text_chunks: List[str] = field(default_factory=list)',
        '    raw_chunk_gaps: List[float] = field(default_factory=list)\n'
        '    chunk_token_counts: List[int] = field(default_factory=list)\n'
        '    text_chunks: List[str] = field(default_factory=list)')
    text = replace_once(text,
        '                                    adjust_itl = chunk_gap / num_new_tokens',
        '                                    output.raw_chunk_gaps.append(chunk_gap)\n'
        '                                    output.chunk_token_counts.append(num_new_tokens)\n'
        '                                    adjust_itl = chunk_gap / num_new_tokens')
    text = replace_once(text,
        '        "itls": [output.itl for output in outputs],',
        '        "itls": [output.itl for output in outputs],\n'
        '        "latencies": [output.latency for output in outputs],\n'
        '        "start_times": [output.start_time for output in outputs],\n'
        '        "successes": [output.success for output in outputs],\n'
        '        "raw_chunk_gaps": [output.raw_chunk_gaps for output in outputs],\n'
        '        "chunk_token_counts": [output.chunk_token_counts for output in outputs],')
    ast.parse(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/data/models/GLM-5.2-MXFP4"))
    parser.add_argument("--cache-dir", type=Path, required=True, help="Compatible cache for this immutable image")
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=31864)
    args = parser.parse_args()
    for path in (args.model, args.cache_dir):
        if not path.is_absolute() or any(c in str(path) for c in '\n\r"$`\\:'):
            parser.error("Model/cache paths must be absolute and shell-safe")
    if not (args.model / "config.json").is_file():
        parser.error("Model is missing")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE]))[0]
    code = "import subprocess,json; print(json.dumps({p:subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip() for p in ['/sglang','/aiter']}))"
    commits = json.loads(subprocess.check_output(["docker", "run", "--rm", "-w", "/", "--entrypoint", "python3", IMAGE, "-c", code]))
    if commits != COMMITS:
        parser.error(f"Unexpected source revisions: {commits}")
    helper = runpy.run_path(str(KIT / "scripts/prepare_workspace.py"))
    helper["prepare"](args.run_dir, args.name, args.port)
    run = args.run_dir.resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    launch = run / "scripts/launch.sh"
    original = launch.read_text()
    text = original.replace(ARCHIVED_IMAGE, IMAGE)
    text = replace_once(text, '-v "$MODEL:$MODEL:ro"', f'-v "{args.model}:$MODEL:ro"')
    text = text.replace('"$W/cache/', f'"{args.cache_dir}/')
    text = replace_once(text, '-e HIP_VISIBLE_DEVICES=0,1,2,3', '-e PYTHONPATH=/sglang/python -e HIP_VISIBLE_DEVICES=0,1,2,3')
    text = replace_once(text, '--cuda-graph-max-bs "$CONC"',
        '--cuda-graph-max-bs "$CONC" --cuda-graph-bs-decode ' + ' '.join(map(str, GRAPH_BATCHES)))
    launch.write_text(text)
    (run / "state/launch-adaptation.diff").write_text(''.join(difflib.unified_diff(
        original.splitlines(True), text.splitlines(True), fromfile="relocated-packup", tofile="unprofiled-sweep")))
    source = subprocess.check_output(["docker", "run", "--rm", "-w", "/", "--entrypoint", "cat", IMAGE,
                                      "/sglang/python/sglang/benchmark/serving.py"], text=True)
    (run / "scripts/original/serving.py").write_text(source)
    client = client_with_details(source)
    (run / "scripts/bench_with_details.py").write_text(client)
    (run / "state/client-details.diff").write_text(''.join(difflib.unified_diff(
        source.splitlines(True), client.splitlines(True), fromfile="native-serving.py", tofile="bench_with_details.py")))
    state = {"image": IMAGE, "archived_image": ARCHIVED_IMAGE, "exact_image_reproduction": False,
             "commits": commits, "container": args.name, "port": args.port, "host_model": str(args.model),
             "cache_dir": str(args.cache_dir), "graph_batch_sizes": GRAPH_BATCHES,
             "max_running_requests": 64, "profiling": False, "packet_capture": "runtime default (no override)",
             "client_change": "Only retain request latencies/start times/success and raw SSE gaps/token counts",
             "client_original_sha256": hashlib.sha256(source.encode()).hexdigest(),
             "client_modified_sha256": hashlib.sha256(client.encode()).hexdigest(),
             "model_hashes": {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
                              for name in ("config.json", "model.safetensors.index.json")}}
    (run / "state/environment.json").write_text(json.dumps(state, indent=2) + "\n")
    (run / "state/image-inspect.json").write_text(json.dumps(image, indent=2) + "\n")
    print(f"Prepared unprofiled sweep: bash {launch} sim server 0.85 64")


if __name__ == "__main__":
    main()

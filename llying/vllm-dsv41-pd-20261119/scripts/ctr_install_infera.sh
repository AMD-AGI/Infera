#!/usr/bin/env bash
# Inside the engine container: install amd-infera 0.3.3 (the InferenceX infera-* router pin) from the local wheels
# with --no-deps, so the image's torch/vllm/transformers stay at their pinned versions; overlay the two vLLM-wrapper
# files of AMD-AGI/Infera dev/pd_opt/vllm_glm52_agentx (infera-overlay/SOURCE_COMMIT): args.py (vLLM >= 0.34 moved
# make_arg_parser to entrypoints/common, which 0.3.3 cannot import; INFERA_VLLM_DP_ENGINE_LB) and gem_mooncake.py
# (MooncakeConnector with its KV cache in amdgpu GEM BOs). Fails unless the worker wrapper parses a DSv4.1 command
# line and the router plus its bundled Rust binary import.
set -eo pipefail
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/env.sh
pip_install=(python3 -m pip install --quiet --no-cache-dir --no-deps)
python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages && pip_install+=(--break-system-packages)
"${pip_install[@]}" "$W"/wheels/amd_infera-0.3.3-py3-none-manylinux_2_28_x86_64.whl "$W"/wheels/nats_py-2.16.0-py3-none-any.whl
python3 -c "import xxhash" 2>/dev/null || "${pip_install[@]}" "$W"/wheels/extra/xxhash-*.whl
pkg="$(python3 -c 'import infera, os; print(os.path.dirname(infera.__file__))')"
cp -r "$W"/infera-overlay/infera/. "$pkg/"
echo "infera overlay from $(cat "$W/infera-overlay/SOURCE_COMMIT"): $(cd "$W/infera-overlay" && find infera -name '*.py' | tr '\n' ' ')"
python3 - "$MODEL" <<'PY'
import importlib, sys
missing = []
for m in ("fastapi", "uvicorn", "httpx", "anyio", "xxhash", "zmq", "msgspec", "msgpack", "transformers",
          "prometheus_client", "nats"):
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m}: {e!r}")
if missing:
    raise SystemExit("infera deps missing: " + "; ".join(missing))
import infera.server.args  # noqa: F401
from infera.engine.vllm.args import parse_vllm_args
from infera.server.launch_rust import _find_binary
a = parse_vllm_args(["--discovery-backend", "etcd", "--etcd-endpoint", "127.0.0.1:1", "--request-transport", "http",
                     "--model", sys.argv[1], "--tokenizer-mode", "deepseek_v41", "--block-size", "128", "--monokernel",
                     "--engram-config", '{"cpu_offload":false}', "--tensor-parallel-size", "4"])
print("infera parse ok: model", a.model, "block", a.block_size, "| rust router", _find_binary())
PY

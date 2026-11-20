#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: build-time self-check of the DeepSeek-V4.1 vLLM nightly image (Dockerfile.vllm.dsv41); also runnable on a
#       published image:  docker run --rm --entrypoint bash <image> /opt/infera/verify_vllm_dsv41_image.sh
# why : every piece this image adds fails SILENTLY when it is missing — an unpatched Mooncake connector hangs requests
#       until a 1800 s pull timeout, a missing GEM allocator only shows as an OOM at KV registration, and a Mooncake
#       without dma-buf falls back to a 5-20x slower transport. Fail the build instead.
set -euo pipefail
fail() { echo "VERIFY FAILED: $*" >&2; exit 1; }
# Count, never `| grep -q`: under pipefail grep -q's early exit SIGPIPEs the producer and the
# pipeline reports failure even on a match.
has() { local n; n=$("${@:2}" 2>/dev/null | grep -c -- "$1" || true); [ "${n:-0}" -gt 0 ]; }

site="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
mc="$site/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
grep -q "RequestStatus.FINISHED_STOPPED," "$mc" || fail "mooncake STOPPED patch missing in $mc"
grep -q "VLLM_MOONCAKE_ZMQ_MAX_SOCKETS" "$mc" || fail "mooncake zmq MAX_SOCKETS patch missing in $mc"
grep -q "storage.nbytes() - tensor_blocks \* block_stride < block_stride" "$mc" || fail "mooncake packed ROCm-tail patch missing in $mc"
echo "vllm $(python3 -c 'import vllm; print(vllm.__version__)'): all three Mooncake connector patches present"

so=/usr/local/lib/libhip_gem_alloc.so
for sym in hip_gem_alloc hip_gem_free hsa_amd_portable_export_dmabuf ibv_reg_mr; do
    has " $sym\$" nm -D --defined-only "$so" || fail "$so does not export $sym"
done
echo "GEM allocator: $so"

mso="$(python3 -c 'import mooncake, os, glob; print(sorted(glob.glob(os.path.join(os.path.dirname(mooncake.__file__), "engine*.so")))[0])')"
for s in MC_DISABLE_HIP ibv_reg_dmabuf_mr; do
    has "$s" strings "$mso" || fail "Mooncake engine $mso lacks $s"
done
echo "Mooncake: $(pip show mooncake-transfer-engine-rocm 2>/dev/null | awk '/^Version/ {print $2}') with dma-buf and MC_DISABLE_HIP"

mori="$(python3 -c 'import mori, os; print(os.path.dirname(mori.__file__))')"
[ -x "$mori/umbp_standalone_server" ] || fail "MORI at $mori has no umbp_standalone_server"
echo "UMBP: $mori/umbp_standalone_server"

/usr/local/bin/infera-router --help >/dev/null 2>&1 || fail "infera-router unusable"
python3 -c 'import infera.engine.vllm.gem_mooncake, infera.engine.vllm.args, infera.server.launch_rust' \
    || fail "infera's vLLM wrapper does not import"
# Parsing `vllm serve` args builds a VllmConfig, which must detect a GPU: only possible at run
# time (build_image.sh re-runs this script in a container with the GPUs).
if [ ! -e /dev/kfd ]; then
    echo "infera imports; DSv4.1 command-line parse check skipped (no GPU visible)"
    echo "VERIFY OK"
    exit 0
fi
python3 - <<'PY'
import json

import infera
import infera.engine.vllm.gem_mooncake  # noqa: F401
from infera.engine.vllm.args import parse_vllm_args
from infera.server.launch_rust import _find_binary

mooncake = {
    "kv_connector": "MooncakeConnector",
    "kv_connector_module_path": "infera.engine.vllm.gem_mooncake",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {"num_workers": 10, "canonical_kv": True},
}
umbp = {
    "kv_connector": "UMBPStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"mode": "standalone", "endpoint": "unix:///tmp/umbp.sock"},
}
kvt = {"kv_connector": "MultiConnector", "kv_role": "kv_producer",
       "kv_connector_extra_config": {"connectors": [mooncake, umbp]}}
a = parse_vllm_args([
    "--discovery-backend", "etcd", "--etcd-endpoint", "127.0.0.1:1", "--request-transport", "http",
    "--model", "/models/DeepSeek-V4.1-Flash", "--tokenizer-mode", "deepseek_v41", "--block-size", "128",
    "--tensor-parallel-size", "1", "--data-parallel-size", "2", "--engram-config", '{"cpu_offload":true}',
    "--kv-cache-dtype", "fp8_e4m3", "--kv-transfer-config", json.dumps(kvt),
])
# Both stacks must claim canonical KV, or the DSv4.1 ROCm backend drops aiter sparse MLA and
# refuses --kv-cache-dtype fp8_e4m3.
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

for leg, c in (("decode", {**mooncake, "kv_role": "kv_consumer"}), ("prefill", kvt)):
    cfg = KVTransferConfig(**c)
    if not KVConnectorFactory.get_connector_class(cfg).only_restores_canonical_kv(cfg):
        raise SystemExit(f"VERIFY FAILED: the {leg} connector stack does not claim canonical KV")
print(f"infera {getattr(infera, '__version__', '?')}: parses a DSv4.1 P-leg command line",
      f"(block {getattr(a, 'block_size', '?')}, dp {getattr(a, 'data_parallel_size', '?')}); router {_find_binary()}")
PY
echo "VERIFY OK"

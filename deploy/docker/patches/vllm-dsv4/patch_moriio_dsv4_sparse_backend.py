###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the MoRIIO DSv4 sparse-attention backend-name fix.

WHAT: for DSv4 sparse MLA (use_mla + hf_config.index_topk + cache_dtype
  fp8_ds_mla), skip the generic get_attn_backend() query and set backend_name
  directly to ROCM_FLASHMLA_SPARSE_DSV4; other models keep the original path.
WHY: backend_name is only a P/D handshake tag, but the generic ROCm selector
  returns ROCM_AITER_MLA_SPARSE (no fp8_ds_mla) -> raises -> prefill worker dies.
DEPS: baked by Dockerfile.vllm; idempotent, no-op if anchor absent.
  VERSION: verified vllm 0.23.x ROCm, DeepSeek-V4-Pro MoRIIOConnector 1P1D.
"""

import os
import sys

try:
    import vllm
except Exception:
    print("moriio-dsv4-backend: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
)
if not os.path.exists(f):
    print(f"moriio-dsv4-backend: {f} not found (no MoRIIO connector) — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "# MORIIO-DSV4-SPARSE-BACKEND-PATCH"
if MARKER in src:
    print("moriio-dsv4-backend: already patched")
    sys.exit(0)

# Generic backend query + its .get_name() usage; matched verbatim so drift
# fails loudly (skips) rather than silently mis-patching.
OLD = (
    "        backend = get_attn_backend(\n"
    "            self.model_config.get_head_size(),\n"
    "            self.model_config.dtype,\n"
    "            self.cache_config.cache_dtype,\n"
    "            use_mla=self.use_mla,\n"
    "        )\n"
    "        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}\n"
    "\n"
    "        # TODO: consider the integration of flashinfer or other backends.\n"
    "        self.backend_name = backend.get_name()\n"
)
NEW = (
    "        # MORIIO-DSV4-SPARSE-BACKEND-PATCH: generic get_attn_backend() returns\n"
    "        # ROCM_AITER_MLA_SPARSE (no fp8_ds_mla) and crashes; backend_name is only\n"
    "        # a P/D handshake tag, so set the fp8_ds_mla-capable backend directly.\n"
    '        _hf_cfg = getattr(self.model_config, "hf_config", None)\n'
    "        _is_dsv4_sparse = (\n"
    "            self.use_mla\n"
    '            and hasattr(_hf_cfg, "index_topk")\n'
    '            and self.cache_config.cache_dtype == "fp8_ds_mla"\n'
    "        )\n"
    "        if _is_dsv4_sparse:\n"
    "            backend = None\n"
    '            self.backend_name = "ROCM_FLASHMLA_SPARSE_DSV4"\n'
    "        else:\n"
    "            backend = get_attn_backend(\n"
    "                self.model_config.get_head_size(),\n"
    "                self.model_config.dtype,\n"
    "                self.cache_config.cache_dtype,\n"
    "                use_mla=self.use_mla,\n"
    "            )\n"
    "        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}\n"
    "\n"
    "        # TODO: consider the integration of flashinfer or other backends.\n"
    "        if backend is not None:\n"
    "            self.backend_name = backend.get_name()\n"
)

if src.count(OLD) != 1:
    print(
        f"moriio-dsv4-backend: anchor found {src.count(OLD)}x (expected 1) — "
        "connector layout drifted, skipping"
    )
    sys.exit(0)

src = src.replace(OLD, NEW, 1)
open(f, "w").write(src)
print(f"moriio-dsv4-backend: patched {f}")

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the MoRIIO WRITE-mode DSA-indexer completion fix.

Baked into the engine image by Dockerfile.vllm (the `for f in vllm/*.py`
loop). Idempotent and self-locating. Apply alongside patch_moriio_hetero (which
makes _compute_block_transfer_offsets per-layer; the indexer's distinct shape
relies on it).

GLM-5.1 (GlmMoeDsa = MLA + DSA "lightning indexer") registers TWO KV caches per
transformer layer:
  - attention latent:  model.layers.N.self_attn.attn            (MLA, head 576)
  - DSA indexer cache: model.layers.N.self_attn.indexer.k_cache (MLA, uint8 quant)
so len(kv_caches) == 2 * num_hidden_layers (= 156 for 78 layers).

In WRITE (push) mode the producer schedules a block write from vLLM's per-layer
`save_kv_layer` hook, which vLLM only invokes for the ATTENTION layers — the
indexer caches are updated by the `sparse_attn_indexer` op and never go through
the connector. So:
  1) the indexer KV is never pushed to decode (decode needs it: the read path
     transfers all 156 layers — see patch_moriio_hetero.py), and
  2) `_finalize_if_complete` (engine.py) gates completion on
     `request_info.writes_done >= self.worker.num_layers` where
     `num_layers == len(kv_caches) == 156`, but only the 78 attn layers are ever
     written, so `writes_done` plateaus at 78 < 156 and the completion branch
     NEVER runs: the producer never sends the write-complete notify to decode
     (decode hangs in WAITING_FOR_REMOTE_KVS) and never appends to
     `done_req_ids`, so the deferred send is reaped after 60s -> HTTP 000.
Proven live via MORIIODBG instrumentation (GLM5.1_MORIIO_PLAN.md §10.6):
SCHED=EXEC=312 (78x4, all attn), GETFIN=0, REAP fires at 60s.

Fix: push the KV-cache layers vLLM never drove through `save_kv_layer` (the
indexer caches) in `wait_for_save` — the single hook vLLM calls once after all
per-layer saves of a forward. This both delivers the indexer KV to decode AND
lets `writes_done` reach `num_layers`, so `_finalize_if_complete` fires.
`schedule_write_blocks` resolves everything from `layer_name` (the tensor arg is
unused), and `_compute_block_transfer_offsets` is per-layer (patch_moriio_hetero),
so the indexer's distinct shape is handled correctly. Homogeneous models have no
undriven layers -> this is a no-op for them.

Run inside a container with vLLM installed:
    docker exec <ctr> python3 patch_moriio_dsa_write.py
Exits 0 (no-op) if the MoRIIO connector isn't present.
"""

import os
import py_compile
import sys

try:
    import vllm
except Exception:
    print("moriio-dsa-write: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
)
if not os.path.exists(f):
    print(f"moriio-dsa-write: {f} not found (no moriio connector) — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "_dsa_save_driven_layers"
if MARKER in src:
    print("moriio-dsa-write: already patched")
    sys.exit(0)

# --- Edit 1: record the layers vLLM drives via worker save_kv_layer (attn) -----
old1 = (
    "        if not self.is_producer:\n"
    "            return\n"
    "        if self.mode == MoRIIOMode.READ:\n"
    "            return\n"
    "        remote_engine_id = None\n"
)
new1 = (
    "        if not self.is_producer:\n"
    "            return\n"
    "        if self.mode == MoRIIOMode.READ:\n"
    "            return\n"
    "        # DSA fix: remember which layers vLLM actually drives through\n"
    "        # save_kv_layer (the attention layers); wait_for_save pushes the rest.\n"
    '        if not hasattr(self, "_dsa_save_driven_layers"):\n'
    "            self._dsa_save_driven_layers = set()\n"
    "        self._dsa_save_driven_layers.add(layer_name)\n"
    "        remote_engine_id = None\n"
)

# --- Edit 2: new worker method to push the undriven (indexer) layers -----------
old2 = (
    "    def _write_blocks_for_req(self, req_id: ReqId, meta: ReqMeta, layer_name, kv_layer):\n"
    "        self.schedule_write_blocks(\n"
    "            request_id=req_id,\n"
    "            transfer_id=meta.transfer_id,\n"
    "            dst_engine_id=meta.remote_engine_id,\n"
    "            local_block_ids=meta.local_block_ids,\n"
    "            remote_block_ids=meta.remote_block_ids,\n"
    "            layer_name=layer_name,\n"
    "            kv_layer=kv_layer,\n"
    "            remote_notify_port=meta.remote_notify_port,\n"
    "            remote_ip=meta.remote_host,\n"
    "        )\n"
)
new2 = old2 + (
    "\n"
    "    def flush_aux_layer_writes(self, metadata) -> None:\n"
    "        # Push KV-cache layers vLLM never drove through save_kv_layer (GLM-5.1\n"
    "        # DSA indexer caches). Without this the indexer KV never reaches decode\n"
    "        # and writes_done can never reach num_layers (== len(kv_caches)), so\n"
    "        # _finalize_if_complete never fires -> write-mode deadlock (HTTP 000).\n"
    "        # Called once per forward (wait_for_save), after all attn save_kv_layer.\n"
    "        if not self.is_producer or self.mode != MoRIIOMode.WRITE:\n"
    "            return\n"
    '        driven = getattr(self, "_dsa_save_driven_layers", None)\n'
    "        if not driven:\n"
    "            return\n"
    "        aux = [\n"
    "            ln\n"
    "            for ln in self.layer_name_to_local_kv_cache_metadata\n"
    "            if ln not in driven\n"
    "        ]\n"
    "        if not aux:\n"
    "            return\n"
    "        for req_id, meta in metadata.reqs_to_save.items():\n"
    "            if meta.remote_engine_id is None:\n"
    "                continue\n"
    "            if (\n"
    "                self.get_engine_name_with_dp(meta.remote_engine_id, 0)\n"
    "                not in self._remote_agents\n"
    "            ):\n"
    "                continue\n"
    "            for layer_name in aux:\n"
    "                self._write_blocks_for_req(\n"
    "                    req_id, meta, layer_name, self.kv_caches[layer_name]\n"
    "                )\n"
)

# --- Edit 3: drive it from the connector-level wait_for_save hook --------------
old3 = "    def wait_for_save(self):\n        pass\n"
new3 = (
    "    def wait_for_save(self):\n"
    "        # DSA fix: push KV-cache layers vLLM never drove through save_kv_layer\n"
    "        # (GLM-5.1 lightning-indexer cache) so decode gets complete KV and\n"
    "        # writes_done can reach num_layers to trigger completion.\n"
    "        if self.connector_worker is not None and isinstance(\n"
    "            self._connector_metadata, MoRIIOConnectorMetadata\n"
    "        ):\n"
    "            self.connector_worker.flush_aux_layer_writes(self._connector_metadata)\n"
)

for tag, old in (("edit1", old1), ("edit2", old2), ("edit3", old3)):
    n = src.count(old)
    if n != 1:
        print(
            f"moriio-dsa-write: {tag} expected 1 match, found {n} — moriio layout changed? aborting"
        )
        sys.exit(1)

src = src.replace(old1, new1).replace(old2, new2).replace(old3, new3)

open(f, "w").write(src)
try:
    py_compile.compile(f, doraise=True)
except py_compile.PyCompileError as e:
    print(f"moriio-dsa-write: py_compile FAILED after patch: {e}")
    sys.exit(1)
print("moriio-dsa-write: patched", f)

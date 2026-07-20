###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Full-loop L3 integration test for a plain (non-DSA) MLA model.

Closes the gap between the two existing gates, which each cover only half:
  * test_vllm_kvd_reload_accounting.py drives the PLANNER
    (get_num_new_matched_tokens + build_connector_meta) but asserts only the
    load/save PLAN — never the reloaded bytes.
  * test_vllm_save_load_roundtrip.py byte-verifies the transport but HAND-BUILDS
    the metadata, bypassing the planner.

This drives the WHOLE loop with the real planner deciding what to move:

  producer:  register real MLA KV  -> get_num_new_matched_tokens (miss, stash
             hashes) -> build_connector_meta (SAVE plan) -> wait_for_save -> L3
  consumer:  register fresh zeroed KV -> get_num_new_matched_tokens (HIT probe
             against L3) -> build_connector_meta (LOAD plan) -> start_load_kv
  assert:    every reloaded page is byte-identical to the producer's source page

Deterministic, CPU-only (no GPU, no LLM), so it is NOT confounded by the
dropped-tail recompute / MoE nondeterminism that makes an E2E output diff
unusable for KV correctness. A separate consumer connector instance sharing the
same daemon + hipfile_roots also exercises cross-instance content-key reuse
(the restart-survival property) without spawning a second process.
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

torch_skip = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _vllm_config(model="mla-l3-fullloop"):
    return SimpleNamespace(
        model_config=SimpleNamespace(model=model, served_model_name=model),
        parallel_config=SimpleNamespace(
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_rank=0,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(),
    )


def _kv_cfg_groups(group_layer_names):
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=list(n)) for n in group_layer_names],
    )


@pytest.fixture
async def kvd_daemon(tmp_path: Path):
    from infera.kvd.server import KvdServer

    socket = tmp_path / f"kvd-fl-{uuid.uuid4().hex[:8]}.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 24)
    await server.start()
    task = asyncio.create_task(server.serve_forever(), name="kvd-fl-test")
    await asyncio.sleep(0)
    yield server, str(socket)
    server.shutdown()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _make_mla_layers(num_layers, num_blocks, block_size, hidden, dtype, device, salt):
    """MLA paged layout [num_blocks, block_size, hidden], deterministic bytes."""
    import torch

    layers = {}
    for li in range(num_layers):
        elems = num_blocks * block_size * hidden
        n = elems * torch.empty(0, dtype=dtype).element_size()
        flat = ((torch.arange(n, dtype=torch.int64) * 7 + li * 37 + salt) & 0xFF).to(torch.uint8)
        layers[f"layer.{li}"] = flat.view(dtype).reshape(num_blocks, block_size, hidden).to(device)
    return layers


def _feed_hashes(connector, req_id, hashes, block_size):
    """Make the planner read our synthetic (req_id, block_hashes) — the same
    stand-in monkeypatch used by test_non_paged_skip.test_hybrid_gates_external_load."""
    connector._req_id_of = lambda r: req_id
    connector._extract_block_hashes_after = lambda r, n: list(hashes)
    connector._block_size_now = lambda r: block_size


@torch_skip
@pytest.mark.parametrize("save_mode", ["staging", "gpu_direct"])
async def test_mla_l3_full_loop_planner_to_bytes(save_mode, kvd_daemon, tmp_path, monkeypatch):
    # Plain (non-DSA) MLA, bf16 — the offloadable case. (A native fp8_ds_mla
    # latent is packed and the guard skips it at register time unless
    # INFERA_KVD_ALLOW_PACKED_KV=1; that path is covered separately by the
    # layout round-trip tests, so it is out of scope for "does L3 work".)
    #
    # Runs the SAME full loop over BOTH save transports:
    #   staging    = phase-1 gather + D2H to pinned host -> POSIX write, load via
    #                _load_chunk_packed (mmap + H2D);
    #   gpu_direct = phase-1 gather to device staging -> hipFileWrite, load via
    #                _load_chunk_packed_hipfile_direct (SSD->GPU DMA). Skips (never
    #                false-greens on the POSIX fallback) unless the host has a
    #                hipfile binding + kernel P2PDMA.
    import torch

    from infera.engine.vllm.kvd_connector import InferaKvdConnector

    if not torch.cuda.is_available():
        pytest.skip("kvd save staging pins host memory — needs a GPU present")

    gpu_direct = save_mode == "gpu_direct"
    _, socket = kvd_daemon
    dtype = torch.bfloat16
    # gpu_direct DMAs straight from GPU memory: hipFile rejects a host pointer
    # (5013 "Memory type ... incompatible"), and the save staging is allocated
    # on the KV cache's device — so gpu_direct needs GPU-resident KV, exactly as
    # in production (the vLLM KV cache always lives on the GPU). staging keeps
    # CPU KV (its D2H->POSIX path handles either device).
    device = torch.device("cuda" if gpu_direct else "cpu")

    NL, NB, BS, HID = 2, 1024, 16, 64
    chunk_tokens = BS * 4  # N = 4 pages / chunk
    n_pages = 8  # 2 full chunks
    monkeypatch.setenv("INFERA_KVD_CHUNK_TOKENS", str(chunk_tokens))
    if gpu_direct:
        # hipfile_direct load only dispatches in async/parallel load mode.
        monkeypatch.setenv("INFERA_KVD_ASYNC_LOAD", "1")
    seen = {"gd_save": 0, "gd_fallback": 0, "hipfile_load": 0}

    root = tmp_path / f"l3-{uuid.uuid4().hex[:6]}"
    root.mkdir()
    hf_roots = {"long": str(root)}
    layer_names = [f"layer.{i}" for i in range(NL)]
    # one shared content identity for the prompt; reused by producer + consumer
    hashes = [(i + 1).to_bytes(8, "little") for i in range(n_pages)]

    # ---------- PRODUCER: register real KV, plan a SAVE, write to L3 ----------
    prod = await asyncio.to_thread(
        InferaKvdConnector,
        _vllm_config(),
        1,
        _kv_cfg_groups([layer_names]),
        socket_path=socket,
        hipfile_roots=hf_roots,
        gpu_direct=gpu_direct,
    )
    if gpu_direct and not getattr(prod, "_gpu_direct", False):
        await asyncio.to_thread(prod.close)
        pytest.skip(
            "gpu_direct inactive (no hipfile binding / kernel P2PDMA) — "
            "would silently exercise the POSIX fallback"
        )
    if gpu_direct:
        assert prod._save_gpu_direct, "gpu_direct save path must be active"
        _orig_save = prod._flush_chunk_save_gpu_direct

        def _spy_save(*a, **k):
            ok = _orig_save(*a, **k)
            seen["gd_save"] += 1
            seen["gd_fallback"] += 0 if ok else 1
            return ok

        monkeypatch.setattr(prod, "_flush_chunk_save_gpu_direct", _spy_save)
    else:
        assert not prod._save_gpu_direct, "staging path expected (gpu_direct off)"
    src = _make_mla_layers(NL, NB, BS, HID, dtype, device, salt=11)
    alloc_p = [200 + i * 3 for i in range(n_pages)]  # producer physical blocks
    try:
        prod.register_kv_caches(src)
        assert prod._group_kv_spec, "MLA spec must populate from real tensors"
        assert int(prod._group_kv_spec[0]["num_kv_channels"]) == 1, "MLA => 1 channel"

        _feed_hashes(prod, "rp", hashes, BS)
        # fresh prefill: nothing in L3 yet -> 0 external, but hashes get stashed
        n_ext, _ = prod.get_num_new_matched_tokens(object(), 0)
        assert (n_ext or 0) == 0, f"cold prefix must miss L3, got {n_ext}"

        new_req = SimpleNamespace(req_id="rp", block_ids=(alloc_p,), num_computed_tokens=0)
        save_meta = prod.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[new_req]))
        assert save_meta.packed_chunks_to_save, "planner must emit SAVE chunks"
        assert not save_meta.packed_chunks_to_load, "cold prefill: nothing to load"
        prod._connector_metadata = save_meta
        await asyncio.to_thread(prod.wait_for_save)
        assert await asyncio.to_thread(lambda: prod.drain_pending_saves(timeout=10.0)) >= 1
        assert list(root.rglob("*.kvcache")), "no chunk file landed on L3"
        if gpu_direct:
            assert seen["gd_save"] >= 1, "gpu_direct save path was not taken"
            assert seen["gd_fallback"] == 0, "gpu_direct save fell back to POSIX"
    finally:
        await asyncio.to_thread(prod.close)

    # ---------- CONSUMER: HIT probe, plan a LOAD, scatter, byte-verify -------
    cons = await asyncio.to_thread(
        InferaKvdConnector,
        _vllm_config(),
        1,
        _kv_cfg_groups([layer_names]),
        socket_path=socket,
        hipfile_roots=hf_roots,
        gpu_direct=gpu_direct,
    )
    if gpu_direct and not getattr(cons, "_gpu_direct", False):
        await asyncio.to_thread(cons.close)
        pytest.skip("gpu_direct inactive on consumer")
    if gpu_direct:
        _orig_load = cons._load_chunk_packed_hipfile_direct

        def _spy_load(entry):
            seen["hipfile_load"] += 1
            return _orig_load(entry)

        monkeypatch.setattr(cons, "_load_chunk_packed_hipfile_direct", _spy_load)
    tgt = _make_mla_layers(NL, NB, BS, HID, dtype, device, salt=0)
    for t in tgt.values():
        t.zero_()
    alloc_c = [700 + i * 5 for i in range(n_pages)]  # DIFFERENT physical blocks
    try:
        cons.register_kv_caches(tgt)
        _feed_hashes(cons, "rc", hashes, BS)

        # THE HIT PROBE: same content is now on L3 -> must report a real match.
        n_ext, _ = cons.get_num_new_matched_tokens(object(), 0)
        assert (n_ext or 0) == n_pages * BS, (
            f"L3 hit probe should match all {n_pages} pages ({n_pages * BS} tokens); got {n_ext}"
        )

        # vLLM sets num_computed_tokens = local(0) + external(n_ext).
        reload_req = SimpleNamespace(req_id="rc", block_ids=(alloc_c,), num_computed_tokens=n_ext)
        load_meta = cons.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[reload_req]))
        assert load_meta.packed_chunks_to_load, "planner must emit LOAD chunks"
        cons._connector_metadata = load_meta
        await asyncio.to_thread(cons.start_load_kv, None)
        await asyncio.sleep(0.1)  # let any async loader drain
        if gpu_direct:
            assert seen["hipfile_load"] >= 1, "hipfile_direct load path not taken"

        # BYTE-VERIFY: each reloaded consumer page == the producer source page.
        for lname in layer_names:
            for pg in range(n_pages):
                s = src[lname][alloc_p[pg]].contiguous().view(torch.uint8)
                d = tgt[lname][alloc_c[pg]].contiguous().view(torch.uint8)
                assert torch.equal(s, d), (
                    f"bf16 MLA: L3 reload byte mismatch layer {lname} page {pg} "
                    f"(producer blk {alloc_p[pg]} -> consumer blk {alloc_c[pg]})"
                )
        # untouched consumer blocks stay zero (scatter hit only the load targets)
        assert torch.all(tgt["layer.0"][0] == 0)
    finally:
        await asyncio.to_thread(cons.close)

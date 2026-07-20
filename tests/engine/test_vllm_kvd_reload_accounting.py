###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""v2 chunked-fusion RELOAD-accounting regression tests.

These guard the scheduler-side `build_connector_meta` LOAD/SAVE split for
the *external-L3 reload* path — the exact accounting that
`fe056f8 (fix kvd L3 KV-cache correctness)` bug #1 was about:

    vLLM reports NewRequestData.num_computed_tokens = local-L1 + external
    (the tokens it expects the connector to LOAD from L3). If the connector
    folds the external part into `skip_blocks`, it emits ZERO load entries,
    the allocated-but-unfilled GPU blocks are attended over, and the model
    produces garbage on every L3 reuse.

The legacy `test_vllm_kvd_connector.py` is module-`skip`ped (v1 per-block API)
and `test_vllm_save_load_roundtrip.py` only exercises the byte-level
gather/scatter round-trip — NEITHER drives `build_connector_meta` with a
realistic reload `NewRequestData` carrying a non-zero `num_computed_tokens`.
So this class of bug had no active regression gate. This file is that gate.

Pure scheduler-side metadata assertions: no GPU, no forward pass — we assert
that the LOAD entries cover exactly the externally-cached prefix region and
target the correct allocated block ids.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from infera.engine.vllm.kvd_connector import InferaKvdConnector
from infera.kvd.server import KvdServer


def _vllm_config(model: str = "reload-acct") -> SimpleNamespace:
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


@pytest.fixture
async def kvd_daemon(tmp_path: Path):
    socket = tmp_path / f"kvd-reload-{uuid.uuid4().hex[:8]}.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever(), name="kvd-reload-test")
    await asyncio.sleep(0)
    yield server, str(socket)
    server.shutdown()
    try:
        await asyncio.wait_for(serve_task, timeout=2.0)
    except asyncio.TimeoutError:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


def _install_spec(connector, *, block_size: int = 16, num_layers: int = 2):
    """Populate `_group_kv_spec` so `build_connector_meta` runs the v2 emit
    without needing real GPU tensors (`register_kv_caches` needs `.device`).

    Chunk grain N = chunk_tokens // block_size is derived from this spec, so
    we set it explicitly and pin INFERA_KVD_CHUNK_TOKENS to match.
    """
    connector._group_kv_spec = {
        0: {
            "layer_names": [f"layer.{i}" for i in range(num_layers)],
            "num_blocks": 4096,
            "block_size": block_size,
            "page_bytes": block_size * 64 * 2 * num_layers,
            "hidden_dim": 64,
            "num_kv_channels": 1,
            "dtype_str": "bf16",
        }
    }
    connector._chunk_tokens_auto = False


def _load_page_ids(meta):
    """Flatten the LOAD entries' per_page_block_ids to the ordered list of
    physical block ids the connector will scatter reloaded KV into."""
    ids: list[int] = []
    for per_page_block_ids, _key, _gid, _layers in meta.packed_chunks_to_load:
        for page in per_page_block_ids:
            ids.append(page[0])
    return ids


@pytest.mark.asyncio
async def test_full_reload_loads_entire_prefix_into_allocated_blocks(kvd_daemon, monkeypatch):
    """A prefix fully evicted from L1 and reloaded from L3:
    num_computed_tokens == num_external (local L1 = 0). The connector MUST
    emit LOAD entries for the whole external region and NOTHING must be
    skipped — else the freshly-allocated blocks are never filled.

    This is the direct regression for bug #1 (folding external into
    skip_blocks -> zero loads -> garbage).
    """
    _, socket = kvd_daemon
    block_size = 16
    chunk_tokens = block_size * 4  # N = 4 pages/chunk
    monkeypatch.setenv("INFERA_KVD_CHUNK_TOKENS", str(chunk_tokens))
    connector = await asyncio.to_thread(
        InferaKvdConnector, _vllm_config(), 0, None, socket_path=socket
    )
    try:
        _install_spec(connector, block_size=block_size)

        # 12 prompt pages = 3 full chunks (N=4). All 12 came from L3.
        n_pages = 12
        n_ext_pages = 12
        hashes = [(i + 1).to_bytes(8, "little") for i in range(n_pages)]
        connector._pending_block_hashes["r1"] = hashes
        connector._pending_external_blocks["r1"] = n_ext_pages

        # vLLM allocates fresh blocks for the whole prefix; block ids are
        # arbitrary physical ids (non-contiguous to catch mis-indexing).
        alloc = [100 + i for i in range(n_pages)]
        new_req = SimpleNamespace(
            req_id="r1",
            block_ids=(alloc,),
            # vLLM sets this = local(0) + external(12*16) = 192.
            num_computed_tokens=n_ext_pages * block_size,
        )
        sched = SimpleNamespace(scheduled_new_reqs=[new_req])

        meta = connector.build_connector_meta(sched)

        loaded = _load_page_ids(meta)
        # Every externally-cached page must be loaded — none skipped.
        assert loaded == alloc, (
            f"full reload must load all {n_pages} pages into their allocated "
            f"blocks; got {loaded} vs expected {alloc}. "
            f"(bug #1 regression: external folded into skip_blocks -> "
            f"missing LOAD entries -> attention over unfilled blocks)"
        )
        # Fully-cached prefix → nothing to save this step.
        assert meta.packed_chunks_to_save == []
    finally:
        await asyncio.to_thread(connector.close)


@pytest.mark.asyncio
async def test_partial_reload_splits_l1_load_and_save_at_boundaries(kvd_daemon, monkeypatch):
    """Mixed case: some prefix is in local L1, some in L3, some brand new.

    Layout (N=4 pages/chunk, 16 tok/page):
      chunk 0 (pages 0-3)  : in L1        -> SKIP (no load, no save)
      chunk 1 (pages 4-7)  : in L3        -> LOAD
      chunk 2 (pages 8-11) : brand new    -> SAVE

    num_computed_tokens = (L1 4 pages + L3 4 pages) * 16 = 128.
    external = 4 pages. skip_blocks must be L1-only = 4 (NOT 8).
    """
    _, socket = kvd_daemon
    block_size = 16
    chunk_tokens = block_size * 4
    monkeypatch.setenv("INFERA_KVD_CHUNK_TOKENS", str(chunk_tokens))
    connector = await asyncio.to_thread(
        InferaKvdConnector, _vllm_config(), 0, None, socket_path=socket
    )
    try:
        _install_spec(connector, block_size=block_size)

        n_pages = 12
        l1_pages = 4
        l3_pages = 4
        hashes = [(i + 1).to_bytes(8, "little") for i in range(n_pages)]
        connector._pending_block_hashes["r1"] = hashes
        connector._pending_external_blocks["r1"] = l3_pages

        alloc = [200 + i for i in range(n_pages)]
        new_req = SimpleNamespace(
            req_id="r1",
            block_ids=(alloc,),
            num_computed_tokens=(l1_pages + l3_pages) * block_size,
        )
        sched = SimpleNamespace(scheduled_new_reqs=[new_req])

        meta = connector.build_connector_meta(sched)

        # LOAD must target exactly the L3 chunk's pages (4..7), NOT the L1
        # prefix (0..3, skipped) and NOT the new chunk (8..11, saved).
        loaded = _load_page_ids(meta)
        assert loaded == alloc[l1_pages : l1_pages + l3_pages], (
            f"partial reload must load ONLY the L3 region pages "
            f"{alloc[l1_pages : l1_pages + l3_pages]}, got {loaded}"
        )

        # SAVE must target exactly the brand-new chunk's pages (8..11).
        saved_pages: list[int] = []
        for per_page_block_ids, _key, _ret, _gid, _layers in meta.packed_chunks_to_save:
            for page in per_page_block_ids:
                saved_pages.append(page[0])
        assert saved_pages == alloc[l1_pages + l3_pages :], (
            f"save must cover ONLY the new chunk pages "
            f"{alloc[l1_pages + l3_pages :]}, got {saved_pages}"
        )
    finally:
        await asyncio.to_thread(connector.close)


@pytest.mark.asyncio
async def test_load_and_save_regions_are_disjoint_and_cover_no_l1(kvd_daemon, monkeypatch):
    """Invariant sweep across external-block counts: for any split, the LOAD
    and SAVE regions must be disjoint, and neither may touch the L1-covered
    prefix. Catches off-by-N in the skip_chunks / n_load_chunks arithmetic.
    """
    _, socket = kvd_daemon
    block_size = 16
    N = 4
    chunk_tokens = block_size * N
    monkeypatch.setenv("INFERA_KVD_CHUNK_TOKENS", str(chunk_tokens))
    connector = await asyncio.to_thread(
        InferaKvdConnector, _vllm_config(), 0, None, socket_path=socket
    )
    try:
        _install_spec(connector, block_size=block_size)
        n_pages = 16  # 4 chunks
        alloc = [300 + i for i in range(n_pages)]

        # Sweep: 0 L1 chunks + k L3 chunks, k in 0..4.
        for l3_chunks in range(0, 5):
            connector._saved_content_keys.clear()
            connector._req_save_state.clear()
            l3_pages = l3_chunks * N
            hashes = [(l3_chunks * 100 + i + 1).to_bytes(8, "little") for i in range(n_pages)]
            rid = f"r{l3_chunks}"
            connector._pending_block_hashes[rid] = hashes
            connector._pending_external_blocks[rid] = l3_pages
            new_req = SimpleNamespace(
                req_id=rid,
                block_ids=(alloc,),
                num_computed_tokens=l3_pages * block_size,  # local L1 = 0 here
            )
            sched = SimpleNamespace(scheduled_new_reqs=[new_req])
            meta = connector.build_connector_meta(sched)

            loaded = set(_load_page_ids(meta))
            saved = set()
            for per_page_block_ids, *_ in meta.packed_chunks_to_save:
                for page in per_page_block_ids:
                    saved.add(page[0])

            assert not (loaded & saved), (
                f"l3_chunks={l3_chunks}: LOAD and SAVE overlap {loaded & saved}"
            )
            # LOAD must be exactly the first l3_pages allocated blocks.
            assert loaded == set(alloc[:l3_pages]), (
                f"l3_chunks={l3_chunks}: loaded={sorted(loaded)} expected={alloc[:l3_pages]}"
            )
    finally:
        await asyncio.to_thread(connector.close)

#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared driver: benchmark a directory THROUGH the real kvd engine connector.

Both ``infera-kvd-probe`` and ``infera-kvd-l3-bench`` call
:func:`run_connector_bench`. It exercises the *production* save/load path —
``InferaKvdConnector.wait_for_save`` / ``.start_load_kv`` — rather than a
hand-written preadv/pwrite/hipFile loop, so the GB/s it reports is what the
engine's chunked-fusion pipeline (per-chunk staging rings, per-layer H2D
overlap, Triton scatter/gather, CUDA-event pipelining, GPU-direct vs POSIX
transport, worker fan-out) actually achieves on this mount.

Protocol (the connector round-trip that the retired ``bench_packed_v2`` drove,
now folded here — see ``infera-kvd-l3-bench``):

  1. Start a real kvd daemon over a tmp UDS (``python -m infera.kvd``),
     shared arena disabled (memfd may be absent in containers).
  2. Allocate real GPU KV-cache tensors, one per layer, in the chosen layout.
  3. Build a real ``InferaKvdConnector`` (hipfile_roots -> ``target_dir``) and
     ``register_kv_caches`` them.
  4. SAVE: bind ``packed_chunks_to_save`` metadata, time ``wait_for_save()``.
  5. LOAD (cold): zero the KV tensors, bind ``packed_chunks_to_load``, time
     ``start_load_kv(None)`` — the zeroed cache forces a genuine reload.
  6. Transport is chosen by env set BEFORE the connector is built; the
     connector reports the RESOLVED path via ``_gpu_direct`` /
     ``_save_gpu_direct`` / ``_no_p2pdma_force_serial`` / ``_layerwise_mode``.

One run reports THREE things (folded in from the retired ``bench_packed_v2``):
THROUGHPUT (batched fan-out GB/s), LATENCY (a separate per-chunk save/load loop
→ p50/p95 ms), and CORRECTNESS (seed a per-layer pattern, save, zero, cold-load,
verify the loaded blocks match — plus the logical block_size sanity check).

``torch`` is imported lazily INSIDE the function so this module imports fine
on a bare host without torch (the package's console entry points must import).
"""

from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

# Supported KV-cache tensor layouts (folded in from the retired bench_packed_v2):
#   regular   : [2, num_blocks, block_size, hidden]  (K/V split, channels=2)
#   mla       : [num_blocks, block_size, hidden]      (combined latent, channels=1)
#   mla-aiter : [num_blocks*block_size, 1, hidden]    (ROCM_AITER_MLA fold)
_LAYOUTS = ("regular", "mla", "mla-aiter")


def _num_kv_channels(layout: str) -> int:
    return 2 if layout == "regular" else 1


def _make_layer(torch, layout, num_blocks, block_size, hidden_dim, dtype, device):
    """One v2-shaped KV cache tensor in the requested attention layout."""
    if layout == "regular":
        shape = (2, num_blocks, block_size, hidden_dim)
    elif layout == "mla":
        shape = (num_blocks, block_size, hidden_dim)
    elif layout == "mla-aiter":
        shape = (num_blocks * block_size, 1, hidden_dim)
    else:
        raise ValueError(f"unknown layout {layout!r}; choose from {_LAYOUTS}")
    return torch.empty(shape, dtype=dtype, device=device)


def _layer_fill_value(layer_idx: int) -> float:
    """Deterministic per-layer pattern value (mirrors bench_packed_v2
    ``_seed_layer``). Distinct per layer so a cross-layer mixup is caught;
    a plain multiple of 1/32 so it round-trips bit-exact through bf16."""
    return ((layer_idx * 31) & 0xFF) / 32.0


def _seed_layer(t, layer_idx: int) -> None:
    """Fill one layer with its deterministic pattern (layout-agnostic)."""
    t.fill_(_layer_fill_value(layer_idx))


def _layer_block_range(t, layout: str, lo: int, hi: int, block_size: int):
    """Slice the contiguous block range ``[lo, hi)`` out of a layer tensor,
    honoring the layout's block axis. For mla-aiter the logical block spans
    ``block_size`` rows of dim 0 (middle dim is 1)."""
    if layout == "regular":
        return t[:, lo:hi]
    if layout == "mla":
        return t[lo:hi]
    if layout == "mla-aiter":
        return t[lo * block_size : hi * block_size]
    raise ValueError(f"unknown layout {layout!r}; choose from {_LAYOUTS}")


def _build_connector(socket_path: str, hipfile_root: str, chunk_tokens: int):
    """Build a connector with hipfile_root pointing at the bench dir.

    Transport env (``INFERA_KVD_GPU_DIRECT`` / ``INFERA_KVD_LAYERWISE_LOAD`` /
    ``INFERA_KVD_*_WORKERS``) is read at ``__init__``; the caller sets it
    before this runs so the bench measures a KNOWN transport."""
    from infera.engine.vllm.kvd_connector import InferaKvdConnector

    os.environ["INFERA_KVD_CHUNK_TOKENS"] = str(chunk_tokens)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model="kvd-bench", served_model_name=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector="InferaKvdConnector",
            kv_role="kv_both",
            kv_connector_extra_config={},
        ),
    )
    return InferaKvdConnector(
        vllm_config,
        1,  # WORKER role
        SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=[])]),
        socket_path=socket_path,
        hipfile_roots={"long": hipfile_root},
    )


def _bind_save(connector, per_page, kvd_key, retention, layer_names):
    from infera.engine.vllm.kvd_connector import InferaKvdConnectorMetadata

    connector._connector_metadata = InferaKvdConnectorMetadata(
        packed_chunks_to_save=[
            (tuple(per_page), kvd_key, retention, 0, list(layer_names)),
        ],
    )


def _bind_load(connector, per_page, kvd_key, layer_names):
    from infera.engine.vllm.kvd_connector import InferaKvdConnectorMetadata

    connector._connector_metadata = InferaKvdConnectorMetadata(
        packed_chunks_to_load=[
            (tuple(per_page), kvd_key, 0, list(layer_names)),
        ],
    )


def _apply_transport_env(transport: str, workers: int) -> None:
    """Set the connector's transport knobs from the bench args BEFORE build.

    - transport: 'auto' -> unset INFERA_KVD_GPU_DIRECT (env/auto-detect),
      'posix' -> '0', 'gpu-direct' -> '1'.
    - workers > 0 -> pin BOTH save & load worker counts (benching knob) and
      force layerwise=parallel so the fan-out actually runs.
    - workers <= 0 -> leave worker envs alone; default layerwise=parallel.
    """
    # INFERA_KVD_AIS is the current knob (INFERA_KVD_GPU_DIRECT is deprecated).
    if transport == "gpu-direct":
        os.environ["INFERA_KVD_AIS"] = "1"
    elif transport == "posix":
        os.environ["INFERA_KVD_AIS"] = "0"
    else:  # auto
        os.environ.pop("INFERA_KVD_AIS", None)
    os.environ.pop("INFERA_KVD_GPU_DIRECT", None)  # drop any stale deprecated override

    if workers and workers > 0:
        os.environ["INFERA_KVD_SAVE_WORKERS"] = str(workers)
        os.environ["INFERA_KVD_LOAD_WORKERS"] = str(workers)
        os.environ["INFERA_KVD_LAYERWISE_LOAD"] = "parallel"
    else:
        os.environ.setdefault("INFERA_KVD_LAYERWISE_LOAD", "parallel")


def _start_daemon(socket_path: str, hipfile_root: str, max_bytes: int):
    """Launch a real kvd daemon and wait for its UDS to appear."""
    kvd_log = Path(hipfile_root) / "kvd.log"
    daemon = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "infera.kvd",
            "--socket",
            socket_path,
            "--max-bytes",
            str(max_bytes),
            # memfd_create is absent in some container/python builds; the
            # v2 file-tier path doesn't need the shared arena.
            "--shared-arena-bytes",
            "0",
        ],
        stdout=open(kvd_log, "wb"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    deadline = time.time() + 30
    while not os.path.exists(socket_path):
        if time.time() > deadline:
            daemon.kill()
            raise RuntimeError(f"kvd daemon did not appear at {socket_path} within 30s")
        time.sleep(0.1)
    return daemon


def _teardown(daemon, hipfile_root: str, socket_path: str) -> None:
    if daemon is not None:
        try:
            daemon.terminate()
            daemon.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                daemon.kill()
            except ProcessLookupError:
                pass
    try:
        shutil.rmtree(hipfile_root, ignore_errors=True)
    except OSError:
        pass
    try:
        os.unlink(socket_path)
    except OSError:
        pass


def run_connector_bench(
    target_dir: str,
    *,
    total_gb: float = 16.0,
    transport: str = "auto",
    chunk_tokens: int | str = "auto",
    chunk_target_mib: int = 128,
    workers: int = 0,
    layers: int = 61,
    hidden_dim: int = 576,
    layout: str = "mla",
    page_tokens: int = 64,
    device: str = "cuda:0",
    warmup: int = 2,
    io_direct: bool | None = None,
    verbose: bool = False,
) -> dict:
    """Drive ``total_gb`` of KV through the real connector save+load path.

    Measures THREE things in ONE run (shared daemon/connector/tensors),
    folded in from the retired ``bench_packed_v2``:

      - THROUGHPUT (batched): every chunk bound in ONE metadata so the
        connector's parallel fan-out runs; aggregate GB/s over the wall.
      - LATENCY (per-chunk): a SEPARATE loop binding ONE chunk at a time and
        timing ``wait_for_save`` / ``start_load_kv`` + ``cuda.synchronize`` —
        p50/p95 ms. Per-chunk is what "latency" means; the batched fan-out is
        what "throughput" means, so both are kept.
      - CORRECTNESS (round-trip): each layer is seeded with a deterministic
        per-layer pattern, saved, the target blocks zeroed, cold-loaded, then
        the loaded blocks are compared to the pattern. Also folds in the
        block_size sanity assert (resolved logical block_size == page_tokens),
        surfaced as ``correct``/``correctness_detail`` rather than a crash.

    Returns a dict:
      {
        "save_gbps": float, "load_gbps": float,
        "save_ms_p50": float, "save_ms_p95": float,
        "load_ms_p50": float, "load_ms_p95": float,
        "correct": bool, "correctness_detail": str,
        "per_chunk_mib": float, "num_chunks": int,
        "resolved": {transport, gpu_direct, save_gpu_direct,
                     force_serial, layerwise, chunk_tokens,
                     save_workers, load_workers, layout},
      }

    Cold-load: the KV tensors are zeroed before the load loop, so each
    ``start_load_kv`` is a genuine reload from the file tier.
    ``io_direct`` is accepted for API symmetry but the connector resolves
    O_DIRECT itself (GPU-direct forces it; POSIX uses storage_classify) — it
    is echoed into ``resolved`` but does not override the connector.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"[connector-bench] {msg}", flush=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("connector bench requires a visible CUDA/ROCm device")

    _apply_transport_env(transport, workers)

    num_kv_channels = _num_kv_channels(layout)
    dtype = torch.bfloat16
    # Match production: INFERA_KVD_CHUNK_TOKENS defaults to "auto", which the
    # connector autosizes to INFERA_KVD_CHUNK_TARGET_MIB (128 MiB) per chunk.
    # A fixed small chunk (512 = 34 MiB for MLA) under-reports load — parallel
    # fan-out needs chunks >= 128 MiB (kvd_connector.py:5568). So "auto"/0 here
    # sizes chunk_tokens to chunk_target_mib, the same target the daemon uses.
    if isinstance(chunk_tokens, str):
        chunk_tokens = 0 if chunk_tokens.strip().lower() in ("auto", "") else int(chunk_tokens)
    if chunk_tokens <= 0:
        # Honor a user-set INFERA_KVD_CHUNK_TOKENS (their production config)
        # first; else size to chunk_target_mib (the daemon's own autosize target).
        env_ct = os.environ.get("INFERA_KVD_CHUNK_TOKENS", "").strip()
        if env_ct.isdigit() and int(env_ct) > 0:
            chunk_tokens = int(env_ct)
            _log(f"chunk_tokens from user env INFERA_KVD_CHUNK_TOKENS={chunk_tokens}")
        else:
            per_token_bytes = num_kv_channels * layers * hidden_dim * 2
            raw = int(chunk_target_mib * (1 << 20) / per_token_bytes)
            chunk_tokens = max(page_tokens, (raw // page_tokens) * page_tokens)
            _log(f"chunk_tokens auto-sized to {chunk_tokens} (~{chunk_target_mib} MiB/chunk)")
    pages_per_chunk = chunk_tokens // page_tokens
    if pages_per_chunk < 1:
        raise ValueError(f"chunk_tokens {chunk_tokens} < page_tokens {page_tokens}")

    # bf16 (2 bytes) × channels (2 regular / 1 MLA) × layers × tokens × hidden.
    per_chunk_bytes = num_kv_channels * layers * chunk_tokens * hidden_dim * 2
    num_chunks = max(1, math.ceil(total_gb * 1e9 / per_chunk_bytes))
    # +slack to fit the disjoint save AND load ranges in one cache; both span
    # (num_chunks + warmup) chunks and the load range starts after the save.
    num_blocks = pages_per_chunk * (num_chunks + warmup) * 2 + 4

    _log(
        f"layout={layout} channels={num_kv_channels} layers={layers} "
        f"hidden={hidden_dim} chunk_tokens={chunk_tokens} pages/chunk={pages_per_chunk} "
        f"per_chunk={per_chunk_bytes / 1024 / 1024:.2f}MiB num_chunks={num_chunks}"
    )

    tmp_socket = tempfile.NamedTemporaryFile(delete=False).name
    os.unlink(tmp_socket)
    os.makedirs(target_dir, exist_ok=True)
    hipfile_root = tempfile.mkdtemp(prefix="kvd-connector-bench-", dir=target_dir)

    device_t = torch.device(device)
    daemon = None
    save_times: list[float] = []
    load_times: list[float] = []
    save_lat: list[float] = []
    load_lat: list[float] = []
    resolved: dict = {}
    correct = True
    correctness_detail = ""
    try:
        daemon = _start_daemon(tmp_socket, hipfile_root, max_bytes=2 << 30)
        _log(f"kvd daemon up (pid={daemon.pid}) socket={tmp_socket}")

        layers_map = {
            f"layer.{i}": _make_layer(
                torch, layout, num_blocks, page_tokens, hidden_dim, dtype, device_t
            )
            for i in range(layers)
        }

        connector = _build_connector(tmp_socket, hipfile_root, chunk_tokens)
        # Provide the LOGICAL block_size via the group's kv_cache_spec so
        # _logical_block_size resolves from config (not the tensor shape) —
        # essential for mla-aiter, whose middle dim is 1. Mirrors production.
        connector._kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(layers_map.keys()),
                    kv_cache_spec=SimpleNamespace(block_size=page_tokens),
                )
            ],
            block_size=page_tokens,
        )
        connector.register_kv_caches(layers_map)

        resolved = {
            "transport": transport,
            "gpu_direct": bool(getattr(connector, "_gpu_direct", False)),
            "save_gpu_direct": bool(getattr(connector, "_save_gpu_direct", False)),
            "force_serial": bool(getattr(connector, "_no_p2pdma_force_serial", False)),
            "layerwise": getattr(connector, "_layerwise_mode", "?"),
            "chunk_tokens": chunk_tokens,
            "save_workers": os.environ.get("INFERA_KVD_SAVE_WORKERS", "auto"),
            "load_workers": os.environ.get("INFERA_KVD_LOAD_WORKERS", "auto"),
            "io_direct": io_direct,
            "layout": layout,
        }
        _log(f"resolved transport: {resolved}")

        # Block-size sanity (folded from bench_packed_v2): the worker must
        # resolve the LOGICAL block_size from kv_cache_spec, NOT the tensor's
        # middle dim (which is 1 for mla-aiter). A desync silently corrupts
        # the scatter, so it belongs to CORRECTNESS — surface it, don't crash.
        correct = True
        correctness_detail = ""
        try:
            resolved_bs = int(connector._group_kv_spec[0]["block_size"])
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            resolved_bs = None
            correct = False
            correctness_detail = f"could not resolve block_size ({exc})"
        if resolved_bs is not None and resolved_bs != page_tokens:
            correct = False
            correctness_detail = (
                f"resolved block_size {resolved_bs} != logical {page_tokens} "
                f"(layout={layout}) — block_size desync"
            )
        _log(f"block_size resolved={resolved_bs} logical={page_tokens} ok={correct}")

        # Deterministic per-layer pattern (distinct per layer — catches a
        # cross-layer mixup; a plain 1/32 multiple so it round-trips bit-exact
        # through bf16, enabling the correctness compare below).
        for li, t in enumerate(layers_map.values()):
            _seed_layer(t, li)
        torch.cuda.synchronize()

        from infera.engine.vllm.kvd_connector import InferaKvdConnectorMetadata

        layer_names = list(layers_map.keys())
        keys = [(c + 1).to_bytes(8, "little") for c in range(num_chunks)]
        save_pages = [
            tuple((c * pages_per_chunk + p,) for p in range(pages_per_chunk))
            for c in range(num_chunks)
        ]
        # Load into a DISJOINT block range so a zeroed target proves a genuine
        # reload (not a read-back of the still-resident save buffers).
        target_offset = num_chunks * pages_per_chunk
        load_pages = [
            tuple((target_offset + c * pages_per_chunk + p,) for p in range(pages_per_chunk))
            for c in range(num_chunks)
        ]

        # SAVE — bind ALL chunks in ONE metadata so the connector's parallel
        # fan-out actually runs. Binding one chunk per call (a naive loop)
        # leaves the executor with a single in-flight future = serial, which
        # massively under-reports throughput. `warmup` passes are discarded;
        # the last pass is measured.
        def _run_save() -> float:
            connector._connector_metadata = InferaKvdConnectorMetadata(
                packed_chunks_to_save=[
                    (save_pages[c], keys[c], "long", 0, layer_names) for c in range(num_chunks)
                ],
            )
            t0 = time.perf_counter()
            connector.wait_for_save()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        for it in range(warmup + 1):
            dt = _run_save()
            if it == warmup:
                save_times.append(dt)

        # COLD LOAD — bind ALL chunks in ONE metadata (parallel executor fans
        # out over N chunks). Zero the target blocks before EACH pass so the
        # measured reload is genuine, not a page-cache/GPU-resident echo.
        def _run_load() -> float:
            for t in layers_map.values():
                t.zero_()
            torch.cuda.synchronize()
            connector._connector_metadata = InferaKvdConnectorMetadata(
                packed_chunks_to_load=[
                    (load_pages[c], keys[c], 0, layer_names) for c in range(num_chunks)
                ],
            )
            t0 = time.perf_counter()
            connector.start_load_kv(None)
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        for it in range(warmup + 1):
            dt = _run_load()
            if it == warmup:
                load_times.append(dt)

        # CORRECTNESS — the last cold LOAD above left the seeded pattern in the
        # DISJOINT load-block range (target was zeroed, then reloaded from the
        # file tier). Compare every layer's loaded blocks against its pattern.
        # A block_size desync (mla-aiter) or a corrupting mount shows up here.
        lo = target_offset
        hi = target_offset + num_chunks * pages_per_chunk
        max_diff = 0.0
        bad_layer = None
        for li, t in enumerate(layers_map.values()):
            sl = _layer_block_range(t, layout, lo, hi, page_tokens)
            diff = (sl.to(dtype=torch.float32) - _layer_fill_value(li)).abs().max().item()
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-3 and bad_layer is None:
                bad_layer = li
        if bad_layer is not None and correct:
            correct = False
            correctness_detail = (
                f"loaded KV mismatch: first bad layer={bad_layer} "
                f"max_abs_diff={max_diff:.4g} (layout={layout})"
            )
        _log(
            f"correctness: correct={correct} max_abs_diff={max_diff:.4g} detail={correctness_detail}"
        )

        # LATENCY — a SEPARATE per-chunk loop: bind ONE chunk, time the single
        # save / load. This is what "latency" means (one chunk end-to-end),
        # distinct from the batched fan-out throughput above. The connector is
        # already warm from the throughput passes, so no extra warmup is needed.
        for c in range(num_chunks):
            connector._connector_metadata = InferaKvdConnectorMetadata(
                packed_chunks_to_save=[(save_pages[c], keys[c], "long", 0, layer_names)],
            )
            t0 = time.perf_counter()
            connector.wait_for_save()
            torch.cuda.synchronize()
            save_lat.append(time.perf_counter() - t0)

        # Zero the whole cache once so each single-chunk load is a genuine cold
        # reload (each chunk writes only its own disjoint blocks).
        for t in layers_map.values():
            t.zero_()
        torch.cuda.synchronize()
        for c in range(num_chunks):
            connector._connector_metadata = InferaKvdConnectorMetadata(
                packed_chunks_to_load=[(load_pages[c], keys[c], 0, layer_names)],
            )
            t0 = time.perf_counter()
            connector.start_load_kv(None)
            torch.cuda.synchronize()
            load_lat.append(time.perf_counter() - t0)
    finally:
        _teardown(daemon, hipfile_root, tmp_socket)

    # Batch throughput: ALL chunks moved in one metadata, so the aggregate is
    # (num_chunks × per_chunk_bytes) / wall — the real multi-chunk fan-out rate.
    total_bytes = num_chunks * per_chunk_bytes
    save_gbps = total_bytes / statistics.median(save_times) / 1e9 if save_times else None
    load_gbps = total_bytes / statistics.median(load_times) / 1e9 if load_times else None

    def _p50_p95_ms(times: list[float]) -> tuple[float | None, float | None]:
        if not times:
            return None, None
        p50 = statistics.median(times) * 1000
        p95 = statistics.quantiles(times, n=20)[-1] * 1000 if len(times) >= 5 else max(times) * 1000
        return p50, p95

    save_ms_p50, save_ms_p95 = _p50_p95_ms(save_lat)
    load_ms_p50, load_ms_p95 = _p50_p95_ms(load_lat)
    return {
        "save_gbps": save_gbps,
        "load_gbps": load_gbps,
        "save_ms_p50": save_ms_p50,
        "save_ms_p95": save_ms_p95,
        "load_ms_p50": load_ms_p50,
        "load_ms_p95": load_ms_p95,
        "correct": correct,
        "correctness_detail": correctness_detail,
        "per_chunk_mib": per_chunk_bytes / 1024 / 1024,
        "num_chunks": num_chunks,
        "resolved": resolved,
    }

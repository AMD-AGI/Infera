###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM `KVConnectorBase_V1` impl that proxies to infera-kvd.

Phase 4.7 — adds **packed multi-layer blobs**, cutting per-block
kvd round-trips from N (layers) down to 1. Phase 4.6's per-layer
path stays as a fallback for partial transfers.

## What's implemented

**Scheduler-side**: `get_num_new_matched_tokens` queries kvd's
`exists` op to detect blocks already cached so vLLM can short-circuit
prefill for matching prefixes.

**Worker-side, two encodings — both processed every step:**

1. **Unpacked** (one kvd key per (block, layer)):
   - `blocks_to_load` → one GET per (block, layer)
   - `blocks_to_save` → one SET per (block, layer) inside `save_kv_layer`
   Best for partial transfers (a single layer's bytes; mixed-engine
   scenarios; small models).

2. **Packed** (one kvd key per block, value = all-N-layers blob):
   - `packed_blocks_to_load` → one GET per block → unpack header →
     dispatch N layer slices into N paged-buffer tensors
   - `packed_blocks_to_save` → `save_kv_layer` STASHES per-layer
     bytes; `wait_for_save` assembles one packed blob per block and
     emits one SET
   Best for full-block transfers — N→1 round-trip reduction.

Format details in `infera/engine/vllm/packed_format.py`.

**Lifecycle**: `update_state_after_alloc`, `build_connector_meta`,
`request_finished`, `register_kv_caches`, `shutdown`.

`wait_for_layer_load` is still a no-op (loads are synchronous in v1).
`wait_for_save` does real work for packed (drains the per-block
accumulator) but stays a no-op for unpacked.

## What's deferred (later phases)

- **Async per-layer pipelining**. Overlap kvd I/O with previous-layer
  attention compute. Hooks already in place — `wait_for_layer_load` /
  `wait_for_save` would gain real semantics.
- **Zero-copy GPU↔host** via pinned-buffer ring + DMA. Today bytes
  flow through `.cpu().view(uint8).numpy().tobytes()` — correctness-
  first, not throughput-first.
- **attn_metadata-driven slot mapping** for save (today we trust the
  scheduler-side save list).

## Lazy imports

vLLM and torch are imported lazily so this module imports on a
router-only host or in unit tests where neither is installed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import errno
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from infera.engine.vllm._l3_reaper import L3FileReaper, startup_budget_clamp
from infera.kvd.client import KvdClient, KvdConnectionError, KvdProtocolError

# AIS/hipFile gates its GPU-direct DMA fastpath to block devices / ext4-ordered
# / xfs and EXCLUDES NFS by default — but our production L3 tier is RDMA-mounted
# NFS, which would then silently bounce through a host buffer (a ~12 s Memcpy
# HtoD per step). The kernel AIS path is generic (vfs O_DIRECT read straight
# into GPU P2P pages) and DMAs NFS fine over NFSoRDMA once hipFile is told to
# allow "unsupported" filesystems via HIPFILE_UNSUPPORTED_FILE_SYSTEMS=1.
# libhipfile reads that env exactly ONCE (cached in a C++ static on the first
# fastpath check), so it MUST be set before any hipFile call — set it at module
# import, the earliest point, well before the lazy `import hipfile` in the
# connector's gpu_direct probe. setdefault: an explicit operator override
# (HIPFILE_UNSUPPORTED_FILE_SYSTEMS=0) still wins. Harmless when gpu_direct is
# off — the fastpath is simply never exercised. Verified on g04u07: with this,
# the connector's 128 MiB NFS chunk reads DMA into VRAM (no host bounce).
os.environ.setdefault("HIPFILE_UNSUPPORTED_FILE_SYSTEMS", "1")

# main's tablespace storage (#22/#37/#41) replaced the composite-filename
# helpers + TIER_* wire constants. The hipFile direct-file path that used
# them is disabled when they're absent; save/load fall back to RPC
# set/get (Phase-1 of the chunked-fusion-onto-main-storage merge).
try:
    from infera.kvd.ssd import (
        _composite_hash,
        _encode_composite,
        _filename_for_composite,
    )
except ImportError:
    _composite_hash = _encode_composite = _filename_for_composite = None  # type: ignore[assignment]
try:
    from infera.kvd.wire import TIER_FILE, TIER_MISS, TIER_RAM
except ImportError:
    TIER_FILE, TIER_MISS, TIER_RAM = object(), object(), object()  # sentinels

if TYPE_CHECKING:
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: F401
        KVConnectorBase_V1 as _RealKVConnectorBase,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata as _RealKVConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole as _RealKVConnectorRole,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        SupportsHMA as _RealSupportsHMA,
    )

    KVConnectorBase_V1 = _RealKVConnectorBase
    KVConnectorMetadata = _RealKVConnectorMetadata
    KVConnectorRole = _RealKVConnectorRole
    SupportsHMA = _RealSupportsHMA
    _VLLM_AVAILABLE = True
    try:
        # The connector-stats API (get_kv_connector_stats /
        # build_kv_connector_stats) exists in both vLLM 0.19.x and 0.22.x.
        # The base lives in the metrics module; fall back to a stub so a
        # vLLM build without it still imports.
        from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
            KVConnectorStats as _RealKVConnectorStats,
        )

        KVConnectorStats = _RealKVConnectorStats
        _KVD_STATS_AVAILABLE = True
    except ImportError:  # pragma: no cover — older vLLM without stats API

        class KVConnectorStats:  # type: ignore[no-redef]
            """Stand-in when this vLLM build lacks the stats base."""

            def __init__(self, data=None) -> None:
                self.data = data or {}

        _KVD_STATS_AVAILABLE = False
except ImportError:  # pragma: no cover — only without vLLM installed

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        """Stand-in so this module imports without vLLM. The real ABC
        replaces this when the worker subprocess loads us."""

        def __init__(self, *args, **kwargs) -> None:
            pass

    class KVConnectorMetadata:  # type: ignore[no-redef]
        """Stand-in metadata."""

    class KVConnectorRole:  # type: ignore[no-redef]
        SCHEDULER = 0
        WORKER = 1

    class SupportsHMA:  # type: ignore[no-redef]
        """Stand-in for vLLM's SupportsHMA mixin."""

    class KVConnectorStats:  # type: ignore[no-redef]
        """Stand-in connector-stats container (no vLLM)."""

        def __init__(self, data=None) -> None:
            self.data = data or {}

    _VLLM_AVAILABLE = False
    _KVD_STATS_AVAILABLE = False


# vLLM's per-connector Prometheus metrics base. A connector that emits stats
# via get_kv_connector_stats MUST also register here via build_prom_metrics,
# else MultiKVConnectorPromMetrics.observe() asserts and kills the engine as
# soon as metrics are on (i.e. WITHOUT --disable-log-stats). Guarded so this
# module still imports on a vLLM build lacking the metrics base (or no vLLM).
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics as _RealKVConnectorPromMetrics,
    )

    KVConnectorPromMetrics = _RealKVConnectorPromMetrics
    _KVD_PROM_AVAILABLE = True
except Exception:  # pragma: no cover — older/absent vLLM metrics module

    class KVConnectorPromMetrics:  # type: ignore[no-redef]
        """Stand-in when this vLLM build lacks the prom-metrics base."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def observe(self, *args, **kwargs) -> None:
            pass

    _KVD_PROM_AVAILABLE = False


class InferaKvdPromMetrics(KVConnectorPromMetrics):
    """No-op Prometheus adapter for the kvd connector.

    vLLM's MultiConnector metrics path asserts that every connector emitting
    stats via ``get_kv_connector_stats`` is ALSO registered here via
    ``build_prom_metrics`` — otherwise enabling metrics (NOT passing
    ``--disable-log-stats``) crashes the engine with
    ``AssertionError: InferaKvdConnector is not contained in the list of
    registered connectors with Prometheus metrics support``.

    kvd's L3 telemetry is already surfaced operator-visibly via
    ``get_kv_connector_stats``; mapping the individual counters onto
    Prometheus gauges is a follow-up, so ``observe`` is a no-op for now — its
    only job is to satisfy the registration contract and stop the crash.
    """

    def observe(self, transfer_stats_data, engine_idx: int = 0) -> None:
        return


# Use vLLM's logger so our startup diagnostics (gpu_direct mode, resolved
# I/O worker count / nconnect, P2PDMA probe) actually surface in the engine
# log — a plain logging.getLogger() does not propagate to vLLM's configured
# handlers, so those INFO lines were invisible. Fall back to stdlib logging
# outside a vLLM process (e.g. the SGLang adapter / unit tests).
try:
    from vllm.logger import init_logger as _init_logger

    logger = _init_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)
_DEFAULT_SOCKET_PATH = "/var/run/infera-kvd.sock"

# Guard rails for reset_cache()'s L3 wipe. A real hipFile L3 root looks like
# /kvd/long or /mnt/.../kvd-pdsw/long. If a root is ever misconfigured (empty,
# "/", or a shallow system dir), wiping its contents would be catastrophic — so
# _safe_wipe_dir_contents refuses anything that isn't a sufficiently-deep,
# non-system, non-root directory. It also deletes only the CONTENTS (never the
# root dir / mount point itself) and never follows symlinks out of the tree.
_WIPE_MIN_DEPTH = 2  # require at least "/a/b"
_WIPE_FORBIDDEN = frozenset(
    (
        "/",
        "/root",
        "/home",
        "/etc",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/usr",
        "/var",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
        "/tmp",
        "/mnt",
        "/opt",
        "/run",
        "/srv",
        "/models",
        "/kvd",
    )
)


def _safe_wipe_dir_contents(path: str) -> int:
    """Delete the CONTENTS of ``path`` (not ``path`` itself) with guard rails so
    a bad/empty/"/" root can never wipe the system.

    Returns the number of top-level entries removed. A missing path or non-dir is
    a no-op (returns 0 — the L3 simply isn't populated yet). Raises ValueError if
    ``path`` fails a safety check; the caller logs it and treats it as a failure.
    """
    if not path or not isinstance(path, str):
        raise ValueError(f"refuse to wipe empty/invalid L3 path: {path!r}")
    # Resolve symlinks, "..", and "//" so the checks see the real target.
    real = os.path.realpath(path)
    if real == os.sep or real in _WIPE_FORBIDDEN:
        raise ValueError(f"refuse to wipe protected/root path: {real!r} (from {path!r})")
    depth = len([p for p in real.split(os.sep) if p])
    if depth < _WIPE_MIN_DEPTH:
        raise ValueError(
            f"refuse to wipe shallow path (depth {depth} < {_WIPE_MIN_DEPTH}): {real!r}"
        )
    if not os.path.isdir(real):
        return 0  # nothing to clear — not an error
    removed = 0
    for name in os.listdir(real):
        child = os.path.join(real, name)
        try:
            # Never rmtree through a symlink (could escape the tree); unlink it.
            if os.path.islink(child) or not os.path.isdir(child):
                os.remove(child)
            else:
                shutil.rmtree(child, ignore_errors=False)
            removed += 1
        except FileNotFoundError:
            pass  # raced with a concurrent wipe / save — fine
    return removed


# GPU-direct (AIS / AMD Infinity Storage hipFile) toggle env var. Renamed
# INFERA_KVD_GPU_DIRECT -> INFERA_KVD_AIS; the legacy name is still
# honored (deprecated) for backward compatibility.
_AIS_ENV = "INFERA_KVD_AIS"
_AIS_ENV_LEGACY = "INFERA_KVD_GPU_DIRECT"


def _read_ais_env(default: str) -> str:
    """Read the AIS/GPU-direct toggle. Prefers INFERA_KVD_AIS; falls back
    to the deprecated INFERA_KVD_GPU_DIRECT (with a one-time warning).
    Returns ``default`` if neither is set."""
    raw = os.environ.get(_AIS_ENV)
    if raw is None:
        raw = os.environ.get(_AIS_ENV_LEGACY)
        if raw is not None:
            logger.warning(
                "%s is deprecated; use %s (it sets GPU-direct / AIS hipFile I/O).",
                _AIS_ENV_LEGACY,
                _AIS_ENV,
            )
    return raw if raw is not None else default


def _is_packed_quant_kv_dtype(dtype: Any) -> bool:
    """True for fp8 / sub-byte KV-cache dtypes whose per-token "hidden" run
    may pack quantization SCALES alongside the data (deepseek_v32 fp8_ds_mla's
    656/584-byte latent, per-token-head-scale fp8 attention, nvfp4).

    The shape probe classifies KV caches by DIMENSIONALITY only, and the
    chunked-fusion gather/scatter kernel copies raw elements with NO scale
    awareness. For a scale-packed cache the copy can silently mis-stride into
    garbage on reload. So register_kv_caches SKIPS a packed-dtype group UNLESS
    it is the provably-safe plain-MLA-fp8 case — hidden == kv_lora_rank +
    qk_rope_head_dim, a plain cast with no interleaved scale (auto-detected via
    `_expected_plain_mla_hidden`; Kimi-K2.6 576 = 512 + 64, validated byte-exact).
    bf16/fp16/fp32 are NOT packed and pass through unconditionally."""
    import torch

    names = (
        "uint8",
        "int8",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    packed = {t for t in (getattr(torch, n, None) for n in names) if t is not None}
    return dtype in packed


def _is_mla_from_config(vllm_config: Any) -> bool:
    """True if the model uses MLA (Multi-head Latent Attention, e.g.
    DeepSeek-V3 / Kimi). MLA caches a single compressed latent that is
    REPLICATED across TP ranks (the TP sharding is only in the
    up-projection at attention time, which is not stored) — so the KV
    written by every TP rank is byte-identical. Detected from config so
    BOTH the scheduler-side and worker-side connectors agree at init,
    before ``register_kv_caches`` runs. Prefers vLLM's ``use_mla`` flag;
    falls back to a ``kv_lora_rank`` marker in the (text) HF config."""
    try:
        mc = getattr(vllm_config, "model_config", None)
        if mc is None:
            return False
        use_mla = getattr(mc, "use_mla", None)
        if isinstance(use_mla, bool):
            return use_mla
        hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
        for cand in (hf, getattr(hf, "text_config", None)):
            klr = getattr(cand, "kv_lora_rank", None)
            if isinstance(klr, int) and klr > 0:
                return True
    except Exception:
        return False
    return False


def _expected_plain_mla_hidden(vllm_config: Any) -> int | None:
    """Return ``kv_lora_rank + qk_rope_head_dim`` from the (text) HF config,
    else None.

    For an MLA model this is the hidden width of the PLAIN compressed latent
    (compressed KV + decoupled RoPE). When a registered fp8/uint8 KV tensor's
    ``hidden_dim`` equals this EXACTLY, the quantized cache is a plain cast
    with NO per-token scale bytes interleaved into the hidden run, so the
    raw-byte chunked gather/scatter round-trips it byte-exact (validated on
    Kimi-K2.6: 576 = 512 + 64, pure fp8, 560/560 L3 hits, 0 miss). A
    scale-PACKED layout (fp8_ds_mla's 656/584 = latent + interleaved tile
    scales + bf16 rope, or nvfp4) stores a LARGER hidden, so it will NOT
    match and stays guarded. This lets register_kv_caches auto-allow the
    provably-safe plain-MLA-fp8 case without the operator toggle, while the
    genuinely scale-packed formats keep skipping (correctness over an
    optimization)."""
    try:
        mc = getattr(vllm_config, "model_config", None)
        if mc is None:
            return None
        hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
        for cand in (hf, getattr(hf, "text_config", None)):
            klr = getattr(cand, "kv_lora_rank", None)
            rope = getattr(cand, "qk_rope_head_dim", None)
            if isinstance(klr, int) and klr > 0 and isinstance(rope, int) and rope > 0:
                return klr + rope
    except Exception:
        return None
    return None


# Substrings (lower-cased) of vLLM v1 KVCacheSpec class names whose state
# is NOT a paged attention KV cache: recurrent / convolutional / linear-
# attention state (Mamba, Mamba2, short-conv, linear attention). The kvd
# L3 connector's chunk format assumes a paged attention layout
# (``[..., num_blocks, block_size, hidden]``); these groups have no
# block-table prefix to reuse and MUST be skipped — never offloaded.
# Critically, a Mamba SSM/conv-state tensor is often 3-D and would
# otherwise fall into the ``len(shape) == 3 → MLA`` branch of
# ``register_kv_caches`` and be mis-detected as an MLA latent cache.
_NON_PAGED_SPEC_NAME_HINTS = (
    "mamba",
    "conv",
    "recurrent",
    "linearattention",
    "lineattn",
    "shortconv",
    "ssm",
)


# DSA (DeepSeek Sparse Attention) indexer cache — used by glm_moe_dsa
# (GLM-5) and deepseek_v32. vLLM registers the indexer via
# DeepseekV32IndexerCache.get_kv_cache_spec() as an MLAAttentionSpec
# ("only one vector instead of K+V"), so it LOOKS like a normal MLA
# latent group to the shape-based path and would be offloaded. But the
# indexer holds an auxiliary sparse-topk index, NOT the reusable verbatim
# KV — the engine recomputes it from the main MLA latent. Offloading it to
# L3 is wasted bytes and risks correctness on reload, so skip the indexer
# group entirely (offload only the main MLA latent). Detected by layer
# name; no current non-DSA model registers "indexer" KV cache layers, so
# this is a strict no-op for everything except glm_moe_dsa / deepseek_v32.
# NOTE (2026-06-24): detector validated by unit test; full-model validation
# pending a glm_moe_dsa-capable vLLM build (no image ships it yet). See
# cache/KVD_GLM5_SUPPORT_ANALYSIS.md.
_DSA_INDEXER_NAME_HINTS = ("indexer",)


def _is_dsa_indexer_group(layer_names: Any) -> bool:
    """True if this kv_cache_group is a DSA sparse-attention *indexer* cache
    (glm_moe_dsa / deepseek_v32) — auxiliary, recomputable, must NOT be
    offloaded to L3. Matches on layer-name substring (case-insensitive)."""
    try:
        for n in layer_names or ():
            ln = str(n).lower()
            if any(h in ln for h in _DSA_INDEXER_NAME_HINTS):
                return True
    except Exception:
        pass
    return False


def _is_dsa_indexer_layer(name: Any) -> bool:
    """True for a single DSA indexer cache layer (`...self_attn.indexer.k_cache`)."""
    ln = str(name).lower()
    return any(h in ln for h in _DSA_INDEXER_NAME_HINTS)


def _split_dsa_layers(layer_names: Any) -> tuple[list, list]:
    """Split a mixed DSA group's layer names into (main-latent, indexer)."""
    main, idx = [], []
    for n in layer_names or ():
        (idx if _is_dsa_indexer_layer(n) else main).append(n)
    return main, idx


# gid base for the indexer sub-spec so it never collides with a real group id.
_DSA_INDEXER_GID_BASE = 1000


def _is_non_paged_kv_spec(spec: Any) -> bool:
    """True if ``spec`` describes a non-attention recurrent/conv state
    group (Mamba, linear attention, short-conv) that the L3 connector
    must NOT offload. Best-effort and version-robust:

      1. Positive attention signal first — both regular ``AttentionSpec``
         and ``MLAAttentionSpec`` expose ``num_kv_heads`` + ``head_size``.
         When present it is a paged attention group we DO handle (MLA
         included), so never skip.
      2. Hard ``isinstance(MambaSpec)`` when that class is importable.
      3. Class-name substring fallback — covers future recurrent spec
         classes and the lightweight stub specs used in unit tests.

    A None spec (older vLLM / stub configs that omit it) is treated as
    NOT non-paged, preserving the legacy shape-based path.
    """
    if spec is None:
        return False
    # (1) Attention specs win outright.
    if (
        getattr(spec, "num_kv_heads", None) is not None
        and getattr(spec, "head_size", None) is not None
    ):
        return False
    # (2) Authoritative isinstance against MambaSpec when present.
    try:
        from vllm.v1.kv_cache_interface import MambaSpec  # type: ignore

        if isinstance(spec, MambaSpec):
            return True
    except Exception:
        pass
    # (3) Name-based fallback.
    name = type(spec).__name__.lower()
    return any(hint in name for hint in _NON_PAGED_SPEC_NAME_HINTS)


def _torch_dtype_to_str(dtype) -> str:
    """Map a torch.dtype to the short string used in v2 ChunkHeader.
    Unknown dtypes default to ``bf16`` since KV is rarely 4 bytes in
    production; the consumer's `dtype_bytes` lookup also defaults to
    2 so the round-trip stays self-consistent."""
    import torch

    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.uint8: "fp8_e4m3",  # vLLM stores fp8 KV as uint8 under the hood
        getattr(torch, "float8_e4m3fn", None): "fp8_e4m3",
        getattr(torch, "float8_e4m3fnuz", None): "fp8_e4m3",
        getattr(torch, "float8_e5m2", None): "fp8_e5m2",
    }.get(dtype, "bf16")


def _chunk_kvd_key_for(block_hashes: list[bytes], start: int, count: int) -> bytes:
    """Compute the 8-byte content-addressed kvd_key for an N-page
    chunk that starts at prompt-block index `start` and covers
    `count` consecutive pages. Stable across processes (block_hashes
    are vLLM block-content hashes, themselves stable when
    ``PYTHONHASHSEED=0`` per memory `project_pythonhashseed_required`).

    Truncated to 8 bytes to match v1's key width — collision risk at
    2^32 chunks is below 1 in 1M, acceptable.
    """
    import hashlib

    h = hashlib.sha256()
    for i in range(start, start + count):
        h.update(block_hashes[i])
    return h.digest()[:8]


# ----------------------------------------------------------------------
# Connector metadata — produced by the scheduler-side connector each
# step, consumed by the worker-side connector to know what to load.
# ----------------------------------------------------------------------


def _vllm_weight_fingerprint(vllm_config: Any) -> str:
    """Compute a stable 16-hex-char fingerprint of the model's
    weights/config — for inclusion in `compat_key` so a fine-tune at
    the same model name doesn't silently reuse the base model's KV
    cache (PR #9 review fix A).

    Strategy:
      1. Look at vllm_config.model_config.model — typically a local
         path or HF id.
      2. If it's a directory: hash `config.json` + `tokenizer.json`
         from inside (cheap, stable, defines model identity).
      3. If it's not resolvable: return "" (caller falls back to the
         rank-only compat_key). Operators see the choice in the
         INFO log emitted by `_extract_compat_key`.

    Why not weight files: hashing GBs of safetensors at every
    connector init is too slow. config.json is enough to distinguish
    different model architectures; tokenizer.json catches different
    vocab/tokenizer versions. A weights-identical-config-different
    finetune isn't catchable without hashing weights — operator
    must rotate `compat_key` explicitly in that case (env override
    is a future hook).
    """
    import hashlib
    from pathlib import Path

    model_path_or_id = ""
    for attr_path in (
        ("model_config", "model"),
        ("model_config", "served_model_name"),
        ("served_model_name",),
        ("model",),
    ):
        cur = vllm_config
        ok = True
        for attr in attr_path:
            cur = getattr(cur, attr, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.strip():
            model_path_or_id = cur.strip()
            break
    if not model_path_or_id:
        return ""
    # Must be a real local directory for us to hash contents.
    p = Path(model_path_or_id)
    if not p.is_dir():
        # HF id or unresolved — return empty to skip the fingerprint.
        # An operator running fine-tunes by HF id must use distinct
        # model names, which they already do.
        return ""
    h = hashlib.sha256()
    found_anything = False
    for filename in ("config.json", "tokenizer.json"):
        candidate = p / filename
        if candidate.is_file():
            try:
                with candidate.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 16), b""):
                        h.update(chunk)
                found_anything = True
            except OSError:
                pass
    if not found_anything:
        return ""
    return h.hexdigest()[:16]


@dataclass
class _SteppedChunkLoadState:
    """Per-(step, chunk) state for layerwise load Variant A.

    Built in ``start_load_kv`` (fetch + decode + CPU staging done once),
    consumed across N ``wait_for_layer_load`` calls (one per layer in
    iteration order). Retired when ``next_layer_idx == len(present)``.

    Variant A keeps the CPU payload in a regular tensor (not mmap'd)
    so ``mm_holder`` is closed eagerly during prep. Variant B will
    instead keep mmap alive and use a per-layer H2D on a dedicated
    copy stream.
    """

    kvd_key: bytes
    cache_group_id: int
    present_layers: tuple[str, ...]
    layer_tensors: list[Any]  # one paged-KV tensor per present layer
    layer_to_idx: dict[str, int]
    block_size: int
    chunk_tokens: int
    num_kv_channels: int  # 1 = MLA, 2 = regular K+V split
    hidden_dim: int
    cpu_payload: Any  # CPU [C, L, T, H] view of decoded bytes
    slot_mapping_device: Any  # int32 device tensor [chunk_tokens]
    next_layer_idx: int = 0


@dataclass
class _PrefetchChunkLoadState:
    """Per-(step, chunk) state for layerwise load Variant B.

    Variant B keeps mmap alive across yields and runs per-layer H2D on
    a dedicated copy_stream with prefetch lookahead. Pinned-host slots
    are reused round-robin (one slot per layer-in-flight, capped at
    ``prefetch_depth``). The default stream waits on copy_stream's
    event before launching each per-layer Triton scatter.
    """

    kvd_key: bytes
    cache_group_id: int
    present_layers: tuple[str, ...]
    layer_tensors: list[Any]  # one paged-KV per present layer
    layer_to_idx: dict[str, int]
    block_size: int
    chunk_tokens: int
    num_kv_channels: int
    hidden_dim: int
    payload_view: Any  # zero-copy memoryview into the mmap'd file
    payload_dtype: Any  # torch dtype (matches layer_tensors[0].dtype)
    per_layer_nbytes: int  # bytes per layer slice
    mm_holder: Any  # mmap; MUST outlive every wait_for_layer_load (or None when hipFile path)
    slot_mapping_device: Any  # int32 device tensor [chunk_tokens]
    # Round-robin per-chunk pinned slots, len = prefetch_depth.
    # pinned_slots[i] holds the bytes for whichever layer was last
    # prefetched into ring index i. None when hipFile-read async path
    # is active (reads go device-direct).
    pinned_slots: list[Any]  # list[_PinnedSlot | None]
    # Per-layer events recorded after H2D enqueue on copy_stream.
    events: Any  # _LayerEventRegistry
    # hipFile FileHandle for cuFileReadAsync direct-to-device path.
    # None when not using hipFile (either gpu_direct off, or no file
    # path, or hipfile binding doesn't support async).
    hipfile_handle: Any = None
    # Byte offset within the on-disk file where the payload begins.
    # For v3 (header-padded) chunks this is rounded up to 4 KiB so
    # per-layer cuFileReadAsync calls land on aligned offsets. For
    # v2 (legacy unpadded) it equals header_end.
    payload_file_offset: int = 0
    # Pre-allocated per-chunk device staging buffer (uint8, 4 KiB-
    # aligned via overallocate-and-offset). Covers
    # prefetch_depth × per_layer_nbytes_aligned. Registered ONCE with
    # hipFile.Buffer at prep time; recycled per ring slot via
    # buffer_offset = ring_idx * per_layer_nbytes_aligned. Avoids
    # per-layer Buffer.register/deregister churn (which trips the
    # "already registered" 5023 error because PyTorch's caching
    # allocator reuses data_ptr() between layers before our async
    # deregister fires).
    hipfile_buf: Any = None  # hipfile.Buffer
    hipfile_buf_dev: Any = None  # torch.Tensor (raw uint8, overallocated)
    hipfile_buf_prefix: int = 0  # bytes to skip in dev tensor to reach 4K boundary
    hipfile_buf_per_layer: int = 0  # rounded-up per_layer_nbytes (≥ per_layer_nbytes, mult of 4K)
    # Per-ring-slot device staging slice views (zero-copy) for scatter.
    # hipfile_slice_views[i] = dev tensor view at buffer_offset =
    # i * hipfile_buf_per_layer, length per_layer_nbytes.
    hipfile_slice_views: list[Any] = field(default_factory=list)
    # Per-layer _AsyncIOHandle (heap-allocated ctypes slots for
    # cuFileReadAsync's in/out pointers). MUST be kept alive until the
    # corresponding cuda.Event signals (driver writes bytes_done at
    # completion time). Cleared in retire alongside Buffer.deregister.
    # Keyed by layer_idx; one entry per in-flight async read.
    hipfile_io_handles: dict = field(default_factory=dict)
    next_layer_idx: int = 0


@dataclass
class _PackedChunkState:
    """Per-(chunk_kvd_key) staging buffer for vLLM-8 chunked-fusion
    saves. One chunk = N consecutive vLLM pages × all layers (within
    one kv_cache_group) → ONE on-disk file in v2 wire format.

    Lives in ``InferaKvdConnector._pending_chunk_saves[chunk_kvd_key]``.
    Filled incrementally as ``save_kv_layer`` calls arrive for each
    (layer, page) tuple of the chunk; flushed once all
    ``num_layers × N`` slots have arrived (or the request finishes
    early and we abort the partial chunk).

    ``buffer`` shape = ``[2, num_layers, chunk_tokens, hidden_dim]``
    matching ``packed_format.ChunkHeader``'s payload layout. Stored
    as a flat byte buffer (uint8) so the same code handles bf16 /
    fp16 / fp8 without dtype-specific branches; the dtype string in
    the header tells the reader how to view it.
    """

    chunk_kvd_key: bytes
    retention: str
    cache_group_id: int
    chunk_tokens: int
    block_size: int
    num_layers: int
    hidden_dim: int
    dtype: str  # "bf16" | "fp16" | "fp8_e4m3" | "fp32"
    layer_names: tuple[str, ...]  # in blob order (= layer_idx → name)
    # Per-page physical block ids per cache group, indexed
    # [page_idx_in_chunk][group_id]. For non-HMA single-group models
    # each inner tuple is a 1-tuple. Used by save_kv_layer to know
    # which block_id this layer's page lives at.
    per_page_block_ids: tuple[tuple[int, ...], ...]
    # The flat staging tensor — allocated lazily on first
    # save_kv_layer call so we don't pay the alloc when this state
    # is constructed early. shape: (2 × num_layers × chunk_tokens ×
    # hidden_dim × dtype_bytes,) uint8 on CPU. Pinned-host so the
    # eventual flush can H2D fast if we ever wire a save-side HIP
    # path; today the flush is POSIX write of the CPU bytes.
    buffer: torch.Tensor | None = None
    # Set of (layer_idx_in_blob, page_idx_in_chunk) tuples that have
    # been filled. When this reaches num_layers × N the chunk flushes.
    filled_slots: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class InferaKvdConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    """Per-step metadata flowing from scheduler → worker.

    One encoding: chunked-fusion (vLLM-8). Each save/load entry
    covers N consecutive vLLM pages × all layers within one
    kv_cache_group, packed as ``[2, num_layers, chunk_tokens,
    hidden_dim]`` (see ``packed_format.ChunkHeader``). One kvd GET
    or SET per chunk × cache_group; the per-(layer, page) v1 form
    was deleted in vLLM-9 as obsolete.
    """

    # Chunked-fusion entries — one per N-page chunk × cache_group.
    # Each entry's ``per_page_block_ids`` is a tuple of length N (pages
    # in the chunk), each element a per-cache-group block_id tuple.
    # ``layer_names`` is the explicit per-layer ordering for the
    # chunk's payload — chosen by the scheduler to match the layer
    # iteration order on the worker (typically vLLM's
    # `kv_caches.keys()` order). Worker reads ``self._chunk_tokens``
    # and ``self._kv_caches`` to know shapes.
    packed_chunks_to_load: list[tuple[tuple[tuple[int, ...], ...], bytes, int, list[str]]] = field(
        default_factory=list
    )
    packed_chunks_to_save: list[tuple[tuple[tuple[int, ...], ...], bytes, str, int, list[str]]] = (
        field(default_factory=list)
    )
    # Parallel to packed_chunks_to_load: the req_id each load entry belongs
    # to, so the worker can group load futures per request for async
    # get_finished reporting (Fix A). Empty/ignored on the sync path.
    load_chunk_req_ids: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------

# Counter keys carried in InferaKvdStats.data. Scheduler-side counts the
# L3 lookup hit/miss (get_num_new_matched_tokens); worker-side counts the
# actual save/load transfers and degrade events. vLLM aggregates the
# scheduler and worker connectors' stats each step, so all keys end up in
# one "KV Transfer metrics: ..." log line.
_KVD_STAT_KEYS = (
    "lookup_requests",  # scheduler: requests probed against L3
    "lookup_chunks",  # scheduler: chunks probed (prefix-aligned)
    "hit_chunks",  # scheduler: leading chunks found in L3
    "saved_chunks",  # worker: chunks written to L3
    "saved_bytes",  # worker: bytes written to L3
    "loaded_chunks",  # worker: chunks read back from L3
    "loaded_bytes",  # worker: bytes read back from L3
    "load_errors",  # worker: chunk loads that fell back to recompute
    "save_pool_misses",  # worker: staging-pool-full degrades
    "evictions",  # connector: saved-key TTL/LRU drops
)

# Emit a periodic INFO summary of cumulative L3 activity every this many
# saved chunks / lookups (see _stat_inc). Makes GPU-direct/file-tier offload
# observable without vLLM's stats logger (which --disable-log-stats mutes) and
# without the daemon statctl (all-zero under file-tier).
_L3_LOG_EVERY = 512


class InferaKvdStats(KVConnectorStats):  # type: ignore[misc]
    """Connector-stats container for infera-kvd.

    ``data`` holds the integer counters in ``_KVD_STAT_KEYS``. vLLM moves
    this object worker→scheduler→logger, calling ``aggregate`` to merge
    across workers/intervals and ``reduce`` to convert to the flat dict
    that gets logged / exported to Prometheus.
    """

    def __init__(self, data: dict | None = None) -> None:
        super().__init__(data=data if data is not None else {})
        for k in _KVD_STAT_KEYS:
            self.data.setdefault(k, 0)

    def reset(self) -> None:
        self.data = {k: 0 for k in _KVD_STAT_KEYS}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        od = getattr(other, "data", {}) or {}
        for k in _KVD_STAT_KEYS:
            self.data[k] = int(self.data.get(k, 0)) + int(od.get(k, 0))
        return self

    def is_empty(self) -> bool:
        return all(int(self.data.get(k, 0)) == 0 for k in _KVD_STAT_KEYS)

    def reduce(self) -> dict[str, int | float]:
        d = self.data
        lookups = int(d.get("lookup_chunks", 0))
        hits = int(d.get("hit_chunks", 0))
        out: dict[str, int | float] = {
            "l3_lookup_reqs": int(d.get("lookup_requests", 0)),
            "l3_lookup_chunks": lookups,
            "l3_hit_chunks": hits,
            "l3_miss_chunks": max(0, lookups - hits),
            "l3_hit_rate": round(hits / lookups, 4) if lookups else 0.0,
            "saved_chunks": int(d.get("saved_chunks", 0)),
            "saved_MiB": round(int(d.get("saved_bytes", 0)) / (1 << 20), 1),
            "loaded_chunks": int(d.get("loaded_chunks", 0)),
            "loaded_MiB": round(int(d.get("loaded_bytes", 0)) / (1 << 20), 1),
            "load_errors": int(d.get("load_errors", 0)),
            "save_pool_misses": int(d.get("save_pool_misses", 0)),
            "evictions": int(d.get("evictions", 0)),
        }
        return out


# ----------------------------------------------------------------------
# Connector
# ----------------------------------------------------------------------


class InferaKvdConnector(KVConnectorBase_V1, SupportsHMA):  # type: ignore[misc]
    """vLLM `KVConnectorBase_V1` impl that talks to infera-kvd.

    Inherits `SupportsHMA` because vLLM's hybrid memory allocator
    splits KV cache into multiple groups for SWA / Mamba models
    (e.g. gpt-oss, GLM with NSA). The scheduler routes those to
    `request_finished_all_groups` and refuses to start with non-HMA
    connectors for hybrid models. Our save/load paths handle the
    per-group block-table mapping in `build_connector_meta` (one
    entry per prompt block carrying `tuple[per-group physical block_id]`)
    and the worker resolves the right block_id per layer via
    `_layer_to_group` populated in `register_kv_caches`.

    Construction:
        config = vllm_config.kv_transfer_config
        connector = InferaKvdConnector(vllm_config, role=KVConnectorRole.SCHEDULER)

    vLLM's `KVConnectorFactory` (registered via the framework's
    `kv_transfer_config.kv_connector` knob) constructs one connector
    per role per process: SCHEDULER in the engine front-end, WORKER
    in each model-runner subprocess. Both connect to the same kvd
    daemon over the same UDS path.
    """

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
        """Require the NHD KV-cache layout.

        The gather/scatter kernel for the vLLM 0.23 FullAttention INTERLEAVED
        layout assumes the per-layer KV tensor is CONTIGUOUS in
        ``[num_blocks, 2, block_size, num_kv_heads, head_dim]`` memory order
        (block → K/V → token → head → head_dim). That holds only when the
        runtime KV layout is NHD (heads adjacent to head_dim, innermost). If
        the layout were HND, num_kv_heads would be pulled forward and the
        tensor would no longer be contiguous in the order the kernel walks,
        so the per-block gap arithmetic would be wrong. Declare NHD so vLLM
        allocates the layout the kernel is written against.
        """
        return "NHD"

    @staticmethod
    def _extract_socket_path(vllm_config: Any, explicit: str | None = None) -> str:
        """Resolve the kvd socket path.

        Precedence (first hit wins):

        1. ``explicit`` — the ``socket_path`` constructor kwarg (tests /
           programmatic callers).
        2. ``kv_transfer_config.kv_connector_extra_config["socket"]`` (or
           ``"socket_path"``) — the config-driven path. This is the natural seam for
           vLLM operators since the connector is instantiated by vLLM's
           ``KVConnectorFactory`` from the ``--kv-transfer-config`` JSON,
           with no infera launcher in the loop to set an env var.
        3. ``INFERA_KVD_SOCKET`` env var — process-wide fallback.
        4. ``_DEFAULT_SOCKET_PATH`` — last resort.
        """
        if explicit:
            return explicit

        try:
            cfg = getattr(vllm_config, "kv_transfer_config", None)
            extra = getattr(cfg, "kv_connector_extra_config", None)
            if isinstance(extra, dict):
                candidate = extra.get("socket") or extra.get("socket_path")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        except Exception:
            # A malformed config shouldn't crash construction — fall through
            # to env/default. Log at debug so the resolution is traceable
            # without spamming the normal startup path.
            logger.debug(
                "kvd socket: kv_connector_extra_config lookup failed; falling back to env/default",
                exc_info=True,
            )

        return os.environ.get("INFERA_KVD_SOCKET") or _DEFAULT_SOCKET_PATH

    def __init__(
        self,
        vllm_config: Any,
        role: Any,
        kv_cache_config: Any = None,
        *,
        socket_path: str | None = None,
        hipfile_roots: dict[str, str] | None = None,
        gpu_direct: bool | None = None,
    ) -> None:
        # `super().__init__` does logging + stashes config; safe to call
        # even when vLLM isn't installed (stand-in is a no-op).
        try:
            super().__init__(vllm_config, role, kv_cache_config)
        except Exception:
            # Stand-in __init__ accepts anything; real one may raise if
            # vllm_config doesn't have kv_transfer_config. We don't catch
            # in production; only here to keep module-import tests sane.
            pass

        self._role = role
        self._socket_path = self._extract_socket_path(vllm_config, socket_path)
        self._model = self._extract_model_id(vllm_config)
        self._compat_key = self._extract_compat_key(vllm_config)
        # MLA TP-dedup (#64): under MLA the cached latent is identical on
        # every TP rank, so _extract_compat_key folds them onto one
        # namespace (tp_rank 0) and only rank 0 persists it (the write
        # gate in wait_for_save). tp_size>1 keeps pure-DP / single-rank
        # correct — there each rank's KV is DISTINCT.
        self._tp_rank, self._tp_size = self._resolve_tp_rank_size(vllm_config)
        # Plain-MLA-fp8 auto-detect: kv_lora_rank + qk_rope_head_dim from the
        # config. A packed KV tensor whose hidden matches this EXACTLY is a
        # plain fp8 cast (no interleaved scale) → safe to offload; register
        # allows it automatically (no toggle). Scale-packed layouts
        # (656/584/nvfp4) have a larger hidden and stay skipped. See
        # `_expected_plain_mla_hidden`.
        self._plain_mla_hidden = _expected_plain_mla_hidden(vllm_config)
        # Set True once a non-paged (Mamba / linear-attention / conv) cache
        # group is observed (register_kv_caches / bootstrap). vLLM 0.22.x does
        # NOT support external KV-connector LOADS for hybrid models — its
        # scheduler `_mamba_block_aligned_split` asserts
        # `num_external_computed_tokens == 0` ("External KV connector is not
        # verified yet"). So on hybrid models we must report ZERO external
        # matched tokens, or the engine dies on the first L3 hit. Saves stay
        # safe (attention-only); only the load advertisement is gated.
        self._has_non_paged_groups = False
        # Set True once a DSA sparse-attention indexer cache group (glm_moe_dsa /
        # deepseek_v32) is observed and skipped in register_kv_caches.
        self._has_aux_skipped_groups = False
        # Warn-once guard for the hybrid no-LOAD path (see get_num_new_matched_tokens).
        self._warned_hybrid_no_load = False
        self._dedup_mla_writes = self._tp_size > 1 and _is_mla_from_config(vllm_config)
        if self._dedup_mla_writes:
            logger.info(
                "kvd MLA TP-dedup ON: tp_rank=%d/%d — %s",
                self._tp_rank,
                self._tp_size,
                "rank 0 persists the shared latent"
                if self._tp_rank == 0
                else "skipping redundant per-rank writes",
            )

        # Background asyncio loop for the sync↔async bridge. Same pattern
        # as SGLang adapter — vLLM calls us synchronously from its worker
        # thread / scheduler step, but kvd is async.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client: KvdClient | None = None
        self._kv_caches: dict[str, torch.Tensor] = {}
        # Layer-name → kv_cache_group index, populated in
        # `register_kv_caches` from `self._kv_cache_config`. Empty
        # default means "single group at index 0" (the non-HMA case).
        # Used by the gather/scatter kernel to project per-page block_id
        # tuples (which carry IDs for every group) down to this layer's
        # specific group's value.
        self._layer_to_group: dict[str, int] = {}
        # Chunk-window size in tokens. Larger = better I/O amortization
        # (and bigger per-call BW), smaller = finer prefix-sharing
        # granularity. Default 512 picked so even small-per-rank models
        # (gpt-oss-120b TP=8 ≈ 4.5 MiB per chunk) cross the LMCache-
        # reported 1-2 MiB efficient I/O floor. Capped at 2048 to bound
        # the in-flight chunk staging buffer per worker.
        # Chunk size. An EXPLICIT token count (INFERA_KVD_CHUNK_TOKENS=N)
        # always wins. Otherwise "auto": derive chunk_tokens after the KV
        # spec is known so each chunk is >= INFERA_KVD_CHUNK_TARGET_MIB
        # (default 128 MiB) — the P2PDMA read bandwidth sweet spot (128 MiB
        # ≈ 21 GB/s; 36 MiB is scatter-bound at ~2.5 GB/s). Per-token KV
        # bytes vary ~10x across models (gpt-oss 72 KiB/tok vs Kimi much
        # more), so a fixed token count mis-sizes the chunk — size by BYTES.
        # The derivation uses the KV spec (identical on scheduler+worker),
        # so the chunk grain — and thus chunk keys — stay consistent.
        _ct_env = os.environ.get("INFERA_KVD_CHUNK_TOKENS", "auto").strip().lower()
        if _ct_env in ("", "auto", "0"):
            self._chunk_tokens = 512  # provisional until _autosize_chunk_tokens
            self._chunk_tokens_auto = True
        else:
            try:
                self._chunk_tokens = max(1, int(_ct_env))
            except ValueError:
                self._chunk_tokens = 512
            self._chunk_tokens_auto = False
        try:
            self._chunk_target_bytes = (
                max(1, int(os.environ.get("INFERA_KVD_CHUNK_TARGET_MIB", "128"))) << 20
            )
        except ValueError:
            self._chunk_target_bytes = 128 << 20

        # Per-cache-group KV shape metadata. Initialized empty here so
        # the scheduler-side `build_connector_meta` can read it without
        # an AttributeError before the worker's `register_kv_caches`
        # fills it from real tensor shapes. We bootstrap with shape
        # info derivable from `kv_cache_config` alone (block_size,
        # num_blocks, layer_names) — enough for the scheduler to plan
        # chunk boundaries. The worker augments with `hidden_dim` and
        # `dtype` at register_kv_caches time, which the flush path
        # needs.
        self._group_kv_spec: dict[int, dict[str, object]] = {}
        self._bootstrap_group_kv_spec_from_config(kv_cache_config)

        # ----- Scheduler-side state for content-hashed save/load keys -----
        # Filled by `get_num_new_matched_tokens` (which sees the Request
        # with its block_hashes), drained by `build_connector_meta`
        # (which sees NewRequestData with allocated block_ids).
        # req_id → list of 8-byte content-derived kvd keys, one per block
        # of the prompt in order.
        self._pending_block_hashes: dict[str, list[bytes]] = {}
        # req_id → number of leading blocks the connector promised to
        # serve from kvd (i.e. external cache hit count). Drives the
        # split between "load from kvd" and "save to kvd" in build_meta.
        self._pending_external_blocks: dict[str, int] = {}
        # req_id → in-flight prefill save progress, for CHUNKED PREFILL.
        # A long prompt prefills over several scheduler steps (8192 tokens
        # each at the default max_num_batched_tokens). The request only
        # appears in `scheduled_new_reqs` on its FIRST step; later steps
        # show up in `scheduled_cached_reqs` with `new_block_ids`. Without
        # tracking those, only the first step's KV ever gets saved (a 52k
        # prompt = 7 steps → ~1/7 coverage). We accumulate the block table
        # across steps and re-emit save chunks each step (content-key
        # dedup in `_emit_v2_chunks` skips the chunks already saved). Each
        # value: {"block_ids_groups": list[list[int]], "block_hashes":
        # list[bytes], "retention": str, "n_load": int(pages)}.
        self._req_save_state: dict[str, dict[str, Any]] = {}
        # Dedupe by CONTENT key — two requests with the same prompt share
        # the same content hashes, so we only need to SET each once.
        # LRU-bounded by `_max_saved_content_keys` to prevent unbounded
        # growth in long-running scheduler processes (PR #9 review fix D).
        # Each entry stores the monotonic timestamp of the SET emission so
        # we can re-emit if the async write failed silently (TTL-based
        # self-healing; the scheduler can't see worker save outcomes
        # cross-process, so optimistic-mark + TTL re-emit is the simplest
        # equivalent of "mark only on success").
        self._saved_content_keys: OrderedDict[bytes, float] = OrderedDict()
        # 1M entries × 40 bytes/(key+ts) ≈ 40 MB RAM cap.
        self._max_saved_content_keys = int(os.environ.get("INFERA_VLLM_SAVED_KEYS_CAP", "1000000"))
        # TTL after which a content key is considered "stale" and
        # re-emitted. Default <0 (disabled — LRU-only). Rationale:
        # content-addressed keys are stable across time and processes;
        # once a chunk is saved its (hash → bytes) mapping doesn't
        # change. The LRU cap (1M keys ~40 MB) is plenty for production
        # working sets, so legitimate cache entries don't get evicted
        # prematurely. With TTL>0 the dedupe expires every N seconds
        # and re-emits — paying full save tax (GPU gather + D2H +
        # executor submit + disk write) for chunks already on disk.
        # Measured cost on the MI355X testbed (Kimi K2.5) c=32 with TTL=30s and
        # iter wall ~60s: every iter re-saved all chunks (6× slowdown
        # vs vanilla vLLM).
        #
        # Failure-recovery trade-off: with TTL disabled, a silently-
        # failed async save leaves its content key in dedupe forever
        # (until LRU evicts or process restart) → next request misses
        # kvd permanently for that chunk. Healthy systems see this
        # rarely (disk full, kvd crash). Operators recover via restart.
        # Set INFERA_VLLM_SAVED_KEYS_TTL_S=30 to opt into the prior
        # TTL self-healing for environments with unreliable storage.
        self._saved_content_ttl_s = float(os.environ.get("INFERA_VLLM_SAVED_KEYS_TTL_S", "-1"))
        # Fix A — async load (env-gated, default off). Returns is_async=True
        # from get_num_new_matched_tokens so vLLM parks the request in
        # WAITING_FOR_REMOTE_KVS instead of blocking the engine step on the
        # (serial) L3 load that pins Running:1 at the cliff. In vLLM 0.19.1
        # async-loading reqs are NOT placed in scheduled_new_reqs, so the
        # load is driven from update_state_after_alloc (which has the
        # allocated blocks) -> stashed in _async_pending_loads -> emitted in
        # build_connector_meta -> kicked non-blocking in start_load_kv ->
        # completion reported via get_finished (NIXL-style). Measured to NOT
        # help on this NFS path (storage-BW bound); kept for completeness.
        self._async_load = os.environ.get("INFERA_KVD_ASYNC_LOAD", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._async_pending_loads: dict[str, list[Any]] = {}
        self._async_inflight: dict[str, list[Any]] = {}
        self._async_inflight_lock = threading.Lock()
        self._closed = False

        # ----- File-tier write roots -----
        # `hipfile_roots[retention]` = directory under which v2 chunk
        # files are written. When `hipfile_roots` is unset, the
        # connector falls back to UDS Set so chunks still reach kvd
        # (RAM tier, no cross-restart persistence).
        self._hipfile_roots = self._resolve_hipfile_roots(hipfile_roots)

        # ----- L3 file-tier reaper (issue #55) -----
        # Connector-owned hipFile chunk files have no daemon-side LRU
        # (the daemon's `--long-bytes` budget covers only RAM-tier
        # entries it owns). Without enforcement, sweeps would grow the
        # L3 dir past the configured budget and fill the underlying FS
        # — observed: NFS hit 100 % full and throughput collapsed.
        # The reaper owns volume accounting + free-space-keyed
        # eviction. Knobs (all opt-in; reaper is a no-op when no
        # hipfile_roots are configured):
        #   INFERA_KVD_L3_BUDGET_BYTES   total file-volume cap
        #                                  (0 = no budget cap; default
        #                                  inherits `--long-bytes` if
        #                                  exposed via env, else 0).
        #   INFERA_KVD_L3_FREE_FLOOR     fraction of FS that must
        #                                  stay free (default 0.05).
        #   INFERA_KVD_L3_REAP_INTERVAL  reaper tick period in
        #                                  seconds (default 30; 0 to
        #                                  disable the bg thread —
        #                                  ENOSPC backstop still fires).
        self._l3_reaper: L3FileReaper | None = None
        if self._hipfile_roots:
            # Budget priority:
            #   1. INFERA_KVD_L3_BUDGET_BYTES  (connector-specific)
            #   2. INFERA_KVD_LONG_BYTES       (the same env the kvd
            #      daemon's `--long-bytes` already reads — the recipe
            #      sets this, so the connector inherits the operator's
            #      intent without an extra knob).
            #   3. 0 (no budget cap; free-space-keyed eviction still runs)
            try:
                declared = int(os.environ.get("INFERA_KVD_L3_BUDGET_BYTES", "0"))
            except ValueError:
                declared = 0
            if declared <= 0:
                try:
                    declared = int(os.environ.get("INFERA_KVD_LONG_BYTES", "0"))
                except ValueError:
                    declared = 0
            try:
                floor = float(os.environ.get("INFERA_KVD_L3_FREE_FLOOR", "0.05"))
            except ValueError:
                floor = 0.05
            try:
                interval = float(os.environ.get("INFERA_KVD_L3_REAP_INTERVAL", "30"))
            except ValueError:
                interval = 30.0
            effective, warnings = startup_budget_clamp(
                self._hipfile_roots.values(),
                declared_budget_bytes=declared,
                free_floor_ratio=floor,
            )
            for w in warnings:
                logger.warning("l3 reaper startup: %s", w)
            if declared > 0 and effective < declared:
                logger.warning(
                    "l3 reaper: declared budget %d clamped to %d (smallest-root free * 0.9)",
                    declared,
                    effective,
                )
            self._l3_reaper = L3FileReaper(
                roots=self._hipfile_roots,
                budget_bytes=effective,
                free_floor_ratio=floor,
                interval_s=interval,
            )
            scanned = self._l3_reaper.scan_existing()
            if scanned:
                logger.info(
                    "l3 reaper: scanned %d pre-existing chunk files (used=%d bytes)",
                    scanned,
                    self._l3_reaper.snapshot()["used_bytes"],
                )
            self._l3_reaper.start()

        # ----- GPU-direct (hipFile) opt-in -----
        # When True, chunk save/load uses hipFile (GPU↔storage DMA via
        # cuFile/hipFile) instead of `.cpu()` + POSIX. Per the
        # `bench/kvcache/hipfile/bench0_sanity.py` measurement on the MI355X testbed
        # (2026-06-02): on Vast NFS, hipFile beats POSIX by 1.5-5×
        # because the Vast client integrates with GPU-direct RDMA; on
        # local ext4, hipFile ties POSIX at large block sizes and
        # loses at small. Default AUTO — enable hipFile GPU-direct iff
        # ais-check reports kernel P2PDMA support on this host; otherwise
        # fall back to POSIX (the safe path on no-P2PDMA hosts). Operator
        # can force with INFERA_KVD_GPU_DIRECT=1/0/on/off (env) or
        # `gpu_direct=True` (kwarg).
        if gpu_direct is None:
            env = _read_ais_env("auto").strip().lower()
            if env in ("auto", ""):
                self._gpu_direct = self._detect_p2pdma_support()
            else:
                self._gpu_direct = env in ("1", "true", "yes", "on")
        else:
            self._gpu_direct = bool(gpu_direct)
        if self._gpu_direct:
            # Lazy import + binding sanity check. If the hipFile shim
            # isn't importable (no binding installed on this host) we
            # fall back to POSIX silently so the connector still
            # works — same opt-in-never-crash policy as the SGLang
            # adapter.
            try:
                from infera.engine.sglang import hipfile_shim

                if not hipfile_shim.is_available():
                    logger.warning(
                        "gpu_direct=True but hipfile shim is_available()=False "
                        "— falling back to POSIX for chunk save/load"
                    )
                    self._gpu_direct = False
            except ImportError as exc:
                logger.warning(
                    "gpu_direct=True but hipfile_shim import failed (%s) "
                    "— falling back to POSIX for chunk save/load",
                    exc,
                )
                self._gpu_direct = False
        # Two modes only: GPU-direct requires kernel P2PDMA too (not just the
        # hipFile binding). Without P2PDMA, hipFile.read silently CPU-bounces
        # through a thread-unsafe path — we refuse that and use POSIX instead.
        # This makes self._gpu_direct the single source of truth:
        # True  <=> hipFile DMA is available AND safe (multi-worker cuFile),
        # False <=> POSIX mode (multi-worker mmap+H2D). No third fallback mode.
        if self._gpu_direct and not self._detect_p2pdma_support():
            logger.info(
                "gpu_direct requested but kernel P2PDMA is unavailable "
                "(ais-check would report amdgpu/P2PDMA False) — using POSIX "
                "mode (multi-worker mmap+H2D), not the hipFile CPU-bounce "
                "fallback."
            )
            self._gpu_direct = False
        if self._gpu_direct:
            logger.info(
                "chunked-fusion gpu_direct ENABLED (chunks written via "
                "hipFileWrite, read via hipFileRead); NFS GPU-direct DMA "
                "enabled via HIPFILE_UNSUPPORTED_FILE_SYSTEMS (set at import)"
            )

        # SAVE-side GPU-direct override. On boxes where hipFile/cuFile write
        # is not true P2P (CPU-bounce fallback), the GPU-direct WRITE path
        # (~3.4 GiB/s) is SLOWER than an explicit D2H->pinned->O_DIRECT-pwrite
        # bounce (~5 GiB/s); reads are unaffected (GPU-direct read stays
        # competitive). INFERA_KVD_SAVE_BOUNCE=1 forces SAVE onto the POSIX
        # bounce path while LOAD keeps GPU-direct. One-flag, fully reversible.
        _save_bounce = os.environ.get("INFERA_KVD_SAVE_BOUNCE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._save_gpu_direct = self._gpu_direct and not _save_bounce
        if self._gpu_direct and _save_bounce:
            logger.info(
                "chunked-fusion SAVE bounce ENABLED (INFERA_KVD_SAVE_BOUNCE): "
                "save via D2H+POSIX, load still GPU-direct"
            )

        self._start_background_loop()
        self._connect_or_raise()

        # Background save pool. Each chunk save splits into:
        #   1. SYNC phase (in `wait_for_save`): Triton gather + D2H to a
        #      CPU bytes object. Must happen while vLLM still owns the
        #      source paged-KV blocks.
        #   2. ASYNC phase (in this executor): POSIX write + atomic rename
        #      + UDS `register_file_entry`. Runs off the forward-pass
        #      critical path so wait_for_save returns immediately after
        #      D2H instead of blocking on disk.
        #
        # Pool size is set to 8 by default — empirically more workers
        # don't help (see project_lmcache_mi300x_agentic_bench_2026_05
        # for the 1.28× ceiling on threaded sync writes). Override with
        # INFERA_KVD_SAVE_WORKERS for benchmarking.
        # Default "auto" → L3 mount nconnect (fallback 16); unified with load.
        save_workers = self._resolve_io_workers("INFERA_KVD_SAVE_WORKERS")
        logger.info(
            "kvd I/O workers: L3 mount nconnect=%s → save=%d (auto unless "
            "INFERA_KVD_SAVE_WORKERS set)",
            self._detect_l3_nconnect(),
            save_workers,
        )
        self._save_executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=save_workers,
                thread_name_prefix="kvd-save",
            )
        )
        self._pending_save_futures: list[concurrent.futures.Future] = []
        self._pending_save_futures_lock = threading.Lock()
        # Race fix: drain in-flight saves at start_load_kv with this
        # bounded timeout (seconds). 0 disables (legacy fire-and-forget,
        # races load↔save under high c). 0.2 s covers UDS-Set / pinned
        # disk writes for chunks up to ~64 MB on MI355X-class hardware
        # without ever stalling the forward pass.
        try:
            self._load_drain_timeout_s = float(os.environ.get("INFERA_KVD_LOAD_DRAIN_S", "0.2"))
        except ValueError:
            self._load_drain_timeout_s = 0.2

        # In-flight + ring chunk cache (worker-side, host RAM):
        #   - `_in_flight_chunks` holds chunks whose async-save future
        #     hasn't fired yet — load can serve from the same host blob
        #     we'd write to disk, skipping kvd RPC + disk IO entirely.
        #   - `_chunk_ring` is a small LRU of recently-completed saves;
        #     the host bytes are already paged in, so re-reading the
        #     same chunk within seconds avoids the disk round trip too.
        # Together: load→save race window is closed without paying any
        # drain latency for the chunks actually in flight, AND repeated
        # reads of hot chunks shortcut the kvd lookup. Disable by
        # setting INFERA_KVD_CHUNK_RING_MAX_BYTES=0 (in-flight cache
        # also disabled; load reverts to kvd-only path).
        self._chunk_ring_max_bytes = self._parse_byte_size(
            os.environ.get("INFERA_KVD_CHUNK_RING_MAX_BYTES", "2G"),
            default=2 * 1024**3,
        )
        self._in_flight_chunks: dict[bytes, bytes] = {}
        self._chunk_ring: OrderedDict[bytes, bytes] = OrderedDict()
        self._chunk_ring_cur_bytes = 0
        self._chunk_cache_lock = threading.Lock()
        # Stats — only updated under lock; surface via __repr__ / logs
        # if a probe needs them.
        self._chunk_cache_inflight_hits = 0
        self._chunk_cache_ring_hits = 0
        self._chunk_cache_misses = 0

        # C3 fix: crash-safe save publish. When set, fdatasync the written
        # file and fsync its parent directory AFTER os.replace. Without
        # this, a power loss between the data write and the FS journal
        # commit can leave a published file whose dir entry exists but
        # whose size/data wasn't committed → loader sees a valid header
        # with truncated/zero payload (combined with C2 silent corruption,
        # this is unsafe). Default OFF because each call costs ~5-10 ms
        # on POSIX path (page-cache flush) or ~1-2 ms on GDS path (size
        # commit only, data already P2PDMA'd to disk). Saves run in the
        # background executor so the cost is off the engine critical
        # path, but it consumes executor throughput. Opt in on
        # deployments without UPS / requiring crash recovery without
        # restart-time re-prefill.
        self._fsync_save = os.environ.get("INFERA_KVD_FSYNC_SAVE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Load executor — mirrors save executor but for the load path
        # (mode=parallel). Storage→GPU reads through ``hipFile.read()``
        # release the GIL while blocked on the kernel cuFile read, so
        # N Python threads doing per-chunk reads truly run in parallel.
        # Microbench on the MI355X testbed (non-RDMA NFS, 128 MiB blocks):
        # N=1 4.4 GB/s, N=4 24.2 GB/s, N=8 22.8 GB/s — knee at N=4-8.
        # RDMA hosts scale further to ~28 GB/s at N=8-16.
        # Load worker pool. TWO modes:
        #   - GPU-direct (self._gpu_direct == hipFile + P2PDMA): multi-worker
        #     cuFile DMA. hipFile.read releases the GIL, so N threads truly fan
        #     out (24-28 GB/s at N=4-16 on 128 MiB blocks; MI355X-testbed microbench).
        #     Default "auto" → L3 mount nconnect (fallback 16).
        #   - POSIX (everything else): SINGLE worker mmap+H2D. Multi-worker was
        #     measured NOT to help here — the reload is H2D + Triton-scatter
        #     bound, not read-transport bound (GPU-E2E 128k DeepSeek-V3.2:
        #     16-worker made it slower, not faster), so parallel reads only add
        #     contention. Keep it simple + safe at 1. (Transport speed comes
        #     from the daemon's zero-copy shared arena, not thread count.)
        # There is NO third "hipFile CPU-bounce fallback" mode: it was
        # thread-unsafe (Memory access fault by GPU on the registered buffer)
        # and is removed. A host without P2PDMA is POSIX (self._gpu_direct was
        # forced False above) — no INFERA_KVD_FORCE_PARALLEL / force-serial clamp.
        # INFERA_KVD_LOAD_WORKERS overrides the count in either mode (benching).
        if self._gpu_direct:
            load_workers = self._resolve_io_workers("INFERA_KVD_LOAD_WORKERS")
        else:
            _lw_env = os.environ.get("INFERA_KVD_LOAD_WORKERS", "").strip().lower()
            load_workers = (
                self._resolve_io_workers("INFERA_KVD_LOAD_WORKERS")
                if _lw_env not in ("", "auto")
                else 1
            )
        self._load_executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=load_workers,
                thread_name_prefix="kvd-load",
            )
        )

        # ----- L3 storage self-check (connector path; aggregated AND PD) -----
        # The daemon logs a self-check for its tablespace POSIX path, but under
        # the connector — and especially PD, where prefill/decode are separate
        # engine processes each with their own connector — the actual L3 I/O is
        # the connector's hipFile/POSIX path with ITS resolved save/load workers
        # and gpu_direct/P2PDMA decision (load may be clamped to 1). So each
        # engine logs its own self-check at boot, reflecting the config that
        # engine will really use. Gated to WORKER-role rank 0 (the rank that
        # owns chunk I/O) so we get one line per engine, not one per TP rank.
        self._maybe_run_connector_selfcheck(save_workers, load_workers)

        # Bounded device staging pool for the gpu_direct SAVE path
        # (LMCache GdsBackend pattern). The gpu_direct save holds a device
        # staging tensor alive until its async hipFileWrite drains; on a
        # large-context / high-concurrency burst those tensors pile up in
        # HBM and OOM the engine. Cap total save staging at
        # INFERA_KVD_SAVE_POOL_MB (default 2048): pre-allocate N flat
        # uint8 slots, hand them out from a free-list, and soft-degrade to
        # the host POSIX path when none is free — never an unbounded
        # torch.empty per chunk. Lazily sized on the first gpu_direct save
        # (slot bytes = first chunk payload). The pool size alone caps
        # VRAM; acquisition is NON-BLOCKING because it runs on vLLM's
        # synchronous per-step engine path (see _acquire_save_slot) — a
        # backpressure sleep there wedges the whole engine while a slow
        # VAST write drains a slot.
        try:
            self._save_pool_bytes = max(
                0, int(os.environ.get("INFERA_KVD_SAVE_POOL_MB", "2048"))
            ) * (1 << 20)
        except ValueError:
            self._save_pool_bytes = 2048 * (1 << 20)
        self._save_pool_slots: list[Any] = []  # flat uint8 device tensors
        self._save_pool_free: list[int] = []  # free slot indices
        self._save_pool_slot_bytes: int = 0
        self._save_pool_misses: int = 0  # pool-full degrade count
        self._save_pool_lock = threading.Lock()

        # Connector telemetry counters (see InferaKvdStats). Worker-side
        # save/load runs in executor threads, so guard with a lock. These
        # accumulate over a logging interval and are drained + zeroed by
        # get_kv_connector_stats(), which vLLM polls each step.
        self._stat_lock = threading.Lock()
        self._stat_counters: dict[str, int] = {k: 0 for k in _KVD_STAT_KEYS}
        # CUMULATIVE totals (never drained) for a periodic INFO summary that is
        # visible WITHOUT vLLM's stats logger. Rationale: get_kv_connector_stats
        # is suppressed by `--disable-log-stats`, and under GPU-direct/file-tier
        # the daemon `statctl` counters are ALL ZERO (the connector writes chunk
        # files directly, bypassing the daemon) — so with neither, an operator
        # has NO signal that L3 is even working. This logs one line every
        # `_L3_LOG_EVERY` saves/lookups. Set INFERA_KVD_LOG_L3=0 to silence.
        self._l3_cum: dict[str, int] = {k: 0 for k in _KVD_STAT_KEYS}
        # First save AND first lookup log immediately (instant "L3 is working"
        # confirmation), then every _L3_LOG_EVERY after.
        self._l3_log_at: dict[str, int] = {"saved_chunks": 1, "lookup_requests": 1}
        self._l3_log_enabled = os.environ.get("INFERA_KVD_LOG_L3", "1").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

        # Pinned-HOST staging pool for the POSIX save path — the symmetric
        # host-side analogue of the device pool above. The POSIX path D2Hs
        # into a pinned slot then writes the chunk straight from it
        # (os.writev, zero-copy). Pooling removes the per-chunk
        # hipHostMalloc (cudaHostAlloc) that otherwise dominates the
        # synchronous wait_for_save phase and wedges the engine under save
        # pressure. Sized like the device pool.
        self._pin_pool_slots: list[Any] = []  # flat pinned uint8 tensors
        self._pin_pool_free: list[int] = []
        self._pin_pool_slot_bytes: int = 0
        self._pin_pool_misses: int = 0
        self._pin_pool_lock = threading.Lock()

        # Layerwise load mode:
        #   "off" (default): start_load_kv runs _load_chunk_packed inline
        #     for every chunk → wait_for_layer_load is a no-op. Highest
        #     overhead on the load-side critical path; matches v2.0 ship.
        #   "stepped": start_load_kv prepares state (fetch + decode + CPU
        #     staging once per chunk), wait_for_layer_load does per-layer
        #     Triton scatter. Saves ~50-100 ms of launch overhead + per-
        #     chunk cuda.synchronize. No H2D / attention overlap.
        #   "prefetch": adds per-layer H2D on a dedicated copy stream
        #     with prefetch_depth lookahead so layer N+depth's H2D is in
        #     flight while layer N attention runs. Closes the c=24 cliff.
        #   "parallel": submit one _load_chunk_packed per chunk to the
        #     load executor pool — N concurrent threads doing blocking
        #     hipFile.read() each. GIL-released kernel read lets reads
        #     truly fan out. Microbench: N=4 reaches 24 GB/s on the MI355X
        #     testbed NFS vs 4.4 GB/s single-stream (5.5x). RDMA hosts push
        #     to ~28 GB/s at N=8-16. Use this when storage→GPU read
        #     bandwidth dominates (large chunks, big working sets).
        #
        # Legacy single-letter aliases "a" / "b" are accepted to avoid
        # breaking existing launcher scripts and bench harnesses; new
        # config should use the descriptive names.
        layerwise_raw = os.environ.get("INFERA_KVD_LAYERWISE_LOAD", "parallel").strip().lower()
        _LAYERWISE_ALIASES = {"a": "stepped", "b": "prefetch"}
        layerwise_mode = _LAYERWISE_ALIASES.get(layerwise_raw, layerwise_raw)
        if layerwise_mode not in ("off", "stepped", "prefetch", "parallel"):
            logger.warning(
                "INFERA_KVD_LAYERWISE_LOAD=%r not in {off,stepped,prefetch,parallel} — defaulting to off",
                layerwise_raw,
            )
            layerwise_mode = "off"
        self._layerwise_mode = layerwise_mode
        # Per-step state for layerwise paths (a and b); list of
        # _SteppedChunkLoadState / _PrefetchChunkLoadState dataclasses. Cleared at
        # the top of each start_load_kv.
        self._inflight_load_state: list[Any] = []

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _stat_inc(self, key: str, n: int = 1) -> None:
        """Thread-safe increment of an interval counter (no-op for an
        unknown key so callers can't typo a metric into existence)."""
        if n == 0 or key not in self._stat_counters:
            return
        _snap: dict[str, int] | None = None
        with self._stat_lock:
            self._stat_counters[key] += n
            self._l3_cum[key] += n
            # Periodic INFO summary on save/lookup milestones (see _L3_LOG_EVERY).
            if (
                self._l3_log_enabled
                and key in self._l3_log_at
                and self._l3_cum[key] >= self._l3_log_at[key]
            ):
                self._l3_log_at[key] = self._l3_cum[key] + _L3_LOG_EVERY
                _snap = dict(self._l3_cum)
        if _snap is not None:
            _lc = _snap["lookup_chunks"]
            _hr = (100.0 * _snap["hit_chunks"] / _lc) if _lc else 0.0
            logger.info(
                "kvd L3 activity: saved=%d chunks/%.1f MiB, loaded=%d chunks/%.1f MiB, "
                "lookups=%d probed=%d hit=%d (%.1f%%), load_errors=%d [role=%s; under "
                "GPU-direct/file-tier the daemon statctl is all-zero by design]",
                _snap["saved_chunks"],
                _snap["saved_bytes"] / (1 << 20),
                _snap["loaded_chunks"],
                _snap["loaded_bytes"] / (1 << 20),
                _snap["lookup_requests"],
                _lc,
                _snap["hit_chunks"],
                _hr,
                _snap["load_errors"],
                getattr(self, "_role", "?"),
            )

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Drain this connector's interval counters into a
        ``InferaKvdStats`` and zero them. vLLM polls this each step on
        BOTH the scheduler and worker connectors and aggregates the two,
        so scheduler-side L3 hit/miss and worker-side save/load merge into
        one record. Returns None when nothing happened this interval, so
        we don't spam empty stats."""
        with self._stat_lock:
            if all(v == 0 for v in self._stat_counters.values()):
                return None
            snapshot = dict(self._stat_counters)
            for k in self._stat_counters:
                self._stat_counters[k] = 0
        return InferaKvdStats(data=snapshot)

    @classmethod
    def build_kv_connector_stats(cls, data: dict | None = None) -> KVConnectorStats:
        """Reconstruct a stats object from the serialized ``data`` dict
        (the convert step vLLM's logger calls before aggregate/reduce)."""
        return InferaKvdStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ):
        """Register this connector with vLLM's per-connector Prometheus
        metrics. Required whenever ``get_kv_connector_stats`` can return
        non-None: MultiKVConnectorPromMetrics.observe() asserts that every
        stats-emitting connector is registered here, so without this the
        engine crashes as soon as metrics are enabled (no ``--disable-log-
        stats``). See ``InferaKvdPromMetrics``."""
        return InferaKvdPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    # ------------------------------------------------------------------
    # Scheduler-side methods
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """vLLM asks: "of the request's tokens beyond what's already
        in my local prefix cache (num_computed_tokens), how many MORE
        can I get from your external cache?"

        v2 saves at chunk granularity — the kvd key is
        ``sha256(block_hashes[start..start+N])[:7] + bytes([gid])`` per
        cache group, never the raw block hash. So the exists probe
        must also walk chunk-aligned groups of N pages and derive the
        SAME key the save path uses. Pivot on gid=0; saves are
        lockstep across cache groups so a hit on gid=0 implies hits on
        every gid.

        Two side-effects beyond returning the hit count:

        1. **Stash the full prompt's content-derived keys** under
           ``_pending_block_hashes[req_id]``. ``build_connector_meta``
           drains this to pair (block_id, key) for both load and save
           metadata.

        2. **Record the promised external-block count (in pages)**
           under ``_pending_external_blocks[req_id]`` so
           ``build_connector_meta`` knows how many chunks to mark
           load-from-kvd.

        Returns (num_external_tokens, is_async). Async loading isn't
        implemented yet → second element is False.
        """
        req_id = self._req_id_of(request)
        full_hashes = self._extract_block_hashes_after(request, 0)
        if req_id is not None:
            # Stash hashes FIRST so the SAVE path (build_connector_meta) can
            # still plan attention-only chunk writes even on hybrid models.
            self._pending_block_hashes[req_id] = full_hashes

        # Hybrid models (Mamba / GDN linear-attention + attention): vLLM's
        # scheduler asserts external-computed-tokens == 0 (it has not verified
        # external connectors against its Mamba-block-aligned split). Reporting
        # a hit here crashes the engine on the first L3 reuse, and the recurrent
        # state can't be externally restored anyway. So gate the LOAD only:
        # report no external match (no load chunks planned -> no crash) while
        # attention-only saves above continue.
        if self._has_non_paged_groups:
            if not self._warned_hybrid_no_load:
                logger.warning(
                    "kvd: hybrid (Mamba/linear-attention) model detected — "
                    "external L3 LOAD disabled (vLLM does not support external "
                    "KV-connector loads for hybrid models); attention-only "
                    "saves continue but will not be reused"
                )
                self._warned_hybrid_no_load = True
            return 0, False

        block_size = self._block_size_now(request)
        if block_size <= 0:
            return 0, False

        # Chunk granularity = N pages per chunk. Must match the
        # build_connector_meta save loop.
        N = max(1, self._chunk_tokens // block_size)
        n_pages = len(full_hashes)
        n_chunks_total = n_pages // N
        if n_chunks_total == 0:
            return 0, False

        # Skip chunks entirely covered by num_computed_tokens (already
        # in vLLM's local prefix cache). Partial-overlap chunks at the
        # boundary stay probed — vLLM clamps in update_state_after_alloc.
        skip_blocks = num_computed_tokens // block_size
        skip_chunks = skip_blocks // N
        if skip_chunks >= n_chunks_total:
            return 0, False

        # Pivot gid: the lowest cache_group_id present in this
        # connector. Saves iterate sorted(group_ids) so the pivot
        # matches whatever save wrote. For single-group models
        # (gpt-oss-120b, Qwen) this is gid=0; for HMA models with
        # gid={0, 1, 2} it's still 0. Falls back to 0 when
        # `_group_kv_spec` hasn't been populated yet — vLLM's
        # scheduler creates the connector before the worker calls
        # `register_kv_caches`, and `kv_cache_config` may be None
        # at scheduler-init time. gid=0 is the right guess for
        # single-group models; multi-group models the worker
        # processes before the scheduler issues its first probe.
        pivot_gid = min(self._group_kv_spec.keys()) if self._group_kv_spec else 0

        chunk_keys: list[bytes] = []
        for chunk_idx in range(skip_chunks, n_chunks_total):
            ck = _chunk_kvd_key_for(full_hashes, chunk_idx * N, N)
            chunk_keys.append(ck[:7] + bytes([pivot_gid & 0xFF]))

        if self._hipfile_roots:
            # File-tier (#46 / LMCache GdsBackend design): the connector
            # owns the on-disk chunk files and the path is derived purely
            # from the content key, so existence is a local stat under the
            # configured roots — no daemon `exists` RPC, no daemon index.
            present = [self._local_chunk_path(ck) is not None for ck in chunk_keys]
        else:
            present = self._run_async(
                self._client.exists(chunk_keys, model=self._model, compat_key=self._compat_key)
            )
        num_hit_chunks = 0
        for p in present:
            if not p:
                break
            num_hit_chunks += 1

        # Telemetry: prefix-aligned L3 probe outcome for this request.
        self._stat_inc("lookup_requests", 1)
        self._stat_inc("lookup_chunks", len(chunk_keys))
        self._stat_inc("hit_chunks", num_hit_chunks)

        logger.debug(
            "get_num_new_matched_tokens: req=%s n_chunks=%d skip_chunks=%d "
            "probe=%d hits=%d pivot_gid=%d N=%d block_size=%d",
            req_id,
            n_chunks_total,
            skip_chunks,
            len(chunk_keys),
            num_hit_chunks,
            pivot_gid,
            N,
            block_size,
        )

        if num_hit_chunks == 0:
            if req_id is not None:
                self._pending_external_blocks[req_id] = 0
            return 0, False

        # Total tokens covered from the start of the prompt:
        # (skip_chunks + num_hit_chunks) full chunks. Subtract what
        # vLLM already has locally to get the "new" count.
        total_tokens_covered = (skip_chunks + num_hit_chunks) * N * block_size
        new_tokens = max(0, total_tokens_covered - num_computed_tokens)
        if req_id is not None:
            # _pending_external_blocks is in PAGES (build_connector_meta
            # converts page count → chunks via // N).
            self._pending_external_blocks[req_id] = new_tokens // block_size
        # Fix A: signal async load so vLLM parks the request instead of
        # blocking the engine step on the L3 read (default off).
        return new_tokens, (self._async_load and new_tokens > 0)

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        """vLLM has allocated the blocks for the load we promised. We
        update ``_pending_external_blocks`` here in case vLLM clamped
        num_external_tokens below what we returned from
        ``get_num_new_matched_tokens`` (it does this when allocated
        block count is less than what we asked to load).

        The (block_id ↔ block_hash) pairing happens in
        ``build_connector_meta``, which sees the final scheduled
        NewRequestData with its block_ids.
        """
        req_id = self._req_id_of(request)
        if req_id is None:
            return
        block_size = self._block_size_now(request)
        actual_external_blocks = num_external_tokens // max(block_size, 1)
        # Take the min — vLLM never gives us MORE blocks than promised;
        # but a misbehaving caller passing a larger number shouldn't
        # cause us to over-load.
        promised = self._pending_external_blocks.get(req_id, actual_external_blocks)
        self._pending_external_blocks[req_id] = min(promised, actual_external_blocks)

        # Fix A: async-loading reqs never reach build_connector_meta's
        # new-reqs loop, so build their load entries HERE (we have the
        # allocated blocks) and stash for build_connector_meta to emit.
        if self._async_load:
            n_load = self._pending_external_blocks.get(req_id, 0)
            block_hashes = self._pending_block_hashes.get(req_id, [])
            if n_load > 0 and block_hashes:
                try:
                    groups = self._normalize_block_id_groups(blocks.get_block_ids())
                except Exception:
                    groups = ()
                if groups and groups[0]:
                    n_pairs = min(min(len(g) for g in groups), len(block_hashes))
                    entries = self._build_async_load_entries(
                        block_hashes, tuple(groups), n_pairs, min(n_load, n_pairs)
                    )
                    if entries:
                        self._async_pending_loads[req_id] = entries
                        # Zero the pending external count so the later
                        # new-reqs save loop does NOT re-emit the load.
                        self._pending_external_blocks[req_id] = 0

    @staticmethod
    def _normalize_block_id_groups(raw: Any) -> tuple[list[int], ...]:
        """Normalize a vLLM block_ids value to per-cache-group form:
        ``tuple[list[int], ...]`` — one list per KV cache group. vLLM v1
        uses this shape; older flat ``list[int]`` becomes a 1-tuple."""
        raw = raw or ()
        if raw and not isinstance(raw[0], (list, tuple)):
            return (list(raw),)
        return tuple(list(g) for g in raw)

    def _build_async_load_entries(
        self,
        block_hashes: list[bytes],
        block_ids_groups: tuple[list[int], ...],
        n_pairs: int,
        n_load: int,
    ) -> list[tuple]:
        """Fix A: build ONLY the leading n_load-page load entries (no save,
        no dedup side effects), mirroring the load branch of _emit_v2_chunks.
        The async path emits loads from update_state_after_alloc because in
        vLLM 0.19.1 async-loading reqs never reach build_connector_meta's
        new-reqs loop. Loads start at chunk 0 (at the cliff, where async
        matters, vLLM's local L1 coverage is 0)."""
        if not self._group_kv_spec:
            return []
        first_group = next(iter(self._group_kv_spec.values()))
        block_size = int(first_group["block_size"])
        if block_size <= 0:
            return []
        N = max(1, self._chunk_tokens // block_size)
        if N <= 0 or N > n_pairs:
            return []
        n_chunks_total = n_pairs // N
        n_load_chunks = min(n_load // N, n_chunks_total)
        if n_load_chunks <= 0:
            return []
        group_ids = sorted(self._group_kv_spec.keys())
        entries: list[tuple] = []
        for chunk_idx in range(n_load_chunks):
            start_page = chunk_idx * N
            chunk_key = _chunk_kvd_key_for(block_hashes, start_page, N)
            per_page_block_ids = tuple(
                tuple(g[start_page + p] for g in block_ids_groups) for p in range(N)
            )
            for gid in group_ids:
                group_chunk_key = chunk_key[:7] + bytes([gid & 0xFF])
                layer_names = list(self._group_kv_spec[gid]["layer_names"])  # type: ignore[arg-type]
                entries.append((per_page_block_ids, group_chunk_key, gid, layer_names))
        return entries

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        """Per-step metadata flowing scheduler → worker. v2 chunked-fusion:
        the request's pages are walked in groups of N (= ``chunk_tokens /
        block_size``); each FULL group emits one ``packed_chunks_to_save``
        (or _load) entry per cache_group. Partial tail (< N pages) is
        dropped per the strict-chunk-aligned policy.

        CHUNKED PREFILL: a long prompt is prefilled over several scheduler
        steps. The request appears in ``scheduled_new_reqs`` only on its
        first step (with the first step's allocated blocks); subsequent
        steps appear in ``scheduled_cached_reqs`` with ``new_block_ids``.
        We must emit save chunks on EVERY step, otherwise only the first
        step's KV (≈1/7 of a 52k prompt) is ever persisted. We accumulate
        the block table per request in ``_req_save_state`` and re-emit each
        step; ``_emit_v2_chunks``'s content-key dedup skips chunks already
        saved, so only the newly-completed chunks actually write.

        If ``get_num_new_matched_tokens`` wasn't called (rare; e.g. the
        request is fully covered by vLLM's local prefix cache) the new
        request is skipped — v2 needs content hashes for chunk_key
        derivation and there's no synthetic-key path.
        """
        meta = InferaKvdConnectorMetadata()
        # Fix A: emit async loads stashed by update_state_after_alloc (async
        # reqs aren't in scheduled_new_reqs, so this is their only path in).
        if self._async_load and self._async_pending_loads:
            for rid, entries in self._async_pending_loads.items():
                for e in entries:
                    meta.packed_chunks_to_load.append(e)
                    meta.load_chunk_req_ids.append(rid)
            self._async_pending_loads = {}
        if not self._group_kv_spec:
            return meta

        # ---- New requests: first prefill step. Seed save state. --------
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or []
        for new_req in new_reqs:
            req_id = getattr(new_req, "req_id", None)
            if req_id is None:
                continue
            # Per-request retention: read from `kv_transfer_params` the
            # router stamped onto the request body. Fall back to "long".
            retention = self._extract_retention(new_req)
            block_hashes = self._pending_block_hashes.pop(req_id, [])
            num_external_blocks = self._pending_external_blocks.pop(req_id, 0)
            block_ids_groups = self._normalize_block_id_groups(getattr(new_req, "block_ids", ()))
            if not block_ids_groups or not block_ids_groups[0]:
                continue
            if not block_hashes:
                logger.warning(
                    "build_connector_meta: req %s has no block_hashes "
                    "(get_num_new_matched_tokens not called?); v2 needs "
                    "content hashes for chunk_key derivation — skipping",
                    req_id,
                )
                continue
            # vLLM-L1 already-covered prefix: chunks fully below this
            # offset are skipped at emit time (no save — vLLM has the KV
            # in L1; no load — vLLM serves from L1 directly). Without
            # this gate we emit 117 redundant save chunks per warm-cache
            # request, which costs ~50 s per c=16 admission step on
            # the MI355X testbed (Kimi K2.5, 15× slower than vanilla vLLM at low c).
            # Cliff-region win is preserved: when L1 evicts under high
            # c, num_computed_tokens drops on the next admission, the
            # recomputed portion falls outside skip_blocks, and gets
            # saved as before.
            num_computed_tokens = int(getattr(new_req, "num_computed_tokens", 0) or 0)
            block_size_n = max(1, self._block_size_now(new_req))
            # CORRECTNESS FIX: vLLM sets num_computed_tokens = local-L1 +
            # external (the tokens it expects US to load from L3). Folding
            # the external part into skip_blocks skips emitting their LOAD
            # entries -> the allocated KV blocks are never filled -> vLLM
            # attends over garbage -> degenerate output. skip_blocks must
            # be the L1-ONLY prefix = num_computed_tokens MINUS external.
            # num_external_blocks is in whole pages, so subtract directly.
            skip_blocks = max(0, num_computed_tokens // block_size_n - int(num_external_blocks))
            # Seed accumulating save state so continuation steps can
            # extend the block table and keep saving. `skip_blocks` is
            # admission-time-fixed (the L1-covered prefix is determined
            # when the request is first scheduled), so continuation
            # steps reuse this value.
            self._req_save_state[req_id] = {
                "block_ids_groups": [list(g) for g in block_ids_groups],
                "block_hashes": block_hashes,
                "retention": retention,
                "n_load": num_external_blocks,
                "skip_blocks": skip_blocks,
            }
            self._emit_req_save_chunks(meta, req_id, emit_load=True)

        # ---- Cached requests: chunked-prefill continuation steps. ------
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            req_ids = getattr(cached, "req_ids", None) or []
            new_block_ids_list = getattr(cached, "new_block_ids", None) or []
            resumed = getattr(cached, "resumed_req_ids", None) or set()
            for i, req_id in enumerate(req_ids):
                st = self._req_save_state.get(req_id)
                if st is None:
                    continue  # not a kvd-tracked prefill, or already complete
                nb = new_block_ids_list[i] if i < len(new_block_ids_list) else None
                if nb is None:
                    continue  # no blocks allocated this step (e.g. pure decode)
                nb_groups = self._normalize_block_id_groups(nb)
                if req_id in resumed:
                    # Preempted-then-resumed: new_block_ids REPLACES the
                    # block table (vLLM reallocated). Content keys are
                    # stable, so re-emit picks up where it left off; the
                    # dedup TTL guards against redundant writes.
                    st["block_ids_groups"] = [list(g) for g in nb_groups]
                else:
                    for g_idx, g in enumerate(nb_groups):
                        if g_idx < len(st["block_ids_groups"]):
                            st["block_ids_groups"][g_idx].extend(g)
                # Continuation: load was already emitted on the first step.
                self._emit_req_save_chunks(meta, req_id, emit_load=False)
                # Once every prompt chunk is covered, drop the state.
                if self._req_save_complete(req_id):
                    self._req_save_state.pop(req_id, None)

        return meta

    def _emit_req_save_chunks(
        self,
        meta: InferaKvdConnectorMetadata,
        req_id: str,
        *,
        emit_load: bool,
    ) -> None:
        """Emit chunk metadata for ``req_id`` from its accumulated block
        table. Re-callable across chunked-prefill steps — dedup in
        ``_emit_v2_chunks`` ensures only newly-completed chunks write."""
        st = self._req_save_state.get(req_id)
        if st is None:
            return
        groups = st["block_ids_groups"]
        block_hashes = st["block_hashes"]
        if not groups or not groups[0]:
            return
        n_blocks = min(len(g) for g in groups)
        n_pairs = min(n_blocks, len(block_hashes))
        if n_pairs <= 0:
            return
        n_load = min(int(st["n_load"]), n_pairs)
        skip_blocks = int(st.get("skip_blocks", 0))
        self._emit_v2_chunks(
            meta=meta,
            block_hashes=block_hashes,
            block_ids_groups=tuple(groups),
            n_pairs=n_pairs,
            n_load=n_load,
            skip_blocks=skip_blocks,
            retention=st["retention"],
            emit_load=emit_load,
        )

    def _req_save_complete(self, req_id: str) -> bool:
        """True once the accumulated block table covers every full chunk
        of the prompt (so no further save chunks can be emitted)."""
        st = self._req_save_state.get(req_id)
        if st is None:
            return True
        n_blocks = min((len(g) for g in st["block_ids_groups"]), default=0)
        return n_blocks >= len(st["block_hashes"])

    def _remember_saved_key(self, kvd_key: bytes, ts: float | None = None) -> None:
        """Add `kvd_key` to the dedupe set and evict LRU entries until
        the set is under the cap. PR #9 review fix D: prevents the
        scheduler-side dedupe set from growing unbounded over a
        long-running connector lifetime.

        `ts` is the monotonic timestamp of the SET emission used by the
        TTL self-healing check in ``_emit_v2_chunks``. Defaults to
        ``time.monotonic()`` if not provided (legacy callers)."""
        self._saved_content_keys[kvd_key] = ts if ts is not None else time.monotonic()
        while len(self._saved_content_keys) > self._max_saved_content_keys:
            self._saved_content_keys.popitem(last=False)
            self._stat_inc("evictions", 1)

    def _emit_v2_chunks(
        self,
        *,
        meta: InferaKvdConnectorMetadata,
        block_hashes: list[bytes],
        block_ids_groups: tuple[list[int], ...],
        n_pairs: int,
        n_load: int,
        retention: str,
        emit_load: bool = True,
        skip_blocks: int = 0,
    ) -> None:
        """vLLM-8 chunked-fusion emit. Coalesces N=chunk_tokens/block_size
        consecutive pages into ONE kvd entry per (chunk, cache_group).

        Per the design doc, ONLY FULL chunks emit — partial tail
        (< N pages) is dropped (recomputed on miss).

        ``skip_blocks`` is the vLLM-L1-covered prefix at admission
        (= ``num_computed_tokens // block_size``). Chunks fully below
        this offset are NOT emitted — no save (vLLM already has the
        KV in L1, the data isn't going anywhere this step), no load
        (vLLM serves from L1 directly). The cliff-region win is
        preserved: when L1 evicts under high c, the next admission of
        the evicted prefix has ``num_computed_tokens=0`` (or low), the
        recomputed portion falls outside ``skip_blocks``, and gets
        saved into kvd as before.

        After ``skip_blocks``: LOAD entries cover the leading chunks
        already externally cached (= ``n_load`` rounded down to
        N-page boundary); SAVE entries cover the rest of the FULL
        chunks.

        For HMA models with G cache_groups: each chunk produces G
        entries (one per group), so the worker can route them through
        its per-group staging buffer.
        """
        # Pages per chunk: validated 1+ at __init__.
        # We pick block_size from group-0's KV spec (the engine's
        # native page size); all groups share block_size in vLLM v1.
        first_group = next(iter(self._group_kv_spec.values()))
        block_size = int(first_group["block_size"])
        if block_size <= 0:
            return  # defensive — no block_size known means no v2
        N = max(1, self._chunk_tokens // block_size)
        if N <= 0 or N > n_pairs:
            # Not enough pages for even one chunk — strict-aligned drops it.
            return

        n_chunks_total = n_pairs // N
        # n_load is in PAGES; convert to chunks (round down — partial
        # leading chunk goes back to "save", not "load").
        n_load_chunks = min(n_load // N, n_chunks_total)
        # skip_chunks: how many leading FULL chunks are vLLM-L1-covered.
        # A partial chunk straddling the L1/non-L1 boundary stays in
        # the emit range — only entirely-covered chunks are skipped.
        skip_chunks = min(max(0, skip_blocks // N), n_chunks_total)
        # The externally-cached (LOAD) region begins AFTER the L1-covered
        # prefix, at absolute chunk index `skip_chunks`, and spans
        # `n_load_chunks` chunks. `n_load` (= _pending_external_blocks) is
        # the count of external pages beyond the L1 prefix — NOT an absolute
        # offset — so the LOAD chunks are the ABSOLUTE indices
        # [skip_chunks, skip_chunks + n_load_chunks). Comparing `chunk_idx`
        # against the bare count `n_load_chunks` (as the code did) only
        # works when skip_chunks == 0 (full reload, local L1 = 0); with any
        # local-L1 prefix (partial reload) every L3 chunk sits at
        # chunk_idx >= skip_chunks >= n_load_chunks, so NONE were marked
        # load -> the allocated blocks were never filled -> attention over
        # stale/garbage KV. The earlier skip_blocks fix was necessary but
        # INCOMPLETE: the is_load boundary was still relative.
        load_chunk_end = skip_chunks + n_load_chunks

        group_ids = sorted(self._group_kv_spec.keys())

        # Iterate chunks; emit one entry per (chunk, group). Skip the
        # L1-covered prefix entirely — vLLM doesn't need us to load or
        # save those.
        for chunk_idx in range(skip_chunks, n_chunks_total):
            start_page = chunk_idx * N
            # Per-chunk content-addressed key (stable across processes
            # via sha256 of constituent page hashes).
            chunk_key = _chunk_kvd_key_for(block_hashes, start_page, N)
            # Per-page per-group physical block ids: shape
            # (N pages, num_groups) — converted to nested tuple for
            # hashing-friendly immutability on the metadata.
            per_page_block_ids = tuple(
                tuple(g[start_page + p] for g in block_ids_groups) for p in range(N)
            )

            is_load = chunk_idx < load_chunk_end
            for gid in group_ids:
                # Compose group-specific chunk key by mixing gid into
                # the content hash, so two groups' chunks land in
                # different files even if their page hashes coincide.
                group_chunk_key = chunk_key[:7] + bytes([gid & 0xFF])
                layer_names = list(self._group_kv_spec[gid]["layer_names"])  # type: ignore[arg-type]
                if is_load:
                    # On chunked-prefill continuation steps the leading
                    # load chunks were already emitted on the first step;
                    # re-emitting them would re-issue the load every step.
                    if emit_load:
                        meta.packed_chunks_to_load.append(
                            (per_page_block_ids, group_chunk_key, gid, layer_names)
                        )
                else:
                    now = time.monotonic()
                    prev_ts = self._saved_content_keys.get(group_chunk_key)
                    if prev_ts is not None and (
                        self._saved_content_ttl_s < 0 or now - prev_ts < self._saved_content_ttl_s
                    ):
                        # Recently saved (or TTL disabled) — skip duplicate
                        # write. `move_to_end` refreshes LRU recency without
                        # bumping the timestamp; a stale entry will still
                        # expire even on repeat hits, forcing re-emission
                        # so a silently-failed write can self-heal.
                        self._saved_content_keys.move_to_end(group_chunk_key)
                        continue
                    self._remember_saved_key(group_chunk_key, now)
                    meta.packed_chunks_to_save.append(
                        (per_page_block_ids, group_chunk_key, retention, gid, layer_names)
                    )

    @staticmethod
    def _req_id_of(request: Any) -> str | None:
        """vLLM's Request carries the id under `request_id`; the
        NewRequestData wrapper under `req_id`. Accept either."""
        return getattr(request, "request_id", None) or getattr(request, "req_id", None)

    @staticmethod
    def _extract_retention(request_or_new_req: Any) -> str:
        """Pull retention from the request's ``kv_transfer_params``.

        Three lookup paths, most authoritative first:

        1. ``request.kv_transfer_params["infera_retention"]`` —
           where vLLM Request stores it after parsing extra_args.
        2. ``request.sampling_params.extra_args["kv_transfer_params"]
           ["infera_retention"]`` — where NewRequestData (which we
           see in build_connector_meta) carries it.
        3. Fallback to ``"long"`` so requests without explicit
           cache_control land in the persistent long-retention tier.
           Earlier versions defaulted to ``"short"`` (spillover); under
           realistic agentic workloads that meant chunks LRU-evicted
           from the spillover region before they could be reused,
           starving the external cache hit rate. See
           bench/kvcache/hipfile/results/* showing 1-3% external
           hit rate at spillover-saturation. Long-tier writes cost
           an fsync per Set but the cache-rescue payoff dominates.

        Returns one of ``"none" | "short" | "long"``.
        """
        # Path 1: Request object.
        kv_tp = getattr(request_or_new_req, "kv_transfer_params", None)
        if isinstance(kv_tp, dict):
            r = kv_tp.get("infera_retention")
            if isinstance(r, str) and r in ("none", "short", "long"):
                return r

        # Path 2: NewRequestData → sampling_params → extra_args.
        sp = getattr(request_or_new_req, "sampling_params", None)
        if sp is not None:
            extra = getattr(sp, "extra_args", None)
            if isinstance(extra, dict):
                kv_tp2 = extra.get("kv_transfer_params")
                if isinstance(kv_tp2, dict):
                    r = kv_tp2.get("infera_retention")
                    if isinstance(r, str) and r in ("none", "short", "long"):
                        return r

        return "long"

    def request_finished(
        self,
        request: Any,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request is done. If we wanted to async-save
        blocks to kvd after this point, we'd return True (keep blocks
        live for our save), then later signal completion through
        `get_finished()`. Skeleton: synchronous-save-only, return False
        immediately (blocks can be freed)."""
        self._req_save_state.pop(self._req_id_of(request) or "", None)
        return False, None

    def request_finished_all_groups(
        self,
        request: Any,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA variant of `request_finished` — invoked by vLLM's
        scheduler instead of `request_finished` whenever the connector
        is `SupportsHMA` (which we are). The default scheduler
        contract (`scheduler.py: _connector_finished`) requires HMA
        connectors implement this; non-HMA connectors are rejected
        for any model with >1 kv_cache_group (e.g. gpt-oss SWA,
        Mamba). `block_ids` is one list per cache group.

        We sync-save inside `wait_for_save` (per-step), so by the
        time a request finishes there's nothing pending for us to
        async-save. Return `(False, None)` so vLLM frees the blocks
        immediately — same semantics as `request_finished` above."""
        self._req_save_state.pop(self._req_id_of(request) or "", None)
        return False, None

    # ------------------------------------------------------------------
    # Worker-side methods
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """vLLM hands us the engine's pre-allocated KV cache tensors,
        keyed by layer name. We capture:

        1. ``self._kv_caches`` — the per-layer device tensors (used by
           the Triton scatter/gather kernel at save/load time).
        2. ``self._layer_to_group`` — layer name → kv_cache_group
           index, derived from ``self._kv_cache_config.kv_cache_groups``.
           Single-group models always map every layer to 0; HMA (SWA,
           Mamba) models split layers across groups with independent
           block tables.
        3. ``self._group_kv_spec`` — per-cache-group shape metadata
           (``num_layers``, ``num_blocks``, ``block_size``,
           ``hidden_dim``, ``dtype``) needed to allocate chunk
           staging buffers and validate ``ChunkHeader`` on load.
           Derived from the FIRST layer of each group; vLLM v1
           invariant guarantees all layers in one group share shape.

        Tensor shape convention (vLLM v1): for most attention backends
        each layer's tensor is ``[2, num_blocks, block_size, num_kv_heads,
        head_dim]`` where the leading ``2`` separates K from V. The
        chunk-format payload flattens this to ``[2, num_layers,
        chunk_tokens, hidden_dim]`` (hidden_dim = num_kv_heads ×
        head_dim) — done by the Triton kernel, not here.

        Best-effort: fake tensors in unit tests don't have ``.shape``
        / ``.dtype``; we swallow the AttributeError and leave
        ``_group_kv_spec`` empty. ``build_connector_meta`` then skips
        v2 emit for that request (logged at debug). Production vLLM
        tensors always have these attributes.
        """
        # Keep only tensor-valued caches. Hybrid models (Mamba / linear-
        # attention / short-conv, e.g. Qwen3.5 GDN) register their recurrent
        # STATE as a *list* of state tensors (conv_state, ssm_state), not a
        # single paged-attention tensor. Those groups are non-pageable and
        # must never be offloaded (see _is_non_paged_kv_spec + the group-spec
        # skips below). Dropping them here keeps every downstream consumer —
        # the gather/scatter and especially `_ensure_save_stream`'s device
        # probe (`next(iter(self._kv_caches.values())).device`) — from ever
        # grabbing a non-tensor and raising ``'list' object has no attribute
        # 'device'``. For non-hybrid models every value is a tensor → no-op.
        self._kv_caches = {k: v for k, v in kv_caches.items() if hasattr(v, "device")}
        _dropped = [k for k in kv_caches if not hasattr(kv_caches[k], "device")]
        if _dropped:
            logger.info(
                "register_kv_caches: dropped %d non-tensor (recurrent/conv "
                "state) cache entries from the offload set, e.g. %r",
                len(_dropped),
                _dropped[:3],
            )

        # Build layer → kv_cache_group index from the engine's
        # KVCacheConfig. Empty default means "single group at index 0".
        self._layer_to_group = {}
        kv_cfg = getattr(self, "_kv_cache_config", None)
        groups = getattr(kv_cfg, "kv_cache_groups", None) if kv_cfg else None
        if groups:
            for gid, group in enumerate(groups):
                for lname in getattr(group, "layer_names", ()) or ():
                    self._layer_to_group[lname] = gid

        # Per-group KV shape metadata. Skipped silently for fake
        # tensors (tests); production tensors always succeed.
        self._group_kv_spec: dict[int, dict[str, object]] = {}
        try:
            if groups:
                for gid, group in enumerate(groups):
                    spec = getattr(group, "kv_cache_spec", None)
                    if _is_non_paged_kv_spec(spec):
                        # Mamba / linear-attention / conv recurrent state:
                        # not a paged attention KV cache. Skip BEFORE the
                        # shape probe so a 3-D state tensor can never be
                        # mis-detected as an MLA latent cache.
                        logger.info(
                            "register_kv_caches: group %d (%s) is a "
                            "non-paged recurrent/conv-state group — "
                            "skipping L3 offload",
                            gid,
                            type(spec).__name__,
                        )
                        self._has_non_paged_groups = True
                        continue
                    layer_names = list(getattr(group, "layer_names", ()) or ())
                    if _is_dsa_indexer_group(layer_names):
                        # DSA sparse-attention model (deepseek_v32 / glm_moe_dsa):
                        # vLLM puts BOTH the main MLA latent (hidden 576) and the
                        # indexer.k_cache (hidden 132) in ONE mixed kv_cache_group.
                        # They have different hidden dims, so SPLIT into two uniform
                        # MLA sub-specs and offload BOTH:
                        #   main    -> this gid (its own block table)
                        #   indexer -> _DSA_INDEXER_GID_BASE+gid, aliasing group-0's
                        #              block ids (per-gid emit/load fall back to
                        #              page_ids[0] for the synthetic gid).
                        # Restoring BOTH on an external L3 hit reproduces exactly
                        # what a native L1 prefix-cache hit reuses: the sparse-attn
                        # scan reads K over the whole sequence from the cache, so
                        # the restored prefix keys make its top-k select correctly.
                        # Both are single co-packed blobs (indexer 128 fp8 + 4-byte
                        # fp32 ue8m0 scale = 132; MLA latent 576), so the raw-byte
                        # gather/scatter round-trips them byte+scale faithfully —
                        # test_dsa_indexer_roundtrip. GPU-E2E on DeepSeek-V3.2-Exp
                        # (deconfounded): reload==cold 8/8, gets>0/misses=0. Speed:
                        # sparse-attn prefill is already cheap, so at achievable
                        # context (<= max_model_len ~8k) L3 reuse is ~break-even and
                        # per-shape-autotune-noisy; the real win is at long context
                        # (the O(N^2) indexer selection dominates). See
                        # KVD_DSA_INDEXER_NOTES.md.
                        main_names, idx_names = _split_dsa_layers(layer_names)
                        bs = self._logical_block_size(group, kv_cfg)
                        self._register_mla_subspec(
                            gid, [n for n in main_names if n in self._kv_caches], bs
                        )
                        self._register_mla_subspec(
                            _DSA_INDEXER_GID_BASE + gid,
                            [n for n in idx_names if n in self._kv_caches],
                            bs,
                        )
                        logger.info(
                            "register_kv_caches: group %d DSA split-offload -> main "
                            "gid=%d (%d layers) + indexer gid=%d (%d layers)",
                            gid,
                            gid,
                            len(main_names),
                            _DSA_INDEXER_GID_BASE + gid,
                            len(idx_names),
                        )
                        # GPU-direct gotcha: the hipFile DMA fast path is gated on
                        # the per-layer byte size being 4 KiB-aligned (cuFile needs
                        # 4 KiB file offsets and layers are packed back-to-back —
                        # see _prepare_chunk_for_prefetch_load). The indexer's
                        # per-layer size is chunk_tokens * hidden_dim * itemsize
                        # (num_kv_channels=1); for the co-packed 132-byte indexer
                        # that is 4 KiB-aligned ONLY when chunk_tokens is a multiple
                        # of 1024 (gcd(132,4096)=4 -> 33*ct must divide 1024). At the
                        # default 256/512 the indexer silently falls back to the
                        # (correct but slower) mmap+H2D path while the main latent
                        # still DMAs -> DSA gets only HALF the GPU-direct win. Warn
                        # so the operator can bump INFERA_KVD_CHUNK_TOKENS.
                        if getattr(self, "_gpu_direct", False):
                            idx_spec = self._group_kv_spec.get(_DSA_INDEXER_GID_BASE + gid)
                            ct = int(getattr(self, "_chunk_tokens", 0) or 0)
                            if idx_spec and ct > 0:
                                idx_dtype = idx_spec.get("dtype")
                                itemsize = idx_dtype.itemsize if idx_dtype is not None else 1
                                idx_per_layer = ct * int(idx_spec["hidden_dim"]) * itemsize
                                if idx_per_layer & (4096 - 1):
                                    logger.warning(
                                        "register_kv_caches: gpu_direct is ON but "
                                        "indexer gid=%d per-layer bytes=%d "
                                        "(chunk_tokens=%d * hidden=%d * %dB) is NOT "
                                        "4 KiB-aligned -> indexer loads fall back to "
                                        "mmap+H2D (non-DMA); only the main latent "
                                        "gets GPU-direct. Set INFERA_KVD_CHUNK_TOKENS "
                                        "to a multiple of 1024 to DMA the indexer too.",
                                        _DSA_INDEXER_GID_BASE + gid,
                                        idx_per_layer,
                                        ct,
                                        int(idx_spec["hidden_dim"]),
                                        itemsize,
                                    )
                        continue
                    present = [n for n in layer_names if n in self._kv_caches]
                    if not present:
                        continue
                    sample = self._kv_caches[present[0]]
                    shape = tuple(sample.shape)
                    # Two tensor layouts in vLLM v1:
                    #   regular: [2, num_blocks, block_size, num_kv_heads, head_dim]
                    #            (4+ dims, leading 2 separates K/V)
                    #   MLA:     [num_blocks, block_size, hidden_dim]
                    #            (3 dims, no leading 2; combined latent)
                    if len(shape) >= 4 and shape[0] == 2:
                        num_kv_channels = 2
                        num_blocks = int(shape[1])
                        block_size_in_shape = int(shape[2])
                        hidden_dim = 1
                        for d in shape[3:]:
                            hidden_dim *= int(d)
                    elif len(shape) >= 4 and shape[1] == 2:
                        # vLLM 0.23 FullAttention INTERLEAVED layout:
                        # [num_blocks, 2, block_size, num_kv_heads, head_dim] —
                        # the size-2 K/V split is at dim 1, not dim 0. The
                        # gather/scatter kernel handles it via KV_INTERLEAVED.
                        num_kv_channels = 2
                        num_blocks = int(shape[0])
                        block_size_in_shape = int(shape[2])
                        hidden_dim = 1
                        for d in shape[3:]:
                            hidden_dim *= int(d)
                    elif len(shape) == 3:
                        num_kv_channels = 1
                        num_blocks = int(shape[0])
                        block_size_in_shape = int(shape[1])
                        hidden_dim = int(shape[2])
                    else:
                        logger.warning(
                            "register_kv_caches: group %d layer %r has "
                            "shape %s that matches neither regular "
                            "[2, num_blocks, block_size, ...] nor MLA "
                            "[num_blocks, block_size, hidden_dim] — "
                            "chunked fusion will skip this group",
                            gid,
                            present[0],
                            shape,
                        )
                        continue
                    # Defensive guard for packed/quantized KV formats. The
                    # shape probe above classifies by dims only; fp8/uint8
                    # caches CAN pack per-token scale bytes INTO the "hidden"
                    # run (fp8_ds_mla's 656/584-byte latent, per-token-head-
                    # scale fp8 attention, nvfp4), and the raw-element gather/
                    # scatter would silently mis-stride into garbage on reload.
                    # Auto-detect the provably-safe case: a plain MLA latent
                    # (num_kv_channels == 1) whose hidden EXACTLY equals
                    # kv_lora_rank + qk_rope_head_dim is a plain fp8 cast with
                    # no interleaved scale (Kimi-K2.6 576 = 512 + 64, validated
                    # byte-exact: 560/560 L3 hits, 0 miss) → offload. Anything
                    # else — larger hidden (= scale packed) or non-MLA packed —
                    # is skipped (no toggle: correctness over the optimization;
                    # a new safe format is added HERE, not via a runtime env).
                    _pmh = getattr(self, "_plain_mla_hidden", None)
                    _plain_mla_fp8 = (
                        num_kv_channels == 1 and _pmh is not None and hidden_dim == _pmh
                    )
                    if _is_packed_quant_kv_dtype(sample.dtype):
                        if _plain_mla_fp8:
                            logger.info(
                                "register_kv_caches: group %d layer %r packed KV "
                                "dtype %s hidden=%d == kv_lora_rank+qk_rope_head_dim "
                                "— plain MLA fp8 cast (no interleaved scale), "
                                "offloading to L3.",
                                gid,
                                present[0],
                                sample.dtype,
                                hidden_dim,
                            )
                        else:
                            logger.warning(
                                "register_kv_caches: group %d layer %r has packed/"
                                "quantized KV dtype %s (shape %s) that is NOT a plain "
                                "MLA fp8 latent (hidden != kv_lora_rank+qk_rope_head_"
                                "dim) — scale-packed/unrecognized layout is unvalidated,"
                                " chunked fusion SKIPS it (no L3, no corruption).",
                                gid,
                                present[0],
                                sample.dtype,
                                shape,
                            )
                            continue
                    # Use the LOGICAL block size from KVCacheConfig, NOT
                    # the tensor's block dim — they differ for MLA latent
                    # layouts (shape dim 1 vs logical 16) and any mismatch
                    # desyncs chunk_tokens from the scheduler, skipping all
                    # saves. See `_logical_block_size`.
                    block_size = self._logical_block_size(group, kv_cfg)
                    if block_size != block_size_in_shape:
                        logger.info(
                            "register_kv_caches: group %d tensor block "
                            "dim=%d but logical (config) block_size=%d — "
                            "using config (MLA latent layout); hidden_dim=%d",
                            gid,
                            block_size_in_shape,
                            block_size,
                            hidden_dim,
                        )
                    self._group_kv_spec[gid] = {
                        "layer_names": present,
                        "num_blocks": num_blocks,
                        "block_size": block_size,
                        # Bytes for ONE block across ALL the group's layers
                        # (chunk auto-sizing); matches the scheduler's
                        # config-derived value for the same model.
                        "page_bytes": (
                            num_kv_channels
                            * hidden_dim
                            * block_size
                            * sample.dtype.itemsize
                            * len(present)
                        ),
                        "hidden_dim": hidden_dim,
                        "num_kv_channels": num_kv_channels,
                        "dtype": sample.dtype,
                        "dtype_str": _torch_dtype_to_str(sample.dtype),
                    }
            if not self._group_kv_spec and self._kv_caches:
                # Fallback for connectors built without KVCacheConfig
                # (older vLLM, unit tests). Treat all layers as group 0.
                first_name = next(iter(self._kv_caches))
                # If KVCacheConfig groups DO exist (e.g. a pure-recurrent
                # model whose only group was skipped above), consult the
                # spec for this layer so a 3-D Mamba state isn't revived
                # as MLA by the shape probe below.
                _fallback_spec = None
                if groups:
                    for group in groups:
                        if first_name in (getattr(group, "layer_names", ()) or ()):
                            _fallback_spec = getattr(group, "kv_cache_spec", None)
                            break
                if _is_non_paged_kv_spec(_fallback_spec):
                    logger.info(
                        "register_kv_caches: fallback layer %r belongs to "
                        "a non-paged recurrent/conv-state group (%s) — "
                        "skipping L3 offload",
                        first_name,
                        type(_fallback_spec).__name__,
                    )
                else:
                    sample = self._kv_caches[first_name]
                    shape = tuple(sample.shape)
                    # Defaults for shapes that match none of the branches
                    # below: num_kv_channels stays 0, which gates every use
                    # of the dims via `if num_kv_channels:` — but initialize
                    # them explicitly so an unrecognized layout can never hit
                    # an UnboundLocalError.
                    num_kv_channels = 0
                    num_blocks = 0
                    block_size_in_shape = 0
                    hidden_dim = 0
                    if len(shape) >= 4 and shape[0] == 2:
                        num_kv_channels = 2
                        hidden_dim = 1
                        for d in shape[3:]:
                            hidden_dim *= int(d)
                        num_blocks, block_size_in_shape = int(shape[1]), int(shape[2])
                    elif len(shape) >= 4 and shape[1] == 2:
                        # vLLM 0.23 FullAttention INTERLEAVED layout
                        # [num_blocks, 2, block_size, num_kv_heads, head_dim].
                        num_kv_channels = 2
                        hidden_dim = 1
                        for d in shape[3:]:
                            hidden_dim *= int(d)
                        num_blocks, block_size_in_shape = int(shape[0]), int(shape[2])
                    elif len(shape) == 3:
                        num_kv_channels = 1
                        num_blocks = int(shape[0])
                        block_size_in_shape = int(shape[1])
                        hidden_dim = int(shape[2])
                    _fb_pmh = getattr(self, "_plain_mla_hidden", None)
                    _fallback_plain_mla_fp8 = (
                        num_kv_channels == 1 and _fb_pmh is not None and hidden_dim == _fb_pmh
                    )
                    if (
                        num_kv_channels
                        and _is_packed_quant_kv_dtype(sample.dtype)
                        and not _fallback_plain_mla_fp8
                    ):
                        # Same packed/quantized-dtype guard as the primary
                        # probe: skip scale-packed fp8/uint8 caches (no L3, no
                        # corruption). The plain-MLA-fp8 case (hidden ==
                        # kv_lora_rank+qk_rope_head_dim, no interleaved scale)
                        # is auto-detected above and offloads normally.
                        logger.warning(
                            "register_kv_caches (fallback): layer %r has packed/"
                            "quantized KV dtype %s (shape %s) that is NOT a plain "
                            "MLA fp8 latent — scale-packed/unrecognized layout is "
                            "unvalidated, chunked fusion SKIPS it (no L3, no "
                            "corruption).",
                            first_name,
                            sample.dtype,
                            shape,
                        )
                        num_kv_channels = 0
                    if num_kv_channels:
                        self._group_kv_spec[0] = {
                            "layer_names": list(self._kv_caches.keys()),
                            "num_blocks": num_blocks,
                            "block_size": block_size_in_shape,
                            "page_bytes": (
                                num_kv_channels
                                * hidden_dim
                                * block_size_in_shape
                                * sample.dtype.itemsize
                                * len(self._kv_caches)
                            ),
                            "hidden_dim": hidden_dim,
                            "num_kv_channels": num_kv_channels,
                            "dtype": sample.dtype,
                            "dtype_str": _torch_dtype_to_str(sample.dtype),
                        }
                        for lname in self._kv_caches:
                            self._layer_to_group.setdefault(lname, 0)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "register_kv_caches: group_kv_spec extraction skipped "
                "(%s); chunked-fusion will be unavailable for this "
                "connector instance",
                exc,
            )
            self._group_kv_spec = {}
        # Now that the real KV spec is known, finalize auto chunk sizing
        # (target INFERA_KVD_CHUNK_TARGET_MIB). Same per-block bytes as
        # the scheduler's config-derived value → identical chunk_tokens.
        self._autosize_chunk_tokens()

    def _split_dsa_entry(self, entry: tuple) -> list[tuple]:
        """Expand a mixed-DSA-group save/load entry into (main, indexer)
        sub-entries so each is gathered/scattered at its OWN hidden_dim.

        Why this is needed: for a DSA model whose main MLA latent (hidden
        576) and sparse-attn ``indexer.k_cache`` (hidden 132) live in ONE
        vLLM kv_cache_group (glm_moe_dsa / GLM-5.2 — DeepSeek-V3.2 instead
        gets two native groups), the scheduler side does NOT split (its
        chunk emit keeps the group's single gid with ALL layer names). The
        worker's ``register_kv_caches`` DID split into gid (main) +
        ``_DSA_INDEXER_GID_BASE+gid`` (indexer). So a raw gid=0 entry carries
        both hidden dims but the gather uses one (576) — the 132-wide indexer
        tensors then overrun their row capacity (cap_rows = numel/576 too
        small) → ``kv_chunk OOB GUARD`` → the save fails and L3 reuse never
        fires. Splitting per-layer by hidden_dim here fixes it while leaving
        the scheduler's chunk_tokens untouched.

        Save entry: ``(block_ids, key, retention, gid, layer_names)`` (len 5).
        Load entry: ``(block_ids, key, gid, layer_names)`` (len 4).
        The indexer sub-entry reuses the SAME per-page block ids (it aliases
        group-0's blocks; the gather falls back to ``page_ids[0]`` for the
        synthetic gid) and derives its key by swapping the trailing gid byte
        (``key[:7] + byte(indexer_gid)``) — matching the scheduler's
        ``chunk_key[:7] + bytes([gid & 0xFF])`` construction.

        Returns ``[entry]`` unchanged for plain MLA, regular attention, or
        DSA models vLLM already grouped separately (no indexer sub-spec
        registered for this gid on this worker).
        """
        if len(entry) == 5:
            block_ids, key, retention, gid, layer_names = entry
        elif len(entry) == 4:
            block_ids, key, gid, layer_names = entry
            retention = None
        else:
            return [entry]
        idx_gid = _DSA_INDEXER_GID_BASE + int(gid)
        if idx_gid not in self._group_kv_spec or not layer_names:
            return [entry]
        idx_names = [n for n in layer_names if _is_dsa_indexer_layer(n)]
        if not idx_names:
            return [entry]  # already split or main-only — nothing mixed
        main_names = [n for n in layer_names if not _is_dsa_indexer_layer(n)]
        idx_key = bytes(key[:7]) + bytes([idx_gid & 0xFF])
        out: list[tuple] = []
        if len(entry) == 5:
            if main_names:
                out.append((block_ids, key, retention, gid, main_names))
            out.append((block_ids, idx_key, retention, idx_gid, idx_names))
        else:
            if main_names:
                out.append((block_ids, key, gid, main_names))
            out.append((block_ids, idx_key, idx_gid, idx_names))
        return out

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Dispatch to one of four load paths based on
        ``self._layerwise_mode`` (set from ``INFERA_KVD_LAYERWISE_LOAD``):
          - ``off`` (default): load everything inline before forward
          - ``stepped``: prepare per-chunk state; per-layer scatter in wait_for_layer_load
          - ``prefetch``: stepped + per-layer H2D on dedicated copy stream + lookahead
          - ``parallel``: submit one _load_chunk_packed per chunk to load executor;
                          GIL-released hipFile.read() lets N threads run truly parallel.
                          Bandwidth fan-out (24+ GB/s on 128 MiB blocks N=4-8) vs
                          ``prefetch``'s compute-overlap pipelining. Pick this when
                          storage→GPU read BW is the bottleneck (large chunks, big
                          per-chunk size, RDMA fabric).
        """
        meta = self._bound_metadata()
        if meta is None:
            return
        # Expand mixed-DSA-group load entries (main + indexer at distinct
        # hidden dims) BEFORE any mode dispatch, keeping load_chunk_req_ids
        # aligned. No-op unless this worker split a DSA group (see
        # _split_dsa_entry).
        if meta.packed_chunks_to_load:
            _rids = meta.load_chunk_req_ids or []
            _new_entries: list[tuple] = []
            _new_rids: list = []
            for _i, _entry in enumerate(meta.packed_chunks_to_load):
                _rid = _rids[_i] if _i < len(_rids) else None
                for _e in self._split_dsa_entry(_entry):
                    _new_entries.append(_e)
                    _new_rids.append(_rid)
            meta.packed_chunks_to_load = _new_entries
            if meta.load_chunk_req_ids:
                meta.load_chunk_req_ids = _new_rids
        # Race fix (companion to TTL self-healing on the scheduler-side
        # dedupe): drain any in-flight async saves before the load tries
        # to fetch chunks. Without this, the load may race a save that
        # wait_for_save submitted-but-didn't-await for one of the chunks
        # we're about to fetch — kvd doesn't have the key yet, the load
        # misses, and vLLM re-prefills despite the prefix having been
        # "saved" moments earlier. Timeout is intentionally short so the
        # forward pass never stalls more than INFERA_KVD_LOAD_DRAIN_S
        # waiting on slow disk.
        if self._load_drain_timeout_s > 0 and meta.packed_chunks_to_load:
            try:
                self.drain_pending_saves(timeout=self._load_drain_timeout_s)
            except Exception:
                logger.debug("start_load_kv: drain_pending_saves raised", exc_info=True)
        # Fix A: async load — submit non-blocking and return immediately so
        # the engine step isn't held hostage by the (serial) L3 read;
        # get_finished reports per-req completion.
        if self._async_load and meta.load_chunk_req_ids:
            self._start_load_kv_async(meta)
            return
        if self._layerwise_mode == "stepped":
            self._start_load_kv_layerwise_a(meta)
            return
        if self._layerwise_mode == "prefetch":
            self._start_load_kv_layerwise_b(meta)
            return
        if self._layerwise_mode == "parallel":
            self._start_load_kv_parallel(meta)
            return
        # "off" → inline sync load.
        for entry in meta.packed_chunks_to_load:
            try:
                self._load_chunk_packed(entry)
            except Exception:
                logger.exception(
                    "start_load_kv: chunk load failed for entry (key=%s gid=%s)",
                    entry[1].hex()[:16] if len(entry) > 1 else "?",
                    entry[2] if len(entry) > 2 else "?",
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Per-layer barrier before that layer's attention runs.

        - ``off``: no-op — start_load_kv already scattered everything.
        - ``stepped``: per-chunk per-layer Triton scatter on default stream.
        - ``prefetch``: wait on copy_stream's H2D event for this layer, then scatter.
        - ``parallel``: no-op — start_load_kv waited for all futures already.
        """
        if self._layerwise_mode == "stepped":
            self._wait_for_layer_load_stepped(layer_name)
            return
        if self._layerwise_mode == "prefetch":
            self._wait_for_layer_load_prefetch(layer_name)
            return
        # "off" → no-op

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """No-op per layer. Chunked-fusion saves all layers in one
        kernel pass at ``wait_for_save`` time (vLLM's contract
        guarantees the source paged-KV blocks remain valid through
        that hook), so there's nothing to do per layer."""

    def wait_for_save(self) -> None:
        """Two-phase chunk save with **batched D2H + dedicated stream**
        for async overlap with model forward.

        Phase 1 (SYNCHRONOUS, runs here): preprocess all chunks (spec
        lookup, path setup), queue all Triton gathers and D2H copies
        onto a dedicated CUDA stream, then ONE save-stream sync. We
        must finish gather here because vLLM's contract guarantees
        the source paged-KV blocks remain valid only through this hook.

        Phase 2 (ASYNCHRONOUS, runs in `self._save_executor`): per
        chunk, write the prepared blob to disk (or UDS Set), then
        register_file_entry. Submitting and returning here lets
        vLLM's forward pass proceed without waiting for disk + RPC.

        Two optimizations vs the original per-chunk sync loop:
          A. **Dedicated CUDA stream** (`self._save_stream`): Triton
             gather and D2H copy run on a separate stream so they
             overlap with the model forward on the default stream
             instead of serializing through it.
          B. **Single sync per wait_for_save**: instead of N implicit
             syncs via per-chunk `.cpu()`, we issue all gathers and
             non-blocking D2H copies (`copy_(..., non_blocking=True)`
             into pinned host buffers), then sync the save stream
             ONCE. Cuts host stall from N×6 ms down to one ~6 ms wait
             that overlaps with model forward.

        Mirrors LMCache's `cache_engine.store()` pattern: their
        `batched_from_gpu` does a single batched D2H, `storage_manager.
        batched_put` returns immediately after enqueueing to its
        worker pool. See project_lmcache_mi300x_agentic_bench_2026_05.
        """
        meta = self._bound_metadata()
        if meta is None or not meta.packed_chunks_to_save:
            return
        # Expand mixed-DSA-group save entries (main + indexer at distinct
        # hidden dims). No-op unless this worker split a DSA group (see
        # _split_dsa_entry). Without this the indexer's 132-wide tensors get
        # gathered at the main latent's 576 → kv_chunk OOB → save fails.
        meta.packed_chunks_to_save = [
            _e for _entry in meta.packed_chunks_to_save for _e in self._split_dsa_entry(_entry)
        ]
        # MLA TP-dedup (#64): the cached latent is byte-identical on every
        # TP rank, so only rank 0 persists it; ranks 1..N skip the
        # redundant D2H + write entirely. compat_key was folded to a
        # shared (tp_rank 0) namespace so all ranks find rank 0's copy on
        # load. No-op unless MLA && tp_size>1 (pure-DP / regular attention
        # keep distinct per-rank data and are unaffected).
        if self._dedup_mla_writes and self._tp_rank != 0:
            return
        try:
            import torch as _torch
        except ImportError:
            return

        # Lazy-init the save stream on first call. We pick the device
        # of any registered KV cache; if none registered yet, fall back
        # to the per-chunk path.
        save_stream = self._ensure_save_stream(_torch)

        # ------------------------------------------------------------
        # Phase 1a: preprocess + queue all GPU work on save_stream
        # ------------------------------------------------------------
        prepared: list[tuple[str, Any, tuple]] = []
        # Each prepared entry: (kind, payload_carrier, args)
        #   kind = "posix" | "uds_set" | "gpu_direct"
        #   payload_carrier =
        #     "posix"/"uds_set": pinned-host tensor receiving D2H
        #     "gpu_direct":      device staging tensor (kept alive)
        #   args = tuple of metadata for the async finisher
        if save_stream is not None:
            stream_ctx = _torch.cuda.stream(save_stream)
        else:
            from contextlib import nullcontext

            stream_ctx = nullcontext()

        with stream_ctx:
            for entry in meta.packed_chunks_to_save:
                try:
                    prep = self._prepare_chunk_save_on_stream(entry, save_stream)
                    if prep is not None:
                        prepared.append(prep)
                except Exception:
                    logger.exception(
                        "wait_for_save: prepare failed for entry (key=%s gid=%s)",
                        entry[1].hex()[:16] if len(entry) > 1 else "?",
                        entry[3] if len(entry) > 3 else "?",
                    )

        # ------------------------------------------------------------
        # Phase 1b: ONE sync for the entire batch
        # ------------------------------------------------------------
        if prepared and save_stream is not None:
            save_stream.synchronize()

        # ------------------------------------------------------------
        # Phase 2: submit each chunk's final-write to the executor
        # ------------------------------------------------------------
        for kind, carrier, args in prepared:
            try:
                cache_key: bytes | None = None
                cache_blob: bytes | None = None
                if kind == "posix":
                    # carrier = (pinned_uint8_view, pin_idx). ZERO-COPY:
                    # the executor writes straight from the pinned view via
                    # os.writev — no bytes() materialize, no header+payload
                    # concat (those two 128 MiB copies were the regression).
                    pinned_view, pin_idx = carrier
                    kvd_key, header_bytes, payload_bytes, tmp, path, retention = args
                    try:
                        future = self._save_executor.submit(
                            self._async_posix_save,
                            pinned_view,
                            pin_idx,
                            kvd_key,
                            header_bytes,
                            payload_bytes,
                            tmp,
                            path,
                            retention,
                        )
                    except Exception:
                        # submit failed (e.g. executor shutdown) → the worker
                        # that would release the slot never runs; release here
                        # so the pinned pool can't leak/exhaust.
                        self._release_pinned_slot(pin_idx)
                        raise
                elif kind == "uds_set":
                    # UDS Set must hand the daemon a bytes blob, so this path
                    # still materializes; release the pinned slot right after.
                    pinned_view, pin_idx = carrier
                    cpu_bytes = bytes(memoryview(pinned_view.numpy()))
                    self._release_pinned_slot(pin_idx)
                    kvd_key, retention, header_bytes = args
                    cache_key = kvd_key
                    cache_blob = header_bytes + cpu_bytes
                    future = self._save_executor.submit(
                        self._async_uds_set_save,
                        kvd_key,
                        cache_blob,
                        retention,
                    )
                elif kind == "gpu_direct":
                    # carrier = device staging tensor; args =
                    # (header_bytes, payload_bytes, kvd_key, tmp, path, retention)
                    future = self._save_executor.submit(
                        self._async_gpu_direct_save,
                        carrier,
                        *args,
                    )
                else:
                    continue
                self._add_pending_save_future(future)
                # In-flight cache: load can serve from `cache_blob` while
                # the disk write is still in flight; on completion the
                # blob moves to the LRU ring buffer for fast re-read.
                # Skipped for gpu_direct (device tensor; would need D2H
                # copy to host, defeating the point of staying GPU-side).
                if cache_key is not None and cache_blob is not None:
                    self._register_inflight_chunk(cache_key, cache_blob, future)
            except Exception:
                logger.exception("wait_for_save: submit failed")

    def _ensure_save_stream(self, torch_mod: Any) -> Any | None:
        """Lazy-init a dedicated CUDA stream for save gather + D2H so
        the work overlaps with the model forward on the default stream
        instead of serializing through it. Returns None if no CUDA
        device is registered yet (registers KV caches haven't fired)
        or torch CUDA isn't available."""
        stream = getattr(self, "_save_stream", None)
        if stream is not None:
            return stream
        if not torch_mod.cuda.is_available():
            self._save_stream = None
            return None
        if not self._kv_caches:
            return None
        # Defense in depth: only a tensor has `.device`. register_kv_caches
        # already filters non-tensor (recurrent/conv state) entries, but guard
        # here too so a stray non-tensor can never crash the save stream.
        sample = next((v for v in self._kv_caches.values() if hasattr(v, "device")), None)
        if sample is None:
            return None
        device = sample.device
        if device.type != "cuda":
            self._save_stream = None
            return None
        try:
            self._save_stream = torch_mod.cuda.Stream(device=device)
        except Exception:
            logger.exception(
                "_ensure_save_stream: torch.cuda.Stream construction failed"
                " — falling back to default stream"
            )
            self._save_stream = None
        return self._save_stream

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report which of `finished_req_ids` have completed their
        async save/load.

        We return (None, None) — same as LMCache's
        `LMCacheConnectorV1.get_finished`. The async save path is
        fire-and-forget for correctness: vLLM is free to recycle the
        request's resources as soon as we return from wait_for_save
        (the D2H/Triton-gather phase has already copied bytes off
        the paged-KV blocks; the async finish only touches the
        already-copied data on disk/UDS). Per-request tracking would
        be needed only if we ever wired our `is_async=True` load
        path or wanted to back-pressure new saves on disk drain.

        Fix A: when async load is enabled, report a req in the SECOND tuple
        element (finished-receiving) once all its load futures are done —
        each finishes only after Triton scatter + cuda.synchronize(), so the
        KV is live before vLLM promotes the req out of WAITING_FOR_REMOTE_KVS.
        """
        if not self._async_load:
            return None, None
        done: set[str] = set()
        with self._async_inflight_lock:
            for rid, futs in list(self._async_inflight.items()):
                if all(f.done() for f in futs):
                    for f in futs:
                        exc = f.exception()
                        if exc is not None:
                            logger.warning(
                                "async load future failed for req %s: %r",
                                rid,
                                exc,
                            )
                    done.add(rid)
                    del self._async_inflight[rid]
        # vLLM unpacks (finished_sending, finished_recving); ours are
        # receiving -> second element so they're promoted, not freed.
        return None, (done or None)

    def shutdown(self) -> None:
        """Tear down the kvd connection and background loop."""
        self.close()

    def reset_cache(self) -> bool:
        """vLLM prefix-cache reset hook.

        Triggered by ``POST /reset_prefix_cache?reset_external=true`` →
        ``Scheduler.reset_prefix_cache`` → ``connector.reset_cache()`` (and
        fanned out by MultiConnector to each child). Lets a prefix-cache reset
        also drop our offloaded L3 KV, so callers don't have to rm files or
        restart the engine.

        Only the SCHEDULER-role instance holds a live client/loop; worker-role
        instances just wipe their (shared) hipFile roots. Returns True on success
        (vLLM treats non-False as success).
        """
        ok = True
        # (1) Clear the daemon's RAM index / daemon-managed tiers. For the
        # hipFile direct-write L3 this is effectively a no-op (the daemon does
        # not own those files), but it is harmless and keeps daemon state tidy.
        if self._client is not None and self._loop is not None and not self._closed:
            try:
                self._run_async(self._client.clear(model=self._model, compat_key=self._compat_key))
            except Exception:
                logger.exception("reset_cache: kvd client.clear failed")
                ok = False
        # (2) Wipe the hipFile L3 root dirs directly — this is what actually
        # frees the offloaded KV on disk. Guard-railed (see
        # _safe_wipe_dir_contents) so a misconfigured root can never rm the
        # system. NOTE: not synchronized with in-flight save/load; prefer
        # ``reset_running_requests=true`` (preempts first) when calling.
        for retention, root in (self._hipfile_roots or {}).items():
            try:
                n = _safe_wipe_dir_contents(root)
                logger.info("reset_cache: wiped L3 root %s=%s (%d entries)", retention, root, n)
            except Exception:
                logger.exception("reset_cache: failed to wipe L3 root %s=%s", retention, root)
                ok = False
        return ok

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 1. Drain pending async-save futures BEFORE we kill the asyncio
        #    loop they call into via self._run_async. Worker threads
        #    hold references to UDS register_file_entry / set RPCs that
        #    must complete (or fail cleanly) before we tear down the
        #    loop. Use getattr in case __init__ raised before the
        #    executor was set up (kvd-unreachable case).
        save_executor = getattr(self, "_save_executor", None)
        if save_executor is not None:
            try:
                save_executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                logger.exception("close: save_executor shutdown raised")
            self._save_executor = None
        # 1b. Load executor (mode=parallel). Same pattern as save,
        #     PLUS: before shutdown, fire one _cleanup_parallel_load_tls
        #     task per worker so each thread cleanly deregisters its
        #     persistent hipFile buffer. Without this, the device tensor
        #     held by the per-thread TLS is GC'd when the thread exits
        #     while hipFile still has its address REGISTERED → driver
        #     cleanup later touches freed device memory → SIGSEGV.
        #
        #     We submit (max_workers * 4) cleanup tasks with a tiny
        #     sleep at the top of each — the sleep keeps each thread
        #     busy long enough that the next task lands on a different
        #     thread, guaranteeing every worker hits its own TLS at
        #     least once. The cleanup itself is idempotent (no-op when
        #     the TLS slot is already None) so over-submission is safe.
        load_executor = getattr(self, "_load_executor", None)
        if load_executor is not None:
            try:
                import time as _time

                n_workers = getattr(load_executor, "_max_workers", 8)

                def _cleanup_with_settle():
                    _time.sleep(0.01)
                    type(self)._cleanup_parallel_load_tls()

                cleanup_futs = [
                    load_executor.submit(_cleanup_with_settle) for _ in range(n_workers * 4)
                ]
                for cf in cleanup_futs:
                    try:
                        cf.result(timeout=5.0)
                    except Exception:
                        pass
            except Exception:
                logger.exception("close: load executor TLS cleanup raised")
            try:
                load_executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                logger.exception("close: load_executor shutdown raised")
            self._load_executor = None
        # 2. Drain layerwise pinned-host pool (Variant B). Each slot is
        #    a pin_memory torch.Tensor; freeing them releases the
        #    underlying cudaHostAlloc allocations.
        pinned_pool = getattr(self, "_layerwise_pinned_pool", None)
        if pinned_pool is not None:
            try:
                pinned_pool.drain()
            except Exception:
                logger.exception("close: pinned pool drain raised")
            self._layerwise_pinned_pool = None
        if self._client is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._client.close(), self._loop).result(
                    timeout=5.0
                )
            except (Exception, TimeoutError):
                pass
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        # Stop the L3 reaper background thread (no-op if never started).
        reaper = getattr(self, "_l3_reaper", None)
        if reaper is not None:
            try:
                reaper.stop()
            except Exception:
                logger.exception("close: l3 reaper stop raised")
            self._l3_reaper = None

    def __del__(self) -> None:  # pragma: no cover — interpreter teardown timing
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal — config extraction, runtime bridge
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_model_id(vllm_config: Any) -> str:
        """vLLM's config nests `model_config.model`; fall back to common
        alternatives so the skeleton works against various vLLM versions
        without breaking."""
        for attr_path in (
            ("model_config", "model"),
            ("model_config", "served_model_name"),
            ("served_model_name",),
            ("model",),
        ):
            cur = vllm_config
            ok = True
            for attr in attr_path:
                cur = getattr(cur, attr, None)
                if cur is None:
                    ok = False
                    break
            if ok and isinstance(cur, str) and cur.strip():
                return cur.strip()
        return ""

    @staticmethod
    def _resolve_tp_rank_size(vllm_config: Any) -> tuple[int, int]:
        """Resolve (tp_rank, tp_size) for THIS process — used to gate the
        MLA write-once dedup (#64). Mirrors the tp resolution order in
        ``_extract_compat_key``: live distributed group → config attr →
        RANK env → 0."""
        parallel = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel, "tensor_parallel_size", 1) or 1) if parallel else 1
        tp_rank: int | None = None
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            tp_rank = int(get_tensor_model_parallel_rank())
        except (ImportError, AssertionError, RuntimeError, AttributeError):
            pass
        if tp_rank is None and parallel is not None:
            v = getattr(parallel, "tensor_parallel_rank", None)
            if isinstance(v, int):
                tp_rank = v
        if tp_rank is None:
            try:
                tp_rank = int(os.environ.get("RANK", "0")) % max(tp_size, 1)
            except ValueError:
                tp_rank = 0
        return tp_rank, tp_size

    @staticmethod
    def _extract_compat_key(vllm_config: Any) -> str:
        """Derive a per-(quant, tp_rank, pp_rank) compat key.

        vLLM v1's `ParallelConfig` carries the cluster TOPOLOGY
        (tp_size, pp_size) but NOT the current process's rank — every
        worker subprocess sees the same config object. The rank lives
        in the torch.distributed state, accessed via vLLM's
        `get_tensor_model_parallel_rank` / `get_pp_group()`.

        Without including the rank in compat_key, two TP workers
        would share the same kvd namespace and OVERWRITE each
        other's KV slices on SET — silent corruption observed on
        MI355X + MiniMax-M2.5 TP=2 (answer turned to garbage after
        a vLLM restart). The fix: ask the live distributed group
        for our rank, fall back to env vars, fall back to "rank 0".
        """
        parallel = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel, "tensor_parallel_size", 1) or 1) if parallel else 1
        pp_size = int(getattr(parallel, "pipeline_parallel_size", 1) or 1) if parallel else 1

        # Rank resolution order (most authoritative first):
        # 1. vLLM's distributed group accessors. These are the only
        #    correct source when called from inside a TP worker
        #    subprocess — the rank is per-process, not in the config.
        # 2. `tensor_parallel_rank` / `pipeline_parallel_rank` on the
        #    config object. Older vLLM stamped these in; tests also
        #    inject them via SimpleNamespace.
        # 3. RANK env var (set by vLLM's spawner) → derive tp/pp from
        #    the topology shape.
        # 4. Default to rank 0.
        tp_rank: int | None = None
        pp_rank: int | None = None

        # (1) live distributed state
        source = "unknown"
        try:
            from vllm.distributed.parallel_state import (
                get_pp_group,
                get_tensor_model_parallel_rank,
            )

            tp_rank = int(get_tensor_model_parallel_rank())
            pp_rank = int(get_pp_group().rank_in_group)
            source = "live-distributed"
        except (ImportError, AssertionError, RuntimeError, AttributeError):
            pass

        # (2) config attrs (tests + older vLLM)
        if tp_rank is None and parallel is not None:
            cfg_tp_rank = getattr(parallel, "tensor_parallel_rank", None)
            if isinstance(cfg_tp_rank, int):
                tp_rank = cfg_tp_rank
                source = "config-attr"
        if pp_rank is None and parallel is not None:
            cfg_pp_rank = getattr(parallel, "pipeline_parallel_rank", None)
            if isinstance(cfg_pp_rank, int):
                pp_rank = cfg_pp_rank
                if source == "unknown":
                    source = "config-attr"

        # (3) RANK env var
        if tp_rank is None or pp_rank is None:
            try:
                global_rank = int(os.environ.get("RANK", "0"))
                if tp_rank is None:
                    tp_rank = global_rank % max(tp_size, 1)
                if pp_rank is None:
                    pp_rank = (global_rank // max(tp_size, 1)) % max(pp_size, 1)
                source = "env-RANK"
            except ValueError:
                pass

        # (4) defaults
        if tp_rank is None:
            tp_rank = 0
            if source == "unknown":
                source = "default-zero"
        if pp_rank is None:
            pp_rank = 0
            if source == "unknown":
                source = "default-zero"

        # PR #9 review fix P1: log the resolved source so operators can
        # detect the silent fallback. If two TP workers BOTH resolve to
        # rank 0 (e.g. because connector ran before
        # init_process_group AND RANK env wasn't exported), they'd share
        # a compat_key and silently clobber each other on kvd. Grepping
        # for "default-zero" makes the misconfig visible.
        log_level = logging.WARNING if source == "default-zero" else logging.INFO
        logger.log(
            log_level,
            "vllm.kvd_connector: resolved TP/PP rank tp=%d/%d pp=%d/%d via %s",
            tp_rank,
            tp_size,
            pp_rank,
            pp_size,
            source,
        )
        # MLA TP-dedup (#64): under MLA the cached latent is REPLICATED
        # across TP ranks (identical bytes), so collapse all of them onto
        # one namespace (tp_rank 0) — combined with the rank-0-only write
        # gate in wait_for_save, this stores ONE copy instead of tp_size.
        # Gated on tp_size>1 so it's a no-op for single-rank / pure-DP
        # (where each DP rank's KV is DISTINCT and must NOT be merged;
        # there the content key already separates ranks' distinct prompts).
        # Regular attention is untouched — its per-rank shards genuinely
        # differ and must keep distinct keys.
        if tp_size > 1 and _is_mla_from_config(vllm_config):
            tp_rank = 0
        # PR #9 review fix A: bake a weight fingerprint into compat_key
        # so two model directories with the same display name but
        # different actual weights/config do NOT share kvd namespace.
        # Without this, a fine-tune at the same model_id silently
        # reuses the base model's KV — shape-compatible, semantically
        # wrong. The fingerprint is a 16-hex-char hash of config.json
        # (or tokenizer.json fallback) from the model dir.
        weight_fp = _vllm_weight_fingerprint(vllm_config)
        if weight_fp:
            return f"tp{tp_rank}of{tp_size}_pp{pp_rank}of{pp_size}_w{weight_fp}"
        return f"tp{tp_rank}of{tp_size}_pp{pp_rank}of{pp_size}"

    @staticmethod
    def _extract_block_hashes_after(request: Any, num_computed_tokens: int) -> list[bytes]:
        """Pull the request's block hashes, return only those whose
        token range starts AT or AFTER num_computed_tokens (the ones
        vLLM hasn't already computed locally)."""
        block_hashes = getattr(request, "kv_block_hashes", None)
        if block_hashes is None:
            block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes:
            return []

        block_size = InferaKvdConnector._extract_block_size(request)
        skip_blocks = num_computed_tokens // max(block_size, 1)
        # Each block hash is an int (XXH3-64). Convert to bytes for the
        # daemon (which uses bytes keys).
        out: list[bytes] = []
        for h in block_hashes[skip_blocks:]:
            if isinstance(h, int):
                out.append(int(h).to_bytes(8, "little", signed=False))
            elif isinstance(h, (bytes, bytearray)):
                out.append(bytes(h))
            else:
                # Unknown hash representation — bail.
                return []
        return out

    @staticmethod
    def _extract_block_size(request: Any) -> int:
        """vLLM's block size is on the kv cache config, but the connector
        sees it through various plumbing. Best-effort across versions."""
        for attr in ("kv_block_size", "block_size", "kv_cache_block_size"):
            val = getattr(request, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        # Last resort. vLLM default in V1 is 16 for many models.
        return 16

    def _block_size_now(self, request: Any = None) -> int:
        """Authoritative KV-cache block size for token/chunk accounting.

        The block size MUST come from the engine's real KV cache, NOT a
        guessed default: the ROCm AITER unified-attn backend (and others)
        override vLLM's v1 default of 16 to **64**. Getting this wrong
        scales `get_num_new_matched_tokens`'s token accounting by
        block_size, so a 16-vs-64 mismatch makes the connector report ¼
        of the truly-cached tokens — vLLM then loads a fraction and the
        external cache looks like a near-total miss.

        `_group_kv_spec[gid]["block_size"]` is captured from the real KV
        tensor shape (worker) or `kv_cache_spec` (scheduler), so prefer
        it. Fall back to request-carried attrs, then the v1 default.
        """
        if self._group_kv_spec:
            for spec in self._group_kv_spec.values():
                try:
                    bs = int(spec.get("block_size", 0) or 0)
                except (TypeError, ValueError):
                    bs = 0
                if bs > 0:
                    return bs
        return self._extract_block_size(request) if request is not None else 16

    def _autosize_chunk_tokens(self) -> None:
        """If chunk sizing is 'auto', derive ``self._chunk_tokens`` so each
        chunk is >= ``self._chunk_target_bytes`` (default 128 MiB) — the
        P2PDMA read sweet spot. Idempotent; safe to call from both the
        scheduler-side bootstrap and the worker-side register_kv_caches.

        Sized in BYTES: ``per_page_bytes`` = Σ over cache groups of (bytes
        for ONE block across ALL the group's layers). ``N = ceil(target /
        per_page_bytes)``; ``chunk_tokens = N × block_size``. Both sides
        compute the same per_page_bytes from the same model, so the chunk
        grain (and chunk keys) match — no scheduler/worker divergence.
        """
        if not getattr(self, "_chunk_tokens_auto", False):
            return
        if not self._group_kv_spec:
            return
        block_size = 0
        per_page_bytes = 0
        for spec in self._group_kv_spec.values():
            try:
                bs = int(spec.get("block_size", 0) or 0)
                pb = int(spec.get("page_bytes", 0) or 0)
            except (TypeError, ValueError):
                continue
            if bs > 0:
                block_size = bs
            per_page_bytes += pb
        if block_size <= 0 or per_page_bytes <= 0:
            return  # spec not yet complete — keep provisional, retry later
        n_pages = max(1, -(-self._chunk_target_bytes // per_page_bytes))  # ceil
        new_ct = n_pages * block_size
        if new_ct != self._chunk_tokens:
            logger.info(
                "kvd auto chunk size: target=%d MiB, per-block(all layers)="
                "%.2f MiB → N=%d pages → chunk_tokens=%d (~%.1f MiB/chunk)",
                self._chunk_target_bytes >> 20,
                per_page_bytes / (1 << 20),
                n_pages,
                new_ct,
                n_pages * per_page_bytes / (1 << 20),
            )
            self._chunk_tokens = new_ct

    def _start_background_loop(self) -> None:
        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_runner, name=f"vllm-kvd-loop-{os.getpid()}", daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread

    def _connect_or_raise(self) -> None:
        client = KvdClient(self._socket_path, client_id=f"vllm-pid{os.getpid()}")
        try:
            asyncio.run_coroutine_threadsafe(client.connect(), self._loop).result()
        except Exception:
            self.close()
            raise
        self._client = client
        logger.info(
            "infera-kvd vLLM connector connected to %s (model=%s, compat_key=%s, role=%s)",
            self._socket_path,
            self._model or "<empty>",
            self._compat_key,
            self._role,
        )

    def _run_async(self, coro):
        if self._loop is None or self._client is None:
            raise RuntimeError("connector is closed or not yet initialized")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ------------------------------------------------------------------
    # Config resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_hipfile_roots(explicit: dict[str, str] | None) -> dict[str, str]:
        """Resolution order: explicit kwarg → env INFERA_KVD_HIPFILE_ROOTS → {}.

        Env format: ``"long=/mnt/long"`` — comma-separated
        ``retention=path`` pairs. SINGLE disk tier only: the spillover/short
        tier has been removed from the config surface, so any key other than
        ``long`` is dropped here (spillover is no longer configurable). The
        internal SpilloverRegion code is deleted in a follow-up.
        """
        if explicit is not None:
            roots = dict(explicit)
        else:
            env = os.environ.get("INFERA_KVD_HIPFILE_ROOTS", "").strip()
            if not env:
                return {}
            roots = {}
            for pair in env.split(","):
                pair = pair.strip()
                if not pair or "=" not in pair:
                    continue
                key, _, value = pair.partition("=")
                key = key.strip()
                value = value.strip()
                if key and value:
                    roots[key] = value
        # Single disk tier: drop any non-'long' root (e.g. the removed 'short'
        # spillover tier) so spillover can't be configured via the connector.
        dropped = [k for k in roots if k != "long"]
        if dropped:
            logger.warning(
                "INFERA_KVD_HIPFILE_ROOTS: ignoring removed tier(s) %s — only "
                "'long' (single disk tier) is supported now.",
                dropped,
            )
        return {k: v for k, v in roots.items() if k == "long"}

    def _local_chunk_path(self, kvd_key: bytes) -> Path | None:
        """Derive the on-disk chunk path purely from the content key and
        return it iff the file exists under any configured hipfile_root.

        This is the #46 / LMCache GdsBackend storage model: the connector
        owns the file tier and there is NO daemon index — the path is a
        pure function of (model, compat_key, kvd_key), so save, the
        availability probe, and load all recompute the same path
        independently. Layout matches the save side exactly:
        ``{root}/{h[:2]}/{h[2:4]}/<urlencoded(model|compat|b64key)>.kvcache``
        where ``h = sha256(composite)``.

        Returns None when no root is configured, the path helpers are
        unavailable (older daemon build), or no file exists under any
        root. Searching every configured retention root (long/short)
        avoids threading the per-chunk retention class through the load
        metadata — there are only a couple of roots, so the extra stats
        are negligible.
        """
        if _encode_composite is None or not self._hipfile_roots:
            return None
        composite = _encode_composite(self._model, self._compat_key, kvd_key)
        fname = _filename_for_composite(composite)
        hexed = _composite_hash(composite)
        rel = f"{hexed[:2]}/{hexed[2:4]}/{fname}.kvcache"
        seen: set[str] = set()
        for root_str in self._hipfile_roots.values():
            if not root_str or root_str in seen:
                continue
            seen.add(root_str)
            p = Path(root_str) / rel
            try:
                if p.is_file():
                    return p
            except OSError:
                continue
        return None

    def _register_mla_subspec(self, gid: int, present: list, block_size: int) -> None:
        """Register a uniform 3-D MLA/indexer sub-spec (num_kv_channels=1) for a
        DSA split. The tensor is a single [num_blocks, block_size(folded),
        hidden] co-packed blob, so the raw-byte gather/scatter round-trips it
        faithfully (byte+scale — see test_dsa_indexer_roundtrip)."""
        if not present:
            return
        sample = self._kv_caches[present[0]]
        shape = tuple(sample.shape)
        if len(shape) != 3:
            logger.warning(
                "_register_mla_subspec: gid %d layer %r shape %s is not the "
                "expected 3-D MLA layout — skipping",
                gid,
                present[0],
                shape,
            )
            return
        num_blocks = int(shape[0])
        hidden_dim = int(shape[2])
        self._group_kv_spec[gid] = {
            "layer_names": present,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "page_bytes": hidden_dim * block_size * sample.dtype.itemsize * len(present),
            "hidden_dim": hidden_dim,
            "num_kv_channels": 1,
            "dtype": sample.dtype,
            "dtype_str": _torch_dtype_to_str(sample.dtype),
        }
        for n in present:
            self._layer_to_group[n] = gid

    @staticmethod
    def _logical_block_size(group: Any, cfg: Any) -> int:
        """The LOGICAL KV block size for a cache group — the page size
        that the engine's block manager addresses ``block_ids`` in, and
        the value the scheduler-side chunk emit + the gather/scatter slot
        map (``block_id * block_size + offset``) both depend on.

        Resolution order mirrors the scheduler bootstrap EXACTLY so the
        two sides can never desync: ``kv_cache_spec.block_size`` →
        ``cfg.block_size`` → 16 (vLLM v1 DEFAULT_BLOCK_SIZE).

        IMPORTANT: this is NOT always the tensor's shape dim. For regular
        attention ``[2, num_blocks, block_size, …]`` the shape dim equals
        this, but some MLA backends (e.g. ROCM_AITER_MLA) lay the latent
        cache out as ``[num_blocks*block_size, 1, head]`` so the tensor's
        block dim is 1 while the logical block size is 16. Deriving
        block_size from the shape (the old behaviour) desynced the worker
        (1) from the scheduler (16): ``_flush_chunk_save`` then rejected
        100% of MLA chunks (``page_count × 1 ≠ chunk_tokens``) → L3 stayed
        empty → ext_hit=0%. It would ALSO have mis-indexed the gather.
        """
        spec = getattr(group, "kv_cache_spec", None)
        try:
            bs_spec = int(getattr(spec, "block_size", 0) or 0)
        except (TypeError, ValueError):
            bs_spec = 0
        try:
            bs_cfg = int(getattr(cfg, "block_size", 0) or 0)
        except (TypeError, ValueError):
            bs_cfg = 0
        return bs_spec or bs_cfg or 16

    def _bootstrap_group_kv_spec_from_config(self, kv_cache_config: Any) -> None:
        """Populate `self._group_kv_spec` from `kv_cache_config` alone
        (no torch tensors required). This runs on BOTH scheduler and
        worker so `build_connector_meta` (scheduler-only) can plan
        chunks without waiting for `register_kv_caches` (worker-only).

        Fields populated here: ``layer_names``, ``num_blocks``,
        ``block_size``. The worker's `register_kv_caches` augments
        with ``hidden_dim``, ``dtype``, ``dtype_str`` from real tensor
        shapes (needed at flush time but not at scheduler-emit time).

        Also populates `self._layer_to_group` so `_block_id_for_layer`
        works on both sides.

        Best-effort: when kv_cache_config is None or has no groups,
        this is a no-op and v2 chunked-fusion stays unavailable on
        this connector (fine for tests with stub configs; production
        always has groups).
        """
        cfg = kv_cache_config or getattr(self, "_kv_cache_config", None)
        if cfg is None:
            return
        groups = getattr(cfg, "kv_cache_groups", None)
        if not groups:
            return
        # block_size + num_blocks come from KVCacheConfig's per-group
        # kv_cache_spec (when present) or from the config's top-level
        # block_size. We grab whichever the engine exposes.
        cfg_num_blocks = int(getattr(cfg, "num_blocks", 0) or 0)
        for gid, group in enumerate(groups):
            layer_names = list(getattr(group, "layer_names", ()) or ())
            if not layer_names:
                continue
            spec = getattr(group, "kv_cache_spec", None)
            if _is_non_paged_kv_spec(spec):
                # Mamba / linear-attention / conv recurrent state — not a
                # paged attention KV cache. Skip on the scheduler side too
                # so build_connector_meta never plans chunks for it (must
                # mirror the worker's register_kv_caches skip).
                logger.info(
                    "_bootstrap_group_kv_spec_from_config: group %d (%s) "
                    "is a non-paged recurrent/conv-state group — skipping",
                    gid,
                    type(spec).__name__,
                )
                self._has_non_paged_groups = True
                continue
            # NOTE: a DSA mixed group is intentionally NOT split on the scheduler
            # side — it is registered here as a single group (the worker's
            # register_kv_caches does the main/indexer sub-spec split). Splitting
            # here too desyncs chunk_tokens (the per-sub-spec page_bytes cannot be
            # derived from the group's single spec.page_size_bytes) and produced
            # L3 key misses + garbage reload in GPU-E2E. The single-group scheduler
            # spec + worker-side split is the validated, correct configuration.
            # Logical block size — same resolution the worker uses, so the
            # two sides compute identical chunk_tokens (see
            # `_logical_block_size`).
            block_size = self._logical_block_size(group, cfg)
            num_blocks = int(getattr(spec, "num_blocks", 0) or 0) or cfg_num_blocks or 0
            # Bytes for ONE block across ALL of this group's layers.
            # `spec.page_size_bytes` is per-LAYER (K+V, one block); the
            # group has len(layer_names) layers. Used by chunk auto-sizing
            # — derivable here on the SCHEDULER from config alone, matching
            # the worker's tensor-derived value (same model).
            try:
                _per_layer_page = int(getattr(spec, "page_size_bytes", 0) or 0)
            except Exception:
                _per_layer_page = 0
            page_bytes = _per_layer_page * len(layer_names)
            self._group_kv_spec[gid] = {
                "layer_names": layer_names,
                "num_blocks": num_blocks,
                "block_size": block_size,
                "page_bytes": page_bytes,
                # hidden_dim/dtype get filled in by the worker's
                # register_kv_caches from the actual tensor shape.
                "hidden_dim": 0,
                "dtype": None,
                "dtype_str": "bf16",
            }
            for lname in layer_names:
                self._layer_to_group.setdefault(lname, gid)
        self._autosize_chunk_tokens()

    def _bound_metadata(self) -> InferaKvdConnectorMetadata | None:
        """Return the scheduler-bound metadata cast to our type, or None
        if nothing was bound for this step. The base class stores it
        under `self._connector_metadata` after `bind_connector_metadata`."""
        meta = getattr(self, "_connector_metadata", None)
        if isinstance(meta, InferaKvdConnectorMetadata):
            return meta
        return None

    # ------------------------------------------------------------------
    # vLLM-8 chunked-fusion save / load (v2 wire format)
    # ------------------------------------------------------------------

    def _add_pending_save_future(self, future: concurrent.futures.Future) -> None:
        """Track a submitted save future + opportunistically prune done
        ones so the list doesn't grow unbounded. The pending list is
        drained on close() or via drain_pending_saves() for
        benchmark-grade cold-vs-warm measurement."""
        with self._pending_save_futures_lock:
            self._pending_save_futures.append(future)
            if len(self._pending_save_futures) > 256:
                self._pending_save_futures = [f for f in self._pending_save_futures if not f.done()]

    def drain_pending_saves(self, *, timeout: float | None = None) -> int:
        """Block until all currently-submitted async saves complete.

        Returns the number of futures we waited on. Intended for
        benchmarks that need an honest cold-vs-warm boundary: after
        a warmup pass, call this so the next measured iter doesn't
        compete with warmup's tail of in-flight writes. Production
        traffic should NOT call this — async save is what gets the
        forward-pass overlap.

        Per LMCache's 2026-05 MI300X blog + their pq_executor.py
        (max_size=0, no back-pressure), competing systems don't
        expose this either; they bench with sustained load instead.
        We expose it because the cliff bench is short-duration and
        needs clean iter boundaries.
        """
        with self._pending_save_futures_lock:
            futures = list(self._pending_save_futures)
        if not futures:
            return 0
        done = concurrent.futures.wait(
            futures,
            timeout=timeout,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        # Prune so the list doesn't carry already-done futures forward.
        with self._pending_save_futures_lock:
            self._pending_save_futures = [f for f in self._pending_save_futures if not f.done()]
        return len(done.done)

    # ------------------------------------------------------------------
    # Active / silent gating — companion to vLLM's is_active() fast-path
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True if this connector has work to do this scheduler step.

        Companion to the optional vLLM-side is_active()-aware fast path
        (see kzjeef/vllm ``dev/jjzhang/kv-connector-is-active-fastpath``
        commit c635630, 5 small patches across scheduler.py + worker
        hooks). On a patched vLLM, returning False causes the scheduler
        to skip per-step metadata build + cross-process IPC/pickle of
        the empty payload.

        **Off by default** — the on-demand-save gate in
        ``build_connector_meta`` (skip_blocks based on
        ``num_computed_tokens``) already recovers ~95 % of the
        vanilla-vLLM scheduling cost at low c without needing the
        framework patch. is_active() opt-in adds ~5 % on top, only on
        a patched vLLM. Three opt-in modes via ``INFERA_KVD_ACTIVE_MODE``:

          - ``always`` (default): always True — vLLM-default behaviour.
            Safe on vanilla vLLM; safe on patched vLLM (no fast-path
            triggered, but no regression either).
          - ``silent``: always False — connector entirely silent. The
            scheduler skips the metadata path; the worker hooks never
            fire; save/load both disabled. Useful for benches isolating
            framework cost and for low-c deployments that registered
            kvd-aware but have no evicting workloads. Equivalent to
            running without ``--kv-transfer-config`` at all.
          - ``on_admission``: True only when ``_pending_block_hashes``
            is non-empty (admission step). False for chunked-prefill
            continuation and decode steps. Best low-c performance with
            cliff救援 retained.

        Legacy: ``INFERA_KVD_SILENT=1`` is honored as a synonym for
        ``INFERA_KVD_ACTIVE_MODE=silent``.

        On vanilla vLLM (no is_active patch) this method is never
        consulted — pure no-op regardless of mode.

        Thread-safety: ``_pending_block_hashes`` is mutated only on the
        scheduler thread; ``is_active()`` is also called on the
        scheduler thread, so no lock needed.
        """
        # Legacy SILENT env: equivalent to ACTIVE_MODE=silent.
        if os.environ.get("INFERA_KVD_SILENT", "").lower() in ("1", "true", "yes", "on"):
            return False
        mode = os.environ.get("INFERA_KVD_ACTIVE_MODE", "always").lower()
        if mode == "silent":
            return False
        if mode == "on_admission":
            return bool(self._pending_block_hashes)
        # Default "always": vLLM-base behaviour — never opt in to the
        # is_active() fast-path. On vanilla vLLM this is the same as not
        # overriding the method at all.
        return True

    # ------------------------------------------------------------------
    # In-flight + ring chunk cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_byte_size(raw: str, *, default: int) -> int:
        """Parse "2G" / "512M" / "1024K" / "1048576" into bytes. On any
        parse failure, return the supplied default."""
        if raw is None:
            return default
        s = str(raw).strip().upper()
        if not s:
            return default
        mult = 1
        if s.endswith("K"):
            mult, s = 1024, s[:-1]
        elif s.endswith("M"):
            mult, s = 1024**2, s[:-1]
        elif s.endswith("G"):
            mult, s = 1024**3, s[:-1]
        elif s.endswith("T"):
            mult, s = 1024**4, s[:-1]
        try:
            return int(float(s) * mult)
        except (ValueError, TypeError):
            return default

    def _cache_get(self, kvd_key: bytes) -> bytes | None:
        """Try in-flight chunks first, then the LRU ring. Returns the
        full blob (header + payload) ready to feed to ``unpack_chunk``,
        or None on miss."""
        if self._chunk_ring_max_bytes <= 0:
            return None
        with self._chunk_cache_lock:
            blob = self._in_flight_chunks.get(kvd_key)
            if blob is not None:
                self._chunk_cache_inflight_hits += 1
                return blob
            blob = self._chunk_ring.get(kvd_key)
            if blob is not None:
                self._chunk_ring.move_to_end(kvd_key)
                self._chunk_cache_ring_hits += 1
                return blob
            self._chunk_cache_misses += 1
            return None

    def _cache_has(self, kvd_key: bytes) -> bool:
        """Cheap presence check (no LRU touch, no stat counter
        update). Used by the hipfile-direct path to short-circuit
        before its lookup_tier RPC."""
        if self._chunk_ring_max_bytes <= 0:
            return False
        with self._chunk_cache_lock:
            return kvd_key in self._in_flight_chunks or kvd_key in self._chunk_ring

    def _register_inflight_chunk(
        self,
        kvd_key: bytes,
        blob: bytes,
        future: concurrent.futures.Future,
    ) -> None:
        """Track an in-flight save: load can serve from `blob` until
        the future completes (skipping kvd RPC), and on success the
        blob moves to the LRU ring for a few more seconds of fast
        re-read."""
        if self._chunk_ring_max_bytes <= 0:
            return
        with self._chunk_cache_lock:
            self._in_flight_chunks[kvd_key] = blob

        def _on_done(fut: concurrent.futures.Future) -> None:
            with self._chunk_cache_lock:
                self._in_flight_chunks.pop(kvd_key, None)
                if fut.exception() is not None:
                    return
                self._ring_put_unlocked(kvd_key, blob)

        future.add_done_callback(_on_done)

    def _ring_put_unlocked(self, kvd_key: bytes, blob: bytes) -> None:
        """Insert into ring buffer, evicting LRU until under cap. MUST
        be called with ``_chunk_cache_lock`` already held."""
        size = len(blob)
        if size > self._chunk_ring_max_bytes:
            # Single chunk larger than the whole budget — don't cache.
            return
        # Refresh recency if already present.
        if kvd_key in self._chunk_ring:
            self._chunk_ring.move_to_end(kvd_key)
            return
        while self._chunk_ring_cur_bytes + size > self._chunk_ring_max_bytes and self._chunk_ring:
            _, evicted = self._chunk_ring.popitem(last=False)
            self._chunk_ring_cur_bytes -= len(evicted)
        self._chunk_ring[kvd_key] = blob
        self._chunk_ring_cur_bytes += size

    def _acquire_save_slot(
        self,
        payload_bytes: int,
        torch_mod: Any,
        device: Any,
    ) -> tuple[int, Any] | None:
        """LMCache-GdsBackend-style bounded device staging allocator.

        Return ``(slot_idx, flat_uint8_tensor)`` from a fixed pool, or
        ``None`` when the chunk is larger than a slot or **no slot is
        free right now**. The caller soft-degrades to the host POSIX
        path when ``None`` — so device staging never exceeds the pool
        size and cannot OOM the engine. Lazily sizes the pool to the
        first chunk.

        CRITICAL: this runs in ``wait_for_save`` Phase 1, which is
        **synchronous and on vLLM's per-step engine critical path**.
        It must therefore be NON-BLOCKING: a busy-wait/backpressure
        sleep here stalls the whole engine step (no new prefills get
        scheduled) for as long as the slow VAST write takes to drain a
        slot — which wedges the engine entirely. So we try-acquire
        once and immediately degrade to the (VRAM-free) host POSIX
        path on miss. The bounded pool alone caps VRAM; we get the
        backpressure-against-OOM property without the backpressure
        stall. Slow-storage backpressure, if ever wanted, belongs in
        the async Phase-2 executor, never here.
        """
        if self._save_pool_bytes <= 0:
            return None
        _PAGE = 4096
        need = (int(payload_bytes) + _PAGE - 1) & ~(_PAGE - 1)
        with self._save_pool_lock:
            if not self._save_pool_slots:
                n = max(2, self._save_pool_bytes // need)
                try:
                    self._save_pool_slots = [
                        torch_mod.empty(need, dtype=torch_mod.uint8, device=device)
                        for _ in range(n)
                    ]
                except Exception:
                    logger.exception(
                        "save staging pool init failed — gpu_direct save "
                        "falls back to host POSIX path",
                    )
                    self._save_pool_slots = []
                    self._save_pool_bytes = 0  # disable; don't retry init
                    return None
                self._save_pool_free = list(range(n))
                self._save_pool_slot_bytes = need
                logger.info(
                    "kvd gpu_direct save staging pool: %d slots x %d MiB "
                    "= %d MiB (INFERA_KVD_SAVE_POOL_MB)",
                    n,
                    need >> 20,
                    (n * need) >> 20,
                )
            if need > self._save_pool_slot_bytes:
                return None  # chunk bigger than a slot — degrade
            # Non-blocking try-acquire (see docstring): NEVER sleep
            # here — this is the engine's synchronous per-step path.
            if self._save_pool_free:
                idx = self._save_pool_free.pop()
                return idx, self._save_pool_slots[idx]
        # Pool momentarily full: degrade this chunk to the host POSIX
        # path (no VRAM use, no engine stall). Counter-logged at debug
        # to avoid spamming the log under sustained save pressure.
        self._save_pool_misses += 1
        self._stat_inc("save_pool_misses", 1)
        if self._save_pool_misses & (self._save_pool_misses - 1) == 0:
            logger.info(
                "kvd save staging pool full — chunk #%d degrades to host "
                "POSIX save (non-blocking, no GPU OOM, no engine stall)",
                self._save_pool_misses,
            )
        return None

    def _release_save_slot(self, idx: int | None) -> None:
        if idx is None:
            return
        with self._save_pool_lock:
            self._save_pool_free.append(idx)

    def _acquire_pinned_slot(
        self,
        payload_bytes: int,
        torch_mod: Any,
    ) -> tuple[int, Any] | None:
        """Pinned-host analogue of _acquire_save_slot. Returns
        ``(slot_idx, flat_pinned_uint8_tensor)`` or ``None`` when the
        pool is momentarily full / the chunk exceeds a slot — caller then
        falls back to a one-off pinned alloc. Non-blocking (runs on the
        synchronous wait_for_save path). Pool bytes reuse
        ``self._save_pool_bytes`` budget."""
        if self._save_pool_bytes <= 0:
            return None
        _PAGE = 4096
        need = (int(payload_bytes) + _PAGE - 1) & ~(_PAGE - 1)
        # A chunk larger than the whole pinned budget can't be pooled without
        # blowing past the budget — degrade to a one-off pinned alloc.
        if need > self._save_pool_bytes:
            return None
        with self._pin_pool_lock:
            if not self._pin_pool_slots:
                # Honor the budget: n = budget // need (>=1 here since
                # need <= budget). Don't force n>=2 — that would allocate
                # 2*need > budget when the budget only fits one slot.
                n = max(1, self._save_pool_bytes // need)
                try:
                    self._pin_pool_slots = [
                        torch_mod.empty(need, dtype=torch_mod.uint8, device="cpu", pin_memory=True)
                        for _ in range(n)
                    ]
                except Exception:
                    logger.exception(
                        "pinned save pool init failed — per-chunk pinned alloc fallback"
                    )
                    self._pin_pool_slots = []
                    return None
                self._pin_pool_free = list(range(n))
                self._pin_pool_slot_bytes = need
                logger.info("kvd POSIX pinned staging pool: %d slots x %d MiB", n, need >> 20)
            if need > self._pin_pool_slot_bytes:
                return None
            if self._pin_pool_free:
                idx = self._pin_pool_free.pop()
                return idx, self._pin_pool_slots[idx]
        self._pin_pool_misses += 1
        return None

    def _release_pinned_slot(self, idx: int | None) -> None:
        if idx is None:
            return
        with self._pin_pool_lock:
            self._pin_pool_free.append(idx)

    def _prepare_chunk_save_on_stream(
        self,
        entry: tuple[tuple[tuple[int, ...], ...], bytes, str, int, list[str]],
        save_stream: Any | None,
    ) -> tuple[str, Any, tuple] | None:
        """Phase 1 of batched two-phase save (see wait_for_save docstring).

        Runs INSIDE the save_stream context: allocates device staging,
        runs Triton gather, and for POSIX/UDS-Set paths issues a
        non-blocking D2H copy to a pinned host buffer. Does NOT sync —
        caller batches all chunks then issues ONE save_stream.sync().

        Returns ``(kind, carrier, args)``:
          - kind = "posix" | "uds_set" | "gpu_direct"
          - carrier (POSIX/UDS): pinned-host torch.Tensor of bytes
          - carrier (gpu_direct): device staging torch.Tensor
          - args = tuple of metadata the matching async finisher needs

        Returns None on any prep failure (chunk is silently skipped).
        """
        # Keep legacy fast-path name accessible for tests / external
        # callers that want a non-batched single-chunk save.
        return self._enqueue_chunk_save_impl(entry, save_stream)

    def _enqueue_chunk_save_impl(
        self,
        entry: tuple[tuple[tuple[int, ...], ...], bytes, str, int, list[str]],
        save_stream: Any | None,
    ) -> tuple[str, Any, tuple] | None:
        from infera.engine.vllm.packed_format import (
            ChunkHeader,
            pack_chunk_header,
            pack_chunk_header_aligned,
        )
        from infera.engine.vllm.triton_kv_gather import kv_chunk_gather

        per_page_block_ids, kvd_key, retention, cache_group_id, layer_names = entry
        if not layer_names:
            return  # nothing to save

        # Resolve cache_group spec (set in register_kv_caches).
        spec = self._group_kv_spec.get(int(cache_group_id))
        if spec is None:
            logger.warning(
                "_flush_chunk_save: unknown cache_group_id=%s; skipping chunk key=%s",
                cache_group_id,
                kvd_key.hex()[:16],
            )
            return
        # Filter to layers actually registered on this worker — vLLM
        # may have already evicted or never allocated some.
        present = [n for n in layer_names if n in self._kv_caches]
        if not present:
            return
        layer_tensors = [self._kv_caches[n] for n in present]

        block_size = int(spec["block_size"])
        hidden_dim = int(spec["hidden_dim"])
        dtype_str = str(spec["dtype_str"])
        num_kv_channels = int(spec.get("num_kv_channels", 2))
        chunk_tokens = block_size * len(per_page_block_ids)
        if chunk_tokens != self._chunk_tokens:
            # Defensive: scheduler used a different chunk_tokens
            # configuration than the worker has. Skip rather than
            # write a malformed chunk.
            logger.warning(
                "_flush_chunk_save: chunk page count %d × block_size %d "
                "= %d tokens ≠ worker self._chunk_tokens %d; skipping",
                len(per_page_block_ids),
                block_size,
                chunk_tokens,
                self._chunk_tokens,
            )
            return

        # Build per-layer layer_to_group view (within this chunk's
        # group, all listed layers ARE in this group, so the kernel's
        # layer_to_group is uniform).
        num_layers = len(present)
        layer_to_group = [0] * num_layers  # all in the same group within one chunk

        # Allocate staging on the SAME device as the layers. Leading
        # dim is 2 for regular attention (K+V split), 1 for MLA
        # (single combined latent). For gpu_direct we take the staging
        # from a BOUNDED, ref-counted device pool (LMCache GdsBackend
        # pattern) so a burst of large-context saves can't OOM HBM; if
        # the pool is exhausted this chunk soft-degrades to the host
        # POSIX path below (its torch.empty is short-lived, freed after
        # the D2H copy, so it does not accumulate on device).
        import torch as _torch

        device = layer_tensors[0].device
        dtype = layer_tensors[0].dtype
        _shape = (num_kv_channels, num_layers, chunk_tokens, hidden_dim)
        _pool_idx: int | None = None
        if self._save_gpu_direct:
            _elt = _torch.empty(0, dtype=dtype).element_size()
            _payload_bytes = num_kv_channels * num_layers * chunk_tokens * hidden_dim * _elt
            _slot = self._acquire_save_slot(_payload_bytes, _torch, device)
            if _slot is not None:
                _pool_idx, _raw = _slot
                staging = _raw[:_payload_bytes].view(dtype).reshape(_shape)
            else:
                staging = _torch.empty(_shape, dtype=dtype, device=device)
        else:
            staging = _torch.empty(_shape, dtype=dtype, device=device)

        # Re-shape per_page_block_ids to per-page-within-this-group
        # (the entry's per-page tuples are already per-group; we
        # extract just THIS group's column).
        per_page_within_group = tuple(
            (page_ids[cache_group_id] if cache_group_id < len(page_ids) else page_ids[0],)
            for page_ids in per_page_block_ids
        )

        try:
            kv_chunk_gather(
                staging,
                layer_tensors,
                per_page_within_group,
                layer_to_group,
                block_size,
                use_triton=True,
            )
            # NOTE: no explicit cuda.synchronize() here. The
            # subsequent staging.cpu() in _flush_chunk_save_posix /
            # _flush_chunk_save_gpu_direct is implicitly synchronous
            # — it waits for any pending work on the default stream
            # that produces the source tensor. An extra synchronize
            # per chunk was costing ~50 ms each (MI355X testbed, Kimi K2.5
            # bench, 2026-06-02): on a 35-chunk request that's
            # ~1.8 s of forward-pass stall per request.
        except Exception:
            logger.exception(
                "_flush_chunk_save: gather kernel failed for key=%s gid=%d",
                kvd_key.hex()[:16],
                cache_group_id,
            )
            return

        # Build v2 header (cheap, sync).
        header = ChunkHeader(
            version=2,
            chunk_tokens=chunk_tokens,
            block_size=block_size,
            num_layers=num_layers,
            layer_names=tuple(present),
            hidden_dim=hidden_dim,
            dtype=dtype_str,
            cache_group_id=int(cache_group_id),
            num_kv_channels=num_kv_channels,
        )
        # When gpu_direct is on, pad header to 4 KiB so per-layer
        # cuFileReadAsync calls land on aligned file offsets. Adds
        # ~3.5 KiB of NULs per ~34 MiB chunk = 0.01 % overhead, lets
        # the consumer skip mmap+H2D entirely. payload_byte_offset
        # in the header tells the consumer where the payload starts.
        if self._save_gpu_direct:
            header_bytes, header = pack_chunk_header_aligned(header, align=4096)
        else:
            header_bytes = pack_chunk_header(header)

        # Single disk tier: all retention classes persist to the one
        # configured disk root (only 'long' survives _resolve_hipfile_roots).
        # retention is now only an eviction-priority hint, NOT a tier selector
        # — so requests without cache_control (retention='none') still cache to
        # disk instead of silently falling back to the RAM-only uds_set path.
        root_str = next(iter(self._hipfile_roots.values()), None) if self._hipfile_roots else None

        # For POSIX/UDS paths we need to D2H — issue a NON-BLOCKING
        # copy into a pinned-host tensor. wait_for_save batches the
        # sync across all chunks so only ONE save_stream.synchronize()
        # is paid per call instead of N implicit syncs via .cpu().
        def _stage_to_pinned_host() -> tuple[Any, int | None, int] | None:
            # D2H into a POOLED pinned slot (no per-chunk hipHostMalloc) and
            # return a flat uint8 view the executor writes straight from
            # (zero extra CPU copy). Falls back to a one-off pinned alloc
            # when the pool is full. Returns (uint8_view, pin_idx, payload_bytes).
            try:
                _elt = _torch.empty(0, dtype=dtype).element_size()
                pb = num_kv_channels * num_layers * chunk_tokens * hidden_dim * _elt
                slot = self._acquire_pinned_slot(pb, _torch)
                if slot is not None:
                    pin_idx, flat = slot
                    view = flat[:pb]
                else:
                    pin_idx = None
                    view = _torch.empty(pb, dtype=_torch.uint8, device="cpu", pin_memory=True)
                view.view(dtype).reshape(
                    num_kv_channels,
                    num_layers,
                    chunk_tokens,
                    hidden_dim,
                ).copy_(staging, non_blocking=True)
                return view, pin_idx, pb
            except Exception:
                logger.exception(
                    "_enqueue_chunk_save: pinned-host stage failed for key=%s",
                    kvd_key.hex()[:16],
                )
                return None

        if not root_str:
            # UDS Set path: stage to pinned host (non-blocking), wait_for_save
            # syncs the batch, executor finishes UDS Set.
            res = _stage_to_pinned_host()
            if res is None:
                return None
            pinned, pin_idx, _pb = res
            return ("uds_set", (pinned, pin_idx), (kvd_key, retention, header_bytes))

        try:
            from infera.kvd.ssd import (
                _composite_hash,
                _encode_composite,
                _filename_for_composite,
            )

            composite = _encode_composite(self._model, self._compat_key, kvd_key)
            fname = _filename_for_composite(composite)
            hexed = _composite_hash(composite)
            path = Path(root_str) / hexed[:2] / hexed[2:4] / f"{fname}.kvcache"
            path.parent.mkdir(parents=True, exist_ok=True)
            # C1 fix: include PID + random suffix so two engines sharing a
            # filesystem root + the same (model, compat_key) — DP replicas,
            # sidecar warmers, or two TP=1 servers — never compute the same
            # tmp inode. Without this, concurrent writers either tear the
            # POSIX file (publish-after-truncate race) or, in the GDS path,
            # interleave hipFile DMAs at the byte level → silent VRAM
            # corruption on the reader. The final published path
            # (`{path}`) is still content-keyed, so dedup / probe / load
            # all still hit the same file once it's `os.replace`-d.
            tmp = path.parent / (f"{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
        except (OSError, Exception):
            logger.exception(
                "_enqueue_chunk_save: path setup failed for key=%s",
                kvd_key.hex()[:16],
            )
            return None

        if self._save_gpu_direct and _pool_idx is not None:
            # GPU-direct: hipFile writes straight from the POOLED device
            # staging slot. The slot is returned to the pool inside
            # _async_gpu_direct_save once the write drains. The save_stream
            # sync (issued by wait_for_save after this returns) guarantees
            # the gather finished before any worker thread touches staging.
            return (
                "gpu_direct",
                staging,
                (header_bytes, header.payload_bytes, kvd_key, tmp, path, retention, _pool_idx),
            )

        # Non-gpu_direct OR gpu_direct with an exhausted staging pool:
        # default POSIX path — stage to pinned host, wait_for_save syncs
        # the batch once, executor does POSIX write + UDS register. (The
        # gpu_direct-degraded torch.empty staging is freed right after the
        # D2H copy below, so it never accumulates on device.)
        res = _stage_to_pinned_host()
        if res is None:
            return None
        pinned, pin_idx, pb = res
        return (
            "posix",
            (pinned, pin_idx),
            (kvd_key, header_bytes, pb, tmp, path, retention),
        )

    # ------------------------------------------------------------------
    # Phase-2 async finishers (run in self._save_executor worker threads)
    # ------------------------------------------------------------------

    def _async_uds_set_save(
        self,
        kvd_key: bytes,
        blob: bytes,
        retention: str,
    ) -> None:
        """Worker-thread phase 2: UDS Set the prepared blob into kvd's
        RAM tier (no file backing). Called when there's no
        hipfile_root configured for this retention class."""
        try:
            accepted, reason = self._run_async(
                self._client.set(
                    kvd_key,
                    blob,
                    retention=retention,
                    model=self._model,
                    compat_key=self._compat_key,
                )
            )
        except Exception:
            logger.exception(
                "_async_uds_set_save: UDS Set raised for key=%s",
                kvd_key.hex()[:16],
            )
            return
        if not accepted:
            logger.debug(
                "_async_uds_set_save: kvd refused UDS Set key=%s reason=%s",
                kvd_key.hex()[:16],
                reason,
            )

    def _fsync_published(self, path: Path) -> None:
        """C3 crash-safety: after os.replace(tmp, path), fdatasync the
        file (commits data + size to disk) and fsync the parent dir
        (commits the dir entry). No-op when INFERA_KVD_FSYNC_SAVE is
        unset. Failures are logged at debug — best-effort; the
        published file is already in the filesystem either way."""
        if not self._fsync_save:
            return
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fdatasync(fd)
            finally:
                os.close(fd)
        except OSError:
            logger.debug(
                "_fsync_published: fdatasync(%s) failed",
                path,
                exc_info=True,
            )
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            logger.debug(
                "_fsync_published: fsync(dir %s) failed",
                path.parent,
                exc_info=True,
            )

    def _async_posix_save(
        self,
        pinned_view: torch.Tensor,
        pin_idx: int | None,
        kvd_key: bytes,
        header_bytes: bytes,
        payload_bytes: int,
        tmp: Path,
        path: Path,
        retention: str,
    ) -> None:
        """Worker-thread phase 2 (POSIX path): ZERO-COPY write of header +
        payload straight from the pooled pinned buffer (os.write from its
        memoryview — no bytes() materialize, no header+payload concat),
        atomic rename, fsync, best-effort register. Always returns the
        pinned pool slot."""

        def _w(fd: int, buf) -> None:
            mv = memoryview(buf)
            off = 0
            while off < len(mv):
                try:
                    n = os.write(fd, mv[off:])
                except InterruptedError:
                    continue  # EINTR — retry the write
                if n <= 0:
                    raise OSError("short write: os.write returned 0")
                off += n

        total_size = len(header_bytes) + int(payload_bytes)
        try:
            try:

                def _write_and_publish() -> None:
                    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    try:
                        _w(fd, header_bytes)
                        _w(fd, memoryview(pinned_view.numpy()).cast("B"))
                    finally:
                        os.close(fd)
                    os.replace(tmp, path)

                # ENOSPC backstop: a full FS triggers an emergency reap +
                # one retry before giving up on this chunk (issue #55).
                self._publish_with_enospc_only(
                    fn=_write_and_publish,
                    want_free=total_size,
                )
            except OSError:
                logger.exception(
                    "_async_posix_save: file write/rename failed for key=%s",
                    kvd_key.hex()[:16],
                )
                return
        finally:
            self._release_pinned_slot(pin_idx)
        self._fsync_published(path)
        # Register with the L3 reaper for LRU + budget accounting.
        self._l3_register_save(str(path), total_size, retention=retention)
        self._register_or_fallback_uds_set(
            kvd_key=kvd_key,
            header_bytes=header_bytes,
            cpu_payload_bytes=None,
            path=path,
            file_total_size=total_size,
            retention=retention,
        )

    def _async_gpu_direct_save(
        self,
        staging: torch.Tensor,
        header_bytes: bytes,
        payload_bytes: int,
        kvd_key: bytes,
        tmp: Path,
        path: Path,
        retention: str,
        pool_idx: int | None = None,
    ) -> None:
        """Worker-thread phase 2 (GPU-direct path): hipFileWrite from
        the device staging tensor, with POSIX fallback if hipFile fails.
        ``pool_idx`` (if set) is the bounded-pool slot backing ``staging``;
        it is returned to the pool in ``finally`` once the write drains so
        the next save can reuse the HBM (backpressure release)."""
        try:
            ok = self._flush_chunk_save_gpu_direct(
                staging=staging,
                header_bytes=header_bytes,
                payload_bytes=payload_bytes,
                kvd_key=kvd_key,
                tmp=tmp,
                path=path,
            )
            if not ok:
                # GDS failed mid-write; fall back to POSIX (still in this
                # worker thread; we already have the device staging).
                if not self._flush_chunk_save_posix(
                    staging=staging,
                    header_bytes=header_bytes,
                    payload_bytes_expected=payload_bytes,
                    kvd_key=kvd_key,
                    tmp=tmp,
                    path=path,
                ):
                    return
            file_total_size = len(header_bytes) + payload_bytes
            # For the GPU-direct path the on-disk bytes are the same as
            # what would have landed via POSIX, but we don't have CPU bytes
            # in hand. The UDS-Set fallback (if register fails) re-reads
            # the file from disk — same behavior as before the refactor.
            self._register_or_fallback_uds_set(
                kvd_key=kvd_key,
                header_bytes=header_bytes,
                cpu_payload_bytes=None,
                path=path,
                file_total_size=file_total_size,
                retention=retention,
            )
        finally:
            # Release the pool slot AFTER the write has fully drained so
            # the device bytes aren't reused under an in-flight hipFileWrite.
            self._release_save_slot(pool_idx)

    def _register_or_fallback_uds_set(
        self,
        *,
        kvd_key: bytes,
        header_bytes: bytes,
        cpu_payload_bytes: bytes | None,
        path: Path,
        file_total_size: int,
        retention: str,
    ) -> None:
        """Save tail for the file tier (#46 / LMCache GdsBackend design).

        The on-disk file we just wrote IS the record: its path is a pure
        function of the content key (see ``_local_chunk_path``), so the
        availability probe and load recompute it without any daemon index.
        We therefore make register_file_entry a BEST-EFFORT call only when
        the client exposes it (builds that keep a daemon-side file tier
        for LRU); on main's tablespace daemon the method is absent and this
        is a clean no-op.

        We deliberately do NOT fall back to a UDS Set. The old fallback
        re-read the whole chunk from disk and pushed it into the daemon's
        RAM tier over the socket — that both defeats the GPU-direct read
        path (load would serve from RAM via UDS, ~25k tok/s instead of
        ~185k) and double-stores every chunk. ``cpu_payload_bytes`` is now
        unused; kept in the signature for call-site compatibility.
        """
        register = getattr(self._client, "register_file_entry", None)
        if register is None:
            # main's daemon has no file tier — the path is self-describing,
            # nothing to register.
            return
        try:
            self._run_async(
                register(
                    kvd_key,
                    model=self._model,
                    compat_key=self._compat_key,
                    path=str(path),
                    file_offset=0,
                    size=file_total_size,
                    version=0,
                    retention=retention,
                )
            )
        except (KvdConnectionError, KvdProtocolError, AttributeError) as exc:
            logger.debug(
                "_register_or_fallback_uds_set: best-effort register failed "
                "for key=%s (%s) — file path is self-describing, ignoring",
                kvd_key.hex()[:16],
                exc,
            )

    def _flush_chunk_save_posix(
        self,
        *,
        staging: torch.Tensor,
        header_bytes: bytes,
        payload_bytes_expected: int,
        kvd_key: bytes,
        tmp: Path,
        path: Path,
        retention: str = "long",
    ) -> bool:
        """POSIX path for chunk save: D2H the device staging buffer to
        CPU bytes, concatenate with header, single POSIX `write_bytes`
        then `os.replace` for atomic publish. Returns True on success.
        """
        import torch as _torch

        try:
            cpu_payload = staging.cpu().view(_torch.uint8).numpy().tobytes()
        except Exception:
            logger.exception(
                "_flush_chunk_save (POSIX): D2H failed for key=%s",
                kvd_key.hex()[:16],
            )
            return False
        if len(cpu_payload) != payload_bytes_expected:
            logger.warning(
                "_flush_chunk_save (POSIX): payload size %d != expected %d for key=%s — skipping",
                len(cpu_payload),
                payload_bytes_expected,
                kvd_key.hex()[:16],
            )
            return False
        total_bytes = len(header_bytes) + len(cpu_payload)
        if not self._publish_with_enospc_retry(
            write_fn=lambda: tmp.write_bytes(header_bytes + cpu_payload),
            replace_fn=lambda: os.replace(tmp, path),
            path=path,
            tmp=tmp,
            kvd_key=kvd_key,
            payload_bytes=total_bytes,
            label="POSIX",
        ):
            return False
        self._fsync_published(path)
        self._stat_inc("saved_chunks", 1)
        self._stat_inc("saved_bytes", total_bytes)
        # Register with the L3 reaper for LRU + budget accounting.
        self._l3_register_save(path, total_bytes, retention=retention)
        return True

    def _flush_chunk_save_gpu_direct(
        self,
        *,
        staging: torch.Tensor,
        header_bytes: bytes,
        payload_bytes: int,
        kvd_key: bytes,
        tmp: Path,
        path: Path,
        retention: str = "long",
    ) -> bool:
        """GPU-direct path for chunk save: header bytes via POSIX
        (small, <1 KiB), payload via `hipFileWrite` straight from the
        device staging tensor. Skips the D2H copy + Python bytes
        materialization. Returns True on success; False on any
        failure (caller falls back to POSIX so the chunk isn't lost).

        Implementation:
          1. POSIX-create the tmp file, write the small header at
             offset 0, then `ftruncate` to total_size so the
             subsequent hipFile write lands in pre-allocated blocks.
          2. Register the device staging buffer with hipFile.
          3. Open the file via the shim's HipFile context manager in
             "r+" mode (binding's `open` reopens the existing extended
             file); call `hf.write(buf, payload_bytes, file_offset=
             header_size, buffer_offset=0)`.
          4. Atomic rename into final path.

        cuFile/hipFile requires page-aligned source buffer; we
        round-down to 4 KiB and pass the prefix as `buffer_offset`.
        Same alignment dance as the legacy v1 hipFile path
        (now-deleted), kept localized here.
        """
        try:
            from infera.engine.sglang.hipfile_shim import (
                HipFile,
                RegisteredBuffer,
            )
        except ImportError:
            return False

        _PAGE = 4096
        # Buffer alignment: hipFile's BufRegister wants page-aligned
        # base. Round DOWN; extend size to cover prefix + actual size,
        # then round UP to page.
        base_ptr = int(staging.data_ptr())
        nbytes = int(staging.numel()) * int(staging.element_size())
        if nbytes != payload_bytes:
            logger.warning(
                "_flush_chunk_save (GDS): staging bytes=%d != expected %d for key=%s — skipping",
                nbytes,
                payload_bytes,
                kvd_key.hex()[:16],
            )
            return False
        prefix = base_ptr & (_PAGE - 1)
        registered_base = base_ptr - prefix
        registered_size = (nbytes + prefix + _PAGE - 1) & ~(_PAGE - 1)

        # 1. POSIX-create + write header + ftruncate.
        total_size = len(header_bytes) + payload_bytes
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                view = memoryview(header_bytes)
                written = 0
                while written < len(view):
                    n = os.write(fd, view[written:])
                    if n <= 0:
                        raise OSError("short write on chunk header")
                    written += n
                os.ftruncate(fd, total_size)
            finally:
                os.close(fd)
        except OSError as exc:
            logger.warning(
                "_flush_chunk_save (GDS): header/truncate failed for "
                "key=%s tmp=%s (%s) — falling back to POSIX",
                kvd_key.hex()[:16],
                tmp,
                exc,
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

        # 2-3. Register staging + hipFileWrite payload.
        try:
            with RegisteredBuffer(registered_base, registered_size) as reg:
                buf = reg.handle
                if buf is None:
                    raise RuntimeError("registered buffer handle is None")
                with HipFile(str(tmp), "r+") as hf:
                    n = hf.write(
                        buf,
                        int(payload_bytes),
                        int(len(header_bytes)),  # file_offset = past header
                        int(prefix),  # buffer_offset = alignment prefix
                    )
                    if int(n) != int(payload_bytes):
                        raise OSError(f"short hipFileWrite: wrote {n} of {payload_bytes}")
        except Exception as exc:
            logger.warning(
                "_flush_chunk_save (GDS): hipFile write failed for key=%s "
                "tmp=%s (%s) — falling back to POSIX",
                kvd_key.hex()[:16],
                tmp,
                exc,
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

        # 4. Atomic rename — with ENOSPC backstop. If the rename fails
        # with ENOSPC, kick the reaper to free space and retry once.
        total_size = len(header_bytes) + payload_bytes
        try:
            self._publish_with_enospc_only(
                fn=lambda: os.replace(str(tmp), str(path)),
                want_free=total_size,
            )
        except OSError as exc:
            logger.warning(
                "_flush_chunk_save (GDS): rename failed for key=%s tmp=%s path=%s (%s)",
                kvd_key.hex()[:16],
                tmp,
                path,
                exc,
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        self._fsync_published(path)
        self._stat_inc("saved_chunks", 1)
        self._stat_inc("saved_bytes", total_size)
        # Register with the L3 reaper for LRU + budget accounting.
        self._l3_register_save(str(path), total_size, retention=retention)
        return True

    # ------------------------------------------------------------------
    # L3 reaper integration helpers
    # ------------------------------------------------------------------

    def _l3_register_save(self, path: Any, size_bytes: int, *, retention: str) -> None:
        """Tell the reaper a new chunk file has been published. No-op
        if the reaper isn't configured (UDS-Set fallback path)."""
        reaper = getattr(self, "_l3_reaper", None)
        if reaper is None:
            return
        try:
            reaper.register(str(path), int(size_bytes), retention=retention)
        except Exception:
            logger.exception("l3 reaper register failed for %s", path)

    def _l3_touch_load(self, path: Any) -> None:
        """LRU-on-read: bump mtime of an entry that was just loaded so
        hot chunks survive eviction. No-op without a reaper."""
        reaper = getattr(self, "_l3_reaper", None)
        if reaper is None:
            return
        try:
            reaper.touch(str(path))
        except Exception:
            logger.exception("l3 reaper touch failed for %s", path)

    def _publish_with_enospc_retry(
        self,
        *,
        write_fn,
        replace_fn,
        path: Any,
        tmp: Any,
        kvd_key: bytes,
        payload_bytes: int,
        label: str,
    ) -> bool:
        """Run (write → replace) with one ENOSPC retry after a forced
        reap. Returns True on publish, False on giving up. Used by the
        POSIX save path; the GDS path uses `_publish_with_enospc_only`
        for the rename since its write step is hipFile-driven."""
        for attempt in (1, 2):
            try:
                write_fn()
                replace_fn()
                return True
            except OSError as exc:
                if exc.errno != errno.ENOSPC or attempt == 2:
                    logger.exception(
                        "_flush_chunk_save (%s): file write failed for key=%s",
                        label,
                        kvd_key.hex()[:16],
                    )
                    try:
                        Path(tmp).unlink(missing_ok=True)
                    except OSError:
                        pass
                    return False
                # ENOSPC + first attempt → trigger an emergency reap.
                reaper = getattr(self, "_l3_reaper", None)
                freed = 0
                if reaper is not None:
                    try:
                        freed = reaper.on_enospc(want_free_bytes=payload_bytes)
                    except Exception:
                        logger.exception("l3 reaper on_enospc raised")
                logger.warning(
                    "_flush_chunk_save (%s): ENOSPC for key=%s, reaped %d bytes, retrying",
                    label,
                    kvd_key.hex()[:16],
                    freed,
                )
        return False

    def _publish_with_enospc_only(self, *, fn, want_free: int) -> None:
        """Run ``fn`` once; on ENOSPC, force a reap and retry once. Any
        other OSError (including a second ENOSPC) propagates so the
        caller can do its normal logging + cleanup."""
        try:
            fn()
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            reaper = getattr(self, "_l3_reaper", None)
            if reaper is not None:
                try:
                    reaper.on_enospc(want_free_bytes=want_free)
                except Exception:
                    logger.exception("l3 reaper on_enospc raised")
            fn()  # retry; propagates if still failing

    def _fetch_chunk_blob(
        self,
        kvd_key: bytes,
    ) -> tuple[memoryview | bytes | None, Any, str | None]:
        """Look up a chunk by kvd_key and return (blob, mm_holder, file_path).

        Returns ``(memoryview/bytes blob, mmap | None mm_holder, str | None file_path)``
        on hit, ``(None, None, None)`` on miss/error. Caller MUST keep
        mm_holder alive until the blob bytes are fully consumed (Triton
        scatter synchronized) — memoryview points into the mmap'd pages.

        ``file_path`` is the on-disk file path when the chunk is in
        file-tier (mm_holder is also non-None). For UDS Get / RAM tier
        the path is None. Used by the layerwise B hipFile-async path
        to open a hipFile.FileHandle for cuFileReadAsync.

        Used by both `_load_chunk_packed` (current sync path) and
        the layerwise A/B drafts.
        """
        # In-flight + ring cache shortcut: if this chunk is still in
        # the host pinned blob from a recent save (or in the LRU ring
        # of recently-completed saves), serve those bytes directly.
        # Closes the load↔save race window without any drain wait,
        # AND avoids the kvd RPC + disk read for hot chunks.
        cached = self._cache_get(kvd_key)
        if cached is not None:
            return memoryview(cached), None, None
        blob: bytes | memoryview | None = None
        # File-tier path derived from the content key (#46 / LMCache
        # GdsBackend design): mmap the on-disk chunk directly — no daemon
        # lookup_tier RPC. When no file exists under a configured root we
        # leave blob None and fall through to UDS Get (covers the RAM-tier
        # / no-hipfile_root configuration).
        path_obj = self._local_chunk_path(kvd_key)
        mm_holder = None
        file_path: str | None = None
        if path_obj is not None:
            import mmap as _mmap

            try:
                size = path_obj.stat().st_size
                fd = os.open(str(path_obj), os.O_RDONLY)
                try:
                    mm_holder = _mmap.mmap(fd, size, access=_mmap.ACCESS_READ)
                finally:
                    os.close(fd)
                blob = memoryview(mm_holder)
                file_path = str(path_obj)
            except OSError as exc:
                logger.warning(
                    "_fetch_chunk_blob: mmap failed for key=%s path=%s (%s) "
                    "— falling back to UDS Get",
                    kvd_key.hex()[:16],
                    path_obj,
                    exc,
                )
                blob = None
                mm_holder = None
                file_path = None

        if blob is None:
            try:
                blob = self._run_async(
                    self._client.get(
                        kvd_key,
                        model=self._model,
                        compat_key=self._compat_key,
                    )
                )
            except Exception:
                logger.exception(
                    "_fetch_chunk_blob: UDS Get raised for key=%s",
                    kvd_key.hex()[:16],
                )
                return None, None, None
            if blob is None:
                logger.debug(
                    "_fetch_chunk_blob: kvd miss on key=%s",
                    kvd_key.hex()[:16],
                )
                return None, None, None
            blob = memoryview(blob)
        return blob, mm_holder, file_path

    # ------------------------------------------------------------------
    # Layerwise load Variant A — per-layer Triton scatter on default
    # stream. H2D + CPU staging still done up-front in start_load_kv.
    # ------------------------------------------------------------------

    # Per-load-thread persistent state. Each worker thread of the
    # _load_executor pool keeps a SINGLE device buffer + hipFile
    # registration alive across chunks — register/deregister + tensor
    # alloc only happen on first call (or on grow).
    #
    # LIFECYCLE: TLS state is created lazily in
    # _get_parallel_load_state on a worker's first read. It MUST be
    # explicitly torn down (via _cleanup_parallel_load_tls running on
    # each worker thread) BEFORE _load_executor.shutdown completes —
    # otherwise the device tensor (held by the TLS dict) is GC'd when
    # the thread exits while hipFile still has the buffer REGISTERED
    # at that address; later driver cleanup then touches freed device
    # memory and SIGSEGVs.
    #
    # KNOWN BUG (no-P2PDMA hosts only — RDMA worker please retest):
    # with N>=4 worker threads in the load executor we see
    # intermittent "Memory access fault by GPU node-X on address
    # 0xXXXXXXe00000 Unknown" during the load loop. N=1 and N=2
    # stable. 5/5 fault rate at N=8 on the MI355X testbed. Adding the setup
    # lock (Buffer.register / FileHandle.open) did NOT fix it, so
    # the race is in the IO path, not the registration table.
    #
    # ROOT CAUSE traced into libhipfile.so source (vendored at
    # /opt/rocm-systems-src/projects/hipfile/):
    #   - state.cpp uses 21 mutex calls — BufferMap/FileMap/StreamMap
    #     are thread-safe.
    #   - But src/amd_detail/backend/fallback.cpp (the CPU-bounce
    #     Fallback backend used when P2PDMA is not available) has
    #     ZERO mutex calls. Its _io_impl does:
    #       1. per-call mmap anonymous bounce buffer
    #       2. pread(fd, bounce_buffer, ...) — per-call fd, OK
    #       3. hipMemcpy(device_ptr, bounce_buffer, ..., H2D)
    #          ← race: bounce_buffer is NOT pinned (just anon mmap),
    #            so hipMemcpy internally stages through a pinned
    #            page pool. Concurrent unpinned H2Ds from N threads
    #            race the driver's pinned-page manager, causing the
    #            fault on the destination device address (the
    #            registered persistent buffer).
    #   - src/amd_detail/backend/fastpath.cpp (P2PDMA path) also has
    #     0 mutex calls but its DMA is driven by the kernel P2PDMA
    #     subsystem, NOT hipMemcpy → likely race-free.
    #
    # Implication: on hosts WITH CONFIG_PCI_P2PDMA + NVMe/RNIC
    # support (any modern RDMA box), the fastpath backend is
    # selected and N=8 should be stable. The MI355X testbed (no P2PDMA → forced
    # to Fallback) is the only environment hitting this.
    #
    # Workaround on no-P2PDMA hosts: INFERA_KVD_LOAD_WORKERS=1 or 2.
    # RDMA worker: N-sweep at 1/2/4/8/16 — should see ~28 GB/s plateau
    # at N=4-8 per the microbench predictions, no faults.
    #
    # Reproduce on a no-P2PDMA host:
    #   INFERA_KVD_LAYERWISE_LOAD=parallel
    #   INFERA_KVD_LOAD_WORKERS=8 INFERA_KVD_GPU_DIRECT=1
    #   infera-kvd-l3-bench --dir /mnt/nvme8/... --transport gpu-direct \
    #     --layout mla --hidden-dim 576 --chunk-tokens 512 --workers 8
    #   (the connector round-trip bench that absorbed bench_packed_v2)
    _PARALLEL_LOAD_TLS = threading.local()

    # Ring of N distinct staging sub-regions per worker thread (LMCache
    # GDS-style: each chunk DMA lands in its own non-overlapping slice of
    # one registered pool). With N>=2 the next chunk's hipFile read goes
    # to a DIFFERENT slot, so it can run while the current chunk's async
    # Triton scatter is still consuming its slot — no per-chunk
    # full-device sync needed. A per-slot cuda.Event gates reuse: before
    # writing slot i again we wait on ONLY that slot's prior scatter
    # event (cheap; it ran N chunks ago), not the whole engine. This
    # replaces the current_stream().synchronize() that serialized the
    # engine (~7x throughput cliff). Override slot count with
    # INFERA_KVD_LOAD_RING_SLOTS (default 3).
    try:
        _LOAD_RING_SLOTS = max(1, int(os.environ.get("INFERA_KVD_LOAD_RING_SLOTS", "3")))
    except ValueError:
        _LOAD_RING_SLOTS = 3

    # Serialize hipFile setup-side calls (Buffer.register/deregister,
    # FileHandle.open/close) across worker threads. Concurrent
    # register + open from N threads triggers the libhipfile.so
    # internal-table race that crashes with "Memory access fault by
    # GPU node-X on address 0xXXXXXXe00000 Unknown" at N>=4. The
    # actual data-plane call (FileHandle.read) runs OUTSIDE this
    # lock so N threads still fan out their reads — only the few
    # microseconds of open + register get serialized.
    _HIPFILE_SETUP_LOCK = threading.Lock()

    # Cache the ais-check probe across connector instances — it's
    # the same machine, the answer doesn't change at runtime.
    _P2PDMA_PROBE_RESULT: bool | None = None
    _P2PDMA_PROBE_LOCK = threading.Lock()

    # Run the L3 storage self-check at most once per process (one connector
    # owns L3 I/O per engine; PD spawns one process per prefill/decode engine,
    # each of which logs its own line).
    _SELFCHECK_DONE = False
    _SELFCHECK_LOCK = threading.Lock()

    @staticmethod
    def _detect_l3_nconnect() -> int | None:
        """Best-effort: the ``nconnect=`` of the NFS mount backing the L3
        hipfile root, so we can default the I/O worker pools to the mount's
        connection count (NFS-over-RDMA load scales near-linearly up to
        nconnect, then plateaus).
        Returns the int, or None for a non-NFS / no-nconnect mount."""
        import re as _re

        roots = os.environ.get("INFERA_KVD_HIPFILE_ROOTS", "")
        paths = [kv.split("=", 1)[1].strip() for kv in roots.split(",") if "=" in kv]
        if not paths:
            return None
        try:
            target = os.path.realpath(paths[0])
            best_mp = ""
            best_opts = ""
            with open("/proc/self/mountinfo") as fh:
                for line in fh:
                    p = line.split()
                    if "-" not in p:
                        continue
                    mp = p[4]
                    sep = p.index("-")
                    opts = p[sep + 3] if len(p) > sep + 3 else ""
                    if (
                        target == mp or target.startswith(mp.rstrip("/") + "/") or mp == "/"
                    ) and len(mp) >= len(best_mp):
                        best_mp, best_opts = mp, opts
            m = _re.search(r"\bnconnect=(\d+)", best_opts)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    def _maybe_run_connector_selfcheck(self, save_workers: int, load_workers: int) -> None:
        """Log an L3 storage self-check (write/read GB/s) for the connector's
        own configured root, using the connector's resolved save/load workers,
        gpu_direct, and P2PDMA verdict.

        Fires once per process on WORKER-role rank 0 only — that is the rank
        that actually performs chunk I/O, and the guard yields exactly one line
        per engine. Under PD each prefill/decode engine is a separate process,
        so each emits its own self-check reflecting the config it will use.
        Best-effort: any failure is swallowed (never blocks engine startup).
        Disable with INFERA_KVD_STORAGE_SELFCHECK=0."""
        try:
            if not self._hipfile_roots:
                return  # no on-disk L3 tier → nothing to probe
            # WORKER role owns chunk I/O; SCHEDULER never touches the files.
            role_worker = getattr(self._role, "value", self._role)
            worker_role = getattr(KVConnectorRole, "WORKER", 1)
            worker_role = getattr(worker_role, "value", worker_role)
            if role_worker != worker_role:
                return
            if self._tp_rank != 0:
                return  # one probe per engine, not per TP rank
            with type(self)._SELFCHECK_LOCK:
                if type(self)._SELFCHECK_DONE:
                    return
                type(self)._SELFCHECK_DONE = True
            from infera.kvd.storage_selfcheck import run_storage_selfcheck

            root = next(iter(self._hipfile_roots.values()), None)
            if not root:
                return
            p2p = self._detect_p2pdma_support()
            extra = (
                f"gpu_direct={'on' if self._gpu_direct else 'off'} "
                f"p2pdma={'yes' if p2p else 'no'}"
                + ("" if (p2p or load_workers > 1) else " (load clamped to 1: no P2PDMA)")
            )
            run_storage_selfcheck(
                root,
                write_workers=save_workers,
                read_workers=load_workers,
                label="L3 connector",
                extra=extra,
            )
        except Exception as exc:  # never block engine startup
            logger.warning("[kvd] connector storage self-check wiring error (non-fatal): %s", exc)

    def _resolve_io_workers(self, env_name: str) -> int:
        """Resolve a load/save worker count. Explicit env wins; otherwise
        ``auto`` → the L3 mount's nconnect, falling back to 16 when it
        can't be detected (non-NFS / parse failure)."""
        v = os.environ.get(env_name, "auto").strip().lower()
        if v not in ("", "auto"):
            try:
                return max(1, int(v))
            except ValueError:
                pass
        nc = self._detect_l3_nconnect()
        return nc if (nc and nc > 0) else 16

    @classmethod
    def _detect_p2pdma_support(cls) -> bool:
        """Best-effort detect kernel P2PDMA support via the
        ``ais-check`` tool that ships with the hipFile install (under
        /opt/rocm/bin/). Returns True iff stdout reports ``amdgpu: True``
        (or, as a fallback, ``Kernel P2PDMA support: True``). Returns False on any error
        (binary missing, permission denied, timeout) — the safe
        default is "no P2PDMA, force LOAD_WORKERS=1" because the
        multi-thread Fallback path races (see KNOWN BUG above).

        Cached across calls (one subprocess invocation per process)
        so the connector init doesn't keep re-running it.
        """
        if cls._P2PDMA_PROBE_RESULT is not None:
            return cls._P2PDMA_PROBE_RESULT
        with cls._P2PDMA_PROBE_LOCK:
            if cls._P2PDMA_PROBE_RESULT is not None:
                return cls._P2PDMA_PROBE_RESULT
            import subprocess

            try:
                out = subprocess.run(
                    ["ais-check"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                text = (out.stdout or "") + (out.stderr or "")
                # ais-check pads the label (e.g. "Kernel P2PDMA support   : True"
                # — tab/spaces before the colon), so match whitespace-tolerantly
                # instead of requiring an exact substring.
                import re as _re

                # Key off the `amdgpu:` line, NOT `Kernel P2PDMA support:`.
                # The latter is unreliable: on driver builds where AIS is
                # initialized (dmesg "AIS: registered ...MB device memory" per
                # GPU) ais-check still reports `Kernel P2PDMA support: False`,
                # because it reads the KFD `capability & 0x40` bit, which these
                # builds do not set. Gating on it makes the connector refuse
                # GPU-direct on hosts where hipFile DMA actually works —
                # measured: gpu_direct=True load path at 20.8 GB/s, zero
                # fallbacks, on a host whose `Kernel P2PDMA support` reads False.
                # `amdgpu: True` is the line that tracks the real driver AIS
                # state and matches runtime behavior (it agrees host and
                # in-container, unlike the P2PDMA line which flips on a missing
                # `dkms`). Accept either, so a genuinely P2PDMA-capable host
                # that reports the old way still passes.
                result = bool(
                    _re.search(r"amdgpu\s*:\s*True", text)
                    or _re.search(r"Kernel\s+P2PDMA\s+support\s*:\s*True", text)
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                # ais-check not on PATH or hung — assume no P2PDMA.
                result = False
            cls._P2PDMA_PROBE_RESULT = result
            logger.info(
                "P2PDMA probe via ais-check: %s",
                "supported" if result else "NOT supported (force single-worker load)",
            )
            return result

    @classmethod
    def _safe_buf_dereg(cls, state: dict) -> None:
        buf = state.get("buf") if state else None
        if buf is not None:
            try:
                with cls._HIPFILE_SETUP_LOCK:
                    buf.deregister()
            except Exception:
                pass

    @classmethod
    def _cleanup_parallel_load_tls(cls) -> None:
        """Per-worker-thread cleanup: deregister this thread's
        persistent hipFile buffer + drop the TLS slot. Idempotent
        (no-op if no state has been initialized on this thread).
        Called from close() via N submits to the executor so each
        worker hits its own TLS exactly once before shutdown."""
        st = getattr(cls._PARALLEL_LOAD_TLS, "state", None)
        if st is None:
            return
        cls._safe_buf_dereg(st)
        cls._PARALLEL_LOAD_TLS.state = None

    @classmethod
    def _get_parallel_load_state(
        cls,
        device: Any,
        payload_nbytes: int,
        page: int = 4096,
    ) -> dict | None:
        """Return per-thread persistent RING of N hipFile-registered
        staging slots, each sized to at least ``payload_nbytes``. One big
        device tensor is registered once and carved into N non-overlapping
        slots (LMCache GDS pool-of-distinct-offsets pattern). Grows by
        doubling on demand and re-registers; otherwise reuses. Returns
        None on hipfile import failure.

        Keys: raw_dev, buf, slot_cap (per-slot bytes), n_slots,
        align_prefix, events (per-slot cuda.Event or None), slot_idx.
        """
        try:
            import ctypes as _ct

            import hipfile as _hipfile
            import torch as _torch
        except ImportError:
            return None

        n_slots = cls._LOAD_RING_SLOTS
        st = getattr(cls._PARALLEL_LOAD_TLS, "state", None)
        # Per-slot capacity: doubling growth, minimum 64 MiB so small
        # chunks don't thrash and we have headroom for the typical
        # Kimi-MLA ~33 MiB case.
        min_cap = max(payload_nbytes, 64 * (1 << 20))
        if (
            st is None
            or st["slot_cap"] < payload_nbytes
            or st["n_slots"] != n_slots
            or st["device"] != str(device)
        ):
            # Free prior registration before allocating new one — the
            # old device tensor would be GC'd anyway but if hipFile
            # still has it registered, the next register at the same
            # data_ptr address (PyTorch caching allocator reuses)
            # races into "already registered" 5023.
            if st is not None:
                cls._safe_buf_dereg(st)
            slot_cap = 1
            while slot_cap < min_cap:
                slot_cap *= 2
            total = slot_cap * n_slots
            raw_dev = _torch.empty(total + page, dtype=_torch.uint8, device=device)
            raw_ptr = int(raw_dev.data_ptr())
            align_prefix = (-raw_ptr) & (page - 1)
            aligned_ptr = raw_ptr + align_prefix
            # Register the FULL N-slot pool once — every chunk slices its
            # own slot at offset (slot_idx * slot_cap), no re-register.
            # slot_cap is a power of two >= 64 MiB so each slot start is
            # 4 KiB-aligned (cuFile requirement). Serialize register call
            # across worker threads to dodge the libhipfile.so
            # multi-thread race; see _HIPFILE_SETUP_LOCK docstring.
            buf = _hipfile.Buffer.from_ctypes_void_p(
                _ct.c_void_p(aligned_ptr),
                total,
                0,
            )
            try:
                with cls._HIPFILE_SETUP_LOCK:
                    buf.register()
            except Exception:
                logger.exception(
                    "_get_parallel_load_state: Buffer.register failed "
                    "for total=%d (%d slots x %d) on device=%s",
                    total,
                    n_slots,
                    slot_cap,
                    device,
                )
                return None
            st = {
                "raw_dev": raw_dev,
                "buf": buf,
                "slot_cap": slot_cap,
                "n_slots": n_slots,
                "align_prefix": align_prefix,
                "events": [None] * n_slots,
                "slot_idx": 0,
                "device": str(device),
            }
            cls._PARALLEL_LOAD_TLS.state = st
        return st

    # Cap how much we read off the head of a chunk file to find the
    # header. Header is msgpack-encoded — a few hundred bytes for
    # typical KV configs (61 layers × ~30 chars/layer-name + scalars
    # comes to ~3 KB). 64 KB is overkill but matches one page set.
    _CHUNK_HEADER_PREFETCH_BYTES = 64 * 1024

    def _load_chunk_packed_hipfile_direct(
        self,
        entry: tuple[tuple[tuple[int, ...], ...], bytes, int, list[str]],
    ) -> None:
        """GPU-direct fast path for parallel-mode load:

          1. UDS lookup_tier → file_path
          2. os.pread of the first 64 KB → parse header (no mmap, no
             page-cache pollution; payload bytes NEVER touched on CPU)
          3. Acquire per-thread persistent device buffer + hipFile
             registration (register-once per thread, reused across
             chunks)
          4. ONE coalesced hipFile.read of the whole payload directly
             into the device buffer (storage → GPU DMA on P2PDMA hosts,
             CPU bounce → GPU on others — both release the GIL so
             N=8 threads fan out)
          5. Reshape (zero-copy) + Triton scatter device→KV-cache

        No mmap, no CPU memcpy of payload, no per-chunk register/
        deregister, no per-chunk alloc. The only per-chunk costs are:
          - 1× UDS RPC (~50us)
          - 1× os.open + pread + close for header (~50us)
          - 1× hipFile open + read + close (open ~10us, read = bulk)
          - 1× Triton scatter (~5ms)

        Falls back to _load_chunk_packed (mmap+H2D) on:
          - chunk in RAM tier (no file_path)
          - payload offset not 4 KiB-aligned (v2 legacy chunks)
          - hipFile binding not importable or register/read raises
        """
        from infera.engine.vllm.packed_format import (
            PackedFormatError,
            unpack_chunk,
        )
        from infera.engine.vllm.triton_kv_gather import kv_chunk_scatter

        per_page_block_ids, kvd_key, cache_group_id, layer_names = entry
        if not layer_names:
            return

        spec = self._group_kv_spec.get(int(cache_group_id))
        if spec is None:
            return
        present = [n for n in layer_names if n in self._kv_caches]
        if not present:
            return
        layer_tensors = [self._kv_caches[n] for n in present]
        block_size = int(spec["block_size"])

        # In-flight + ring cache shortcut: when we already have the
        # chunk bytes in host RAM, going through hipFile would force
        # us to wait for the on-disk file to materialize (it may not
        # yet — the async save is still pending). The mmap+H2D path
        # via `_load_chunk_packed` will hit the cache in
        # `_fetch_chunk_blob` and scatter directly from RAM, which
        # also dodges the per-chunk hipFile open/register overhead.
        if self._cache_has(kvd_key):
            self._load_chunk_packed(entry)
            return

        # 1. Derive the file path from the content key (#46 / LMCache
        #    GdsBackend design) — connector owns the file tier, no daemon
        #    lookup. We only need the path; the payload comes via hipFile.
        path_obj = self._local_chunk_path(kvd_key)
        if path_obj is None:
            # Not on disk under any configured root → RAM tier / true miss.
            # Fall back to the mmap+H2D path (handles UDS Get for RAM tier).
            self._load_chunk_packed(entry)
            return
        file_path = str(path_obj)
        # LRU-on-read: mark this chunk as recently used so it survives
        # the next reaper tick. Cheap (registry hashmap touch).
        self._l3_touch_load(file_path)

        # 2. pread the header. 64 KB is enough for any realistic
        #    config (typical header <4 KB). os.pread blocks but is
        #    GIL-releasing so other load threads keep going.
        try:
            fd = os.open(str(file_path), os.O_RDONLY)
        except OSError as exc:
            logger.debug(
                "_load_chunk_packed_hipfile_direct: os.open(%s) failed (%s); fall back to mmap+H2D",
                file_path,
                exc,
            )
            self._load_chunk_packed(entry)
            return
        try:
            header_buf = os.pread(fd, self._CHUNK_HEADER_PREFETCH_BYTES, 0)
        except OSError:
            os.close(fd)
            self._load_chunk_packed(entry)
            return
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

        try:
            # header_only=True: parse just the geometry from the 64 KiB
            # pread window and SKIP payload validation (the 130+ MiB
            # payload is DMA'd straight to device, never held in host).
            header, _ignored_payload_view = unpack_chunk(header_buf, header_only=True)
        except PackedFormatError as exc:
            logger.debug(
                "_load_chunk_packed_hipfile_direct: header parse failed key=%s (%s); fall back",
                kvd_key.hex()[:16],
                exc,
            )
            self._load_chunk_packed(entry)
            return

        # Shape compat — same checks as the slow path.
        if header.num_layers != len(present):
            logger.warning(
                "_load_chunk_packed_hipfile_direct: blob num_layers=%d != worker %d key=%s",
                header.num_layers,
                len(present),
                kvd_key.hex()[:16],
            )
            return
        if header.num_kv_channels != int(spec.get("num_kv_channels", 2)):
            return
        if header.hidden_dim != int(spec["hidden_dim"]):
            return

        import struct as _struct

        payload_file_offset = (
            header.payload_byte_offset
            if header.payload_byte_offset > 0
            else (4 + _struct.unpack("<I", header_buf[:4])[0])
        )
        # Total payload size from the header geometry — single source
        # of truth (no mmap to .stat for file length).
        # CORRECTNESS FIX: header.dtype is the packed-format name ("bf16",
        # "fp8_e4m3", ...), NOT the torch name — the old dict keyed by torch
        # names always missed and defaulted to 2 (a 2x payload-size error for
        # fp8). Use the header's own byte width.
        bpe = header.dtype_bytes
        payload_nbytes = (
            header.num_kv_channels
            * header.num_layers
            * header.chunk_tokens
            * header.hidden_dim
            * bpe
        )

        # C2 fix: hipFile.read uses `payload_nbytes` derived purely from
        # the header geometry and never cross-checks against the file's
        # actual size. A truncated file (writer crashed mid-flush, half-
        # published tmp, etc.) would still pass the existence probe in
        # `_local_chunk_path` (it's a regular file with a valid header)
        # and would then DMA whatever the FS returns for the missing tail
        # into the registered device buffer — silent VRAM corruption.
        # Validate file size up front; on mismatch, fall back to the
        # mmap+H2D path which validates via `unpack_chunk(blob)` (no
        # header_only) and safely returns miss → vLLM re-prefills.
        expected_size = payload_file_offset + payload_nbytes
        try:
            actual_size = os.stat(file_path).st_size
        except OSError:
            self._load_chunk_packed(entry)
            return
        if actual_size < expected_size:
            logger.warning(
                "_load_chunk_packed_hipfile_direct: file %s truncated "
                "(size=%d expected>=%d, key=%s) — falling back to mmap+H2D "
                "(which validates payload)",
                file_path,
                actual_size,
                expected_size,
                kvd_key.hex()[:16],
            )
            self._load_chunk_packed(entry)
            return

        _PAGE = 4096
        # GPU-direct requires 4 KiB alignment on file offset; v3
        # chunks (pack_chunk_header_aligned at save) guarantee this.
        if (payload_file_offset & (_PAGE - 1)) != 0:
            self._load_chunk_packed(entry)
            return

        # 3. Per-thread persistent device buffer + registration.
        #    Register-once / reuse across chunks — the BIG overhead
        #    win vs the prior per-chunk register/dereg.
        device = layer_tensors[0].device
        # Pin this _load_executor worker thread to the layer's GPU — its
        # torch current-device defaults to 0, so under TP>1 the registered
        # buffer alloc, hipFile read, and Triton scatter would all target
        # the wrong device → GPU fault. See _load_chunk_packed for detail.
        if device.type == "cuda":
            import torch as _torch_dev

            _torch_dev.cuda.set_device(device)
        kv_dtype = layer_tensors[0].dtype
        st = self._get_parallel_load_state(device, payload_nbytes, _PAGE)
        if st is None:
            self._load_chunk_packed(entry)
            return
        raw_dev = st["raw_dev"]
        buf = st["buf"]
        align_prefix = st["align_prefix"]
        # --- pick the next ring slot + gate on its prior scatter -------
        # Rotate to a DISTINCT slot so this read doesn't alias the slot a
        # still-running scatter is consuming. Wait on ONLY this slot's
        # previous scatter event (it ran n_slots chunks ago, so this is
        # almost always already-signalled = no block) — replaces the
        # per-chunk current_stream().synchronize() that serialized the
        # whole engine.
        slot = st["slot_idx"]
        st["slot_idx"] = (slot + 1) % st["n_slots"]
        slot_byte = slot * st["slot_cap"]  # offset within registered pool
        _prev_ev = st["events"][slot]
        if _prev_ev is not None:
            _prev_ev.synchronize()

        # 4. ONE coalesced hipFile.read into the persistent buffer.
        # FileHandle.open + close are wrapped in the setup lock (the
        # binding's table races at N>=4 otherwise); fh.read runs
        # OUTSIDE the lock so N threads still fan out their reads.
        try:
            import hipfile as _hipfile

            with type(self)._HIPFILE_SETUP_LOCK:
                fh = _hipfile.FileHandle(str(file_path), os.O_RDONLY)
                fh.open()
        except Exception:
            logger.exception(
                "_load_chunk_packed_hipfile_direct: hipfile.FileHandle(%s) failed key=%s",
                file_path,
                kvd_key.hex()[:16],
            )
            self._load_chunk_packed(entry)
            return
        read_err = None
        try:
            # buffer_offset=0 → always read into the head of the
            # registered region; we slice the view by payload_nbytes.
            # NO lock here — N threads truly parallel-read.
            fh.read(buf, payload_nbytes, payload_file_offset, slot_byte)
        except Exception as e:
            read_err = e
        # close under lock regardless of read outcome
        try:
            with type(self)._HIPFILE_SETUP_LOCK:
                fh.close()
        except Exception:
            pass
        if read_err is not None:
            logger.exception(
                "_load_chunk_packed_hipfile_direct: hipFile.read failed "
                "key=%s — falling back to mmap+H2D",
                kvd_key.hex()[:16],
            )
            self._load_chunk_packed(entry)
            return

        # 5. Reshape device bytes → KV layout (zero-copy view) and
        #    Triton scatter into the paged KV cache. The on-disk bytes are
        #    already in the live cache dtype (fp8 passthrough included —
        #    when vLLM runs --kv-cache-dtype fp8 the cache is uint8/fp8 and
        #    the chunk holds those exact bytes), so view as the cache dtype.
        staging_view = raw_dev[align_prefix + slot_byte : align_prefix + slot_byte + payload_nbytes]
        try:
            staging_typed = staging_view.view(kv_dtype).reshape(
                header.num_kv_channels,
                header.num_layers,
                header.chunk_tokens,
                header.hidden_dim,
            )
        except RuntimeError:
            logger.exception(
                "_load_chunk_packed_hipfile_direct: dtype view/reshape failed key=%s payload=%d",
                kvd_key.hex()[:16],
                payload_nbytes,
            )
            return

        per_page_within_group = tuple(
            (page_ids[cache_group_id] if cache_group_id < len(page_ids) else page_ids[0],)
            for page_ids in per_page_block_ids
        )
        layer_to_group = [0] * len(present)
        try:
            # NOTE: sync=False here on purpose. The EVENT-SCOPED wait below
            # (_ev.record/synchronize) already guarantees the scatter
            # completes before this worker returns, at a fraction of the
            # cost of a full current_stream().synchronize(). Passing
            # sync=True would reintroduce that full-stream sync INSIDE the
            # kernel — draining every other load thread's queued work on
            # the shared default stream → the ~7x throughput cliff this
            # ring design exists to avoid (collapses the 16-worker NFS
            # fan-out to single-stream). slot_mapping lifetime is safe via
            # torch's stream-ordered free + the event wait; the GPU-fault
            # crash was fixed by the device pin above, not by this sync.
            kv_chunk_scatter(
                staging_typed,
                layer_tensors,
                per_page_within_group,
                layer_to_group,
                block_size,
                use_triton=True,
            )
            # CORRECTNESS — replaces the per-chunk
            # current_stream().synchronize() with an EVENT-SCOPED wait that
            # gives the SAME two guarantees at a fraction of the cost:
            #
            #   (1) KV-live-before-this-future-completes. The async load
            #       contract (_start_load_kv_async) is: future.done() =>
            #       the chunk's KV is live on the GPU, so vLLM may promote
            #       the request out of WAITING_FOR_REMOTE_KVS and run
            #       attention. The Triton scatter is ASYNC; if we returned
            #       here without waiting, the future would complete while
            #       the scatter is still pending and attention would read
            #       not-yet-scattered (stale/other-chunk) KV — exactly the
            #       systematic cross-request corruption we saw. So we MUST
            #       block this worker until THIS chunk's scatter finishes.
            #   (2) staging-slot-reuse safety. The persistent raw_dev slot
            #       is reused by a later chunk's hipFile DMA, which is NOT
            #       stream-ordered against this async scatter. Once the
            #       scatter event is signalled the slot bytes are fully
            #       consumed and safe to overwrite.
            #
            # We record a cuda.Event on the (current) stream the scatter
            # ran on and synchronize() on THAT event — a targeted wait for
            # this one scatter, NOT a full current_stream().synchronize()
            # that drained every other thread's queued work on the device
            # (the ~7x throughput cliff). The event is also stashed per
            # slot so the next rotation's gate finds it already signalled.
            import torch as _t_sync

            _ev = _t_sync.cuda.Event()
            _ev.record()
            st["events"][slot] = _ev
            _ev.synchronize()
            self._stat_inc("loaded_chunks", 1)
            self._stat_inc("loaded_bytes", payload_nbytes)
        except Exception:
            logger.exception(
                "_load_chunk_packed_hipfile_direct: scatter failed key=%s",
                kvd_key.hex()[:16],
            )

    def _start_load_kv_async(self, meta: Any) -> None:
        """Fix A: submit each load chunk to the load executor WITHOUT
        waiting, grouping futures per req_id. Each _load_chunk_packed
        future completes only after its Triton scatter + cuda.synchronize(),
        so future.done() => that chunk's KV is live on the GPU. Returns
        immediately; get_finished polls the futures and reports a req as
        finished-receiving once all its chunks landed -> vLLM promotes it
        out of WAITING_FOR_REMOTE_KVS."""
        executor = self._load_executor
        chunks = meta.packed_chunks_to_load
        rids = meta.load_chunk_req_ids
        # self._gpu_direct is True only when hipFile + P2PDMA are both usable
        # (set in __init__), so it alone selects DMA vs POSIX — no extra guard.
        use_hipfile_direct = self._gpu_direct
        per_chunk = (
            self._load_chunk_packed_hipfile_direct
            if use_hipfile_direct
            else self._load_chunk_packed
        )
        if executor is None:
            for entry in chunks:
                try:
                    per_chunk(entry)
                except Exception:
                    logger.exception("async load (inline fallback) failed")
            return
        with self._async_inflight_lock:
            for entry, rid in zip(chunks, rids, strict=False):
                fut = executor.submit(per_chunk, entry)
                self._async_inflight.setdefault(rid, []).append(fut)

    def _start_load_kv_parallel(self, meta: Any) -> None:
        """Mode=parallel: submit one ``_load_chunk_packed`` per chunk
        to ``self._load_executor`` (default 8 workers, override via
        ``INFERA_KVD_LOAD_WORKERS``) and wait for all futures before
        returning.

        Why this works (verified by microbench on the MI355X testbed + per the
        prompt for the RDMA host):

        - ``_load_chunk_packed`` reads its chunk's entire packed blob in
          ONE call via ``_fetch_chunk_blob`` (mmap for file-tier or UDS
          Get for RAM-tier). Reads are already coalesced within a chunk.
        - For file-tier hits the underlying call is ``hipFile.read()``
          (or POSIX ``read`` on the mmap fallback). Both release the GIL
          while blocked on the kernel cuFile/read path. N Python threads
          doing concurrent reads therefore truly run in parallel — the
          microbench shows 5-7x lift at N=4-8 on 128 MiB blocks.
        - Chunks are independent (no overlapping KV pages between
          chunks emitted in the same step, per the v2 chunk-key /
          page-id projection in build_connector_meta). Triton scatter
          for each chunk writes to its own KV pages — no cross-chunk
          serialization needed.

        Sync semantics match mode=off: we wait for ALL chunks to land
        before returning, so ``wait_for_layer_load`` stays a no-op and
        attention sees fully-populated KV cache. The win is purely
        bandwidth fan-out, NOT compute overlap (that's mode=prefetch).

        Use this mode when:
          - storage→GPU read BW dominates (large chunks, big working set)
          - RDMA / P2PDMA fabric where per-chunk parallel reads scale
          - chunks are >=128 MiB each (per-chunk smaller = N-thread
            overhead vs read time becomes unfavorable; consider
            ``--chunk-tokens`` >= 2048 for MLA models)
        """
        executor = self._load_executor
        chunks = meta.packed_chunks_to_load
        # Pick the per-chunk loader (two modes, decided by self._gpu_direct):
        #   - GPU-direct (hipFile + P2PDMA) → hipFile.read direct to device
        #     (GIL-released, N-way fan-out)
        #   - POSIX (everything else, incl. no-P2PDMA hosts) → mmap+H2D
        #     (GIL-released read + torch H2D, N-way fan-out, thread-safe).
        #     The thread-unsafe hipFile CPU-bounce fallback was removed.
        # self._gpu_direct is True only when hipFile + P2PDMA are both usable
        # (set in __init__), so it alone selects DMA vs POSIX — no extra guard.
        use_hipfile_direct = self._gpu_direct
        per_chunk = (
            self._load_chunk_packed_hipfile_direct
            if use_hipfile_direct
            else self._load_chunk_packed
        )
        if executor is None or not chunks:
            # Executor torn down or nothing to load — fall back inline.
            for entry in chunks:
                try:
                    per_chunk(entry)
                except Exception:
                    logger.exception(
                        "parallel load (inline fallback): chunk failed for key=%s gid=%s",
                        entry[1].hex()[:16] if len(entry) > 1 else "?",
                        entry[2] if len(entry) > 2 else "?",
                    )
            return

        futures = [executor.submit(per_chunk, entry) for entry in chunks]
        for fut, entry in zip(futures, chunks, strict=False):
            try:
                # Per-chunk timeout: matches the spillover-region read
                # ceiling at worst-case (a 1 GiB chunk over a 100 MB/s
                # NFS would be ~10s; 60s is conservative). Adjust via
                # env if you have pathological storage latencies.
                fut.result(timeout=60.0)
            except concurrent.futures.TimeoutError:
                logger.exception(
                    "parallel load: chunk timeout (>60s) for key=%s gid=%s "
                    "— treating as miss, will re-prefill",
                    entry[1].hex()[:16] if len(entry) > 1 else "?",
                    entry[2] if len(entry) > 2 else "?",
                )
            except Exception:
                logger.exception(
                    "parallel load: chunk failed for key=%s gid=%s",
                    entry[1].hex()[:16] if len(entry) > 1 else "?",
                    entry[2] if len(entry) > 2 else "?",
                )
        # ONE batch-end sync: drains all in-flight Triton scatters
        # across worker threads. Each scatter was launched on the
        # default cuda stream (cheap, async kernel launch). Pulling
        # the sync out of per-chunk into here turns N×sync (slow,
        # serializes the parallel fan-out) into 1×sync at the end
        # of start_load_kv — matches the mode=off semantics that
        # downstream attention assumes (KV cache is fully populated
        # when start_load_kv returns).
        try:
            import torch as _torch

            _torch.cuda.synchronize()
        except ImportError:
            pass

    def _start_load_kv_layerwise_a(self, meta: Any) -> None:
        """Variant A: build per-chunk state with CPU payload already
        decoded + slot_mapping precomputed; defer per-layer GPU work
        to wait_for_layer_load."""
        # Defensive: prior step may have thrown mid-wait_for_layer_load.
        self._inflight_load_state = []
        for entry in meta.packed_chunks_to_load:
            try:
                st = self._prepare_chunk_for_stepped_load(entry)
            except Exception:
                logger.exception(
                    "start_load_kv_layerwise_a: prepare failed (key=%s gid=%s)",
                    entry[1].hex()[:16] if len(entry) > 1 else "?",
                    entry[2] if len(entry) > 2 else "?",
                )
                continue
            if st is not None:
                self._inflight_load_state.append(st)

    def _prepare_chunk_for_stepped_load(
        self,
        entry: tuple[tuple[tuple[int, ...], ...], bytes, int, list[str]],
    ) -> _SteppedChunkLoadState | None:
        """Fetch + decode + CPU staging + slot_mapping precompute. Same
        prelude as `_load_chunk_packed` but stops before H2D and Triton
        scatter — those move to per-layer wait_for_layer_load calls."""
        from infera.engine.vllm.packed_format import (
            PackedFormatError,
            unpack_chunk,
        )

        per_page_block_ids, kvd_key, cache_group_id, layer_names = entry
        if not layer_names:
            return None

        spec = self._group_kv_spec.get(int(cache_group_id))
        if spec is None:
            logger.debug(
                "_prepare_chunk_for_stepped_load: unknown cache_group_id=%s — miss",
                cache_group_id,
            )
            return None
        present = [n for n in layer_names if n in self._kv_caches]
        if not present:
            return None
        layer_tensors = [self._kv_caches[n] for n in present]
        block_size = int(spec["block_size"])

        blob, mm_holder, _file_path = self._fetch_chunk_blob(kvd_key)
        if blob is None:
            return None
        try:
            header, payload_view = unpack_chunk(blob)
        except PackedFormatError as exc:
            logger.debug(
                "_prepare_chunk_for_stepped_load: unpack failed key=%s (%s) — miss",
                kvd_key.hex()[:16],
                exc,
            )
            self._safe_close_mmap(mm_holder)
            return None

        # Shape validation — same checks as _load_chunk_packed.
        if header.num_layers != len(present):
            logger.warning(
                "_prepare_chunk_for_stepped_load: blob num_layers=%d != worker %d "
                "for key=%s gid=%d — miss",
                header.num_layers,
                len(present),
                kvd_key.hex()[:16],
                cache_group_id,
            )
            self._safe_close_mmap(mm_holder)
            return None
        if header.num_kv_channels != int(spec.get("num_kv_channels", 2)):
            logger.warning(
                "_prepare_chunk_for_stepped_load: blob num_kv_channels=%d != worker %d "
                "for key=%s gid=%d — miss",
                header.num_kv_channels,
                spec.get("num_kv_channels", 2),
                kvd_key.hex()[:16],
                cache_group_id,
            )
            self._safe_close_mmap(mm_holder)
            return None
        if header.hidden_dim != int(spec["hidden_dim"]):
            logger.warning(
                "_prepare_chunk_for_stepped_load: blob hidden_dim=%d != worker %d "
                "for key=%s gid=%d — miss",
                header.hidden_dim,
                spec["hidden_dim"],
                kvd_key.hex()[:16],
                cache_group_id,
            )
            self._safe_close_mmap(mm_holder)
            return None

        # Materialize bytes into a CPU tensor we own (so mm_holder can
        # close immediately). This is the byte memcpy we'd be doing
        # anyway in _load_chunk_packed line 2338 today.
        import numpy as _np
        import torch as _torch

        try:
            arr = _np.frombuffer(payload_view, dtype=_np.uint8)
            cpu_buf = _torch.from_numpy(arr.copy())
            kv_dtype = layer_tensors[0].dtype
            cpu_payload = cpu_buf.view(kv_dtype).reshape(
                header.num_kv_channels,
                header.num_layers,
                header.chunk_tokens,
                header.hidden_dim,
            )
        except Exception:
            logger.exception(
                "_prepare_chunk_for_stepped_load: payload decode failed key=%s",
                kvd_key.hex()[:16],
            )
            self._safe_close_mmap(mm_holder)
            return None
        # Variant A is done with mmap now; Variant B would stash it.
        self._safe_close_mmap(mm_holder)

        # Precompute the per-group slot_mapping ONCE here so the
        # per-layer scatter doesn't rebuild it 61× per chunk
        # (see triton_kv_gather.py:317-329 — that builder is called
        # inside _kv_chunk_transfer_triton). We pass our precomputed
        # slot_mapping by reusing kv_chunk_scatter's per-page interface
        # for now — the optimization to bypass the builder belongs in
        # a follow-up.
        device = layer_tensors[0].device
        # Project per_page_block_ids to this group's column.
        per_page_within_group = tuple(
            (page_ids[cache_group_id] if cache_group_id < len(page_ids) else page_ids[0],)
            for page_ids in per_page_block_ids
        )
        slot_list: list[int] = []
        for (bid,) in per_page_within_group:
            base = int(bid) * block_size
            for off in range(block_size):
                slot_list.append(base + off)
        slot_mapping_device = _torch.tensor(
            slot_list,
            dtype=_torch.int32,
            device=device,
        )

        return _SteppedChunkLoadState(
            kvd_key=kvd_key,
            cache_group_id=int(cache_group_id),
            present_layers=tuple(present),
            layer_tensors=layer_tensors,
            layer_to_idx={n: i for i, n in enumerate(present)},
            block_size=block_size,
            chunk_tokens=int(header.chunk_tokens),
            num_kv_channels=int(header.num_kv_channels),
            hidden_dim=int(header.hidden_dim),
            cpu_payload=cpu_payload,
            slot_mapping_device=slot_mapping_device,
        )

    def _wait_for_layer_load_stepped(self, layer_name: str) -> None:
        """Variant A per-layer dispatch: scan in-flight chunks, scatter
        the layer slice into the matching paged-KV. Retire chunks
        whose cursor reaches num_layers. Runs on default stream so
        attention (next launch on default) sees the writes."""
        state_list = self._inflight_load_state
        if not state_list:
            return

        import torch as _torch

        # Lazy import the triton helpers we drive directly.
        try:
            from infera.engine.vllm.triton_kv_gather import (
                _TRITON_AVAILABLE,
                _kv_chunk_transfer_kernel,
            )
        except ImportError:
            _kv_chunk_transfer_kernel = None
            _TRITON_AVAILABLE = False

        retired: list[int] = []
        for st_idx, st in enumerate(state_list):
            idx = st.layer_to_idx.get(layer_name)
            if idx is None:
                # This chunk doesn't cover this layer (HMA cross-group);
                # skip without touching the cursor.
                continue
            try:
                self._ship_one_layer_stepped(
                    st,
                    idx,
                    _torch,
                    _kv_chunk_transfer_kernel,
                    _TRITON_AVAILABLE,
                )
            except Exception:
                logger.exception(
                    "_wait_for_layer_load_stepped: scatter failed key=%s layer=%s",
                    st.kvd_key.hex()[:16],
                    layer_name,
                )
                # Partial-fail: skip rest of this chunk's layers
                # (same silent-miss policy as _load_chunk_packed).
                st.next_layer_idx = len(st.present_layers)
            else:
                st.next_layer_idx += 1
            if st.next_layer_idx >= len(st.present_layers):
                retired.append(st_idx)
        # Pop retired entries from the back so indices stay stable.
        for st_idx in reversed(retired):
            del state_list[st_idx]

    def _ship_one_layer_stepped(
        self,
        st: _SteppedChunkLoadState,
        layer_idx: int,
        torch_mod: Any,
        kernel: Any,
        triton_available: bool,
    ) -> None:
        """One-layer H2D + Triton scatter for Variant A.

        Slices the per-chunk CPU payload at ``layer_idx`` (zero-copy
        view), sends to device, and launches the kernel with
        ``num_layers=1``. No explicit cuda.synchronize — relies on
        default-stream in-order ordering with the subsequent attention
        launch. Falls back to a Python scatter when Triton is unavailable
        (tests on CPU)."""
        layer_tensor = st.layer_tensors[layer_idx]
        device = layer_tensor.device

        # Slice → contiguous → device. Per-layer slice is ~540 KiB for
        # Kimi K2.5 MLA (1 × 1 × 512 × 576 × 2). H2D is blocking;
        # a pinned-host ring is the next step. Keep non_blocking=False
        # until pinning lands.
        cpu_slice = st.cpu_payload[:, layer_idx : layer_idx + 1, :, :].contiguous()
        staging = cpu_slice.to(device, non_blocking=False).contiguous()

        if not triton_available or device.type == "cpu":
            # Fallback path for tests + CPU-only environments. Uses
            # the public scatter helper which handles both 2- and 1-
            # channel layouts (see triton_kv_gather.py
            # _kv_chunk_scatter_python).
            from infera.engine.vllm.triton_kv_gather import kv_chunk_scatter

            # Reproject per_page from precomputed slot_mapping
            # back to per_page_block_ids for the public helper. The
            # python fallback wants the same shape kv_chunk_scatter
            # accepts; we just send one layer.
            per_page = tuple(
                (st.slot_mapping_device[i * st.block_size].item() // st.block_size,)
                for i in range(st.chunk_tokens // st.block_size)
            )
            kv_chunk_scatter(
                staging,
                [layer_tensor],
                per_page,
                [0],
                st.block_size,
                use_triton=False,
            )
            return

        # Triton fast path: drive _kv_chunk_transfer_kernel directly
        # with num_layers=1.
        layer_k_stride = (
            layer_tensor.shape[1] * st.block_size * st.hidden_dim if st.num_kv_channels == 2 else 0
        )
        layer_ptrs = torch_mod.tensor(
            [int(layer_tensor.data_ptr())],
            dtype=torch_mod.int64,
            device=device,
        )
        TOKEN_BLOCK = 64
        grid = (
            st.num_kv_channels * 1,  # single layer
            (st.chunk_tokens + TOKEN_BLOCK - 1) // TOKEN_BLOCK,
        )
        kernel[grid](
            staging,
            layer_ptrs,
            st.slot_mapping_device,
            layer_k_stride=layer_k_stride,
            layer_slot_stride=st.hidden_dim,
            num_layers=1,
            chunk_tokens=st.chunk_tokens,
            hidden_dim=st.hidden_dim,
            NUM_KV_CHANNELS=st.num_kv_channels,
            DIRECTION=0,  # LOAD
            TOKEN_BLOCK=TOKEN_BLOCK,
        )

    @staticmethod
    def _safe_close_mmap(mm_holder: Any) -> None:
        if mm_holder is None:
            return
        try:
            mm_holder.close()
        except BufferError:
            pass

    # ------------------------------------------------------------------
    # Layerwise load Variant B — per-layer H2D on a dedicated copy
    # stream with prefetch_depth lookahead. Default stream waits on
    # the H2D event before launching scatter, so attention on layer N
    # runs concurrently with H2D for layer N+1..N+depth-1.
    # ------------------------------------------------------------------

    def _start_load_kv_layerwise_b(self, meta: Any) -> None:
        """Variant B: build per-chunk state keeping mmap alive, then
        prime the prefetch pipeline by issuing H2D for the first
        ``prefetch_depth`` layers of every chunk on copy_stream."""
        self._inflight_load_state = []
        for entry in meta.packed_chunks_to_load:
            try:
                st = self._prepare_chunk_for_prefetch_load(entry)
            except Exception:
                logger.exception(
                    "start_load_kv_layerwise_b: prepare failed (key=%s gid=%s)",
                    entry[1].hex()[:16] if len(entry) > 1 else "?",
                    entry[2] if len(entry) > 2 else "?",
                )
                continue
            if st is not None:
                self._inflight_load_state.append(st)

        if not self._inflight_load_state:
            return

        # Prime: kick H2D for layers [0..depth-1] on copy_stream.
        try:
            import torch as _torch
        except ImportError:
            return
        from infera.engine.vllm.layerwise_load_helper import (
            _ensure_load_copy_stream,
            get_prefetch_depth,
        )

        copy_stream = _ensure_load_copy_stream(self, _torch)
        depth = get_prefetch_depth()
        for st in self._inflight_load_state:
            n = min(depth, len(st.present_layers))
            for layer_idx in range(n):
                self._kick_h2d_for_layer_prefetch(st, layer_idx, _torch, copy_stream)

    def _prepare_chunk_for_prefetch_load(
        self,
        entry: tuple[tuple[tuple[int, ...], ...], bytes, int, list[str]],
    ) -> _PrefetchChunkLoadState | None:
        """Fetch chunk via mmap (keep alive), decode header, build
        slot_mapping + event registry + empty pinned-slot ring.
        Defer ALL H2D / scatter to start_load_kv_layerwise_b's prime
        + wait_for_layer_load_layerwise_b's per-layer dispatch."""
        from infera.engine.vllm.layerwise_load_helper import (
            _LayerEventRegistry,
            get_prefetch_depth,
        )
        from infera.engine.vllm.packed_format import (
            PackedFormatError,
            unpack_chunk,
        )

        per_page_block_ids, kvd_key, cache_group_id, layer_names = entry
        if not layer_names:
            return None

        spec = self._group_kv_spec.get(int(cache_group_id))
        if spec is None:
            return None
        present = [n for n in layer_names if n in self._kv_caches]
        if not present:
            return None
        layer_tensors = [self._kv_caches[n] for n in present]
        block_size = int(spec["block_size"])

        blob, mm_holder, file_path = self._fetch_chunk_blob(kvd_key)
        if blob is None:
            return None
        try:
            header, payload_view = unpack_chunk(blob)
        except PackedFormatError as exc:
            logger.debug(
                "_prepare_chunk_for_prefetch_load: unpack failed key=%s (%s)",
                kvd_key.hex()[:16],
                exc,
            )
            self._safe_close_mmap(mm_holder)
            return None

        # Shape validation (same as Variant A).
        if (
            header.num_layers != len(present)
            or header.num_kv_channels != int(spec.get("num_kv_channels", 2))
            or header.hidden_dim != int(spec["hidden_dim"])
        ):
            logger.warning(
                "_prepare_chunk_for_prefetch_load: shape mismatch key=%s gid=%d",
                kvd_key.hex()[:16],
                cache_group_id,
            )
            self._safe_close_mmap(mm_holder)
            return None

        import torch as _torch

        kv_dtype = layer_tensors[0].dtype
        device = layer_tensors[0].device
        dtype_bytes = header.dtype_bytes
        per_layer_nbytes = (
            header.num_kv_channels * header.chunk_tokens * header.hidden_dim * dtype_bytes
        )

        # hipFile fast path: when gpu_direct + we have an on-disk path +
        # the binding exposes async I/O, open a FileHandle for
        # cuFileReadAsync direct-to-device reads. Skip the mmap+pinned-
        # host+memcpy chain entirely. Per-layer reads go straight from
        # NVMe → device staging on copy_stream (real GPU-Direct path
        # on P2PDMA hosts; CPU bounce on non-P2PDMA hosts but still
        # one syscall less than the mmap path).
        hipfile_fh = None
        hipfile_buf = None
        hipfile_buf_dev = None
        hipfile_buf_prefix = 0
        hipfile_buf_per_layer = 0
        hipfile_slice_views: list[Any] = []
        # payload_byte_offset for v3-padded chunks (from msgpack body)
        # OR fallback to header_end for v2 unpadded chunks (legacy).
        # cuFile / hipFile-async requires 4 KiB file_offset alignment;
        # v3 chunks are written with payload_byte_offset rounded up to
        # 4 KiB by pack_chunk_header_aligned at save time.
        payload_file_offset = (
            header.payload_byte_offset
            if header.payload_byte_offset > 0
            else (len(blob) - len(payload_view))
        )
        depth = get_prefetch_depth()
        _PAGE = 4096
        if (
            self._gpu_direct
            and file_path is not None
            and (payload_file_offset & (_PAGE - 1)) == 0  # aligned at save
            and (per_layer_nbytes & (_PAGE - 1)) == 0  # naturally aligned
        ):
            try:
                import ctypes as _ct

                import hipfile as _hipfile

                if hasattr(_hipfile, "supports_async") and _hipfile.supports_async():
                    fh_obj = _hipfile.FileHandle(file_path, os.O_RDONLY)
                    fh_obj.open()
                    self._ensure_hipfile_stream_registered(_torch)

                    # Pre-allocate ONE big device buffer for this chunk's
                    # prefetch ring. Overallocate by 4 KiB so we can shift
                    # the base pointer to the next 4 KiB boundary —
                    # PyTorch's caching allocator gives us ≥256-byte
                    # alignment but cuFile wants 4 KiB. The whole region
                    # registers ONCE (vs per-layer churn that trips the
                    # 5023 "already registered" race against torch's
                    # cache reuse).
                    region_nbytes = depth * per_layer_nbytes
                    raw_dev = _torch.empty(
                        region_nbytes + _PAGE,
                        dtype=_torch.uint8,
                        device=device,
                    )
                    raw_ptr = int(raw_dev.data_ptr())
                    align_prefix = (-raw_ptr) & (_PAGE - 1)
                    aligned_ptr = raw_ptr + align_prefix
                    buf = _hipfile.Buffer.from_ctypes_void_p(
                        _ct.c_void_p(aligned_ptr),
                        region_nbytes,
                        0,
                    )
                    buf.register()

                    # Pre-build zero-copy slice views for each ring slot.
                    # Each slice covers per_layer_nbytes bytes starting
                    # at ring_idx * per_layer_nbytes within the aligned
                    # region.
                    for i in range(depth):
                        start = align_prefix + i * per_layer_nbytes
                        hipfile_slice_views.append(raw_dev[start : start + per_layer_nbytes])

                    hipfile_fh = fh_obj
                    hipfile_buf = buf
                    hipfile_buf_dev = raw_dev
                    hipfile_buf_prefix = align_prefix
                    hipfile_buf_per_layer = per_layer_nbytes
            except Exception:
                logger.exception(
                    "_prepare_chunk_for_prefetch_load: hipfile prep failed for "
                    "key=%s path=%s — falling back to mmap+H2D",
                    kvd_key.hex()[:16],
                    file_path,
                )
                if hipfile_fh is not None:
                    try:
                        hipfile_fh.close()
                    except Exception:
                        pass
                if hipfile_buf is not None:
                    try:
                        hipfile_buf.deregister()
                    except Exception:
                        pass
                hipfile_fh = None
                hipfile_buf = None
                hipfile_buf_dev = None
                hipfile_slice_views = []

        # Per-group slot_mapping precomputed once for all 61 layers'
        # scatters (see Variant A — same optimization).
        per_page_within_group = tuple(
            (page_ids[cache_group_id] if cache_group_id < len(page_ids) else page_ids[0],)
            for page_ids in per_page_block_ids
        )
        slot_list: list[int] = []
        for (bid,) in per_page_within_group:
            base = int(bid) * block_size
            for off in range(block_size):
                slot_list.append(base + off)
        slot_mapping_device = _torch.tensor(
            slot_list,
            dtype=_torch.int32,
            device=device,
        )

        # Pinned-slot ring sized at prefetch_depth (each slot is one
        # layer's bytes at a time). Lazily acquired in _kick_h2d.
        # When hipFile-async path is active, pinned_slots stays
        # all-None and the per-chunk hipfile_buf + slice_views handle
        # per-layer device staging via one pre-registered Buffer.
        # `depth` was already computed above for the hipFile prep.
        pinned_ring: list[Any] = [None] * depth

        events = _LayerEventRegistry(num_layers=header.num_layers)

        # If we took the hipFile path, the mmap was opened just for
        # blob/header decoding — we don't need it across layer reads.
        # cuFileReadAsync reads directly from disk via the FileHandle.
        if hipfile_fh is not None:
            self._safe_close_mmap(mm_holder)
            mm_holder = None
            payload_view = None  # bytes come from hipFile reads now

        return _PrefetchChunkLoadState(
            kvd_key=kvd_key,
            cache_group_id=int(cache_group_id),
            present_layers=tuple(present),
            layer_tensors=layer_tensors,
            layer_to_idx={n: i for i, n in enumerate(present)},
            block_size=block_size,
            chunk_tokens=int(header.chunk_tokens),
            num_kv_channels=int(header.num_kv_channels),
            hidden_dim=int(header.hidden_dim),
            payload_view=payload_view,
            payload_dtype=kv_dtype,
            per_layer_nbytes=per_layer_nbytes,
            mm_holder=mm_holder,
            hipfile_handle=hipfile_fh,
            payload_file_offset=payload_file_offset,
            hipfile_buf=hipfile_buf,
            hipfile_buf_dev=hipfile_buf_dev,
            hipfile_buf_prefix=hipfile_buf_prefix,
            hipfile_buf_per_layer=hipfile_buf_per_layer,
            hipfile_slice_views=hipfile_slice_views,
            slot_mapping_device=slot_mapping_device,
            pinned_slots=pinned_ring,
            events=events,
        )

    def _ensure_hipfile_stream_registered(self, torch_mod: Any) -> Any | None:
        """Lazy-init: register self._load_copy_stream with hipFile so
        cuFileReadAsync / cuFileWriteAsync can target it. Idempotent."""
        st_obj = getattr(self, "_load_copy_stream_hipfile", None)
        if st_obj is not None:
            return st_obj
        from infera.engine.vllm.layerwise_load_helper import (
            _ensure_load_copy_stream,
        )

        copy_stream = _ensure_load_copy_stream(self, torch_mod)
        if copy_stream is None:
            return None
        try:
            import hipfile as _hipfile

            st_obj = _hipfile.Stream(copy_stream.cuda_stream)
            st_obj.register()
            self._load_copy_stream_hipfile = st_obj
            return st_obj
        except Exception:
            logger.exception("_ensure_hipfile_stream_registered: failed to register stream")
            return None

    def _kick_h2d_for_layer_prefetch_hipfile(
        self,
        st: _PrefetchChunkLoadState,
        layer_idx: int,
        torch_mod: Any,
        copy_stream: Any,
    ) -> Any:
        """hipFile fast-path: cuFileReadAsync layer slice into the
        pre-registered per-chunk device buffer (ring slot = layer_idx
        % prefetch_depth) on copy_stream. Records cuda.Event for the
        consumer wait.

        File and buffer offsets are both 4 KiB-aligned by construction:
          * file_offset = payload_file_offset + layer_idx * per_layer_nbytes
              — payload_file_offset was padded to 4 KiB at save time
                (pack_chunk_header_aligned) and per_layer_nbytes is a
                multiple of 4 KiB by tensor shape for typical models.
          * buffer_offset = ring_idx * hipfile_buf_per_layer — the
                base buffer is aligned (overallocated + shifted by
                hipfile_buf_prefix), so each ring slot is aligned too.

        No Buffer.register/deregister churn — the buffer was
        registered ONCE at prep time and is recycled across layers.
        Returns the slice view (zero-copy into the registered region)
        or None on failure.
        """
        depth = len(st.hipfile_slice_views)
        ring_idx = layer_idx % depth

        file_off = st.payload_file_offset + layer_idx * st.per_layer_nbytes
        buf_off = ring_idx * st.hipfile_buf_per_layer
        try:
            # read_async returns an _AsyncIOHandle owning the four
            # ctypes slots (size_p, file_off_p, buf_off_p, bytes_done_p).
            # MUST be kept alive past stream sync — the driver writes
            # bytes_done after the I/O completes on copy_stream. Stash
            # it in st.hipfile_io_handles[layer_idx]; retire (or the
            # next prep for this layer slot, if we ever reuse) drops it.
            io_handle = st.hipfile_handle.read_async(
                st.hipfile_buf,
                st.per_layer_nbytes,
                file_off,
                buf_off,
                copy_stream.cuda_stream,
            )
            ev = torch_mod.cuda.Event()
            ev.record(copy_stream)
            st.events.record(layer_idx, ev)
            st.hipfile_io_handles[layer_idx] = io_handle
        except Exception:
            logger.exception(
                "_kick_h2d_for_layer_prefetch_hipfile: read_async raised "
                "key=%s layer=%d file_off=%d buf_off=%d",
                st.kvd_key.hex()[:16],
                layer_idx,
                file_off,
                buf_off,
            )
            return None

        # Stash the slice view under layer_idx so _ship_one_layer_prefetch
        # picks it up via the same _device_staging_by_layer dict the
        # mmap path uses.
        slice_view = st.hipfile_slice_views[ring_idx]
        cache = getattr(st, "_device_staging_by_layer", None)
        if cache is None:
            cache = {}
            st._device_staging_by_layer = cache
        cache[layer_idx] = slice_view
        return slice_view

    def _kick_h2d_for_layer_prefetch(
        self,
        st: _PrefetchChunkLoadState,
        layer_idx: int,
        torch_mod: Any,
        copy_stream: Any,
    ) -> Any:
        """Issue H2D for ``st.cpu_payload[:, layer_idx]`` on copy_stream.

        Dispatch: when ``st.hipfile_handle`` is set (gpu_direct + file
        path + supports_async), route to the hipFile-async fast path
        which skips mmap + pinned-host entirely. Otherwise use the
        pinned-host ring + PyTorch H2D path.

        Acquires a pinned-host slot (round-robin index =
        ``layer_idx % prefetch_depth``), copies the layer slice from
        the mmap'd payload into pinned host, kicks off
        ``device_staging.copy_(pinned, non_blocking=True)`` on
        copy_stream, records a cuda.Event, stashes it in the registry.

        Returns the device staging tensor (caller stashes alongside).
        On any failure (pool empty, alloc fail, CUDA failure), returns
        None and the caller will fall back to a synchronous inline
        H2D in wait_for_layer_load.
        """
        if st.hipfile_handle is not None:
            return self._kick_h2d_for_layer_prefetch_hipfile(
                st,
                layer_idx,
                torch_mod,
                copy_stream,
            )
        from infera.engine.vllm.layerwise_load_helper import (
            get_or_create_pinned_pool,
        )

        depth = len(st.pinned_slots)
        ring_idx = layer_idx % depth
        # If a prior layer's slot is still here, release it back to the
        # pool. (The earlier wait_for_layer_load already issued
        # wait_event + scatter for that layer, so the pinned bytes are
        # safe to recycle.)
        pool = get_or_create_pinned_pool(self)
        if st.pinned_slots[ring_idx] is not None:
            pool.release(st.pinned_slots[ring_idx])
            st.pinned_slots[ring_idx] = None

        # 1. Acquire pinned host slot.
        slot = pool.acquire(st.per_layer_nbytes, torch_mod)
        if slot is None:
            # Pool cap hit — wait_for_layer_load_b will do inline blocking H2D.
            return None
        st.pinned_slots[ring_idx] = slot

        # 2. Copy the layer slice from mmap'd payload into the pinned host.
        # The mmap holds the bytes; we slice [num_kv_channels × layer ×
        # chunk_tokens × hidden_dim × dtype_bytes] starting at
        # layer_idx × (chunk_tokens × hidden_dim × dtype_bytes).
        dtype_bytes = st.per_layer_nbytes // (st.num_kv_channels * st.chunk_tokens * st.hidden_dim)
        slice_size = st.chunk_tokens * st.hidden_dim * dtype_bytes
        # Payload is laid out [C, L, T, H]; for each channel we copy
        # slice_size bytes starting at (c*num_layers + layer_idx) *
        # slice_size.
        try:
            dst = slot.tensor[: st.per_layer_nbytes]
            payload_bytes = st.payload_view
            dst_off = 0
            for c in range(st.num_kv_channels):
                start = (c * len(st.present_layers) + layer_idx) * slice_size
                end = start + slice_size
                dst[dst_off : dst_off + slice_size] = torch_mod.frombuffer(
                    payload_bytes[start:end],
                    dtype=torch_mod.uint8,
                )
                dst_off += slice_size
        except Exception:
            logger.exception(
                "_kick_h2d_for_layer_prefetch: pinned-copy failed key=%s layer=%d",
                st.kvd_key.hex()[:16],
                layer_idx,
            )
            pool.release(st.pinned_slots[ring_idx])
            st.pinned_slots[ring_idx] = None
            return None

        # 3. H2D into device staging on copy_stream + record event.
        device = st.layer_tensors[0].device
        try:
            from contextlib import nullcontext

            ctx = torch_mod.cuda.stream(copy_stream) if copy_stream is not None else nullcontext()
            with ctx:
                staging_dev = torch_mod.empty(
                    st.per_layer_nbytes,
                    dtype=torch_mod.uint8,
                    device=device,
                )
                staging_dev.copy_(dst, non_blocking=copy_stream is not None)
                if copy_stream is not None:
                    ev = torch_mod.cuda.Event()
                    ev.record(copy_stream)
                    st.events.record(layer_idx, ev)
                else:
                    # No copy stream → no event needed; subsequent
                    # default-stream Triton scatter implicitly serializes.
                    pass
        except Exception:
            logger.exception(
                "_kick_h2d_for_layer_prefetch: H2D enqueue failed key=%s layer=%d",
                st.kvd_key.hex()[:16],
                layer_idx,
            )
            pool.release(st.pinned_slots[ring_idx])
            st.pinned_slots[ring_idx] = None
            return None

        # Stash the device staging tensor on the state under layer_idx
        # so wait_for_layer_load_b can find it. Use a side dict because
        # we don't want to bloat _PrefetchChunkLoadState with another
        # num_layers-sized list.
        cache = getattr(st, "_device_staging_by_layer", None)
        if cache is None:
            cache = {}
            st._device_staging_by_layer = cache
        cache[layer_idx] = staging_dev
        return staging_dev

    def _wait_for_layer_load_prefetch(self, layer_name: str) -> None:
        """Per-layer dispatch for Variant B. For each in-flight chunk:
        1. Look up this chunk's local layer_idx for layer_name.
        2. If H2D event was recorded: default_stream.wait_event(event);
           else: inline blocking H2D (fallback for pool-cap-hit case).
        3. Launch Triton scatter on default stream (num_layers=1).
        4. Release pinned slot (its bytes are now on device).
        5. Kick H2D for layer_idx + prefetch_depth on copy_stream.
        6. Advance cursor; retire when cursor reaches num_layers.
        """
        state_list = self._inflight_load_state
        if not state_list:
            return

        import torch as _torch

        from infera.engine.vllm.layerwise_load_helper import (
            _ensure_load_copy_stream,
            get_or_create_pinned_pool,
            get_prefetch_depth,
        )

        try:
            from infera.engine.vllm.triton_kv_gather import (
                _TRITON_AVAILABLE,
                _kv_chunk_transfer_kernel,
            )
        except ImportError:
            _kv_chunk_transfer_kernel = None
            _TRITON_AVAILABLE = False

        copy_stream = _ensure_load_copy_stream(self, _torch)
        depth = get_prefetch_depth()
        pool = get_or_create_pinned_pool(self)

        retired: list[int] = []
        for st_idx, st in enumerate(state_list):
            idx = st.layer_to_idx.get(layer_name)
            if idx is None:
                continue
            try:
                self._ship_one_layer_prefetch(
                    st,
                    idx,
                    _torch,
                    copy_stream,
                    _kv_chunk_transfer_kernel,
                    _TRITON_AVAILABLE,
                )
            except Exception:
                logger.exception(
                    "_wait_for_layer_load_prefetch: layer ship failed key=%s layer=%s",
                    st.kvd_key.hex()[:16],
                    layer_name,
                )
                st.next_layer_idx = len(st.present_layers)
            else:
                st.next_layer_idx += 1
                # Prefetch the layer that just rolled into the window.
                next_layer = idx + depth
                if next_layer < len(st.present_layers):
                    try:
                        self._kick_h2d_for_layer_prefetch(
                            st,
                            next_layer,
                            _torch,
                            copy_stream,
                        )
                    except Exception:
                        logger.exception(
                            "_wait_for_layer_load_prefetch: prefetch failed key=%s next_layer=%d",
                            st.kvd_key.hex()[:16],
                            next_layer,
                        )

            if st.next_layer_idx >= len(st.present_layers):
                retired.append(st_idx)
                # Release any remaining pinned slots + close mmap +
                # deregister per-chunk hipFile buffer + close handle.
                for s in st.pinned_slots:
                    pool.release(s)
                st.pinned_slots = [None] * len(st.pinned_slots)
                if st.hipfile_buf is not None:
                    try:
                        st.hipfile_buf.deregister()
                    except Exception:
                        logger.exception(
                            "retire: hipfile_buf.deregister raised key=%s",
                            st.kvd_key.hex()[:16],
                        )
                    st.hipfile_buf = None
                # Drop refs to slice views + raw device tensor. Refs go
                # via cuda lazy free; events already recorded on them
                # are owned by the events list and don't need the
                # tensor object alive after their wait fires.
                st.hipfile_slice_views = []
                st.hipfile_buf_dev = None
                if st.hipfile_handle is not None:
                    try:
                        st.hipfile_handle.close()
                    except Exception:
                        pass
                    st.hipfile_handle = None
                # Drop the per-layer _AsyncIOHandle slots. Safe here:
                # all per-layer events have been waited on by the
                # consumer (st.next_layer_idx == len(st.present_layers)),
                # so the driver has written bytes_done into the slots
                # and won't touch them again.
                st.hipfile_io_handles.clear()
                self._safe_close_mmap(st.mm_holder)
                st.mm_holder = None
        for st_idx in reversed(retired):
            del state_list[st_idx]

    def _ship_one_layer_prefetch(
        self,
        st: _PrefetchChunkLoadState,
        layer_idx: int,
        torch_mod: Any,
        copy_stream: Any,
        kernel: Any,
        triton_available: bool,
    ) -> None:
        """Variant B per-layer ship: wait on H2D event (or inline H2D
        fallback), launch Triton scatter on default stream. Reinterprets
        the device staging uint8 tensor as the KV dtype just before the
        kernel — same shape contract as Variant A's scatter."""
        layer_tensor = st.layer_tensors[layer_idx]
        device = layer_tensor.device

        cache = getattr(st, "_device_staging_by_layer", None) or {}
        # POP (not get): drop the dict's ref to this layer's device staging
        # tensor as soon as we take it. The local ``staging_dev_uint8`` keeps
        # it alive through the scatter-kernel launch below; on function return
        # the last ref dies and the caching allocator frees it stream-ordered
        # (safe — the scatter was enqueued on the default stream first, so any
        # reuse is ordered after it). Without this pop the dict retains every
        # layer's staging until the whole chunk retires, so device staging
        # grows to num_layers × per_layer_nbytes per in-flight chunk. With a
        # large pinned cap (full overlap → every layer takes this fast path)
        # that balloon starves the KV pool and kills the engine
        # (kv_cache_usage→0.97 at step 0). Bounding it to ~prefetch_depth
        # live tensors is what makes a high PINNED_CAP safe.
        staging_dev_uint8 = cache.pop(layer_idx, None)
        if staging_dev_uint8 is not None and copy_stream is not None:
            # Fast path: H2D was prefetched, just wait for it.
            default_stream = torch_mod.cuda.current_stream(device=device)
            st.events.wait(layer_idx, default_stream)
        else:
            # Slow path (no copy_stream, or pool was capped and prefetch
            # returned None): inline blocking H2D into a fresh device
            # tensor on the default stream.
            dtype_bytes = st.per_layer_nbytes // (
                st.num_kv_channels * st.chunk_tokens * st.hidden_dim
            )
            slice_size = st.chunk_tokens * st.hidden_dim * dtype_bytes
            host_buf = torch_mod.empty(
                st.per_layer_nbytes,
                dtype=torch_mod.uint8,
                device="cpu",
            )
            dst_off = 0
            payload_bytes = st.payload_view
            for c in range(st.num_kv_channels):
                start = (c * len(st.present_layers) + layer_idx) * slice_size
                end = start + slice_size
                host_buf[dst_off : dst_off + slice_size] = torch_mod.frombuffer(
                    payload_bytes[start:end],
                    dtype=torch_mod.uint8,
                )
                dst_off += slice_size
            staging_dev_uint8 = host_buf.to(device, non_blocking=False)

        # Reinterpret bytes as the cache dtype + reshape to scatter input
        # shape. The staging bytes are already in the live cache dtype
        # (payload_dtype == layer_tensors[0].dtype), fp8 passthrough
        # included — when vLLM runs --kv-cache-dtype fp8 the cache is
        # uint8/fp8 and these are the exact bytes vLLM wrote. The scatter
        # requires staging.dtype == layer.dtype, which this view satisfies.
        staging_typed = staging_dev_uint8.view(st.payload_dtype).reshape(
            st.num_kv_channels,
            1,
            st.chunk_tokens,
            st.hidden_dim,
        )

        if not triton_available or device.type == "cpu":
            from infera.engine.vllm.triton_kv_gather import kv_chunk_scatter

            per_page = tuple(
                (st.slot_mapping_device[i * st.block_size].item() // st.block_size,)
                for i in range(st.chunk_tokens // st.block_size)
            )
            kv_chunk_scatter(
                staging_typed,
                [layer_tensor],
                per_page,
                [0],
                st.block_size,
                use_triton=False,
            )
            return

        # Triton fast path on default stream.
        layer_k_stride = (
            layer_tensor.shape[1] * st.block_size * st.hidden_dim if st.num_kv_channels == 2 else 0
        )
        layer_ptrs = torch_mod.tensor(
            [int(layer_tensor.data_ptr())],
            dtype=torch_mod.int64,
            device=device,
        )
        TOKEN_BLOCK = 64
        grid = (
            st.num_kv_channels * 1,
            (st.chunk_tokens + TOKEN_BLOCK - 1) // TOKEN_BLOCK,
        )
        kernel[grid](
            staging_typed,
            layer_ptrs,
            st.slot_mapping_device,
            layer_k_stride=layer_k_stride,
            layer_slot_stride=st.hidden_dim,
            num_layers=1,
            chunk_tokens=st.chunk_tokens,
            hidden_dim=st.hidden_dim,
            NUM_KV_CHANNELS=st.num_kv_channels,
            DIRECTION=0,
            TOKEN_BLOCK=TOKEN_BLOCK,
        )

    def _load_chunk_packed(
        self,
        entry: tuple[tuple[tuple[int, ...], ...], bytes, int, list[str]],
    ) -> None:
        """Load one v2 chunk: look up tier, read the blob (mmap if
        file tier, UDS Get if RAM tier), unpack the header, run
        Triton scatter to fill per-layer × per-page slots.

        ``entry`` shape (from `_emit_v2_chunks`):
          (per_page_block_ids, kvd_key, cache_group_id, layer_names)
        """
        from infera.engine.vllm.packed_format import (
            PackedFormatError,
            unpack_chunk,
        )
        from infera.engine.vllm.triton_kv_gather import kv_chunk_scatter

        per_page_block_ids, kvd_key, cache_group_id, layer_names = entry
        if not layer_names:
            return

        spec = self._group_kv_spec.get(int(cache_group_id))
        if spec is None:
            logger.debug(
                "_load_chunk_packed: unknown cache_group_id=%s; treating as miss",
                cache_group_id,
            )
            return
        present = [n for n in layer_names if n in self._kv_caches]
        if not present:
            return
        layer_tensors = [self._kv_caches[n] for n in present]
        block_size = int(spec["block_size"])

        blob, mm_holder, _file_path = self._fetch_chunk_blob(kvd_key)
        if blob is None:
            # Promised by the scheduler probe but not loadable here
            # (file vanished / RAM-tier miss) → vLLM recomputes it.
            self._stat_inc("load_errors", 1)
            return

        # 2. Unpack header.
        try:
            header, payload_view = unpack_chunk(blob)
        except PackedFormatError as exc:
            logger.debug(
                "_load_chunk_packed: unpack failed for key=%s (%s) — miss",
                kvd_key.hex()[:16],
                exc,
            )
            if mm_holder is not None:
                try:
                    mm_holder.close()
                except BufferError:
                    pass
            return

        # Validate shape compat with our registered KV caches.
        if header.num_layers != len(present):
            logger.warning(
                "_load_chunk_packed: blob num_layers=%d != worker layers %d "
                "for key=%s gid=%d — miss",
                header.num_layers,
                len(present),
                kvd_key.hex()[:16],
                cache_group_id,
            )
            if mm_holder is not None:
                try:
                    mm_holder.close()
                except BufferError:
                    pass
            return
        if header.num_kv_channels != int(spec.get("num_kv_channels", 2)):
            logger.warning(
                "_load_chunk_packed: blob num_kv_channels=%d != worker %d "
                "for key=%s gid=%d — miss (probably MLA-vs-regular mismatch)",
                header.num_kv_channels,
                spec.get("num_kv_channels", 2),
                kvd_key.hex()[:16],
                cache_group_id,
            )
            if mm_holder is not None:
                try:
                    mm_holder.close()
                except BufferError:
                    pass
            return
        if header.hidden_dim != int(spec["hidden_dim"]):
            logger.warning(
                "_load_chunk_packed: blob hidden_dim=%d != worker hidden_dim %d "
                "for key=%s gid=%d — miss",
                header.hidden_dim,
                spec["hidden_dim"],
                kvd_key.hex()[:16],
                cache_group_id,
            )
            if mm_holder is not None:
                try:
                    mm_holder.close()
                except BufferError:
                    pass
            return

        # 3. Materialize payload into a device tensor and Triton-scatter.
        # Zero-copy from memoryview → numpy → torch is possible for
        # CPU but staging must be on device for the scatter kernel.
        # H2D once for the whole chunk (= one big copy) — same shape
        # as LMCache's pattern.
        import numpy as _np
        import torch as _torch

        try:
            arr = _np.frombuffer(payload_view, dtype=_np.uint8)
            # Reshape to dtype + KV_2LTD shape; .view() reinterprets
            # bytes as the kv dtype, then .reshape() applies the
            # logical shape (no copy).
            staging_cpu = _torch.from_numpy(arr.copy())  # one full memcpy
            kv_dtype = layer_tensors[0].dtype
            # Re-view as the KV (cache) dtype after the byte copy. Leading
            # dim is num_kv_channels (2 for regular, 1 for MLA) — taken
            # from the header which is the source of truth. The payload
            # bytes are already in the live cache dtype (fp8 passthrough
            # included: --kv-cache-dtype fp8 → uint8/fp8 cache holds these
            # exact bytes), so viewing as kv_dtype is the correct decode.
            staging_typed = staging_cpu.view(kv_dtype).reshape(
                header.num_kv_channels,
                header.num_layers,
                header.chunk_tokens,
                header.hidden_dim,
            )
        except Exception:
            logger.exception(
                "_load_chunk_packed: payload decode failed for key=%s",
                kvd_key.hex()[:16],
            )
            if mm_holder is not None:
                try:
                    mm_holder.close()
                except BufferError:
                    pass
            return
        # mm holder no longer needed after the .copy() above
        if mm_holder is not None:
            try:
                mm_holder.close()
            except BufferError:
                pass

        device = layer_tensors[0].device
        # This runs in a _load_executor worker thread (parallel/async
        # modes), whose torch current-device defaults to 0 — NOT the
        # worker's assigned GPU. Under TP>1 that means the H2D + Triton
        # scatter below would launch on device 0 while the layer tensors
        # live on cuda:rank → "Memory access fault" GPU crash. Pin the
        # thread to the layer's device. (In "off" mode this runs on the
        # engine main thread where the device is already correct; the call
        # is a cheap no-op there.)
        if device.type == "cuda":
            _torch.cuda.set_device(device)
        staging = staging_typed.to(device, non_blocking=False)

        # Per-page block ids within this group — same projection as
        # save side.
        per_page_within_group = tuple(
            (page_ids[cache_group_id] if cache_group_id < len(page_ids) else page_ids[0],)
            for page_ids in per_page_block_ids
        )
        layer_to_group = [0] * len(present)
        try:
            # sync=False: the explicit synchronize() below already covers
            # scatter completion before `staging` is freed/reused; passing
            # sync=True would just double the full-stream sync.
            kv_chunk_scatter(
                staging,
                layer_tensors,
                per_page_within_group,
                layer_to_group,
                block_size,
                use_triton=True,
            )
            if device.type == "cuda":
                _torch.cuda.synchronize()
            self._stat_inc("loaded_chunks", 1)
            self._stat_inc("loaded_bytes", int(staging.numel()) * int(staging.element_size()))
        except Exception:
            logger.exception(
                "_load_chunk_packed: scatter kernel failed for key=%s gid=%d",
                kvd_key.hex()[:16],
                cache_group_id,
            )
            return


# ---------------------------------------------------------------------------
# Register this out-of-tree connector by NAME in vLLM's KVConnectorFactory.
# vLLM resolves the connector for *creation* via kv_connector_module_path, but
# MultiConnector's stats reconstruction (build_kv_connector_stats) looks each
# child up by name via get_connector_class_by_name(), which only consults the
# factory registry. Without a name entry, PD + kvd (kvd wrapped in a
# MultiConnector alongside an RDMA transport) crashes the engine on the first
# connector-stats record: "Connector 'InferaKvdConnector' is not registered."
# Idempotent and import-safe.
try:  # pragma: no cover - import-time registration side effect
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory as _KVCF,
    )

    if "InferaKvdConnector" not in getattr(_KVCF, "_registry", {}):
        _KVCF.register_connector(
            "InferaKvdConnector",
            "infera.engine.vllm.kvd_connector",
            "InferaKvdConnector",
        )
except Exception:  # noqa: BLE001 - never let registration break module import
    pass

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""KV storage tiers below HBM: host DRAM, local SSD, remote/pooled storage.

A tier holds prefix-cache blocks that HBM evicted, so a later request can load
them back instead of recomputing them. What a tier changes is therefore *reuse*,
not admission: a request being decoded still needs its blocks in HBM, so a tier
raises how much prefix an engine keeps cached and leaves how many requests it
runs at once where HBM put it.

Each tier is described per GPU, the way the hardware is: every GPU of a replica
holds its own shard of every cached block and loads that shard over its own
link, so a load costs ``shard_bytes / bandwidth + blocks * latency`` and the
GPUs of a replica load in parallel. A link shared by several GPUs (a node's
NVMe array, a rack's flash enclosure) is entered as its per-GPU share.

Placement is capacity-aware and recency-ordered: a block enters HBM when it is
computed or loaded, and when a tier is full its least-recently-used *tail*
block moves one tier down rather than being dropped -- the same leaf-first rule
the HBM radix cache already uses, applied per tier, so a shared head is never
pushed below its own continuation. Only the last tier drops blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

_GiB = 1024.0**3

# name -> (per-GPU GB/s into HBM, per-block access latency in us)
#
# dram: PCIe 5 x16 host link, ~64 GB/s nominal.
# ssd:  a node's NVMe array shared by its GPUs -- 8 Gen5 drives at ~14 GB/s
#       sequential read behind 8 GPUs is ~14 GB/s each before contention, and
#       ~12 after the filesystem and the host bounce buffer.
# remote: rack- or cluster-level KV storage over the GPU's RDMA NIC; 400 Gb/s
#       is 50 GB/s on the wire and the store, not the NIC, usually binds.
TIER_DEFAULTS: dict[str, tuple[float, float]] = {
    "dram": (64.0, 2.0),
    "host": (64.0, 2.0),
    "cxl": (48.0, 4.0),
    "ssd": (12.0, 80.0),
    "nvme": (12.0, 80.0),
    "remote": (25.0, 20.0),
}


@dataclass(frozen=True)
class KvTier:
    """One storage tier below HBM, per GPU."""

    name: str
    capacity_gb: float
    bw_gbps: float
    latency_us: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity_gb < 0:
            raise ValueError(f"KV tier {self.name}: capacity must be >= 0")
        if self.bw_gbps <= 0:
            raise ValueError(f"KV tier {self.name}: bandwidth must be > 0")
        if self.latency_us < 0:
            raise ValueError(f"KV tier {self.name}: latency must be >= 0")

    @classmethod
    def parse(cls, spec: str) -> KvTier:
        """``NAME:CAPACITY_GB[:BW_GBPS[:LATENCY_US]]``, e.g. ``ssd:4000`` or
        ``dram:512:64:2``. Bandwidth and latency default by name."""
        parts = [p.strip() for p in str(spec).split(":")]
        if len(parts) < 2 or not parts[0]:
            raise ValueError(
                f"bad KV tier '{spec}' (expected NAME:CAPACITY_GB[:BW_GBPS[:LATENCY_US]])"
            )
        name = parts[0].lower()
        bw, lat = TIER_DEFAULTS.get(name, (None, 0.0))
        if len(parts) > 2 and parts[2]:
            bw = float(parts[2])
        if len(parts) > 3 and parts[3]:
            lat = float(parts[3])
        if bw is None:
            raise ValueError(f"KV tier '{name}' has no default bandwidth; give it as NAME:GB:GBPS")
        return cls(name=name, capacity_gb=float(parts[1]), bw_gbps=float(bw), latency_us=float(lat))

    def capacity_blocks(self, bytes_per_token: float, block_size: int) -> int:
        per_block = bytes_per_token * max(1, int(block_size))
        if per_block <= 0:
            return 0
        return int(self.capacity_gb * _GiB // per_block)

    def block_load_ms(self, bytes_per_token: float, block_size: int) -> float:
        """Time to bring one block's per-GPU shard into HBM."""
        per_block = bytes_per_token * max(1, int(block_size))
        return per_block / (self.bw_gbps * 1e9) * 1e3 + self.latency_us * 1e-3


def resolve_tiers(specs, offload_gb: float = 0.0, offload_bw: float = 64.0) -> list[KvTier]:
    """Tiers from ``--kv-tier`` specs, else the legacy single host-offload tier.

    Ordered as given: the first spec is the tier directly below HBM.
    """
    tiers = [s if isinstance(s, KvTier) else KvTier.parse(s) for s in (specs or [])]
    if not tiers and offload_gb and offload_gb > 0:
        tiers = [
            KvTier("dram", float(offload_gb), float(offload_bw or 64.0), TIER_DEFAULTS["dram"][1])
        ]
    return [t for t in tiers if t.capacity_gb > 0]


def kv_bytes_per_token(inference_config) -> float:
    """Per-GPU KV bytes one token of context occupies."""
    from .kv_cache import estimate_kv_cache
    from .memory import _layers_on_rank

    return float(
        estimate_kv_cache(
            inference_config, _layers_on_rank(inference_config), concurrency=1, context_len=1
        ).bytes_per_token
    )


def hbm_kv_pool_tokens(inference_config, hbm_gb: float | None) -> int:
    """Tokens of KV the HBM left after weights and activations holds, per replica.

    The same budget the memory projection reports, expressed in tokens so the
    DES can size its pool and its HBM block tier from it when no engine log
    says what the pool was.
    """
    from .memory import project_inference_memory

    if not hbm_gb:
        return 0
    mem = project_inference_memory(inference_config, hbm_capacity_gb=hbm_gb, verbose=False)
    usable = mem.hbm_capacity_bytes or 0
    frac = inference_config.request_config.kv_cache_memory_fraction
    if frac:
        usable = int(usable * float(frac))
    free = usable - mem.weight_bytes - mem.activation_bytes
    bpt = kv_bytes_per_token(inference_config)
    if free <= 0 or bpt <= 0:
        return 0
    return int(free // bpt)


def describe(
    tiers: list[KvTier], bytes_per_token: float, block_size: int, seq_tokens: int = 0
) -> list[str]:
    """One line per tier: capacity, bandwidth, and what that is in tokens --
    and, given a sequence length, in whole sequences of it."""
    out = []
    for t in tiers:
        blocks = t.capacity_blocks(bytes_per_token, block_size)
        seqs = (
            f", {blocks * block_size // seq_tokens} seqs of {seq_tokens}" if seq_tokens > 0 else ""
        )
        out.append(
            f"{t.name}: {t.capacity_gb:g} GB/GPU @ {t.bw_gbps:g} GB/s, "
            f"{t.latency_us:g} us/block -> {blocks * block_size / 1e6:.2f}M tokens "
            f"({blocks} blocks{seqs}), {t.block_load_ms(bytes_per_token, block_size):.3f} ms/block"
        )
    return out

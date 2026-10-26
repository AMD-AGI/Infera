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

The exception is a tier whose keyspace belongs to the node rather than to the
GPU (UMBP, below): what it holds in distinct tokens depends on how the worker
keys it, so it is entered as the node total and converted to the per-GPU
convention from the layout of the pool it serves (:meth:`KvTier.for_layout`).

Placement is capacity-aware and recency-ordered: a block enters HBM when it is
computed or loaded, and when a tier is full its least-recently-used *tail*
block moves one tier down rather than being dropped -- the same leaf-first rule
the HBM radix cache already uses, applied per tier, so a shared head is never
pushed below its own continuation. Only the last tier drops blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

_GiB = 1024.0**3

# name -> (per-GPU GB/s into HBM, per-block access latency in us)
#
# dram: PCIe 5 x16 host link, ~64 GB/s nominal.
# ssd:  a node's NVMe array shared by its GPUs -- 8 Gen5 drives at ~14 GB/s
#       sequential read behind 8 GPUs is ~14 GB/s each before contention, and
#       ~12 after the filesystem and the host bounce buffer.
# remote: rack- or cluster-level KV storage over the GPU's RDMA NIC; 400 Gb/s
#       is 50 GB/s on the wire and the store, not the NIC, usually binds.
# umbp: MoRI's UMBP standalone server, a per-node host-DRAM store that SGLang's
#       unified radix tree loads and offloads pages against directly, with no
#       HiCache host tier in between (InferenceX's umbp-linker arm). Its data
#       plane is MoRI over host memory registered for GPU access, so a load
#       crosses the same PCIe 5 x16 host link as ``dram`` and takes its
#       numbers. The per-request lookup against the standalone server (gRPC
#       over a unix socket) is not modelled: nothing has measured it.
TIER_DEFAULTS: dict[str, tuple[float, float]] = {
    "dram": (64.0, 2.0),
    "host": (64.0, 2.0),
    "cxl": (48.0, 4.0),
    "ssd": (12.0, 80.0),
    "nvme": (12.0, 80.0),
    "remote": (25.0, 20.0),
    "umbp": (64.0, 2.0),
}

# Tiers whose capacity is given as the node's total rather than per GPU.
NODE_TIERS = frozenset({"umbp"})


@dataclass(frozen=True)
class KvTier:
    """One storage tier below HBM, per GPU.

    ``per_node`` marks a capacity that is still the node's total and has to go
    through :meth:`for_layout` before it sizes anything; ``node_gb`` keeps that
    total on the converted tier so it can be reported.
    """

    name: str
    capacity_gb: float
    bw_gbps: float
    latency_us: float = 0.0
    per_node: bool = False
    node_gb: float = 0.0

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
        ``dram:512:64:2``. Bandwidth and latency default by name. A node tier
        (``umbp:1500``) takes its capacity as the node's total."""
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
        return cls(
            name=name,
            capacity_gb=float(parts[1]),
            bw_gbps=float(bw),
            latency_us=float(lat),
            per_node=name in NODE_TIERS,
        )

    def for_layout(self, ranks: int, attention_dp: int = 1) -> KvTier:
        """This node tier in the per-GPU convention, for a worker of ``ranks``
        GPUs running ``attention_dp``-way attention-DP.

        The DES sizes a tier at ``capacity_gb / per-GPU bytes per token``
        distinct tokens, so the per-GPU-equivalent capacity is whatever makes
        that come out at what the node actually holds. How UMBP keys a page
        decides it (InferenceX ``server_sglang.sh``, umbp-linker arm):

        * Pure TP: every rank's keys carry its own suffix and the MLA latent is
          replicated across TP, so each rank stores its own copy and the node
          holds ``node / (ranks * per-GPU bytes)`` distinct tokens -- the
          node's total split over its GPUs, the same as a per-GPU tier.
        * Attention-DP: the keys collapse to rank 0 and the node holds one
          shared keyspace, every token once, so ``node / per-GPU bytes``
          distinct tokens: ``node * dp / ranks`` per GPU, which is the whole
          node when every rank is a DP rank.

        One worker per node is assumed: the recipe refuses multi-node prefill
        workers, and two workers sharing a node's server is not a layout it
        runs. A tier that was already per GPU is returned unchanged.
        """
        if not self.per_node:
            return self
        ranks = max(1, int(ranks))
        dp = min(ranks, max(1, int(attention_dp or 1)))
        return replace(
            self,
            capacity_gb=self.capacity_gb * dp / ranks,
            per_node=False,
            node_gb=self.capacity_gb,
        )

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


def tiers_for_pool(tiers: list[KvTier], inference_config) -> list[KvTier]:
    """``tiers`` with every node tier converted for the pool that owns them.

    The worker's GPUs are its TP x PP ranks; its attention-DP degree is the
    pool's own (a split's prefill pool sets it separately from decode).
    """
    from .kv_cache import attention_dp_size

    mp = inference_config.model_parallel_config
    ranks = max(1, int(mp.tensor_model_parallel_size or 1)) * max(
        1, int(mp.pipeline_model_parallel_size or 1)
    )
    dp = attention_dp_size(inference_config)
    return [t.for_layout(ranks, dp) for t in tiers]


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
        if t.per_node:
            size = f"{t.capacity_gb:g} GB/node (not yet converted per GPU)"
        elif t.node_gb:
            size = f"{t.node_gb:g} GB node total = {t.capacity_gb:g} GB/GPU-equivalent"
        else:
            size = f"{t.capacity_gb:g} GB/GPU"
        out.append(
            f"{t.name}: {size} @ {t.bw_gbps:g} GB/s, "
            f"{t.latency_us:g} us/block -> {blocks * block_size / 1e6:.2f}M tokens "
            f"({blocks} blocks{seqs}), {t.block_load_ms(bytes_per_token, block_size):.3f} ms/block"
        )
    return out

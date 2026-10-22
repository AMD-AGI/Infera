from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.utils import get_dsa_seed_metadata_dim
from sglang.srt.layers.attention.dsa.utils import (
    should_remap_pd_dsa_seed_to_local_slots,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.eagle_info import EagleDraftInput

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


# [yihou] YIHOU_DSA_SEED_GUARD -- mount patch, not upstream (R029). off (default):
# no behaviour change. count: per-site device-side counters of DSA seed entries
# outside [-1, kv_slot_bound) plus draft graph-fallback counters, logged at most
# once a minute. mask: count + set those entries to -1 (and refill the eager
# draft-extend seed buffer with -1 on every reuse, in eagle_worker_v2).
_yihou_logger = logging.getLogger("sglang.yihou.seedguard")
YIHOU_DSA_SEED_GUARD = os.environ.get("YIHOU_DSA_SEED_GUARD", "off").strip().lower()
if YIHOU_DSA_SEED_GUARD not in ("off", "count", "mask"):
    raise ValueError(f"YIHOU_DSA_SEED_GUARD must be off|count|mask, got {YIHOU_DSA_SEED_GUARD!r}")


class _YihouSeedGuardStats:
    def __init__(self):
        self.bad = {}  # site -> device int64 counter of out-of-range entries
        self.rows = {}  # site -> host count of seed rows checked
        self.host = {}  # name -> host int counters (graph fallbacks)
        self.last_log = time.monotonic()
        self.logged = {}

    def maybe_log(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_log < 60.0:
            return
        self.last_log = now
        parts = []
        for site, cnt in self.bad.items():
            total = int(cnt.item())  # one sync per minute
            parts.append(f"{site}.bad_entries={total} {site}.rows={self.rows.get(site, 0)}")
        for name, val in self.host.items():
            parts.append(f"{name}={val}")
        if parts:
            _yihou_logger.warning("YIHOU-SEEDGUARD mode=%s %s", YIHOU_DSA_SEED_GUARD, " ".join(parts))


_yihou_stats = _YihouSeedGuardStats()


def yihou_kv_slot_bound(token_to_kv_pool_allocator) -> int | None:
    """Exclusive upper bound of KV slot ids (KV buffer rows = size + page_size)."""
    size = getattr(token_to_kv_pool_allocator, "size", None)
    if size is None:
        return None
    return int(size) + int(getattr(token_to_kv_pool_allocator, "page_size", 1) or 1)


def yihou_guard_dsa_seed(seed, bound, site: str):
    """Count (and in mask mode, -1) seed entries outside [-1, bound). -1 is the
    valid 'no index' sentinel. Never syncs except in the per-minute log."""
    if YIHOU_DSA_SEED_GUARD == "off" or seed is None or bound is None:
        return seed
    bad = (seed < -1) | (seed >= bound)
    cnt = _yihou_stats.bad.get(site)
    if cnt is None:
        cnt = torch.zeros((), dtype=torch.int64, device=seed.device)
        _yihou_stats.bad[site] = cnt
    cnt += bad.sum()
    _yihou_stats.rows[site] = _yihou_stats.rows.get(site, 0) + int(seed.shape[0])
    if YIHOU_DSA_SEED_GUARD == "mask":
        seed = seed.masked_fill(bad, -1)
    _yihou_stats.maybe_log()
    return seed


def yihou_count(name: str, inc: int = 1) -> None:
    if YIHOU_DSA_SEED_GUARD == "off":
        return
    _yihou_stats.host[name] = _yihou_stats.host.get(name, 0) + inc
    _yihou_stats.maybe_log()


def _requires_dsa_seed_for_cuda_graph(hf_config, topk: int) -> bool:
    return topk == 1 and get_dsa_seed_metadata_dim(hf_config) > 0


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> EagleDraftInput:
    # Adaptive spec moves the step count after publish, and this runs once per
    # prebuilt batch.
    spec = get_spec()
    num_states = spec.speculative_eagle_topk
    if spec.enable_multi_layer_eagle:
        num_states *= spec.speculative_num_steps

    topk_p = torch.stack(
        [
            torch.as_tensor(
                req.output_topk_p[:num_states],
                device=batch.device,
                dtype=torch.float32,
            )
            for req in batch.reqs
        ],
        dim=0,
    )
    topk_index = torch.stack(
        [
            torch.as_tensor(
                req.output_topk_index[:num_states],
                device=batch.device,
                dtype=torch.int64,
            )
            for req in batch.reqs
        ],
        dim=0,
    )

    hidden_states = torch.stack(
        [req.hidden_states_tensor for req in batch.reqs], dim=0
    ).to(batch.device)

    dsa_topk_indices = None
    dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
    if dsa_indices_list and all(t is not None for t in dsa_indices_list):
        dsa_topk_indices = torch.stack(dsa_indices_list, dim=0).to(batch.device)
        if should_remap_pd_dsa_seed_to_local_slots():
            # PD sends request-relative positions; fused TopK consumes
            # decode-local physical slots. Remap once before the draft loop/graph.
            req_to_token = batch.req_to_token_pool.req_to_token
            table_width = req_to_token.shape[1]
            valid_positions = dsa_topk_indices >= 0
            gather_positions = dsa_topk_indices.clamp(min=0, max=table_width - 1).to(
                torch.int64
            )
            local_slots = req_to_token[
                batch.req_pool_indices[:, None], gather_positions
            ]
            invalid_rows = torch.any(
                (dsa_topk_indices < -1)
                | (dsa_topk_indices >= batch.seq_lens[:, None])
                | (dsa_topk_indices >= table_width)
                # Slot 0 is the reserved padding sink; real KV allocations
                # start at 1, and untouched req-to-token entries remain 0.
                | (valid_positions & (local_slots <= 0)),
                dim=1,
            )
            local_slots.masked_fill_(~valid_positions, -1)
            local_slots.masked_fill_(invalid_rows[:, None], -1)
            dsa_topk_indices = local_slots
        dsa_topk_indices = yihou_guard_dsa_seed(
            dsa_topk_indices,
            yihou_kv_slot_bound(batch.token_to_kv_pool_allocator),
            "pd_seed",
        )
        if torch.any(torch.all(dsa_topk_indices < 0, dim=1)).item():
            dsa_topk_indices = None

    requires_dsa_seed_for_cuda_graph = _requires_dsa_seed_for_cuda_graph(
        batch.model_config.hf_config,
        spec.speculative_eagle_topk,
    )

    spec_info = EagleDraftInput(
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
        dsa_topk_indices=dsa_topk_indices,
        cuda_graph_compatible=not (
            requires_dsa_seed_for_cuda_graph and dsa_topk_indices is None
        ),
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
    if not spec_info.cuda_graph_compatible:
        yihou_count("pd_seedless_batches")

    if batch.enable_overlap:
        spec_info.future_dsa_topk_indices_available = dsa_topk_indices is not None
        spec_info.future_indices = batch.req_pool_indices
        # Seed the relay buf with the known seq_lens; publish's chained record
        # keeps the in-flight forward's fence intact (see FutureMap.publish).
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )

    return spec_info

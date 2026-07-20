###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ATOM Mooncake PD coordination protocol.

ATOM ships its own disaggregation proxy
(``atom.kv_transfer.disaggregation.proxy``) that fronts a prefill/decode
pair: it forwards a single-token, non-streaming request to prefill, waits
for the response to learn the ``transfer_id`` (and the re-predicted
``first_token_id``), then issues the decode request carrying that hand-off
plus the peer's mooncake bootstrap addressing. infera replaces that proxy
with its own :class:`~infera.router.disagg.DisaggRouter`; this protocol
reproduces the exact body shaping the ATOM proxy does.

Because the decode body needs ``transfer_id`` / ``first_token_id`` from the
prefill *response*, the topology is ``serial-pull`` (POST prefill, read its
JSON, then POST decode) — not the concurrent push SGLang / vLLM-mooncake
use.

The mooncake transport is push/WRITE: the consumer (decode) opens a ZMQ
side channel to the producer's (prefill's) ``handshake_port`` and the
producer RDMA-writes KV into the consumer's registered GPU memory. Workers
advertise their ``host`` / ``http_port`` / ``handshake_port`` / ``tp_size``
/ ``dp_size`` in ``disagg_meta['params']`` at registration time.

Only routable through ``/v1/completions``: ATOM threads
``kv_transfer_params`` to and from the engine on the text-completion path
only (the chat path neither accepts nor returns it).

DP-attention (``dp_size > 1``): ATOM internally chooses the producer DP rank
and reports the rank that stored KV as ``dp_rank`` in the prefill response.
Decode must reuse that rank as ``remote_dp_rank`` because the side-channel
port is ``base_handshake_port + dp_rank*tp_size + tp_rank``.

So we mirror ATOM's proxy / atomesh path: pass the full prefill
``kv_transfer_params`` into decode, then enrich only ``remote_dp_size`` /
``remote_tp_size`` and ``remote_dp_rank``. For ``dp_size == 1`` this naturally
collapses to rank 0.
"""

from __future__ import annotations

from typing import Literal

from infera.common.worker_pool import WorkerInfo


def _params(w: WorkerInfo, side: str) -> dict:
    params = w.disagg_meta.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(
            f"{side} worker {w.worker_id}: missing disagg_meta['params'] "
            f"(atom-mooncake needs host/http_port/handshake_port/tp_size/dp_size)"
        )
    return params


def _required(params: dict, key: str, w: WorkerInfo, side: str):
    if key not in params or params[key] is None:
        raise ValueError(f"{side} worker {w.worker_id}: disagg_meta.params[{key!r}] missing")
    return params[key]


class AtomMooncakeProtocol:
    """Serial-pull: POST prefill (1 token), pull transfer_id from its
    response, then POST decode with the mooncake hand-off."""

    name: str = "atom-mooncake"
    topology: Literal["concurrent", "serial-pull"] = "serial-pull"

    def annotate_prefill(self, body: dict, p: WorkerInfo, d: WorkerInfo, room_id: int) -> dict:
        d_params = _params(d, "decode")
        out = dict(body)
        # Prefill only computes the KV cache: single token, non-streaming.
        out["stream"] = False
        out["max_tokens"] = 1
        if "max_completion_tokens" in out:
            out["max_completion_tokens"] = 1
        out.pop("stream_options", None)
        out["kv_transfer_params"] = {
            "remote_dp_size": _required(d_params, "dp_size", d, "decode"),
            "remote_tp_size": _required(d_params, "tp_size", d, "decode"),
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_handshake_port": _required(d_params, "handshake_port", d, "decode"),
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": _required(d_params, "host", d, "decode"),
            "remote_port": _required(d_params, "http_port", d, "decode"),
        }
        return out

    def annotate_decode(
        self,
        body: dict,
        p: WorkerInfo,
        d: WorkerInfo,
        room_id: int,
        prefill_handoff: dict | None,
    ) -> dict:
        p_params = _params(p, "prefill")
        handoff = prefill_handoff or {}
        if "transfer_id" not in handoff:
            raise ValueError(
                "atom-mooncake decode requires 'transfer_id' from the prefill "
                f"hand-off, got keys={sorted(handoff)}"
            )
        out = dict(body)
        # NOTE: do NOT decrement max_tokens — ATOM's decode re-predicts T0
        # (overridden by the prefill's first_token_id) and that step still
        # counts toward max_tokens, so keeping the original value matches
        # the non-PD output length (see the ATOM proxy comment).
        #
        # Reuse the prefill response's kv_transfer_params verbatim (it is the
        # `handoff`): it already carries do_remote_prefill=True /
        # do_remote_decode=False, the producer's real remote_host /
        # remote_port / remote_handshake_port, transfer_id, first_token_id,
        # remote_block_ids, remote_engine_id, local_slot_index, tp_size and
        # the actual dp_rank. This mirrors atomesh's relay, which pipes the
        # prefill response's kv straight into the decode body.
        ktp: dict = dict(handoff)
        # enrich_decode_kv: the three fields the prefill omits but decode's
        # ReqMeta needs. remote_dp_size/remote_tp_size describe the PRODUCER's
        # parallelism (so the consumer maps tp shards + computes the port
        # stride); remote_dp_rank is the producer rank that holds the KV,
        # copied from the response's dp_rank so the consumer dials
        # remote_handshake_port + remote_dp_rank*tp_size + tp_rank. The
        # response's own dp_rank is left in the body (verbatim reuse, as
        # atomesh does); ATOM's consumer ignores it and reads only
        # remote_dp_rank.
        ktp["remote_dp_size"] = _required(p_params, "dp_size", p, "prefill")
        ktp["remote_tp_size"] = _required(p_params, "tp_size", p, "prefill")
        ktp.setdefault("tp_size", ktp["remote_tp_size"])
        dp_rank = handoff.get("dp_rank")
        # bool is an int subclass; guard against a stray True/False.
        if isinstance(dp_rank, int) and not isinstance(dp_rank, bool):
            ktp["remote_dp_rank"] = dp_rank
        out["kv_transfer_params"] = ktp
        return out

    def extract_handoff(self, prefill_response_payload: dict) -> dict:
        kv = prefill_response_payload.get("kv_transfer_params")
        if not isinstance(kv, dict):
            raise ValueError(
                "atom-mooncake prefill response missing top-level "
                "'kv_transfer_params' (route prefill via /v1/completions, "
                "not /v1/chat/completions)"
            )
        if "transfer_id" not in kv:
            raise KeyError("atom-mooncake prefill response kv_transfer_params lacks 'transfer_id'")
        # Hand the FULL prefill-response kv_transfer_params to annotate_decode:
        # the decode leg reuses it verbatim (+ enrichment), exactly like ATOM's
        # own proxy / atomesh mesh. A shallow copy keeps the router from
        # mutating the parsed response.
        return dict(kv)

    def request_id_for(self, p: WorkerInfo, d: WorkerInfo, room_id: int) -> str | None:
        # ATOM correlates P/D via transfer_id in the body, not the request id.
        return None

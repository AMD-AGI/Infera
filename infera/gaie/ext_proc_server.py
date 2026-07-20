###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Envoy ext_proc (External Processing) gRPC server implementing a GAIE
Endpoint Picker (EPP) on top of Infera's routing brain.

An Envoy-based Inference Gateway (kGateway) opens a bidirectional ``Process``
stream per HTTP request and asks this server which backend endpoint to route
to. Because the model name and prompt live in the request *body* (not the
headers), the server buffers the request body, runs :class:`EndpointPicker`,
and answers with header mutations:

  * ``x-gateway-destination-endpoint`` — host:port the gateway routes to
    (the decode/primary worker's frontend sidecar = the InferencePool member).
  * ``x-worker-instance-id`` — the chosen worker's logical id (honoured by a
    worker-side frontend running ``--router-mode direct``).
  * ``x-prefill-instance-id`` — the prefill worker id (disaggregated only).

Protocol note — FULL_DUPLEX_STREAMED:
    kGateway configures the ext_proc filter with ``request_body_mode`` and
    ``response_body_mode`` = ``FULL_DUPLEX_STREAMED``. In that mode Envoy
    streams the body to the EPP and expects the EPP to stream it back via
    ``BodyMutation.streamed_response`` (the EPP is on the data path); it does
    NOT expect a 1:1 response per streamed chunk. We therefore:
      * answer request/response *headers* 1:1 with CONTINUE (header_mode=SEND),
      * buffer request *body* chunks and emit a SINGLE request_body response on
        end_of_stream that carries the routing header mutations + the buffered
        body echoed back, and
      * pass response *body* chunks straight through (echo each chunk back).
    Emitting an extra/unsolicited response triggers Envoy's "Spurious response
    message" guard and resets the stream (HTTP 500).

The picker's in-flight bookkeeping is released when the per-request stream
ends (success, client disconnect, or error).
"""

from __future__ import annotations

import json
import logging

from infera.gaie.endpoint_picker import EndpointPicker, PickResult
from infera.gaie.proto import ext_proc_pb2 as pb
from infera.gaie.proto import ext_proc_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

DEST_HEADER = "x-gateway-destination-endpoint"
WORKER_HEADER = "x-worker-instance-id"
PREFILL_HEADER = "x-prefill-instance-id"


def _set_header(name: str, value: str) -> pb.HeaderValueOption:
    """Build an overwrite-or-add header mutation. The value goes in
    ``raw_value`` (bytes), which is what current Envoy expects for ext_proc
    header mutations (the string ``value`` field is deprecated)."""
    return pb.HeaderValueOption(
        header=pb.HeaderValue(key=name, raw_value=value.encode()),
        append_action=pb.OVERWRITE_IF_EXISTS_OR_ADD,
    )


class ExtProcServicer(pb_grpc.ExternalProcessorServicer):
    """Bidirectional ext_proc handler. One ``Process`` stream == one HTTP
    request/response lifecycle. Implements FULL_DUPLEX_STREAMED body handling
    (see module docstring)."""

    def __init__(self, picker: EndpointPicker) -> None:
        self._picker = picker

    async def Process(self, request_iterator, context):  # noqa: N802 (gRPC name)
        pick: PickResult | None = None
        req_body = bytearray()
        routed = False
        try:
            async for req in request_iterator:
                kind = req.WhichOneof("request")
                if kind == "request_headers":
                    if req.request_headers.end_of_stream:
                        # No request body (e.g. GET /v1/models): decide now and
                        # mutate routing headers on the headers response.
                        pick = self._route(b"")
                        routed = True
                        yield self._headers_decision(pick)
                    else:
                        # Body is coming; defer the decision and CONTINUE.
                        yield _continue(request_headers=True)
                elif kind == "request_body":
                    req_body.extend(req.request_body.body)
                    if req.request_body.end_of_stream:
                        if not routed:
                            pick = self._route(bytes(req_body))
                            routed = True
                        # Single response carrying routing headers + the buffered
                        # body streamed back to Envoy with end_of_stream.
                        yield self._request_body_decision(pick, bytes(req_body))
                    # Non-final chunk: accumulate only (no response in duplex mode).
                elif kind == "response_headers":
                    yield _continue(request_headers=False)
                elif kind == "response_body":
                    # Pass the upstream response body straight through, chunk by
                    # chunk, preserving streaming (e.g. SSE for stream=true).
                    yield _passthrough_body(req.response_body.body, req.response_body.end_of_stream)
                elif kind == "request_trailers":
                    yield pb.ProcessingResponse(request_trailers=pb.TrailersResponse())
                elif kind == "response_trailers":
                    yield pb.ProcessingResponse(response_trailers=pb.TrailersResponse())
        finally:
            if pick is not None:
                self._picker.release(pick)

    # --- routing ---

    def _route(self, raw_body: bytes) -> PickResult | None:
        body: dict = {}
        if raw_body:
            try:
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict):
                    body = parsed
            except ValueError:
                logger.warning("ext_proc: request body is not valid JSON; routing without it")
        model = body.get("model")
        try:
            result = self._picker.pick(model, body)
        except Exception:
            logger.exception("ext_proc: pick failed for model=%r", model)
            return None
        if result is None:
            logger.warning("ext_proc: no active worker for model=%r", model)
        else:
            logger.info(
                "ext_proc: routed model=%r -> endpoint=%s worker=%s prefill=%s",
                model,
                result.destination_endpoint,
                result.worker_instance_id,
                result.prefill_instance_id or "-",
            )
        return result

    # --- response builders ---

    def _routing_headers(self, pick: PickResult | None) -> pb.CommonResponse:
        # clear_route_cache so the gateway re-evaluates routing against the
        # destination header we just set.
        common = pb.CommonResponse(clear_route_cache=True)
        if pick is not None:
            opts = [
                _set_header(DEST_HEADER, pick.destination_endpoint),
                _set_header(WORKER_HEADER, pick.worker_instance_id),
            ]
            if pick.prefill_instance_id:
                opts.append(_set_header(PREFILL_HEADER, pick.prefill_instance_id))
            common.header_mutation.set_headers.extend(opts)
        return common

    def _headers_decision(self, pick: PickResult | None) -> pb.ProcessingResponse:
        return pb.ProcessingResponse(
            request_headers=pb.HeadersResponse(response=self._routing_headers(pick))
        )

    def _request_body_decision(self, pick: PickResult | None, body: bytes) -> pb.ProcessingResponse:
        common = self._routing_headers(pick)
        # Echo the buffered body back to Envoy so it can forward it upstream
        # (FULL_DUPLEX_STREAMED: the EPP is on the request data path).
        common.body_mutation.streamed_response.body = body
        common.body_mutation.streamed_response.end_of_stream = True
        return pb.ProcessingResponse(request_body=pb.BodyResponse(response=common))


def _continue(*, request_headers: bool) -> pb.ProcessingResponse:
    """No-op CONTINUE for a headers phase (header_mode=SEND is 1:1)."""
    resp = pb.HeadersResponse(response=pb.CommonResponse())
    if request_headers:
        return pb.ProcessingResponse(request_headers=resp)
    return pb.ProcessingResponse(response_headers=resp)


def _passthrough_body(chunk: bytes, end_of_stream: bool) -> pb.ProcessingResponse:
    """Echo a response body chunk straight back (FULL_DUPLEX_STREAMED)."""
    common = pb.CommonResponse()
    common.body_mutation.streamed_response.body = chunk
    common.body_mutation.streamed_response.end_of_stream = end_of_stream
    return pb.ProcessingResponse(response_body=pb.BodyResponse(response=common))

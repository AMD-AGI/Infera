#!/usr/bin/env python3
"""Persist OTLP/gRPC spans as normalized, append-only JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import threading
from concurrent import futures
from pathlib import Path
from typing import Any

import grpc
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def any_value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    raw = getattr(value, kind)
    if kind == "array_value":
        return [any_value(item) for item in raw.values]
    if kind == "kvlist_value":
        return {item.key: any_value(item.value) for item in raw.values}
    if kind == "bytes_value":
        return bytes(raw).hex()
    return raw


def attributes(values: Any) -> dict[str, Any]:
    return {item.key: any_value(item.value) for item in values}


class Collector(TraceServiceServicer):
    def __init__(self, output: Path):
        self._lock = threading.Lock()
        self._stream = output.open("x", encoding="utf-8", buffering=1)
        self._stream.write(
            json.dumps(
                {
                    "record_type": "metadata",
                    "schema_version": 1,
                    "started_at": utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        self.span_count = 0

    def close(self) -> None:
        with self._lock:
            self._stream.flush()
            self._stream.close()

    def Export(self, request, context):  # noqa: N802 - OTLP API name
        received_at = utc_now()
        rows = []
        for resource_spans in request.resource_spans:
            resource_attrs = attributes(resource_spans.resource.attributes)
            service_name = str(resource_attrs.get("service.name", ""))
            for scope_spans in resource_spans.scope_spans:
                scope_name = scope_spans.scope.name
                scope_version = scope_spans.scope.version
                for span in scope_spans.spans:
                    rows.append(
                        {
                            "record_type": "span",
                            "schema_version": 1,
                            "received_at": received_at,
                            "service_name": service_name,
                            "resource_attributes": resource_attrs,
                            "scope_name": scope_name,
                            "scope_version": scope_version,
                            "trace_id": bytes(span.trace_id).hex(),
                            "span_id": bytes(span.span_id).hex(),
                            "parent_span_id": bytes(span.parent_span_id).hex(),
                            "name": span.name,
                            "kind": int(span.kind),
                            "start_time_ns": int(span.start_time_unix_nano),
                            "end_time_ns": int(span.end_time_unix_nano),
                            "attributes": attributes(span.attributes),
                            "events": [
                                {
                                    "name": event.name,
                                    "time_ns": int(event.time_unix_nano),
                                    "attributes": attributes(event.attributes),
                                }
                                for event in span.events
                            ],
                            "status_code": int(span.status.code),
                            "status_message": span.status.message,
                        }
                    )
        with self._lock:
            for row in rows:
                self._stream.write(json.dumps(row, sort_keys=True) + "\n")
            self.span_count += len(rows)
        return ExportTraceServiceResponse()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=4317)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not 1 <= args.port <= 65535 or args.workers <= 0:
        parser.error("invalid port or worker count")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    collector = Collector(args.output)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
    )
    add_TraceServiceServicer_to_server(collector, server)
    bound_port = server.add_insecure_port(f"0.0.0.0:{args.port}")
    if bound_port != args.port:
        collector.close()
        raise SystemExit(f"failed to bind OTLP port {args.port}")
    server.start()
    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(
            json.dumps(
                {
                    "ready_at": utc_now(),
                    "port": args.port,
                    "output": str(args.output),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    stopping = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopping.wait(1.0):
        pass
    server.stop(grace=10).wait()
    collector.close()
    print(json.dumps({"spans": collector.span_count, "stopped_at": utc_now()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

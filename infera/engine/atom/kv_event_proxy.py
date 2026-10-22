###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Process-safe relay for ATOM KV-cache events.

ATOM creates one ``EngineCore`` process per data-parallel rank. Each process
owns an independent ``BlockManager`` and therefore needs to publish its own KV
events, while Infera advertises one logical KV-event endpoint per ATOM worker.
The relay gives all EngineCore publishers a local XSUB ingress and forwards
their streams through one externally advertised XPUB endpoint.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid

import zmq

logger = logging.getLogger(__name__)


class AtomKvEventProxy:
    """Relay multiple child PUB sockets through one advertised endpoint."""

    def __init__(self, external_bind_endpoint: str) -> None:
        suffix = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.external_bind_endpoint = external_bind_endpoint
        self.ingress_endpoint = f"ipc:///tmp/infera-atom-kv-{suffix}.sock"
        self._control_endpoint = f"inproc://infera-atom-kv-control-{suffix}"
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._controller: zmq.Socket | None = None

    def start(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="atom-kv-event-proxy",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("ATOM KV-event proxy did not start")
        if self._error is not None:
            raise RuntimeError("ATOM KV-event proxy failed to start") from self._error

        controller = zmq.Context.instance().socket(zmq.PAIR)
        controller.setsockopt(zmq.LINGER, 0)
        controller.connect(self._control_endpoint)
        self._controller = controller

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        if self._controller is not None:
            try:
                self._controller.send(b"TERMINATE")
            except zmq.ZMQError:
                logger.exception("failed to stop ATOM KV-event proxy cleanly")
            self._controller.close(linger=0)
            self._controller = None
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error("ATOM KV-event proxy did not stop within %.1fs", timeout)
        self._thread = None

    def _run(self) -> None:
        context = zmq.Context.instance()
        ingress = context.socket(zmq.XSUB)
        egress = context.socket(zmq.XPUB)
        control = context.socket(zmq.PAIR)
        for socket in (ingress, egress, control):
            socket.setsockopt(zmq.LINGER, 0)
        try:
            ingress.bind(self.ingress_endpoint)
            egress.bind(self.external_bind_endpoint)
            control.bind(self._control_endpoint)
            logger.info(
                "ATOM kv-events: proxy ingress=%s advertise-bind=%s",
                self.ingress_endpoint,
                self.external_bind_endpoint,
            )
            self._ready.set()
            zmq.proxy_steerable(ingress, egress, None, control)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            logger.exception("ATOM KV-event proxy failed")
        finally:
            ingress.close(linger=0)
            egress.close(linger=0)
            control.close(linger=0)

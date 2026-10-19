#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Keep the API process heap flat so its gen-2 GC pauses stay short.

Under AgentX load both ATOM API processes spent ~45% of the wall clock in
stop-the-world gen-2 GC passes, nearly all triggered inside a /metrics render,
and each pass grew with the number of completed requests (D ~0.08 ms per
request: ~0.65 s per pass after an hour). Every pass freezes the event loop,
so request admission, the P result hand-back and SSE delivery all wait; it
added ~0.7 s to each TTFT. Two things grow with completed requests:

1. ``CoreManager.outputs_queue``. The output thread put every batch of finished
   sequences on it, but only ``LLMEngine.step()`` / ``generate()`` and asyncio
   mode read it; the OpenAI server delivers through the per-sequence stream
   callbacks instead. So every finished ``Sequence`` (token arrays, KV transfer
   block-id lists) stayed reachable for the life of the process and each gen-2
   pass walked all of them -- the same leak ATOM already fixed for
   ``stream_outputs_queue``. The API server is the only caller that wires
   ``_flush_stream_batch_fn``; with it set and asyncio mode off, nothing will
   ever read the queue, so the batch is not queued.
2. The per-request context gauges (``atom:{prefill,decode}_request_context_tokens``,
   labelled by request id and never removed). The multiprocess collector
   re-reads and json-parses every series on each render, so render time and the
   objects it promotes grow with every request served. They are opt-in now:
   ``ATOM_METRICS_PER_REQUEST_CONTEXT=1`` records them as before.

Written against ATOM 661ef766 in rocm/atom-dev:nightly_202610131505.
Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

PATCH = {
    "model_engine/engine_core_mgr.py": (
        "# infera: nothing reads outputs_queue online",
        [
            (
                "                        self.outputs_queue.put_nowait(seqs)\n",
                "                        # infera: nothing reads outputs_queue online. The\n"
                "                        # API server (the one caller that wires the flush\n"
                "                        # hook) delivers through the stream callbacks; only\n"
                "                        # LLMEngine.step()/generate() and asyncio mode read\n"
                "                        # this queue. Queued anyway, every finished Sequence\n"
                "                        # stayed reachable and each gen-2 GC pass grew.\n"
                "                        if self._flush_stream_batch_fn is None or getattr(\n"
                "                            self, \"_asyncio_mode\", False\n"
                "                        ):\n"
                "                            self.outputs_queue.put_nowait(seqs)\n",
            ),
        ],
    ),
    "metrics/scheduler.py": (
        "# infera: per-request context gauges are opt-in",
        [
            ("import time\n", "import os\nimport time\n"),
            (
                "    def _record_request_context(self, metric, req_id, seq, tokens):\n",
                "    # infera: per-request context gauges are opt-in. Each request adds a\n"
                "    # series that is never removed, and every /metrics render of the API\n"
                "    # process re-reads all of them (render time and GC grow with requests).\n"
                "    _PER_REQUEST = os.environ.get(\"ATOM_METRICS_PER_REQUEST_CONTEXT\") == \"1\"\n"
                "\n"
                "    def _record_request_context(self, metric, req_id, seq, tokens):\n"
                "        if not self._PER_REQUEST:\n"
                "            return\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch api-gc-stalls", PATCH, package="atom"))

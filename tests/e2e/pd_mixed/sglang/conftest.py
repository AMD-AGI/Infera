###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SGLang engine adapter + the shared ``worker`` fixture.

Workers are spawned as subprocesses via ``python -m infera.engine.sglang``
(required: ``SglangEngine.stop()`` kills all children of the current PID). Only
the engine-specific argv mapping lives here; the spawn lifecycle + ``worker``
fixture are shared (see :mod:`tests.e2e.harness.fixtures`).
"""

from __future__ import annotations

from ...harness import EngineAdapter, EngineParams
from ...harness.fixtures import make_worker_fixture


class SglangAdapter(EngineAdapter):
    engine = "sglang"
    module = "infera.engine.sglang"

    def gpus_per_worker(self, params: EngineParams) -> int:
        return max(1, params.tensor_parallel_size)

    def pick_port(self) -> int:
        # SGLang derives several *fixed* TCP ports from the base HTTP port. The
        # one that bites us is DP attention: with --enable-dp-attention the
        # dist-init / tokenizer / detokenizer / rpc / metrics / scheduler-input
        # ports become TCP (not IPC) and sit at base+ZMQ_TCP_PORT_DELTA(=233) ..
        # base+239 (see PortArgs.init_new in sglang server_args.py). There's also
        # base+10000 for the optional gRPC port and base+0..31 for a few ZMQ
        # control sockets.
        #
        # The flaky "scheduler_input_port at <p> is not available in 30 seconds"
        # abort happens when one of those derived ports lands INSIDE the OS
        # ephemeral range (/proc/sys/net/ipv4/ip_local_port_range, 32768-60999 on
        # our hosts): between our free-check here and sglang actually binding the
        # port, the kernel hands that exact port to some *other* bind(("",0))
        # socket — frequently one of the engine's OWN startup sockets — so sglang
        # sees the port "already in use" (by itself, same pid) and dies before
        # becoming ready. The kernel allocates ephemeral ports from the low end
        # up, so a base just above 32768 (derived band ~33000) fails often, while
        # a high base usually gets lucky — hence the intermittent failures.
        #
        # Root-cause fix: only ever hand out a base whose ENTIRE derived TCP
        # footprint lies strictly BELOW the ephemeral range, so the kernel can
        # never auto-assign any of these ports to an unrelated socket during
        # startup. We still free-check the whole footprint (a plain bind() with
        # no SO_REUSEADDR also treats a TIME_WAIT port as busy, which is what we
        # want) to avoid a lingering just-torn-down worker.
        import random
        import socket

        def _free(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return True
                except OSError:
                    return False

        # base+0..31 (ZMQ ctrl), base+233..239 (DP-attention TCP band),
        # base+10000 (gRPC). The last is the widest offset and caps the base.
        derived_offsets = [*range(32), *range(233, 240), 10000]
        try:
            eph_low = int(open("/proc/sys/net/ipv4/ip_local_port_range").read().split()[0])
        except (OSError, ValueError, IndexError):
            eph_low = 32768
        # Keep base + every derived offset strictly below the ephemeral range.
        lo = 2048
        hi = eph_low - max(derived_offsets) - 1
        if hi <= lo:  # pathologically small ephemeral range; degrade gracefully
            hi = eph_low - 1
        for _ in range(500):
            base = random.randint(lo, hi)
            if all(_free(base + off) for off in derived_offsets):
                return base
        raise RuntimeError(
            "no free base port with the whole SGLang derived TCP footprint "
            "(base+0..31, +233..239, +10000) below the ephemeral range"
        )

    def build_argv(
        self,
        params: EngineParams,
        *,
        port: int,
        host: str,
        server_ctx: dict,
        gpu_ids: list[int],
    ) -> list[str]:
        argv = [
            "python3",
            "-m",
            self.module,
            "--model-path",
            params.model,
            "--port",
            str(port),
            "--host",
            host,
            # Pin discovery to etcd (launcher default is kubernetes) so the
            # worker registers in the prefix the in-process server watches.
            "--discovery-backend",
            "etcd",
            "--etcd-endpoint",
            server_ctx["etcd_endpoint"],
            "--etcd-prefix",
            server_ctx["etcd_prefix"],
            # The in-process server routes over HTTP + subscribes to KV events
            # over ZMQ; pin both off the launcher's NATS defaults or the worker
            # can't be reached.
            "--request-transport",
            "http",
            "--kv-event-transport",
            "zmq",
            "--trust-remote-code",
            # e2e sends single requests; cap CUDA-graph capture (default bs=512)
            # so tp>1 MoE warmup doesn't blow past the ready timeout (larger
            # batches fall back to eager, correctness unchanged).
            "--cuda-graph-max-bs",
            "8",
        ]

        tp = max(1, params.tensor_parallel_size)
        if tp > 1:
            argv += ["--tp-size", str(tp)]
        if params.dp_attention:
            argv += ["--enable-dp-attention", "--dp-size", str(tp)]
        if params.expert_parallel:
            argv += ["--ep-size", str(tp)]

        argv += list(params.extra_args)
        return argv


worker = make_worker_fixture(SglangAdapter)

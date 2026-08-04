###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A worker that registers into the fleet and serves tokens, without a GPU.

Everything above the engine — discovery, routing, failover, the circuit breaker,
drain, and any autoscaling loop — is testable without weights. What blocked that
until now was simply that there was no way to *be* a worker without loading a
model: ``tests/e2e/harness`` starts real engines in containers, so the cheapest
fleet anyone could build cost a GPU and a multi-minute weight load per member.

This is that missing piece. It registers through the **real** registration
clients with a real :class:`EngineConfig`, so the fake cannot drift from the
contract a genuine worker satisfies — if the payload changes, this changes with
it or fails loudly. What it fakes is only what happens after a request arrives.

The parts worth having are the ones that make the *hard* problems reproducible:

* ``--startup-delay-s`` simulates the 5-15 minute weight load. That delay is the
  single biggest reason naive autoscaling overshoots — an unready replica counts
  as consuming 0% of the metric, so the loop keeps asking for more. Reproducing
  it costs nothing here and a GPU-hour on real hardware.
* ``--max-concurrency`` gives requests somewhere to queue, which is what makes
  ``num_requests_waiting`` mean anything. Without a queue the metric everyone
  scales on is identically zero.
* SIGTERM deregisters *before* draining, mirroring the real worker, so the
  window where a terminating pod is still receiving traffic is observable.

Not simulated, deliberately: real KV transfer. A PD fake accepts the bootstrap
fields and answers, but no KV moves between prefill and decode. That makes it
useful for testing pool membership, routing and scaling, and useless for testing
the transfer itself. Do not use it to conclude anything about Mooncake.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from infera.common.worker_pool import DisaggMode, EngineType, KvRegistrationMetadata
from infera.engine.base import EngineConfig
from infera.router.disagg_protocols import _PROTOCOLS
from infera.router.dp_routing import DP_RANK_HEADER

logger = logging.getLogger("infera.fakeworker")

# Roughly the shape of an English word, so token counts and byte counts stay in
# a believable ratio for anything measuring throughput.
_FILLER = "token "


@dataclass
class Behaviour:
    """What this worker pretends about its own performance."""

    ttft_ms: float = 50.0
    itl_ms: float = 10.0
    max_concurrency: int = 8
    #: Requests beyond max_concurrency wait here; this is what makes the
    #: queue-depth metric — the one the whole industry scales on — non-zero.
    max_kv_blocks: int = 1024
    blocks_per_request: int = 8
    fail_rate: float = 0.0
    #: Serve 5xx for the first N requests, then recover. For exercising the
    #: circuit breaker's half-open probe without killing the process.
    fail_first: int = 0


@dataclass
class State:
    running: int = 0
    waiting: int = 0
    served: int = 0
    failed: int = 0
    ready: bool = False
    started_at: float = field(default_factory=time.monotonic)
    _sem: asyncio.Semaphore | None = None
    draining: bool = False

    #: Per-DP-rank request counts, keyed by the X-Data-Parallel-Rank header the
    #: router sent. With a real engine you cannot easily see which rank the
    #: router *intended* -- the engine just serves. Counting them here is what
    #: makes DP-attention routing assertable at all.
    by_dp_rank: dict[str, int] = field(default_factory=dict)
    #: The PD handoff fields the router injected on the last request. Whether
    #: the router shaped the body correctly is invisible from outside; a real
    #: engine either works or hangs on KVPoll with no explanation.
    last_handoff: dict = field(default_factory=dict)


def deterministic_canary(model_name: str) -> list[int]:
    """A stand-in for the real tokenizer canary.

    Real workers tokenize a fixed probe string and register the ids, so that a
    fleet running mismatched tokenizers under one model name is rejected at
    registration rather than at inference time. A fake has no tokenizer, but it
    still has to agree with *other fakes* for the same model, or the second one
    to register is silently dropped from the pool.

    Deriving it from the model name gives exactly that: all fakes for a model
    agree, and different models differ. It will not match a real worker's canary
    — see the README; do not mix fakes and real workers under one model name.
    """
    h = hashlib.sha256(f"infera-fake-canary::{model_name}".encode()).digest()
    return [int.from_bytes(h[i : i + 2], "big") for i in range(0, 16, 2)]


def build_app(cfg: EngineConfig, behaviour: Behaviour, state: State) -> FastAPI:
    app = FastAPI(title="infera fake worker")

    def _engine_metric(name: str) -> str:
        # Engine-native metric names, so a scaling rule written against a fake
        # fleet transfers to a real one unchanged. vLLM's names are from its
        # published metrics doc; SGLang's are second-hand (see README) and worth
        # checking against a live engine before relying on them.
        prefix = "vllm" if cfg.engine == EngineType.VLLM else "sglang"
        sglang = {
            "num_requests_waiting": "num_queue_reqs",
            "num_requests_running": "num_running_reqs",
            "gpu_cache_usage_perc": "token_usage",
        }
        return f"{prefix}:{sglang[name] if prefix == 'sglang' else name}"

    async def _admit() -> bool:
        """Returns False if this request should be rejected outright."""
        if state.draining:
            return False
        if behaviour.fail_first and state.served + state.failed < behaviour.fail_first:
            state.failed += 1
            return False
        return True

    async def _generate(prompt_tokens: int, max_tokens: int):
        """Occupy a concurrency slot for a believable amount of time."""
        assert state._sem is not None
        state.waiting += 1
        async with state._sem:
            state.waiting -= 1
            state.running += 1
            try:
                # TTFT scales with prompt length the way a real prefill does,
                # so prefill-heavy and decode-heavy load look different to
                # anything measuring them.
                await asyncio.sleep(behaviour.ttft_ms / 1000.0 * max(1.0, prompt_tokens / 512))
                for _ in range(max_tokens):
                    yield _FILLER
                    await asyncio.sleep(behaviour.itl_ms / 1000.0)
                state.served += 1
            finally:
                state.running -= 1

    def _parse(body: dict) -> tuple[int, int]:
        text = json.dumps(body.get("messages") or body.get("prompt") or "")
        prompt_tokens = max(1, len(text) // 4)
        return prompt_tokens, int(body.get("max_tokens") or 16)

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()

        # Record what the router decided *before* deciding whether to serve, so
        # a refused request still shows up in the routing evidence.
        rank = request.headers.get(DP_RANK_HEADER) or "-"
        state.by_dp_rank[rank] = state.by_dp_rank.get(rank, 0) + 1
        handoff = {
            k: body[k]
            for k in (
                "bootstrap_host",
                "bootstrap_port",
                "bootstrap_room",
                "disagg_prefill_dp_rank",
                "kv_transfer_params",
            )
            if k in body
        }
        if handoff:
            state.last_handoff = handoff

        if not await _admit():
            return JSONResponse({"error": "fake worker refusing"}, status_code=503)
        prompt_tokens, max_tokens = _parse(body)

        if body.get("stream"):

            async def sse():
                async for chunk in _generate(prompt_tokens, max_tokens):
                    payload = {"choices": [{"delta": {"content": chunk}}]}
                    yield f"data: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        out = "".join([c async for c in _generate(prompt_tokens, max_tokens)])
        return JSONResponse(
            {
                "id": f"fake-{state.served}",
                "model": cfg.model_name,
                "choices": [{"message": {"role": "assistant", "content": out}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max_tokens,
                    "total_tokens": prompt_tokens + max_tokens,
                },
            }
        )

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": cfg.model_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        # Unready until the simulated weight load finishes. This is the whole
        # point of --startup-delay-s: a replica that exists but cannot serve is
        # exactly what makes autoscalers overshoot.
        if not state.ready:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok", "running": state.running, "waiting": state.waiting}

    @app.get("/metrics")
    async def metrics():
        used = min(1.0, (state.running * behaviour.blocks_per_request) / behaviour.max_kv_blocks)
        lines = [
            f"{_engine_metric('num_requests_running')} {state.running}",
            f"{_engine_metric('num_requests_waiting')} {state.waiting}",
            f"{_engine_metric('gpu_cache_usage_perc')} {used:.4f}",
            f"infera_fake_worker_served_total {state.served}",
            f"infera_fake_worker_refused_total {state.failed}",
            f"infera_fake_worker_ready {1 if state.ready else 0}",
            f"infera_fake_worker_draining {1 if state.draining else 0}",
        ]
        for rank, n in sorted(state.by_dp_rank.items()):
            lines.append(f'infera_fake_worker_requests_by_dp_rank{{dp_rank="{rank}"}} {n}')
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/debug/routing")
    async def routing():
        """What the router actually decided, which is otherwise unobservable.

        A real engine given a malformed PD handoff does not complain -- it hangs
        on KVPoll until a ~300s timeout, and the failure surfaces nowhere near
        the router that caused it. This turns that into an assertion.
        """
        return {
            "worker_id": f"{cfg.host}:{cfg.port}",
            "disagg_mode": cfg.disagg_mode.value,
            "dp_rank": cfg.dp_rank,
            "dp_size": cfg.dp_size,
            "requests_by_dp_rank": state.by_dp_rank,
            "last_handoff": state.last_handoff,
        }

    return app


def _disagg_meta(args) -> dict:
    """Mirror what a real worker advertises.

    Only PREFILL carries a bootstrap endpoint; DECODE tags the protocol so the
    router can fail fast on a cross-protocol pairing, and has nothing else to
    say. Getting this wrong does not fail loudly at registration -- it fails at
    the first PD request, as a protocol error that reads like a router bug.
    """
    if args.disagg_mode == "mixed":
        return {}
    params: dict = {}
    if args.disagg_mode == "prefill":
        host = args.advertise_host or args.host
        params["bootstrap_addr"] = f"{host}:{args.bootstrap_port}"
    return {"protocol": args.pd_protocol, "params": params}


def build_config(args) -> EngineConfig:
    kv = None
    if args.kv:
        canary = deterministic_canary(args.model_name)
        kv = KvRegistrationMetadata(
            engine_block_size=args.kv_block_size,
            index_block_size=args.kv_block_size,
            tokenizer=args.model_name,
            tokenizer_digest=hashlib.sha256(args.model_name.encode()).hexdigest(),
            tokenizer_canary=canary,
            supports_events=False,  # no ZMQ publisher; router falls back to load-only
        )
    return EngineConfig(
        model_name=args.model_name,
        host=args.advertise_host or args.host,
        port=args.port,
        engine=EngineType(args.engine),
        disagg_mode=DisaggMode(args.disagg_mode),
        disagg_meta=_disagg_meta(args),
        kv=kv,
        kv_block_size=args.kv_block_size if args.kv else None,
        dp_rank=args.dp_rank,
        dp_size=args.dp_size,
        request_transport="http",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="infera-fake-worker",
        description="Register into an Infera fleet and serve tokens without a GPU.",
    )
    p.add_argument("--model-name", required=True, help="must match what the router routes for")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--advertise-host",
        default=os.environ.get("POD_IP"),
        help="address peers use to reach this worker; defaults to $POD_IP. "
        "0.0.0.0 is never routable from another pod.",
    )
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--engine", default="sglang", choices=[e.value for e in EngineType])
    p.add_argument("--disagg-mode", default="mixed", choices=[m.value for m in DisaggMode])
    p.add_argument(
        "--pd-protocol",
        default="sglang-bootstrap",
        choices=sorted(_PROTOCOLS),
        help="must match the router's registry; a decode worker advertising a "
        "different one is rejected as a protocol mismatch",
    )
    p.add_argument(
        "--bootstrap-port",
        type=int,
        default=8998,
        help="advertised in disagg_meta by a prefill worker. Nothing listens on "
        "it -- no KV is transferred (see README).",
    )
    p.add_argument("--dp-rank", type=int, default=None)
    p.add_argument("--dp-size", type=int, default=None)

    p.add_argument(
        "--discovery-backend",
        default=os.environ.get("INFERA_DISCOVERY_BACKEND", "kubernetes"),
        choices=["kubernetes", "etcd"],
    )
    p.add_argument("--etcd-endpoint", default=os.environ.get("INFERA_ETCD_ENDPOINT"))
    p.add_argument("--etcd-prefix", default="/infera/workers/")

    p.add_argument(
        "--kv",
        action="store_true",
        help="register a KV metadata block with a synthetic tokenizer canary. "
        "All fakes for a model agree; a fake and a REAL worker will not -- the "
        "second to register is silently dropped. See the README.",
    )
    p.add_argument("--kv-block-size", type=int, default=64)

    p.add_argument("--ttft-ms", type=float, default=50.0)
    p.add_argument("--itl-ms", type=float, default=10.0)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--max-kv-blocks", type=int, default=1024)
    p.add_argument(
        "--startup-delay-s",
        type=float,
        default=0.0,
        help="seconds of simulated weight loading before /health goes green. "
        "Set this to your real cold start to reproduce autoscaler overshoot.",
    )
    p.add_argument(
        "--fail-first",
        type=int,
        default=0,
        help="refuse the first N requests with 503, then recover -- exercises "
        "the router's circuit breaker and its half-open probe.",
    )
    p.add_argument("--drain-timeout", type=float, default=30.0)
    return p.parse_args(argv)


async def _serve(args) -> None:
    import uvicorn

    cfg = build_config(args)
    behaviour = Behaviour(
        ttft_ms=args.ttft_ms,
        itl_ms=args.itl_ms,
        max_concurrency=args.max_concurrency,
        max_kv_blocks=args.max_kv_blocks,
        fail_first=args.fail_first,
    )
    state = State()
    state._sem = asyncio.Semaphore(args.max_concurrency)
    app = build_app(cfg, behaviour, state)

    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    )
    serve_task = asyncio.create_task(server.serve())

    # Do not register until the socket is actually bound. uvicorn logs a bind
    # failure and gives up, but the process keeps running -- so without this
    # check a port collision produces a worker that is in the pool and serves
    # nothing. That is a routing black hole, and it is exactly the failure this
    # tool exists to help find rather than create.
    for _ in range(100):
        if server.started or serve_task.done():
            break
        await asyncio.sleep(0.05)
    if not server.started:
        serve_task.cancel()
        raise SystemExit(f"failed to bind {args.host}:{args.port} -- not registering")

    if args.startup_delay_s > 0:
        logger.info("simulating weight load for %.0fs", args.startup_delay_s)
        await asyncio.sleep(args.startup_delay_s)
    state.ready = True

    # Register only once ready, exactly like a real worker: a worker that is in
    # the pool but cannot serve is a routing black hole.
    if args.discovery_backend == "etcd":
        from infera.common.registration import RegistrationClient

        reg = RegistrationClient(args.etcd_endpoint, prefix=args.etcd_prefix)
    else:
        from infera.common.registration_k8s import K8sRegistrationClient

        reg = K8sRegistrationClient()
    worker_id = await reg.register(cfg)
    # register() only writes the record; keeping it alive is the caller's job in
    # both backends, exactly as in the real worker entrypoint. Without this the
    # etcd lease (30s) expires and the worker silently vanishes from the pool
    # about half a minute after it appears -- which looks like a discovery bug
    # and is not one.
    hb_task = asyncio.create_task(reg.heartbeat_loop(), name="fake-worker-heartbeat")
    logger.info(
        "registered %s model=%s mode=%s via %s",
        worker_id,
        cfg.model_name,
        cfg.disagg_mode.value,
        args.discovery_backend,
    )

    stop = asyncio.Event()

    async def _shutdown() -> None:
        # Deregister BEFORE draining, matching the real worker: the router must
        # stop sending new work before we stop accepting it, or the gap shows up
        # to clients as failures rather than as a clean drain.
        state.draining = True
        hb_task.cancel()
        try:
            await reg.deregister()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.warning("deregister failed: %s", exc)
        deadline = time.monotonic() + args.drain_timeout
        while state.running and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if state.running:
            logger.warning("drain timeout with %d request(s) still in flight", state.running)
        server.should_exit = True
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown()))

    await stop.wait()
    await serve_task


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

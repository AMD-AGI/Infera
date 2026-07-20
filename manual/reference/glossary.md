# Glossary

Short definitions for the terms used throughout this manual.

```{glossary}
Infera
  The inference-orchestration layer. "Infera" is the project; `infera` is the
  Python package you install and run.

Server
  `infera.server` — the FastAPI process exposing the OpenAI API and holding the
  router. Stateless with respect to models; run many replicas.

Worker / Engine
  One model engine (vLLM, SGLang, or ATOM) on one or more GPUs, launched via
  `python -m infera.engine.<name>`. Self-registers into the discovery backend.

etcd
  A small key-value store used as the shared registry of which workers exist and
  what each can do. The server watches it; workers write to it (lease + heartbeat).

Router
  The component inside the server that picks a worker per request. `AutoRouter`
  chooses between `MixedRouter` (single worker) and `DisaggRouter` (PD pair).

Routing policy
  How the router ranks workers: `round-robin` (even spread) or `kv-aware` (prefix
  cache locality).

KV cache
  The per-token key/value tensors the model produces while reading a prompt.
  Reusing them ("a cache hit") skips recomputing a shared prefix.

Prefix
  The leading tokens common to many requests (system prompt, document, chat
  history). The unit KV-aware routing and prefix caching optimize for.

PD / disaggregation
  Prefill/Decode disaggregation — running prefill (read the prompt) and decode
  (generate tokens) on separate GPUs/nodes, with a KV transfer between them.

Prefill / Decode
  The two inference phases. Prefill is compute-bound; decode is
  memory-bandwidth-bound and scales with concurrency.

1P1D / 2P1D / 1P2D
  PD topology shorthand: (prefill workers)P(decode workers)D.

Connector / Transport
  The mechanism that moves KV between PD workers: SGLang bootstrap,
  Mooncake (`MooncakeConnector`), or MoRI (`MoRIIOConnector`).

MoRI / MoRIIO
  AMD's IO/transfer layer. **MoRIIO** is a PD KV transport (vLLM `MoRIIOConnector`,
  SGLang `--disaggregation-transfer-backend mori`), run in read/pull mode. (MoRI
  also provides the MoE all-to-all, `--moe-a2a-backend mori` — a separate use.)

RDMA / RoCEv2
  Remote Direct Memory Access over the NIC, used for cross-node KV transfer. The
  RoCEv2 GID index matters cross-host — set it to the routable (v2) index via
  `MC_GID_INDEX` (find it with `show_gids`).

xGMI
  AMD's intra-node GPU interconnect (die-to-die / socket-to-socket link between
  GPUs on the same host).

ROCm
  AMD's GPU compute platform — the runtime/driver stack Infera and the engines run
  on. `gfx950` is the MI355X architecture.

AINIC
  AMD's AI NIC (Pensando / ionic) — the RoCEv2 RDMA NIC on MI355X hosts that
  carries the cross-node KV transfer.

RCCL
  AMD's collective-communication library; the `NCCL_*` env vars (e.g.
  `NCCL_IB_GID_INDEX`) configure it on ROCm.

kvd
  `infera.kvd` — the external tiered KV-cache daemon (RAM → NVMe → network).

L1 / L2 / L3 / L4
  KV cache tiers: GPU HBM / host RAM / local disk / distributed store. See
  [KV-Cache Management](../components/kvd.md).

Shared arena (CopyFree)
  The memfd-backed pinned RAM region kvd shares with engines via an FD passed
  over `SCM_RIGHTS`, enabling zero-copy GETs.

hipFile / AIS
  AMD Infinity Storage — GPU-direct storage that DMAs KV chunks straight between
  disk and GPU VRAM. kvd's L3 GPU-direct path uses it; gated by
  `INFERA_KVD_AIS`.

Retention
  Per-request cache-class hint `infera_retention` via OpenAI
  `extra_body.kv_transfer_params`. The vLLM connector honors `none|short|long`
  (default `long`); `ephemeral` and per-request TTL are daemon-level only.

NATS
  The default message broker for request and KV-event transport in the
  operator/production setup. The no-broker path uses `http` + `zmq` instead.

MI355X
  The AMD Instinct GPU Infera targets (gfx950): fp8 + mxfp4, AMD AINIC NIC.

fp8 / mxfp4
  Weight/activation quantization formats — both supported on MI355X. mxfp4 is the
  newer 4-bit format (smaller, faster).
```


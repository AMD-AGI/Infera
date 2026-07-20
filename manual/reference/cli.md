# CLI reference

The flags you'll reach for most, by entry point. This is a curated reference —
run `python -m infera.<thing> --help` for the complete, authoritative list.

```{admonition} Match transport/discovery fleet-wide
:class: important
`--discovery-backend`, `--request-transport`, and `--kv-event-transport` must be
the **same** on the server and every worker. See
[Routing & transport](../features/routing_and_transport.md).
```

## Server — `python -m infera.server`

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `8000` | OpenAI-compatible HTTP bind |
| `--router-tokenizer-path` | (required) | tokenizer for prefix hashing; must match workers |
| `--router-mode` | `auto` | `auto` (in-process policy) \| `direct` (trust a GAIE EPP's `x-worker-instance-id` pick) |
| `--router-policy` | `kv-aware` | `kv-aware` \| `round-robin` (used when `--router-mode auto`) |
| `--kv-overlap-weight` | `1.0` | KV-aware: cache locality vs load balance |
| `--kv-prefill-overlap-weight` | (unset) | KV-aware PD: prefill-side weight (typical `20.0`); overrides the global |
| `--kv-decode-overlap-weight` | (unset) | KV-aware PD: decode-side weight (typical `2.0`); overrides the global |
| `--request-max-retries` | `1` | retry on an alternate worker on pre-response failure (never mid-stream); `0` disables |
| `--discovery-backend` | `kubernetes` | `kubernetes` \| `etcd` |
| `--etcd-endpoint` | — | required for `--discovery-backend etcd` |
| `--etcd-prefix` | `/infera/workers/` | etcd key prefix the fleet registers under |
| `--k8s-label-selector` | `$INFERA_K8S_LABEL_SELECTOR` | worker-Pod selector (k8s discovery) |
| `--k8s-namespace` | (Pod's own namespace) | namespace to watch (k8s discovery) |
| `--request-transport` | `nats` | `nats` \| `http` |
| `--kv-event-transport` | `nats` | `nats` \| `zmq` |
| `--nats-server` | `$NATS_SERVER` / `nats://127.0.0.1:4222` | broker URL |
| `--nats-req-idle-timeout` | `900`s | per-chunk inactivity timeout (`0` = forever) |
| `--nats-req-max-duration` | `0` (off) | hard wall-clock cap per request |
| `--nats-req-max-pending` | `0` (off) | per-worker admission limit (→ 429) |
| `--kvd-socket-path` | — | kvd UDS; enables `POST /v1/cache/prewarm` (L3 prefetch), else 503 |
| `--enable-profiling` | off | torch-profiler control plane |

## Worker — `python -m infera.engine.{sglang,vllm,atom}`

```{admonition} Infera flags vs forwarded engine flags
:class: note
The worker launcher parses the **Infera** flags below and forwards everything
else **verbatim to the underlying engine**. So `--advertise-host`,
`--discovery-backend`, `--request-transport`, `--kv-event-transport`,
`--enable-kv-events` are Infera's; engine-native flags like `--tp-size` /
`--tensor-parallel-size`, `--page-size` / `--block-size`, and the
`--disaggregation-*` set are passed through to SGLang/vLLM/ATOM (and validated
by them, not by Infera). Consult the engine's own docs for those.
```

Common:

| Flag | Default | Meaning |
|---|---|---|
| `--model-path` (sglang/atom: `--model`) | (required) | served model |
| `--port` (atom: `--server-port`) | — | engine HTTP port |
| `--host` | `0.0.0.0` | bind address |
| `--etcd-endpoint` | — | shared etcd (etcd discovery) |
| `--advertise-host` | = `--host` | routable host/IP for cross-node |
| `--discovery-backend` | `kubernetes` | `kubernetes` \| `etcd` |
| `--request-transport` | `nats` | `nats` \| `http` |
| `--enable-kv-events` / `--no-enable-kv-events` | on | publish KV events for KV-aware routing |
| `--kv-event-transport` | `nats` | `nats` \| `zmq` |

Parallelism:

| Flag | Engine | Meaning |
|---|---|---|
| `--tp-size N` | SGLang | tensor-parallel across N GPUs |
| `--tensor-parallel-size N` | vLLM | tensor-parallel across N GPUs |
| `-tp N` | ATOM | tensor-parallel across N GPUs |
| `--page-size 16` / `--block-size 16` | SGLang / vLLM | KV block size (match fleet-wide) |

PD disaggregation:

| Flag | Engine | Meaning |
|---|---|---|
| `--disaggregation-mode prefill\|decode\|null` | SGLang | role; `null` = colocated |
| `--disaggregation-bootstrap-port` | SGLang | prefill bootstrap port |
| `--disaggregation-transfer-backend mooncake\|mori` | SGLang | KV transport |
| `--kv-transfer-config '{...}'` | vLLM / ATOM | connector + `kv_role` (producer/consumer) |

## KV-cache daemon — `python -m infera.kvd`

| Flag | Default | Meaning |
|---|---|---|
| `--socket` | — | UDS path engines connect to |
| `--max-bytes` | — | host RAM (L2) budget |
| `--shared-arena-bytes` | `auto` | zero-copy arena size (`auto` = `--max-bytes`; `0` = off) |
| `--long-path` / `--long-paths` | — | L3 region: single device / **striped** across devices (mutually exclusive) |
| `--long-bytes` | — | L3 region size |
| `--use-tablespace` | *(deprecated)* | no-op, accepted for backward compatibility; the L3 tier is always the container-file tablespace layout now |
| `--long-backend` | `tablespace` | `tablespace` (L3) \| `mooncake` \| `lmcache` (L4) — **one only** |
| `--io-mode` | `auto` | `auto` \| `direct` \| `buffered` (auto picks O_DIRECT on NVMe, buffered on NFS/SATA) |
| `--long-workers-per-shard` | `8` | intra-shard IO parallelism for the striped (`--long-paths`) region |

### Per-request KV params (OpenAI `extra_body.kv_transfer_params`)

| Key | Values | Meaning |
|---|---|---|
| `infera_retention` | `none\|short\|long` (default `long`) | cache class. The vLLM connector honors `none`/`short`/`long`; `ephemeral` exists as a daemon-side eviction class but is not selectable per-request via the vLLM connector (it maps to `long`). |

Per-request **TTL** (`infera_ttl_seconds`) is supported by the daemon/wire
protocol but is **not** currently propagated by the vLLM connector — TTL is a
daemon-level concern today, not a per-request knob on the vLLM path.

## Important environment variables

| Var | Where | Why |
|---|---|---|
| `PYTHONHASHSEED=0` | vLLM worker | **mandatory** for cross-restart cache hits (vLLM salts block hashes per process) |
| `HIP_VISIBLE_DEVICES` | worker | which GPU(s) the worker uses |
| `MC_DISABLE_HIP_TRANSPORT=1` | PD worker | force RDMA, not the empty XGMI shortcut |
| `MC_GID_INDEX=1` / `NCCL_IB_GID_INDEX=1` | cross-node PD | RoCEv2 GID (default 0 is link-local) |
| `VLLM_HOST_IP` | vLLM PD | routable host IP for the connector |

# Environment variables

Every environment variable Infera reads, grouped by subsystem. These are set on
the **process** (engine, daemon, or server) — most have a CLI-flag equivalent
where noted, and flags win over env vars. Each feature one-pager repeats the
subset relevant to it; this page is the single place that lists them all.

```{tip}
The three that bite people most: `PYTHONHASHSEED=0` (vLLM cross-restart cache
hits), `MC_GID_INDEX=1` (cross-node RDMA), and `--ipc=host` so the engine and kvd
share the L2 arena.
```

## kvd cache — connector (set on the engine process)

| Env | Default | What it does |
|---|---|---|
| `INFERA_KVD_SOCKET` | `/var/run/infera-kvd.sock` | Unix socket the engine uses to reach the kvd daemon. |
| `INFERA_KVD_HIPFILE_ROOTS` | *(none)* | L3 write roots, `retention=path` CSV (e.g. `long=/nvme/kvd-long`). Set this to enable a disk tier; also turns on the GPU-direct chunk-file path. |
| `INFERA_KVD_AIS` | `auto` | GPU-direct (hipFile) read/write: `auto` probes P2PDMA, `1`/`0` force. Use `1` on NFS/AIC, measure on local NVMe. Legacy name: `INFERA_KVD_GPU_DIRECT`. |
| `INFERA_KVD_CHUNK_TOKENS` | `auto` | Tokens per L3 chunk; `auto` targets `INFERA_KVD_CHUNK_TARGET_MIB` (default `128`). |
| `INFERA_KVD_LAYERWISE_LOAD` | `parallel` | L3 load mode: `off` / `stepped` / `prefetch` / `parallel`. |
| `INFERA_KVD_L3_BUDGET_BYTES` | `0` (uncapped) | Hard cap on the **GPU-direct** L3 file volume. These files are connector-owned, so the daemon's `--long-bytes` does **not** bound them — this does. |
| `INFERA_KVD_L3_FREE_FLOOR` | `0.05` | Free-space fraction the connector's L3 reaper keeps; evicts (retention-priority, LRU) when violated. Matters most on a shared filesystem. |
| `INFERA_KVD_L3_REAP_INTERVAL` | `30` (s) | L3 reaper tick period. |
| `INFERA_KVD_SAVE_WORKERS` / `INFERA_KVD_LOAD_WORKERS` | `auto` | L3 I/O worker counts (`auto` = mount `nconnect`; load drops to 1 without P2PDMA). |
| `INFERA_KVD_RETENTION_DEFAULT` | `long` (vLLM) | Default per-request retention when the request carries none. **Required on SGLang** to exercise the long tier (its SPI doesn't propagate per-request retention). |
| `HIPFILE_UNSUPPORTED_FILE_SYSTEMS` | *(unset)* | **libhipfile** env (not Infera): set `1` to allow hipFile DMA reads on NFS. |

Per-request (not env) cache hint goes through `extra_body.kv_transfer_params`:
`infera_retention` — the vLLM connector honors `none|short|long` (default
`long`). See [KV-Cache Offload](../features/kv_cache_offload.md).

```{admonition} Advanced kvd knobs
:class: note
Edge-case tuning (async-load depth, ring buffers, save pool, fsync policy, arena
hugetlb/prefault, per-shard workers) exists as additional `INFERA_KVD_*` vars.
A normal deployment never needs them; run `python -m infera.kvd --help` for the
ones that matter.
```

## PD disaggregation — RDMA transport (set on each engine)

| Env | Typical | What it does |
|---|---|---|
| `MC_GID_INDEX` | `1` | Mooncake **RoCEv2 (routable) GID index** — `1` on our ionic fleet; verify with `show_gids` (often 1 or 3). Index 0 is the link-local/RoCEv1 GID and times out cross-node. |
| `NCCL_IB_GID_INDEX` | `1` | Same idea for the collectives (RCCL) path. |
| `MC_DISABLE_HIP_TRANSPORT` | `1` (cross-node) | Force RDMA instead of the intra-node HIP/XGMI shortcut (which advertises an empty segment a remote peer rejects). |
| `MORI_IB_GID_INDEX` | `1` | RoCEv2 GID index for the MoRI transport. |
| `VLLM_HOST_IP` | *(none)* | Routable IP a vLLM worker advertises for KV transfer (cross-node). Pair with `--advertise-host`. |
| `RDMAV_FORK_SAFE` | `1` | libibverbs fork-safety; set for RDMA workers. |

See [PD disaggregation](../features/pd_disaggregation.md) for the bring-up
checklist and the RDMA self-check.

## Routing & transport (set on the server and every worker)

| Env | Default | What it does |
|---|---|---|
| `NATS_SERVER` / `NATS_URL` | *(injected by operator)* | NATS broker address when `--request-transport nats`. |
| `INFERA_NATS_REQ_IDLE_TIMEOUT` | `900` (s) | Max wait for the next reply chunk; expiry → 504 + cancel. `0` = forever. |
| `INFERA_NATS_REQ_MAX_DURATION` | `0` (off) | Hard wall-clock cap on a whole request. |
| `INFERA_NATS_REQ_MAX_PENDING` | `0` (off) | Per-worker admission limit; backlog at the cap → 429. |
| `INFERA_REQUEST_MAX_RETRIES` | `1` | Retry on an alternate worker before the first token (never mid-stream). `0` = fail fast. |
| `INFERA_K8S_LABEL_SELECTOR` | *(operator default)* | Pod selector for `--discovery-backend kubernetes`. |
| `INFERA_DRAIN_TIMEOUT` | `30` (s) | Graceful-drain window on shutdown. |

Each has a CLI flag (flag > env > default); see [Routing & transport](../features/routing_and_transport.md)
and the [CLI reference](cli.md).

## Engine correctness

| Env | Value | What it does |
|---|---|---|
| `PYTHONHASHSEED` | `0` | **Mandatory on vLLM** — vLLM v1 salts block hashes per process, so without a fixed seed you get **0 cross-restart cache hits**. SGLang doesn't need it. |
| `VLLM_SERVER_DEV_MODE` | `1` | Mounts dev routes including `POST /reset_prefix_cache` (used to clear L1 + kvd L3 without a restart). |

## Related

- [CLI reference](cli.md) — the flag equivalents.

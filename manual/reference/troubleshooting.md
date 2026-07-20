# Troubleshooting

The most common "why isn't this working" cases, grouped by subsystem. Most are a
single flag that has to match across the whole fleet.

## Fleet / routing

| Symptom | Likely cause | Fix |
|---|---|---|
| Worker never shows in `GET /v1/workers` | server and worker disagree on discovery/transport | Match `--discovery-backend` and `--request-transport` on **both** (see [Routing & transport](../features/routing_and_transport.md)). |
| Server can't reach a worker across hosts | worker advertised a non-routable address | Pass `--advertise-host <routable-ip>` on the worker; shared `--etcd-endpoint`. |
| KV-aware routing not improving TTFT | events off, or block size mismatch | `--enable-kv-events` on workers + `--page-size/--block-size 16` everywhere + server `--router-policy kv-aware`. |

## PD disaggregation

| Symptom | Likely cause | Fix |
|---|---|---|
| PD worker refuses to start on TCP | PD hard-fails on TCP (5–20× slower than RDMA) | Fix the RDMA device flag, or benchmark-only `--disaggregation-allow-tcp`. |
| Cross-node QP times out at `QP → RTR` | GID index 0 is link-local, not the RoCEv2 index | Set the RoCEv2 GID index (`MC_GID_INDEX` + `NCCL_IB_GID_INDEX`) — find it with `show_gids` (often 1 or 3). |
| `received packet mismatch` | prefill and decode on different RDMA rails | Put both on the same `ib_device`. |

## Tiered KV cache (kvd)

| Symptom | Likely cause | Fix |
|---|---|---|
| Worker aborts: "infera-kvd unreachable" | daemon not started or wedged | Start `python -m infera.kvd` first; if the probe times out, restart it. (Intentional fail-fast.) |
| GETs not zero-copy (inline-bytes fallback) | kvd and engine not in the same IPC namespace | Docker `--ipc=host`; in k8s co-locate kvd + engine in one pod. |
| vLLM: 0 cross-restart hits | vLLM v1 salts block hashes per process | Set `PYTHONHASHSEED=0` on the vLLM process (mandatory; SGLang doesn't need it). |
| kvd won't start: "filesystem rejects O_DIRECT" | tablespace defaults to O_DIRECT, unsupported on tmpfs/some NFS | `--io-mode auto` (or `buffered`). |
| GPU-direct silently CPU-bounces (`READ` clamped to 1 worker, no error) | `ais-check` says `Kernel P2PDMA support: False` **only because the image ships no `/boot/config-*`** — a false-negative, not missing hardware (`amdgpu: True` confirms the driver op is present) | Mount the host kernel config: `-v /boot:/boot:ro` (shipped k8s/operator now do this). See [GPU-direct P2PDMA false-negative](#gpu-direct-p2pdma-false-negative). |

### GPU-direct P2PDMA false-negative

`ais-check` decides `Kernel P2PDMA support` by reading `/boot/config-$(uname -r)`.
Engine images don't ship that file, so on a perfectly P2PDMA-capable host ais-check
prints `False` purely because the file is absent. The kvd connector treats that as
"no P2PDMA" and silently downgrades GPU-direct L3 loads to the CPU-bounce path
(reads clamped to 1 worker, ~8× slower) — no error, just lost throughput.

Tell a real failure from a false-negative by the `amdgpu` line:

```
ais-check
#   Kernel P2PDMA support : False   <- brittle (reads /boot/config-*, absent in image)
#   amdgpu                : True    <- runtime fact: the driver AIS op is present
```

`amdgpu: True` with `P2PDMA: False` is the false-negative. Give the container the
host's config to fix it:

```
docker run ... -v /boot:/boot:ro ...   # auto-matches the shared kernel version
```

ais-check then reads `Kernel P2PDMA support: True`. The shipped k8s manifest and
operator pod builder both mount `/boot:ro` for this reason; confirm on
a node with `infera-preflight --firmware`.

## Diagnostic commands

```bash
# live worker registry as the server sees it
curl -s localhost:8000/v1/workers | python -m json.tool

# router's mirrored cache view for one worker (KV-aware)
curl -s localhost:8000/v1/admin/cache-view/<host:port>

# live kvd counters — the daemon logs them periodically at --log-level INFO
#   (entries, gets/sets/hits/misses, long-tier bytes)
```

## When all else fails

- Re-read the relevant feature one-pager — most issues are a flag that must match
  fleet-wide.
- Check the [environment variables](environment.md) reference — a missing or
  mismatched env var (GID index, `PYTHONHASHSEED`, `--ipc=host`) is a frequent
  root cause.

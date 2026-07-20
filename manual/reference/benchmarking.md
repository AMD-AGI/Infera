# Benchmarking & storage probe

Two tools answer "is my L3 storage fast enough, and what does kvd actually get on
it?" Both drive the **real `InferaKvdConnector` save/load path** — they start a
real kvd daemon, allocate GPU KV-cache tensors, and time the production
`wait_for_save` / `start_load_kv`. So the numbers reflect the engine's actual
chunked-fusion pipeline (per-chunk staging, per-layer H2D overlap, Triton
scatter, GPU-direct-vs-POSIX transport, worker fan-out), not a synthetic IO loop.
Run them **inside the engine image** — they need torch + a GPU + `ais-check`.

- **Storage probe** — a quick "will this mount work for kvd L3?" check.
- **L3 throughput bench** — read + write GB/s under a chosen transport / chunk /
  worker config.

## Storage probe

```bash
infera-kvd-probe --dir /mnt/nvme8/kvd-probe --size-gb 16
```

Drives the connector's real save/load against `--dir` and reports write (save) and
cold read (load) GB/s, plus the resolved transport and the P2PDMA verdict
(`ais-check`). It honors your `INFERA_KVD_CHUNK_TOKENS` — the chunk you actually
run in production — and logs the resolved chunk size, so the number is what kvd's
L3 would get on this mount. A false-negative P2PDMA verdict (container missing
`/boot/config-*`) clamps L3 loads to 1 worker — the probe surfaces exactly that.

## L3 throughput bench

```bash
infera-kvd-l3-bench --dir /mnt/nvme8/l3-bench --total-gb 16
```

| Flag | Maps to |
|---|---|
| `--transport {auto,posix,gpu-direct}` | connector transport (`INFERA_KVD_AIS`): POSIX mmap+H2D vs hipFile/P2PDMA S2D |
| `--chunk-tokens N \| auto` | `INFERA_KVD_CHUNK_TOKENS`; `auto` sizes to `--chunk-target-mib` (matches the daemon's autosize) or honors a user-set env value |
| `--chunk-target-mib N` | per-chunk MiB target for `auto` (default 128, = `INFERA_KVD_CHUNK_TARGET_MIB`) |
| `--workers N` | save+load executor fan-out (`INFERA_KVD_*_WORKERS`) |

Both transports land data in **GPU HBM** (that's where kvd's L3 goes); they differ
only in how the bytes get there — `posix` stages through a pinned host buffer
(CPU-bounce, the no-P2PDMA path), `gpu-direct` DMAs NVMe↔HBM directly (hipFile/AIS,
needs a P2PDMA-capable driver). READ is measured **cold** (KV zeroed before the
load) so it is a genuine reload.

**Chunk size matters:** the parallel load fan-out only pays off at chunks
**≥ 128 MiB** (smaller = N-thread overhead > read time). `auto` (the default) sizes
to 128 MiB, matching production; a fixed small `--chunk-tokens` will under-report.

### Reference numbers — indicative only

> **Hardware:** AMD Instinct MI355X (gfx950), ROCm 7.1.1; L3 on 8× 7.6 TB NVMe in
> RAID-0 (md, 256 KiB stripe). **Config:** real connector path, `auto` chunk
> (~128 MiB), 8 workers, Kimi-MLA KV shape (hidden 576, 61 layers).
>
> | transport | cold LOAD (reload) |
> |---|---|
> | gpu-direct (P2PDMA, S2D) | ~19 GB/s |
> | posix (mmap + H2D) | ~7–8 GB/s |
>
> At production chunk size (≥ 128 MiB) GPU-direct wins the reload via parallel
> fan-out; at small chunks (e.g. 34 MiB) both fall to ~6 GB/s and the win vanishes.

**These are a rough reference, not a target.** Your GPU, storage (single disk vs
RAID, NVMe vs NFS/WekaFS), chunk size, model KV shape, and host load all move
these numbers — sometimes by a lot. Always measure your own mount with the tools
above rather than trusting a number from another machine.

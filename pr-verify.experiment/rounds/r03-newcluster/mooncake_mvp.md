# r03 — cross-node mooncake MVP: works, over `mlx5_0` only

Measured 2026-08-24 between `crsuse2-m2m-237` (job 58799) and `crsuse2-m2m-106`
(job 58800), using `scripts/mvp_mooncake_loopback.py` in
`infera-local:sglang-prverify-20260824`, both containers `--network=host` with
`--device /dev/infiniband --cap-add IPC_LOCK -e RDMAV_FORK_SAFE=1`.

## Headline

**Cross-node mooncake RDMA works.** This is the path n06-33 could not build, so
**#33970 can be validated as a real 2-node 1P1D** rather than the single-node
TP4+TP4 loopback fallback that `plan.md` step 1 assumes.

```
[initiator] register_memory(0x76e6ed6fb010, 8388608) rc=0
[initiator] transfer_sync_read rc=0 in 24.01 ms
[initiator] first16=abababababababababababababababab all_0xAB=True
RESULT: PASS
```

`rc=0`, all 8 MiB arrive byte-correct. Reproduced (24.01 ms, then 26.08 ms).

## Use `mlx5_0`. The ionic rails silently fall back to TCP.

This is the trap-1 lesson again, and it would have been easy to miss: **the rail
run also returned `rc=0` with correct data, and was 4x faster** — but it was not
RDMA at all.

| device | topology discovery | transport | time (8 MiB) |
|---|---|---|---|
| `mlx5_0` | `Found 1 HCAs` | `installTransport, type=rdma` | 24.0 / 26.1 ms |
| `ionic_0` | **`Found 0 HCAs`** | **`TcpTransport: listen on port 15230`** | 6.6 ms |

The `ionic_0` run never installed an RDMA transport. It fell through to TCP over
the management network, reported `RESULT: PASS`, and looked *better* than the
working RDMA path. Reading only `rc` and the byte check would have recorded a
rail RDMA result that never happened.

**Cause is in the logs**, not a mystery:

```
libibverbs: Warning: Driver ionic does not support the kernel ABI of 4
            (supports 1 to 1) for device /sys/class/infiniband/ionic_N
```

The container's `libibverbs` is newer than the host's `ionic` driver ABI, so all
eight ionic devices are rejected at discovery and only `mlx5_0` survives. This is
the same class of problem the n06-33 kit worked around by bind-mounting the
host's `libionic.so` into the container (see the Quick Start in the repo README,
`-v /usr/lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro`). Not
attempted here: `mlx5_0` works and is sufficient for #33970, whose claim is about
a race, not about bandwidth.

So: **the user's instruction to use mlx5 is what the measurement supports.**

## GID index: mooncake selects it itself here

```
rdma_context.cpp:1362  Find best gid index: 3 on mlx5_0/ (network state: with network device)
rdma_context.cpp:290   RDMA device: mlx5_0, LID: 0, GID: (GID_Index 3) ...ff:ff:0a:f5:9a:bf
```

Index 3 is the RoCE v2 IPv4 entry I enumerated independently
(`rdma_topology.md`). **`MC_GID_INDEX` was not set** in these runs and did not
need to be — mooncake's own selection was correct. Do not carry n06-33's
`MC_GID_INDEX=1` over; it would be wrong here, and unnecessary.

## What the timings do and do not mean

24-26 ms for 8 MiB (~2.7 Gb/s) is **not** steady-state bandwidth and should not
be quoted as such. Each run is a fresh process that pays
`openSegment`/handshake — visible as
`transferSync, cache not found, openSegment with target ...` on every run. The
line-rate figure for this path is the `ib_write_bw` number, 174.58 Gb/s
(`rdma_topology.md`). This MVP establishes **correctness and reachability**, which
is what it was for.

## Practical notes

- The **first** initiator run printed no result line at all; the second, identical
  invocation worked. Run it twice before concluding anything from a silent run.
- The target holds for 180 s (`time.sleep(180)`), so re-run the initiator inside
  that window or restart the target.
- `Failed to get NUMA node ... Operation not permitted` is benign — it appears on
  the working RDMA path too.
- Container needs `--device /dev/infiniband`, `--cap-add IPC_LOCK`,
  `--network=host`, and `RDMAV_FORK_SAFE=1` (trap 2).

## Consequence for #33970

`plan.md` step 1 is written around the single-node TP4+TP4 workaround and its
scope caveat — *"CANNOT establish behaviour under real cross-node RDMA latency …
loopback is faster, so the race window is narrower"*. That caveat **no longer
applies**: the 2-node configuration the PR actually targets is available. Plan the
A/B as 1P on 237 + 1D on 106 over `mlx5_0`, and drop the loopback scope
disclaimer from the PR.

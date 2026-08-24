# r03 — cross-node RDMA topology on crsuse2-m2m (237 / 106)

All measured first-hand 2026-08-24 with `ib_write_bw` via `spur exec`.
Nodes: `crsuse2-m2m-237` (job 58799), `crsuse2-m2m-106` (job 58800).

## Headline

**Cross-node RDMA works, on both the management NIC and the rails.** This is the
path n06-33 could not build (its peer n01-33 had no rail route). The 2-node #33970
run is therefore on the table here — subject to mooncake itself working, which
`ib_write_bw` does **not** establish (trap 1 in `context.md`).

| Path | Device | GID idx | rdma_cm (`-R`) | BW @1 MiB | RTT |
|---|---|---|---|---|---|
| management | `mlx5_0` / `ens3` | 3 (RoCE v2, IPv4) | **works** | **174.58 Gb/s** | 0.56 ms |
| rail 0 | `ionic_0` / `enP2p0s9` | 1 (RoCE v2, IPv6) | **fails** — `Unexpected CM event 8` | **375.14 Gb/s** | 0.15 ms |

Rail is 2.15x the bandwidth and 3.7x lower latency, but only over the non-`-R`
(out-of-band TCP parameter exchange) path.

## Correcting an earlier reading

An initial `ip -o -4 addr` sweep showed no address on any `enP*` interface, which
read as "the rails carry no IP". That was wrong: **the rails are IPv6-addressed**.
Each has a global `fc01::/16` address handed out by RA, plus a link-local `fe80::`.

```
237 ionic_0  fc01:0800:700d:2d3f:0690:81ff:fe44:b669   (enP2p0s9,  mtu 9000)
106 ionic_0  fc01:0800:a00d:2d5e:0690:81ff:fe45:3529   (enP2p0s9,  mtu 9000)
```

Same-index rails share the second field (`0800` for rail 0 on both nodes) but sit
in different /64s, so they are **routed, not on one link**:

```
ip -6 route get fc01:0800:a00d:2d5e:0690:81ff:fe45:3529
  via fe80::9e69:edff:fe3b:e017 dev enP2p0s9 mtu 9000
ping6 -I enP2p0s9 <106 rail0>  ->  3/3, 0.152 ms avg
```

## Full rail map (identical device ordering on both nodes)

`rdma link show`: `ionic_0..3` -> `enP2p0s9..12`, `ionic_4..7` -> `enP3p0s9..12`,
`mlx5_0` -> `ens3`. Unlike n06-33, the mapping **is** in numeric order here — do
not carry over trap 4's out-of-order table.

Rail second-field per index (237 / 106):

| ionic | 237 | 106 |
|---|---|---|
| 0 | `fc01:0800:700d:2d3f` | `fc01:0800:a00d:2d5e` |
| 1 | `fc01:0700:6f0d:2d3f` | `fc01:0700:9f0d:2d5e` |
| 2 | `fc01:0500:6d0d:2d3f` | `fc01:0500:9d0d:2d5e` |
| 3 | `fc01:0600:6e0d:2d3f` | `fc01:0600:9e0d:2d5e` |
| 4 | `fc01:0400:6c0d:2d3f` | `fc01:0400:9c0d:2d5e` |
| 5 | `fc01:0300:6b0d:2d3f` | `fc01:0300:9b0d:2d5e` |
| 6 | `fc01:0100:690d:2d3f` | `fc01:0100:990d:2d5e` |
| 7 | `fc01:0200:6a0d:2d3f` | `fc01:0200:9a0d:2d5e` |

## GID indices on this fabric — do not copy `MC_GID_INDEX=1` blindly

Enumerated per trap 1's instruction:

- `mlx5_0`: gid[0]/[1] link-local, gid[2] RoCE v1 IPv4, **gid[3] RoCE v2 IPv4**
  (`::ffff:10.245.x.x`). For the management path the index is **3**.
- `ionic_N`: gid[0] link-local (RoCE v2), **gid[1] global `fc01::` (RoCE v2)**.
  For the rail path the index is **1** — which coincidentally matches n06-33's
  value, but for a different reason (there it was an IPv4-mapped `::ffff:192.168.x.x`).

## The open problem: rdma_cm fails on the rails

```
ib_write_bw -d ionic_0 -x 1 -R -F --ipv6-addr <106 rail0>
  -> Unexpected CM event 8      (client cannot init the connection)

ib_write_bw -d ionic_0 -x 1 -F <106 mgmt IPv4>     # no -R
  -> 375.14 Gb/s, remote GID fc01:0800:a00d:... (confirmed the real rail)
```

So the rail carries RDMA traffic fine; it is the **rdma_cm connection manager**
that will not establish over it. The management NIC accepts `-R` without trouble.

This matters because `working_process.md` records mooncake as using rdma_cm — if
that holds, mooncake may be confined to the 174 Gb/s `mlx5_0` path even though the
rails are faster. **Not yet established either way.** The measurement that settles
it is the mooncake `TransferEngine` MVP (`scripts/mvp_mooncake_loopback.py`,
adapted to two hosts), not another perftest run. Note also that on n06-33
`ib_write_bw` needed `-R` to produce data rows at all (trap 5); here it is the
opposite, which is another reason not to reason across fabrics.

## Practical notes

- `ib_write_bw` rejects `-6`; the flags are `--ipv6` (GID) and `--ipv6-addr`
  (parameter negotiation).
- Passing a bare IPv6 literal without `--ipv6-addr` fails with
  `Address family for hostname not supported for ai_family: 2`.
- `spur exec` runs with `HOME=/opt/spur`, which is **not writable**, and `pwd=/`.
  Export `HOME=/home/yihou` and `cd` explicitly, or docker and anything else that
  wants a home directory fails with `mkdir /opt/spur/.docker: permission denied`.
- A backgrounded `spur exec` dies when the launching shell exits — use
  `setsid nohup ... & disown` **inside** the remote command to detach it.

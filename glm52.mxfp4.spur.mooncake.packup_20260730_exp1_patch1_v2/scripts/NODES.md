# Node assignment — three-way experiment, 2026-07-30

Held 2026-07-30 (`sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00`).
All six passed the GPU health gate (`torch.cuda.is_available() -> True 8`) and
run container `dbg2` from image `infera.yihou.sglang.1.0`.

| exp | leg     | job   | node            | ens3 IP         | port  |
|-----|---------|-------|-----------------|-----------------|-------|
| E1  | prefill | 14315 | crsuse2-m2m-118 | 10.245.159.138  | 30000 |
| E1  | decode  | 14316 | crsuse2-m2m-003 | 10.245.157.171  | 30001 |
| E2  | prefill | 14317 | crsuse2-m2m-234 | 10.245.152.243  | 30000 |
| E2  | decode  | 14318 | crsuse2-m2m-259 | 10.245.155.111  | 30001 |
| E3  | prefill | 14320 | crsuse2-m2m-074 | 10.245.154.156  | 30000 |
| E3  | decode  | 14321 | crsuse2-m2m-072 | 10.245.144.119  | 30001 |

Routers run on each experiment's PREFILL node, one per experiment, on distinct
ports so a circuit breaker left open by one arm cannot be mistaken for another
arm's failure (PITFALLS P4):

| exp | router port | prometheus port |
|-----|-------------|-----------------|
| E1  | 8110        | 29110           |
| E2  | 8120        | 29120           |
| E3  | 8130        | 29130           |

## What each experiment changes

| exp | patch 1 | patch 2a | patch 2b | patch 3 | patch 4 | IndexShare | prefill MTP |
|-----|---------|----------|----------|---------|---------|------------|-------------|
| E1  | **v2 (#32762 shape)** | ours | ours | ours | ours | on (default) | off |
| E2  | v1 (ours) | — | — | ours | — | **off** | **on** |
| E3  | v1 (ours) | ours | **#32209 shape** | ours | **#32209 shape** | on (default) | off |

E1 and E3 keep `prefill MTP` off because that is the configuration all our
existing evidence was collected under; changing it there would confound the
patch comparison. E2 turns it on because reproducing llying's run is the point
of that arm and their launcher enables it on both legs.

## Acceptance for all three (as instructed)

1. 4-prompt sequential probe, 4/4, `spec_accept_length > 1`
2. conc=32 x 512 tokens, 32/32, no hang, no `KVTransferError`

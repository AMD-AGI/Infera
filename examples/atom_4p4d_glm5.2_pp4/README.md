# GLM-5.2-MXFP4 — ATOM 4P4D with a PP4 prefill (debug)

Debug variant of [`atom_4p4d_glm5.2_dcp`](../atom_4p4d_glm5.2_dcp/) for a
performance comparison: the prefill on GPUs 0-3 runs **TP1 × PP4** with the
settings of ATOM's GLM-5.2 AgentX recipe
(`/app/ATOM/recipes/Agentic-GLM-5.2.md` in the image) instead of TP4 + DCP4.
Like [`atom_8p4d_glm5.2_dcp`](../atom_8p4d_glm5.2_dcp/), this kit is only an
`env.sh` over the 4P4D kit, whose scripts run unchanged
(`PREFILL_PP=4` selects the PP prefill in its `worker.sh`). Read the 4P4D README
for the setup, the image and the gotchas.

```bash
$EDITOR env.sh               # the only file you edit
bash engine/up.sh            # etcd -> prefill + decode -> router, waiting on health
bash engine/smoke.sh         # correctness through the router (real MTP acceptance)
bash engine/agentx.sh        # one AgentX point (start with MTP_AL=2.99 for the reference settings)
bash engine/down.sh
```

## Differences from the DCP kit

| | DCP kit | this kit | why |
|---|---|---|---|
| prefill parallelism | TP4 + DCP4 | TP1 × PP4, layers per stage `20,20,20,18` | ATOM recipe |
| prefill execution | CUDA graphs up to 2 × CONC | `--enforce-eager` | ATOM runs PP without CUDA graphs |
| prefill `--max-num-batched-tokens` | 16384 | 8192 | ATOM recipe (chunked PP) |
| prefill mem util | 0.90 | 0.85 | ATOM recipe |
| `OFFLOAD_MIN_LOAD_TOKENS` | 8192 | 0 | ATOM recipe |
| `LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3` | one per DCP rank | one per PP stage | same value |
| block size, both legs | 64 | 16 | ATOM recipe; the non-DCP producer lays tokens into each decode DCP rank's pages with one block size, so both legs share it |
| Mooncake send workers per prefill rank | 16 (ATOM default) | 4 | see below |
| decode | TP4 + DCP4 | unchanged apart from the block size | |

Not taken from the recipe: `OFFLOAD_PROFILE=1` (logging only) and
`--timeout-keep-alive 900` (the infera disagg router does not reuse idle
connections). The recipe's decode settings (mem util 0.94,
`--max-num-batched-tokens 2048`, `AITER_REUSE_IDENTICAL_COMM_GROUPS=1`, graphs
up to 256) are not applied: only the prefill changes.

KV moves from each PP stage (its own layers) to every decode DCP rank: the
producer gathers the tokens a rank owns into a staging slot
(`ATOM_PD_MLA_STAGING`, default on) and the decode lands it
(`ATOM_PD_MLA_LANDING`, default on). Mooncake handshakes are base + PP stage
on the prefill, 21301-21304. AgentX metadata reports the prefill as TP1 × PP4,
so throughput per GPU is still over 8 GPUs.

## Known issue: send workers

With ATOM's default 16 Mooncake send workers per stage, every AgentX run on
MI355X (crsuse2-m2m-140) failed within its first four requests: one RDMA write
from a stage to one decode rank never completed (`staged-mla` / `landed-mla
RDMA chunk error -1 ... Mooncake's sync timeout`, Mooncake `CQ completion
timeout diagnostic`, then `Failed to poll CQ` and ionic `wqe is not posted`
without end). Every later transfer to that rank failed, so every request did.
It hit `ionic_4` and `ionic_5`, with forced and with real MTP acceptance. One
send worker removed the hang, but a 443k-token prompt then missed ATOM's fixed
120 s transfer deadline (`PD_TRANSFER_DEADLINE_S`, queueing included). Four
workers (`PREFILL_SEND_WORKERS`, this kit's default) ran C64 and C96 without a
transfer failure. The DCP kit, whose prefill ranks each write to one decode
rank without staging, never showed the hang.

## Results

AgentX, 140, image `infera-atom:nightly-202610131505-pr198fix`, same settings
as the DCP kit's reference runs (MTP K3, forced AL 2.99, 10 warm-up requests
per lane, 3600 s), one run per point. Per GPU over 8 GPUs; TTFT in s, ITL in ms.

| CONC | kit | input tok/s/GPU | output tok/s/GPU | TTFT p50 / p90 / p99 | ITL p50 / p90 | prefill hit GPU / CPU / total % | errors |
|---:|---|---:|---:|---|---|---|---|
| 64 | DCP | 30739.05 | 242.78 | 3.23 / 17.81 / 49.41 | 16.03 / 17.30 | 79.8 / 16.7 / 96.5 | 1 / 7024 |
| 64 | PP4 | **33389.60** (+8.6%) | 262.13 | 2.09 / 5.17 / 15.74 | 16.46 / 17.59 | 77.7 / 17.2 / 95.0 | 1 / 7682 |
| 96 | DCP | 20902.07 | 193.03 | 51.39 / 103.55 / 161.52 | 13.75 / 16.48 | 16.3 / 77.6 / 93.9 | 0 / 5664 |
| 96 | PP4 | **30115.94** (+44.1%) | 274.59 | 24.98 / 51.53 / 76.31 | 16.75 / 18.74 | 34.6 / 52.0 / 86.6 | 0 / 8142 |

## Status

Debug only. Besides the runs above: smoke 6/6 and GSM8K 0.98 (200 questions,
real acceptance, 16 send workers). On shutdown each PP stage logs a few
`TypeError` tracebacks from aiter `shm_broadcast`; none appear while serving.

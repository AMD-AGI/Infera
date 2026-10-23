# GLM-5.2 PD cross-node 135 -> 136: DCP4 KV-transfer fix and the new DCP / DPA best (2026-10-22..23)

**Result (AgentX C56, 3600 s, one A1 each, sim acceptance 3.61 - no correctness claim):**

| round | decode | score tok/s/chip | TPOT ms | TTFT p50/p90 s | KV transfer p50/p90 s |
|---|---|---|---|---|---|
| **dpax1** | DPA4 (TP4/DP4, mem 0.95) | **26,321** (new best) | 16.7 | 3.32 / 18.18 | 0.83 / 1.76 |
| **dcpx11** | DCP4 (TP4 EP4 DP1 dcp4, mem 0.85, graph 1..96 13 sizes) | **25,197** (DCP best) | 18.4 | 3.46 / 15.32 | 0.88 / 1.95 |
| dcpx12 | dcpx11 + decode mem 0.87 | 24,651 | 18.3 | 3.56 / 15.32 | - |
| dcpx10 | DCP4 + pack fix | 24,753 | 17.7 | 3.70 / 17.48 | 0.93 / 1.97 |
| dcpx8c | DCP4 before the pack fix | 19,997 | 16.9 | 11.57 / 42.57 | 12.29 / 32.28 |
| g3x1 (10-21, v0521) | DPA4, previous best | 25,524 | 15.9 | 4.28 / 21.96 | 1.00 / 1.95 |

**What fixed DCP:** with a DCP decode every prefill rank sends each request to all 4 decode ranks through
`send_kvcache_dcp`, which first packs the strided KV rows on the GPU (`try_pack_dcp_src`). Measured with a log-only
instrumentation (`[yihou-xfer]` lines): the pack took 78 ms per call and 91 % of the send time while RDMA itself ran at
~12 GB/s per call. The pack kernel alone takes 0.66 ms for 8192 rows on an idle GPU. Two prefill-side patches:
1. Mooncake transfer queue sharded by a hash of `bootstrap_room` (the stock port-sum shard is constant for a DP1 decode,
   so one of four workers did all sends; `room % 4` alone equals the prefill DP rank). Same idea as open sglang #42163.
2. DCP pack: per-layer metadata cached on the device (the stock kernel wrapper does a pageable `torch.tensor(...,
   device=)` H2D every call) and the gather on a priority -1 stream, no default-stream wait. 78 -> 41 (no-wait only)
   -> 0.6 ms per pack; KV transfer p50 12.3 s -> 0.93 s.

Start with `opt.readme.md` (every optimization with evidence and how to apply it). Then `REPRODUCE.md`, `notes.md`.

## Layout
- `opt.readme.md` - the optimizations, ordered by impact, with measurements and apply steps.
- `patches/` - mounted files (`conn.yihou-dcpshard2*.py`, `dcp_pack.yihou-nowait2.py`), their generators (`mk_*.py`
  applied to overlay 93034eba91), the log-only stats tails, and `tests/` (pack kernel equality + timing on one GPU).
- `results/summary.tsv`, `results/xfer_stats.txt` (per-round pack/RDMA aggregates), `results/rounds/<round>/`,
  `results/probes/` (py-spy, top -H, RDMA counters, per-CPU load), `results/dryruns/`.
- `scripts/envs/`, `scripts/harness/` (config.yihou.dcpx.sh with all knobs, mounts tsv, audit/agentx_env fixes),
  `scripts/tools/` (orchestrators, xsum/xsub/pdbal analysis).
Related: `../glm52.dcpx-xnode-dcp4decode.packup_20261022/` (dcpx1-6, image build), `../glm52.dcp4-decode-gmu.packup_20261023/`
(other session: decode mem > 0.85 via empty_cache), `../glm52.llying-dcp-s1s3.packup_20261022/` (llying v0522 same-node config).

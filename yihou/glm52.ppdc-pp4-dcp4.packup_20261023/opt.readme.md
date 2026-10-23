# Optimizations added in the ppdc rounds (on top of ../glm52.dcpx-pack-fix-best.packup_20261023/opt.readme.md)

Knobs are in `scripts/harness/config.yihou.dcpx.sh` (YIHOU_OVERRIDES of `scripts/envs/env.ppdc1-135x136.yihou.sh`).
Round 1 changed all of these at once (user: apply llying's options, choose trade-offs) - their single contributions are
not separated except write_back.

| # | change | knob | source / evidence | taken? |
|---|---|---|---|---|
| 1 | Prefill PP4 x TP1 (stage k = GPU k), `SGLANG_PP_LAYER_PARTITION=20,20,20,18`, chunk + max-prefill-tokens 8192, `--disable-overlap-schedule`, deepseek_v2 PP fix (`patches/pp-deepseek.patch`, mounted file `deepseek_v2.yihou-ppdc.py`) | `DCPX_PP=4` (also router affinity 0) | llying 2p1d-v0522 kit-v6-pp4 (4+8: +5.9..17 %, mixed with prefill MTP) | yes |
| 2 | Prefill `--schedule-policy shortest-prefill-first` | `DCPX_SPF=1` | llying 2p1d-aus +15.4 % (C176) | yes |
| 3 | Prefill HiCache write_back | `PREFILL_HICACHE_WRITE_POLICY=write_back` | llying +19.3 % (C176); here 0 % at C56 (ppdc3) | yes (neutral at C56) |
| 4 | No OffloaderV2 on the PP prefill (per stage ~94-120 GB weights, KV pool 10.74 M tokens) | `DCPX_NOOFFLOAD=1` | other-session S3 -19 % with it (not paired) | yes |
| 5 | Prefill mem 0.88, max-running 256 | overrides | llying PP kit | yes |
| 6 | Decode all-rank `torch.cuda.empty_cache()` when free < 6 GiB every 8 loops + memlog (`patches/ec03.diff`, `memlog01.diff`) | `DCPX_DECODE_EC=1` | other session gmu pack-up: 0.88 stable, 27,685 same-node | yes, with decode mem 0.88 |
| 7 | Decode max-running 96 = graph max 96 (no eager full-width DSA allocations above graph bs) | override | llying dcp4-improve §4.1 | yes |
| - | decode mem 0.92 | - | slower in the other session (22,540 vs 27,685) | no |
| - | allocator envs, ctx 256K, topk v2, HSA_NO_SCRATCH_RECLAIM, pp async depth | - | no PD A/B / did not help | not yet |
| - | R1 / R2 / session affinity / decode request balance | - | one prefill DP + one decode DP: nothing to choose | n/a for DCP (applies to a DPA decode) |
| 8 | AgentX per-GPU denominator under PP: `agentx_env.py` reports TP = tp*pp | harness | otherwise per_gpu = total/5 | required |

Per-stage transfer (prefill `[yihou-xfer]`, C56): 4 stages balanced, ~3.1 GB/s each, 0.6 ms/pack (pack 23 % of send
time - smaller per-stage sends, 1.9 ms per RDMA call at 12.8 GB/s).

Candidates from llying's 10-23 work (not yet run here; `llying/dcp4-improve-20261023`, `llying/atom-pr209-analysis-20261023`):
DCP index-K sharded per rank like ATOM (+~11-18 % decode KV), draft-layer KV counted x4 (`pool_configurator.py:303`,
~4 % slots), eager full-width DSA buffers, custom AR costs 14 GiB/GPU under DCP, PP4 + DPA4 decode (their 4+4 C80 37,310
vs DCP4 30,903 - decode step ~30 % slower under DCP at equal load), decode request balance (+6.9 % at 4+8 C88).

# Environment

Both rounds ran on the same node and the same image; only the decode mount + 2 env vars differ (dryrun diff).

## Time

| round | chain start | bring-up done (bench A1 start) | A1 end | A2 end | A3 end |
|---|---|---|---|---|---|
| tb56 (runshare v2) | 2026-10-13 06:36:12Z | 06:53:15Z | 08:22:58Z | 09:52:48Z | 11:33:42Z |
| tb56n (no patch) | 2026-10-13 12:04:36Z (tb56 stopped 12:03:54-12:04:36) | 12:17:29Z | 13:45:11Z | 15:11:28Z | 16:48:29Z |

## Hardware — `crsuse2-m2m-085` (captured 2026-10-14 01:57 UTC, after the rounds; raw: `environment-raw/collect_env.085.20261014.txt`)

| item | value |
|---|---|
| cluster / job | spur (Crusoe), partition `amd-spur`, account `amd-oai`, QoS `amd-oai-qos`, job **211169** (24 h hold, gpu:8, -c 160) |
| host / data IP | `crsuse2-m2m-085` / `10.245.153.145` (NIC `ens3`, mlx5_0) |
| OS / kernel | Ubuntu 24.04.4 LTS / 6.8.0-107-generic |
| CPU / RAM | 2x AMD EPYC 9575F 64-Core (236 logical CPUs visible), 2.7 TiB RAM, no swap |
| GPU | 8x AMD Instinct MI355X (gfx950), 309,220,868,096 B VRAM each; amdgpu driver 6.14.14 (rocm-smi) |
| RDMA | `ionic_0..7` all Active, LinkUp, 400 Gb/s, Ethernet (RoCE); `mlx5_0` 200 Gb/s (ens3) |
| docker | 29.7.2 |
| ECC | not captured for 085 in this session [gap]; no fault lines in either round's server logs (`Traceback`/`Memory access fault` count 0 in decode log, both rounds) [M] |

Topology (both rounds): same-node 1P1D P4D4 — prefill GPU 0-3 (`HIP_VISIBLE_DEVICES=0,1,2,3`), decode GPU 4-7, each
TP4 + DP4 with DP attention, EP1. Rails passed: `--disaggregation-ib-device {"0":"ionic_0",...,"3":"ionic_3"}` on both
legs; env (dryrun) `MC_TE_FILTERS=yihou_no_such_rdma_device`, `MC_DISABLE_HIP_TRANSPORT=1`,
`MOONCAKE_DISABLE_HIP_DMABUF=0`, `MC_ENABLE_DEST_DEVICE_AFFINITY=1` [M, dryrun]. Which mooncake transport actually
carried the KV in these rounds was not re-checked for this packup [gap; see the baseline kit for the transport notes].

## Software

| item | value |
|---|---|
| image | `infera-sglang:v0521-yihou-1010-mc4340`, ID **`sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`** on 085 [M collect_env] |
| image tar | `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` (loaded on 085 2026-10-13 05:10-05:14) |
| image content | sglang v0.5.21 recipe + Mooncake #4340 HIP-IPC sub-allocation relocation fix (faae8dd4 port); engine.so sha256 3c96489a… (per `config.yihou.mc4340n2.sh` header) |
| sglang source (code reading) | `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang`; `srt/disaggregation/decode.py` sha256 `97a2abfa61b256e4b840395677601ef342de9c0831ef2ad128cd80c4198a0c1b` (md5 bd1e5fe7…) = base of the runshare2 patch |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (read-only mount) |
| etcd | `quay.io/coreos/etcd:v3.5.14` (per round container) |
| AgentX client | InferenceX checkout `/shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX` @ `918524ff94045b3f091115f1051c22a8588edf2b` (via symlink `$WS/cache/InferenceX`) |
| harness | `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` (not in git; verbatim copy in `scripts/harness/`) |
| repo | `/home/yihou/dev/git/infera.glm52.pd` branch `dev/pd_opt/glm_5.2_agentx` @ `a3ba39687e884201839f75a45374cc145d07e7fa` (dirty tree; the harness does not live in the repo; collect_env could not record git) |

## Engine config (both rounds, from dryrun; tb56 adds the decode mount + env)

| | prefill | decode |
|---|---|---|
| mem-fraction-static | 0.88 | 0.95 |
| max_total_num_tokens / DP rank [M log] | 2,433,088 | 2,762,176 |
| max-running-requests (arg) | 64 | 96 |
| HiCache | `--enable-hierarchical-cache --hicache-ratio 1.5` (no size) = 3,649,664 host tokens/rank [M JSON `cpu_total_tokens`], write_through, kernel IO, page_first, `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` | none |
| DSA backends | triton / triton | triton / triton |
| other | chunked prefill 32768, cuda-graph-max-bs-prefill 64, kv fp8_e4m3, index_share_for_mtp_iteration=false, kv-events on | EAGLE 5/1/6, `SGLANG_SIMULATE_ACC_LEN=3.61` (match-expected, real-draft-token), cuda-graph-max-bs-decode 64, `--disable-custom-all-reduce` |
| runtime mounts (both legs) | `kv_events_publisher.yihou-noblock.py`, `forward_mla_rocm.yihou-noabsorbcfg.py`, `conn.yihou-regbase.py` | same + (tb56 only) `decode.yihou-runshare2.py` |

Correctness is waived by construction (`SGLANG_SIMULATE_ACC_LEN=3.61` forces the accept count).

## Gaps

- 085 env captured after the rounds (2026-10-14), not during; GPU VRAM in that snapshot is from the later x128p stack.
- No ECC readout for 085 recorded.
- Exact sbatch submit line and JIT-seed command not logged (REPRODUCE marks them [inferred]).
- Nodes 026 / 206 env not captured (not used by these rounds).

# SGLang 1P1D — GLM-5.2-FP8 + MTP + DP-attention on MI325X (runnable scripts)

Runnable reproduction package for **one prefill node + one decode node**, fronted by
the Infera router, with the KV cache moved over **Mooncake RDMA (RoCEv2)**.

The engine recipe is the one already validated aggregated (single-node) in
`inference_glm5p2_sglang/run_sglang_mtp.sh`: GLM-5.2-FP8 weights, fp8 KV cache, DSA
sparse attention on the tilelang backends, DP-attention at TP8/DP8, and MTP (EAGLE
against the checkpoint's own nextn layer). PD changes only the disaggregation flags —
if a number here looks wrong, compare against `sglang_naive_engine.sh`, which runs the
identical recipe on one node.

Every script is driven by environment variables; you should not need to edit any of
them. The IPs below are this cluster's — substitute your own.

## Scripts

| Script                       | Where        | Purpose                                                   |
| ---------------------------- | ------------ | --------------------------------------------------------- |
| `install.sh`                 | both nodes   | Install Infera + deps + etcd into the running container.   |
| `patch_mooncake_hip.sh`      | both nodes   | Gate Mooncake's HIP IPC transport so PD uses RDMA.         |
| `patch_sglang.sh`            | both nodes   | Apply the two out-of-tree SGLang patches.                  |
| `preflight_rdma.sh`          | both nodes   | RDMA devices, GID index, cross-node bandwidth.             |
| `infera_0_etcd.sh`           | prefill node | etcd (the shared worker registry).                         |
| `infera_1_server.sh`         | prefill node | Infera router — the single OpenAI endpoint on `:8000`.     |
| `infera_2_sglang_prefill.sh` | prefill node | SGLang prefill leg, 8 GPUs, TP8/DP8.                       |
| `infera_3_sglang_decode.sh`  | decode node  | SGLang decode leg, 8 GPUs, TP8/DP8.                        |
| `curl.sh`                    | prefill node | Smoke-test the pair through the router.                    |
| `bench.sh`                   | prefill node | Throughput sweep via `sglang.bench_serving`.               |
| `sglang_naive_engine.sh`     | one node     | Aggregated baseline: same recipe, no PD, no Infera.        |
| `stop.sh`                    | both nodes   | Tear down and wait for VRAM to come back.                  |

## Topology (1P1D)

| Node                            | Role                                                    |
| :------------------------------ | :------------------------------------------------------ |
| prefill node (`10.32.17.210`)   | etcd + Infera router + prefill leg + verify + benchmark |
| decode node (`10.32.17.209`)    | decode leg                                              |

```
client ──HTTP──> router :8000 ──HTTP──> prefill :30001  (8x MI325X, TP8/DP8)
                    │                        │
                 etcd :2379              Mooncake RDMA over rdma0 (RoCEv2)
                    │                        ▼
                    └──HTTP────────────> decode :31001   (8x MI325X, TP8/DP8)
```

etcd and the router live on the prefill node only. Both legs self-register into etcd
with a lease; the router watches the prefix and pairs them. There is no static worker
list and nothing to reconfigure when a leg restarts.

## The recipe

Both legs run the **same** flags and differ only by `--disaggregation-mode`, port, and
the prefill leg's bootstrap port:

| | prefill | decode |
| --- | --- | --- |
| GPUs / parallelism | 8, TP8 + DP8 (`--enable-dp-attention`) | 8, TP8 + DP8 |
| port / bootstrap | `30001` / `8998` | `31001` / — |
| mem-fraction | `0.85` | `0.85` |

Shared: `--kv-cache-dtype fp8_e4m3`, `--dsa-prefill-backend tilelang
--dsa-decode-backend tilelang`, `--speculative-algorithm EAGLE --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`,
`--disable-custom-all-reduce`, `--reasoning-parser glm45 --tool-call-parser glm47`,
`--disaggregation-transfer-backend mooncake --disaggregation-ib-device rdma0`, plus env
`MC_GID_INDEX=3 HSA_NO_SCRATCH_RECLAIM=1 SGLANG_DSA_TRITON_PREFILL=1`.

SGLang adjusts a few of these itself and says so in the log: DP-attention divides
`chunked-prefill-size` 131072 → 16384, and speculative decoding caps
`max-running-requests` at 48.

## 1. Prerequisites

```text
Hardware: 2 nodes, 8x MI325X (gfx942) each, ROCm 7.2.0, 8x Broadcom bnxt_re RoCE rails
Model:    /wekafs/models/GLM-5.2-FP8  (~704 GiB, 78 layers + 1 nextn/MTP layer)
Image:    lmsysorg/sglang:v0.5.16-rocm720-mi30x
```

This cluster hands you a **live container** per node (a hostNetwork Kubernetes pod), not
a docker socket — so unlike `../sglang_1p2d_kimi2.6` these scripts do not `docker run`
anything, they start processes inside the container you are already in. The pod must be
created with the RDMA devices (`/dev/infiniband/uverbs*` usable, `CAP_IPC_LOCK`,
unlimited memlock); nothing in here can grant that after the fact.

Run on **both** nodes:

```bash
bash install.sh
```

It installs Infera editable plus the handful of pure-python deps the SGLang image is
missing, and drops the etcd static binary on `/wekafs` (one download per cluster).

### 1b. Patch the container

The image cannot be rebuilt here, so three fixes that belong in it are reproduced at
runtime instead. Run on **both** nodes, before the legs:

```bash
bash patch_mooncake_hip.sh   # ~30 s (incremental C++ build, cached on /wekafs)
bash patch_sglang.sh         # instant
```

Both are idempotent and have a `--status` mode. What they buy:

| Patch | Without it |
| --- | --- |
| Mooncake HIP IPC gate | Every request dies in `hipIpcOpenMemHandle` — HIP IPC is intra-node only and cannot reach the peer. 20 min of loading, then failure on request one. |
| `sglang_disagg` KV wait-event | **Silent corruption.** Every prefill chunk but the last is RDMA-read while the forward still writes those pages, so prompts over ~16k come back partially wrong with no error anywhere. |
| `sglang_dsa` padded rows | Only with `MTP=1`: the HIP DSA indexer asserts `lengths.size(0) == B` as soon as two DP ranks hold a near-but-unequal request count. |

The leg scripts refuse to start when the gate or the KV wait-event is missing, and refuse
to start with `MTP != 0` when the DSA patch is missing — so a forgotten step fails in
seconds instead of twenty minutes, or worse, silently. Rationale and evidence for each:
`deploy/docker/patches/{mooncake_cpp,sglang_disagg,sglang_dsa}/`.

`MTP=1` needs one thing that is not a patch: GLM-5.2's IndexShare must be off, via
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'`. Both leg scripts
add it automatically whenever `MTP != 0`, so this is only worth knowing if you hand-roll a
launch. With IndexShare on, the decode leg hangs in PD warmup forever and `/health` never
leaves 503: `dsa_topk_indices` is None on a PREBUILT batch, which turns `can_cuda_graph`
into a per-rank decision, and the ranks split between the draft CUDA graph and eager with
mismatched collectives. REPORT.zh.md §1.2 has the full trace.

Both MTP prerequisites — this override and the `sglang_dsa` patch — exist because of the
same upstream commit, #30839, which is in v0.5.16. Upstream is tracking the deadlock as
issue #32527 with fix PR #32209, and the padded-row assert as PR #32762 (NPU) and the 4th
item of #32209 (TRT-LLM); all were still open as of 2026-07-29, and no one has done the
aiter/HIP side of the padded-row fix, which is what our patch is. Once #32209 lands, both
workarounds can go.

Do not read PR #31477 as an alternative fix for the deadlock. It is a performance change
for the opposite case — making a seed that *does* arrive consumable by fused top-k — worth
about 3% of TPOT, and it does nothing when the seed is absent, which is the case that
deadlocks. It matters here for a different reason: it is what makes the IndexShare override
free today, since fused top-k is currently off under PD anyway. When #31477 lands, the
override starts costing that 3% and should be traded for #32209.

## 2. Verify the RDMA fabric first

PD moves the KV cache across the fabric on every request. Over TCP the pair is slower
than not disaggregating at all, and the failure is silent — so check before bringing up
a 20-minute model load.

On **both** nodes:

```bash
bash preflight_rdma.sh
```

Expect 8 active ports and a routable RoCEv2 IPv4 GID. `MC_GID_INDEX` matters: these are
Broadcom `bnxt_re` NICs where index 1 is the link-local IPv6 GID and the routable
IPv4-mapped entry is **index 3** (Infera's fleet default of 1 is for ionic NICs). The
wrong index hangs rather than errors. Each run prints that node's `rdma0` rail IP.

Then measure the link, decode node first:

```bash
# decode node
bash preflight_rdma.sh server
# prefill node — PEER is the rail IP the decode node printed
PEER=10.115.45.101 bash preflight_rdma.sh client
```

Both nodes must pin the **same rail** (`rdma0` by default, i.e. `tw-eth0`): same-rail
peers are one hop apart on the fabric, and prefill and decode landing on different rails
is how the RoCE QP fails to reach RTR and the transfer times out under concurrency.

## 3. Run

Order: `etcd → router → prefill → decode → verify → benchmark`.

On the **prefill node** (`10.32.17.210`):

```bash
bash infera_0_etcd.sh
bash infera_1_server.sh
bash infera_2_sglang_prefill.sh
```

On the **decode node** (`10.32.17.209`) — `ETCD_ENDPOINT` must point at the prefill node:

```bash
ETCD_ENDPOINT=10.32.17.210:2379 bash infera_3_sglang_decode.sh
```

Cold start is **~20 minutes per leg**: 704 GiB of weights, read twice because MTP
extracts the nextn layer as the EAGLE draft model. Both legs read the same WekaFS mount,
so starting them together is slower per leg than starting them alone — still much faster
than serially. Do not kill a slow launch; the scripts already raise SGLang's bootstrap
timeout and Infera's readiness timeout well past the default.

The launch scripts end in `tail -f`. `Ctrl-C` stops the tail, not the engine — use
`stop.sh`.

## 4. Verify

On the prefill node:

```bash
bash curl.sh
```

Expect exactly one `prefill` and one `decode` worker, both `active`, and a coherent
answer. Workers only:

```bash
curl -s 127.0.0.1:8000/v1/workers | python3 -m json.tool
```

To confirm the KV actually moved over RDMA rather than falling back, the decode log
should show Mooncake picking the rail:

```bash
grep -E "GID index|installTransport|Device rdma" infera_3_sglang_decode.log
```

For the full correctness suite (needle-in-haystack, HumanEval, determinism), point
`inference_glm5p2_sglang/verify_correctness.py` at the router:

```bash
/wekafs/llying/code/inference_glm5p2_sglang/verify_correctness.py \
    --base-url http://127.0.0.1:8000 --json-out /tmp/verify_pd.json
```

## 5. Benchmark

```bash
bash bench.sh                                  # PD, through the router
CONC="64 128" ISL=4096 OSL=512 bash bench.sh   # narrower sweep
```

Results land in `bench_pd/`. For the aggregated comparison, stop the PD stack, run
`sglang_naive_engine.sh` on one node, and sweep it with `PORT=30000 TAG=agg bash bench.sh`.

## 6. Stop

```bash
bash stop.sh          # engines + router on this node
bash stop.sh --all    # also etcd
```

`stop.sh` waits for `rocm-smi` to report the VRAM back. Relaunching before it does is
how you get an OOM that looks like a config problem.

## Notes & gotchas

1. **Advertise an address the peer can reach.** `HOST_IP` defaults to `$POD_IP`, i.e.
   the node's management IP on `10.32.17.0/21`. That carries the control plane (etcd,
   router→worker HTTP, the Mooncake handshake); the KV itself goes over the RoCE rail,
   which Infera pins separately from the GID at `MC_GID_INDEX`. Do not point `HOST_IP`
   at a rail IP — `hostname -I` lists the rails first, which is why the scripts prefer
   `$POD_IP` over it.
2. **KV events are off on both legs.** With them on, Infera adds
   `--disaggregation-decode-enable-radix-cache`, which SGLang rejects together with
   speculative decoding — MTP would not start. PD here routes round-robin, so a
   prefix-cache view buys the router nothing anyway.
3. **`--disable-custom-all-reduce` is not optional.** `--enable-aiter-allreduce-fusion`
   produces NaN logits on ~8% of long cold prefills for this model; see
   `inference_glm5p2_sglang/KNOWN_ISSUES.md`. The MTP recipe avoids the custom-allreduce
   path entirely.
4. **MTP is capped at 3 steps on gfx942.** `--speculative-num-steps > 3` fails to build.
   3 steps / 4 draft tokens is the verified point; observed accept length is ~3.7–3.9
   of 4.
5. **GLM-5.2 is a DSA model but not DeepSeek-V4.** Both carry `index_topk`, and Infera
   used to key its gfx942 DeepSeek-V4 policy off that alone, which force-injected
   `--attention-backend dsv4 --disable-shared-experts-fusion` and the FlashMLA hack onto
   GLM. Fixed in `infera/engine/dsv4_gfx942.py`; the log should say
   `Use dsa attention backend for DeepSeek with DSA` and nothing about dsv4.
6. **Both legs must agree** on model, TP/DP and KV dtype. A flag that drifts between
   them surfaces as a KV shape mismatch twenty minutes into the load, so the two launch
   scripts deliberately repeat the same block rather than sharing a sourced file.

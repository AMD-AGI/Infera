# Environment

Raw per-node captures in `env/env_crsuse2-m2m-135.txt` and
`env/env_crsuse2-m2m-138.txt`. Those were taken at 06:35 UTC on 2026-09-18, a few
hours before these rounds, on the same two nodes with no reboot or driver change
in between — the hardware and fabric sections are current for this experiment.
Flagging the timing rather than implying they were captured during the rounds.

## Nodes

| role | host | data IP | GPUs used |
|---|---|---|---|
| prefill | `crsuse2-m2m-135` | `10.245.148.209` | devices **2,3,4,5** |
| decode | `crsuse2-m2m-138` | `10.245.157.237` | devices **2,3,4,5** |

Control node and image builder: `crsuse2-m2m-135`.

## Hardware

| | |
|---|---|
| GPU | AMD Instinct MI355X ×8 per node, `gfx950`, 309.22 GB each |
| GPU driver | ROCm `6.14.14` |
| CPU | AMD EPYC 9575F 64-Core ×2 sockets, 236 threads |
| RAM | 2.7 TiB |
| OS | Ubuntu 24.04.4 LTS |
| kernel | 6.8.0-107-generic |

**Only four GPUs per node are used.** On 135, GPU[1] is held by a root-owned
Kubernetes vLLM pod (`pod5d84e491`, `vllm serve Qwen3-32B`, ~97 GB). It is a live
serving workload with no sudo and no kubectl available; it must not be touched.
Devices 2,3,4,5 avoid it and keep `mem_fraction_static` at 0.85.

## RDMA fabric

RoCE over `ionic` HCAs. Rails are **physically isolated** — `ionic_i` reaches
only `ionic_i` — so a mismatched pairing is unreachable rather than slow.

| | |
|---|---|
| rails used | `ionic_2`, `ionic_3`, `ionic_4`, `ionic_5`, all `PORT_ACTIVE`, Ethernet, both nodes |
| per-GPU pinning | `{"0":"ionic_2","1":"ionic_3","2":"ionic_4","3":"ionic_5"}` — keys are the **local (visible)** device index |
| GID index | `MC_GID_INDEX=1` — mandatory; index 0 is link-local `fe80::` and not routable |
| host lib | `/lib/x86_64-linux-gnu/libionic.so`, bind-mounted into the container at `/host-libionic/libionic.so` |
| known bad | 135's `ionic_7` is defective (no netdev, GID index 1 all-zero). Not used here. |

Verification of the four rails at the time of these rounds:
`spec/preflight-nodes/rails-crsuse2-m2m-{135,138}.txt`.

## Software

| | |
|---|---|
| base image | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917` |
| base digest | `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32` |
| base local id | `sha256:ec3b41253277651c5aab25f92a24d04b5cfd5fc8bc2c0b1885bfdb30967133dd` (identical on both nodes) |
| engine image, un-patched | `infera-sglang:v0519-yihou-0917`, id `sha256:4190c3a99d0ea8b195580008e7f37fb2f1dfdbe6254b019884d4bef5721041cd` (identical on both nodes) |
| engine image, NextN fix | `infera-sglang:v0519-yihou-0917-nextnfix` — **135**: `sha256:6d6393c4070a6030917c7d92ad156980f59d4c12022b2ed23883d6218bc01078`, **138**: `sha256:e17780d2d0fc01f5a2f88136aba0a94d929fdbbbfcd06ea38ecf3b5c5b406770` |
| in-image sglang | `0.5.19.dev20260917+ga9fb1c3238` |
| repo | `AMD-AGI/Infera`, branch `dev/pd_opt/glm_5.2_agentx`, HEAD `5a342acffe10e09729662ff40e81b22a4367fd75` |

The two `nextnfix` image IDs **differ by design**: the thin layer was built
independently on each node rather than shipped, because the base is 66 GB and was
already present on both. Content equivalence is established by running the import
check on each node (REPRODUCE.md §3), not by comparing IDs.

The tag is not an identity. `issue.md` §3.2 in the repo records a campaign where
image drift under a stable tag hid a regression; the in-image
`sglang.__version__` string above is the reliable identifier. Read it with:

```bash
docker run --rm --entrypoint python3 <image> -c "import sglang; print(sglang.__version__)"
```

### Patches carried by the engine image

The base engine image applies the repo's DSA patch set at build time
(`deploy/docker/scripts/apply_sglang_dsa_patches.sh`, arm `full`). It leaves
`*.orig` backups, so its presence is checkable inside the image:

```
srt/disaggregation/decode.py            srt/managers/schedule_batch.py
srt/layers/attention/dsa_backend.py     srt/model_executor/forward_batch_info.py
srt/managers/scheduler_components/dp_attn.py
srt/speculative/eagle_worker_v2.py      srt/speculative/base_spec_worker.py
```

That set is cut `--fuzz=0` against the **20260916** nightly; on 20260917 the
`draft_cuda_graph_dp_vote` diff was re-cut because upstream moved the DP-sync
slot it rides. Its column layout on this base was audited by hand during this
investigation and is **correct** — see `notes.md` §4.

## Model and data

| | |
|---|---|
| weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (shared NFS, same absolute path on both nodes), quant `quark` |
| served name | `glm5.2-mxfp4` |
| dataset | none — all probes are literal prompts embedded in `scripts/probe.yihou.sh` and in REPRODUCE.md |

## Shape

P4+DPA / D4+DPA: TP4, DP4, EP1, DP attention on, `mem_fraction_static` 0.85,
`kv-cache-dtype fp8_e4m3`, HiCache off both sides, `page_size` 64.
MTP: EAGLE, 5 steps, topk 1, 6 draft tokens.
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'`.
Simulated acceptance **off** (`DECODE_SIMULATE_ACC_LEN=""`) in every round here —
that is what makes the acceptance numbers real rather than forced.

## Secrets

**None required.** No registry login (all images are local to the nodes), no API
keys, no tokens, no etcd credentials — `launch.sh` starts its own etcd on the
control node. Access is ssh to the two hosts plus local Docker. The engine's
`server_args` dumps contain `api_key` and `ssl_keyfile_password` fields; both are
`None` in every captured artifact in this packup.

# GLM-5.2 on MI355X: public reproduction and upstream patch map

Updated 2026-10-28 (UTC). Audience: InferenceX and other external reproductions.
Start here for the experiment; use [PR_PACKUP_MAP.md](PR_PACKUP_MAP.md) for every
maintained patch's purpose, PR state, evidence archive and applicability.
[PR_STATUS.md](PR_STATUS.md) retains the maintenance history.

## 1. What result can be reproduced?

The requested reference is **llr37-c104** in
[glm52.llrepro-c104-conc.packup_20261027](../glm52.llrepro-c104-conc.packup_20261027/README.md):
**48,713.28739 total tokens/s/GPU**, over **eight participating MI355X GPUs**,
with four prefill GPUs on one node and four decode GPUs on another.

This is a **historical performance experiment with simulated MTP acceptance**,
not a correctness-qualified production result. Its real-acceptance counterpart
garbled outputs; the bisect implicated prefill PP communication overlap and decode
custom all-reduce. Applying the PRs listed here does not establish that this
48,713 configuration is correct. No stack-level score is evidence for an individual PR.

| Reference | Reported total tokens/s/GPU | Interpretation |
|---|---:|---|
| llr37, C104 | 48,713.28739 | Requested historical peak, one run; simulated acceptance 2.99 |
| llr37r, C104 | 48,103 | Later repeat; two-run rounded mean 48,408 |
| llr48, C88 | 44,245 | Later experiment with PP overlap and custom AR off; still simulated acceptance, one run, correctness not certified |

The repeat and llr48 are in the newer
[2026-10-28 sweep](../glm52.llrepro-garblefree-sweep.packup_20261028/README.md).
That archive explicitly says the approximately 48k stack is not a deliverable
correctness result. Turning off the two implicated switches is a control to test,
not proof that all long-context or fused-kernel problems are fixed.

The [raw llr37 JSON](../glm52.llrepro-c104-conc.packup_20261027/results/llr/llr37-c104/agentx_conc104.json)
reports **48,310.33559 input** and **402.95180 output** tokens/s/GPU. The total
counts replayed input tokens, including cache hits; it is not decode-only token
generation. It records 11,627 profiled records, 1,155 warm-up records dropped,
five error records, 3.20894 mean QPS, TTFT p50/p90 5.97246/14.97585 s and mean
TPOT 15.03 ms. Preserve that accounting when comparing results.

## 2. Frozen software and public availability

The deployment uses **one integrated Infera/SGLang image containing Mooncake**.
Mooncake is loaded as a compiled transfer library in the engine; a separate
Mooncake server image is not part of this recipe. A stock SGLang image alone
does not contain the Infera engine wrapper and Rust router used by llr37.

| Component | Frozen reference / source | Availability and limits |
|---|---|---|
| Experiment archives | Infera branch `dev/pd_opt/glm_5.2_agentx`; audited archive revision `629913ea317293f8d260d4d227fe4206d3cd956f` | Public Git repository; all paths below are repository-relative, not private NFS dependencies |
| SGLang base image | `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018` | Public Docker Hub tag verified 2026-10-28; linux/amd64 manifest `sha256:8f7d8034546b17cc4ae57c63f3f65c584eb73e62378243d1416b8385aa7be286` |
| Infera build source | [`fdbafb6065dae6d1c53936c7d2cf1f5022a514d0`](https://github.com/AMD-AGI/Infera/tree/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0) | Build its `deploy/docker/Dockerfile.sglang`, not the current branch's Dockerfile |
| Historical integrated image | `infera-sglang:v0522-fdbafb60` | Local tag, **not a public registry pull URL**. Recorded image ID `sha256:f69f49e39cbf97e933c742f61d0d9a3c45914e51ab01c99ac4b6eb859858a967`; this is a local config/image ID, not a registry manifest digest |
| SGLang reported version | `0.5.22.dev20261018+g8dee3f6bdb` | From the archived image record; runtime source is further patched by the frozen Dockerfile |
| Mooncake source | [`faae8dd4a6309c3ecd47e0721a83b0250d686fa2`](https://github.com/kvcache-ai/Mooncake/tree/faae8dd4a6309c3ecd47e0721a83b0250d686fa2) plus the two RDMA diffs below | Built inside the Infera image; do not substitute an arbitrary current wheel |
| ROCm / Aiter | ROCm 7.2 base image; retain its Aiter build | Kernel tuning CSVs are build/architecture-specific; record Aiter source and installed package versions in any new result |
| Model | [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4) | Public, ungated API access verified. Current public revision `386bd0e4ec821f7b07975701cec3c3b953a5576a`; archived local weights have no published revision/checksum binding, so byte identity to llr37 is **unverified** |
| Benchmark | [InferenceX `918524ff94045b3f091115f1051c22a8588edf2b`](https://github.com/SemiAnalysisAI/InferenceX/tree/918524ff94045b3f091115f1051c22a8588edf2b) | Exact source revision in archived `runtime.env`; use its client dependency constraints |
| Trace | [semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126), train, 393 entries | Public, ungated API access verified. Current revision `23f152f6f0f9399a85901b89a6458def0ef16729`; original cached dataset revision was not recorded |
| Auxiliary service | `quay.io/coreos/etcd:v3.5.14` | Selected by the archived kit; record the resolved digest on reproduction |

Public availability checks do not mean weights/images were downloaded or a new
model run was performed. The pinned source recipe still resolves some pip/cargo
dependencies at build time; a rebuild is not guaranteed to produce the historical
image ID. Retain the new image ID/digest, `pip freeze`, build log and model/trace
checksums. These remaining identity gaps prevent a claim of bit-for-bit reproduction.
The machine-readable values are in [SOURCE_PINS.json](repro/SOURCE_PINS.json).

## 3. Exact llr37 patch-to-artifact correspondence

There are two layers: patches already baked into the frozen image and files
mounted at launch. **Do not apply all open PRs on top of that image.** A current-main
PR is the upstream maintenance form of a fix, not necessarily a patch that applies
to the October 18 image's source layout.

### 3.1 Patches already baked into the image

Paths in this table are under
[`deploy/docker/` at fdbafb60](https://github.com/AMD-AGI/Infera/tree/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0/deploy/docker).
The [frozen Dockerfile](https://github.com/AMD-AGI/Infera/blob/fdbafb6065dae6d1c53936c7d2cf1f5022a514d0/deploy/docker/Dockerfile.sglang)
is the authoritative application list and order.

| Image input | Why it exists | Upstream correspondence / action for llr37 |
|---|---|---|
| `patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py` | Align Aiter's physical query rows and real-row metadata | Same problem family as SGLang [#33059](https://github.com/sgl-project/sglang/pull/33059); image script and current PR are different revisions. Already baked; do not stack the current-main diff |
| `patches/sglang_dsa/dsa_dp_sync.diff` | Avoid host synchronizations on rank-divergent DSA eager paths | Historical [#33973](https://github.com/sgl-project/sglang/pull/33973), now maintained as [#46830](https://github.com/sgl-project/sglang/pull/46830); already baked |
| `patches/sglang_dsa/v0522/dsa_page_table_rows.diff` | Match per-request page tables to padded/per-token MTP rows | Separate image patch; **not** #33059 and not graph-output padding D23/#46424. No independent yihou upstream PR assigned; preserve frozen source |
| `patches/sglang_disagg/v0522/patch_prefill_nonfinal_chunk_wait_event.py` | Record producer completion before a non-final chunk send | Consumer fix [#41395](https://github.com/sgl-project/sglang/pull/41395) is in the base; producer/PP work is related to [#45524](https://github.com/sgl-project/sglang/pull/45524). Preserve this historical producer patch; do not assume the newer PR is byte-equivalent |
| `patches/sglang_responses/v0522/patch_responses_unstreamed_tool_args.py` | Reconcile unfinished Responses tool-argument deltas | Inherited image behavior, not a claimed AgentX speedup or a new yihou PR |
| `patches/sglang_rocm/patch_hicache_rocm_staged_write_back.py` | Keep unsupported ROCm staged host-cache writes on the supported path | Inherited HiCache integration; not OffloaderV2 S06. Preserve frozen source |
| `patches/sglang_rocm/v0522/patch_mla_absorb_bmm_default_config.py` | Avoid the hand-tuned configuration that triggers a Triton/LLVM abort | S02 workaround family; tracked as [issue #46829](https://github.com/sgl-project/sglang/issues/46829), deliberately not a workaround PR |
| `patches/sglang_kv_events/patch_kv_metrics_send_noblock.py` | Prevent backpressured auxiliary metrics from freezing the scheduler | S01 / [#46820](https://github.com/sgl-project/sglang/pull/46820); already baked, no extra runtime mount |
| `patches/mooncake_cpp/rdma_dest_local_rail.diff` | Select a source rail compatible with the destination's HCA | llying's inherited RDMA patch; not yihou M01/M02, no ownership claim |
| `patches/mooncake_cpp/rdma_ack_timeout.diff` | Make RC ACK timeout configurable for the fabric | llying's inherited RDMA patch; selected with `MC_IB_TIMEOUT=18`, not Mooncake #5005 |
| Infera engine wrapper, Rust router and host-libionic entrypoint | Worker discovery, KV-aware routing, per-decode demand and NIC ABI support | Build all of the pinned Infera source. [#210](https://github.com/AMD-AGI/Infera/pull/210) is merged recipe history, not an instruction to replace fdbafb60 with arbitrary main |

The DSA installer **asserts** the already-upstream draft CUDA-graph DP vote
(#32196) and NextN fusion architecture; it does not reapply their old diffs.
The decode-radix-spec installer is a no-op for this base, and llr37 does not turn
on the rejected decode-radix experiment.

Infera [#218](https://github.com/AMD-AGI/Infera/pull/218) is a build-check-only
follow-up: it prevents SIGPIPE from falsely rejecting a valid Mooncake artifact.
It is not a runtime performance patch in the historical image. An optional
source rebuild backport is shown below and must be recorded as a build-input change.

### 3.2 Files added or replaced at launch

| Inventory item | Public input and exact destination | llr37 requirement / PR |
|---|---|---|
| **S13** | [kit `patches/deepseek_v2.py`](../glm52.llrepro-pp4-dpa4.packup_20261024/kit/sglang_2p1d_glm5.2/patches/deepseek_v2.py) → prefill `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py:ro` | Required PP non-first-stage hidden-size fix. llying's earlier kit version is used. No new PR pending attribution agreement; old #34465 is closed unmerged |
| **K01** | [complete NEWFILES diff](patches/K01-kf-decode-kernel-overlay.NEWFILES.diff) → reconstructed `overlay/` → decode `/yihou_overlay:ro` | Required historical KF sparse-MLA, indexer/top-k and MoE candidates; no upstream PR, not a qualified kernel release |
| **A01, first CSV** | [75-row DPA/K3 CSV](../glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/aiter_retune/merged/yihou_glm52_dpa_bf16_tuned_gemm.csv) → decode `/sgl-workspace/aiter/aiter/configs/model_configs/yihou_glm52_dpa_bf16_tuned_gemm.csv:ro` | Mounted in llr32 onward; no Aiter PR yet |
| **A01, second CSV** | [49-row PTPC CSV](../glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/aiter_retune/ptpc/yihou_glm52_ptpc_bf16_tuned_gemm.csv) → same directory, original filename | Mounted in llr35/llr37; no Aiter PR yet; the second CSV alone did not establish a speedup |
| **I07** | c104 [`scripts/worker.sh`](../glm52.llrepro-c104-conc.packup_20261027/scripts/worker.sh) and [`common.sh`](../glm52.llrepro-c104-conc.packup_20261027/scripts/common.sh) replace the corresponding base-kit files | Required role-specific environment, custom-AR and multiple-mount hooks. Reproduction tooling, not an Infera core bugfix/feature PR |
| llr37 configuration | [`sites/llr37.site.env`](../glm52.llrepro-c104-conc.packup_20261027/sites/llr37.site.env) with only site values remapped | C104 plus the llr35 stack; no additional C104 code patch |

Use the inventory's **complete K01 diff**, not the older llr5–llr7 overlay copy.
The named `glm52.kf-dcp-and-fused-indexer.packup_20261025` directory cited by the
original inventory is absent from this public branch. Its full overlay source is
available in K01 instead. We reconstructed it and verified that its import-hook
file plus the later llr42 alias diff equals the archived llr42 hook byte-for-byte.
Do **not** apply that alias diff or enable `YIHOU_KF_FUSED_INDEXER` for llr37;
that later experiment was not retained. The DCP overlay switches are also off.

No OffloaderV2 weight offload, DCP branch, DCP packing optimization, hashed-room
transfer mount, periodic `empty_cache`, Mooncake HIP-priority change or online
MXFP4 draft patch was added to llr37. Its **FP8 draft is an existing flag**
(`SGLANG_GLM_NEXTN_MOE_PTPC=1`), distinct from excluded S14's MXFP4 experiment.
HiCache is a KV-cache host tier and must not be confused with OffloaderV2 weights.

## 4. Operating point and measurement contract

| Setting | llr37 value |
|---|---|
| Prefill | One worker, **PP4 × TP1**, four MI355X; layer split `20,20,20,18`; EP1 |
| Decode | One worker, **TP4 / attention-DP4**, four MI355X; EP1; DCP1 |
| GPU denominator | 4 prefill + 4 decode = **8**, regardless of unused GPUs in either host |
| Prefill batching | Chunk/max-prefill tokens 8192; static memory 0.88; `--pp-async-batch-depth 1`; shortest-prefill-first |
| Prefill HiCache | 250 GB per PP rank; ratio 1.5; `write_back`, `kernel`, `page_first`; host-register chunk 512 GB |
| Decode memory / graphs | Static memory 0.95; max-running 256; launcher graph bound 256 (effective per-DP-rank bound 64); four tokenizer workers |
| KV / DSA | FP8 E4M3 KV; Triton DSA prefill/decode backends |
| MTP, **both roles** | EAGLE steps/top-k/draft-tokens **3/1/4**; `index_share_for_mtp_iteration=true` |
| Simulated acceptance | Decode `SGLANG_SIMULATE_ACC_LEN=2.99`, `match-expected`, `real-draft-token` |
| Decode overlay | `YIHOU_KF_SPARSE=1`, `YIHOU_KF_INDEXER=1`, `YIHOU_KF_MOE=1`, `SGLANG_OPT_USE_TOPK_V2=false` |
| Other retained settings | Decode `--stream-interval 30`, `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`, FP8 draft flag above |
| Known problematic settings | Prefill `SGLANG_PP_COMM_OVERLAP=1`; decode `DECODE_CUSTOM_AR=1`; inherited fused QK/RoPE and Aiter all-reduce fusion flags remain in the worker |
| Router | Infera Rust, KV-aware; overlap weights 20/2; session affinity on prefill; guard release `completion`; decode input demand on |
| Mooncake / fabric | RDMA, eight ionic rails in the archive, GID index 1, dma-buf enabled; `MC_DISABLE_HIP_TRANSPORT=1`, destination-device affinity/local rail on, timeout 18, max MR 1 GiB |
| AgentX | C104, duration 3600 s, 10 warm-up requests per lane, one A1 run; HTTP session IDs from correlation IDs |

Select GPU IDs, RDMA devices, GID and libionic according to your allocated hosts.
The archive used GPUs 4–7 on each host; the assembler defaults to four dedicated
GPUs 0–3. Provide sufficient pinned host RAM for approximately 1 TB of prefill
HiCache plus the model/runtime. Driver, NIC topology and host RAM changes are part
of the reproduction environment, not innocuous filename substitutions.

**Metadata discrepancy:** the old AgentX wrapper reports prefill TP4/PP1 in its
JSON although the actual worker launches PP4/TP1. The eight-GPU denominator is
correct. Report the actual topology above alongside the untouched raw JSON;
do not silently rewrite archived results. The base pack-up's `environment.md`
also contains older DCP-run details: use the llr37 site, worker and engine-config
records for this recipe.

## 5. Reproduce from public inputs

These commands prepare an independent deployment on **your allocated hosts**.
No access to the archived AMD hostnames or private NFS is needed. Shell examples
assume a large shared data/NFS directory available at the same absolute path on
both nodes. Set Docker's build storage and caches on a sufficiently large data
disk. The helper only assembles source/configuration; it does not start containers
or access GPUs, and refuses an existing output directory.

### Fetch and assemble

```bash
export GLM_REPRO_ROOT=/data/yihou/glm52-public
mkdir -p "$GLM_REPRO_ROOT"
git clone --branch dev/pd_opt/glm_5.2_agentx https://github.com/AMD-AGI/Infera.git "$GLM_REPRO_ROOT/Infera"
export GLM_REPO="$GLM_REPRO_ROOT/Infera"
export GLM_INV="$GLM_REPO/yihou/glm52.patch-inventory_20261027"
git -C "$GLM_REPO" rev-parse HEAD
(cd "$GLM_REPO" && sha256sum -c yihou/glm52.patch-inventory_20261027/repro/SHA256SUMS)

# With huggingface_hub installed, download the currently public pinned model.
# This pin has not been proven identical to the archived local weights.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "amd/GLM-5.2-MXFP4",
    revision="386bd0e4ec821f7b07975701cec3c3b953a5576a",
    local_dir="/data/yihou/models/GLM-5.2-MXFP4",
)
PY

python "$GLM_INV/repro/prepare_llr37.py" \
  --output "$GLM_REPRO_ROOT/llr37" \
  --prefill-ip 192.0.2.10 --decode-ip 192.0.2.11 \
  --decode-ssh your-decode-host \
  --model /data/yihou/models/GLM-5.2-MXFP4 \
  --rdma-devices ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
```

Replace the documentation IPs/SSH name and device list. The helper composes the
base PP4 kit, c104 hooks, K01 source, both CSVs and the exact llr37 tuning values.
It emits `site.env`, `real-acceptance.env` and a per-file `ASSEMBLY.json` hash/source
manifest. Review the generated site before launch; choose a unique `--container-prefix`
and free ports for each independent deployment. Do not execute the archived
`run_llr.yihou.sh`, queue/guard scripts or `nfs_cleanup/` scripts unchanged: they
contain original host paths and lifecycle/cleanup actions. The steps here use the
kit directly instead.

### Build the integrated image

```bash
export GLM_IMAGE_CONTEXT="$GLM_REPRO_ROOT/image-fdbafb60"
mkdir "$GLM_IMAGE_CONTEXT"
git -C "$GLM_REPO" archive fdbafb6065dae6d1c53936c7d2cf1f5022a514d0 | tar -x -C "$GLM_IMAGE_CONTEXT"

# Optional build-check backport, independent of runtime performance:
git -C "$GLM_IMAGE_CONTEXT" apply --no-index --check \
  --include=deploy/docker/scripts/build_mooncake_sglang.sh "$GLM_INV/reviews/I05c-ready.diff"
git -C "$GLM_IMAGE_CONTEXT" apply --no-index \
  --include=deploy/docker/scripts/build_mooncake_sglang.sh "$GLM_INV/reviews/I05c-ready.diff"

docker build --network host \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang-rocm@sha256:8f7d8034546b17cc4ae57c63f3f65c584eb73e62378243d1416b8385aa7be286 \
  -f "$GLM_IMAGE_CONTEXT/deploy/docker/Dockerfile.sglang" \
  -t infera-sglang:v0522-fdbafb60 "$GLM_IMAGE_CONTEXT"
docker image inspect infera-sglang:v0522-fdbafb60 --format '{{.Id}} {{json .RepoDigests}}'
docker save infera-sglang:v0522-fdbafb60 | ssh your-decode-host docker load
```

The optional #218 backport changes only artifact inspection, not Mooncake source
or its compilation. A failed `--check` is a source mismatch to investigate, not
permission to force a patch. The original image build recipe already applies
section 3.1; **do not cherry-pick #33059/#46820/#46830 again into it**. Keep the same
resulting integrated image on both hosts. No full image rebuild was performed as
part of preparing this guide.

### Launch the historical performance experiment

On the prefill host, with the image, model and assembled kit visible on both hosts:

```bash
source "$GLM_REPRO_ROOT/llr37/site.env"
export RUN_ID=public-llr37-c104
export GLM_KIT="$GLM_REPRO_ROOT/llr37/kit/sglang_2p1d_glm5.2"
bash "$GLM_KIT/engine/up.sh"
bash "$GLM_KIT/engine/agentx/agentx.sh"
```

`up.sh` starts etcd, the two roles and the router, waits for health and PD worker
registration, and writes the effective engine settings. `agentx.sh` fetches the
pinned InferenceX commit and records the replay command and metrics. Both nodes
need passwordless SSH and Docker access; model, overlay and CSV paths must exist
on the node that mounts them. Snapshot all container logs/inspect output before
stopping only your own deployment. Start each repeat with fresh service processes;
record whether JIT caches are warm.

Before trusting a score, confirm in logs/inspect:

1. PP4/TP1 prefill, TP4/attention-DP4 decode, and matching 3/1/4 MTP flags on both roles.
2. Prefill `deepseek_v2.py` bind mount and all three decode mounts (overlay + two CSVs).
3. `[yihou-kf] installed` and per-shape checks/dispatch; no overlay import failure
   or unexpected permanent fallback. A configured environment variable alone
   does not prove a kernel executed.
4. Aiter reports merging both CSV filenames without duplicate shape keys.
5. Actual RDMA devices/transport, image IDs, simulated-acceptance flags and all
   error/timeout counts. The archive had failures; do not describe it as zero-error.

### Validate correctness separately

Use a **fresh deployment** sourced from generated `real-acceptance.env` instead
of `site.env`; it removes simulated acceptance, turns off custom AR and PP
communication overlap. Then use the kit's `engine/smoke.sh` and `engine/eval.sh`,
plus long-context/needle checks representative of the intended benchmark.
These controls are not certified to pass: inherited fused flags, KF numerical
coverage, long-context correctness and transfer timeouts still require attribution.
Never report the historical 48,713 number for this changed configuration without
measuring it. Any additional correction is a separately identified recipe revision.

## 6. How to use the current PRs instead of the frozen image

[PR_PACKUP_MAP.md](PR_PACKUP_MAP.md) maps each PR to its archive and distinguishes
`BAKED`, `MOUNTED`, `BUILD-ONLY`, and `NOT IN LLR37`. Full PR heads/bases and state
timestamps are in [PR_SNAPSHOT.json](repro/PR_SNAPSHOT.json). Ready for review is
not approval, merge, or whole-stack correctness certification.

For a current-main integration, resolve each fix against that exact main/image
revision, drop superseded fixes, and port only missing behavior with tests.
Review dependencies first: #44322 → #46824 → #46826 describes the **separate DCP
recipe**, not llr37; #46828 is a DCP/Q-projection option. #46819 intentionally
retains the producer wait, unlike the archived DCP experiment. Do not transfer
old combined performance claims to these revised PRs. This guide does not claim
that current upstream main plus all listed PRs has reproduced the frozen score.

For external handoff, the remaining gaps are concrete:

- No public registry location has been established for the historical integrated
  image ID; rebuild from public pinned sources or publish a verified immutable image.
- Bind historical model/trace bytes and full dependency versions to checksums.
- Qualify the retained KF kernels and Aiter CSVs; neither has a ready upstream PR.
- Resolve S13 attribution with llying before filing it; its archived source is
  already available for reproducing the old deployment.
- Finish real-acceptance correctness and representative long-context validation;
  report the resulting configuration's own repeated performance and failure rates.

## 7. Evidence and maintenance

The reproducibility chain is:

| Pack-up | Role in llr37 |
|---|---|
| [PP4/DPA4 base, Oct 24](../glm52.llrepro-pp4-dpa4.packup_20261024/README.md) | Frozen image recipe/ID; full kit and S13 prefill source; llr1–llr3 baseline |
| [KF/AR/top-k, Oct 24](../glm52.llrepro-kf-ar-topk.packup_20261024/README.md) | llr5–llr7 activation evidence; its older overlay is superseded by K01 for this guide |
| [PP overlap/concurrency, Oct 25](../glm52.llrepro-ppcomm-c88.packup_20261025/README.md) | Historical switch/concurrency exploration; correctness caveat remains |
| [K3 MTP, Oct 26](../glm52.llrepro-k3-mtp.packup_20261026/README.md) | 3/1/4 speculation and acceptance sensitivity |
| [K3/FP8/GEMM/memory, Oct 27](../glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/README.md) | llr35/35r stack; two CSVs; memory 0.95 and FP8 draft |
| [C104, Oct 27](../glm52.llrepro-c104-conc.packup_20261027/README.md) | llr37 site/hooks/raw result; C96/C104/C112 comparison |
| [Follow-up sweep, Oct 28](../glm52.llrepro-garblefree-sweep.packup_20261028/README.md) | llr37r repeat, rejected later flags and overlap/custom-AR-off results |

Guide validation was **CPU/read-only source verification**: public URLs/revisions,
artifact hashes, exact source composition, zero-fuzz K01 reconstruction, shell
syntax, generated worker command/environment checks and optional #218 backport
application. No new GPU run or speedup is claimed. See [VALIDATION.md](repro/VALIDATION.md).

When a PR changes state or head, update the map, snapshot and live status together.
When runtime files/configuration change, issue a new recipe identity and preserve
the old site's evidence. Keep runtime caches, credentials, worktrees and private
logs out of commits; never silently replace the archived result files.

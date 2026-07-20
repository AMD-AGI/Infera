# Example: SGLang 1P1D (PD-disaggregated) — DeepSeek-V4-Pro on MI355X

A concrete, copy-paste walk-through of a **PD-disaggregated** SGLang pair — one
**prefill** worker + one **decode** worker on separate nodes — served through the
standard Infera stack, with KV transferred over **Mooncake RDMA** on the ionic
fabric, and throughput-swept end to end. Same tuned DeepSeek-V4-Pro (`fp4`: FP8
attention + MXFP4 MoE) recipe as the aggregated example, split across two 8×MI355X
nodes.

For the single-node aggregated counterpart see
[SGLang mixed — DeepSeek-V4-Pro](sglang_mix_dsv4.md). For the concepts, read
[PD disaggregation](../features/pd_disaggregation.md) first.

```{admonition} PD only pays off over real RDMA
:class: important
Cross-node PD moves the KV cache over the fabric every request. It's a win only
on an RDMA fabric (ionic RoCEv2 here) — over TCP it's *slower*. Infera's launch
preflight refuses a config prone to silent TCP fallback. Bring the fabric up
first ([cross-node prereqs](../features/pd_disaggregation.md#cross-node-prerequisites-rdma)).
```

## The tuned recipe in one line

The PD engine reuses the aggregated recipe's two levers (`--attention-backend
dsv4` + the full fused-compress env set) and adds **DP-attention always on** plus
a large prefill chunk. Both legs share these; they differ only by
`--disaggregation-mode`:

| flag group | value |
|---|---|
| base | `--attention-backend dsv4 --disable-radix-cache --page-size 256 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion --cuda-graph-max-bs 512 --context-length 9472` |
| DP-attn | `--tp-size 8 --dp 8 --enable-dp-attention --ep-size 8 --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000` |
| prefill chunk | `--chunked-prefill-size 163840 --max-prefill-tokens 163840` (÷8 = 20480/rank) |
| PD | `--disaggregation-mode prefill\|decode --disaggregation-transfer-backend mooncake` (prefill also `--disaggregation-bootstrap-port 8998`) |
| mem-fraction | prefill `0.85` (DP-attn prefill OOMs high), decode `0.90` |

## 0. Shared variables (both nodes)

```bash
HF_CACHE=/path/to/hf-cache                     # TODO: HuggingFace cache / model dir (host path)
MODEL=$HF_CACHE/DeepSeek-V4-Pro                 # fp4 DSv4-Pro checkpoint (real weights, not stubs)
IMG=inferaimage/infera:infera-sglang-...        # TODO: update to the image you validated
P_IP=10.0.0.1                                   # TODO: prefill node data-plane (RDMA-rail) IP
D_IP=10.0.0.2                                   # TODO: decode  node data-plane (RDMA-rail) IP
ETCD=$P_IP:2379                                 # etcd lives on the prefill node
```

## 1. etcd — shared registry (on the prefill node host)

```bash
docker run -d --name infera-etcd --network host quay.io/coreos/etcd:v3.5.14 \
  etcd --advertise-client-urls http://$P_IP:2379 --listen-client-urls http://0.0.0.0:2379
```

## 2. Worker container + check RDMA (on **each** node host)

Start one persistent container per node (host net + `/dev/infiniband` = RDMA
passthrough):

```bash
docker run -d --name pd_bench --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined -v /mnt/vast:/mnt/vast --entrypoint "" "$IMG" sleep infinity
```

Now **check RDMA is visible inside the container** — the count must equal the
node's active RDMA ports (e.g. `8`):

```bash
docker exec pd_bench bash -lc "ibv_devinfo | grep -c PORT_ACTIVE"
```

**If that prints `0`** (or `ibv_devinfo` errors), the container's bundled ionic
provider doesn't match the host kernel driver — RDMA would silently degrade to
TCP. Fix it by overlaying the host's `libionic.so` onto both the soname and the
libibverbs provider symlink, then re-check:

```bash
HL=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=$(basename "$HL"); docker cp "$HL" pd_bench:/usr/lib/x86_64-linux-gnu/$B
docker exec pd_bench bash -lc "cd /usr/lib/x86_64-linux-gnu && ln -sf $B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../$B libionic-rdmav34.so && ldconfig; ibv_devinfo | grep -c PORT_ACTIVE"
```

For a fuller cross-node check (RoCE bandwidth + Mooncake KV transfer over the
fabric, not just device visibility), run the preflight suite before launching —
see [Verify the RDMA fabric](#verify-the-rdma-fabric-optional) below.

Run the rest **inside `pd_bench`** on each node; re-export the shared vars.
Export your token first if the model is gated:

```bash
export HF_TOKEN=hf_...                          # TODO: set if the model/tokenizer is gated
```

## 3. Infera router — `infera.server` (inside `pd_bench` on the prefill node, :8000)

One router fronts the whole pair. It watches etcd and **auto-detects the
prefill+decode pair** — no static prefill/decode list, no `sglang_router`:

```bash
python -m infera.server --host 0.0.0.0 --port 8000 --etcd-endpoint $ETCD \
  --router-tokenizer-path $MODEL --router-policy round-robin \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq --router-backend rust
```

## 4. PD engines — one command per leg (inside `pd_bench`)

Export the fused-compress lever on **both** nodes (paste as-is):

```bash
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0 SGLANG_OPT_USE_JIT_INDEXER_METADATA=false SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false SGLANG_DP_USE_GATHERV=1
```

**Prefill leg (prefill node):**

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m infera.engine.sglang --model-path $MODEL --tp-size 8 --trust-remote-code --host 0.0.0.0 --port 30000 --advertise-host $P_IP --etcd-endpoint $ETCD --discovery-backend etcd --request-transport http --kv-event-transport zmq --attention-backend dsv4 --disable-radix-cache --page-size 256 --cuda-graph-max-bs 512 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion --context-length 9472 --dp 8 --enable-dp-attention --ep-size 8 --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 --chunked-prefill-size 163840 --max-prefill-tokens 163840 --mem-fraction-static 0.85 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998
```

**Decode leg (decode node)** — same recipe, `--disaggregation-mode decode`, mem-fraction `0.90`, no bootstrap port, plus `--no-enable-kv-events` (see the note below):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m infera.engine.sglang --model-path $MODEL --tp-size 8 --trust-remote-code --host 0.0.0.0 --port 30000 --advertise-host $D_IP --etcd-endpoint $ETCD --discovery-backend etcd --request-transport http --kv-event-transport zmq --no-enable-kv-events --attention-backend dsv4 --disable-radix-cache --page-size 256 --cuda-graph-max-bs 512 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion --context-length 9472 --dp 8 --enable-dp-attention --ep-size 8 --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 --chunked-prefill-size 163840 --max-prefill-tokens 163840 --mem-fraction-static 0.90 --disaggregation-mode decode --disaggregation-transfer-backend mooncake
```

```{admonition} Why --no-enable-kv-events on the decode leg (SWA models)
:class: important
With KV events on, Infera auto-appends `--disaggregation-decode-enable-radix-cache`
to a mooncake **decode** worker (so the router can steer prefix repeats). But
DeepSeek-V4-Pro is a **sliding-window-attention (SWA)** model, and SGLang rejects
that flag on SWA — the decode scheduler dies at startup with
`--disaggregation-decode-enable-radix-cache is incompatible with sliding window
attention (SWA) models`. `--no-enable-kv-events` skips the auto-append. PD here
uses round-robin, not KV-aware routing, so nothing is lost. Non-SWA models can
leave KV events on.
```

```{admonition} What Infera fills in for you
:class: note
Infera auto-sets the RDMA env on ROCm (set-if-unset): `MC_GID_INDEX=1`,
`MC_DISABLE_HIP_TRANSPORT=1`, and pins the KV-transfer host IP to the RDMA rail —
so you don't hand-set them (`infera/engine/rocm_rdma_env.py`). `--advertise-host`
must be each node's **data-plane (RDMA-rail) IP** so the peer can reach it.
`--kv-cache-dtype fp8_e4m3` is the default. For cold cross-node bootstrap, bump
`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800`.
```

## 5. Verify (curl through the router)

```bash
curl -sf http://$P_IP:8000/health && echo OK
curl -s http://$P_IP:8000/v1/workers | python3 -m json.tool          # expect 2: one prefill, one decode
curl -s http://$P_IP:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=?\"}],\"max_tokens\":32,\"temperature\":0}"
```

Two workers with `disagg_mode` `prefill` and `decode`, plus a coherent
completion, means the router paired them and the KV hand-off works.

## 6. Sweep (InferenceX-aligned, through the router)

Point `sglang.bench_serving` at the **router** on :8000 with `sglang-oai`. Same
aligned knobs as the aggregated example:

```bash
for C in 64 128 256 384 512; do
  python3 -m sglang.bench_serving --backend sglang-oai --base-url http://$P_IP:8000 \
    --model $MODEL --tokenizer $MODEL \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
    --max-concurrency $C --num-prompts $((C*10)) --warmup-requests $((C*2)) \
    --request-rate inf --output-file pd_c${C}.jsonl 2>&1 | tail -20
done
```

Read `total_token_throughput` per jsonl for each concurrency point.

## Verify the RDMA fabric (optional)

The `ibv_devinfo` check in §2 confirms the container *sees* the NICs; it doesn't
confirm the two nodes can actually move KV over RDMA. Infera ships a preflight
suite (`infera-preflight`) that measures cross-node RoCE bandwidth and Mooncake
KV-transfer over the fabric — run it before a real bring-up to catch a
degraded/TCP-fallback path early. Run it **inside the container** on both nodes
against a shared output dir:

```bash
# both nodes, shared --dump-path; SLURM srun coordinates the two tasks
srun --nodelist=<prefill-host>,<decode-host> -N2 --ntasks-per-node=1 \
  python -m infera.tools.preflight --dump-path <shared-dir> --netperf --mooncake
```

It writes a per-host JSON + a combined HTML report; healthy same-rail RoCE is
~325–345 Gb/s, and Mooncake should show an `rdma` (not just `tcp`) number. See
`infera/tools/preflight/README.md` for the full check list.

---

## Notes & gotchas

1. **Infera router auto-pairs; don't use `sglang_router`/mini-lb.** `infera.server`
   watches etcd and shapes each request to the prefill+decode pair. That's the
   "use Infera's own router" path — one server, no static `--prefill/--decode`
   list.
2. **Advertise the data-plane IP**, not the public NIC. `--advertise-host $P_IP/$D_IP`
   must be the RDMA-rail IP the peer can reach; Infera also pins the KV host IP to
   the rail, but the etcd-advertised URL comes from `--advertise-host`.
3. **mem-fraction is role-asymmetric**: prefill `0.85` (DP-attn prefill OOMs at
   `0.90` under high conc), decode `0.90`. See
   [DP prefill OOM](../reference/troubleshooting.md).
4. **Mooncake vs MoRI**: this image's MoRI backend regressed; use
   `--disaggregation-transfer-backend mooncake`. The two are otherwise
   throughput-equivalent for DSv4 (small MLA KV). Both legs must match.
5. **Cold start ~30 min** (weights + cuda-graph capture, same as the aggregated
   example) — don't kill a slow launch; shrink `--cuda-graph-max-bs` to cut it.
6. **Switching topology/backend**: kill both legs (`pkill -9 -f
   infera.engine.sglang`), wait for VRAM≈0 on both nodes, then relaunch — else the
   next run OOMs.

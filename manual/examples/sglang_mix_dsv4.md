# Example: SGLang mixed (aggregated) — DeepSeek-V4-Pro on MI355X

A concrete, copy-paste walk-through of an **aggregated** ("mixed") SGLang worker
— prefill and decode colocated on the same 8 GPUs — served through the standard
Infera stack (etcd + `infera.server` router) and throughput-swept end to end. It
runs the tuned recipe for DeepSeek-V4-Pro (`fp4`: FP8 attention + MXFP4 MoE) on a
single 8×MI355X node.

For the disaggregated (1P1D) counterpart see
[SGLang 1P1D — DeepSeek-V4-Pro](sglang_1p1d_dsv4.md). For the concepts, start
from [Engines](../components/engines.md) and the
[Quickstart](../getting_started/quickstart.md).

```{admonition} What "mixed / aggregated" means here
:class: tip
One worker does **both** phases (no KV transfer, no RDMA). This is the simplest,
usually-fastest topology for a balanced input:output ratio — reach for
[PD disaggregation](../features/pd_disaggregation.md) only when one phase
dominates. Everything below is single-node; no ionic/RDMA setup is required.
```

## The tuned recipe in one line

Two levers make DSv4 hit its target throughput: `--attention-backend dsv4`
**and** the full fused-compress env set. Miss either and throughput drops to
~50%. Pick the variant by concurrency:

| conc | variant | parallel flags |
|---|---|---|
| **≤ 128** | `nodp` (pure TP8) | `--chunked-prefill-size 8192` |
| **≥ 256** | `dp` (DP-attention) | `--dp 8 --enable-dp-attention --ep-size 8 --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 --chunked-prefill-size 65536` + env `SGLANG_DP_USE_GATHERV=1` |

## 0. Shared variables

```bash
HF_CACHE=/path/to/hf-cache                     # TODO: your HuggingFace cache / model dir (host path)
MODEL=$HF_CACHE/DeepSeek-V4-Pro-fixed          # fp4 DSv4-Pro checkpoint
HOST=127.0.0.1                                  # single node
ETCD=$HOST:2379
IMG=inferaimage/infera:infera-sglang-...        # TODO: update to the image you validated
```

## 1. etcd — shared registry (on the host)

Start this first; the router and worker both register here.

```bash
docker run -d --name infera-etcd --network host quay.io/coreos/etcd:v3.5.14 \
  etcd --advertise-client-urls http://$HOST:2379 --listen-client-urls http://0.0.0.0:2379
```

## 2. Worker container (on the host)

One persistent container, all 8 GPUs, host net (router :8000 reachable), with the
HF cache bind-mounted in:

```bash
docker run -d --name sgl_mix --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $HF_CACHE:$HF_CACHE --entrypoint "" "$IMG" sleep infinity
```

Everything from here on runs **inside `sgl_mix`** — either `docker exec -it
sgl_mix bash` and paste, or prefix each command with `docker exec sgl_mix`.
Re-export `HF_CACHE/MODEL/HOST/ETCD` inside the shell. If the checkpoint or
tokenizer is gated, also export your token first:

```bash
export HF_TOKEN=hf_...                          # TODO: set if the model/tokenizer is gated
```

## 3. Infera router — `infera.server` (inside `sgl_mix`, :8000)

```bash
python -m infera.server --host 0.0.0.0 --port 8000 --etcd-endpoint $ETCD \
  --router-tokenizer-path $MODEL --router-policy round-robin \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```

## 4. SGLang worker — tuned recipe, one command (inside `sgl_mix`)

First export the fused-compress lever (paste the whole block as-is):

```bash
export SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_AITER_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=1 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false SGLANG_ROCM_USE_MULTI_STREAM=false SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true SGLANG_EAGER_INPUT_NO_COPY=true SGLANG_USE_ROCM700A=0 SGLANG_OPT_USE_JIT_INDEXER_METADATA=false SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false SGLANG_OPT_USE_TILELANG_MHC_POST=false
```

**`nodp` (concurrency ≤ 128)** — pure TP8:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m infera.engine.sglang --model-path $MODEL --tp-size 8 --trust-remote-code --host 0.0.0.0 --port 30000 --advertise-host $HOST --etcd-endpoint $ETCD --discovery-backend etcd --request-transport http --kv-event-transport zmq --attention-backend dsv4 --disable-radix-cache --page-size 256 --cuda-graph-max-bs 128 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion --mem-fraction-static 0.90 --chunked-prefill-size 8192
```

**`dp` (concurrency ≥ 256)** — add `SGLANG_DP_USE_GATHERV=1` and the DP-attention block:

```bash
SGLANG_DP_USE_GATHERV=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m infera.engine.sglang --model-path $MODEL --tp-size 8 --trust-remote-code --host 0.0.0.0 --port 30000 --advertise-host $HOST --etcd-endpoint $ETCD --discovery-backend etcd --request-transport http --kv-event-transport zmq --attention-backend dsv4 --disable-radix-cache --page-size 256 --cuda-graph-max-bs 128 --swa-full-tokens-ratio 0.15 --disable-shared-experts-fusion --mem-fraction-static 0.90 --dp 8 --enable-dp-attention --ep-size 8 --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000 --chunked-prefill-size 65536 --max-running-requests 256
```

```{admonition} What Infera fills in for you
:class: note
`--kv-cache-dtype fp8_e4m3` is Infera's default (halves the KV footprint) — you
don't pass it. The three dev flags (`--discovery-backend etcd --request-transport
http --kv-event-transport zmq`) must match the server; drop them only on the
production NATS + Kubernetes plane.
```

Cold start is slow (weights off shared storage + cuda-graph capture). It's ready
when the log prints `The server is fired up and ready to roll!`.

## 5. Verify (curl through the router)

```bash
curl -sf http://$HOST:8000/health && echo OK
curl -s http://$HOST:8000/v1/workers | python3 -m json.tool          # expect 1 worker
curl -s http://$HOST:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=?\"}],\"max_tokens\":32,\"temperature\":0}"
```

A coherent completion means the router found the worker and served it.

## 6. Sweep (InferenceX-aligned, through the router)

Point `sglang.bench_serving` at the **router** on :8000 with the OpenAI backend
(`sglang-oai`), not the raw engine port. Aligned knobs: random dataset,
`--random-range-ratio 1.0`, `num-prompts = 10×conc`, `warmup = 2×conc`,
`--request-rate inf`.

```bash
for C in 32 64 128; do            # 256 512 1024 → restart the worker in dp variant first
  python3 -m sglang.bench_serving --backend sglang-oai --base-url http://$HOST:8000 \
    --model $MODEL --tokenizer $MODEL \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
    --max-concurrency $C --num-prompts $((C*10)) --warmup-requests $((C*2)) \
    --request-rate inf --output-file mix_c${C}.jsonl 2>&1 | tail -20
done
```

Read `output_throughput` / `total_token_throughput` from each jsonl; divide by 8
for the per-GPU number that lines up with the InferenceX CSV.

---

## Notes & gotchas

1. **Two levers are non-negotiable.** `--attention-backend dsv4` *and* the full
   fused-compress env set. Either one missing → ~50% throughput at high conc.
2. **Variant by concurrency**, not a single config: `nodp` (pure TP8) is optimal
   ≤128; ≥256 needs the `dp` (DP-attention) variant or throughput falls off a
   cliff (c256 ~50% → >100% target). Restart the worker to switch variants.
3. **Sweep the router, not the engine.** `--backend sglang-oai --base-url
   http://host:8000`. Hitting the raw engine port (30000) with `--backend sglang`
   bypasses the Infera router and doesn't exercise the product path.
4. **`--kv-events off`** is a valid simplification for a single mix worker (no
   KV-aware routing to feed); it skips the worker-side KV plane. Left on here to
   match the documented dev path.
5. **Cold start is dominated by cuda-graph capture — be patient (~30 min).** After
   weights load, SGLang captures a cuda graph per batch size up to
   `--cuda-graph-max-bs`; on an 8×MI355X TP8 run this alone can take tens of
   minutes and looks like a hang. Don't kill it — confirm forward progress by
   watching the JIT/graph build dir grow (e.g. `watch du -sh
   ~/.cache/flashinfer /sgl-workspace/aiter/aiter/jit`). To cut the wait, lower or
   shrink the capture set: a smaller `--cuda-graph-max-bs` (e.g. 32) or an
   explicit `--cuda-graph-bs` list captures fewer graphs — at the cost of
   replaying graphs only up to that batch size. See SGLang's
   [hyperparameter tuning](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)
   and [server arguments](https://docs.sglang.io/advanced_features/server_arguments.html)
   (raising `--cuda-graph-max-bs` costs memory — drop `--mem-fraction-static` to match).

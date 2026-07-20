# Quickstart

The fastest path from nothing to a served model you can `curl`. Single host,
two GPUs, no Docker orchestration. About five minutes.

You'll start four things: **etcd**, **one server**, **two engine workers**, then
send a request.

## Choose your path

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🐍 pip — this page
`pip install amd-infera`, then run `python -m infera.*` by hand. The fastest
way to see it work on one host — no images, no orchestrator.
:::

:::{grid-item-card} 🐳 Container
:link: installation
:link-type: doc
Prebuilt engine images — a reproducible single-host stack.
:::

:::{grid-item-card} ☸️ Kubernetes
:link: ../serving/kubernetes
:link-type: doc
Multi-host, autoscale, and PD via the operator — the production path.
:::

::::

The rest of this page walks the **pip** path.

## 1. Start etcd

etcd is the shared registry the server and workers coordinate through. For dev,
one container is enough:

```bash
docker run -d --name infera-etcd --net host quay.io/coreos/etcd:v3.5.14 \
  etcd --advertise-client-urls http://127.0.0.1:2379 \
       --listen-client-urls http://0.0.0.0:2379
```

## 2. Start a server

The server holds the router, not the model. It needs a small tokenizer for the
router's prefix hashing (any small model works):

```bash
python -m infera.server --port 8000 --etcd-endpoint 127.0.0.1:2379 \
  --router-tokenizer-path Qwen/Qwen3-0.6B \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```

```{admonition} Why those last three flags?
:class: important
By default Infera runs the **production** plane — `--discovery-backend
kubernetes` (needs a k8s API + label selector) and `--request-transport nats`
(needs a NATS broker). This quickstart uses the **no-broker dev path** instead:
`etcd` discovery + `http` request transport + `zmq` KV events. Set the same
three flags on **every** server and worker, or they won't find each other. See
[Routing & transport](../features/routing_and_transport.md).
```

```{tip}
Run the server on as many ports/hosts as you like, all pointing at the same
`--etcd-endpoint`. They share one view of the fleet — put a load balancer in
front and you have HA for free.
```

## 3. Start two engine workers

Pick **one** engine. Each worker takes a GPU and self-registers into etcd. The
`--discovery-backend etcd --request-transport http --kv-event-transport zmq`
flags match the server (the no-broker dev path from step 2).

::::{tab-set}

:::{tab-item} SGLang
```bash
HIP_VISIBLE_DEVICES=0 python -m infera.engine.sglang \
  --model-path Qwen/Qwen3-0.6B --port 30000 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq

HIP_VISIBLE_DEVICES=1 python -m infera.engine.sglang \
  --model-path Qwen/Qwen3-0.6B --port 30001 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```
:::

:::{tab-item} vLLM
```bash
HIP_VISIBLE_DEVICES=0 python -m infera.engine.vllm \
  --model Qwen/Qwen3-0.6B --port 30000 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq

HIP_VISIBLE_DEVICES=1 python -m infera.engine.vllm \
  --model Qwen/Qwen3-0.6B --port 30001 --host 0.0.0.0 \
  --etcd-endpoint 127.0.0.1:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```
:::

:::{tab-item} ATOM
```bash
# ATOM registers via etcd only — it has no --discovery-backend/--request-transport/
# --kv-event-transport knobs (those are vLLM/SGLang-only). Just point it at etcd.
HIP_VISIBLE_DEVICES=0 python -m infera.engine.atom \
  --model Qwen/Qwen3-0.6B --server-port 30000 --host 0.0.0.0 \
  -tp 1 --etcd-endpoint 127.0.0.1:2379

HIP_VISIBLE_DEVICES=1 python -m infera.engine.atom \
  --model Qwen/Qwen3-0.6B --server-port 30001 --host 0.0.0.0 \
  -tp 1 --etcd-endpoint 127.0.0.1:2379
```
:::

::::

Give them a few seconds to load (each engine pulls + loads the model).

## 4. Verify

Check the server is up and both workers have registered:

```bash
curl -sf localhost:8000/health && echo OK
curl -s localhost:8000/v1/workers | python -m json.tool   # expect 2 workers
```

## 5. Send a request

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B",
       "messages":[{"role":"user","content":"1+1=?"}],
       "max_tokens":50}'
```

You should get an OpenAI-style completion (abbreviated):

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "1 + 1 = 2."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
}
```

The router picked one of the two workers for you — that's a working Infera stack.

```{admonition} Nothing back yet?
:class: tip
- **`503` or empty `/v1/workers`** — the engines are still loading the model; wait
  a few seconds and retry (watch a worker's log for "registered").
- **Connection refused** — the server takes a moment to bind `:8000`; retry.
- **Workers up but the request errors/hangs** — the server *and* every worker must
  share the same `--etcd-endpoint` and the three dev flags
  (`--discovery-backend etcd --request-transport http --kv-event-transport zmq`).
```

## Where to go next

- **Route by cache locality** instead of round-robin →
  [KV-aware routing](../features/kv_aware_routing.md)
- **Split prefill and decode** across GPUs →
  [PD disaggregation](../features/pd_disaggregation.md)
- **Keep KV warm** across restarts on RAM/NVMe →
  [KV-Cache Offload](../features/kv_cache_offload.md)
- **Deploy for real** with Kubernetes →
  [Deployment](../serving/deployment.md)

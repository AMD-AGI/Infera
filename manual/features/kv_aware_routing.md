# KV-aware routing

```{admonition} One-pager
:class: tip
**What:** route each request to the worker that already has its prompt prefix
cached. **Why:** skip recomputing the shared prefix → lower TTFT, higher
throughput. **Cost:** the server subscribes to each worker's KV-cache events.
```

```{graphviz}
digraph kv_aware {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.2,0.12"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10];

  REQ [label="Request\nprompt + shared prefix"];
  RTR [shape=diamond style=filled fillcolor="#fff3cd" color="#caa300"
       label="Router\nscore each worker:\ncost = overlap × miss-blocks + load"];
  W0  [label="Worker 0\nalready caches the prefix" fillcolor="#e2efdd" color="#6a9a4a"];
  W1  [label="Worker 1\ncold" fillcolor="#f4f4f4" color="#999999"];
  HIT [label="Prefix cache HIT\nlow TTFT" fillcolor="#e2efdd" color="#6a9a4a"];

  REQ -> RTR;
  W0 -> RTR [label="KV-cache events" style=dashed color="#8a8a8a"];
  W1 -> RTR [label="KV-cache events" style=dashed color="#8a8a8a"];
  RTR -> W0 [label="lowest cost =\nbest prefix overlap" penwidth=1.8 color="#5577cc"];
  W0 -> HIT;
}
```

## The idea

Many workloads share long prefixes — a fixed system prompt, a RAG document, a
growing chat history. If two requests with the same prefix land on *different*
workers, the second one recomputes a prefix the first already cached. KV-aware
routing sends them to the **same** worker so the prefix is a cache hit.

## How it picks

Each worker streams **KV-cache events** to the server, which mirrors that
worker's cached-block set. For an incoming request the router scores every
worker:

```text
cost(w) = overlap_weight × (request_blocks − cache_hits(w)) + active_blocks(w)
```

- `request_blocks − cache_hits(w)` — how many prompt blocks worker *w* would
  have to compute from scratch (lower = better cache locality).
- `active_blocks(w)` — distinct in-flight prompt blocks on *w* (shared prefixes
  counted once); a load term so a hot worker isn't overwhelmed.
- `overlap_weight` (`--kv-overlap-weight`, default `1.0`) — the dial between
  **cache locality** (raise it → lower TTFT) and **load balance** (lower it →
  steadier ITL under load). At `0` the cache term drops out entirely and routing
  is pure load balance.

The worker with the lowest cost wins.

```{admonition} Worked example
:class: note
A 10-block prompt with `--kv-overlap-weight 1.0`:

| Worker | cache hits | miss blocks (10 − hits) | active blocks | cost |
|---|---:|---:|---:|---:|
| **W0** — caches the prefix | 8 | 2 | 6 | `1.0 × 2 + 6` = **8** |
| **W1** — cold | 0 | 10 | 1 | `1.0 × 10 + 1` = **11** |

W0 wins: skipping 8 blocks of prefill beats being a little busier. Raise the
weight and W0 wins by more; set it to `0` and only `active_blocks` counts.
```

### Tuning prefill vs decode separately (PD)

Under [PD disaggregation](pd_disaggregation.md) the two sides want *opposite*
routing. Prefill benefits enormously from a prefix cache hit (it's the whole
cost of prefill), while decode should mostly balance load. So the router takes
**per-role** overlap weights that override the global `--kv-overlap-weight`:

| Flag | Typical | Effect |
|---|---|---|
| `--kv-prefill-overlap-weight` | `20.0` | strongly prefer the prefill worker that already caches the prefix |
| `--kv-decode-overlap-weight` | `2.0` | mild locality on decode; mostly route by load |

Leave them unset to fall back to the single `--kv-overlap-weight` for both.

## Turn it on

Two halves: events on the **workers**, the policy on the **server**. The
examples use the no-broker dev flags (`--discovery-backend etcd
--request-transport http --kv-event-transport zmq`); see the
[Quickstart](../getting_started/quickstart.md) for why, and drop them to use the
production NATS+k8s defaults.

::::{tab-set}

:::{tab-item} SGLang
```bash
python -m infera.engine.sglang \
  --model-path <model> --port 30000 --host 0.0.0.0 \
  --page-size 16 --enable-kv-events \
  --etcd-endpoint <etcd>:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```
:::

:::{tab-item} vLLM
```bash
python -m infera.engine.vllm \
  --model <model> --port 30000 --host 0.0.0.0 \
  --block-size 16 --enable-kv-events \
  --etcd-endpoint <etcd>:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```
:::

::::

```bash
# server
python -m infera.server --port 8000 --etcd-endpoint <etcd>:2379 \
  --router-policy kv-aware \
  --router-tokenizer-path Qwen/Qwen3-0.6B \
  --kv-overlap-weight 1.0 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```

```{important}
The block/page size must match across the fleet and the router (16 is the
canonical value). The router tokenizes the prompt and hashes it into blocks
with a chained XXH3-64 — the tokenizer it uses is `--router-tokenizer-path`.
```

## Verify it's working

```bash
# the router's mirrored view of one worker's cache
curl localhost:8000/v1/admin/cache-view/0.0.0.0:30000
# → {"worker_id":"0.0.0.0:30000","block_count":47}
```

Send two requests with the same long prefix and watch the second one's TTFT drop
and `block_count` stay flat (it reused, didn't re-add).

## When to use it

- **Yes:** shared system prompts, RAG over repeated documents, multi-turn chat,
  agentic loops with a growing prefix.
- **Less so:** every request is unique and short — there's no prefix to hit, so
  plain round-robin balances load just as well at lower overhead.

## Environment variables

KV-aware routing is configured by **flags** (`--router-policy`,
`--kv-overlap-weight`, `--kv-event-transport`), not env vars. The one env var that
affects whether the prefix cache it relies on persists:

| Env | Value | Why it matters here |
|---|---|---|
| `PYTHONHASHSEED` | `0` | On vLLM, required for stable block hashes — without it cross-restart prefix hits are zero, defeating the routing win. |

Full list on the [environment variables](../reference/environment.md) page.

## Related

- [Routing & transport](routing_and_transport.md) — all policies + the event
  transport.
- [KV-Cache Offload](kv_cache_offload.md) — keep prefixes warm even *below* the
  worker's GPU cache.

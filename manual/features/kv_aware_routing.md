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

## Multimodal (image) affinity

Vision/audio requests can't use the block-hash cache locality above: the router
tokenizes **text only**, and the image-placeholder token id is identical no
matter which image was sent — so two requests with the same surrounding text but
different images hash to the same blocks. Trusting that would serve one image's
KV for another (silent corruption).

So when a request carries image content the kv-aware policy drops the text
overlap term and routes by **image affinity** instead. It derives a stable key
per image — `XXH3` of the image *reference*: the `http(s)` URL, or the whole
`data:` URI whose base64 payload content-hashes the bytes — and keeps a small
per-worker LRU of the images each worker recently served. A repeat image is
drawn back to the worker already holding its warm vision/prefix cache; an unseen
image has no affinity and falls back to load balance.

This is **engine-agnostic** — the key indexes the router's *own* image→worker
map, never the engine's KV hashes — so SGLang, vLLM and ATOM share one code
path. Under [PD disaggregation](pd_disaggregation.md) it applies in the
**prefill** pool (the image is encoded during prefill); decode routes by load.

**No extra configuration.** It's part of `--router-policy kv-aware` and kicks in
automatically whenever a request body carries `image_url` / `image` /
`input_audio` blocks (OpenAI or Anthropic shape) or a top-level `images` list.
The pick log shows it:

```text
pick policy=kv-aware role=prefill picked=<worker> mm_images=1 mm_affinity_hits=1
```

`mm_images` is how many images the request carried; `mm_affinity_hits` is how
many of them the chosen worker already held — `0` on the first request for a
given image, `≥1` once that worker is warm. A runnable PD demo lives in
[`examples/vlm_pd_mm_routing`](https://github.com/AMD-AGI/Infera/tree/main/examples/vlm_pd_mm_routing).

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

## Optional Decode input-demand routing (Rust)

The Rust router can select Decode workers/ranks by the input-token demand of
in-flight requests. It defaults to `off` and requires `--router-policy kv-aware`.

```bash
# Add to the existing entrypoint configuration.
RUST_LOG=info,infera_router::decode_demand=debug \
python -m infera.server --router-backend rust --router-policy kv-aware \
  --router-tokenizer-path /path/to/model --decode-input-demand shadow
```

Keep the deployment's discovery, transport and tokenizer/template options.
Without the flag the Rust router reads `INFERA_DECODE_INPUT_DEMAND`; the Python
backend ignores that variable.

| Mode | Selection and overhead |
|---|---|
| `off` | Original routing; no input-token counting or reservations |
| `shadow` | Original selection plus demand bookkeeping, a suggested target and decision metrics |
| `on` | Select the lowest projected input demand |

For a candidate target, the projected demand is its in-flight input tokens plus
this request's input tokens times `n` (an invalid `n` counts as 1). Equal costs
use original load as the tie-break, then candidate order. The chosen target keeps
normal block/cache bookkeeping. Prefill and aggregated/Mixed selection are
unchanged. Decode input length needs neither KV block metadata nor Decode radix.

A request is rendered and tokenized at most once per worker template variant:
its Prefill and Decode picks share the result. The count is exact unless the
request is multimodal or the worker is known to render differently (its
render-parity probe diverged); then the router's own count is an estimate. A
worker whose render has not been, or cannot be, checked counts as exact. When the
router cannot tokenize a request at all, it estimates string bytes / 4, skipping
inline binary payloads (`data:` URLs and `data` fields such as base64 images or
audio). Estimates take part in selection like exact counts. Debug events
(`infera_router::decode_demand`) show the candidates' projected tokens, the
original choice, the suggestion and the actual choice, without prompt or session
text.

Selection and reservation share one short mutex critical section. The reservation
moves from the pick into the PD response guard and is released exactly once, on
response EOF, cancellation, failure or an abandoned pick; release never panics.
With `--pd-prefill-guard-release completion` it stays on the Decode guard
after Prefill's load is released. Live reservations survive a worker leaving
discovery, so late completions cannot subtract from a newly recreated entry.
State is local to one router process.

With `--session-affinity both`, a request whose session is already bound keeps
its Decode rank and is still booked; demand only chooses a session's first target
or its replacement. Use `--session-affinity prefill` to balance every Decode pick
on demand.

`/metrics` exposes `infera_router_decode_demand_tokens{target}`, the in-flight
input tokens per Decode route key, and
`infera_router_decode_demand_decisions_total{mode,input,agrees}`, where `input`
is `exact` or `estimated` and `agrees` says whether the suggestion matched the
original policy.

Only input tokens are counted. In a GLM-5.2 AgentX workload a request averaged
about 112k input and 930 output tokens, so input dominates the KV a Decode rank
holds and the context each decode step reads. Generated-token growth, compute and
capacity differences are not modeled; for long-output workloads, track generated
tokens as they arrive rather than reserving `max_tokens` up front. This is not
physical KV allocation either: shared prefixes count once per request, backend
reclamation lag is ignored, and there is no hard admission limit. Start with
homogeneous Decode targets; Radix/HiCache reuse can make the estimate
conservative.

In a P8D4 GLM-5.2 AgentX run with Prefill-only session affinity, an earlier
implementation of this policy evened KV occupancy across the Decode ranks and cut
the mean Decode allocation wait from 10.5 s to 3.3 s. That run also moved to
different nodes, so its throughput change is not attributable to demand routing
alone.

### Request-count weight

Under DP attention every decode step waits for the busiest rank, and that rank's
step time grows with its request count as well as its context. Token-only demand
evens the context but can leave one rank with many short requests.
`--decode-request-weight W` (or `INFERA_DECODE_REQUEST_WEIGHT`; requires
`--decode-input-demand shadow|on`) changes a candidate's cost to

    in-flight input tokens + this request's tokens + W × requests the target holds

where held requests are the reservations above (booked at the pick, released at
response EOF, cancellation or failure). `W = 0`, the default, is token-only
demand. Dynamo's `router-decode-active-request-weight` scores Decode workers the
same way (`decode blocks + weight × active requests`).

Choose `W` as the context, in tokens, that costs one decode step as much as one
more request on the busiest rank. A least-squares fit of step time against the
busiest rank's requests and tokens (3 s bins of the Decode log) gave about 1.05M
for GLM-5.2 1P1D C88 on SGLang v0.5.22 and 0.17-0.19M for 2P1D C176 on v0.5.21.
On that 1P1D C88 deployment, `W = 1000000` raised p90 interactivity from 55.7 to
79.6 at the same throughput and raised TTFT (mean +15%), because context spreads
less evenly; see `examples/sglang_2p1d_glm5.2/README.md`.

`/metrics` adds `infera_router_decode_demand_requests{target}`, the held
requests per Decode route key.

### Future: one tokenization end to end

The router and each engine still tokenize the same prompt separately. Sending the
router's token ids to the engines would remove that work and match block hashes to
the engine's KV exactly, but the router's render would become the model's input:
measure the gain first (engine tokenization's share of TTFT, render-parity
coverage of real traffic) and limit it to workers with confirmed render parity.

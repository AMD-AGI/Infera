# Overview

This page is the conceptual tour — the mental model, no commands. If you'd rather
get something running first, jump to the [Quickstart](quickstart.md) and come back.

## What problem does Infera solve?

A single model server (`vllm serve`, `sglang launch`) is great until you need
**more than one GPU's worth of serving**. Then come the questions one server
can't answer on its own — each maps to a feature:

- I have 8 GPUs. How do requests spread across them *without* throwing away the
  prompt cache each time? → **[KV-aware routing](../features/kv_aware_routing.md)**
- My prompts share a long prefix (a system prompt, a document, a conversation).
  Can each request go to the GPU that already holds that prefix? →
  **[KV-aware routing](../features/kv_aware_routing.md)**
- Prefill (reading the prompt) and decode (generating tokens) want very different
  hardware. Can I run them on separate GPUs and move the KV between them? →
  **[PD disaggregation](../features/pd_disaggregation.md)**
- The KV cache only lives in GPU memory. Can I keep it warm in RAM / on disk so a
  restart — or the next user — doesn't pay to recompute it? →
  **[KV-Cache Offload](../features/kv_cache_offload.md)**

Infera is the thin layer that answers these. It does **not** replace vLLM /
SGLang / ATOM — it sits in front of them and orchestrates a fleet.

## The three moving parts

```{graphviz}
digraph infera_arch {
  rankdir=TB; bgcolor="transparent"; compound=true;
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.22,0.13"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10 color="#5577cc"];

  client [label=<client<br/><font color="#8a8a8a">OpenAI API</font>> fillcolor="#f4f4f4" color="#999999"];
  server [label=<<b>infera.server</b> · FastAPI :8000<br/><font color="#5b6f9e">OpenAI endpoints + router · run N replicas</font>>];
  etcd   [shape=cylinder fillcolor="#fff3cd" color="#caa300" label=<<b>etcd</b><br/><font color="#998400">shared worker registry</font>>];

  subgraph cluster_workers {
    label=<<font color="#5e8048">engine workers — one model each, self-registered</font>>;
    labeljust="l"; style="rounded"; color="#cfe0cf"; fontname="Helvetica,Arial,sans-serif"; fontsize=10;
    w0 [label=<vLLM / SGLang / ATOM<br/><font color="#5e8048">GPU 0</font>> fillcolor="#e2efdd" color="#6a9a4a"];
    w1 [label=<vLLM / SGLang / ATOM<br/><font color="#5e8048">GPU 1</font>> fillcolor="#e2efdd" color="#6a9a4a"];
    wn [label=<…<br/><font color="#5e8048">GPU N</font>> fillcolor="#e2efdd" color="#6a9a4a"];
  }

  kvd [fillcolor="#dbe7ff" color="#5577cc" label=<<b>kvd</b> · tiered KV-cache daemon <font color="#8a8a8a">(per host, optional)</font><br/><font color="#5b6f9e">L2 RAM → L3 NVMe → L4 network · survives restart · shared across engines</font>>];

  client -> server [label="request"];
  server -> etcd  [label="watch" style=dashed];
  w0 -> etcd [label="register + heartbeat (lease)" style=dashed];
  w1 -> etcd [style=dashed];
  wn -> etcd [style=dashed];
  server -> w0 [label="route" color="#6a9a4a" constraint=false];
  w1 -> kvd [ltail=cluster_workers label="KV spill / reuse (CopyFree UDS)" color="#5577cc"];
}
```

**1. The server (`infera.server`)** — a FastAPI process exposing the
OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`) on port 8000. It
holds no model — it holds the **router**. Run as many replicas as you like behind
a load balancer; they share one view of the fleet.

**2. The workers (`infera.engine.{vllm,sglang,atom}`)** — each is one model
engine on one (or a few) GPUs. A worker **registers itself** into etcd at startup
and heartbeats; when it dies, its lease expires and it drops out. You never hand
the server a static worker list — the fleet is discovered.

**3. etcd** — a small key-value store: the **shared source of truth** for "which
workers exist, and what each can do." The server watches it; workers write to it.
That's the whole coordination story.

**(Optional) kvd — KV-Cache Management.** A per-host daemon that gives the workers
a **tiered KV cache**: blocks spill from GPU HBM → host RAM → NVMe → network and
stay warm across engine restarts, shared by every engine on the host (read back
zero-copy via a shared-memory arena). Turn it on when you want prefixes to survive
restarts or to reuse them across workers. See
[KV-Cache Management](../components/kvd.md).

```{admonition} The one-sentence version
:class: tip
**Workers announce themselves to etcd; the server watches etcd and routes each
request to the best worker.**
```

## How a request flows

```{graphviz}
digraph req_flow {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.2,0.12"];
  edge [color="#5577cc" penwidth=1.3 fontname="Helvetica,Arial,sans-serif" fontsize=10];

  c [label="client"];
  r [shape=diamond style=filled fillcolor="#fff3cd" color="#caa300" label="router\npicks worker(s)"];
  w [label=<engine worker(s)<br/><font color="#5e8048">prefill + decode</font>> fillcolor="#e2efdd" color="#6a9a4a"];

  c -> r [label="POST /v1/chat/completions"];
  r -> w [label="route by policy\n(round-robin · KV-aware · PD)"];
  w -> c [label="stream tokens"];
}
```

1. A client sends `POST /v1/chat/completions` to any server replica.
2. The **router** reads the live worker list (from etcd) and the request, then
   picks a worker — round-robin, KV-aware (prefix locality), or PD. See
   [Routing & transport](../features/routing_and_transport.md) for the policies.
3. On a **prefill/decode disaggregated** path, the router dispatches to a prefill
   *and* a decode worker and coordinates the KV hand-off over the KV transport
   (Mooncake / MoRI).
4. The worker streams tokens back; the server relays them to the client.

The client never knows how many GPUs are involved — it's just the OpenAI API.

## Where it runs

Infera is **ROCm-native**, built and validated on AMD Instinct **MI355X** GPUs:

| Capability | MI355X |
|---|---|
| Quantization | fp8 **and** mxfp4 |
| RDMA NIC | AMD AINIC |

The RDMA NIC carries the KV transfer in
[PD disaggregation](../features/pd_disaggregation.md); mxfp4 widens which
quantized checkpoints you can load.

## What to read next

- Get it running → [Quickstart](quickstart.md)
- Pick an engine → [Engines](../components/engines.md)
- Deploy for real → [Deployment](../serving/deployment.md)
- The KV cache everyone keeps mentioning → [KV-Cache Offload](../features/kv_cache_offload.md)


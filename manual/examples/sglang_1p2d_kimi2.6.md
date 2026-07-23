# Example: SGLang 1P2D (PD-disaggregated) — Kimi-K2.6-MXFP4 on MI355X

A **PD-disaggregated** SGLang deployment for `amd/Kimi-K2.6-MXFP4` on AMD MI355X:
one **prefill** worker feeding **two** **decode** workers (**1P2D**), fronted by
the Infera router, with the KV cache moved over **Mooncake RDMA**, then
throughput-swept end to end.

The runnable scripts for this example live in the repo under
[`examples/sglang_1p2d_kimi2.6/`](https://github.com/AMD-AGI/Infera/tree/main/examples/sglang_1p2d_kimi2.6).
This page is the short overview; the
[`README.md`](https://github.com/AMD-AGI/Infera/blob/main/examples/sglang_1p2d_kimi2.6/README.md)
in that directory is the authoritative, copy-paste step-by-step guide. Every
script is driven by environment variables and needs no editing.

The package covers two tests — run the single-node one first:

1. **SGLang single-node** — sanity-check that `Kimi-K2.6-MXFP4` starts and serves
   on one 4-GPU SGLang instance.
2. **Infera router + SGLang PD (1P2D)** — the real multi-node target.

For the concepts behind PD, read [PD disaggregation](../features/pd_disaggregation.md) first.

```{admonition} PD only pays off over real RDMA
:class: important
Cross-node PD moves the KV cache over the fabric on **every** request. It's a win
only on an RDMA fabric (ionic RoCEv2 here) — over TCP it is *slower* than a single
node. The container's ionic provider must match the host driver, or KV transfer
silently falls back to TCP. So run `preflight_rdma.sh` on every PD node **before**
the bring-up and confirm the visible port count equals the node's active ports (not
`0`) — see §3 of the
[`README.md`](https://github.com/AMD-AGI/Infera/blob/main/examples/sglang_1p2d_kimi2.6/README.md).
```

## Topology (1P2D)

`1P2D = 1 prefill + 2 decodes`, spread over three nodes (8× MI355X each):

| Node     | Role                                                           |
| -------- | -------------------------------------------------------------- |
| `node-0` | etcd + Infera router + prefill + verify + benchmark entrypoint |
| `node-1` | decode-0                                                       |
| `node-2` | decode-1                                                       |

## The tuned recipe

Both PD legs share the `aiter` attention backend, fp8 KV, DP-attention, and
Mooncake KV transfer. They differ mainly by GPU count and `--disaggregation-mode`:

| leg         | GPUs    | parallelism                               | mem-fraction | key flags                                                            |
| ----------- | ------- | ----------------------------------------- | ------------ | -------------------------------------------------------------------- |
| **prefill** | 4 (TP4) | `--dp-size 4 --enable-dp-attention`       | `0.85`       | `--disaggregation-mode prefill --disaggregation-bootstrap-port 8998` |
| **decode**  | 8 (TP8) | `--dp-size 8 --enable-dp-attention` (EP1) | `0.90`       | `--disaggregation-mode decode`                                       |

Shared by both: `--attention-backend aiter --kv-cache-dtype fp8_e4m3 --trust-remote-code --no-enable-kv-events --disaggregation-transfer-backend mooncake`, plus env `MC_GID_INDEX=1 SGLANG_USE_AITER=1`.

## Prerequisites

- **Hardware:** 3 nodes, 8× MI355X each (ROCm 7.2.0), Docker with GPU access.
  node-0 prefill uses 4 GPUs; each decode node uses all 8.
- **Model:** `amd/Kimi-K2.6-MXFP4` (~550B, license modified-mit, NOT gated).
- **Image:** `rocm/infera:sglang-v0.1.1`.

Download the model to a **local directory** (it is bind-mounted read-only) and
point `MODEL` at it. Benchmarking needs a local clone of InferenceX (`IX`). Full
details, including the model download and the "adapt to your cluster" env vars, are
in §1 *Prerequisites* / §2 *Adapt to your cluster* of the
[`README.md`](https://github.com/AMD-AGI/Infera/blob/main/examples/sglang_1p2d_kimi2.6/README.md).

## Scripts

All scripts are in
[`examples/sglang_1p2d_kimi2.6/`](https://github.com/AMD-AGI/Infera/tree/main/examples/sglang_1p2d_kimi2.6):

| Script                         | Purpose                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------- |
| `sglang_naive_engine.sh`       | Single-node SGLang server, default 4 GPUs / TP4.                                |
| `preflight_rdma.sh`            | RDMA preflight: container port visibility + (optional) cross-node fabric check. |
| `infera_0_etcd.sh`             | Start etcd (PD shared registry).                                                |
| `infera_1_server.sh`           | Start the Infera router/server (PD).                                            |
| `infera_2_sglang_prefill.sh`   | SGLang prefill leg, default 4 GPUs / TP4 / DP4.                                 |
| `infera_3_sglang_decode.sh`    | SGLang decode leg, default 8 GPUs / TP8 / DP8 / EP1.                            |
| `curl.sh`                      | Smoke-test the router (workers + chat).                                         |
| `bench_inferencex.sh`          | Benchmark via InferenceX.                                                       |

## Run

Export the shared vars (`MODEL`, `IX`, `DATA_NET`, `IMAGE`) on every node, then run in this order:

```text
single-node: engine -> verify -> benchmark
1P2D:        preflight -> etcd -> server -> prefill -> decode -> verify -> benchmark
```

**Single-node** (any 4-GPU MI355X):

```bash
IMAGE=rocm/infera:sglang-v0.1.1 \
MODEL=/your/path/Kimi-K2.6-MXFP4 bash sglang_naive_engine.sh
```

**1P2D** — first run the RDMA preflight on **each** PD node (the reported port count
must equal the node's active ports, not `0`):

```bash
bash preflight_rdma.sh
```

Then, on `node-0` (example data-net IP `10.0.0.1`):

```bash
bash infera_0_etcd.sh
ETCD_ENDPOINT=10.0.0.1:2379 bash infera_1_server.sh
ETCD_ENDPOINT=10.0.0.1:2379 bash infera_2_sglang_prefill.sh
```

On `node-1` and `node-2` (`ETCD_ENDPOINT` **must** point at node-0):

```bash
ETCD_ENDPOINT=10.0.0.1:2379 bash infera_3_sglang_decode.sh
```

Verify through the router (`bash curl.sh`) — expect **1 prefill + 2 decodes** plus
a coherent completion — then sweep concurrency with `bench_inferencex.sh` (we swept
32→2048 for 1P2D). See §4 *Run* in the
[`README.md`](https://github.com/AMD-AGI/Infera/blob/main/examples/sglang_1p2d_kimi2.6/README.md)
for the exact commands, verify/benchmark details, and how to stop the services.

## Notes & gotchas

1. **Infera router auto-pairs; no static worker list.** `infera.server` watches
   etcd and shapes each request to the 1P2D group — one server, `--router-policy
   round-robin`.
2. **Advertise the data-network IP.** `--advertise-host` (and `SGLANG_HOST_IP` /
   `HOST_IP`) must be each node's data-net IP the peers can reach, not the public NIC.
3. **RDMA passthrough on the PD legs.** `--device=/dev/infiniband`,
   `--cap-add=IPC_LOCK`, `MC_GID_INDEX=1`, and the host `libionic.so` bind-mount keep
   the container's RDMA provider aligned with the host driver — otherwise KV transfer
   silently degrades to TCP. Run the preflight checks first.
4. **Attention backend is `aiter`** for both single-node and PD on this image.

The full gotchas list (cold-start behaviour, `--no-enable-kv-events` rationale, the
decode-leg MoRI env) is in *Notes & gotchas* of the
[`README.md`](https://github.com/AMD-AGI/Infera/blob/main/examples/sglang_1p2d_kimi2.6/README.md).

# Example: PD-disaggregated 1P1D on Kubernetes — SGLang over Mooncake

A **prefill/decode-disaggregated** recipe: one router, one **prefill** worker on
node A and one **decode** worker on node B (**1P1D**, tp=1 each). KV is streamed
prefill → decode by the **Mooncake** transfer engine over the ionic **RoCE**
fabric, while Mooncake's RPC/handshake rides the pod-overlay (flannel) — so no
host networking is needed. Uses the **libionic-baked** engine image
(`rocm/infera:sglang-v0.1.2`), so RDMA works without any runtime host-lib
injection.

The manifest is
[`examples/k8s-deployments/pd-1p1d-mooncake.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/k8s-deployments/pd-1p1d-mooncake.yaml).
Read [Kubernetes deployment](../serving/kubernetes.md) for the general flow and
the single-node [Kimi-K3 recipe](k8s_kimi_k3.md) for the aggregated (mixed) case.

```{admonition} The hard prerequisite: a routable RoCE fabric between the two nodes
:class: important
PD only works if the **prefill and decode nodes can open an RDMA queue-pair over
ionic** — i.e. their RoCE NICs are on the **same rail/subnet or a routed fabric**.
Mooncake connects the RPC over the pod network, exchanges segment descriptors,
then does the KV move over RDMA; if the two nodes' ionic GIDs are on isolated
`/64`s with no route between them, the QP never comes up and every request fails
with `remote mooncake session … is not alive` / `Failed to get kvcache from
prefill`. Confirm the fabric first:

    # on each node — the group-4 field of the GID_Index-1 RoCE v2 GID:
    for d in 0 1; do cat /sys/class/infiniband/ionic_$d/ports/1/gids/1; done

If node A's `ionic_0` and node B's `ionic_0` share no routable subnet, pick a
pair of nodes that do (a multi-rail training fabric pairs rail *i* on every node).
```

## Topology

| Component | Node | GPUs | Runs |
|---|---|---|---|
| `qwen-pd-server` | A | 0 | `infera.server` — OpenAI endpoint + PD-aware router (`:8000`) |
| `qwen-pd-prefill` | A | 1 | `infera.engine.sglang --disaggregation-mode prefill …mooncake` |
| `qwen-pd-decode` | B | 1 | `infera.engine.sglang --disaggregation-mode decode …mooncake` |

The router auto-selects the disaggregated dispatcher once a model has **both**
prefill and decode workers registered (it dual-dispatches each request: prompt to
prefill, then decode pulls the KV).

## Prerequisites

- A **two-node** k3s cluster (control-plane + one agent), both GPU nodes on a
  routable RoCE fabric (see the admonition). Join the second node with its
  data-dir on a large disk:
  ```bash
  # on the agent node (K3S_URL = the server's IP):
  curl -sfL https://get.k3s.io | K3S_URL=https://<server-ip>:6443 \
    K3S_TOKEN=$(ssh <server> sudo cat /var/lib/rancher/k3s/server/node-token) \
    INSTALL_K3S_EXEC="--data-dir /mnt/<big-disk>/k3s" sh -
  kubectl get nodes          # both Ready
  ```
- AMD GPU device plugin (auto-schedules on both nodes) and the infera-operator —
  see [Kubernetes deployment](../serving/kubernetes.md).
- The **libionic-baked** engine image in **each** node's containerd:
  ```bash
  docker build -f deploy/docker/Dockerfile.sglang \
    --build-arg INSTALL_LIBIONIC=1 -t rocm/infera:sglang-v0.1.2 .
  docker save rocm/infera:sglang-v0.1.2 -o /tmp/img.tar   # on each node
  sudo k3s ctr images import /tmp/img.tar                 # (or build locally)
  ```
- The model at the **same path on both nodes** — a shared FS (`/mnt/vast/…`) is
  simplest; or the model-cache Job on a RWX PVC (see [Kimi-K3 recipe](k8s_kimi_k3.md)).

## Deploy & verify

```bash
# edit <PREFILL_NODE>/<DECODE_NODE>/<MODEL_DIR>/<MODEL_PATH> in the manifest
kubectl create namespace infera
kubectl apply -f examples/k8s-deployments/pd-1p1d-mooncake.yaml

kubectl -n infera get pods -o wide      # prefill on A, decode on B, both Running
kubectl -n infera port-forward svc/qwen-pd-server 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_PATH>","messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":40}'
```

A completion confirms the full path: router → prefill → **Mooncake KV over RoCE**
→ decode.

```{admonition} What is validated
:class: note
On a two-node k3s (MI355X) the operator reconcile, cross-node worker registration,
PD dual-dispatch, the baked-libionic RDMA device discovery (8 active ionic HCAs
in-pod, no injection), and the pod-overlay Mooncake RPC were all confirmed. The
final KV move additionally needs the two nodes on a **mutually routable RoCE
fabric** (above) — a pair whose ionic rails don't interconnect will reconcile and
dispatch but fail the transfer.
```

## Notes & gotchas

- **libionic ABI.** The stock sglang base ships libionic ABI 1; the ionic kernel
  needs ABI 4, or `ibv_get_device_list` returns 0 HCAs and Mooncake silently
  drops to TCP (KV transfer then fails). `Dockerfile.sglang` now bakes ABI-4
  libionic (`INSTALL_LIBIONIC=1`); the entrypoint still injects a host build if
  `/host-libionic` is mounted, as a belt-and-braces version match.
- **No hostNetwork.** Mooncake's RPC advertises `POD_IP`; on the pod-overlay that
  is a flannel address both nodes route to. hostNetwork would advertise the public
  host IP and, on a single node, also collide prefill/decode on `:30000`.
- **Slow-load probe.** `skipReadinessProbe: true` on the workers so the operator
  doesn't restart them during model load / graph capture.
- **mori alternative.** `--disaggregation-transfer-backend mori` is the other
  RoCE path; it needs an active RDMA device in-pod too (same libionic fix) and, on
  a single node, `MORI_DISABLE_AUTO_XGMI=0` for the intra-node XGMI fallback.

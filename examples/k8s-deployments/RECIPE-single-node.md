# Recipe: single-node aggregated (Qwen3-0.6B), vLLM & SGLang

A concrete, copy-paste recipe that brings up one `InferaDeployment` (router +
1 GPU worker) and serves an OpenAI-compatible endpoint — the infera analog of a
vendor "recipe". Two engine variants, **validated end to end on single-node k3s
(MI355X, `amd.com/gpu`)**:

| Variant | Manifest | Worker entrypoint |
|---|---|---|
| SGLang | `single-node-qwen-sglang.yaml` | `infera.engine.sglang` |
| vLLM | `single-node-qwen-vllm.yaml` | `infera.engine.vllm` |

## sglang vs vllm — what differs

The **router/server is engine-agnostic and identical** across both. Swapping
engines is *not* just the image; four coupled things change, all on the worker:

| | SGLang | vLLM |
|---|---|---|
| `spec.backendFramework` | `sglang` | `vllm` |
| container `image` | `rocm/infera:sglang-v0.1.1` | `inferaimage/infera:vllm-v0.1.0-rc5` |
| worker module | `infera.engine.sglang` | `infera.engine.vllm` |
| engine flags | `--model-path … --tp-size 1 --attention-backend aiter --mem-fraction-static 0.8` | `--model … --tensor-parallel-size 1 --gpu-memory-utilization 0.85` |

The two CLIs differ (`--model-path`/`--tp-size` vs `--model`/`--tensor-parallel-size`),
so the worker `command` must match the engine — you cannot reuse one manifest and
only change the image.

## Prerequisites

1. A Kubernetes cluster. Single node is fine — e.g. k3s **with its data-dir on a
   big disk** so image imports don't fill root:
   ```bash
   curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--write-kubeconfig-mode=644 --data-dir /mnt/<big-disk>/k3s" sh -
   export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
   ```
2. AMD GPU device plugin (exposes `amd.com/gpu`):
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml
   kubectl describe node <node> | grep amd.com/gpu   # -> amd.com/gpu: 8
   ```
3. The infera-operator (CRD + controller); no LWS/NATS needed for single-node aggregated:
   ```bash
   INSTALL_LWS=false INSTALL_NATS=false ../../deploy/scripts/deploy-k8s.sh
   ```
4. Engine image present in the **node container runtime**. k3s uses containerd
   (not docker), so import a local image into it (tar, not a stream):
   ```bash
   docker save rocm/infera:sglang-v0.1.1 -o /tmp/img.tar
   sudo k3s ctr images import /tmp/img.tar
   ```
5. Model on a host dir the pod hostPath-mounts at `/models`:
   ```bash
   python3 -c 'from huggingface_hub import snapshot_download as d; d("Qwen/Qwen3-0.6B", local_dir="/mnt/<big-disk>/models/qwen/Qwen3-0.6B")'
   ```

## Deploy & verify

```bash
kubectl create namespace infera
kubectl apply -f single-node-qwen-sglang.yaml     # or single-node-qwen-vllm.yaml

# operator reconciles the CR into server+worker Deployments, a ClusterIP :8000
# and the k8s-discovery ServiceAccount; wait for it to go ready:
kubectl -n infera get inferadeployment -w         # STATE -> ready (~2-3 min: model load + graphs)
kubectl -n infera get pods                        # server 1/1, worker 1/1

# call it (port-forward, or curl the ClusterIP from the node):
kubectl -n infera port-forward svc/qwen-agg-server 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/Qwen3-0.6B","messages":[{"role":"user","content":"What is Kubernetes?"}],"max_tokens":64}'
```

(For the vLLM variant the served model name is `Qwen3-0.6B` and the service is
`qwen-agg-vllm-server`.) A real completion in `choices[0].message.content`
confirms the operator → router → engine path end to end.

## Notes
- `hostPath` keeps the recipe self-contained (no CSI). For a real cluster use a
  PVC instead (see `pvc-models.yaml` + `single-node-aggregated.yaml`).
- Model mount requires `extraPodSpec` with an explicit `command`; the operator's
  simpler `args` mode auto-builds the entrypoint but only mounts dshm + `/boot`.
- k3s image imports land under `--data-dir`; putting that on a large disk avoids
  the DiskPressure eviction you get when 30–80 GB engine images fill root.

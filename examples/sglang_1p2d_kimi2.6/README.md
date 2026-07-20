# SGLang 1P2D — Kimi-K2.6-MXFP4 on MI355X (runnable scripts)

Runnable reproduction package for two tests on AMD MI355X:

1. **SGLang single-node** — sanity-check that `amd/Kimi-K2.6-MXFP4` starts and
   serves on one 4-GPU SGLang instance.
2. **Infera router + SGLang PD (1P2D)** — one **prefill** worker feeding **two**
   **decode** workers, fronted by the Infera router, with the KV cache moved over
   **Mooncake RDMA**, then throughput-swept end to end.

Every script is driven by environment variables — you should not need to edit any
script. The node roles (`node-0`, `node-1`, `node-2`) and IPs (`10.0.0.*`) in the
comments and commands are placeholders; substitute your own.

For the concise, prose walk-through and the concepts behind PD, see the manual
page [`manual/examples/sglang_1p2d_kimi2.6.md`](../../manual/examples/sglang_1p2d_kimi2.6.md)
and [PD disaggregation](../../manual/features/pd_disaggregation.md).

## Scripts

| Script                       | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| `sglang_naive_engine.sh`     | Single-node SGLang server, default 4 GPUs / TP4. |
| `preflight_rdma.sh`          | RDMA preflight: container port visibility + (optional) cross-node fabric check. |
| `infera_0_etcd.sh`           | Start etcd (PD shared registry).                 |
| `infera_1_server.sh`         | Start the Infera router/server (PD).             |
| `infera_2_sglang_prefill.sh` | SGLang prefill leg, default 4 GPUs / TP4 / DP4.  |
| `infera_3_sglang_decode.sh`  | SGLang decode leg, default 8 GPUs / TP8 / DP8 / EP1. |
| `curl.sh`                    | Smoke-test the router (workers + chat).          |
| `bench_inferencex.sh`        | Benchmark via InferenceX.                        |

## Topology (1P2D)

`1P2D = 1 prefill + 2 decodes`, spread over three nodes (8× MI355X each):

| Node     | Role                                                           |
| :------- | :------------------------------------------------------------- |
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

## 1. Prerequisites

### 1.1 Hardware / software

```text
Hardware: 3 nodes, 8x MI355X each (ROCm 7.2.0), Docker with GPU access
          node-0 prefill uses 4 GPUs; each decode node uses all 8
Model:    amd/Kimi-K2.6-MXFP4  (~550B, license modified-mit, NOT gated)
Image:    inferaimage/infera:<current-tag>   (infera-sglang — used for both tests)
```

Both the single-node and the PD test use the same **infera-sglang** image; its tag
is updated per test round (we will hand you the current tag). Set it once with
`export IMAGE=inferaimage/infera:<current-tag>`.

### 1.2 Download the model weights

The model is [amd/Kimi-K2.6-MXFP4](https://huggingface.co/amd/Kimi-K2.6-MXFP4)
(license `modified-mit`, ~550B params). It is **not gated**, so no access request is
needed, but it uses custom code (all scripts already pass `--trust-remote-code`).
Download it to a local path and point `MODEL` at that path — it must be a **local
directory** (the scripts bind-mount it read-only with `-v "$MODEL:$MODEL:ro"`), not
the HuggingFace repo id:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download amd/Kimi-K2.6-MXFP4 --local-dir /your/path/Kimi-K2.6-MXFP4
export MODEL=/your/path/Kimi-K2.6-MXFP4
```

### 1.3 Get InferenceX (needed only for benchmarking)

`bench_inferencex.sh` runs `utils/bench_serving/benchmark_serving.py`, which is
**not** baked into the image — it is mounted from the host. Clone the InferenceX
repo and point `IX` at it:

```bash
git clone https://github.com/SemiAnalysisAI/InferenceX.git /your/path/InferenceX
export IX=/your/path/InferenceX
```

## 2. Adapt to your cluster

Everything is controlled by env vars, so you only need to tell the scripts about
your network and paths.

- **Data-network prefix** — the scripts auto-detect a node's data-network IP by
  matching `DATA_NET` against `hostname -I`. The default is `10.0.0.`. Set your
  own prefix (note the trailing dot):

```bash
export DATA_NET=<your-data-net-prefix>.   # e.g. 192.168.10.
```

- **etcd endpoint** — etcd runs on `node-0`. On `node-0`, the etcd/server/prefill
  scripts auto-detect its IP from `DATA_NET` (if that picks the wrong NIC, override
  with `ETCD_HOST_IP=<node-0-data-ip> bash infera_0_etcd.sh`). On the decode nodes
  you **must** pass `node-0`'s data-network IP explicitly:

```bash
export ETCD_ENDPOINT=<node-0-data-ip>:2379
```

- **Model / InferenceX / image** — set `MODEL`, `IX`, `IMAGE` as shown above. The
  same infera-sglang `IMAGE` is used for both the single-node and the PD test.

## 3. Verify the RDMA fabric (run before the PD bring-up)

Cross-node PD moves the KV cache over the fabric on every request, so a healthy
RDMA path is a **prerequisite** — over TCP the pair is slower than single-node.
`preflight_rdma.sh` does two checks in one; run it before the PD bring-up.

**Device visibility (always).** It counts the RDMA ports the container can see on the
node; the number must equal the node's active ports (e.g. `8`), not `0`. A `0` means
the container's ionic provider doesn't match the host driver and RDMA would silently
fall back to TCP (the `libionic.so` bind-mount added by the image entrypoint keeps
them aligned). Run it on **each** PD node (needs `IMAGE`, see §1.1):

```bash
export IMAGE=inferaimage/infera:<current-tag>
bash preflight_rdma.sh
```

**Cross-node bandwidth + Mooncake KV transfer (when `DUMP_PATH` is set).** Device
visibility alone doesn't prove the nodes can move KV over RDMA. Set a shared
`DUMP_PATH` and launch **one task per node** — with SLURM:

```bash
export IMAGE=inferaimage/infera:<current-tag> DUMP_PATH=<shared-dir>
srun --nodelist=<node-0>,<node-1>,<node-2> -N3 --ntasks-per-node=1 bash preflight_rdma.sh
```

No SLURM? Run it on **each** node with the rank set by hand (a lone single-node run
skips the cross-node probes):

```bash
SLURM_PROCID=0 SLURM_NNODES=3 SLURMD_NODENAME=$(hostname) DUMP_PATH=<shared-dir> bash preflight_rdma.sh
```

A healthy same-rail RoCE link is ~40–43 GB/s (the preflight report's netperf matrix is
in GB/s), and Mooncake should report an `rdma` number (not just `tcp`). See
`infera/tools/preflight/README.md` for the full check list.

## 4. Run

Execution order:

```text
single-node: engine -> verify -> benchmark
1P2D:        etcd -> server -> prefill -> decode -> verify -> benchmark
```

### 4.1 SGLang single-node (sanity check)

On any 4-GPU MI355X, using the same **infera-sglang** image (`$IMAGE`):

```bash
IMAGE=inferaimage/infera:<current-tag> \
MODEL=/your/path/Kimi-K2.6-MXFP4 bash sglang_naive_engine.sh
```

Watch it come up, then `curl http://127.0.0.1:8000/v1/models`. Benchmark it via
§4.4 (`HOST=127.0.0.1`). If you ran this on `node-0`, remove it
(`docker rm -f sglang-kimi-engine`) before the PD bring-up — the Infera router also
binds `:8000`.

### 4.2 Infera + SGLang PD (1P2D)

On every node, export the following (set `IMAGE` to the infera-sglang tag):

```bash
export DATA_NET=x.x.x.   # your data-net prefix (note the trailing dot)
export IMAGE=inferaimage/infera:<current-tag>
export MODEL=/your/path/Kimi-K2.6-MXFP4
export IX=/your/path/InferenceX
```

On `node-0` (example data-net IP `10.0.0.1`):

```bash
bash infera_0_etcd.sh
ETCD_ENDPOINT=10.0.0.1:2379 bash infera_1_server.sh
ETCD_ENDPOINT=10.0.0.1:2379 bash infera_2_sglang_prefill.sh
```

On `node-1` and `node-2` (`ETCD_ENDPOINT` **must** point at node-0):

```bash
ETCD_ENDPOINT=10.0.0.1:2379 bash infera_3_sglang_decode.sh
```

Cold start is slow (weights + graph capture) — don't kill a slow launch.

### 4.3 Verify (through the router, on `node-0`)

```bash
bash curl.sh
```

Workers only:

```bash
curl -s 127.0.0.1:8000/v1/workers | python3 -m json.tool     # expect 1 prefill + 2 decodes
```

Three workers (`prefill` + two `decode`) plus a coherent completion means the
router paired them and the KV hand-off works.

### 4.4 Benchmark (InferenceX, through the router, on `node-0`)

`bench_inferencex.sh` defaults: `ISL=1024, OSL=1024, RATE=inf, RANGE=0.8,
NUM_PROMPTS=CONC*10`. Results land next to the script as `<TAG>.json` / `<TAG>.log`.

For **single-node**, we tested concurrencies 32, 64, 96, 128:

```bash
for C in 32 64 96 128; do
  MODEL=/your/path/Kimi-K2.6-MXFP4 IX=/your/path/InferenceX CONC=$C bash bench_inferencex.sh
done
```

For **1P2D**, we tested 32, 64, 96, 128, 256, 512, 1024, 2048 (same form, run on
`node-0`, varying `CONC`).

### 4.5 Stop services

```bash
# PD
docker rm -f infera-sgl-etcd infera-sgl-server infera-sgl-prefill infera-sgl-decode
# single-node
docker rm -f sglang-kimi-engine
```

`Ctrl-C` only stops `docker logs -f`, not the container — use `docker ps` to
confirm, and remove containers before relaunching or the next run OOMs.

## Notes & gotchas

1. **Infera router auto-pairs; no static worker list.** `infera.server` watches
   etcd and shapes each request to the 1P2D group — one server, `--router-policy
   round-robin`.
2. **Advertise the data-network IP.** `--advertise-host` (and `SGLANG_HOST_IP` /
   `HOST_IP`) must be each node's data-net IP the peers can reach, not the public NIC.
3. **RDMA passthrough on the PD legs.** `--device=/dev/infiniband`,
   `--cap-add=IPC_LOCK`, `MC_GID_INDEX=1`, and the host `libionic.so` bind-mount keep
   the container's RDMA provider aligned with the host driver — otherwise KV transfer
   silently degrades to TCP. Run the preflight checks in §3 first.
4. **Attention backend is `aiter`** for both single-node and PD on this image.
5. **`--no-enable-kv-events` on the PD legs.** PD here routes round-robin, not
   KV-aware, so nothing is lost. The decode leg's `MORI_IB_GID_INDEX` /
   `RCCL_MSCCL_ENABLE` only matter if you raise `--ep-size` (MoRI all-to-all); at EP1
   they're harmless no-ops.

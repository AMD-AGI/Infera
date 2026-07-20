# DeepSeek-V4-Pro — one-click reproduction kit (MI355X)

Self-contained scripts to reproduce DSv4-Pro throughput tests on 8×MI355X nodes,
launched **the infera way** (`infera.server` router + `infera.engine.*` workers +
etcd discovery). No local paths are baked in — you set everything via env vars.

## Matrix

| engine | mix (single-node) | pd_mooncake | pd_mori |
|--------|:---:|:---:|:---:|
| atom   | `engine/mix/atom`   | `engine/pd_mooncake/atom`   | — |
| sglang | `engine/mix/sglang` | `engine/pd_mooncake/sglang` | `engine/pd_mori/sglang` |
| vllm   | `engine/mix/vllm`   | `engine/pd_mooncake/vllm`   | — |

- **mix**: one worker, both phases, single node (no RDMA).
- **pd_\***: disaggregated prefill+decode over two nodes; KV moves over RDMA
  (mooncake or mori backend).

## 1. Set environment variables

Every script sources `common.sh`, which hard-fails (red error + exit) on any
missing required var. Set these first:

| var | required | default | meaning |
|-----|:---:|---------|---------|
| `INFERA_IMAGE` | yes | — | container image for the engine (per-engine tag, see below) |
| `INFERA_MODEL` | yes | — | DSv4-Pro checkpoint path (native `deepseek_v4`, fp8; see note below) |
| `INFERA_TOKENIZER` | no | `$INFERA_MODEL` | tokenizer path override — set only if the checkpoint's own tokenizer files are missing/broken |
| `INFERA_MODEL_MOUNT` | yes | — | shared-fs mount holding the checkpoint (bind-mounted into the container); `INFERA_MODEL` must live under it |
| `NODE_IP` | mix | `127.0.0.1` | this node's internal IP (advertise/etcd) |
| `PREFILL_IP` / `DECODE_IP` | pd | — | each node's **data-plane (RDMA-rail) IP** |
| `PREFILL_NODE` / `DECODE_NODE` | pd | — | ssh host names for the two nodes |
| `ETCD_PORT` | no | `2379` | etcd client port |
| `ROUTER_PORT` | no | `8000` | infera.server OpenAI HTTP port |
| `GID_INDEX` | no | `1` | RoCEv2 GID index (0 is link-local — keep 1) |
| `RDMA_NIC` | no | auto | data-plane NIC name (pd only; auto-detected if unset) |

### Checkpoint

All three engines serve the **same** native `deepseek_v4` fp8 checkpoint (FP8
attention + FP4 MoE). Point `INFERA_MODEL` at a directory whose `config.json` has
`model_type: deepseek_v4` and whose weight shards + `tokenizer.json` resolve (not
dangling symlinks). The model itself is **not** included — source it yourself.

If a checkpoint's tokenizer files are broken but its weights are fine, set
`INFERA_TOKENIZER` to a sibling checkpoint with an intact tokenizer.

## 2. Run a case

Each **mix** case has a `run.sh` (single entry) plus part scripts
(`engine.sh` / `smoke.sh` / `bench.sh` / `down.sh`). Example:

```bash
export INFERA_MODEL_MOUNT=<shared-fs-mount>   # e.g. the filer holding your checkpoints
export INFERA_IMAGE=<sglang-image> INFERA_MODEL=$INFERA_MODEL_MOUNT/<path-to>/DeepSeek-V4-Pro
bash engine/mix/sglang/run.sh          # bring up etcd + router + worker, then smoke
bash engine/mix/sglang/bench.sh 64     # optional: throughput sweep at conc 64
bash engine/mix/sglang/down.sh         # teardown
```

Each **pd** case has `up.sh` (2-node bring-up) / `down.sh` / `smoke.sh` /
`bench.sh`. Example:

```bash
export INFERA_IMAGE=<sglang-image> INFERA_MODEL=<path-to>/DeepSeek-V4-Pro
export PREFILL_NODE=<prefill-host> DECODE_NODE=<decode-host> PREFILL_IP=<ip> DECODE_IP=<ip>
bash engine/pd_mooncake/sglang/up.sh           # default TOPO=2p1d, num_request=640
bash engine/pd_mooncake/sglang/smoke.sh
bash engine/pd_mooncake/sglang/down.sh
```

### sglang PD topology option

Only the sglang PD cases take a `TOPO` option (mooncake & mori). Set `TOPO`:

| TOPO | layout | notes |
|------|--------|-------|
| `1p1d` | 1×prefill(TP8) + 1×decode(TP8) | chunk follows the experiment recipe |
| `2p1d` **(default)** | 2×prefill + 1×decode | chunk 163840, `num_request=640` — best perf |
| `2p2d` | 2×prefill(TP8) + 2×decode(TP4) | heterogeneous; two TP4 decodes co-located via `--base-gpu-id` |

```bash
TOPO=1p1d bash engine/pd_mooncake/sglang/up.sh
```

atom PD and vllm PD are 1p1d only (no `TOPO`).

## 3. Verify / bench

`up.sh`/`run.sh` end by printing worker list + a smoke completion through the
router. For throughput, run `bench.sh <conc...>` (random dataset,
num-prompts=10×conc, warmup=2×conc, sweep the router :8000, not the raw engine
port). Read `total_token_throughput` from the output jsonl; divide by GPU count
for the per-GPU number.

## Notes

- **Cold start ~30 min** (weights off shared storage + cuda-graph capture). Not a
  hang — `wait_*` helpers poll patiently and early-exit only on real errors.
- **PD needs real RDMA.** Over TCP it is slower; infera preflight rejects configs
  prone to silent TCP fallback. Bring the fabric up first.
- **Between runs**, `down.sh` reaps engines and waits for VRAM to drain — relaunch
  before that drains will OOM.

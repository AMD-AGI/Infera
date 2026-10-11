# Reproduction kit — TP4 / EP1 / DP-attention internal decode sweep

Goal: reproduce the eight TP4/EP1/DPA-on points (and optionally the six EP4 control points) from a
clean machine with cluster access.

**Estimated time:** ~16 min for the first point (cold AITER JIT + page cache; `load_pool_capture`
alone is ~14 min) and ~3–5 min per point afterwards. Full 8-point sweep ≈ 45 min; adding the 6
control points ≈ 75 min total. **Do not kill a point that looks silent — CUDA-graph capture is slow.**

---

## 0. Prerequisites

### Machine
- **One** node on the AMD `amd-spur` cluster with 8× MI355X; this run used `crsuse2-m2m-217`.
  Four GPUs (`0,1,2,3`) are used. Single node only — no cross-node traffic, so the RDMA fabric is
  irrelevant to the result.
- Obtained by **holding an existing allocation** (`squeue`), not by requesting a new one. This run
  used job `133750` (08:00Z start, 8 h limit). The scripts hard-refuse to run if
  `squeue -j $JOB_ID -h -o '%u %T %N'` does not equal `<you> RUNNING <NODE>`.
- The scripts also hard-refuse nodes `crsuse2-m2m-234`, `-036`, `-249` (peer-owned).
- The node must be exclusively yours. Verify with `docker ps` and `rocm-smi` before starting; a
  foreign container holding VRAM will invalidate the numbers (this happened — see `notes.md`).

### Secrets and access (values NOT included here — source them yourself)
- **Cluster access**: an account on the `amd-spur` cluster with `spur exec` rights to your own job.
  No SSH key or token is embedded anywhere in this packup.
- **Docker**: local daemon access on the compute node (group membership). No registry login is
  needed — the image is loaded from a local archive, not pulled.
- **Model / HF**: none. `HF_HUB_OFFLINE=1`; weights are read from a local read-only mount.
- No API keys, S3, etcd or registry credentials are required by any step below.

### External dependencies (absolute paths, not in this packup)
| What | Path | Notes |
|---|---|---|
| Model weights | `/shared_nfs/models/GLM-5.2-MXFP4` | NFS, mounted read-only |
| Pinned image archive | `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst` | 23,912,216,852 bytes, SHA256 `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2` |
| Wrapper repo | `infera.dev.yihou.sglang.bench.fast.script` | branch `dev.yihou.sglang.bench.fast.script`, HEAD `4b2b3bc39cc019a9c651c1bfdfd35eb6e63dcbae` |

### Repo state
The benchmark code that ran is in `evidence/code_snapshot/` (`profile_decode.py`, `topology.py`,
`batch_state.py`, `compare_server.py`, `check_compare_paths.py`) with
`code_hashes.sha256` — verify against those hashes before trusting a re-run.
`evidence/code_snapshot/code.diff` is the working-tree diff **at run time**; it touches
`compare_server.py` and files under `comparisons/`, which are **left over from a previous task and
not on this sweep's code path** (`profile_decode.py`, `topology.py` and `batch_state.py` are all
clean). No patch is required to reproduce.

---

## 1. Confirm the allocation and the node is clean

```bash
export JOB_ID=<your running job id>
export NODE=<crsuse2-m2m-NNN>
export CONTAINER=yihou-noep-dpa-<ddmm>      # must match ^yihou-[A-Za-z0-9_-]+$

squeue -j "$JOB_ID" -h -o '%u %T %N'        # must print: <you> RUNNING <NODE>
spur exec "$JOB_ID" bash -lc 'docker ps --format "{{.Names}} {{.Status}}"'
```

If a foreign container is holding GPUs, resolve ownership **before** proceeding. Do not silently
benchmark on a shared node.

## 2. Load the pinned image (skip if `docker image inspect` already resolves it)

```bash
spur exec "$JOB_ID" bash -lc '
  docker image inspect sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d >/dev/null 2>&1 ||
  zstd -dc /shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst | docker load'
```
Takes a few minutes and ~92 GB of image store (the node's NVMe, not `/`).

## 3. Create the container

```bash
export ROOT=<path to>/infera.dev.yihou.sglang.bench.fast.script/glm52_decode_internal_yihou_20260909_1057
bash scripts/create_container_yihou.sh
```
This pins the image by **digest**, mounts the repo and the read-only model, gives `/dev/kfd` +
`/dev/dri`, and sets the environment the result depends on (`HIP_VISIBLE_DEVICES=0,1,2,3`,
`AITER_JIT_DIR=/tmp/yihou-aiter-b9a83742f631`, `AITER_USE_FLYDSL_MOE_SORTING=1`,
`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`, `SGLANG_OPT_USE_TOPK_V2=false`, `HF_HUB_OFFLINE=1`).
It refuses to clobber an existing container.

## 4. Run the sweep (the eight EP1 points)

```bash
export OUTPUT_ROOT=$ROOT/sweeps/<your-sweep-dir>/iterations
bash scripts/run_tp4_noep_dpa_sweep_yihou.sh 4 8 16 20 24 32 40 48
```

Per point this calls `scripts/run_decode.sh`, which snapshots code + hashes + git HEAD + diff, runs

```
python3 $ROOT/bench/profile_decode.py \
  --model-path /shared_nfs/models/GLM-5.2-MXFP4 --result-dir <OUT> \
  --tp-size 4 --ep-size 1 --enable-dp-attention --batch-size <C> \
  --input-len 70000 --output-len 10000 --accept-length 3.61 \
  --warmup-steps 10 --enable-aiter-allreduce-fusion \
  --enable-fused-qk-norm-rope --mem-fraction-static 0.85
```

inside the container, and writes `launch_status.json` with the exit code and wall time.

**Run it detached and redirect to a file** (`setsid nohup … > driver.log 2>&1 &`). Do **not** pipe the
driver output through `tail`: `runtime.log` contains multi-megabyte single-line tqdm bars and doing so
previously triggered host memory pressure that killed the driver shell (see `notes.md`).

### Flags worth understanding
- `--batch-size` is the **GLOBAL** concurrency. With `--enable-dp-attention`, `dp_size = tp_size = 4`,
  so the per-rank batch is `C/4` and the CUDA-graph batch size is that local value. **C must be
  divisible by 4.**
- `--ep-size 1` is a plain TP MoE configuration; `server_cli` already forces `--moe-a2a-backend none`.
- `--accept-length 3.61` drives simulated acceptance (`match-expected`, `real-draft-token`); EAGLE is
  fixed at steps=5 / draft-tokens=6 / topk=1 with `kv-cache-dtype fp8_e4m3` and FlyDSL dsa backends.
- `--mem-fraction-static 0.85` sizes the KV pool **independently of C** (see `notes.md` for the
  headroom arithmetic — no adjustment was needed up to C=48).

## 5. (Optional) Run the EP4 control points

```bash
export OUTPUT_ROOT=$ROOT/sweeps/<your-sweep-dir>/control_iterations
bash scripts/run_ep4_control_yihou.sh 4 16 24 32 40 48
```
Identical parameters except `--ep-size 4`. Run these on the **same node and container** — that is the
whole point of the control.

## 6. Collect and verify

```bash
python3 scripts/collect_noep_sweep_yihou.py "$OUTPUT_ROOT" --output summary_yihou.csv
echo "exit=$?"     # must be 0
```

The collector is the gate, not a formatter: it asserts `complete`, exact `useful_output_tokens =
C × 10000`, `batch_size / input_len / output_len`, `(ep_size, tp_size, dp_size, enable_dp_attention)
== (1, 4, 4, True)` and `launch_status.exit_code == 0`, and exits non-zero otherwise. A row can come
back `missing`, `incomplete`, `configuration_mismatch`, `topology_mismatch`, `process_failed` or
`process_unconfirmed`.

## Expected output

`summary_yihou.csv` with eight `pass` rows and:

| C | TPOT ms | output tok/s |
|---:|---:|---:|
| 4 | 6.2456 | 640.45 |
| 8 | 7.8147 | 1023.71 |
| 16 | 9.8216 | 1629.06 |
| 20 | 10.7144 | 1866.65 |
| 24 | 11.3620 | 2112.31 |
| 32 | 12.7571 | 2508.40 |
| 40 | 14.3614 | 2785.25 |
| 48 | 15.4295 | 3110.92 |

Every point should also show `realized_accept_length = 3.6134393063583814` and
`verify_iterations = 2768` — these are deterministic given the parameters, so a mismatch means the
configuration is not the one documented here, regardless of how close the timing looks.

These are **single runs** with no repeats, so treat small differences (a few percent) as within
unmeasured run-to-run variation.

## If it doesn't reproduce
See `notes.md` — foreign containers holding VRAM, the cold-cache first point, the driver-shell kill,
the FlyDSL shape fallback, and what is and is not a real mismatch.

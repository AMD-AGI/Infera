# Reproduction kit — TP4 / DPA high-concurrency EP on vs EP off

Goal: reproduce the twelve measured points (C = 64 / 96 / 128, EP off and EP on, on two nodes) and,
if you want it, the C=160 infeasibility result.

**Estimated time.** First point on a cold node ~16 min (AITER JIT + page cache; `load_pool_capture`
alone is ~14 min), ~5-8 min per point afterwards. One arm of three points ~30 min; both arms on one
node ~55 min; both arms on both nodes in parallel ~1 h. Loading the image costs a few extra minutes
per node the first time.
**Do not kill a point that looks silent — CUDA-graph capture is slow.**

---

## 0. Prerequisites

### Machines
- **Two** nodes on the AMD `amd-spur` cluster with 8x MI355X; this run used `crsuse2-m2m-254`
  (job 136670) and `crsuse2-m2m-267` (job 136669). Four GPUs (`0,1,2,3`) are used per node.
  Single node per run — no cross-node traffic, so the RDMA fabric is irrelevant to the result.
- One node is enough to reproduce the EP comparison; the second exists only to replicate it and
  measure node-to-node spread. **Both arms of a comparison must run on the same node** — a
  cross-node EP comparison is confounded (this is why the prior sweep added a same-node control).
- Obtained by **holding an existing allocation** (`squeue`), never by requesting one. The scripts
  hard-refuse unless `squeue -j $JOB_ID -h -o '%u %T %N'` equals `<you> RUNNING <NODE>`, and
  hard-refuse nodes `crsuse2-m2m-234`, `-036`, `-249` (peer-owned).
- The node must be free of foreign GPU work. Verify with `docker ps` **and**
  `ls /sys/class/kfd/kfd/proc | wc -l` (0 = no GPU users). A foreign container holding VRAM
  invalidates the numbers — this happened here, see `notes.md`.

### Secrets and access (values NOT in this packup — source them yourself)
- **Cluster**: an `amd-spur` account with `spur exec` rights to your own job. No SSH key or token is
  embedded anywhere here.
- **Docker**: daemon access on the compute node. No registry login — the image is loaded from a
  local archive, never pulled.
- **Model / HF**: none. `HF_HUB_OFFLINE=1`; weights are read from a local read-only mount.
- No API keys, S3, etcd or registry credentials are used by any step below.

### External dependencies (absolute paths, not in this packup)
| What | Path | Notes |
|---|---|---|
| Model weights | `/shared_nfs/models/GLM-5.2-MXFP4` | NFS, mounted read-only |
| Pinned image archive | `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst` | 23,912,216,852 B, SHA256 `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2` |
| Wrapper repo | `infera.dev.yihou.sglang.bench.fast.script` | branch `dev.yihou.sglang.bench.fast.script`, HEAD `e38dadae464e558795e762a96033595e56a3e51a` |

### Repo state
The benchmark code that ran is in `evidence/code_snapshot/` with `code_hashes.sha256` — verify
against those hashes before trusting a re-run. `evidence/code_snapshot/code.diff` is the
working-tree diff at run time; it touches `compare_server.py` and `comparisons/`, **left over from a
previous task and not on this sweep's code path** (`profile_decode.py`, `topology.py` and
`batch_state.py` are clean). **No patch to the benchmark code is required.**

---

## 1. Confirm the allocation and that the node is clean

```bash
export JOB_ID=<your running job id>
export NODE=<crsuse2-m2m-NNN>
export CONTAINER=yihou-hc-<nnn>-<ddmm>       # must match ^yihou-[A-Za-z0-9_-]+$

squeue -j "$JOB_ID" -h -o '%u %T %N'         # must print: <you> RUNNING <NODE>
spur exec "$JOB_ID" bash -lc '
  docker ps --format "{{.Names}} | {{.Image}} | {{.Status}}"
  echo "kfd procs: $(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)"'
```

If a foreign container holds GPUs, resolve ownership **before** proceeding — check whether it is a
live job (`docker top <name>`, CPU time) and whether a peer holds a concurrent allocation on the same
node. Do not silently benchmark on a shared node, and do not kill someone's running job.

## 2. Load the pinned image (skip if `docker image inspect` already resolves it)

```bash
spur exec "$JOB_ID" bash -lc '
  docker image inspect sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d >/dev/null 2>&1 ||
  zstd -dc /shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst | docker load'
```
A few minutes and ~92 GB of image store (the node's NVMe, not `/`). See `logs/imgload_*.log`.

## 3. Create the container

```bash
export ROOT=<path to>/infera.dev.yihou.sglang.bench.fast.script/glm52_decode_internal_yihou_20260909_1057
bash scripts/create_container_yihou.sh
```
Pins the image **by digest**, mounts the repo and the read-only model, gives `/dev/kfd` + `/dev/dri`,
and sets the environment the result depends on. It refuses to clobber an existing container.

## 4. Run one arm, then the other, on the SAME node

```bash
export OUTPUT_ROOT=$ROOT/sweeps/<your-sweep-dir>/iterations_<node>

EP_SIZE=1 bash scripts/run_highconc_yihou.sh 64:0.85 96:0.85 128:0.88   # EP off
EP_SIZE=4 bash scripts/run_highconc_yihou.sh 64:0.85 96:0.85 128:0.88   # EP on
```

Points are given as `C:MEM_FRACTION`; the mem-fraction is **required per point** and never defaulted,
because at these concurrencies the KV pool has to be sized deliberately. Per point the driver calls
`scripts/run_decode.sh`, which snapshots code + hashes + git HEAD + diff and then runs, inside the
container:

```
python3 $ROOT/bench/profile_decode.py \
  --model-path /shared_nfs/models/GLM-5.2-MXFP4 --result-dir <OUT> \
  --tp-size 4 --ep-size <1|4> --enable-dp-attention --batch-size <C> \
  --input-len 70000 --output-len 10000 --accept-length 3.61 \
  --warmup-steps 10 --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope \
  --mem-fraction-static <MF> --max-running-requests <C>
```

**Run detached and redirect to a file** (`setsid nohup ... > driver.log 2>&1 &`). Do **not** pipe the
driver output through `tail`: `runtime.log` holds multi-megabyte single-line tqdm bars and doing so
previously triggered host memory pressure that killed a driver shell. Use `grep -c` or `wc -c`.

### Flags that matter
- `--max-running-requests <C>` is **mandatory above C=48** and is the single most important flag
  here. It is **global**; SGLang divides it by `attn_dp_size`. Without it SGLang prints
  *"Max running requests is reset to 48 for speculative decoding"*, giving 48/4 = **12** request
  slots per rank and capping the local batch at 12 no matter how large the KV pool is. See `notes.md`.
- `--batch-size` is the **GLOBAL** concurrency. With `--enable-dp-attention`, `dp_size = tp_size = 4`,
  so the per-rank batch is `C/4` and the CUDA-graph batch size is that local value. **C must divide
  by 4.**
- `--mem-fraction-static`: **0.85** at C=64/96 (2,436,864 pool tokens against 1,281,024 / 1,921,536
  required) and **0.88** at C=128 (0.85 gives only 2,436,864 against 2,562,048 required). It sizes
  the pool only; it does not change kernel timing.
- `--ep-size 1` is a plain TP MoE configuration; `server_cli` already forces `--moe-a2a-backend none`.
- `--accept-length 3.61` drives simulated acceptance (`match-expected`, `real-draft-token`); EAGLE is
  fixed at steps=5 / draft-tokens=6 / topk=1 with `kv-cache-dtype fp8_e4m3` and FlyDSL DSA backends.

## 5. (Optional) Reproduce the C=160 infeasibility

```bash
EP_SIZE=1 bash scripts/run_highconc_yihou.sh 160:0.97 160:0.974 160:0.978 160:0.98
EP_SIZE=4 bash scripts/run_highconc_yihou.sh 160:0.97 160:0.9772 160:0.978 160:0.98
EP_SIZE=1 bash scripts/run_highconc_xs_yihou.sh 160:0.98    # with expandable_segments
```
All of these are expected to **fail**, each in a documented way (`notes.md`). Every attempt is
preserved in `evidence/points/`.

## 6. Collect and verify

```bash
python3 scripts/collect_highconc_yihou.py <iterations_dir> [<iterations_dir2>] --output summary_yihou.csv
echo "exit=$?"
```

The collector is a **gate, not a formatter**: it asserts `complete`, exact
`useful_output_tokens = C x 10000`, `batch_size / input_len / output_len`,
`(ep_size, tp_size, dp_size, enable_dp_attention)` and `launch_status.exit_code == 0`, and exits
non-zero if any row is not `pass`. Rows can come back `missing`, `incomplete`,
`configuration_mismatch`, `topology_mismatch`, `process_failed`, `process_failed_no_result` or
`process_unconfirmed`.

**Exit code 1 is expected if you also ran step 5** — the deliberate C=160 failures live in the same
tree. With only steps 1-4 present, all rows are `pass` and the exit code is 0.

## Expected output

| C | EP off TPOT ms | EP on TPOT ms |
|---:|---:|---:|
| 64 | 17.519 – 17.553 | 17.947 – 17.977 |
| 96 | 21.936 – 21.999 | 22.162 – 22.375 |
| 128 | 25.530 – 25.792 | 25.781 – 25.919 |

(ranges are the two nodes). Every point must also show
`realized_accept_length = 3.6134393063583814` and `verify_iterations = 2768` — both are
deterministic given the parameters, so a mismatch means the configuration is not the one documented
here, however close the timing looks.

These are **single runs**. Treat differences below ~1 % as within unmeasured variation — that is the
node-to-node spread measured here.

## If it doesn't reproduce
See `notes.md`: the 48-request speculative-decoding cap, the foreign container holding VRAM, the
cold-cache first point, the `tail`/memory-pressure trap, the C=160 memory wall, and the
`AITER_COMMIT` env-var trap.

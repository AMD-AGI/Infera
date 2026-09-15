# Reproduction kit — GLM-5.2 TP8 / DP-attention decode sweep, EP8 and EP1

Goal: reproduce the twenty points (ten EP8 + ten EP1, C = 48 … 320) from a clean machine.

**Estimated time.** The first point of a cold machine is the slow one: AITER JIT compiles and the
408 GB model is read from NFS with a cold page cache. Once warm, a point costs ~3.5 min at C=48
rising to ~6.5 min at C=320 (measured `launch_wall_seconds`: 214 → 384). A full ten-point sweep is
~52 min (measured: 3107 s for the ten EP8 points, 3055 s for the ten EP1 points, summing
`launch_wall_seconds`); both sweeps ~1.7 h of compute plus the cold start. **Do not kill a point that looks
silent — CUDA-graph capture is slow and prints nothing.**

---

## 0. Prerequisites

### Machine
- **One** bare-metal host with **8× MI355X (gfx950)**; this run used `smci355-ccs-aus-n06-25`.
  All eight GPUs are used (`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`). Single node — no cross-node
  traffic, so the RDMA fabric is irrelevant to the result.
- **No scheduler.** Unlike the spur packups, there is no Slurm job to hold and no `spur exec`; the
  scripts here talk to the local Docker daemon directly and contain no allocation guards.
- **The host must be idle and exclusively yours.** Check before starting — a foreign container
  holding VRAM invalidates the numbers:
  ```bash
  docker ps --format '{{.Names}} {{.Status}}'
  rocm-smi --showuse --showmemuse
  ```

### Secrets and access (values NOT included here — source them yourself)
- **Docker**: local daemon access on the host (group membership). **No registry login** — the image
  is already present locally and is never pulled.
- **Model / HF**: none. `HF_HUB_OFFLINE=1`; weights are read from a local read-only mount.
- No API keys, SSH keys, S3, etcd or vault credentials are used by any step below. No secret value
  appears anywhere in this packup.

### External dependencies (absolute paths, not in this packup)
| What | Path / id | Notes |
|---|---|---|
| Model weights | `/perf_apps/data/models/GLM-5.2-MXFP4` | NFS (`10.235.192.7:/performanceapps`), 408 GB, mounted read-only |
| Container image | `sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0` | local tag `rocm-llm-bench:latest`, 66.3 GB, built 2026-09-08 |
| Wrapper repo | `infera.dev.yihou.sglang.bench.fast.script` | branch `dev.yihou.sglang.bench.fast.script`, HEAD `e38dadae` (see `evidence/code_snapshot/git_head.txt`) |
| Benchmark code | `<repo>/glm52_decode_internal_yihou_20260909_1057/bench/` | used **unmodified**; hashes in `evidence/code_snapshot/code_hashes.sha256` |

**If you do not have that image digest**, you need an image carrying SGLang
`402df1e1e453e1e85ec0f5ac4052d36598cc691a` and AITER `2c71811b32c8ce2e1266aedaec199df7d90f597d` on
torch `2.9.1+rocm7.2.0`. The spur packups pin a *different* digest with the same stack — see
`notes.md` → "Deviations".

### Repo state
`evidence/code_snapshot/code.diff` is the working-tree diff at run time and is **empty**: the run
used the committed `bench/*.py` unchanged. Verify against `code_hashes.sha256` before trusting a
re-run. No patch is required.

---

## 1. Set the two paths everything else derives from

```bash
export REPO=<path to>/infera.dev.yihou.sglang.bench.fast.script
export WS=$REPO/glm52_tp8ep8_dpa_c128_yihou_20260914-0423   # or a fresh workspace dir
```

If you start from a fresh workspace, copy this packup's `scripts/` into `$WS/scripts/` and create
`$WS/{iterations,results,logs}`. The scripts derive `WS` from their own location and `REPO` from
`$WS/..`, and they read the benchmark from `$REPO/glm52_decode_internal_yihou_20260909_1057/bench/`.

## 2. Create the container

```bash
bash "$WS/scripts/create_container_yihou.sh"
```

This pins the image **by digest**, mounts the repo read-write and the model read-only, gives
`/dev/kfd` + `/dev/dri`, and sets the environment the result depends on (`HIP_VISIBLE_DEVICES=0..7`,
`AITER_JIT_DIR=/tmp/yihou-aiter-b5aa5bd3d828`, `AITER_USE_FLYDSL_MOE_SORTING=1`,
`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`, `SGLANG_OPT_USE_TOPK_V2=false`, `HF_HUB_OFFLINE=1`).
It refuses to clobber an existing container.

It also applies the four fixes this host needs — each one cost a failed run to find, all are
explained in `notes.md`:
1. runs as the **host uid/gid** (the repo is on an NFS export with `root_squash`);
2. bind-mounts a host-owned copy of the image's `/tmp/aiter_configs` (root-owned in the image);
3. bind-mounts a **synthesized** `/etc/passwd` + `/etc/group` carrying your uid (it is LDAP-only,
   and `getpass.getuser()` raises `KeyError` without it);
4. sets an explicit `PATH` **without `/root/.cargo/bin`** (unreadable as non-root, which turns a
   PATH walk into `EACCES` and kills torch inductor on `nvcc`).

Sanity-check before spending an hour on a sweep:
```bash
docker exec yihou-glm52-tp8ep8-0914 bash -lc \
  'echo $PATH; id; python3 -c "import getpass,shutil,torch; print(getpass.getuser(), shutil.which(\"nvcc\"), torch.cuda.device_count())"'
# expect: PATH without /root/.cargo/bin; uid=<you>; "<you> None 8"
```

## 3. Run the EP8 sweep

```bash
cd "$WS"
setsid nohup bash "$WS/scripts/run_tp8_ep8_dpa_yihou.sh" 48 64 96 128 160 192 224 256 288 320 \
  > "$WS/logs/driver_sweep_yihou.log" 2>&1 &
```

Per point this calls `scripts/run_decode.sh`, which asserts the container's image digest and owner,
snapshots code + hashes + git HEAD + diff, runs

```
python3 $REPO/glm52_decode_internal_yihou_20260909_1057/bench/profile_decode.py \
  --model-path /perf_apps/data/models/GLM-5.2-MXFP4 --result-dir <OUT> \
  --tp-size 8 --ep-size 8 --enable-dp-attention --batch-size <C> \
  --max-running-requests <C> \
  --input-len 70000 --output-len 10000 --accept-length 3.61 \
  --warmup-steps 10 --enable-aiter-allreduce-fusion \
  --enable-fused-qk-norm-rope --mem-fraction-static 0.85
```

inside the container, and writes `launch_status.json` with the exit code and wall time.

**Run it detached and redirect to a file.** Do **not** pipe the driver output through `tail`:
`runtime.log` contains multi-megabyte single-line tqdm bars, and doing that on the spur runs
previously triggered host memory pressure that killed the driver shell.

`run_decode.sh` refuses to overwrite an existing iteration directory, so the sweep is **resumable**:
if you have to restart it, pass only the C values that have no directory yet.

### Flags worth understanding
- `--batch-size` is the **GLOBAL** concurrency. With `--enable-dp-attention`, `dp_size = tp_size = 8`,
  so the per-rank batch is `C/8` and the CUDA-graph batch size is that local value. **C must be
  divisible by 8.**
- `--max-running-requests <C>` is **mandatory, and is not tuning.** SGLang otherwise auto-sets it to
  48 ("Max running requests is reset to 48 for speculative decoding"), and
  `kv_cache_configurator.py:1882` sizes the per-worker `ReqToTokenPool` as
  `max_running_requests // attn_dp_size` = `48 // 8` = **6 slots** — fewer than any local batch here,
  so `alloc_req_slots` fails outright. Passing C yields exactly `C/8` slots. See `notes.md`.
- `--ep-size 8` vs `--ep-size 1` is the only difference between the two sweeps; `server_cli` already
  forces `--moe-a2a-backend none` in both.
- `--accept-length 3.61` drives simulated acceptance (`match-expected`, `real-draft-token`); EAGLE is
  fixed at steps=5 / draft-tokens=6 / topk=1 with `kv-cache-dtype fp8_e4m3` and FlyDSL dsa backends.
- `--mem-fraction-static 0.85` sizes the KV pool **independently of C**. It was never changed.

## 4. Run the EP1 sweep — same container, immediately after

```bash
cd "$WS"
setsid nohup bash "$WS/scripts/run_tp8_noep_dpa_yihou.sh" 48 64 96 128 160 192 224 256 288 320 \
  > "$WS/logs/driver_noep_sweep_yihou.log" 2>&1 &
```

Identical parameters except `--ep-size 1`. **Run it on the same node and the same container** — that
is the whole point of the comparison. Do not recreate the container between the two sweeps.

## 5. Collect and verify — this is the gate, not a formatter

```bash
python3 "$WS/scripts/collect_sweep_yihou.py" "$WS/iterations" \
  --output "$WS/results/sweep_summary_yihou.csv"
echo "exit=$?"     # must be 0

python3 "$WS/scripts/collect_sweep_yihou.py" "$WS/iterations" \
  --prefix tp8_noep_dpa_on_c --expect-ep 1 \
  --output "$WS/results/sweep_summary_noep_yihou.csv"
echo "exit=$?"     # must be 0
```

The collector asserts, per point: `complete`; exact `useful_output_tokens = C × 10000`;
`input_len/output_len = 70000/10000`; `(ep_size, tp_size, dp_size, enable_dp_attention)` equal to
`(8|1, 8, 8, True)`; `local_batch_size == C/8`; `moe_a2a_backend == "none"`;
`realized_accept_length == 3.6134393063583814`; and `launch_status.exit_code == 0`. It exits non-zero
otherwise. A row can come back `missing` (still running or died before writing a result) or `fail`
with the specific problems listed.

`scripts/verify_point_yihou.py <point> [--expect-ep 1]` gates a single point the same way.

## Expected output

Two CSVs with ten `pass` rows each, matching the tables in `README.md`. Every point should also show
`realized_accept_length = 3.6134393063583814` and `verify_iterations = 2768` — these are
deterministic given the parameters and **identical at every concurrency and both EP sizes**, so a
mismatch means the configuration is not the one documented here, no matter how plausible the timing
looks.

These are **single runs** with no repeats. Treat differences of a few percent as within unmeasured
run-to-run variation.

## Capacity ceiling — where this stops, and why

Per rank the KV pool is `#tokens = 3,448,128` (EP8) / `3,471,936` (EP1), and each request reserves
`80,064` tokens, so a point needs `local_batch × 80,064`. C=320 needs `3,202,560` — 92.9 % of the
EP8 pool — and passed. The next step up, C=352, would need `44 × 80,064 = 3,522,816`, which exceeds
**both** pools. This is arithmetic from measured numbers, not a performance prediction; it says the
sweep would next fail on **reservation**, not on device memory (`max_memory_allocated_bytes` only
grew from 253.45 GB at C=48 to 256.36 GB at C=320, on 288 GB cards — measured on the EP8 sweep only).

## If it doesn't reproduce
See `notes.md`: the five environment failures (all cost a run each), the cold-cache first point, the
`--max-running-requests` trap, and what is and is not a real mismatch.

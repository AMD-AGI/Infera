# Reproduction kit — GLM-5.2 decode profiling, TP8 / EP8 / dpa / C=256

Goal: reproduce the top-10 kernel ranking and the graph ON/OFF comparison from a clean machine.

**Estimated time.** One full run is **~6 min** once weights are page-cached and the AITER JIT cache
is warm (measured `launch_wall_seconds`: 357 s graph-ON, 487 s graph-OFF, 470 s with-stack). A cold
first run is far slower — AITER JIT compiles and 408 GB of weights come off NFS. **CUDA-graph capture
is slow and prints nothing; do not kill a silent run.**

---

## 0. Prerequisites

### Machine
- **One** bare-metal host with **8× MI355X (gfx950)**; this run used `smci355-ccs-aus-n06-25`.
  All eight GPUs (`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`). Single node.
- **No scheduler** — the scripts talk to the local Docker daemon directly.
- **Host must be idle and exclusively yours.** This matters more for profiling than for throughput
  benchmarking: a neighbour job perturbs per-kernel event capture, and it cost us a wrong conclusion
  once (see `notes.md`, "Contention corrupts profiler counts").
  ```bash
  docker ps --format '{{.Names}} {{.Status}}'
  rocm-smi --showuse
  ```

### Secrets and access
None. No registry login (image is local, never pulled), `HF_HUB_OFFLINE=1`, weights from a
world-readable NFS mount. No API key, SSH key, S3 or vault credential is used by any step here.

### External dependencies (absolute paths, not in this packup)
| What | Path / id | Notes |
|---|---|---|
| Model weights | `/perf_apps/data/models/GLM-5.2-MXFP4` | NFS, 408 GB, mounted read-only |
| Container image | `sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0` | local tag `rocm-llm-bench:latest`, 66.3 GB |
| Wrapper repo | `infera.dev.yihou.sglang.bench.fast.script` | branch `dev.yihou.sglang.bench.fast.script`; profiling feature is commit **`23f472d2`** |
| Benchmark code | `<repo>/glm52_decode_internal_yihou_20260909_1057/bench/` | `profile_decode.py` + `profiling_yihou.py`; hashes in `evidence/code_snapshot/code_hashes.sha256` |

If you lack that digest you need an image carrying SGLang
`402df1e1e453e1e85ec0f5ac4052d36598cc691a` and AITER `2c71811b32c8ce2e1266aedaec199df7d90f597d` on
torch `2.9.1+rocm7.2.0`.

### Repo state
**Unlike the throughput packup, this one requires the profiling feature.** It is committed, so no
patch is needed — but a checkout older than `23f472d2` will not have `--profile` at all. Verify:
```bash
python3 <repo>/glm52_decode_internal_yihou_20260909_1057/bench/profile_decode.py --help | grep -c '^\s*--profile'
# expect 9 or more; --help is stdlib-only by design and needs no GPU
```
`evidence/code_snapshot/bench_pristine/` holds the pre-feature `bench/*.py` so you can diff and
re-verify the default-off equivalence claim yourself.

---

## 1. Paths

```bash
export REPO=<path to>/infera.dev.yihou.sglang.bench.fast.script
export WS=$REPO/glm52_decode_profiling_yihou_20260914-0851    # or a fresh workspace
```
From a fresh workspace: copy this packup's `scripts/` into `$WS/scripts/` and create
`$WS/{iterations,results,logs}`. Scripts derive `WS` from their own location and `REPO` from `$WS/..`.

## 2. Create the container

```bash
bash "$WS/scripts/create_container_yihou.sh"
```
Pins the image **by digest**, mounts repo rw and model ro, `/dev/kfd` + `/dev/dri`, and applies the
four host fixes each of which cost a failed run on this machine (NFS `root_squash` → run as host
uid/gid; root-owned `/tmp/aiter_configs` → bind-mount a host-owned copy; LDAP-only uid →
synthesized `/etc/passwd`+`/etc/group`; unreadable `/root/.cargo/bin` on `PATH` → explicit `PATH`).
Full explanation in the throughput packup's `notes.md`.

Sanity-check:
```bash
docker exec yihou-glm52-tp8ep8-0914 bash -lc \
  'echo $PATH; id; python3 -c "import getpass,torch; print(getpass.getuser(), torch.cuda.device_count())"'
# expect: PATH without /root/.cargo/bin; your uid; "<you> 8"
```

## 3. The primary run — graph ON

```bash
cd "$WS"
setsid nohup env MAX_STEPS=0 PROFILE_START=1384 PROFILE_NUM=10 GRAPH=on \
  bash scripts/run_profile_yihou.sh p1b_ep8_c256_graphon_midwindow_yihou \
  > logs/p1b_driver.log 2>&1 &
```
expands to (verify with `--dry-run`):
```
python3 $REPO/glm52_decode_internal_yihou_20260909_1057/bench/profile_decode.py \
  --model-path /perf_apps/data/models/GLM-5.2-MXFP4 --result-dir <OUT> \
  --tp-size 8 --ep-size 8 --enable-dp-attention --batch-size 256 --max-running-requests 256 \
  --input-len 70000 --output-len 10000 --accept-length 3.61 --warmup-steps 10 --max-steps 0 \
  --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85 \
  --profile --profile-start-step 1384 --profile-num-steps 10 --profile-ranks 0
```

### Why each profiling flag is what it is — none of these are free choices
- **`--max-steps 0` (full OSL), window at 1384.** 0 means "run the whole OSL". The window sits at the
  run **mean** context (≈75000 of the 70000→80000 sweep). A window at step 20 would sit 152 tokens
  into the sweep. *Measured*: this changes almost nothing (only the DSA logits kernel moves, +7.0 %,
  tracking context; kernel-sum per iteration 88.47 → 88.78 ms, +0.35 %, ranking identical). Run the
  full OSL anyway — it also gives the end-to-end TPOT cross-check.
- **`--profile-ranks 0`.** 8 ranks × traces explodes disk for no extra insight at TP8.
- **`--profile-num-steps 10`.** 10 iterations ≈ 0.95 s of decode → 870 KB trace. Enough to average.
- **`--profile-with-stack` OFF by default.** It costs ~40× trace size per iteration (16.7 MB for 2
  iterations vs 7.96 MB for 10 without). Turn it on only to resolve a kernel's identity — see step 5.
- **`--max-running-requests 256` is mandatory and is not tuning.** SGLang auto-sets 48, and
  `kv_cache_configurator.py:1882` sizes the per-worker `ReqToTokenPool` as
  `max_running_requests // attn_dp_size` = `48 // 8` = 6 slots < local batch 32 → `alloc_req_slots`
  fails outright.
- **No `DEBUG_*` runtime overrides.** Stock defaults, because that is what the published number used.
  In particular **do not set `DEBUG_CLR_GRAPH_PACKET_CAPTURE`** — see `notes.md`.

**Run detached, redirect to a file, and do not pipe through `tail`:** `runtime.log` contains
multi-megabyte single-line tqdm bars.

`run_profile_yihou.sh` refuses to clobber an existing iteration directory and writes
`launch_started.json` **before** launching (so a run that dies pre-launch still leaves a trace — it
did not, once, and 25 minutes of idle GPU went unnoticed).

## 4. The comparison run — graph OFF

```bash
cd "$WS"
setsid nohup env MAX_STEPS=0 PROFILE_START=1384 PROFILE_NUM=10 GRAPH=off \
  bash scripts/run_profile_yihou.sh p0b_ep8_c256_graphoff_midwindow_yihou \
  > logs/p0b_driver.log 2>&1 &
```
Identical except `--disable-cuda-graph`. **Same host, same container, back to back** — that is what
makes it single-variable. Do not recreate the container between them.

## 5. Optional — identify a kernel hiding behind a generic name

```bash
setsid nohup env MAX_STEPS=0 PROFILE_START=1384 PROFILE_NUM=2 GRAPH=off WITH_STACK=1 \
  bash scripts/run_profile_yihou.sh p2_shapes_stack_graphoff_yihou \
  > logs/p2_driver.log 2>&1 &
```
`--profile-with-stack` emits `python_function` events forming a timestamped Python call tree. Match
each `hipLaunchKernel` by `ts` to its innermost enclosing frame. This is how `main_kernel` was
identified. **`--profile-record-shapes` cannot do this** — it is an aten-dispatcher feature and the
kernel bypasses the dispatcher; see `notes.md`.

## 6. Verify — these are the gates, before any analysis

Per run, from `result_yihou.json`:

| check | expected |
|---|---|
| `complete` | `true` |
| `verify_iterations` | `2768` |
| `useful_output_tokens` | `2560000` (= C × 10000, exact) |
| `realized_accept_length` | **`3.6134393063583814`** — bit-identical, both modes |
| `(ep_size, tp_size, dp_size, enable_dp_attention)` | `(8, 8, 8, true)` |
| `local_batch_size` / `moe_a2a_backend` | `32` / `none` |
| `target_graph_iterations` | `2768` graph-ON / `0` graph-OFF |
| `is_performance_measurement` | `false` |
| end-to-end TPOT | within 5 % of 26.2519 ms (we got 26.3154, +0.24 %) |

`realized_accept_length` is the sharpest gate: it is deterministic given the parameters, so an exact
match proves the harness is the one that produced the published number. A 40-iteration run gives
3.80 instead — short runs are not representative of anything.

Then read `profile/top_ops_rank_0_yihou.json`:
**use `top_by_self_device_time_device_kernels_only`.** The sibling `top_by_self_device_time`
contains `record_function` spans, which torch reports as `DeviceType.CUDA` rows that **overlap the
kernels beneath them** — `step[TARGET_VERIFY bs=32]` would otherwise rank first at 838.93 ms and
double-count everything. See `notes.md`.

## Expected output
`profile/` per run: chrome trace (rank 0), `top_ops_rank_0_yihou.{json,txt}`,
`device_timer_rank_{0..7}_yihou.json`, `profile_meta_yihou.json`. Numbers should match
`README.md`'s tables. **Single runs, no repeats** — treat a few percent as unmeasured variance.

## If it doesn't reproduce
`notes.md` covers: the annotation double-counting trap, the DeviceTimer category names changing
between graph modes, why contention corrupts profiler counts, the `MAX_STEPS=0` guard bug, and the
`-w /` CWD/EPERM crash in `--enable-profile-cuda-graph`.

# Notes — gotchas, wrong turns, open questions

Everything here is written as **what / why / how / context** so a future reader learns why a step
matters, not just that it exists. The five failures below each cost a run; four of them are pure
consequences of moving this method off the spur cluster onto a bare-metal host with an NFS home and
LDAP accounts.

---

## The five failed attempts, in the order they were hit

All five are preserved under `evidence/aborted/<name>/` with the exact `config_yihou.json`,
`launch_status.json`, a gzipped `runtime.log` and a plain-text `failure_tail.txt` so you can read
the traceback without gunzipping. Nothing was deleted.

### 1. `aborted_rootsquash_c128_yihou` — the container could not write its own results

**What:** `tee: .../runtime.log: Permission denied` seconds after launch; the run would also have
failed to write `result_yihou.json`.
**Why:** the repo lives on an NFS home export with **`root_squash`**. The container ran as root by
default, NFS mapped root to `nobody`, and `nobody` cannot write into a `yihou`-owned directory.
Verified directly: `docker exec <c> touch <workspace>/logs/x` → `EPERM` as root.
**How fixed:** run the container as the host user — `--user $(id -u):$(id -g)` plus
`--group-add video --group-add render` so the ROCm devices remain usable.
**Context:** the spur packups never saw this because their `$ROOT` was on a share without
`root_squash`. This is the root cause of failures 2–4 as well: every one of them is something that
only breaks when the container is not root.

### 2. `aborted_aiterconfigs_c128_yihou` — AITER could not lock its tuned-GEMM tables

**What:** `PermissionError: [Errno 13] Permission denied: '/tmp/aiter_configs/bf16_tuned_gemm.csv.lock'`.
**Why:** `/tmp/aiter_configs` ships **root-owned** in the image, and AITER writes lock files next to
its CSVs.
**How fixed:** copy the image's own six CSVs out to a host directory
(`/var/tmp/yihou-glm52-aiter-configs`), `chown` them to the run user, and bind-mount that over
`/tmp/aiter_configs`. `create_container_yihou.sh` does this automatically on first use.
**Context:** those CSVs are **tuned kernel-selection tables — they affect which GEMM kernels get
picked, hence performance.** They were copied byte-for-byte from the image rather than starting
empty, precisely so the tuning is preserved. Starting from an empty directory would silently change
kernel selection and quietly invalidate any comparison against the spur numbers.

### 3. `aborted_getpwuid_c128_yihou` — torch died on import

**What:** `KeyError: 'getpwuid(): uid not found: 100882'` from
`torch/_inductor/runtime/cache_dir_utils.py` → `getpass.getuser()`, at import time.
**Why:** the uid is resolved on the host through **LDAP/sssd**, so it exists in neither `/etc/passwd`
on the host nor anywhere in the image. As root this never happens (uid 0 is always in `/etc/passwd`).
**How fixed:** synthesize a passwd/group pair — `cat /etc/passwd; getent passwd $(id -u)` into
`/var/tmp/yihou-glm52-nss/passwd` (same for group, plus `video` and `render`) — and bind-mount them
read-only over `/etc/passwd` and `/etc/group`. `USER`/`LOGNAME` are also exported, which alone would
satisfy `getpass.getuser()`, but not a direct `pwd.getpwuid()` call.
**Context:** mounting the host's `/etc/passwd` directly does **not** work here — the user is not in
it. That was checked (`grep -c yihou /etc/passwd` → 0) before synthesizing one.

### 4. `aborted_nvcc_path_c128_yihou` — `PermissionError: 'nvcc'` during CUDA-graph capture

**What:** after 779 s, deep inside draft CUDA-graph capture:
`Exception: Capture cuda graph failed: PermissionError: [Errno 13] Permission denied: 'nvcc'`.
**Why:** two things compound.
- The image's login-shell PATH begins with **`/root/.cargo/bin`**, and `/root` is `drwx------`. When
  `Popen` walks the PATH candidates for `nvcc`, `execv("/root/.cargo/bin/nvcc")` returns **EACCES**;
  CPython's `_posixsubprocess` keeps the first errno that is neither `ENOENT` nor `ENOTDIR`, so the
  final exception is `PermissionError` even though every later candidate is a plain `ENOENT`. The
  caller tolerates `FileNotFoundError`, not `PermissionError`.
- The `nvcc` probe is **not a real compile step**. The chain is
  `_inductor/compile_fx.py:1211 codegen_and_compile` → `torch._dynamo.repro.after_aot.save_graph_repro`
  → `generate_compiler_repro_string` → `_cuda_system_info_comment()`. It is inductor's repro-dump
  helper gathering CUDA version info on a ROCm-only image. **There is no `nvcc` anywhere in the image**
  (`find /opt /usr -name nvcc` is empty) and no earlier inductor error in the log — the whole failure
  is the repro helper tripping over an unreadable PATH entry.

**How fixed:** give the container an explicit `PATH` with `/root/.cargo/bin` removed. Nothing in this
benchmark needs cargo.
**Verified by decisive experiment**, not by inference:
```
docker exec -e PATH=/root/.cargo/bin:/opt/venv/bin:/usr/bin:/bin <c> \
  python3 -c "import subprocess; subprocess.check_output(['nvcc','--version'])"
# -> PermissionError: [Errno 13] Permission denied: 'nvcc'
# same command with that first entry removed -> FileNotFoundError: [Errno 2]
```
**Context / trap:** an early hypothesis blamed the benign AITER hipify warnings
(`Failed to save /aiter/csrc/include/*_hip.h with "Permission denied", leaving ... unchanged`) that
appear in the same log. **They are unrelated.** The image is pre-hipified, AITER compiled fine, and
the run reached graph capture. Do not chase them.

### 5. `aborted_reqpool_c128_yihou` — the one that is not an environment problem

**What:** after 491 s — i.e. **past** a fully successful CUDA-graph capture —
`RuntimeError: alloc_req_slots runs out of memory. Please set a smaller number for
--max-running-requests. req_to_token_pool.available_size()=6, num_reqs=16`.

**Why:** SGLang auto-sets `max_running_requests` to 48 (the log says
`Max running requests is reset to 48 for speculative decoding. You can override this by explicitly
setting --max-running-requests.`), and `mem_cache/kv_cache_configurator.py:1882` then sizes the
**per-DP-worker** `ReqToTokenPool` as `max_running_requests // attn_dp_size`. With `attn_dp_size = 8`
that is **6 slots**, while the local batch at C=128 is 16.

Confirmed by measurement rather than by reading alone — the same CLI resolved through the pinned
parser inside the container:
```
docker exec <c> python3 -c "...; sa = prepare_server_args(<our cli>); print(sa.max_running_requests)"
# -> 48
```

**How fixed:** pass `--max-running-requests <C>`, giving exactly `C/8` slots. `profile_decode.py`
passes unknown flags through to the pinned `ServerArgs` parser and `--max-running-requests` is not in
its "owned" flag set, so this is legitimate passthrough.

**This is a capacity knob, not tuning.** It sizes the request-slot table so the requested concurrency
can be admitted at all; it changes no performance parameter, and `--mem-fraction-static` stayed at
0.85 throughout. It is nonetheless a **deviation from the spur packups' `server_cli`** and is
recorded as such.

**Important trap:** SGLang's error message suggests lowering `--max-running-requests`, and the
surrounding advice block suggests lowering `--mem-fraction-static` / `--cuda-graph-max-bs-decode`.
**All of that is wrong here.** This is a slot count, independent of KV bytes: the KV pool was
`#tokens = 3,448,128` per rank and the point needed `16 × 80,064 = 1,281,024`, which the harness's
own check had already passed. Following the message would have produced a smaller, silently
different benchmark.

**Corollary worth knowing — the spur TP4 sweep was sitting exactly on this limit.** There
`attn_dp_size = 4`, so `48 // 4 = 12`, which is *exactly* the local batch of its largest point
(C=48). One step further and it would have failed identically. Anyone pushing that method to higher
concurrency must pass `--max-running-requests` explicitly.

---

## Operational gotchas

- **CUDA-graph capture is slow and silent.** On this host, capture alone was ~5 min at C=128
  (target verify ≈ 130 s × 3 stages, draft decode 56.5 s, draft extend 135.4 s). **Do not kill a
  quiet run.** Watch `wc -c` on `runtime.log` and `du -sh $AITER_JIT_DIR` inside the container.
- **Never `cat`/`tail -f` `runtime.log` without a byte limit.** It contains multi-megabyte
  single-line tqdm progress bars; doing so on the spur runs previously caused host memory pressure
  that killed the driver shell. Use `tail -c N`.
- **Run drivers detached** (`setsid nohup … > file 2>&1 &`) and never pipe them through `tail`.
- **Model load is not the bottleneck here.** Cold it took ~60 s for all 282 shards (408 GB over NFS);
  warm, ~9 s to 59 %. `load_pool_capture` was ~65 s at C=128 with a warm AITER cache. Budget the time
  for capture, not for loading.
- **`run_decode.sh` refuses to overwrite an existing iteration directory**, which makes a sweep
  resumable: re-launch with only the C values that have no directory yet.
- **Keep the container alive between sweeps.** The EP8→EP1 comparison is only single-variable
  because both sweeps ran in the same container instance with the same warm AITER JIT cache.

## Deviations from the spur packups (all deliberate, all necessary)

1. **Different image digest.** `b5aa5bd3d828…` here vs `b9a83742f631…` there; that archive does not
   exist on this host. The stack was probed and is identical (SGLang `402df1e1e45…`, AITER
   `2c71811b32c…`, torch `2.9.1+rocm7.2.0`). A different build of the same sources — **not** the same
   image, and not claimed to be.
2. **No scheduler.** All Slurm/spur guards (`squeue` ownership assertion, `spur exec`, the
   `crsuse2-m2m-*` node regex and the forbidden-node list) were removed; this is a bare-metal host we
   hold exclusively. The container image-digest and owner assertions were **kept**.
3. **Non-root container + three bind-mounts + explicit PATH** — failures 1–4 above.
4. **`--max-running-requests <C>` added** — failure 5 above.
5. **Model path** `/perf_apps/data/models/GLM-5.2-MXFP4` instead of `/shared_nfs/models/...`.
6. **8 GPUs instead of 4**, TP8 instead of TP4.

## Open questions (deliberately not answered)

- **The C=224 sign flip.** EP1 is faster than EP8 at nine of the ten concurrencies; at C=224 it is
  0.19 % slower — the only point where the sign reverses. With single runs and no repeats, this
  cannot be distinguished from run-to-run noise. **The measurement that would settle it:** repeat
  C=224 (both EP sizes) three or more times and compare the spread to the 0.19 % gap.
- **Why EP1's KV pool is larger.** EP1 consistently got `#tokens = 3,471,936` per rank versus EP8's
  `3,448,128` (+0.69 %), stable across all ten points on each side. The obvious story — "EP1 shards
  the MoE weights more cheaply, leaving more room" — is **contradicted by measurement**: EP1's
  `target_initialized_bytes` is *larger* (165.61 GB vs 164.48 GB). The pool is sized post-capture from
  free memory, so explaining this needs a staged memory measurement across load → capture → pool
  sizing. Not done.
- **The shape of the TPOT-vs-C curve.** TPOT rises 12.083 → 30.772 ms and per-GPU throughput 496.56
  → 1299.89 tok/s over C=48 → 320. No per-stage attribution (attention vs MoE vs collective) was
  performed, so no mechanism is claimed.
- **The AITER build-arg mismatch** in the image environment (`AITER_COMMIT=d9e5ef7ce08…` vs the
  checked-out `/aiter` HEAD `2c71811b32c…`). Recorded in `environment.md`; not investigated.

## What "pass" does and does not mean

- `complete: true` means the wrapper's requested useful-output count was reached — **not** that the
  generated text was correct. The prefix KV contents and the acceptance are synthetic; the weights,
  the MoE routing and the kernels are real.
- Throughput counts only useful output tokens. Input tokens and terminal over-computation are not
  counted.
- `tok/s = C × 1000 / TPOT` by construction. Reporting both is reporting one measurement twice.
- `verify_iterations = 2768` and `realized_accept_length = 3.6134393063583814` are deterministic
  given these parameters and came back **identical at all 20 points**. They are the cheapest
  configuration check available: if they differ, the configuration is not the documented one,
  regardless of how plausible the timing looks.

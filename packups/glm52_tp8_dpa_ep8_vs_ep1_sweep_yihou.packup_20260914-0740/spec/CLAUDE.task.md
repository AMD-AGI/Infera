# Task — GLM-5.2 internal decode bench: TP8 / EP8 / DP-attention, concurrency sweep

## Task description
Measure decode performance of GLM-5.2-MXFP4 on **8× MI355X** with the scheduler-free internal
harness (`bench/profile_decode.py`).

**Phase 2 (current): concurrency sweep** C = 48, 64, 96, 128, 160, 192, 224, 256, 288, 320.
C=128 is already measured (`iterations/tp8_ep8_dpa_on_c128_yihou`, TPOT 17.8478 ms,
7171.77 tok/s) and is reused, not re-run. Everything else below is unchanged per point.

Capacity arithmetic for the sweep (per rank): the KV pool measured at C=128 was
`#tokens = 3,448,128`, and each request reserves `80,064` tokens, so
`local_batch × 80,064` must fit: C=320 ⇒ `40 × 80,064 = 3,202,560` (93 % of the pool) — it fits,
but only just, and the pool shrinks slightly as `--max-running-requests` grows the
`req_to_token` table. **If a high-C point fails on capacity or OOM, report it — do not lower
`--mem-fraction-static`.**

**Phase 3 (queued): the same sweep with expert parallelism OFF** — `--ep-size 1`, a plain TP MoE,
every other flag byte-for-byte identical, same C list, run in the **same container on the same
node** so that EP8-vs-EP1 is a single variable. Driver: `scripts/run_tp8_noep_dpa_yihou.sh`;
iteration names `tp8_noep_dpa_on_c<C>_yihou`; gate with
`collect_sweep_yihou.py --prefix tp8_noep_dpa_on_c --expect-ep 1`.
Do **not** recreate the container between phases — that would destroy the comparison.

Phase 1 (done) measured exactly one operating point:

| Knob | Value |
|---|---|
| tp_size | 8 |
| ep_size | 8 |
| DP attention | **on** (`--enable-dp-attention`, so `dp_size = 8`, local batch = 128/8 = **16**) |
| global concurrency C | **128** (sweep: see above) |
| `--max-running-requests` | **= C** (required; see "Other notable details") |
| ISL / OSL | 70000 / 10000 |
| accept length | 3.61 |
| mem-fraction-static | 0.85 |
| warmup steps | 10 |

Deliverable: a verified `result_yihou.json` with `complete: true`,
`useful_output_tokens == 128 × 10000 == 1280000`, topology `(tp=8, ep=8, dp=8, dpa=true)`,
driver `exit_code == 0`, plus TPOT / throughput reported to the user in Chinese.

## Background
This is a **new machine we now hold** (`smci355-ccs-aus-n06-25`, bare metal, no slurm/spur).
The method and all reference numbers come from prior runs on the `amd-spur` cluster, packed up in
`packups/glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/` (TP4/EP1 sweep + TP4/EP4 control) and
`packups/glm52_tp4_ep4_dpa_sweep_yihou.packup_20260910-062400/`.
The harness intercepts SGLang's model forward via `TpModelWorker` + `EAGLEWorkerV2`: **no HTTP
server, no Scheduler, no PD fake-input path**. Real weights, real MoE routing; synthetic prefix KV
and simulated acceptance.

## Context — what is different on this machine
| Item | spur packup | here |
|---|---|---|
| Allocation | slurm job + `spur exec` | bare metal, plain `docker` (all slurm/spur guards **removed**) |
| Node guards | `crsuse2-m2m-*` regex, forbidden node list | not applicable; host is exclusively ours |
| Model | `/shared_nfs/models/GLM-5.2-MXFP4` | **`/perf_apps/data/models/GLM-5.2-MXFP4`** (NFS, 408 G, mounted `:ro`) |
| Image | `sha256:b9a83742f631…` (archive not present here) | **`sha256:b5aa5bd3d828…`** = local tag `rocm-llm-bench:latest` |
| GPUs | 4 of 8 | **8 of 8** (`HIP_VISIBLE_DEVICES=0..7`) |

**Why the substitute image is acceptable (first-hand, probed 2026-09-14):** `rocm-llm-bench:latest`
carries SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a` (the pinned commit), AITER
`2c71811b32c8ce2e1266aedaec199df7d90f597d`, torch `2.9.1+rocm7.2.0` — all identical to the packup's
recorded stack. Only the image **digest** differs (different build of the same sources). This must be
stated in the report; it is a different digest, not a different stack.

## Key references
- `packups/glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/REPRODUCE.md` — the procedure
- `packups/.../environment.md` — pinned stack, env vars, resolved `server_cli`
- `packups/.../notes.md` — failure modes (foreign containers, cold cache, driver-shell kill)
- `glm52_decode_internal_yihou_20260909_1057/bench/` — the benchmark code (used **unmodified**)
- `glm52_decode_internal_yihou_20260909_1057/README.md` — harness semantics and caveats

## Core principles
1. **No code change to `bench/*.py`.** If the run needs one, stop and report — do not silently patch.
2. **Never tune a number or relax a validation to make the point pass.** `complete: true` and the
   exact token count are gates, not formatting.
3. **All work stays inside this workspace.** Nothing outside is created, moved or deleted.
   Deleting anything whose path lacks `yihou` is forbidden, always.
4. **Suspend, don't conclude.** Report what was measured. Do not explain a number you have not
   attributed with a measurement.
5. Run inside the container; do not run benchmark workloads on the host directly.
6. Work in English; report to the user in Chinese.

## Other notable details
- **CUDA-graph capture is slow and silent.** Cold AITER JIT + cold NFS page cache ⇒ expect
  `load_pool_capture` ~15 min, possibly more here (408 G over NFS, first read). **Do not kill a
  quiet run.** Watch `runtime.log` size and `$AITER_JIT_DIR` growth instead.
- **Run the driver detached** (`setsid nohup … > driver.log 2>&1 &`). Do **not** pipe driver output
  through `tail`/`head` on the host: `runtime.log` contains multi-megabyte single-line tqdm bars and
  doing so previously killed the driver shell via host memory pressure.
- `AITER_JIT_DIR` is keyed to the image digest and persists across runs; it affects startup only.
- Expected invariants if the config is the documented one:
  `realized_accept_length = 3.6134393063583814`, `verify_iterations = 2768`. A mismatch means the
  configuration is not the intended one, no matter how plausible the timing looks.
- Memory headroom note: at TP4/C=48 the per-rank KV pool was 2,436,864 tokens with
  `reserved_tokens_per_request = 80064`. Here local batch is 16 ⇒ 1,281,024 tokens reserved per rank.
  If the pool comes back smaller than that, **report it**; do not quietly lower `--mem-fraction-static`.
- **`--max-running-requests C` is mandatory**, not tuning. SGLang auto-sets it to 48
  ("Max running requests is reset to 48 for speculative decoding"), and
  `kv_cache_configurator.py:1882` then sizes the per-worker `ReqToTokenPool` as
  `max_running_requests // attn_dp_size`. With `attn_dp_size = 8` that is 6 slots — fewer than
  any local batch in this sweep — and `alloc_req_slots` fails. Passing C gives exactly `C/8`
  slots. (The spur TP4 packup never hit this: `48 // 4 = 12` was exactly its largest local batch.)
- Environment fixes this host needs, all already in `scripts/create_container_yihou.sh`: non-root
  container (NFS root_squash), bind-mounted `/tmp/aiter_configs` (root-owned in the image),
  synthesized passwd/group (uid is LDAP-only, `getpass.getuser()` raises), and an explicit `PATH`
  without `/root/.cargo/bin` (unsearchable dir makes a PATH walk return EACCES, so torch's
  inductor repro helper dies on `nvcc` with `PermissionError` instead of `FileNotFoundError`).
- `tok/s = C × 1000 / TPOT` by definition in this harness — one measurement in two units, not two
  independent results.

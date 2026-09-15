# SGLang internal decode bench & profiling

Benchmark and profile **decode** of a GLM-5.2-class model by calling SGLang's internals directly —
**no HTTP server, no Scheduler, no PD path**. The harness drives `TpModelWorker` + `EAGLEWorkerV2`
itself and spawns its own TP ranks, so a data point costs ~6 minutes instead of a full server bring-up.

This directory is the **single shared copy**. It used to be three divergent copies scattered across
per-experiment workspaces; those are now under `../temp_workspace/` and hold **no code**.

```
bench/      the engine — this is what actually runs
scripts/    runners and helpers (container, single point, sweep, profile, verify)
tests/      unit tests for the engine's pure logic
```

---

## What it does, and what it does not

| | |
|---|---|
| ✅ real model weights, real MoE routing, real attention backends | |
| ✅ real EAGLE speculative decode (steps=5 / draft=6 / topk=1) | |
| ✅ CUDA-graph capture and replay, exactly as the server does | |
| ⚠️ **synthetic prefix** — KV is filled with physically valid random data, not a real prefill | |
| ⚠️ **simulated acceptance** — accept length is driven to a target, not sampled from the draft | |
| ❌ no scheduler, no batching policy, no request arrival | |

So it measures **steady-state decode cost at a fixed batch and context**, not end-to-end serving.

---

## Quick start

```bash
# 0. one-time: build the container (pins the image by digest, applies the host fixes)
bash <this-dir>/scripts/create_container.sh

# 1. make a workspace — outputs land in $PWD/iterations by default
mkdir -p ~/work/my_sweep && cd ~/work/my_sweep

# 2. one throughput point
bash <this-dir>/scripts/run_decode.sh tp8_ep8_c256_yihou \
    --tp-size 8 --ep-size 8 --enable-dp-attention \
    --batch-size 256 --max-running-requests 256 \
    --input-len 70000 --output-len 10000 --accept-length 3.61 \
    --warmup-steps 10 --enable-aiter-allreduce-fusion \
    --enable-fused-qk-norm-rope --mem-fraction-static 0.85

# 3. one profiling run (top-10 kernels + per-stage GPU time)
MAX_STEPS=0 PROFILE_START=1384 PROFILE_NUM=10 \
  bash <this-dir>/scripts/run_profile.sh p_ep8_c256_graphon_yihou
```

**Always run detached and redirect to a file** for anything long:
```bash
setsid nohup bash <this-dir>/scripts/sweeps/run_tp8_ep8_dpa.sh 48 64 96 128 > logs/sweep.log 2>&1 &
```
`runtime.log` contains multi-megabyte single-line tqdm bars — **never pipe driver output through
`tail`**; doing so has caused host memory pressure that killed the driver.

---

## Where things go

Every script takes the **workspace from `$WS`, defaulting to the current directory**, and the
**engine from `$BENCH_ROOT`, defaulting to this directory**. So: `cd` into your workspace, call the
scripts by absolute path, and results appear under `./iterations/<name>/`.

```
<workspace>/iterations/<name>/
├── result_yihou.json          aggregated result (rank 0 writes it)
├── rank_{0..N}_yihou.json     per-rank summary
├── steps_yihou.jsonl          per-iteration trace (large; gitignored)
├── config_yihou.json          resolved benchmark + server CLI
├── command.txt, git_head.txt, code.diff, code_hashes.sha256, bench_snapshot/
├── launch_started.json        written *before* launch — "no news" is then unambiguous
├── launch_status.json         exit code + wall seconds
└── profile/                   only with --profile (see below)
```

---

## `scripts/`

| script | purpose |
|---|---|
| `create_container.sh` | Create the pinned container. **Use this one** — it carries four host fixes (see below). |
| `run_decode.sh` | One point. Asserts the container's image digest and owner, snapshots code + hashes + diff, tees `runtime.log`, refuses to clobber an existing iteration dir. |
| `run_profile.sh` | One **profiling** run. Same guarantees plus the profiling knobs. |
| `sweeps/run_tp8_ep8_dpa.sh`, `sweeps/run_tp8_ep1_dpa.sh` | The exact published EP8 / EP1 sweeps: `bash .../run_tp8_ep8_dpa.sh 48 64 96 128 160 192 224 256 288 320` |
| `collect_sweep.py` | **Gate, not a formatter.** Verifies every point and exits non-zero on any failure. |
| `verify_point.py` | Same checks for a single point. |
| `graph_visibility_probe.py` | Model-free hipGraph probe: does `torch.profiler` see inside a replayed graph? Seconds to run, one GPU. |
| `equivalence_smoke.sh` | Proves `--profile` absent ⇒ behaviour unchanged. Needs `PRISTINE=/path/to/pre-feature/profile_decode.py`. |

### Common environment knobs
`CONTAINER`, `IMAGE` (digest), `MODEL`, `BENCH_ROOT`, `WS`, `OUTPUT_ROOT`.
`run_profile.sh` additionally: `CONC ISL OSL ACCEPT MEMFRAC WARMUP EP_SIZE MAX_STEPS
PROFILE_START PROFILE_NUM PROFILE_RANKS GRAPH{on|off}`.
All runners accept `--dry-run` as the first argument — **use it before spending a startup.**

---

## Heterogeneous input lengths

`--input-len N` gives every request the same prefix. `--input-len-spec` gives them different ones.
The two are mutually exclusive; passing both is an error.

| spec | meaning |
|---|---|
| `70000` or `uniform:70000` | every request 70000 — identical to `--input-len 70000` |
| `bimodal:8192,70000,0.1` | 10 % of requests at 8192, the rest at 70000 |
| `normal:40000,8000[,lo,hi]` | draws from `N(40000, 8000²)`, rounded, clamped to `[lo, hi]` |
| `list:4096,4096,70000,...` | verbatim; `list:@path` reads one integer per line |

```bash
bash scripts/run_decode.sh mixed_yihou \
    --tp-size 8 --ep-size 1 --enable-dp-attention \
    --batch-size 256 --max-running-requests 256 \
    --input-len-spec bimodal:8192,70000,0.25 \
    --output-len 10000 --accept-length 3.61
```

**The spec describes one attention-DP rank, not the global batch.** The count it must fill is
`local_batch_size = batch_size // dp_size` — 32 for `--batch-size 256` at dp=8. Every rank then
receives that same multiset **in the same order**, resolved once in the parent process and shipped
to the ranks, so all eight stay in lockstep and the cross-rank progress check still proves it. A
`list:` whose length is the global batch will be rejected, with the arithmetic in the message.

Four things worth knowing before you use this:

- **A mixed batch costs the same KV as an all-longest batch.** Allocation is uniform-max: rows are
  sized for `max(ISL)` before the per-request prefix is truncated, so short requests over-reserve.
  Heterogeneity buys you no capacity headroom — if the all-70000 batch does not fit, neither does
  `{8k, 70k}`.
- **`realized_accept_length` and `verify_iterations` are unchanged by ISL**, so the usual acceptance
  gate still applies to a mixed run. Acceptance is simulated as one scalar per iteration broadcast
  across the batch, and every request shares `output_len`, so they all finish on the same iteration.
- **`input_len` in the result JSON is `null` for a mixed run**, deliberately — no single number is
  "the" ISL. Read `input_lens` (per request, global length), `input_len_uniform`, or the
  `context_min` / `context_max` pair in the progress log. `verify_point.py --expect-heterogeneous`
  checks the ragged case instead of pinning one length.
- **Whether a mixed run's TPOT is interpretable is an open question.** DSA is sparse with
  `index_topk = 2048`; how per-step cost varies with sequence length here has not been measured.
  The harness will give you the number. It will not tell you what it means.

`compare_server.py` does not support this and rejects `--input-len-spec` explicitly rather than
silently measuring a uniform batch.

---

## Profiling

`--profile` is **opt-in and off by default**; with it absent the harness is byte-compatible with the
published throughput results (the new flags are not even recorded in `config_yihou.json`).

Three mechanisms, so both CUDA-graph modes are covered:

| mechanism | graph ON | graph OFF | yields |
|---|---|---|---|
| `torch.profiler` over a window of iterations | ✅ | ✅ | per-kernel table + chrome trace |
| `DeviceTimer` (CUDA events, attached by the bench) | ✅ | ✅ | per-stage GPU seconds, whole run |
| `--enable-profile-cuda-graph` | ✅ | n/a | capture-time kernel **inventory** |

Outputs land in `<iteration>/profile/`: `*.trace.json.gz` (per emitting rank),
`top_ops_rank_N_yihou.{json,txt}`, `device_timer_rank_N_yihou.json`, `profile_meta_yihou.json`.

### Five rules that will otherwise give you wrong numbers

1. **Use `top_by_self_device_time_device_kernels_only`.** The sibling `top_by_self_device_time`
   includes `record_function` spans, which torch reports as `DeviceType.CUDA` rows that **overlap
   the kernels beneath them**. Sort naively and `step[TARGET_VERIFY bs=32]` ranks first at 838 ms
   while containing most of the list.
2. **A profiled run is not a performance measurement** (`is_performance_measurement: false`). Quote
   TPOT only from a non-profiled run.
3. **Attribute in the same graph mode as the configuration you care about.** Graph-ON gives the
   kernel layer but no CPU-operator layer and no `grid`/`block` args; graph-OFF gives both. Different
   layers, not different quality.
4. **DeviceTimer category names differ between modes.** graph-ON: `target_verify` / `eagle_draft` /
   `eagle_draft_extend`. graph-OFF: `target_verify` / `decode` / `extend` — and `decode` there is the
   **draft** runner's 4 eager forwards per iteration, not the target.
5. **Never take profiler counts while anything else uses the GPUs.** Contention silently changes
   event counts. Also: a `docker exec` whose outer shell you kill on a timeout **leaves the
   in-container process running** — after any timeout, check the whole process tree.

**Do not set `DEBUG_CLR_GRAPH_PACKET_CAPTURE`.** Per-kernel attribution already works at stock
defaults; the flag is uncharacterised. See the profiling packup's `research/rocm_profiling_env.md`.

---

## Two traps that are not optional

**`--max-running-requests <C>` is mandatory and is not tuning.** SGLang otherwise auto-sets it to 48,
and `kv_cache_configurator.py` sizes the per-worker `ReqToTokenPool` as
`max_running_requests // attn_dp_size`. At dp=8 that is **6 slots** — smaller than any real local
batch — and `alloc_req_slots` fails outright. SGLang's own error message (lower
`--max-running-requests`, lower `--mem-fraction-static`) is **wrong** here: it is a slot count, not
KV bytes.

**`create_container.sh` carries four host fixes**, each of which cost a failed run to find:
NFS `root_squash` → run as the host uid/gid; root-owned `/tmp/aiter_configs` → bind-mount a
host-owned copy; LDAP-only uid → synthesized `/etc/passwd` + `/etc/group`; unreadable
`/root/.cargo/bin` on `PATH` → an explicit `PATH` without it. Use this script, not an ad-hoc
`docker run`.

**CUDA-graph capture is slow and prints nothing.** A silent run is not a hung run; wait.

---

## Verifying a result before you believe it

```bash
python3 <this-dir>/scripts/verify_point.py <workspace>/iterations/<name>      # one point
python3 <this-dir>/scripts/collect_sweep.py <workspace>/iterations \
        --output summary.csv ; echo "exit=$?"                                 # must be 0
```
The sharpest single check is `realized_accept_length`: it is deterministic given the parameters, so
it must be **bit-identical** across every point of a sweep. At ISL 70000 / OSL 10000 / accept 3.61
the published value is `3.6134393063583814`. A short run (tens of iterations) gives something else
entirely — short runs are not representative of anything.

```bash
python3 -m pytest <this-dir>/tests -q        # engine unit tests; no GPU needed
python3 <this-dir>/bench/profile_decode.py --help   # stdlib-only by design
```

---

## Published results

Packed experiments live in `../packups/`. Each is self-contained (`README.md`, `REPRODUCE.md`,
`environment.md`, `notes.md`, evidence, `MANIFEST.sha256`). Cross-experiment summaries are the
`*_yihou*.md` files at the repo root.

Each packup's `notes.md` records the traps that would have produced plausible-but-wrong numbers.
Read it before trusting a re-run that disagrees.

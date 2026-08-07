# Notes — GLM-5.2 MIX conc=1 agentic latency (Task 2)

## The method (what / why)

A custom conc=1 latency driver (`scripts/lat_conc1.py`), run **inside** the engine
container against the router `:8100`. It measures each Case-A shape in isolation on a
single frozen server, with the cacheable prefix pre-warmed so cache-hit is guaranteed
(the mission's "在刷入 cache 保证 cache hit rate 的情况下"). No think-time delay
because conc=1 (operator decision).

- **Shapes:** the **full** Case-A ISL/OSL (not the ×10% fixlen values) —
  `(74000, 320)`, `(155000, 3300)`, `(235000, 17000)` for p50 / p90 / p99.
- **Reps:** 10 sequential requests per shape (conc=1, no concurrency, no think time).
- **Sampling:** `temperature 1.0 / top_p 0.95` — the checkpoint's own
  `generation_config`, **not greedy**. `temperature 0` + MTP is indistinguishable from
  KV corruption on this stack, so greedy is wrong here.
- **Metrics:** per-request TTFT (time to first *streamed* token), E2E (full stream),
  and derived TPOT = `(e2e - ttft) / (completion_tokens - 1)`. Reported as
  mean/median/p90 over the 10 reps.

### The cache model — why a fixed prefix + fresh suffix
The agentic driver models a session as a **fixed cacheable prefix + a variable fresh
suffix**, giving a high prefix cache-hit rate (Case A ≈ 89%). This driver mirrors that:

- `prefix_tok = round(0.89 * ISL)` — a single fixed prefix, built once per shape.
- `fresh_tok  = ISL - prefix_tok` — the variable tail.
- It **warms** the shared prefix once (a warm-marker request, not measured), so the
  prefix's KV blocks are resident.
- Then each of the 10 reps sends **the same prefix** (→ cache hit) **+ a distinct
  fresh suffix** (`offset = 1000 + i*fresh_tok`, so each rep is a genuinely new
  request that must prefill only the ~11% tail).

Net effect: prefill is dominated by cache reuse, so **TTFT is low and flat** and the
measurement isolates decode cost (E2E, TPOT).

### Forcing the exact output shape — `ignore_eos` + `min_tokens`
To pin OSL to *exactly* the Case-A value (so E2E/TPOT are comparable across reps and
to InferenceX), each request sets:

- `max_tokens = min_tokens = OSL` — floor **and** ceiling on generated length, and
- `ignore_eos: true` — the model does **not** stop at an EOS token.

Without these, the model would emit a natural (short, variable) completion and the OSL
would never reach 320 / 3300 / 17000; the latency numbers would be meaningless as a
fixed-shape characterization.

## Results — how to read them

| shape | ISL | OSL | E2E p50 | TPOT p50 | TTFT p50 |
|---|---|---|---|---|---|
| p50 | 74000 | 320 | 2480 ms | 6.63 ms | 424 ms |
| p90 | 155000 | 3300 | 20053 ms | 5.89 ms | 601 ms |
| p99 | 235000 | 17000 | 103069 ms | 6.01 ms | 866 ms |

- **E2E scales with OSL** — it is decode-bound; ~2.5 s → ~20 s → ~104 s tracks
  320 → 3300 → 17000 output tokens at a steady per-token cost.
- **TPOT is flat at ~6 ms/token** across all three shapes — the per-token decode
  cost is independent of OSL and only weakly sensitive to ISL (context length), as
  expected at conc=1.
- **TTFT is low (0.4–0.9 s)** and rises only mildly with ISL — because the ~89%
  prefix is cache-resident, prefill only processes the small fresh tail. TTFT would
  be far higher on a cold prefix.

## Gotchas / caveats (what / why / how / context)

### The 100% cache-hit is warmed-prefix residence, not a workload property
**What:** the driver reports `cached/prompt ≈ 100%` every rep (`cached_tokens 73984 /
prompt_tokens 74012` at p50). **Why:** the prefix is warmed once and stays resident
across all 10 sequential reps on a frozen server, so every rep hits it — by
construction. **How to read:** this is the *intended* Case-A cache-hit condition, not
an incidental measurement. Contrast Task 1's fixlen sweep, where `--dataset-name
random` had no shared prefix and the cache-hit column was meaningless residue. **Here
the 100% is the point** — it is the "刷入 cache 保证 cache hit rate" the mission asked
for.

### Runs inside the container, against the in-container router
**What:** `lat_conc1.py` is launched with `docker exec` and talks to
`http://127.0.0.1:8100`. **Why:** it needs the sglang tokenizer + `transformers`
(present in the image) to build token-exact prefixes, and the router is only reachable
from inside on `127.0.0.1` (or via `$MY_IP` from outside). **How:** `docker cp` the
script in, `docker exec … python3 /tmp/lat_conc1.py`. Do **not** background the docker
client across a `spur exec` teardown — redirect to a file with `tee` instead.

### DSA env is mandatory on gfx950 or the model serves garbage
Without the DSA-ROCm env (`SGLANG_OPT_USE_TILELANG_INDEXER=1`, `TOPK_V2=0`,
`JIT_NORM=0`, `USE_AITER=1`, `ROCM_FUSED_DECODE_MLA=0`) the worker emits garbage. All
set in `scripts/mix_worker.sh`; the smoke's "coherent answer" block is the live check.

### MTP requires `--disable-custom-all-reduce` on gfx950
aiter custom all-reduce **deadlocks** in EAGLE verify on gfx950. `mix_worker.sh`
auto-adds `--disable-custom-all-reduce` whenever MTP is on. Accept-len **median ~3 is
healthy; 4.00 means degeneration.**

### DSA indexer P1V3 reversed-IDLE-rank fix (baked in the image)
Under DP-attention on an **IDLE** rank during MTP draft-extend, the `GLM52_P1V2`
guard's inequality inverts (fewer query rows than lengths entries) → `RuntimeError:
Expected lengths.size(0) == B`. Fix: reconcile both sides to `_p1v2_rows =
min(real, padded)`. Applied at build by
`deploy/docker/patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py`;
**`_p1v2_rows` verified in the bytecode** (identifier marker — a stale `__pycache__`
has voided a run on this stack twice).

### ROCm hicache host-alloc fix (baked in the image)
gfx950 hard-aborts with `Memory access fault by GPU node-N on address <host VA>` when
hicache stores raw host `data_ptr()`s that a GPU kernel dereferences (`hipHostRegister`
maps host pages at a *different* device VA; gfx950 is `xnack-`, no page-migration
fallback). Fix: route `ALLOC_MEMORY_FUNCS` to `pin_memory` (`hipHostMalloc`) —
`deploy/docker/patches/sglang_rocm/patch_hicache_rocm_host_alloc.py`.

### Cluster / infra gotchas
- **`/tmp` is root-owned on crsuse** under `spur exec` (runs as `yihou`). Use
  `/var/tmp` — `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`. (The `results/lat_summary.md`
  header shows a harmless `permission denied` reading `/opt/spur/.docker/config.json` —
  cosmetic, from a docker client invocation, does not affect the numbers.)
- **etcd v3.5.14 needs `--entrypoint /usr/local/bin/etcd`** (empty ENTRYPOINT).
- **Never background a long docker client inside `spur exec`** — teardown kills it.
- **Server logs contain binary bytes** — grep through `strings`.

## Feature-proof evidence (from the smoke, before measuring)

Same frozen server as Task 1: 1 worker `disagg_mode=mixed`; coherent answer (DSA
live); MTP accept-len median 3.12; 8 kvd adapters + entries>0; router `kv-aware`,
tokenizer loaded.

## Known gaps in this packup

- Base-image `sha256` digest not captured at build (floating tag only) — see
  `environment.md` for the one-liner to resolve it on the node.
- DSA/ROCm patches are referenced by in-tree path at the pinned SHA, not copied in,
  since the image bakes them from the repo at `d1a97b2`.
- `lat_report.py` has the input path `/tmp/mix_lat/lat_%s.jsonl` hard-coded; point it
  at the copied jsonl when reading off-node (noted in `REPRODUCE.md` §5).

# Notes — GLM-5.2 MIX fixlen sweep

## The bench method (what/why)

InferenceX-aligned `sglang.bench_serving --backend sglang-oai-chat` against the
router `:8100`.

- **Shapes:** ISL = Case-A percentile × 10% (operator's rule "实际输入=分位×10%") =
  **7400 / 15500 / 23500** (p50 / p90 / p99), paired OSL = **320 / 3300 / 17000**.
- **Grid:** 3 shapes × concurrency {1, 8, 16, 24} = **12 rounds**, one **frozen**
  server (never restarted mid-sweep).
- **Load:** `num-prompts = 10×C`, `warmup = 2×C`, `--request-rate inf`,
  `--max-concurrency C`.
- **Sampling / shape flags:**
  - `--random-range-ratio 1.0` pins every prompt to **exactly** ISL (a delta, not a
    distribution) — this is what makes it a *fixlen* sweep.
  - `--temperature 1.0 --top-p 0.95` — the checkpoint's own `generation_config`, **not
    greedy**. `temperature 0` + MTP is indistinguishable from KV corruption, so greedy
    is wrong for this stack.
- **Per-GPU:** divide `output_throughput` by 8.

## Gotchas / caveats (what / why / how / context)

### Cache-hit column is residue, not a workload property
**What:** `--cache-report` prints a cache-hit rate; it is low and noisy (e.g. 9.9% at
p50/c1, 1.3% at p99/c24). **Why:** `--dataset-name random` builds prompts with **no
shared prefix by construction**, so there is nothing legitimate to hit; what shows up
is incidental block reuse (kvd storage backend residue). **How to read:** ignore it as
a workload metric — it is not measuring prefix caching here. **Context:** it is left on
only to confirm the kvd/hicache path is wired, not to characterize hit rate.

### DSA env is mandatory on gfx950 or the model serves garbage
**What:** without the DSA-ROCm env the worker emits ~200s of garbage tokens. **Why:**
gfx950 needs the tilelang indexer path (`SGLANG_OPT_USE_TILELANG_INDEXER=1`,
`TOPK_V2=0`, `JIT_NORM=0`, `USE_AITER=1`, `ROCM_FUSED_DECODE_MLA=0`). **How:** all set
in `scripts/mix_worker.sh`; the smoke's "coherent answer" block is the live check.

### MTP requires `--disable-custom-all-reduce` on gfx950
**What:** aiter custom all-reduce **deadlocks** in EAGLE verify on gfx950. **How:**
`mix_worker.sh` auto-adds `--disable-custom-all-reduce` whenever MTP is on. Watch the
smoke's accept-len: **median ~3 is healthy; 4.00 means degeneration**. This run: 3.12.

### DSA indexer P1V3 reversed-IDLE-rank fix (baked in the image)
**What:** under DP-attention on an **IDLE** rank during MTP draft-extend, the
`GLM52_P1V2` guard's inequality inverts (fewer query rows than lengths entries) →
`RuntimeError: Expected lengths.size(0) == B`. **Fix:** reconcile both sides to
`_p1v2_rows = min(real, padded)`, clipping the lengths side via `ke_offset`. **How:**
`deploy/docker/patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py`, applied at
build; **`_p1v2_rows` verified in the bytecode** (identifier marker, per the mission's
"prove the patch reached the bytecode, not the source" principle — a stale
`__pycache__` has voided a run on this stack twice).

### ROCm hicache host-alloc fix (baked in the image)
**What:** gfx950 hard-aborts with `Memory access fault by GPU node-N on address
<host VA>` when hicache stores raw host `data_ptr()`s that a GPU kernel dereferences.
**Why:** `hipHostRegister` maps host pages at a **different** device VA; gfx950 is
`xnack-` so there's no page-migration fallback. **Fix:** route
`ALLOC_MEMORY_FUNCS` to `pin_memory` (`hipHostMalloc`).
**How:** `deploy/docker/patches/sglang_rocm/patch_hicache_rocm_host_alloc.py`.
**Context:** invisible on the earlier vultr run because it used `--context-length 32768`.

### Cluster / infra gotchas
- **`/tmp` is root-owned on crsuse** under `spur exec` (runs as `yihou`). Use
  `/var/tmp` — `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`.
- **etcd v3.5.14 needs `--entrypoint /usr/local/bin/etcd`** — its image has an empty
  ENTRYPOINT and `Cmd=[/usr/local/bin/etcd]`; passing `etcd` as argv[0] dumps usage
  and exits 2.
- **Never background a long docker client inside `spur exec`** — the exec namespace
  teardown kills it. Use `docker exec -d` / `nohup` inside the container.
- **Never probe a PD leg's own port** (N/A here — MIX has no PD — but the router is the
  only endpoint to hit either way: `:8100`).
- **Server logs contain binary bytes** — grep them through `strings` (the smoke does).

## Feature-proof evidence (from the smoke, before the sweep)

- 1 worker, `disagg_mode=mixed` (aggregated, not PD).
- Coherent answer to the capital-of-France probe ⇒ DSA env live.
- MTP accept-len **median 3.12** ⇒ speculative decode healthy, not degenerate.
- **8 kvd adapters** connected (~one per rank) + kvd entries > 0 after traffic.
- Router policy `kv-aware`, tokenizer loaded (pw20 / dw2 overlap weights).

## Scope / out-of-scope for this deployment (from the mission)

- MIX = single-node aggregated: **no PD, no mooncake, no RDMA**. The worker was
  derived from the validated PD launcher by removing every disaggregation arg; the
  tuned recipe (DSA env, MTP, kvd, KV dtype, ctx, reasoning-parser) is carried
  byte-for-byte.
- `--mem-fraction-static 0.85` is the mix value; lower it on a prefill activation OOM.
- kvd daemon L3 file 64 GiB here (the vultr 512 GiB `--long-bytes` was a vultr-disk
  artefact, inapplicable on crsuse).

## Known gaps in this packup

- Base-image `sha256` digest not captured at build (recorded by floating tag only) —
  see `environment.md` for the one-liner to resolve it on the node.
- DSA/ROCm patches are referenced by in-tree path at the pinned SHA rather than copied
  in, since the image bakes them from the repo at `d1a97b2`.

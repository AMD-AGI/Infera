> **DATED 2026-09-18 — PREDATES THE RUN OF RECORD.** This file was written by the
> config-authoring role before the 2026-09-19 garbled-decode discovery, and it describes
> `config.full.sh`'s design, not the final configuration. Two things changed afterwards:
> the final `config.yihou.p8d8.sh` **re-adds** `index_share_for_mtp_iteration=false`
> (assigned *after* the source, because `config.full.sh:90` clears it), and custom
> all-reduce stays **ON** with correctness waived rather than being settled by a probe.
> Read `../notes.md` and `../results/sweep_results.yihou.md` for the configuration that
> actually ran.

# Config design — P8D8 vs P4D4 bench configs

Rationale for `scripts/config.yihou.p8d8.sh`, `scripts/config.yihou.p4d4.sh`, and
`scripts/probe_mtp.yihou.sh`. Every claim below is tagged **[first-hand]** (read
in this worktree's source, or verified by sourcing the config) or **[second-hand]**
(taken from a reference packup / another teammate, not re-verified here). This
role launches no container and touches no GPU.

Source of truth read: `/home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd/`
`{config.full.sh,config.sh,engine.sh,launch.sh,tools/topology.py}`.

## Shape

| | P8D8 | P4D4 |
|---|---|---|
| prefill / decode | 137 / 136 | 137 / 136 |
| GPUs per leg | 0-7 (all) | 0,1,2,3 |
| TP / DP | 8 / 8 | 4 / 4 |
| CONC (fast target) | 80 | 40 |
| CONTAINER_PREFIX | glm52-pd-yihou-p8d8 | glm52-pd-yihou-p4d4 |

- **P8D8 TP8/DP8/devices 0-7 equal config.full.sh:29-30,48,50,64,66 defaults**
  **[first-hand]**, but are **restated** in the config so the shape is legible
  without cross-reading and cannot silently change if that default moves. This
  is a deliberate, commented deviation from "override only what differs" for the
  experiment's defining parameter.
- **P4D4 TP4/DP4 are genuine overrides** of the config.full.sh 8/8 default
  **[first-hand]**.
- **GPUs 0,1,2,3 for P4D4** is hwprep's recommended quad (`scripts/rdma_map.yihou.md`):
  single NUMA0 / PCI domain 0002, rails 08/07/05/06, all ACTIVE on both nodes
  **[second-hand: hwprep's first-hand re-measurement]**. 4,5,6,7 is a valid mirror
  quad; switching would require swapping the RDMA map to ionic_4-7.
- **CONC is set in-config with `:=`** so it is overridable for the phase-E sweep.
  config.full.sh:128 defines CONC **[first-hand]**; agentx_bench also takes CONC
  as a required argument, which supersedes the in-config value. The in-config
  value documents the phase-1 fast target only.

## MTP (decode leg only)

- **DECODE_MTP=1, SPEC_STEPS=5, SPEC_TOPK=1, SPEC_DRAFT_TOKENS=6, EAGLE** are the
  config.full.sh:78-81 defaults, inherited unchanged, verified against
  engine.sh:200-206 — `DECODE_MTP` gates the block (`:200`), and the three counts
  map to `--speculative-num-steps` (`:203`), `--speculative-eagle-topk` (`:204`),
  `--speculative-num-draft-tokens` (`:205`), plus `--speculative-algorithm EAGLE`
  (`:202`) **[first-hand]**. Left unset in the config so they inherit; restating
  would only invite drift.

- **Simulated acceptance — the one that is easy to get wrong.**
  config.full.sh:82 assigns `DECODE_SIMULATE_ACC_LEN="${DECODE_SIMULATE_ACC_LEN-3.61}"`
  with **no colon** **[first-hand]**. Semantics of `${VAR-word}`: substitute
  `word` only when VAR is *unset*; a *set-but-empty* VAR stays empty. Therefore:
  - unset ⇒ 3.61 ⇒ simulation ON ⇒ used by the two fast runs.
  - `DECODE_SIMULATE_ACC_LEN=` on the command line ⇒ empty ⇒ engine.sh:149
    `[[ -n "${DECODE_SIMULATE_ACC_LEN:-}" ]]` is false ⇒ the SGLANG_SIMULATE_ACC_*
    env block is skipped ⇒ simulation OFF ⇒ used by the correctness probe.

  **Does `DECODE_SIMULATE_ACC_LEN=` actually work through the bench's KEY=VALUE
  handling? YES — verified first-hand by sourcing.** Trace: engine.sh:24-28 (and
  launch.sh:10-14) validate the arg against `^[A-Za-z_][A-Za-z0-9_]*=.*$` (the
  `.*` matches empty, so `KEY=` passes) and `export` it *before* sourcing the
  config; config.full.sh:82's no-colon default then leaves the set-but-empty value
  empty; engine.sh:149 drops the sim block. Sourcing config.yihou.p8d8.sh with
  `export DECODE_SIMULATE_ACC_LEN=` yielded an empty value **[first-hand]**. The
  config deliberately never assigns `DECODE_SIMULATE_ACC_LEN`; converting it to
  `:=` would clobber the empty override, so it is left untouched.

## `--disable-custom-all-reduce` toggle (OPEN question)

- Whether TP8 needs the GLM-5.2-MTP garbled-decode fix is **unsettled** — the
  five-round single-variable A/B that established it
  (`glm52-mtp-garbled-decode-rootcause.packup_20260918`) was **P4D4/TP4 only**,
  and phase 1 saw coherent TP8/DP8 output at real accept 2.57 **[second-hand:
  reference packup + mission.md:70-74]**. So it is **not** hard-coded either way.
- Delivered as a single switch: `DECODE_NO_CUSTOM_AR` (default unset ⇒ flag OFF).
  When set, the config assigns `DECODE_EXTRA_ARGS=--disable-custom-all-reduce`.
  engine.sh:63 reads `DECODE_EXTRA_ARGS`, engine.sh:227-230 word-splits it onto
  the decode argv **[first-hand]**. Verified by sourcing with
  `DECODE_NO_CUSTOM_AR=1` ⇒ `DECODE_EXTRA_ARGS=--disable-custom-all-reduce`
  **[first-hand]**. The mission notes this needs patch
  `0001-engine-sh-extra-args-env`; the handling is already present in this
  worktree's engine.sh, so no patch step remains **[first-hand]**.
- The **probe** (`probe_mtp.yihou.sh`, simulation off) settles it per shape before
  the timed runs; the leader flips `DECODE_NO_CUSTOM_AR` per shape based on the
  probe's coherence verdict.

## DSA backends — the substitution, and the one second-hand item

- config.full.sh:87-89 requests `DSA_PREFILL_BACKEND=flydsl`,
  `DSA_DECODE_BACKEND=flydsl`, `DSA_TOPK_BACKEND=aiter` **[first-hand]**.
- The config substitutes `tilelang` for both compute backends (set with `:=`
  before the source, so config.full.sh:87-88's `${VAR:-flydsl}` keeps tilelang)
  and **clears `DSA_TOPK_BACKEND` AFTER the source** — because config.full.sh:89
  uses `${DSA_TOPK_BACKEND:-aiter}` and `:-` treats an empty string as unset, so
  clearing it *before* would silently restore "aiter". Cleared post-source ⇒
  engine.sh:181-182 omits `--dsa-topk-backend` ⇒ sglang uses its own default
  (sgl-kernel). Verified by sourcing: `DSA_PREFILL_BACKEND=tilelang`,
  `DSA_TOPK_BACKEND` empty **[first-hand]**.
- **[second-hand — the one item I could not verify first-hand]** *That the pinned
  nightly's argparse rejects `flydsl` and `aiter` but accepts `tilelang` and
  (implicitly) `sgl-kernel`.* Taken from
  `glm52-1p1d-samerail-c32-c40.packup_20260918/scripts/config.yihou.full.sh:29-34`,
  which read `sglang/srt/arg_groups/fields/{exec_,spec}.py` first-hand inside the
  **sibling** image `infera-sglang:v0519-yihou-0917` at the **same** nightly
  `0.5.19.dev20260917+ga9fb1c3238`. Not re-verified against
  `...-nextnfix-hicache` because that requires launching the image, which this
  role does not do. **Risk:** low — same nightly, and tilelang is engine.sh's own
  fallback default (engine.sh:164-165). **To promote to first-hand**, a
  GPU/​container-touching teammate can run, on a node holding the image:
  `docker run --rm <image> python3 -c "import sglang.srt... ; print choices"` for
  `dsa_prefill_backend` / `dsa_topk_backend`, or `--help | grep dsa`.

## HiCache, HSA reclaim, TOPK_V2 — inherited, verified, not re-set

- **Prefill HiCache ON** (config.full.sh:57 `PREFILL_HICACHE=1`), **decode HiCache
  OFF** (config.full.sh:73 `DECODE_HICACHE=0`). engine.sh:76-79 **hard-rejects**
  decode HiCache + MTP (`exit 2`), so #37152 can only help the prefill leg here
  **[first-hand]**.
- **HSA_NO_SCRATCH_RECLAIM role-scoped**: prefill 0 (config.full.sh:56), decode 1
  (config.full.sh:72), selected per role at engine.sh:49/62, forwarded to the
  container at engine.sh:121 **[first-hand]**.
- **SGLANG_OPT_USE_TOPK_V2=false** is the config.full.sh:101 default **[first-hand]**,
  forwarded to the container env at engine.sh:123 (`-e SGLANG_OPT_USE_TOPK_V2=...`).
  Not re-set (mission required). Confirm-in-live-container is a deploy-time check
  (mission §1 / PLAN "Still to be confirmed in the live container env").
- **JSON_MODEL_OVERRIDE_ARGS empty** (config.full.sh:90) ⇒ the
  `index_share_for_mtp_iteration=false` workaround is removed; engine.sh:183-184
  omits `--json-model-override-args` when empty **[first-hand]**. This is a
  config.full.sh design choice inherited as-is; the correctness probe is what
  guards against any resulting garbling.

## Same-rail KV path

Two independent halves, both present:
1. **Router rank affinity** `PD_DP_RANK_AFFINITY=1` (config.full.sh:107 default,
   restated for intent). launch.sh:128-132 translates `1 → true` for the router's
   clap `ArgAction::Set` (which rejects `1`) **[first-hand]**.
2. **Per-GPU HCA pinning** via the `RDMA_DEVICE` JSON map + `MC_TE_FILTERS` comma
   list. Values are hwprep's ready-to-paste blocks **[second-hand: hwprep's
   first-hand re-measurement on 137+136]**: `GPU_n ↔ ionic_n` is NUMA-local, both
   nodes topologically identical, all rails ACTIVE, and same rail-id per index
   across the two nodes ⇒ prefill rank i and decode rank i share a rail, which is
   what `MC_ENABLE_DEST_DEVICE_AFFINITY=1` (config.full.sh:96) needs.
   `MC_GID_INDEX=1` (config.full.sh:94) is the live rail per rdma_map.yihou.md,
   inherited unchanged.
   - The JSON is assigned with an `if [[ -z ]]` test, **not** `${VAR:=...}`: a bare
     `}` inside a `${VAR:=...}` expansion terminates it and truncates the JSON
     (reference warning, reproduced). Verified by sourcing: the 8-entry map
     survives intact **[first-hand]**.

## Sourcing the full target from outside the bench tree

These configs live in the **view workspace**, not inside the bench tree, so the
reference's relative `../../config.full.sh` idiom does not apply. They source
`"$BENCH_DIR/config.full.sh"` with `BENCH_DIR` defaulting to
`/home/yihou/dev/git/infera.yihou.glm52.p8p4/bench/glm5p2_pd` and overridable
**[first-hand]**. **Deploy dependency:** `config.full.sh` (and the bench scripts)
must exist at `BENCH_DIR` on whichever node sources the config — launch.sh:101-105
runs engine.sh on the *remote* node with `CONFIG=<abs path>`, so the config file
and its `BENCH_DIR` target must both be present there. This is a deploy/sync
concern outside config authoring; flagged for the leader.

## Probe (`probe_mtp.yihou.sh`)

- Sends **one fixed prompt N=16 times at temperature=0**, concurrently (so decode
  is under load), prints every completion, and computes a coherence verdict:
  healthy &rArr; all 16 contain the first ten primes IN ORDER, none empty, no degenerate
  repeated-token tail. NOT byte-identity: a healthy temperature-0 run on this stack
  produced 16 DIFFERENT strings (DP attention + batching reduce in different orders),
  so "distinct completions > 1" is not evidence of garbling. Corrected 2026-09-18.
  the GLM-5.2-MTP garble signature is variation or a degenerate tail. Embedded
  Python validated against fixtures **[first-hand]**.
- Reads **per-DP-rank `spec_accept_length` and `spec_accept_rate`** from the decode
  engine's `/metrics` — snapshots idle (the trap: an idle rank reads 0), scrapes
  **mid-flight while the 16 requests run** (the measurement), and scrapes once more
  after completion as a fallback (the gauges are running averages). Reports only
  ranks whose counters moved; "no active rank" prints INCONCLUSIVE, never a zero.
  Metric-parsing keys on the whole label set, so it is robust to the DP-rank label
  name (`dp_rank` / `engine` / …). Validated against fixtures **[first-hand]**.
- **Addresses [first-hand from topology.yihou.tsv + topology.py + config ports]:**
  router `http://10.245.153.247:28000` (137, ROUTER_PORT 28000); decode
  `10.245.154.168:29002` (136, ENGINE_PORT_BASE 29001 + decode's topology row
  index 1). topology.py sets index = line-2, so prefill=0/decode=1. All
  overridable via env.
- **The probe deployment MUST be launched with `DECODE_SIMULATE_ACC_LEN=`**
  (simulation off) — simulated acceptance masks garbling and inflates the gauge.
  The coherence verdict, not the acceptance number, decides the all-reduce flag.

## Open dependencies / second-hand summary

1. **DSA argparse choices** — second-hand from the sibling image at the same
   nightly; promotion path given above. The only correctness-affecting item not
   verified first-hand here.
2. **RDMA map values / P4D4 GPU quad** — second-hand from hwprep's first-hand
   re-measurement (`scripts/rdma_map.yihou.md`); now confirmed, no longer an open
   dependency. (It was an open dependency at first authoring; hwprep delivered
   the map and the config values match it verbatim.)
3. **Deploy-time confirmations (leader/deploy-config):** `config.full.sh` present
   at `BENCH_DIR` on the remote nodes; `SGLANG_OPT_USE_TOPK_V2=false` reaching the
   live container env; `MODEL` path (config.full.sh:15,
   `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`) mounted on 137/136.

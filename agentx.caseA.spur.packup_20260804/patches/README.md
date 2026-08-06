# Patches and fixes

No source patch was authored for this run. Every engine-side fix this
deployment depends on is **baked into the image** by the branch's Dockerfile and
verified in compiled bytecode by `scripts/start_ctr.sh` before the legs start.
This section records what those fixes are, how they were verified here, and the
one operational fix that *was* needed (which is env, not code).

---

## 1. Engine fixes — baked into the image, verified in BYTECODE

`scripts/start_ctr.sh` runs a 9-assertion gate against **freshly compiled
bytecode**, never the `.py` source. Reading source proves the file was edited;
reading the `.pyc` proves the interpreter will run it. A stale `__pycache__` has
silently reverted a patch and invalidated a full experiment in this tree twice.

Gate output on both nodes for this run:

```
ROCm hicache host alloc     GLM52_ROCM_HOST_ALLOC   pyc_hits=1  OK
mooncake early-send         wait_event              pyc_hits=2  OK
mooncake early-send (utils) wait_event              pyc_hits=1  OK
mooncake early-send (prefill) wait_event            pyc_hits=1  OK
ALLOC_MEMORY_FUNCS dispatch is_hip=True -> alloc_with_pin_memory  OK
DSA p1: hip dp rows         _p1v2_trim              pyc_hits=1  OK
DSA p1v3: reversed padding  _p1v2_rows              pyc_hits=1  OK
DSA p2b: page-table rows    _glm52_match_page_table_rows  pyc_hits=1  OK
DSA p3: draft-graph DP vote requires_dp_attention_eager_forward  pyc_hits=2  OK
BYTECODE_GATE OK
```

| fix | what | why needed | context (symptom cured) |
|---|---|---|---|
| **GLM52_ROCM_HOST_ALLOC** | routes `ALLOC_MEMORY_FUNCS["cuda"]` to `alloc_with_pin_memory` on HIP | on ROCm, `hipHostRegister` maps host pages at a *different* device address than the host VA; the hicache pools hand raw host `data_ptr()`s to GPU kernels via device-side pointer tables | `Memory access fault by GPU node-N on address <host VA>` with kvd + long prompts on gfx950 (`xnack-`, no page-migration fallback) |
| **mooncake early-send** | overlaps KV transfer with the suffix forward | throughput | — |
| **DSA p1 / p1v3** | trims the aiter paged-MQA rows under DP padding, **both directions** | p1v2 handled only `real < padded`; on an IDLE rank under MTP draft-extend the inequality inverts | `Expected lengths.size(0) == B` crash, 2–13 min into a run |
| **DSA p2b** | page-table row matching | DPA + PD | — |
| **DSA p3** | DP-group vote on the draft-graph/eager decision | a rank-divergent decision deadlocks the group | hang under MTP + DPA |

### The `merged-mtp` vs `final-pr` difference — caught by the gate

The first node pair carried only `infera/engine-sglang:merged-mtp`. The gate
reported:

```
DSA p1v3: reversed padding  _p1v2_rows  pyc_hits=0  MISSING
BYTECODE_GATE FAILED(1)
```

That image needs `apply_p1v3.py` applied inside the decode container (kept in
`scripts/` for that path). The procedure is:

```bash
docker exec <ctr> md5sum .../dsa/dsa_indexer.py     # expect 632f17acd38737459b43f830ee60ee89
docker cp apply_p1v3.py <ctr>:/tmp/ && docker exec <ctr> python3 /tmp/apply_p1v3.py
docker exec <ctr> rm -f .../dsa/__pycache__/dsa_indexer*.pyc   # THEN relaunch the leg
```

**The `.pyc` deletion and relaunch are load-bearing.** A running engine has
already imported the old bytecode; the patch script printing `patched OK` proves
nothing about what is executing.

This was done and verified (`_p1v2_rows=1` in a freshly compiled `.pyc`) on the
first node pair. The **final** run did not need it — `final-pr`, rebuilt for
this experiment, has P1V3 baked in and passes `BYTECODE_GATE OK`.

---

## 2. The one fix authored here — `OUT` must be inside the mounted path

**Not a code patch. An environment fix, applied without editing the customer's
script** (`scripts/run_caseA.sh`).

- **What.** `OUT=$HERE/results`, where `$HERE` is the bench directory.
- **Why.** The customer's `replay_caseA.sh` mounts only `$HERE` into the aiperf
  container. An `--output-artifact-dir` outside it resolves inside the
  *container namespace*.
- **How it manifests.** The CSV dies with the container; the host-side
  `[ -f "$csv" ]` test fails; `summary.csv` records **`FAILED`** — for a run that
  completed and produced a full metrics table in its console output. The failure
  is silent and looks like a benchmark failure.
- **Context.** Observed on a prior vultr attempt where both concurrency points
  recorded `FAILED` despite completing. With `OUT` inside `$HERE`, both points
  wrote clean rows here.

---

## 3. Deployment parameters that are NOT patches but are load-bearing

| setting | value | why it is not free |
|---|---|---|
| prefill `--chunked-prefill-size` | 65536 | sglang divides this by `dp_size` **only** when DP-attention is on (`server_args.py:4902`). With DPA off the engine resolves it to 65536 — 8× the per-forward batch of a dp8 leg given the same CLI number. |
| prefill `--mem-fraction-static` | 0.70 | pairs with the above: the larger per-forward batch needs more activation headroom. Prefill activation OOM is fixed by *lowering* this, the opposite direction from the decode-side retract fix. |
| `--disaggregation-ib-device` | `mlx5_0` | spur's KV fabric. The 8 `ionic_*` devices also present on these nodes are the **vultr** fabric; binding them here silently falls back to TCP. |

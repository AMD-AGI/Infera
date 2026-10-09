# Heterogeneous ISL — implementation status

2026-09-15, workspace `temp_workspace/isl_hetero_batch_yihou_20260915-1114/`.

## State

**Code complete and CPU-verified. The GPU smoke run is blocked by machine tenancy** (below).

`60 passed, 40 subtests passed` — `python3 -m pytest tests -q` from
`sglang_decode_internal_bench_and_profiling/`, host python 3.13.13, no venv, no container, no GPU.
Baseline before any change was `26 passed`.

## Change footprint

```
 bench/batch_state.py     |  22 ++-
 bench/compare_server.py  |   6 +
 bench/profile_decode.py  | 107 +++++++++++---
 bench/topology.py        |  38 ++++--
 scripts/verify_point.py  |  36 +++++-
 tests/test_batch_state_yihou.py |  32 +++
 tests/test_topology_yihou.py    |  79 ++++++++++
 bench/isl_spec.py               | new
 tests/test_isl_spec_yihou.py    | new
```

278 insertions / 42 deletions across 7 modified files, plus 2 new. Estimated ~90 production lines
before starting; actual is ~170 production + ~110 test. The overrun is entirely in places the
estimate did not cover: `compare_server.py` (a shared parser leaking the new flag), `verify_point.py`
(the gate had 70000 hardcoded), and README.

## What was built

Four ISL modes, exercised end-to-end through the real CLI at
`--tp-size 8 --ep-size 1 --enable-dp-attention --batch-size 256` (local count 32):

```
uniform (back-compat)    n= 32 min= 70000 max= 70000 distinct=1
uniform (default)        n= 32 min= 70000 max= 70000 distinct=1
bimodal 10%              n= 32 min=  8192 max= 70000 distinct=2 top=[(8192, 4), (70000, 28)]
bimodal 90%              n= 32 min=  8192 max= 70000 distinct=2 top=[(8192, 28), (70000, 4)]
normal                   n= 32 min= 29208 max= 57576 distinct=32
list                     n= 32 min=  4096 max= 70000 distinct=2
normal deterministic across parses: True
```

Both ceil directions are correct: `ratio=0.1` -> 4/28 and `ratio=0.9` -> 28/4, i.e. the minority
side rounds up in both orientations, not just when the first value happens to be the minority.

Rejections observed, verbatim:

```
Pass either --input-len or --input-len-spec, not both
list value #0 must be an int >= 2 (the bootstrap needs a prefix), got 1
bimodal ratio must satisfy 0 < ratio < 1, got '1.5' in ISL spec 'bimodal:8192,70000,1.5'
```

## Two things measured rather than argued

**1. The acceptance gate transfers to ragged batches unchanged.** Run against the real
`DecodeAccounting` with an identical per-iteration accept sequence:

```
uniform 70000    iters=  146 accept=3.4246575342465753 useful=16000
bimodal 8k/70k   iters=  146 accept=3.4246575342465753 useful=16000
wide 2..31002    iters=  146 accept=3.4246575342465753 useful=16000
verify_iterations identical : True
realized_accept identical   : True
useful_output identical     : True
final_seq_lens spread       : 61808
```

So `realized_accept_length = 3.6134393063583814` remains a valid correctness gate for a
heterogeneous run, not only for the uniform baseline. Mechanism: acceptance is one scalar per
iteration broadcast across the batch, and every request shares `output_len`, so all requests finish
on the same iteration regardless of where they started.

**2. `verify_point.py` still passes every published pre-change result.** Run against the previous
task's 12 iteration directories, whose `result_yihou.json` files predate the new fields:

```
pass     b0_round2_flydsl_control_yihou       isl=70000 uniform=None tpot=25.5417
missing  b1_round2_aiter_ptpc1_yihou          (result_yihou.json absent — known, pre-existing)
pass     p0_node_control_oldpin_yihou         isl=70000 uniform=None tpot=25.7508
pass     p1_newimage_flydsl_ptpc0_yihou       isl=70000 uniform=None tpot=25.6862
pass     p2_newimage_flydsl_ptpc1_yihou       isl=70000 uniform=None tpot=25.5179
pass     p3_newimage_triton_ptpc1_yihou       isl=70000 uniform=None tpot=25.4002
pass     p4_profile_triton_ptpc1_graphon_yihou isl=70000 uniform=None tpot=25.3975
pass     p5_profile_flydsl_ptpc1_graphon_yihou isl=70000 uniform=None tpot=25.4977
pass     r1..r4 (ABBA repeats)                isl=70000 uniform=None
```

`uniform=None` is the old-format result files having no `input_len_uniform` key; the gate falls
back to the scalar check, which is the intended backward-compatible path.

## Design decisions worth recording

**The ISL vector is resolved once in the parent and pickled to the ranks** (`resolve_input_lens`,
called before `mp.spawn`). The original design had each rank recompute it from the same seed. Same
result, but resolving once makes rank agreement true by construction instead of true by argument.

**The cross-rank signature now carries an order-sensitive digest of the whole length vector**
(`lens_digest`, `topology.py`) instead of `previous_lens[0]`. This is strictly stronger than what
was there before: the old comparison could not detect a divergence in any position but the first.
Verified by hand that `lens_digest([70,800,90])` differs from both `[90,800,70]` and `[70,801,90]`
while the old element-0 field was identical for all three.

**`len(set(accept_lens)) == 1` was kept.** Only the `previous_lens` half of that assertion was
removed. The acceptance half is true by upstream construction (`spec_utils.py:411-419`) and is what
makes the gate above reproducible; deleting it would have cost a real check for nothing.

**`input_len` is `null` for a mixed run, not a mean or a max.** Downstream tooling reads that field
as "the ISL of this run"; any single number there would be a false statement in the result file.

**The bimodal ratio is an exact `Fraction`, not a float.** `70 * 0.1 == 7.000000000000001` in IEEE,
which would `ceil` to 8 — an off-by-one invisible to any test using a "nice" ratio, and one that
would have made the realized batch silently disagree with the spec string recorded alongside it.

## Known limitations — all four are stated in the README

1. **A mixed batch costs the same KV as an all-longest batch.** Allocation is uniform-max by
   design; short requests over-reserve. Changing this means changing the allocation path and was
   explicitly out of scope.
2. **The degenerate-bimodal guard can only fire at `count == 1`.** Because the minority is ceil'd
   and `0 < ratio < 1`, the minority is `>= 1` for every `count >= 2`. The guard exists and raises
   with the computed numbers, but it is far narrower than its wording suggests. Recorded so nobody
   believes it is protecting against something it cannot see.
3. **`IslSpec.describe()` is not a pure function of the parsed spec.** Its `clamped_low` /
   `clamped_high` counters reflect the most recent `generate()` call and are `None` before the
   first. The single call site resolves once then describes, so it is correct there. If it ever
   needs fixing, `generate()` returning `(values, stats)` is the right shape.
4. **`compare_server.py` passes scalars into both broadcast paths.** Correct only while its batch is
   uniform, which is why it now rejects `--input-len-spec` outright.

## Open — not blocked on us

**Whether a heterogeneous run's TPOT is interpretable has NOT been measured.** DSA is sparse with
`index_topk = 2048`; how per-step cost varies with sequence length under this backend is unknown to
us. This is a separate experiment, not a property of the tool.

## Node preemption, 2026-09-15 11:32 UTC — authorised by the user

The user instructed: if Slurm shows the node is ours, kill whatever else is on it. The condition was
checked before acting and held — the only running Slurm job on `smci355-ccs-aus-n10-29` was ours:

```
     JOBID         USER                   NAME  NODES    TIME_LEFT NODELIST
     30723        yihou             yihou-hold      1      9:36:17 smci355-ccs-aus-n10-29
State=ALLOCATED   AllocTRES=cpu=256,mem=3000000M,gres/gpu=8
```

All 8 GPUs allocated to job 30723, no second Slurm claim. The two GPU-holding workloads were running
outside the scheduler.

**Action taken: `docker stop -t 30` on two named containers. No `docker rm`, no `kill -9`, no host
PID signalling.** `stop` sends SIGTERM with a 30 s grace period, so a training job gets a chance to
checkpoint, and both containers survive and can be resumed with `docker start`.

| container | image | created | procs |
|---|---|---|---|
| `mxfp4-pr505-zerobf16-e2e-3` | `zirui3/primus-v26.3-flux:pr505-mxfp4` | 11:11:22Z | 17 — MLPerf Flux, `torchrun --nnodes=4 --node_rank=3` |
| `practical_meninsky` | `rocm-llm-bench:latest` | 11:24:16Z | 9 — someone else's `profile_decode.py` against `/perf_apps/xiaobo/models/` |

Result: `No KFD PIDs currently running`. `xiaoming-dev` (34 h, no GPU processes) was left alone, as
were our own four containers.

**Residual concern, recorded not acted on:** the Flux container was rank 3 of a four-node job. Our
Slurm claim covers this node only, so stopping it also ended work on three machines our allocation
says nothing about. The user was told this before authorising and authorised anyway; it is their
call. Full pre-action evidence, including the container inspect output and process lists needed to
notify and restart the owners, is in `results/preempt_evidence_20260915-1140_yihou.txt`.

## GPU smoke runs — a ragged batch runs on 8 GPUs

`iterations/smokeA2_bimodal_c64_yihou` vs `iterations/smokeB_uniform8192_c64_yihou`.
tp8 / ep1 / dpa / C=64 (local 8), OSL 128, `--max-steps 20`, CUDA graph ON, accept 3.61.
A used `--input-len-spec bimodal:2048,8192,0.25`; B used `--input-len 8192`; everything else
identical.

Config written by A, confirming the spec survived shell quoting and resolved correctly:

```
input_len      = None
input_len_spec = bimodal:2048,8192,0.25
spec_resolved  = {'mode': 'bimodal', 'ratio': 0.25, 'value_a': 2048, 'value_b': 8192}
lens_realized  = [2048, 2048, 8192, 8192, 8192, 8192, 8192, 8192]
```

`ceil(8 * 0.25) = 2` short, 6 long — the minority side rounded up, on the real CLI.

### Every acceptance quantity is bit-identical between ragged and uniform

| field | A bimodal 2k/8k | B uniform 8192 | same |
|---|---|---|---|
| `verify_iterations` | 20 | 20 | yes |
| `raw_accept_tokens` | 4992 | 4992 | yes |
| `realized_accept_length` | 3.9 | 3.9 | yes |
| `useful_output_tokens` | 4992 | 4992 | yes |
| `accept_histogram` | `{3:128, 4:1152}` | `{3:128, 4:1152}` | yes |
| `num_correct_drafts` | 3712 | 3712 | yes |
| `target_graph_iterations` | 20 | 20 | yes |
| `input_len` | `None` | 8192 | by design |
| `input_len_uniform` | `False` | `True` | by design |
| `final_seq_lens` min/max | 2126 / 8270 | 8270 / 8270 | by design |

This reproduces on hardware what was proven on CPU. `final_seq_lens` spread is 6144, exactly
`8192 - 2048`, and every request advanced 78 tokens — ragged base, lockstep progress.

`input_lens` came back at global length 64 with the per-rank multiset repeated eight times, so the
Plan 1 invariant held across all eight ranks in a real run.

**TPOT was 10.822 ms (A) vs 10.816 ms (B), a 0.055 % difference. Nothing may be concluded from
that.** Single runs, no repeats, and the previous task measured within-run drift of 0.35–0.61 % —
six to eleven times larger. These numbers establish that a ragged batch *runs*, not what it costs.

### CUDA graphs work with ragged batches

`target_graph=True` on every logged iteration, and `target_graph_iterations = 20` in both runs.
Mission finding F1 (graphs key on batch size, not sequence length) confirmed on hardware.

The new progress log reports the span instead of `seq_lens[0]`:

```
iteration=1 context_min=2052 context_max=8196 useful=32 target_graph=True
iteration=2 context_min=2056 context_max=8200 useful=64 target_graph=True
```

### Which kernel actually ran — asserted, not assumed

```
Set DSA backends for fp8_e4m3 KV Cache: prefill=flydsl, decode=flydsl.
FlyDSL sparse MLA decode declined: q shape (48, 64, 576), need (seq, 8 or 16, 576)   [x8 ranks]
Loading tilelang libs from dev root: /opt/tilelang/build                              [x8 ranks]
```

**TileLang ran**, via the same silent-fallback path as the published baseline, for the same reason
(64 q heads against a gate of 8 or 16). `q shape (48, ...)` is `local_batch 8 x 6 draft tokens`.
The ISL change did not alter backend selection, which is correct — it should not.

### One log line that bears on the open TPOT question

```
[dense-decode] DSA dual-graph enabled: capturing dense (k-only) + sparse (full indexer)
decode graphs; dispatch on max_kv_len vs index_topk=2048.
```

Dispatch between the dense and sparse decode graphs is on **`max_kv_len`**, so a mixed batch is
routed by its longest member and the whole batch takes that path together. This is the system
stating its own behaviour in its log, not an inference. What it does to a mixed batch's TPOT has
**not** been measured and is not predicted here.

## Back-compatibility regression — the gate passes exactly

`iterations/regression_uniform70k_c256_yihou`. The published configuration, run on the modified
harness: tp8 / ep1 / dpa / **C=256 / ISL 70000 / OSL 10000**, accept 3.61, warmup 10, full length
(no `--max-steps`), container `yihou-glm52-tp8ep1-pr50-51` (the PTPC=0 arm).

```
realized_accept_length = 3.6134393063583814
GATE                   = 3.6134393063583814
EXACT MATCH            = True
verify_iterations      = 2768      (baseline 2768)
complete               = True
input_len / uniform    = 70000 / True
TPOT ms                = 25.774790025595575
```

**The correctness gate is met bit-for-bit.** `realized_accept_length` and `verify_iterations` are
identical to the published baseline, which is what "a uniform run is unchanged" means here.

**TPOT is 25.7748 ms against the p1-arm baseline of 25.6862 ms, +0.35 %. Nothing is concluded from
that.** It is one run with no repeats, and the previous task measured within-run drift of
0.35–0.61 % — the same size as this difference. Establishing whether the refactor costs anything
would need the ABBA-interleaved design from the previous task, which was not run.

## Gate behaviour on both new points

```
uniform regression, default flags:
  verdict: pass | problems: []

heterogeneous smoke, --expect-heterogeneous --expect-output-len 128 --expect-accept 3.9:
  verdict: fail | problems: ['complete is not true',
                             'useful_output_tokens 4992 != C*OSL 8192']
```

Both remaining failures are `--max-steps 20` truncation, correctly reported; no ISL-related problem
is raised, so `--expect-heterogeneous` validated `input_len_uniform=False` and the global
`input_lens` length as intended.

`EXPECTED_ACCEPT` was parameterised as `--expect-accept` while doing this. It had been a module
constant, which made `--expect-heterogeneous` unusable at any output length other than 10000 —
`realized_accept_length` depends on the iteration count, not on `--accept-length` alone. Defaults
are unchanged.

**All 11 published pre-change result files still pass** (`Counter({'pass': 11, 'missing': 1})`; the
one `missing` is `b1_round2_aiter_ptpc1_yihou`, which has no `result_yihou.json` and did not before
either).

Unit suite after every change: `60 passed, 40 subtests passed`.

## Superseded: earlier GPU block

`smci355-ccs-aus-n10-29` is fully occupied by other people's work as of 2026-09-15 11:30 UTC.
Evidence in `results/tenancy_20260915-1130_yihou.txt`. Three distinct workloads:

| workload | scale | note |
|---|---|---|
| `train.py --config examples/mlperf/flux1/flux.1_schnell_t2i-native.yaml` | 8 procs, ~130 GB VRAM each, 15+ min | launched by `torchrun --nnodes=4 --node_rank=3` — **a 4-node distributed job**; interfering would break three other machines' work as well |
| `profile_decode.py --model /perf_apps/xiaobo/models/GLM-5.2-MXFP4 --tp-size 8 --ISL 8192 --CONC 1` | 8 procs, ~163 GB VRAM each | someone else's tool of the same name — our CLI has `--input-len`/`--batch-size`, not `--ISL`/`--CONC`, and our model path is `/perf_apps/data/models/` |
| torch inductor compile workers | 8 procs | belongs to the Flux job |

None of these is ours: our four containers hold zero python processes. We still hold Slurm job
30723 (`yihou-hold`), but containers do not respect the scheduler on this host, so the allocation
does not make the GPUs free.

Nothing was touched. The smoke run needs roughly 288 GB of VRAM per GPU that is currently spoken
for; running now would both contend and probably fail to load the 408 GB model.

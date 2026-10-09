# Heterogeneous input lengths in the internal SGLang decode harness

**2026-09-15.** Adds the ability to bench a batch whose requests have **different ISLs**, four ways
to specify the mix, and the evidence that a uniform run is unchanged.

## What this is

Before this change `--input-len` was a single integer and batch-uniformity was **asserted every
iteration** — `topology.py:69` demanded `len(set(previous_lens)) == 1`, so a ragged batch was not
merely unsupported, it was actively rejected on the first step. This packup contains the change
that removes that restriction without giving up what the assertion was protecting.

## Result at a glance

| claim | evidence |
|---|---|
| A ragged batch runs on 8 GPUs with CUDA graph ON | `results/points/smokeA2_bimodal_c64_yihou/` |
| A uniform run is unchanged, bit-for-bit | `realized_accept_length = 3.6134393063583814`, `verify_iterations = 2768` — identical to the published baseline. `results/points/regression_uniform70k_c256_yihou/` |
| Acceptance accounting is ISL-independent | every acceptance field identical between a ragged run and its uniform control, on hardware |
| The kernel that actually ran was TileLang | `FlyDSL sparse MLA decode declined` on all 8 ranks, same as the published baseline |
| All 11 published pre-change results still pass the gate | `verify_point.py` re-run against the previous task's iterations |
| Unit suite | `60 passed, 40 subtests passed` (was `26 passed` before) |

**No TPOT conclusion is drawn anywhere in this packup.** Every performance number here comes from a
single run without repeats, and the previous task measured within-run drift of 0.35–0.61 %, which
is larger than any difference observed. These runs establish that ragged batches *work*, not what
they cost.

## The four modes

```
--input-len 70000                             # unchanged, still works
--input-len-spec uniform:70000                # same thing, explicit
--input-len-spec bimodal:8192,70000,0.25      # 25 % short; the MINORITY side rounds UP
--input-len-spec normal:40000,8000,2,70000    # gaussian, rounded, clamped by saturation
--input-len-spec list:4096,4096,70000,...     # verbatim; list:@path reads one int per line
```

**The spec describes one attention-DP rank, not the global batch.** At `--batch-size 256` with
dp=8 it must fill `local_batch_size = 32`. Every rank then gets that same multiset in the same
order, resolved once in the parent and shipped to the ranks.

## Where to start

| you want | read |
|---|---|
| to reproduce any number here | `REPRODUCE.md` |
| what was decided and why | `design_spec_yihou.md`, then `results/implementation_status_yihou.md` |
| the traps, and the two mistakes made along the way | `notes.md` |
| the code | `sources/` (final files) and `patches/modified_files_yihou.diff` |
| the original task definition | `mission_book_yihou.md` |

## Scope, stated plainly

**Plan 1 was implemented; Plan 2 was considered and rejected.** Plan 1 gives every DP rank the same
ISL multiset. Plan 2 would have split one global distribution unevenly across ranks, which is more
general — and would have required deleting the cross-rank determinism check, this harness's
strongest invariant. The cost of Plan 1 is granularity: with dp=8 the finest expressible unit is
one request per rank, i.e. 8 of a 64-request global batch. That is a real limitation, not a detail.

## Known limitations

1. **A mixed batch costs the same KV as an all-longest batch.** Allocation is uniform-max by
   design; short requests over-reserve. Heterogeneity buys no capacity headroom.
2. **The degenerate-bimodal guard can only fire at `count == 1`.** It exists and raises with the
   computed numbers, but is far narrower than its wording suggests.
3. **`IslSpec.describe()` is not pure.** Its clamp counters reflect the most recent `generate()`.
4. **`compare_server.py` does not support ragged batches** and rejects `--input-len-spec` rather
   than silently measuring a uniform one.

## The open question

**Whether a heterogeneous run's TPOT is interpretable has not been measured.** One relevant fact
was read from the runtime log and is recorded without interpretation:

```
[dense-decode] DSA dual-graph enabled: capturing dense (k-only) + sparse (full indexer)
decode graphs; dispatch on max_kv_len vs index_topk=2048.
```

Dispatch between the dense and sparse decode graphs is on `max_kv_len`, so a mixed batch is routed
by its longest member and the whole batch takes that path together. What that does to TPOT is a
separate experiment.

## The C=64 300K comparison — data included, conclusion explicitly NOT drawn

`results/points/c64_all70k_control_yihou/` vs `results/points/c64_one300k_yihou/`. Same image, same
session, C=64 / EP1 / dpa, OSL 10000, full length. The ragged arm used
`list:300000,70000,70000,70000,70000,70000,70000,70000` — one 300K request **per rank**, which under
Plan 1 is the finest expressible granularity (8 of the 64 global requests).

| | all-70K | 1x300K + 7x70K per rank |
|---|---|---|
| TPOT ms | 12.5537 | **12.0989** |
| `verify_iterations` | 2768 | 2768 |
| `realized_accept_length` | 3.6134393063583814 | 3.6134393063583814 |
| `useful_output_tokens` | 640000 | 640000 |
| `reserved_tokens` | 640512 | 2480128 |
| peak allocated | 173.5 GB | 226.4 GB |
| `final_seq_lens` min/max | 80000 / 80000 | 80000 / 310000 |

The ragged arm came out **3.62 % faster**, which is both counterintuitive and larger than the
0.35–0.61 % within-run drift measured in the previous task.

**No conclusion is drawn from this and none should be.** It is one run per arm, and run order is
completely confounded with arm — the control ran first and the 300K arm second, so thermal state
differs exactly along the contrast being measured. The previous task built an ABBA-interleaved
design specifically because a difference this size can be manufactured by ordering alone. Those
repeats have **not** been run, and profiling was deliberately not started: profiling an
unreplicated effect risks explaining a phantom.

What the pair *does* establish, because these are exact: both arms completed 2768 iterations with
a bit-identical acceptance gate, so the comparison is between two valid runs of the same work.

Note `reserved_tokens` is ~3.9x higher in the ragged arm for a single long request per rank — that
is the uniform-max allocation (limitation 1 below), not a property of the workload.

## Not in this packup

Profiling of the 300K comparison, the ABBA repeats that would make its bench delta interpretable,
and the 25/50/75/100 % 300K sweep requested afterwards.

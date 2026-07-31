# Merging kvaware+kvd, PD+DPA+MTP, and the mooncake early-send fix

**Ran:** 2026-07-31, 12:22–13:40 UTC
**Author:** yihou
**Nodes:** `chi2879` (prefill) + `chi2867` (decode), 8× MI355X gfx950 each, ionic RoCE
**Model:** GLM-5.2-MXFP4, 2-node PD over mooncake RDMA
**Status:** **PASS** — all four gates met.

## Goal

Three workstreams had been validated separately and never together:

1. **kvaware + kvd** — KV-aware routing plus tiered KV offload (`kvaware_kvd_pr.packup_20260731.pr.final`)
2. **PD + DP-attention + EAGLE MTP** — the 3 sglang DSA patches (PR58, `glm52.mxfp4.spur.mooncake.packup_20260731_main_converged`)
3. **PD + mooncake long-prompt corruption** — the early-send wait event (PR56)

This run merges all three into one configuration and validates it end to end.
**Spec:** `spec.md` (the mission file, copied verbatim).

The three do not conflict — they touch almost disjoint files. What the merge
exposed is different: **pre-existing infera code meeting a configuration nobody
had run**, because kvaware/kvd was never validated with MTP on and the MTP work
drove `sglang.launch_server` directly, bypassing the infera wrapper.

## Success criteria and result

Four gates, each loading one more thing onto the last, so a failure localises.

| gate | criterion | actual | verdict |
|---|---|---|---|
| **G0** | patches must not break kvaware+kvd: 4/4 correctness, kvd restart-replay shows `gets>0`, `hits>0`, `sets` unchanged | 4/4, 32/32 prefix reuse, kvd **102 gets / 102 hits / 102 sets (unchanged)** after engine restart | ✅ |
| **G1** | + MTP on decode: 4/4, `acc_len>1`, router KV view non-empty | 4/4, 32/32, **`accept len` 2.48–2.58**, router prefill view **51 blocks** | ✅ |
| **G2** | + prompt spanning >1 prefill chunk: needle correct at every depth | **5/5**, prompt really split `8192+8192+1728` | ✅ |
| **stress** | conc=16 then conc=128, 0 HTTP errors | **64/64** and **256/256**, 0 corrupt, 0 errors | ✅ |

`Traceback` on either leg across all gates: **0**.

### Against the spec's four goals

`spec.md` asks for four things. Two were deferred by the operator during the
session, in favour of getting the experiment green first:

| # | spec goal | status |
|---|---|---|
| 1 | new worktree + unified experiment branch | **deferred** — "先实验修复，不谈分支pr和提交" |
| 2 | deliver the final Dockerfile | **done** — `deliverable/deploy/docker/Dockerfile.sglang` (**not built**, see below) |
| 3 | deliver a built image on the cluster | **deferred** — "基于 vultr 集群已有的 kv-aware 镜像快速 patch 验证先"; the run used in-container patching |
| 4.1 | verify kvd + kvaware still work | **done** — G0 |
| 4.2 | mtp+dpa+pd / pd+mooncake correct; single-request, conc=16, conc=128 | **done** — G1, G2, stress |
| 4.3 | packup via the packup skill | **done** — this kit |

Goals 1 and 3 remain open work, not failures.

### The two numbers that actually discriminate

A green run proves little on its own; these two are the ones that would have gone
red if a fix were absent or wrong.

**kvd is serving, not merely wired.** A speed-up proves nothing — SGLang's in-GPU
radix cache serves a repeated prefix without touching L3. Restarting the prefill
engine empties that cache while the kvd daemon and its L3 keep running:

| | gets | hits | sets | misses |
|---|---:|---:|---:|---:|
| after first reuse run | 0 | 0 | 102 | 0 |
| **after restart + replay** | **102** | **102** | **102 (unchanged)** | 0 |

102 reads, zero new writes, zero misses — reuse that could only have come from L3.

**The bigram fix works, and produces the *right* hashes.** `is_eagle` is a global
server arg, so with MTP on the prefill leg's kv-events carry bigram pairs
`(t[i], t[i+1])`. Unfixed, the router hashes the pairs and the view reads **0**.
It reads **51 blocks — byte-identical to the plain-int path in G0**, which shows
the flattened keys chain to the same hashes rather than merely being non-empty.

## The patch set: 5 in, 7 out

| # | patch | from | fixes |
|---|---|---|---|
| 1 | `dsa_indexer_hip_dp_padded_rows.diff` | PR58 | HIP paged-MQA DP-padded rows |
| 2 | `dsa_backend_dp_sync_and_page_table_rows.diff` | PR58 | DP host-sync + MTP page-table rows |
| 3 | `draft_cuda_graph_dp_vote.diff` | PR58 | per-rank draft graph/eager divergence → deadlock |
| 4 | `patch_mooncake_early_send_wait_event.py` | PR56 | non-final prefill chunk RDMA-read races the forward |
| 5 | `patch_infera_kvevent_bigram.py` | PR56, re-cut as a self-locating script | kv-aware view empty under MTP |
| 6 | `patch_infera_decode_radix_vs_mtp.py` | **written during this run** | kvaware's decode-radix append is rejected under EAGLE |
| 7 | `patch_infera_decode_kvd_skip.py` | **written during this run** | kvd is write-only on a PD decode leg |

6 and 7 are the merge-exposed defects. They are independent — different flags,
different gates, different SGLang checks — so neither subsumes the other.

Prerequisite for 1–3: the GLM-5.2 nextn `eh_proj` quark-exclude, already in the
base image; the apply script **asserts** it rather than assuming.

## A finding worth reading on its own

**kvd on a PD decode leg is write-only, in every configuration.** SGLang's
`_prefetch_kvcache` is the sole caller of `prefetch_from_storage`, and
`_add_request_to_queue` only invokes it on the NULL and PREFILL branches — the
DECODE branch has no equivalent. The backup path still runs, so L3 fills and is
never read. Measured: prefill 102 sets / **102 gets**, decode 180 sets / **0
gets**, 318 MB of host memory for zero reads. Not MTP-specific.

Full analysis, including the open question it leaves: `notes.md` §5.

## What this kit does NOT establish

- **The merged image was never built.** Everything was validated by patching the
  running containers, with each patch verified in the *bytecode* on both nodes.
  `deliverable/deploy/docker/Dockerfile.sglang` applies the same scripts in the
  same order, but that is an argument, not a measurement. **Build it and re-run
  G0–G2 before shipping.**
- **No differential control in this run.** Each patch's necessity was established
  in the earlier per-workstream kits, not re-proven here.
- **No performance comparison** against any baseline. Every number here is
  correctness or count.
- **One configuration**: context 32768, chunk 8192/rank, MTP on the decode leg
  only, python router backend. A **Rust-router deployment with MTP still has the
  original bigram bug** — see `notes.md` §7.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable steps from a clean cluster to the result |
| `environment.md` | hardware, RDMA fabric, image digests, git SHAs, external paths, secrets needed |
| `notes.md` | the gotchas, the wrong turns, the three self-corrections — the most re-read file |
| `spec.md` | the originating mission file, verbatim |
| `working_process.md` | the round-by-round log as it was written during the run |
| `patches/` | all 7 patches + `README.md` giving what/why/how/context for each |
| `deliverable/` | the merged Dockerfile, its patch dirs, and the infera source diff + tests |
| `scripts/` | every script that ran, verbatim |
| `results/raw/` | needle + stress JSON, kvd counters, per-node env snapshots |
| `logs/` | both legs, router, gzipped (2.7 MB → 153 KB) |

## Related kits

These live in the **experiment checkout**, not in the repository — of the three,
only this kit is committed. (`kvaware_kvd_pr.packup_20260731/` is on the branch
too, carried in by the PR #59 cherry-picks.)

- `kvaware_kvd_pr.packup_20260731.pr.final` — workstream 1 alone
- `glm52.mxfp4.spur.mooncake.packup_20260731_main_converged` — workstream 2 alone
- `work.merge_20260731/` — the live scratch workspace this kit was built from (kept intact)

## Where this sits in the branch

`MERGE_BRANCH.md` at the repository root maps the four commit groups, says which
upstream PR each belongs to, and gives the exact `git rebase --onto` that drops
the temporary ones. Read that first if you are here to rebase rather than to
reproduce.

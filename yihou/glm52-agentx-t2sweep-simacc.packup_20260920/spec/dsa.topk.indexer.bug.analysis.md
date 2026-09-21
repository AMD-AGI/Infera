# GLM-5.2 on MI355X: decode-side GPU memory access fault in the DSA top-k indexer path

Analysis of a reproducible, non-deterministic GPU illegal-memory-access that kills
the decode worker of a GLM-5.2-MXFP4 1P1D disaggregated deployment under EAGLE
MTP. Written 2026-09-19 from five runs on 2026-09-18.

**Status: not root-caused, not fixed upstream, two viable workarounds identified,
and one likely configuration error found in our own launcher.**

---

## 1. Executive summary

| | |
|---|---|
| Symptom | `Memory access fault by GPU node-N ... Reason: Unknown` → `Fatal Python error: Aborted` on the **decode** worker. Note what is *absent*: no `HSA_STATUS*`, no kernel name, no `hipError*` — the log gives nothing beyond the faulting address |
| Frequency | **4 faults in 5 runs** of `config.full.sh`; 32 / 39 / 54 / 71 minutes into sustained load |
| Not | OOM, a bad GPU, custom all-reduce, the image layer, settle time, or any engine argument |
| Upstream | **Open**: sgl-project/sglang **#39517**, **#37648**. No fix in `main` as of `5d703de9e4` (2026-09-18) |
| Our own tracker | `bench/glm5p2_pd/issue.md` §3.3, previously marked "needs confirming whether v0.5.19 is still affected" — **this work confirms it is** |
| Likely config error | We run `--dsa-*-backend tilelang`; AMD's validated MI355X-MXFP4 TP4 recipe uses **`triton`** |
| Chosen workaround | `index_share_for_mtp_iteration=false` (see §8) |

---

## 2. Environment

| | |
|---|---|
| Hardware | 2 × AMD Instinct MI355X node, `gfx950`, driver 6.14.14, Ubuntu 24.04.4, kernel 6.8.0-107 |
| Nodes | `crsuse2-m2m-135` prefill, `crsuse2-m2m-138` decode, GPUs 2,3,4,5 each |
| Model | `amd/GLM-5.2-MXFP4` (Quark), `index_topk=2048`, `index_n_heads=32`, `index_head_dim=128`, `n_routed_experts=256`, `n_group=1`, `num_attention_heads=64`, `kv_lora_rank=512` |
| Engine | sglang `0.5.19.dev20260917+ga9fb1c3238`, image `infera-sglang:v0519-yihou-0917` |
| Shape | 1P1D, TP4 / DP4 / EP1, DP attention, `kv_cache_dtype fp8_e4m3`, `mem_fraction_static 0.85`, `chunked_prefill 32768` |
| Spec decode | EAGLE, `speculative_num_steps 5`, `speculative_eagle_topk 1`, `speculative_num_draft_tokens 6` — **decode leg only**; the prefill leg has `speculative_algorithm=None` |
| DSA backends | `--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --dsa-topk-backend sgl-kernel` |
| Relevant env | `SGLANG_OPT_USE_TOPK_V2=false`, `SGLANG_DSA_FUSE_TOPK` **unset → default True**, `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1` |

---

## 3. Fault signature

```
2026-09-18T16:29:13.156Z  Memory access fault by GPU node-6 (Agent handle: 0x637863981680)
                          on address 0x7d73b0400000. Reason: Unknown.
2026-09-18T16:29:13.xxx   GPU coredump: execvp failed: No such file or directory
2026-09-18T16:29:13.xxx   Fatal Python error: Aborted
   ... +10 min ...
                          [rank0] Watchdog caught collective operation timeout:
                          WorkNCCL(SeqNum=521, OpType=_ALLGATHER_BASE,
                          NumelIn=12288, NumelOut=49152, Timeout(ms)=600000)
                          ran for 600003 milliseconds before timing out
                          ProcessGroupNCCL … taking the entire process down
```

One DP rank faults; the surviving ranks then die ten minutes later on the NCCL
collective timeout. **The NCCL timeout and any downstream Mooncake
`transport retry counter exceeded` / `Connection refused` are consequences, not
causes** — the peer rank is already dead.

### The trap this creates

The HTTP server outlives the scheduler. After the fault you see: `/metrics`
answering 200, `num_running_reqs` frozen at a non-zero value per rank, aggregate
client counters frozen, `queue=1r/0w`, and engine logs showing only `GET /metrics`.
It looks like a hang. It is a corpse.

---

## 4. Reproduction record

All five runs used the same two nodes, same GPUs, same model, same topology.

| run | image | `disable_custom_all_reduce` | `index_share` | vs C40 `server_args` | faulting GPU | time to fault |
|---|---|---|---|---|---|---|
| **C40** (2026-09-18 06:51) | base `v0519-yihou-0917` | False | **on** | — | none | **completed 3600 s, 4128 req, 0 err** |
| **T1** (11:04) | `+nextnfix+hicache` | True | **off** | n/a (different config) | none | **completed, 1067 req, 0 err** |
| T2 run 1 (12:01) | `+nextnfix+hicache` | True | on | 1 diff | node-5 = GPU[3] | 32 min |
| T2 run 2 (12:56) | `+nextnfix+hicache` | True | on | 1 diff | node-7 = GPU[5] | 54 min |
| T2 run 3 (14:21) | `+nextnfix+hicache` | **False** | on | **0 diffs** | node-5 = GPU[3] | 39 min |
| T2 run 4 (15:17) | **base `v0519-yihou-0917`** | **False** | on | **0 diffs** | node-6 = GPU[4] | 71 min |

`GPU node-N` maps to physical `GPU[N-2]` (from each node's own `rocm-smi` node-ID
table). State immediately before every fault was healthy: `accept len` 3.50–3.79,
`token usage` 0.03–0.66, `errors=0`, `cuda graph: True`.

---

## 4a. The fault is not TP4-specific — corroboration from a second shape

Added 2026-09-19 from a parallel session on the same image and the same cluster,
running **TP8/DP8** where all of §4 is TP4/DP4. Their deployment:

| | |
|---|---|
| shape | TP8 / DP8, PD 1P1D |
| `index_share_for_mtp_iteration` | **ON** (verified: no override in the launch argv, so GLM-5.2's `config.json` default `true` stood) |
| fused top-k | ON (no `SGLANG_DSA_FUSE_TOPK` in that launch) |
| custom all-reduce | ON |
| decode ready | 2026-09-18 14:40:26 |
| **fault** | 2026-09-18 16:06:58 — `Memory access fault by GPU node-3 … Reason: Unknown.` → `Fatal Python error: Aborted`, one decode rank dead |
| **time to fault** | **1 h 26 min** after ready, ≈40 min into sustained AgentX C80 load |

**86 minutes sits inside our TP4 range of 32/39/54/71 minutes, not outside it.**
So the phenomenon spans both shapes and "TP8 is not susceptible" is ruled out.
Five faults are now on record across two TP sizes, all with IndexShare ON.

The same session separately established the **coherent** cell our own rounds
never had: TP8, simulation **off**, IndexShare **on**, custom all-reduce **on**,
16/16 temperature-0 probes coherent at real `spec_accept_length` **4.69**. That
run was only ~26 minutes under intermittent load, so it says nothing about
fault-freedom — they were explicit about that when asked.

## 5. What was eliminated, and how

Each row is an experiment or a first-hand measurement, not an inference.

| eliminated | evidence |
|---|---|
| **A single bad GPU** | four faults on GPU[3], [5], [3], [4] |
| **OOM / resource exhaustion** | zero `HSA_STATUS*`, `hipError*`, `out of memory`, `OUT_OF_RESOURCES` in any crash log; `#retracted-req: 0`; and — the strongest form — **the fault fires at LOW occupancy (`token usage` 0.03–0.66) while the same runs sustained a whole-run peak of 1.00 / 0.98 / 0.90 without faulting** |
| **Custom all-reduce** | run 3 dropped `--disable-custom-all-reduce`, reaching 0 `server_args` differences from C40 — still faulted |
| **The thin image layer** (NextN shared-experts fusion fix, PR #37152 HiCache) | run 4 used C40's exact base image — still faulted |
| **Node environment drift** | `collect_env.sh` diff, morning vs evening, on the decode node: only monotonic GFX-activity counters differ. Driver, kernel, OS, HCA inventory identical; 0 retired pages |
| **Settle time between runs** | run 2 launched onto GPUs idle ~23 min and faulted anyway |
| **Preemption / request retraction** | `#retracted-req: 0` in all four crash logs |
| **`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`** | C40 ran with it and completed |
| **`--dsa-topk-backend sgl-kernel`** | resolves to `sgl-kernel` in C40, T1 **and** T2 — passing it explicitly is a no-op |
| **Any other engine argument or container env** | a 21-agent exhaustive audit across prefill args, router/client config, InferenceX + dataset version, launcher diffs, machine state and fault forensics returned **zero surviving differences** after adversarial verification. All 34 `-e` vars engine.sh passes match C40 byte-for-byte |

**Conclusion of §5: C40 and the faulting runs are configurationally identical.
C40 is a lucky survivor of a non-deterministic defect, not a known-good
configuration.** Its margin was thin — it completed in ~80 min of load; run 4
died at 71 min.

---

## 6. The code path

### 6.1 What the indexer top-k is

GLM-5.2 uses DeepSeek Sparse Attention. For **each query token**, a "lightning
indexer" scores the cached history and selects the **top `index_topk = 2048` KV
positions**; attention then runs only over those. This selection is computed at
target prefill, at target verify, at draft extend, and at every draft decode
step.

It is **not** the MoE expert-routing top-k, and **not** a top-k over draft logits.

### 6.2 Switch A — `SGLANG_DSA_FUSE_TOPK` (`environ.py:166`, default **True**)

Controls whether the *top-k transform* is fused into the top-k kernel
(`dsa_backend.py:3554` docstring). The transform maps a selected local position
`p` through the page table to a physical `page_size=1` KV slot:
`real_page_table[row, p//ps]*ps + (p%ps)`.

- **Fused**: one kernel selects and emits physical slot indices. The decode CUDA
  graph then *drops the wide `page_size=1` page table* entirely
  (`dsa_backend.py:1330-1331`, `1370-1371`). Downstream consumption is the
  identity `page_table_1 = _get_fused_topk_page_table(topk_indices)`
  (`dsa_backend.py:2266-2267`).
- **Unfused**: the kernel returns raw local indices
  (`dsa_topk_backend.py:106-107` short-circuit); a separate Triton pass
  `transform_index_page_table_decode` (`transform_index.py:180-210`) maps them,
  reading the full wide table.

Backend constraint: under fused, only `sgl-kernel` and `flashinfer` are permitted
(`dsa_indexer_kpool.py:773-781`); `torch` requires `SGLANG_DSA_FUSE_TOPK=false`.

**Measured cost of disabling**, from PR **#36714**'s own speed table on
**MI355X gfx950 / GLM-5.2-MXFP4 / 1P1D PD / TP2**, acceptance simulation off:

| | fused off | fused on |
|---|---|---|
| p50 TPOT | 23.16 ms | **7.94 ms** |
| MTP accept length | 3.12 | 3.12 |
| TTFT | flat (1.01×) |

→ **disabling costs ~2.92× decode TPOT, with no acceptance change.** (The same
fusion is worth only ~3 % on NVIDIA H20 per #31477 — the ROCm unfused path is
relatively much slower.)

### 6.3 Switch B — `index_share_for_mtp_iteration` (model config, **true** for GLM-5.2)

Reuses the indexer top-k *selection* across MTP draft steps instead of
recomputing it each step.

- Seeded at **draft extend** from the last verified token, not from draft-decode
  step 0 (`eagle_worker_v2.py:293-294`).
- Capture → gather → carry: `forward_mha.py:100-118` fills
  `dsa_seed_topk_capture`; `eagle_worker_v2.py:938-949` and `1096-1103` gather it
  at `select_index`; `:1152-1153` stows it on `next_draft_input.dsa_topk_indices`;
  `IndexTopKShareState.mtp_iteration` (`index_topk_share.py:24-97`, driven from
  `eagle_worker_v2.py:714-718`) carries it across steps.
- Buffer width `dsa_seed_topk_width = index_topk + index_kpool - 1`
  (`model_config.py:357-361`).
- Gated on `topk == 1`, because "`select_top_k_tokens` reorders rows, desyncing
  indices" (`eagle_worker_v2.py:288`).

**Cost of disabling**: the indexer top-k runs on every in-loop draft-decode
forward — `speculative_num_steps - 1 = 4` extra selections per verify cycle
instead of 0. **Compute only**; the target verify recomputes its own selection, so
served output is unchanged. Per PR **#29787**, index-share moves accept length by
at most **+0.061** — its value is saving compute, not raising acceptance.
No upstream benchmark isolates its end-to-end cost.

### 6.4 Are the two switches the same thing? **No, on the decode leg.**

`dsa_backend.py:430` sets `use_fused_topk = should_use_dsa_fused_topk(seed_dsa_topk_from_draft_extend)`
— the only assignment in the file. Expanding `dsa/utils.py:99-114` for our decode
leg (ROCm, `disaggregation_mode == "decode"`, no hisparse, no dcp), where
`should_remap_pd_dsa_seed_to_local_slots()` reduces to the env value:

| `index_share` | `use_fused_topk` |
|---|---|
| on | `ENV and (not True or ENV)` = **ENV** |
| off | `ENV and (not False or …)` = **ENV** |

**Disabling B does not disable fused top-k on decode.** The single coupling point
engages only on the prefill worker. A and B are disjoint here, and B is the
narrower change.

### 6.5 Why the prefill leg still carries DSA seed machinery

Our prefill has no draft model. `scheduler.py:1499`:

```
# The PD metadata wire schema must match on P and D even when only D
# enables spec decoding; a seedless prefill writes the invalid sentinel.
output_dsa_topk_indices_dim = get_dsa_seed_metadata_dim(self.model_config.hf_config)
```

The seed field is reserved on both legs to keep the wire format symmetric; the
prefill writes a sentinel. **The seed itself is produced on the decode worker at
draft extend.**

---

## 7. Upstream status — not fixed

Verified 2026-09-19 against a freshly fetched `origin/main = 5d703de9e4`.

| question | answer | evidence |
|---|---|---|
| Are the matching issues fixed? | **No** | **#37648** (MI355X gfx950, GLM-5.3-MXFP4 **TP8 EAGLE**, long context) — the closest neighbour — and **#39517** are both **OPEN**, no linked fix. **On #39517, be careful**: it is Qwen3.5-122B, **TP1, non-PD, aiter/ck_tile**, and its log names `FmhaBatchPrefillWithPagedKVCache` *and* `HSA_STATUS_ERROR_MEMORY_FAULT`. Our four decode logs contain **neither string** (verified: 0 occurrences of each) — only the bare `Reason: Unknown`. **Same fault class, not the same signature**; an earlier draft of this document overstated that |
| Is there a fix in the 96 commits between our build `a9fb1c3238` and `main`? | **No** | per-file `git log a9fb1c3238..origin/main` over `dsa_topk_backend.py`, `dsa_indexer_kpool.py`, `dsa_backend.py`, `index_topk_share.py`, `dsa_backend_mtp_precompute.py` = **0 commits each**. Pickaxe for `clamp` / `padded` / `idle` / `uint32` over those paths: nothing relevant |
| Would upgrading the image help? | **No** | the hazard files are **byte-identical** between our build and `main` (matching git blob hashes). There is nothing to inherit |
| Was there a candidate fix that failed? | **Yes** | **#38575** (merged 2026-09-10, in our build) was hoped to be it; #39517's reporter: *"the 20260912 build that carries it faults identically. Six tries, six faults."* |
| Is the enabling PR reverted or guarded for gfx950? | **No** | **#36714** (merged 2026-08-29, ancestor of our build) flipped `should_remap_pd_dsa_seed_to_local_slots()` from `is_cuda()` to `(is_cuda() or is_hip())`; that line is unchanged at `origin/main:dsa/utils.py:88-91`. Its author validated ROCm accuracy only on **gfx942 (MI300X)**; gfx950 was measured for **speed only** |

### The negative-length hypothesis — checked and **rejected for our stack**

`dsa_topk_backend.py:303-307` warns that a negative (DP-padded / idle-companion)
per-row length read as `uint32` "reinterprets as ~4e9 tokens and
illegal-addresses". This is attractive — we run DP attention with very uneven
per-rank load (`#running-req` at fault time: `1/1/1/1`, `4/6/2/1`, `7/1/5/3`,
`4/1/1/1`, so padded rows always exist).

**It does not apply here.** That site lives only in the v2 fused kernel
`topk_transform_paged_v2`, gated by
`should_use_topk_v2() = is_sgl_kernel() AND SGLANG_OPT_USE_TOPK_V2` — and our
`config.sh:93` sets `SGLANG_OPT_USE_TOPK_V2=false`. Moreover PR **#30378**
already clamps padded rows to 0 there, and both named producers
(`fused_dsa_draft_extend_metadata`, `seqlens_expand_kernel`) clamp in our build
**and** in `main`.

So the mechanism stays **open**. What remains live on our decode leg is #36714's
seed remap itself, driven by `SGLANG_DSA_FUSE_TOPK`, independent of
`USE_TOPK_V2`, unreverted and unguarded for gfx950.

---

## 8. Remedies

### Fault-free evidence for `index_share=false`, across both shapes

| run | shape | IndexShare | custom AR | outcome |
|---|---|---|---|---|
| ours ×5 (`t2`, `t2b`, `t2c`, `t2d`, and the original rounds' configuration) | TP4 | **on** | varies | **4 faults** at 32 / 39 / 54 / 71 min |
| peer deployment B | TP8 | **on** | on | **fault** at 1 h 26 m |
| ours `t2e` | TP4 | **off** | **on** | 4 h 36 m, 0 faults, 2 points completed |
| ours `t2f` (running) | TP4 | **off** | **off** | 3 h 39 m so far, 0 faults, 2 points completed |
| peer current (running) | TP8 | **off** | **on** | 3 h 19 m so far, 0 faults, 2 points completed |

**5 faults with it on, 0 faults in ~11.5 accumulated hours with it off, across two
TP sizes and both settings of custom all-reduce.** The peer's run is what
separates the two switches: all of our own `index_share=false` evidence carried
`--disable-custom-all-reduce`, theirs does not, and it is clean — so the
IndexShare switch is doing the work, not the all-reduce change.

Still **correlational**. No faulting kernel has been identified by anyone.

### Measured costs — both were open questions until 2026-09-19

| change | measurement | source |
|---|---|---|
| `index_share_for_mtp_iteration=false` | **≈0.3 %** — 19,513 vs 19,575 tok/s/chip at TP8 CONC 80, inside run-to-run noise. Unmeasured above that concurrency | peer, same image |
| `--disable-custom-all-reduce` | **+0.7 % at CONC 40** (158,390 vs 157,362 tok/s total — i.e. no measurable cost) and **−4.2 % at CONC 56** (117,407 vs 122,579). Single-variable A/B, same stack, same 3600 s window | ours, `t2f` vs `t2e` |

So both mitigations are cheap. **Note what this corrects:** an earlier draft of
§8 carried a ~2.92× figure against `--disable-custom-all-reduce`. That number is
from PR #36714 and belongs to **`SGLANG_DSA_FUSE_TOPK`**, a different switch; it
was never a measurement of custom all-reduce and should not have been placed
beside it.

| # | change | evidence | cost |
|---|---|---|---|
| **1** | **`--dsa-prefill-backend triton --dsa-decode-backend triton`** | AMD's cookbook (`docs/cookbook/autoregressive/GLM/GLM-5.2.mdx:121,127` at `origin/main`) states the **MI355X MXFP4 gfx950 tp=4** recipe uses the **Triton** DSA backends, "also SGLang's ROCm default", and that **five-step MTP is validated** for `amd/GLM-5.2-MXFP4` on MI355X with image `…-20260916`, TP4/EP4 with HiCache. It explicitly says the **tp=8** recipes are the ones that stay on **tilelang**. #39517's reporter found `triton` "works, and works well" on gfx950 (**+54 %**) where the aiter path faults in ~3 min | possibly **faster** |
| **2** | `index_share_for_mtp_iteration=false` | our own T1: ~40 min of real MTP, zero faults. Narrower than #1 or #3 (decode draft path only) | 4 extra indexer selections per verify cycle; accept-length effect ≤ 0.061 per #29787; wall-clock unmeasured |
| **3** | `SGLANG_DSA_FUSE_TOPK=0` | disables #36714's seed remap and restores the wide page table | **~2.92× decode TPOT** per #36714's own table on our exact chip/model |
| — | upgrade the base image | **rejected** — hazard files byte-identical to `main` | — |

### A likely configuration error of our own

`bench/glm5p2_pd/engine.sh:164-165` hard-codes:

```bash
--dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}"
--dsa-decode-backend  "${DSA_DECODE_BACKEND:-tilelang}"
```

Every run in this investigation — C40, T1, and all four T2 runs — therefore ran
**tilelang**, which AMD documents as the **tp=8** choice. Our shape is **tp=4
MXFP4**, for which the documented and validated backend is **triton**, described
as the configuration "the gfx950 sparse-MLA kernels are tuned for" (16 query
heads per rank on an FP8 KV cache). This should be tested.

---

## 9. What is NOT established

1. **The mechanism.** No faulting kernel or instruction has been identified. Our
   own `issue.md` §3.3 says the same: "only narrows it to the fused DSA indexer
   or the metadata/workspace/buffer lifetimes it changes; no specific faulting
   kernel or instruction located." This analysis does not improve on that.
2. **That `index_share=false` fixes it.** As of 2026-09-19 the count is 5 faults
   with it on versus 0 in ~11.5 accumulated hours with it off, across TP4 and
   TP8 and both settings of custom all-reduce (see §8). That is a much stronger
   correlation than the n=1 this section originally recorded — but it is still
   correlation. No faulting kernel has been identified, so the mechanism by
   which the switch helps is unknown, and a defect that merely became rarer
   would look identical from here.
3. **That triton avoids it.** Untested by us.
4. **Why C40 survived.** Best available explanation is chance, given a
   non-deterministic fault and a thin margin, but this is not proven.

---

## 10. Wrong turns (recorded so they are not repeated)

Each cost real time; each was wrong for a reason worth knowing.

1. **"Custom all-reduce is the cause."** Asserted after a 1-difference
   `server_args` diff. Refuted by run 3. *Lesson: a single surviving difference
   is a candidate, not a cause.*
2. **"The image layer is the cause."** Asserted after run 3. Refuted by run 4.
   *Same lesson, second time.*
3. **Reporting `json_model_override_args='{}'` as "identical to C40" without
   resolving it.** The empty override means the **model default applies**, and
   GLM-5.2's `config.json` has `index_share_for_mtp_iteration = true`. Reporting
   the literal config value while omitting its runtime meaning hid the single
   most relevant switch for hours. *Lesson: resolve config values to runtime
   effect before reporting them as equal.*
4. **"The seed is computed on prefill and shipped across machines."** Wrong: our
   prefill has `speculative_algorithm=None` and writes only a sentinel
   (`scheduler.py:1499`). *Lesson: a docstring describing a general topology is
   not a description of the deployment in front of you.*
5. **"DP-padded negative length read as uint32."** Plausible and matched every
   symptom — but the site is in a kernel our config does not use, and it is
   already clamped. *Lesson: check the gate before adopting a mechanism.*
6. **Treating a parked `Phase warmup progress` counter as a stall.** AgentX stops
   emitting those lines in the warmup tail and never emits a profiling-phase line;
   C40's last line is `434/444` and it completed normally. The liveness signals
   are `tot in=/out=`, `unique_in_srv` and `tput_in_srv`.
7. **Quoting a pre-fault maximum as a whole-run peak.** The first draft
   eliminated OOM using "peak `token usage` 0.66" — but 0.66 is only the highest
   value seen *immediately before* a fault; the same runs reached 1.00 / 0.98 /
   0.90 earlier without faulting. The conclusion survived and in fact got
   stronger (the fault fires at *low* occupancy while high occupancy is
   sustained fine), but the number as stated was wrong. *Lesson: say which
   window a maximum is over.*
8. **Claiming an "identical signature" to upstream #39517.** That issue is a
   different model, TP1, non-PD, a different backend, and its log names both a
   kernel and `HSA_STATUS_ERROR_MEMORY_FAULT`; ours names neither. *Lesson:
   matching one log line is not a matching signature — diff the whole report.*
9. **Not reading `bench/glm5p2_pd/issue.md` §3.3.** The defect was documented in
   the very directory this work was carried out in, with the open question
   "confirm whether v0.5.19 is still affected". *Lesson: read the repo's own
   issue register before searching upstream.*

---

## 11. Contribution back

`issue.md` §3.3 asked whether the v0.5.19 fused path is still affected. **It is.**
Four faults, on an image (`20260917` nightly base) newer than the `20260911` the
note suggested trying, with `#36714` and `#38575` both present. That, plus the
byte-identical-to-`main` finding, is worth adding to §3.3 and to upstream #39517.

## 12. References

| kind | ref |
|---|---|
| our tracker | `bench/glm5p2_pd/issue.md` §3.3 |
| upstream issues | [#39517](https://github.com/sgl-project/sglang/issues/39517), [#37648](https://github.com/sgl-project/sglang/issues/37648), [#37478](https://github.com/sgl-project/sglang/issues/37478) |
| upstream PRs | #36714 (enabler + speed table), #38575 (failed candidate fix), #30378 / #30512 (v2-kernel clamps), #30506 (ROCm forces `USE_TOPK_V2=false`), #31477 / #29787 / #28192 (fused-topk and index-share lineage), #39631 / #40148 (HIP top-k, cookbook) |
| cookbook | `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx:120,121,127` @ `origin/main` |
| code | `environ.py:166`; `dsa/utils.py:88-114`; `dsa_backend.py:430,1330,1370,2266,3554,3575`; `dsa_topk_backend.py:106,303-307`; `dsa_indexer_kpool.py:748,941,773-781`; `eagle_worker_v2.py:151,288-303,714-718,938-949,1096-1103,1152-1153`; `index_topk_share.py:24-97`; `model_config.py:357-361`; `scheduler.py:1499`; `engine.sh:164-165` |
| raw evidence | `bench/glm5p2_pd/results/yihou-agentx-hicache/{t2,t2b,t2c,t2d}-crash/`, `working_process.md`, `notes/poll-log.md` |
| archived so §5 is checkable | `notes/evidence/` — retired-page dumps for both nodes, `collect_env` morning vs evening for the decode node, and the decode `server-info` for C40 and for runs 1/3/4 (the comparisons §5 rests on originally lived outside this workspace) |
| independent verification of this document | `notes/writeup-verification.md` — claim-by-claim, 13 pass / 2 fail. Both failures are corrected above and recorded in §10 |

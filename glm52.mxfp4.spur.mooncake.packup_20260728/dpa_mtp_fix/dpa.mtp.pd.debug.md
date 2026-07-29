# DPA + MTP + PD — debug master doc

Consolidated 2026-07-29. Everything needed to resume debugging without prior conversation:
the bug in plain language, what is proven vs inferred, the research, the open options, and
the concepts that kept coming up.

Companion files (same dir): `CODE_VERIFICATION_bug2.md` (source reading),
`RESEARCH_bug2.md` (upstream + CUDA comparison + docs), `HANDOFF_pd_mtp_hang.md`
(chronological trail incl. refuted hypotheses), `NOTES_rootcause_and_fix.md` (Bug 1).

---

## 0. Status in one table

| | State |
|---|---|
| **Bug 1** — `lengths.size(0) == B` crash (DPA+MTP, single node) | ✅ **FIXED, shipped.** Patch in `infera` branch `worktree-dsa-hip-dp-rows-fix` (`9bcec49`), **not yet pushed** — user submits the PR |
| **Bug 2** — PD + MTP decode deadlock | ❌ **Root-caused, reproduced, NOT fixed.** Fix designed, not implemented |
| Single-node mix, DPA+MTP | ✅ 256/256 |
| PD + DPA, **no** MTP | ✅ 1792/1792 over conc 64/128/256 (control) |
| PD + DPA + MTP | ❌ boots clean, `health=200`, **first routed request hard-deadlocks** |

---

## 1. The bug in plain language

**Two facts that collide.**

*Fact A — the 8 GPUs must move in lockstep.* DP attention makes 8 GPUs act as one unit.
Periodically they must pool and redistribute intermediate results — a **collective**
(`all_gather`, `broadcast`). The iron rule: **all 8 must arrive, or the other 7 wait
forever.** So even a GPU with no work must run a dummy "idle" forward to keep the
collectives whole. That is the `IDLE` forward mode — by design.

*Fact B — the GPU is asynchronous.* The CPU queues work and races ahead without waiting.
That is where the performance comes from. But when the CPU needs to **read an actual number
off the GPU** (e.g. "how long is the longest sequence in this batch?"), it must stop and
wait for the queue to drain. `.max().item()` is one innocuous-looking Python line that is
in fact **a hard brake**. NVIDIA's own docs warn about exactly this:

> *"Something that looks trivial in Python, like `if tensor.item() > 0:`, may trigger a
> host-device synchronization under the hood."*

**The bug: the brake sits on a fork only some ranks take.**

On the PD decode leg, work arrives per-request, so "only 1 of 8 ranks has work" is routine —
we measured `global_num_tokens_cpu = [0,0,4,0,0,0,0,0]`. And the metadata-construction code
is guarded by *"am I idle? if so, skip."* So:

- **the rank with work** → enters → hits the brake → CPU blocks waiting on GPU
- **the 7 idle ranks** → skip the whole block → race ahead into the next collective → wait

7 ranks wait at the collective for the 8th; the 8th waits at the brake for the GPU.
**Deadlock.** Because they drop out at different moments they end up spread across *two
different* collectives — py-spy caught 2 in `broadcast`, 5 in `all_gather`, 1 at the brake.
That is not "slow", that is fully desynchronized. Per PyTorch/Meta fleet data:

> *"Almost all NCCL watchdog timeouts are caused by collective desync (mismatch), not
> slowness — meaning increasing the timeout will not fix the issue."*

**Why NVIDIA is fine: the same code never runs there.** On CUDA the whole metadata
construction is captured into a CUDA Graph and replayed, so the number is never needed
at runtime. The guard is *"idle **or** graph-capable → skip"*, and CUDA satisfies the second
clause — **all** ranks skip, uniformly, no fork. On HIP that graph is never captured, so the
"fallback" path becomes **the only** path, taken every step.

**One-line summary:**

> The defect is not *"a sync exists"* — it is *"a sync exists on a branch only some ranks
> take."* The same `cudaStreamSynchronize`, hoisted to a point every rank reaches, is
> harmless (the scheduler already has one). Left behind `if I-have-work`, it deadlocks.

---

## 2. Concepts (asked during the session — keep for the next reader)

### draft decode vs draft extend

One speculative-decoding cycle, say we're at token 100:

1. **draft** — the small model (nextn) guesses ahead. With `num_steps=3`: guess 101 → 102 →
   103. Each guess is **1 token in** → shaped like decode → **draft decode**.
2. **verify** — the big model checks all 4 tokens **in parallel** (`TARGET_VERIFY`).
   Say it accepts 101, 102 and rejects 103.
3. **draft extend** — the small model's KV cache is now stale: it knows what it *guessed*,
   not what was *ratified*. So the confirmed tokens are fed to it **all at once** →
   **many tokens in** → shaped like prefill → **draft extend** (`DRAFT_EXTEND_V2`).
   **This is where our bug lives.**

| | what | tokens per pass | shape |
|---|---|---|---|
| draft **decode** | small model guesses forward | 1 | like decode |
| draft **extend** | small model catches up | many (=4) | like prefill |

Both are the *same small model* — only the shape differs. Attention kernels optimise the two
shapes very differently, so sglang gives them **separate backends** and captures **separate
CUDA graphs**. Hence our log:

```
Capture draft decode CUDA graph  → 1   ✅ captured on HIP
Capture draft extend CUDA graph  → 0   ❌ not captured
```

### `--speculative-attention-mode`

Picks which backend config draft-extend borrows, since its shape is prefill-like:

```python
backend_name = "decode_attention_backend" if mode == "decode" else "prefill_attention_backend"
```

`prefill` (default) → uses `--nsa-prefill-backend`; `decode` → uses `--nsa-decode-backend`.
It exists because draft-extend, while prefill-shaped, has very short sequences (4 tokens) and
tiny batches, where decode kernels sometimes win.

### "the CPU mirror"

`seq_lens` (how long each request currently is) lives in **VRAM**. The CPU cannot dereference
VRAM, so to read it a copy must be made into host memory — that copy is `seq_lens_cpu`, the
"mirror".

*Why the CPU needs it:* scheduling and metadata construction run in Python and need **real
integers** — how wide to slice a table, how big a buffer to allocate. A device tensor cannot
be the `n` in `x[:n]`.

*Why the value lives on the GPU:* in speculative decoding, how many tokens got accepted is
computed **by the verify kernel on the GPU**, so sequence lengths change there. The
authoritative copy is necessarily in VRAM.

*It doesn't have to block:* upstream built pinned host memory + a private D2H stream + event
gating so the copy overlaps asynchronously. `needs_cpu_seq_lens = False` says "don't even do
that for me" — a claim DSA makes because it assumes graph replay, which needs no such value.

---

## 3. What is PROVEN (measured, reproducible)

1. **`is_idle(rank) == (global_num_tokens_cpu[rank] == 0)`** — 56/56 probe samples, zero
   mismatches.
2. **That predicate gates the eager metadata call**, so the branch is rank-ragged whenever
   occupancy is partial.
3. **The stall site**: `dsa_backend.py:767` (in the patched build) = the `else` arm's
   `seq_lens.max().item()` — a GPU→CPU sync. py-spy sampled twice, zero movement.
4. **The stuck rank follows whoever holds the work** — DP1, DP2, DP5, DP6 across runs. A
   race, not a fixed rank.
5. **Arm selection tracks forward mode**: `IDLE` → mirror present → HOST arm;
   `DRAFT_EXTEND_V2` → mirror None → GPU_SYNC arm.
6. **MTP is the sole trigger**: same image, same script, only `MTP=1→0` → 1792/1792 pass.
7. **Not transport**: hung vs passing runs byte-identical (mlx5_0, 8× rdma, 0 ionic, 0
   `KVTransferError`).
8. **Not Bug 1**: that assert never reappears under PD with the patch.
9. **`can_cuda_graph` is uniformly False on HIP** — draft-extend graph never captured
   (`grep -c "Capture draft extend CUDA graph begin"` = 0 in **both** the hung PD run and the
   passing single-node run).

### Explicitly REFUTED (do not revisit)

- ❌ *"`can_cuda_graph` diverges per rank"* — it is uniformly False on HIP. This was the
  previous top hypothesis **and is what every upstream report blames**; it is not our bug.
- ❌ *"Flip `needs_cpu_seq_lens = True`"* (option b) — **declined upstream** by merged
  PR #29798: it would tax every graph-replay deployment. See §5.
- ❌ *"Patch only line 746"* — insufficient; lines 806-813 sync unconditionally on the same
  branch and would deadlock identically.
- ❌ *"`Device2ExtendCudaGraphRunner` lacks a `hip` key, so lookup fails"* — **my error.**
  On ROCm `torch.zeros(1,device="cuda").device.type == "cuda"`, so the lookup hits.
- ❌ *"#27091 is the issue to read"* — it is a **merged PR** about SWA index translation. The
  `schedule_meta` line is a rejected-alternative note justifying pre-pad ordering. Unrelated.

### Inferred, NOT proven

- That a per-rank host sync *causes* the ragged collective. Source proves CUDA skips the
  function and HIP does not; the causal step rests on the py-spy evidence.
- That lines 806-813 are dead for `DRAFT_EXTEND_V2` (dataflow says yes; `forward_batch` is
  mutated in place, so **grep for post-call readers before removing**).

---

## 4. Root cause (full chain, every link measured)

1. A PD decode step gives work to a subset of DP ranks — routine, since batches are shaped by
   KV arrival from prefill.
2. `is_idle(rank) == (gnt[rank] == 0)` → busy rank gets `DRAFT_EXTEND_V2`, rest get `IDLE`.
3. Forward mode decides the mirror: `IDLE` → `seq_lens_cpu` present; `DRAFT_EXTEND_V2` →
   `None` (`gpu_only`, `base_spec_worker.py:112`).
4. Missing mirror → `else` arm → `seq_lens.max().item()` = **GPU→CPU sync**, while idle peers
   take the free host path into the next collective.
5. The syncing rank cannot retire; peers hold the stream. **Deadlock.**

### The contract violation underneath

`DeepseekSparseAttnBackend` declares `needs_cpu_seq_lens = False` (twice), telling the
scheduler not to publish the mirror. Its own comment states the assumption:

> *"Decode/verify/draft **graph replay** rebuilds metadata from static buffers … **The eager
> fallback derives lengths from GPU seq_lens.**"*

The design **assumes graph replay**. The `.item()` is the acknowledged fallback, believed
rare. On HIP the graph is never captured → the fallback is the only path, every step.

---

## 5. Upstream research (see `RESEARCH_bug2.md` for citations)

**The failure *class* is known and hot; our *mechanism* is unreported.**

- **#32527** (opened 2026-07-27, one day before us) — identical topology and precondition.
- **#32209** (open, **CI red**) — upstream's fix attempt: all-gather the graph-vs-eager
  decision so ranks vote. **Cannot fix ours** — a host-side sync is invisible to such a vote.
- **#32722** (open, **red by design**) — first CI case for PD+DPA+MTP; proves **no existing
  coverage** of this combination.

All of them blame `can_cuda_graph`, which we measured is uniformly False here. Exhaustive
search found **no** report of a busy rank stalling on `.max().item()` while idle peers
advance. **Still broken on `main`** as of 2026-07-28.

**Upstream direction endorses our fix.** v0.5.15 release notes / LMSYS blog:

> *"…made `seq_lens_cpu` optional for DSA to **drop the D2H sync**…"*

Removing this host dependency is upstream's own goal, framed as perf. Our contribution
reframes it as also a correctness fix — that is the PR framing to use.

**Why option (b) is out** — merged PR #29798 (whose diff *is* lines 806-813) states:

> *"An earlier local workaround was to set `…needs_cpu_seq_lens=True`. That fixes the
> assertion but makes **all** spec-v2 DSA draft decode materialize a CPU sequence-length
> mirror, **including CUDA graph replay cases**…"*

Note #29798's scope: it repaired the eager path for batches exceeding `--cuda-graph-max-bs`
— a **global** property entered by all ranks together. Upstream never anticipated **per-rank
asymmetric entry**. That gap is our finding.

**CUDA may not be permanently immune.** `can_run_dp_cuda_graph` is an all-gathered *min* over
ranks; if it goes False under partial occupancy, CUDA falls eager and hits the same bug.
Unresolved statically — strengthens the case for fixing it properly rather than per-platform.

---

## 6. Options

### Fix A (recommended) — use the static page-table width

Replace the `else` arm (`dsa_backend.py`, ~line 746 stock / 767 patched):

```python
max_seqlen_k = self.req_to_token.shape[1]
```

**Idiom is established in-tree, three ways:**
- `triton_backend.py:704-708` — *"gpu_only: seq_lens_sum may be None; **over-allocate is
  safe** (ragged write)"*
- `trtllm_mha_backend.py:555-559` — *"Static upper-bound page-table width … **never a host
  max / seq_lens_cpu D2H sync**"*
- `dsa_backend.py:689-695` — **DSA's own graph path already does exactly this.**

**Over-sizing proven safe** by full consumer trace: the wide table is only indexed *through*
top-k, and top-k masks per row by `lengths` derived from `cache_seqlens` — independent of
`max_seqlen_k`. Extra columns get `-inf`, are never selected, never dereferenced.
**Cost:** the indexer logits buffer (~16.8 MB/call at 128K ctx, bs=8, next_n=4 vs a few MB
realistic). Bounded, but **must be benchmarked** — the one open cost question.

### Fix A2 (required, same change) — lines 806-813

Two unconditional syncs on the same branch; without these the hang persists. Verify by
instrumentation before removing (see §3 "inferred").

### Fix C (fallback) — uniform entry
Make all ranks take the branch. Works, but taxes idle ranks with a pointless sync. Composes
with #32209's philosophy. Use only if A/A2 stall.

### Fix B (separate work, NOT for the hang) — enable draft-extend graph on HIP
Would close the eager gap and is real perf. **Deliberately deferred:** it only *hides* the
defect (a batch over `--cuda-graph-max-bs` falls back to eager and breaks again — on CUDA
too). Est. 3 gates ≈ half a day mechanical, then **1-3 days unknown** making HIP's non-fused
metadata path work *under capture* (stricter than eager: no host syncs, no dynamic alloc).
Estimate unverified.

### Config workarounds evaluated — both dead ends

- **`--speculative-draft-attention-backend aiter`** → ❌ **Does not work for this model.**
  It *would* satisfy the HIP gate (`isinstance(..., AiterMultiStepDraftBackend)`), but aiter
  has **zero** indexer support (`grep -c indexer` = 0, `is_sparse=False` hardcoded) while the
  model is `GlmMoeDsaForCausalLM` with `index_topk=2048`. Swapping it replaces sparse
  attention with dense → OOM or **silently wrong output**. The gate targets non-DSA models.
- **`--speculative-attention-mode decode`** → ⚠️ **Low expectation, cheap to test.** Both our
  backends are tilelang, so it likely still yields a DSA class and the gate still fails. But
  it builds `DeepseekSparseAttnMultiStepBackend` via a different factory, so the metadata path
  *might* differ. 10 min to falsify.

---

## 7. Second known ROCm deadlock — MUST rule out

> EAGLE on gfx942/gfx950 requires **`--disable-custom-all-reduce`**: the aiter custom
> all-reduce kernel deadlocks during EAGLE verify at high concurrency. (sglang GLM-5.1
> cookbook; cf. #28815 / #31071 / PR #31478.)

**Our scripts do not set this.** Add it to the next run so a persisting hang cannot be
misattributed to the indexer sync.

⚠️ Also unverified but high-impact: the GLM-5.2 cookbook reportedly **disables MTP for AMD
outright** — *"MTP on gfx950 still depends on the spec-decode draft kernel, which isn't yet
validated on this hardware."* Rests on a search-index summary of a JS-rendered page. If true,
we are first to exercise this path on gfx950 and should expect further bugs behind this one.

---

## 8. Live environment

| | |
|---|---|
| Jobs | `9005` → `crsuse2-m2m-244` (**10.245.157.105**, prefill), `9006` → `crsuse2-m2m-029` (**10.245.146.21**, decode) |
| Container | `dbg2` on both; Bug 1 patch applied; **no** Bug 2 probes/fix currently installed |
| Prefill leg | ✅ **already up**, `health=200`, `MTP=0` (fix-agnostic, pre-warmed) |
| Caches | `/home/yihou/glm52_fix/{inductor_cache,triton_cache}` — host bind-mount, persists across containers |
| Image | `infera.yihou.sglang.1.0` ← `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`. **Note: PyPI tops out at 0.5.10 — 0.5.15.post1 ships only as this image** |

```bash
export DOCKER_CONFIG=/tmp/dockercfg     # before EVERY docker call
spur exec 9006 bash -c "... docker exec dbg2 ..."
```

### Scripts
`scripts/` — `pd_leg_spur.sh` (both legs), `probe.py` (4-prompt correctness),
`instrument_divergence.py` (rank-tagged branch probe), `instrument_dsa_stall.py`,
`instrument_v2.py` (which arm), `fix_bug2.py` (**partial — do NOT ship alone**).
All idempotent, anchor-match-verified, `py_compile`d, `--revert` supported.

### Traps that cost a boot cycle (~10 min each)
1. **Never `pkill -f launch_server`** — orphans schedulers, leaks ~82% VRAM, next boot wedges
   silently. Use `docker rm -f` + recreate; confirm `rocm-smi --showmemuse` reads 0%.
2. **`TORCHINDUCTOR_COMPILE_THREADS=4` is NOT enough** on a cold cache — the boot-time
   Inductor deadlock recurred with it set (DP0-2 in `synchronize`/`_make_launchers`, DP3-7 in
   `broadcast`, >17 min, `health=503`, no exception). Use `=1` **plus** the persistent cache
   dirs above.
3. **Distinguish the two deadlocks:** boot-time → never reaches `ready to roll`, Inductor
   frames. Bug 2 → reaches `health=200`, hangs only on a routed request, one rank in
   `init_forward_metadata`.
4. **Router:** fresh `--port` *and* `--prometheus-port` every restart; `pkill -9 -f
   launch_router` first; restart it *after* decode reports 200.
5. Spur evicts jobs without warning (lost all 4 mid-session) and `JobHoldMaxRequeue` bounces
   are normal — retry-loop the `sbatch`.

### Diagnostic one-liner
```bash
docker exec dbg2 bash -c 'for p in $(ps -eo pid,args --no-headers | \
  grep -oE "^ *[0-9]+ .*sglang::scheduler_DP[0-9]" | awk "{print \$1}"); do
    N=$(ps -p $p -o args= | grep -oE "DP[0-9]"); \
    echo "$N: $(py-spy dump --pid $p 2>&1 | sed -n 5p | xargs)"; done'
```
Sample twice ~6 s apart — identical output = hang.

---

## 9. Plan

1. **10-min cheap shot** — relaunch decode with `--speculative-attention-mode decode` **and**
   `--disable-custom-all-reduce`. Falsifies the config workaround and clears the second known
   ROCm deadlock in one boot. Low expectation, high information/cost ratio.
2. **Implement Fix A + A2**, instrument to confirm 806-813 are dead for this mode, verify with
   the 4-prompt probe, then the c64/128/256 sweep against the DPA-only control numbers
   (2051 / 3743 / 6243 out tok/s).
3. **Benchmark the widened logits buffer** — the one open cost question for Fix A.
4. If A/A2 stall → fall back to Fix C (uniform entry).
5. Consider an upstream issue: our mechanism is unreported, `main` is still broken, and
   #32722 shows they want coverage here. Framing: *"extends the v0.5.15 D2H-sync removal to
   the eager path, where it also fixes a DP-rank-divergence hang."*

**Not in scope now:** Fix B (HIP draft-extend graph) — separate perf work, deliberately
deferred.

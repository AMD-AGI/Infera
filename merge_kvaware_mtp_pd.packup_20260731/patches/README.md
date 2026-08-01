# The patch set — what, why, how, context

Seven patches. 1–5 were planned; **6 and 7 were written during the run**, when
turning MTP on exposed two pre-existing infera auto-appends that no prior
validation had exercised.

Applied by `scripts/apply_all_in_container.sh`, which runs them in order and then
verifies every one reached the **bytecode**. See §"Why bytecode" below.

| # | file | layer | source |
|---|---|---|---|
| 1 | `dsa_indexer_hip_dp_padded_rows.diff` | sglang | PR58 |
| 2 | `dsa_backend_dp_sync_and_page_table_rows.diff` | sglang | PR58 |
| 3 | `draft_cuda_graph_dp_vote.diff` | sglang | PR58 |
| 4 | `patch_mooncake_early_send_wait_event.py` | sglang | PR56 |
| 5 | `patch_infera_kvevent_bigram.py` | **infera** | PR56, re-cut |
| 6 | `patch_infera_decode_radix_vs_mtp.py` | **infera** | new, this run |
| 7 | `patch_infera_decode_kvd_skip.py` | **infera** | new, this run |

**5–7 edit infera's own code and belong as source commits, not build layers.**
They ship here as self-locating scripts only so the running containers could pick
them up without a rebuild. Their committed form, with tests, is
`../deliverable/infera_source_changes.diff`.

---

## Prerequisite (not one of the seven)

The GLM-5.2 nextn `eh_proj` quark-exclude fix must already be present. It is —
the base image's Dockerfile patch loop applies it (verified at
`deepseek_nextn.py:363`).

`apply_sglang_dsa_patches.sh` **asserts** it rather than applying it. That script
is idempotent, so a silent "skipped" would go unnoticed until runtime, where it
surfaces as GLM-5.2 dying at draft weight-load with a `3072 vs 6144` shape error.

---

## 1–3. The DSA patch set (sglang, from PR58)

**Context.** Without these, PD + DP-attention + EAGLE MTP on GLM-5.2/gfx950
crashes on the first batch or deadlocks the whole DP group under concurrency.

**What / why**, per patch:

| # | what it fixes | shape |
|---|---|---|
| 1 | HIP/aiter paged-MQA sizes its output from **DP-padded** rows while `lengths` is sized to **real** rows → `Expected lengths.size(0) == B` | after upstream [#32762](https://github.com/sgl-project/sglang/pull/32762) (NPU, same bug class): one boolean gates both trim and restore, post-kernel row count asserted before padding is restored |
| 2 | (a) `seq_lens.max().item()` is a host sync on a branch only *some* DP ranks take → collectives desynchronize; (b) page table has one row per **request**, top-k one per **token** under MTP → assert | ours; no upstream counterpart found |
| 3 | the draft graph/eager choice is made **per rank** from rank-dependent inputs and diverges on the PD decode leg → deadlock | after upstream [#32209](https://github.com/sgl-project/sglang/pull/32209): the vote rides an all-gather the scheduler already performs — **zero** extra collectives |

**How.** `patch -p1 --fuzz=0` against the pinned sglang commit
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`. `--fuzz=0` is deliberate: a fuzzy
apply that "succeeds" against a different base is worse than a clean failure.

**Necessity is not re-proven in this kit** — it was established in the earlier
per-workstream kits (patch 3's same-node revert control: 0/4, deadlock).

**There is a configuration-only alternative to parts of 2 and 3**:
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'`. Each diff
header records its own status relative to that. Not adopted here.

---

## 4. `patch_mooncake_early_send_wait_event.py` (sglang, from PR56)

**Context — the symptom.** With chunked prefill over the mooncake KV transport,
prompts longer than one prefill chunk come back **partially wrong**, and nothing
raises. On GLM-5.2 DSA a needle read returns the first digits and then repeats
`</think>`:

    want=2183762  got='2183</think>2183</think>218</think>218</think> the'

The corruption boundary lands exactly on the chunk boundary — a needle in the
*final* chunk is retrieved correctly, and the same model and chunk size in an
aggregated single-node server passes.

**Why.** `prefill.py`'s early-send path already records a completion event as the
barrier, but **only `mori/conn.py` ever read it**. `mooncake/conn.py` has no
`wait_event` / `synchronize()` anywhere, and the overlap path that moves
non-final chunks recorded none. So the transfer worker RDMA-reads pages while the
forward that writes them is still running. The final chunk is always correct
because it goes through the sampling path, which already synchronizes.

**What.** Three files, mirroring what `mori` already does: `TransferKVChunk`
carries a `wait_event`; mooncake's `send()` picks it up, `add_transfer_request()`
forwards it, and `transfer_worker` synchronizes on it before reading device
memory; `prefill.py`'s overlap non-final-chunk send records one.

**How.** Self-locating idempotent Python. **All three files or none** — a
half-patched tree still corrupts, so an anchor that is missing or no longer
unique writes nothing and exits 1.

Not DSA-specific: any PD deployment running chunked prefill over mooncake with
overlap scheduling is affected. **Validated here by G2** (5/5 with the prompt
confirmed split into 3 chunks).

---

## 5. `patch_infera_kvevent_bigram.py` (infera, from PR56)

**Context.** kv-aware routing under MTP scores every worker zero and degrades to
round-robin, with nothing in any log.

**Why.** With EAGLE/MTP on, SGLang keys its radix tree on bigrams
(`RadixKey.is_bigram`, set from `is_eagle`) and the kv-event emitter reports a
block's tokens as overlapping pairs:

    sglang/srt/mem_cache/events.py:65
        if is_bigram:
            page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]

infera hashed the pairs as-is. The query side chunks the **flat** token slice, so
the resulting view can never be matched.

**Note `is_eagle` is a global server arg** — this affects the **prefill** leg too,
not only the leg running MTP. That is what makes it observable in this run.

**What.** `_flat_tokens` takes the first element of each pair, rebuilding
`t[start:end]`; radix nodes split on page boundaries, so the two chunkings stay
aligned. The msgspec schema widens to `list[int | tuple[int, int]]` so the pairs
decode at all.

**How.** Self-locating idempotent Python, patching **every** infera copy it finds
— see the two-copies trap in `../notes.md` §3.

**Verified** by G1's router cache-view reading 51 blocks, byte-identical to G0's
plain-int path. Unfixed it reads 0. Three unit tests fail on pre-fix code.

**Not applied to the Rust router.** `rust/router/src/kv_event.rs` has the same
bug; every run here uses `--router-backend python` (the default). See
`../notes.md` §7.

---

## 6. `patch_infera_decode_radix_vs_mtp.py` (infera, written during this run)

**Context.** The G1 decode leg died in argument parsing, before loading a weight:

    ValueError: --disaggregation-decode-enable-radix-cache is incompatible with
    speculative decoding (--speculative-algorithm EAGLE)

**Why.** infera appends `--disaggregation-decode-enable-radix-cache` on a
mooncake decode leg whenever **kv-events** are on (`args.py:255`), so the router
can steer repeats to the rank holding the prefix. SGLang hard-forbids that flag
under speculative decoding (`arg_groups/pd_disaggregation_hook.py:41`).

**This is not a conflict between the merged workstreams** — no line of it
changed. It is pre-existing code meeting a configuration nobody had run: the
kvaware/kvd validation never enabled MTP, and the MTP validation drove
`sglang.launch_server` directly, bypassing the infera wrapper that appends.

**What.** Extend the existing gate — which already excludes non-mooncake backends
for the same "SGLang rejects it" reason — to also exclude speculative decoding,
and log why rather than dropping the flag silently.

**Consequence, deliberately not hidden.** SGLang then takes its `else` branch and
forces `disable_radix_cache = True`. The decode leg contributes little or nothing
to the router's KV view under MTP; prefix-aware routing runs on the prefill-side
view. Measured: decode cache-view **0 blocks** in G1 vs 90 in G0.

**How.** Self-locating idempotent Python, all infera copies. Two tests
(`test_decode_radix_vs_speculative.py`), verified to fail on pre-fix code.

---

## 7. `patch_infera_decode_kvd_skip.py` (infera, written during this run)

**Context.** Fixing 6 exposed the next crash immediately:

    ValueError: The arguments enable-hierarchical-cache and disable-radix-cache
    are mutually exclusive

**Why the crash.** infera's kvd wiring unconditionally appends
`--enable-hierarchical-cache` (`kvd_wiring.py:51`). Patch 6 causes
`disable_radix_cache = True`, and SGLang refuses the combination
(`server_args.py:5772`). That check comes from upstream
[#9452](https://github.com/sgl-project/sglang/pull/9452): hicache is implemented
as a `RadixCache` subclass and its keys are radix-node hash values, so a
`ChunkCache` has nothing to key on — `'ChunkCache' object has no attribute
'prefetch_from_storage'`. An implementation dependency, not a law.

**Why the patch is scoped wider than the crash.** kvd on a PD decode leg is
**write-only in every configuration**, MTP or not:
`Scheduler._add_request_to_queue` calls `_prefetch_kvcache` on the NULL and
PREFILL branches only, and that method is the sole caller of
`prefetch_from_storage`. Measured in G0, with the decode leg holding a full
`HiRadixCache` (the most favourable case): prefill **102 sets / 102 gets**,
decode **180 sets / 0 gets** — 318 MB of host memory and the D2H bandwidth to
move it, for zero reads.

**What.** Skip kvd wiring entirely when `disaggregation_mode == "decode"`, at
INFO. The decode leg then gets no hicache flags at all, so the collision above
cannot arise either.

**How.** Self-locating idempotent Python, all infera copies. Four tests
(`test_decode_leg_gating.py`) asserting the **observable** effect — which flags
reach the sglang subprocess argv — rather than the presence of the helper. See
`../notes.md` §8 for why that distinction mattered.

**Open question, deliberately left open.**
`--disaggregation-decode-enable-offload-kvcache` drives a separate decode-side
mechanism (`DecodeKVCacheOffloadManager`) that also requires a hicache storage
backend. We never enable it and did **not** check whether it reads back from L3.
This patch disables that path too. Full analysis:
`../notes.md` §5.

---

## Why bytecode verification

Python caches compiled modules in `__pycache__` keyed on source mtime. A patch
script that restores a backup with `shutil.copy2` preserves the original mtime,
so an edited `.py` can match a stale `.pyc` and CPython silently runs the
**unpatched** bytecode. **This has already invalidated a full experiment on this
stack** — the source showed the fix, the runtime did not have it.

So `apply_all_in_container.sh` drops `__pycache__`, recompiles, and greps the
fresh `.pyc` for an identifier the patch introduces. It greps for **identifiers,
never comment markers**: the compiler discards comments, so a comment marker
reads as a false negative.

Three things cannot be checked in bytecode and are source-checked instead, each
with the reason recorded at the check: the nextn prerequisite (an f-string split
across constants), patch 2a (changes an expression, introduces no new
identifier), and the msgspec field type (a lazily-evaluated annotation).

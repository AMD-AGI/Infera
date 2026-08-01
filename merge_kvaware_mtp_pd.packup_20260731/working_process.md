# Merge validation — working process

Goal: one image carrying **kvaware+kvd** ∪ **mtp+dpa+pd (DSA patches)** ∪
**pd+mooncake early-send fix**, validated end to end on chi2879 (prefill) +
chi2867 (decode), GLM-5.2-MXFP4, 2-node PD over mooncake RDMA (ionic).

Base image: `infera/engine-sglang:kvaware-kvd`
(`sha256:c9292019d0c5a36d97fb87e7b2944b3b6382bab2f33410b89ed03103192a0d80`),
already present on both nodes. Patched **in-container** for this experiment
phase; the same three scripts become a Dockerfile layer once green.

## Patch set

Started as 5, ended as 7: patches 6 and 7 were written during R2, when turning MTP
on exposed two pre-existing infera auto-appends that no prior run had exercised.

| # | patch | from | fixes |
|---|---|---|---|
| 1 | `dsa_indexer_hip_dp_padded_rows.diff` | PR58 | HIP paged-MQA DP-padded rows |
| 2 | `dsa_backend_dp_sync_and_page_table_rows.diff` | PR58 | DP host-sync + MTP page-table rows |
| 3 | `draft_cuda_graph_dp_vote.diff` | PR58 | per-rank draft graph/eager divergence → deadlock |
| 4 | `patch_mooncake_early_send_wait_event.py` | PR56 | non-final prefill chunk RDMA-read races the forward |
| 5 | `patch_infera_kvevent_bigram.py` | PR56 (re-cut as a self-locating script) | kv-aware view empty under MTP (bigram token pairs) |
| 6 | `patch_infera_decode_radix_vs_mtp.py` | **new, R2** | kvaware's decode-radix append is rejected under EAGLE |
| 7 | `patch_infera_decode_kvd_skip.py` | **new, R2** | kvd is write-only on a PD decode leg |

Prerequisite for 1–3: GLM-5.2 nextn `eh_proj` quark-exclude — **already in the
image** (verified: `deepseek_nextn.py:363`), asserted by the apply script.

Patches 5–7 edit infera's own code and belong as **source commits**, not build
layers; they ship as scripts here only so the running containers could pick them
up without a rebuild. `deliverable/infera_source_changes.diff` is their committed
form, with tests.

## Deferred (TODO, not blocking)

- `dsv4_gfx942` architecture-based detection (PR56) — `apply_gfx942_dsv4` returns
  early on non-gfx942, so it is a **no-op on MI355X**. Needed only for MI325X.
- `INFERA_SGLANG_READY_TIMEOUT` (PR56) — convenience; 1800 s has been enough here.
- Rust router bigram decode (PR56, `rust/router/src/kv_event.rs`) — every run uses
  `--router-backend python` (the default).

## Validation gates

| gate | config | pass criterion |
|---|---|---|
| **G0** | kvaware + kvd + DPA + PD, **MTP off** | 4/4 correctness; kvd restart-replay `gets>0`, `hits>0`, `sets` unchanged |
| **G1** | G0 + **MTP on decode leg** | 4/4, `acc_len>1`; router KV view non-empty (prefix reuse measurable) |
| **G2** | G1 + prompt spanning >1 prefill chunk | needle correct at every depth (no chunk-boundary corruption) |
| **stress** | after G2 | conc=16 then conc=128, 0 HTTP errors |

## Rounds

| round | purpose | outcome |
|---|---|---|
| R0 | apply all 5 patches in-container on both nodes; launch G0 (MTP=0) | patches ✅ verified in bytecode on both nodes; G0 legs launched 12:33 UTC |
| R1 | **G0** — kvaware+kvd+DPA+PD, MTP off: did the patches break the baseline? | **PASS**, see below |
| R2 | **G1** — same + MTP on the decode leg | 2 crashes → 2 new infera patches → **PASS**, see below |
| R3 | **G2** — G1 + a prompt spanning >1 prefill chunk (mooncake early-send) | **PASS** 5/5, see below |
| R4 | conc=16 then conc=128 under the full merged config | **PASS** 64/64 and 256/256, see below |

### R1 — G0 result (baseline replay, MTP off)

Command line is byte-identical to the kvaware/kvd baseline (`MTP=0` ⇒ no
`MTP_ARGS`, no `--disable-custom-all-reduce`), so any change here is attributable
to the 5 patches alone.

| check | result |
|---|---|
| 4-prompt correctness probe | **4/4** |
| prefix reuse, cold + reuse phase | **32/32** correct (16+16) |
| kvd wiring | `--enable-hierarchical-cache` appended, `hicache-storage-backend infera`, **8× `infera-kvd adapter connected`**, `KV plane up` |
| both legs registered in etcd | prefill + decode, `dp_size=8` each |
| `Traceback` either leg | **0** |

**kvd attribution (restart-and-replay).** The prefill engine was restarted (190 s)
while the kvd daemon and its L3 kept running — that empties the in-GPU radix cache,
so any reuse afterwards can only come from L3:

| | gets | hits | sets | misses |
|---|---:|---:|---:|---:|
| after first reuse run | 0 | 0 | 102 | 0 |
| **after restart + replay** | **102** | **102** | **102 (unchanged)** | 0 |

102 reads, zero new writes, zero misses — same signature as the baseline packup.

**Router KV view is populated** (the G1 baseline to diff against):
`/v1/admin/cache-view/<worker>?dp_rank=N` summed over 8 ranks —
prefill **51** blocks, decode **90** blocks. Under MTP without the bigram fix
these must read **0**; that is the discriminator R2 will use.

### R0 detail

**Reset ritual**: both nodes torn down (old `kvaware_kvd_final` container + engine
procs killed, GPUs confirmed idle), fresh container from
`infera/engine-sglang:kvaware-kvd`, 8× `PORT_ACTIVE` verified inside each
container before anything else.

**Patch verification** (identical on chi2879 and chi2867):

```
PREREQ nextn eh_proj      -> src=1        patch2a max_seqlen_k -> src=1
dsa_indexer :: _p1v2_trim                       pyc=1
dsa_backend :: _glm52_match_page_table_rows     pyc=1
dp_attn / eagle_worker_v2 / eagle_draft_cuda_graph_runner /
  forward_batch_info / schedule_batch / decode  pyc=1..2   (patch 4, 6 files)
common/utils.py :: wait_event                   pyc=1
mooncake/conn.py :: _early_send_wait_event      pyc=1   + synchronize() src=1
prefill.py :: _early_send_wait_event            pyc=1
/opt/infera/infera/...client.py :: _flat_tokens pyc=2
/opt/infera/infera/...events.py bigram type     src=1
_flat_tokens smoke                              OK
```

**Trap hit and fixed**: the image carries **two** infera copies —
`/opt/venv/.../site-packages/infera` and `/opt/infera/infera` (the image WORKDIR,
which shadows site-packages for any process started there, i.e. every
`docker exec`). The first patch run edited only what `find_spec` returned and the
verification then imported the *other* copy and failed with `ImportError:
cannot import name '_flat_tokens'`. `patch_infera_kvevent_bigram.py` now patches
every copy it can find. Worth carrying into the Dockerfile layer.

### R2 — G1 result (MTP on the decode leg)

Two crashes before it ran. Neither is a conflict between the merged workstreams —
nothing they touch overlaps. Both are **pre-existing infera code meeting a
configuration nobody had run**: the kvaware/kvd validation never enabled MTP, and
the PD+DPA+MTP validation drove `sglang.launch_server` directly, bypassing the
infera wrapper that does the appending. The merge is simply the first time both
switches are on at once.

Two independent auto-appends, two independent gates, so **two patches** — neither
subsumes the other:

| infera appends | in | gated on | SGLang check it hits |
|---|---|---|---|
| `--disaggregation-decode-enable-radix-cache` | `args.py:255` | **kvaware** (`enable_kv_events`) | `pd_disaggregation_hook.py:41` — forbidden with `--speculative-algorithm` |
| `--enable-hierarchical-cache` | `kvd_wiring.py:51` | **kvd** (`infera_kvd_socket`) | `server_args.py:5772` — forbidden with `disable_radix_cache` |

The second is reached *because* of the first: rejecting the decode radix cache
forces `disable_radix_cache = True` (`pd_disaggregation_hook.py:56`), which then
collides with the hierarchical-cache flag.

- `patches/patch_infera_decode_radix_vs_mtp.py` — skip the decode-radix append
  under speculative decoding, and log why.
- `patches/patch_infera_decode_kvd_skip.py` — do not wire kvd on a PD decode leg
  at all. Scoped wider than the crash on purpose: kvd is **write-only** there in
  every configuration. See `notes/decode_leg_kvd_is_write_only.md`.

**Result** — prefill MTP off / decode MTP on, kvaware both legs, kvd prefill only:

| check | G0 (MTP off) | G1 (MTP on) | verdict |
|---|---|---|---|
| 4-prompt correctness | 4/4 | **4/4** | ✅ |
| prefix reuse | 32/32 | **32/32** | ✅ |
| MTP genuinely active | n/a | **`accept len: 2.48`, `2.58`** (> 1) | ✅ |
| `Traceback` either leg | 0 | **0** | ✅ |
| router cache-view, **prefill** | 51 blocks | **51 blocks** | ✅ bigram fix works |
| router cache-view, **decode** | 90 blocks | **0 blocks** | expected — kvd off there |
| prefill kvd | 102 sets / 102 gets | **102 sets (unchanged) / 204 gets** | ✅ L3 still serving |
| decode kvd | 180 sets / 0 gets | **unchanged** — no new writes | ✅ patch effective |

**The prefill 51 is the discriminator for the bigram fix.** `is_eagle` is a global
server arg, so with MTP on the prefill leg's radix keys are bigrams too and its
kv-events carry `(t[i], t[i+1])` pairs. Without `_flat_tokens` the router would
hash the pairs and the view would read **0**. It reads 51 — byte-identical to the
plain-int path in G0, which also shows the flattened keys chain to the same
hashes rather than merely being non-empty.

Measured *after* a prefix-reuse run: the view lives in the router process, so a
freshly restarted router reads 0 for a trivial reason. The first G1 sample was
taken too early and showed 0/0; re-measured after driving traffic.

`accept len` is logged as `accept len: N`, not `accept_len=N` — the first grep
used the wrong pattern and reported "MTP absent" when it was running fine.

### R3 — G2 result (long prompt across prefill chunks)

Same G1 configuration. 23797-token prompts, needle placed at 5 depths, on top of
the merged patch set (so with the mooncake early-send wait event in force).

**The prompt really was chunked** — from the prefill engine's own log, two of the
needle requests show `new-token: 8192, 8192, 1728` and `8192, 3904, 6080`. Three
chunks at the 8192-per-rank boundary. Without this check a green needle result
would be worthless: chunked prefill quietly not engaging looks identical to a fix.

| depth | needle | prompt tok | completion tok | finish | `</think>` | found |
|---|---|---:|---:|---|---:|---|
| 0.0 | 10000 | 23797 | 85 | stop | 1 | ✅ |
| 0.25 | 17919 | 23797 | 142 | stop | 1 | ✅ |
| 0.5 | 25838 | 23798 | 142 | stop | 1 | ✅ |
| 0.75 | 33757 | 23798 | 155 | stop | 1 | ✅ |
| 1.0 | 41676 | 23798 | 79 | stop | 1 | ✅ |

**5/5.** Every response terminated on its own (`finish=stop`) with `</think>`
appearing exactly once — the corruption signature this patch exists to prevent is
a truncated needle followed by `</think>` repeating dozens of times.

**A false alarm on the first attempt, worth recording.** The probe initially ran
with `max_tokens=256` and reported **2/5**, with `</think>` repeating 144–240
times — exactly the documented corruption signature. It was an artifact: 256 cut
the model off mid-reasoning and the run-on tail is what fills the rest. Re-running
with 2048 gives 5/5 at 79–155 completion tokens, i.e. nowhere near either cap.

The lesson is that `</think>` repetition alone does not discriminate — it is
produced both by KV corruption and by truncating a reasoning model. `finish_reason`
is what separates them, and the probe now records it. A depth sweep helps too: real
chunk-boundary corruption spares depth 1.0 (the final chunk goes through the
sampling path, which already synchronizes), while truncation hits all depths.

### R4 — stress under the full merged configuration

Same G1/G2 legs. ISL/OSL 1024, needle-in-log prompts, temp=0.

| run | conc | requests | CLEAN | TAIL_REPEAT | BAD | duration |
|---|---:|---:|---:|---:|---:|---:|
| `stress_c16.json` | 16 | 64 | **64** | 0 | **0** | 18.5 s |
| `stress_c128_v2.json` | 128 | 256 | **250** | 6 | **0** | 22.5 s |

`TAIL_REPEAT` = needle retrieved, `finish=stop`, only the post-`</think>` tail
loops. Not a correctness failure; the BAD tally is `DIGIT_LOOP + CORRUPT_REASONING`.

**The classifier had two defects, both found here and both fixed.** The first
conc=128 run reported `CORRUPT_REASONING 1/256`. It was a false positive:

```
idx=239 expect=12662 finish=stop ctok=283  FOUND_IN_TEXT=True
1.  **分析请求：** ... 记录 SECRET-239：“轨道陀螺仪的校准常数正好是 12662。”
    ...</think>Based on the log, Record SECRET-239 explicitly states that the
    calibration constant for the orbital gyroscope is exactly 12662.
```

Coherent, correct, self-terminated — GLM-5.2 simply reasoned in Chinese on an
English prompt, and the classifier had a `>5 CJK chars → salad` rule.

Fixing that exposed the worse defect underneath: the documented corruption
signature `2183</think>2183</think>218</think>218</think>` was classified
**WRONG**, and WRONG is not in the BAD tally. The word-repetition regex cannot see
it (the digits differ between tags, so no token is adjacent to a copy of itself).
The one failure mode this gate exists to catch was invisible.

Changes to `scripts/stress_capture.py`:
- drop the CJK rule — script choice is not a corruption signal;
- check "needle present AND `finish=stop`" *before* the heuristics, so a
  stylistic signal cannot masquerade as KV corruption;
- add `</think>` count ≥ 3 as a salad signal, which is what actually catches the
  chunk-boundary mode.

8 classifier unit cases, including both real corruption shapes and the coherent
Chinese response, pass. conc=128 re-run under the fixed classifier: **250 CLEAN /
6 TAIL_REPEAT / 0 BAD**.

## Result — all gates pass

| gate | criterion | result |
|---|---|---|
| **G0** | patches do not break kvaware+kvd | 4/4, 32/32, kvd 102 gets / 102 hits after restart |
| **G1** | + MTP on decode | 4/4, 32/32, `accept len` 2.48–2.58, router view 51 blocks |
| **G2** | + prompt across prefill chunks | 5/5 needles, prompt really split 8192+8192+1728 |
| **stress** | conc=16, conc=128 | 64/64 and 256/256, 0 BAD |

## Deliverable

`deliverable/` — the Dockerfile carrying all three workstreams, the two patch
directories it applies, and `infera_source_changes.diff` (4 infera files + 3 new
test modules, 14 tests). See `deliverable/README.md` for layer order and rationale.

**Tests, and the discipline applied to them.** Each of the three infera fixes was
reverted in place and its test module re-run, to confirm the tests fail without the
fix — `scripts/revert_check_tests.sh` automates this.

The first pass reported all three "failing on reverted code", but two of those were
**collection errors**: the test imported the new symbol by name, so reverting the
fix produced `ImportError` and every test in the module errored out. That proves
the symbol is absent, not that the behaviour is wrong — a test that cannot run is
not a test that fails. Both were rewritten:

- `test_kv_event_bigram.py` — fetch `_flat_tokens` via `getattr` and skip only the
  helper's own unit tests, so the three behavioural tests still execute on pre-fix
  code. They now fail for three distinct reasons: wrong hashes, bigram and plain
  views disagreeing, and the wire decode rejecting pairs.
- `test_decode_leg_gating.py` — assert the **observable** effect (which flags reach
  the sglang subprocess argv) instead of importing the guard helper.

After the rewrite all three fail behaviourally (`1–2 failed`, no collection error).

Full unit suite after the changes: **1161 passed, 1 skipped** (`tests/unit`, minus
`gaie` which needs grpc). The container image ships only `infera/`, not `tests/`,
so the suite runs on the dev box; the two sglang-dependent modules run in the
container, where they pass 14/14.

**The image itself has not been built.** Everything above was validated by
in-container patching, with each patch verified in the bytecode on both nodes. The
Dockerfile runs the same scripts in the same order, but that is an argument rather
than a measurement — build it and re-run G0–G2 before shipping.

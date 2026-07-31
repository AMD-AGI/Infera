# Notes — gotchas, traps, and the wrong turns

Ordered roughly by how likely each is to bite a reproducer.

---

## 1. Cold start is ~5–9 minutes and is not a hang

Weight load (~35 s) plus CUDA-graph capture. On this stack a leg that has printed
nothing for five minutes is normal. **Poll, do not kill:**

    grep -ac "ready to roll" $LOG      # 1 = up
    grep -ac Traceback       $LOG      # 0 = healthy

Four cold starts are on the critical path of a full reproduction (G0 ×2, the G0
restart, G1 ×2 — G1's prefill restarts too), which is where most of the ~2 h goes.

---

## 2. The reset ritual between rounds is mandatory

**What.** `node_reset_and_patch.sh` tears down the container and engine
processes, **waits for the GPUs to return to idle**, starts a fresh container,
and verifies 8 `PORT_ACTIVE` inside it before anything else.

**Why.** Carrying GPU state between RDMA rounds is the top cause of wasted hours
on this stack. Each skipped step maps to a concrete failure: skip the
memory-release wait → OOM mid-run that looks like a regression in your latest
change; skip the `PORT_ACTIVE` check → the libionic injection silently failed,
mooncake drops to TCP, and the run "works" while measuring nothing.

**Context.** The `PORT_ACTIVE` check is cheap and catches the failure that is
hardest to diagnose from downstream symptoms.

---

## 3. The image carries TWO infera copies, and the wrong one wins

**What happened.** The first patch run edited only what `importlib.find_spec`
returned — `/opt/venv/lib/python3.10/site-packages/infera`. Verification then
imported the *other* copy and failed:

    ImportError: cannot import name '_flat_tokens' from 'infera.router.kv_event.client'
    (/opt/infera/infera/router/kv_event/client.py)

**Why.** `/opt/infera` is the image's `WORKDIR`. Any process started there — i.e.
**every `docker exec`** — gets `/opt/infera/infera` ahead of site-packages on
`sys.path`.

**How it was fixed.** All three infera patch scripts now enumerate every copy
they can find (`find_spec` result plus `/opt/infera/infera`) and patch each.

**Context.** Worth carrying into the Dockerfile layer, and worth checking in any
image where a package is both pip-installed and present as a source tree. The
verification block prints which copies it found, so a future single-copy image
degrades cleanly.

---

## 4. The router KV view is per-process — measure it after driving traffic

**What.** `/v1/admin/cache-view/<worker>?dp_rank=N` reads a view held in the
**router process**. A freshly restarted router reports 0 blocks for every worker.

**Why it matters here.** That count is the discriminator for the bigram fix. The
first G1 sample was taken right after restarting the router and read 0/0 — which
looks exactly like the fix having failed. Re-measured after `prefix_reuse.py`: 51
blocks on prefill, matching G0.

**How to avoid.** Always: restart router → drive traffic → *then* read the view.
`scripts/cache_view.sh` carries this warning in its header.

---

## 5. kvd on a PD decode leg is write-only — in every configuration

**What.** SGLang never issues a storage prefetch on a PD decode worker. The
backup path still runs, so L3 fills and is never read.

**Why — the source.** `scheduler.py:2309`:

```python
def _add_request_to_queue(self, req, is_retracted=False):
    if   self.disaggregation_mode == DisaggregationMode.NULL:    self._prefetch_kvcache(req)
    elif self.disaggregation_mode == DisaggregationMode.PREFILL: self._prefetch_kvcache(req)
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
        # <-- no _prefetch_kvcache call
```

`_prefetch_kvcache` (`scheduler.py:2284`) is the **only** caller of
`tree_cache.prefetch_from_storage(...)`, and it is itself called from exactly
those two places.

**Why — the measurement.** G0, kvd on **both** legs with separate daemons, MTP
off — so the decode leg had a full `HiRadixCache`, the most favourable case:

| leg | node | sets | gets | hits | bytes |
|---|---|---:|---:|---:|---:|
| prefill | chi2879 | 102 | **102** | **102** | 180 MB |
| decode | chi2867 | 180 | **0** | **0** | 318 MB |

The prefill numbers come from the restart-and-replay test. The decode leg was
never restarted, so its `gets=0` alone would be ambiguous — the in-GPU cache
could have absorbed everything. The source read is what removes the ambiguity:
no code path could have produced a get.

**Context — this is not the MTP restriction.** The merge hit it from the MTP
direction via a real but different chain (patches 6 and 7 in
`patches/README.md`), which made it *look* like "kvd and MTP are incompatible on
the decode leg". The weaker precondition already fails.

**Open question, not resolved.** `disaggregation_decode_enable_offload_kvcache`
drives a separate mechanism, `DecodeKVCacheOffloadManager`
(`disaggregation/decode.py:1984`, constructed at `scheduler.py:465`), which
requires `hicache_storage_backend`. **We never enable it and did not test it.**
Whether it reads back from L3 — and therefore whether the decode-leg kvd skip
needs an exception carved for it — is unchecked. Read that manager first if you
need the flag.

**Not reported upstream.** Worth raising as a question rather than a bug report:
the fix could reasonably be either wiring the prefetch or documenting that decode
should not enable hicache storage.

---

## 6. Two probe defects that each produced a wrong verdict

Both were in **our measurement tooling**, not the system under test. Both would
have been believed if the numbers had been slightly less surprising.

### 6a. `max_tokens=256` manufactures the corruption signature

**What.** G2 first reported **2/5** needles, with `</think>` repeating 144–240
times — exactly the documented chunk-boundary corruption signature.

**Why it was wrong.** 256 cut the reasoning off mid-thought; the run-on tail is
what fills the rest. Re-run at 2048: **5/5**, completions 79–155 tokens, nowhere
near either cap.

**How to tell them apart.** `</think>` repetition alone does **not**
discriminate — both KV corruption and truncating a reasoning model produce it.
Two things separate them, and the probe now records both:

- `finish_reason`: `length` means we cut it off; `stop` means the model chose to.
- A **depth sweep**: real chunk-boundary corruption spares depth 1.0 (the final
  chunk goes through the sampling path, which already synchronizes), while
  truncation hits every depth uniformly.

### 6b. The stress classifier hid the one failure mode it existed to catch

**What.** conc=128 first reported `CORRUPT_REASONING 1/256`. The response:

    idx=239 expect=12662 finish=stop ctok=283  FOUND_IN_TEXT=True
    1. **分析请求：** ... 记录 SECRET-239：“轨道陀螺仪的校准常数正好是 12662。”
       ...</think>Based on the log, Record SECRET-239 explicitly states that the
       calibration constant for the orbital gyroscope is exactly 12662.

Coherent, correct, self-terminated. GLM-5.2 simply reasoned in Chinese on an
English prompt, and the classifier had a `>5 CJK chars → salad` rule.

**The worse defect underneath.** Removing that rule exposed it: the documented
signature `2183</think>2183</think>218</think>218</think>` was classified
**WRONG**, and `WRONG` is not in the BAD tally. The word-repetition regex cannot
see it — the digits differ between tags, so no token is adjacent to a copy of
itself. **The one failure mode this gate exists to catch was invisible.**

**How it was fixed** (`scripts/stress_capture.py`):

- dropped the CJK rule — script choice is not a corruption signal;
- check "needle present AND `finish=stop`" *before* the heuristics, so a
  stylistic signal cannot masquerade as KV corruption;
- added `</think>` count ≥ 3 as a salad signal — this is what actually catches
  the chunk-boundary mode.

Eight classifier unit cases now cover both real corruption shapes and the
coherent-Chinese response. conc=128 re-run: **250 CLEAN / 6 TAIL_REPEAT / 0 BAD**.

**Context — the general lesson.** A green result from a classifier you have not
adversarially tested is worth very little. Both defects were found only because a
result looked *odd*, not because anything failed.

---

## 7. Deferred from PR56 — what is NOT in this merge

| item | why deferred | when it matters |
|---|---|---|
| `dsv4_gfx942` architecture-based detection | `apply_gfx942_dsv4` returns early on non-gfx942 → **no-op on MI355X** | MI325X, where GLM-5.2 (`glm_moe_dsa`, `index_topk` 2048) would otherwise be handed the dsv4 knobs |
| `INFERA_SGLANG_READY_TIMEOUT` | convenience; 1800 s has sufficed | very large checkpoints, or both PD legs loading off one filesystem |
| `net.py` NodePort-range skip | vultr is bare metal, no kube-proxy, `ip_local_port_range=32768 60999` — cannot be hit here | any Kubernetes deployment |
| **Rust router bigram decode** | every run uses `--router-backend python` (the default) | **a Rust-router deployment with MTP still has the original bug** — kv-aware silently degrades to round-robin |

The Rust one is the sharp edge: it is a real, unfixed instance of the same bug,
invisible in exactly the same way, and only the default backend choice keeps it
out of scope here.

---

## 8. A test that cannot run is not a test that fails

**What.** The revert check (`scripts/revert_check_tests.sh`) reverts each infera
fix in place and re-runs its test module, asserting it fails. The first pass
reported all three "failing" — but two were **collection errors**: the test
imported the new symbol by name, so reverting produced `ImportError` and every
test in the module errored out.

**Why that is not good enough.** It proves the symbol is absent, which is trivial.
It says nothing about whether the behaviour is wrong — and behaviour is what the
fix is for.

**How both were rewritten:**

- `test_kv_event_bigram.py` — fetch `_flat_tokens` via `getattr` and skip only the
  helper's own unit tests, so the three behavioural tests still execute on
  pre-fix code. They now fail for three distinct reasons: wrong hashes, bigram
  and plain views disagreeing, and the wire decode rejecting pairs.
- `test_decode_leg_gating.py` — assert the **observable** effect (which flags
  reach the sglang subprocess argv) instead of importing the guard helper.

After the rewrite all three fail behaviourally. Full unit suite after the
changes: **1161 passed, 1 skipped**.

---

## 9. Cluster hygiene

These are shared nodes. Other people's containers (`mlperf_gptoss2`,
`primus_train`, the `robust-*` exporters) were running throughout and were left
alone. Nothing outside `merge_g0` / `merge_g0_etcd` was stopped, and no images
were pruned. Before removing any container on `chi28xx`, prove it is yours via
`docker inspect` (Binds / Env / Created).

The original scratch workspace `work.merge_20260731/` and the on-node staging dir
`/mnt/vast/c_huggingface/merge_20260731/` are both intact — this packup is a copy.

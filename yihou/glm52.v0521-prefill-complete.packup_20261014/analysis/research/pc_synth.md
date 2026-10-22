# prefill_complete usage guide for our harness (GLM-5.2 P4D4 same-node, v0.5.21 image)

Path conventions:
- `S` = `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/srt/`
- `H` = `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness/`

Where a report and its verify pass disagreed, I used the verify pass.

## (A) What it does

1. Decode does not reserve a request's KV pages, req slot or metadata buffer until prefill has finished computing that request.
   - On arrival, decode sends a BOOTSTRAP message and parks the request in the prealloc queue with `waiting_for_input=False` (`S/disaggregation/common/conn.py:2099-2113`).
   - `pop_preallocated` skips such requests (`S/disaggregation/decode.py:1361-1366`).
2. Prefill admits Bootstrapping requests as optimistic, with no cap on attempts (`S/disaggregation/prefill.py:516-528`).
   - The admission is still gated on: not retracted, the SWA/write-through clause, and a free metadata buffer.
   - A chunked request does not yield to other requests (`prefill.py:1394-1404`).
   - It never falls back to the bootstrap queue (`prefill.py:1719-1726`).
3. On the last chunk, prefill calls `mark_prefill_complete()`, which sends one `WaitingForInput` to decode (`prefill.py:1049-1053`; `common/conn.py:1847-1855`).
   - If decode's BOOTSTRAP arrives after prefill completes, `register()` sends this message instead (`S/disaggregation/common/bootstrap.py:66-75`).
   - The finished request is then parked in `disagg_prefill_inflight_queue`, still holding its device KV (`prefill.py:1180-1187`).
4. Decode then allocates through the normal admission path (formula unchanged, `decode.py:485-494`) and sends the destination indices.
   - Prefill unparks the request and sends the whole KV in one `send_kv_chunk(last_chunk=True)` (`prefill.py:1349-1366`, `:1183-1185`).
   - So KV transfer no longer overlaps with chunked prefill.
5. Prefill releases its KV only when the sender reports Success (`prefill.py:1191-1199`).
   - The sender-side `MooncakeKVSender.poll` holds `Transferring` until worker cleanup finishes (`S/disaggregation/mooncake/conn.py:3010`, `:3066-3080`).
   - This hold is on prefill, not decode, so it lengthens how long prefill holds KV.

Provenance: upstream PR #40703, in our image (`S/../_version.py` = `0.5.21.dev20261008+g11b1fb21de`, which is 285 commits ahead of `5b5d7212`). It is opt-in.
- The PR's own benchmark: output tok/s +26.8%, TTFT p95 −48.9%, ITL p99 +92.8% (worse).
- The PR's own caveat: "not a general throughput improvement."

## (B) Flags per leg and how to pass them

**Required on both legs:** `--disaggregation-decode-allocation-policy prefill_complete` (`S/arg_groups/fields/disagg.py:59-64`).
- If the legs disagree, startup still succeeds but every request fails with `policy mismatch` (`common/conn.py:2056-2063`).
- The harness has no knob for this flag (grep finds no `allocation` in `engine.sh`, `config.sh` or `config.yihou.*.sh`).
- Pass it with `PREFILL_EXTRA_ARGS_ADD` / `DECODE_EXTRA_ARGS_ADD`:
  - they are appended at `H/config.yihou.v0521-base.sh:243-244`;
  - picked up at `H/bench-harness/engine.sh:50,63`;
  - word-split into the command at `engine.sh:321-323`.

**Forced second change on prefill:** `PREFILL_HICACHE_WRITE_POLICY=write_back`.
- The base default is `write_through` (`base.sh:98`, also `bench-harness/config.sh:56`), emitted at `engine.sh:305`.
- With `write_through`, startup fails: `ValueError: --disaggregation-decode-allocation-policy prefill_complete does not yet support prefill HiCache other than L2 write_back` (`S/arg_groups/pd_disaggregation_hook.py:231-238`, `:244-247`).

**Set automatically, nothing to pass:** prefill `optimistic_prefill_attempts` goes 0 → 1 (`pd_disaggregation_hook.py:49-55`). It survives with `write_back` (`serving_hook.py:476-495`). If it were resolved back to 0, startup would fail at `serving_hook.py:497-505`.

**Must stay at default:**
- `--disaggregation-decode-host-receive-threshold 0.0` (`disagg.py:179-182`; checked at hook `:224`).
- `DECODE_HICACHE=0` (`base.sh:92`).
- No decode radix cache (`disagg.py:85-88`).
- `SGLANG_DISAGG_STAGING_BUFFER` and `SGLANG_RUST_SERVER` unset (`environ.py:803`, `:1871`).

**Overlay, in this order.** Set everything after sourcing, because `YIHOU_MOUNTS` is assigned unconditionally at `base.sh:80`.
```
source <q4 config: config.yihou.mc4340n.sh, or v0521-base.sh + mc4340 chain>
YIHOU_MOUNTS=<...>/mounts.yihou.v2-regbase.tsv        # as config.yihou.mc4340.sh:33-36 does
PREFILL_EXTRA_ARGS_ADD="${PREFILL_EXTRA_ARGS_ADD} --disaggregation-decode-allocation-policy prefill_complete"
DECODE_EXTRA_ARGS_ADD="${DECODE_EXTRA_ARGS_ADD} --disaggregation-decode-allocation-policy prefill_complete"
PREFILL_HICACHE_WRITE_POLICY=write_back
PREFILL_HICACHE_SIZE=370          # base default 450 (base.sh:97); emitted at base.sh:238
PREFILL_MEM_FRACTION=0.88         # base 0.85
DECODE_MEM_FRACTION=0.96          # base 0.95 (base.sh:88; engine.sh:56,261)
DECODE_MAX_RUNNING=96
CONC=48                           # base 44 (base.sh:197)
HICACHE_HOST_REGISTER_CHUNK_GB=<same as q4>   # 512 is the tested value (config.yihou.E2-target.sh:7-12); prefill env only (base.sh:248-249)
```
- Keep the `${...}` prefix on both EXTRA_ARGS lines. Base already uses these hooks for `--hicache-size` and `--disable-custom-all-reduce` (`base.sh:237-238`), so overwriting them would drop those args. [inferred]
- q4 (`H/config.yihou.mc4340n.sh`; `H/dryrun.q4-137.yihou.txt`) already had CONC 48, decode 0.96, decode max_running 96, prefill 0.88, prefill max_running 64 and chunk 32768. Relative to q4, this run changes only three things: the allocation policy, `write_through`→`write_back`, and hicache 450→370.
- Run `DRY_RUN=1` first and diff the result against `dryrun.q4-137.yihou.txt`. Expect exactly those three deltas, with the policy flag present on both legs.
- `BOOTSTRAP_TIMEOUT` (default 1800) drives both `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (`engine.sh:215-216`). They cannot be tuned separately without an extra env entry.

## (C) Incompatibilities and risks for our setup

**EAGLE MTP 5/1/6 on decode:** not rejected. Only decoupled speculation is excluded (hook `:217-218`).

**DPA TP4/DP4:**
- The startup check is `tp_size != (dp_size if enable_dp_attention else 1)` (hook `:213-215`). It passes only if DPA is on for both legs; the engine flag was not re-read.
- At runtime the peer must have attention TP1/CP1/PP1 (`common/conn.py:2065-2068`). With DPA, attn_tp = 1. [inferred]

**KV pool state types:**
- KV-manager init raises if any state type is not `DSA`: `prefill_complete currently supports full-attention and DSA KV pools only` (`common/conn.py:375-379`).
- `DSA_TAIL` is added only when `index_kpool>1 && index_kpool_compress` (`mem_cache/memory_pool.py:5219`, `:5272-5274`; `disaggregation/utils.py:1564-1576`).
- The GLM-5.2-MXFP4 `config.json` has no `index_kpool*` keys, and `index_kpool_compress` defaults to False (`configs/model_config.py:446-447`). So the check should pass. [inferred]
- Unverified: whether the fp8_e4m3 KV (`base.sh:118`) lands in a non-DSA pool type such as `MHATokenToKVPoolMXFP8`/`BLOCK_SCALE` (`utils.py:1358-1361`).
- The first launch settles both questions.

**HiCache 370 with write_back:**
- `write_back` is a second variable that the code forces on us (see notes rule 4).
- Under `write_back`, the checkpoint uses `checkpoint_kv_cache` at every chunk and again at unpark (`prefill.py:594-599`, `:1384`, `:1363`).
- The policy code does not reference the host size.
- `--hicache-size` is GB per process (`mem_cache/pool_host/base.py:211-214`), so prefill's host total is about 4×370 GB. [inferred]
- `engine.sh:304` still emits `--hicache-ratio`. That size>0 overrides it is per the base comment only (`base.sh:93-95`), not tested.

**Mounted files:**
- `conn.yihou-regbase.py` = image mooncake `conn.py` plus three added hunks (`82a83,133`, `382a434`, `507a560`), all diagnostic logging. The `defer_decode_allocation` branch is kept (image `:3075` → mount `:3128`).
- `kv_events_publisher` has one changed hunk (`90c95,103`, non-blocking send) and no allocation logic.
- `forward_mla_rocm` adds lines only (`158a159-162`).
- All the other prefill_complete logic is unmounted image code.

**Prefill memory pressure:**
- Parked requests keep holding prefill device KV and metadata buffers until decode allocates and the sender reports Success.
- Prefill metadata buffers = max_running × 2 = 64 × 2 = 128 per DP rank (`managers/scheduler.py:1549-1553`). When they run out, `pop_bootstrapped` stops admitting (`prefill.py:527-528`).
- If decode is the bottleneck, parked requests should build up on prefill and squeeze both the prefill pool and the evictable radix/L1 pages. [inferred, not measured]

**Timeouts:**
- Decode phase 1 covers prefill queueing plus compute, against WAITING: `Timed out waiting for prefill completion` (`common/conn.py:2124-2131`).
- Prefill's BOOTSTRAP window is measured from `_prefill_complete_time` (`:1939-1960`).
- The DeferredBootstrap TTL is WAITING+BOOTSTRAP (`:380-384`).
- The ZMQ send timeout is 1 s under this policy, versus 30 s under early (`:1458-1462`).

**Missing upstream fixes:**
- #43442 is absent, but it explicitly skips prefill_complete.
- #43719 is partially present: the poison machinery exists (`common/conn.py:368-370`, `:920-933`), but `_poison_unsettled_send` does not.
- On upgrade, #43407 flips the threshold encoding: an explicit `0.0` would then mean always-on, which conflicts with this policy.

## (D) Checks after launch to prove it is active

1. **Server-args dump:**
   - `disaggregation_decode_allocation_policy='prefill_complete'` on both legs. This is the only proof that the `infera.engine.sglang` wrapper (`engine.sh:252`) forwards the arg; that is unverified.
   - Prefill: `optimistic_prefill_attempts=1` and `hicache_write_policy='write_back'`.
2. **No startup `ValueError`** from `pd_disaggregation_hook.py:244-247`, `serving_hook.py:497-505` or `common/conn.py:375-379`. The `YIHOU-REGBASE` / state_types log should show DSA only.
3. **Prefill log line** (`managers/scheduler_components/metrics_reporter.py:740-746`). [inferred]
   - `#bootstrap-req` should stay about 0.
   - `#optimistic-req` should be about the whole batch.
   - `#inflight-req` includes parked requests and is the buildup gauge.
4. **Decode log line** (`metrics_reporter.py:997-1000`). [inferred]
   - `#prealloc-req` now includes requests still being prefilled.
   - `#transfer-req` and `pre-allocated usage` cover only finished requests, so usage should be lower than q4 at the same load.
5. **Strings that should be absent:**
   - `exhausted optimistic prefill attempts falling back to bootstrap queue` (suppressed, `prefill.py:1719-1726`).
   - Failure strings: `policy mismatch` (`common/conn.py:2060-2063`), `requires peer attention TP1/CP1/PP1` (`:2068`), `Timed out waiting for prefill completion` (`:2130`), `timed out after ... in KVPoll.Bootstrapping` (`:1958`), `in KVPoll.WaitingForInput` (`:2358`), `Duplicate decode bootstrap room` (`:2034`), `Duplicate or cancelled bootstrap room` (`:1789`).

## (E) Open questions only an experiment can answer

1. Whether the `infera.engine.sglang` wrapper passes the flag through unchanged (D1).
2. Whether the KV state types really are DSA only, i.e. no `DSA_TAIL` and not an MXFP8/BLOCK_SCALE pool.
3. Whether prefill_complete changes AgentX C48 throughput, TTFT or ITL against q4. The PR's +27% / −49% TTFT p95 / +93% ITL p99 came from a decode-KV-capped mixed 131K/4K workload. Our setup differs: decode KV is not capped and transfer is same-node HIP-IPC.
4. How much each variable contributes. The planned run changes three things against q4. Separating them needs:
   - an `early` + `write_back` + 370 control, to isolate the policy;
   - optionally `early` + `write_through` + 370, to isolate the write policy.
5. How large prefill parked-request buildup gets (`#inflight-req`, prefill token usage), and whether it hits the 128 metadata-buffer cap or squeezes the prefill pool at mem 0.88.
6. Whether 370 GB at the 256 GB default register chunk makes 2 register calls, and whether that matters. Keeping q4's chunk value avoids the question.
7. Whether the 1 s ZMQ send timeout under defer causes BOOTSTRAP send aborts at C48.
8. Whether the 1800 s timeouts are enough under phase-1 semantics, where the wait now includes prefill queueing.
# GLM-5.2 internal decode bench — TP8 / EP8 / DP-attention, concurrency sweep

## Sweep result summary (Phase 2)

Concurrency sweep C = 48, 64, 96, 128, 160, 192, 224, 256, 288, 320 at the identical configuration
(TP8/EP8/DP-attention-on, ISL 70000/OSL 10000, accept length 3.61, warmup 10,
`--mem-fraction-static 0.85`, `--max-running-requests <C>`). All 10 points measured and gate-checked
by `scripts/collect_sweep_yihou.py iterations --output results/sweep_summary_yihou.csv`:
**`exit=0`, all 10 rows `verdict=pass`, no problems.** No point failed on capacity/OOM at any C,
including C=320 (`local_batch=40`, KV reservation `40 x 80064 = 3,202,560` = 92.9% of the
3,448,128-token pool).

`verify_iterations=2768` and `realized_accept_length=3.6134393063583814` are **identical across all
10 rows** — the configuration did not drift across the sweep.

| C | local batch | TPOT (ms) | output tok/s | tok/s per GPU | decode (s) | reserved_tokens | max_memory_allocated_bytes |
|---|---|---|---|---|---|---|---|
| 48  | 6  | 12.083191042300314 | 3972.460572042904  | 496.557571505363   | 120.83186839299742 | 480384   | 253454164992 |
| 64  | 8  | 13.142841791105457 | 4869.57090538156   | 608.696363172695   | 131.42838889098493 | 640512   | 253462555136 |
| 96  | 12 | 16.237883286201395 | 5912.100629617084  | 739.0125787021356  | 162.37871910299873 | 960768   | 253479336448 |
| 128 | 16 | 17.847756853699686 | 7171.769598231988  | 896.4711997789985  | 178.47753150801873 | 1281024  | 253638683648 |
| 160 | 20 | 20.842523304204224 | 7676.613702896799  | 959.5767128620998  | 208.42523304204224 | 1601280  | 254090995712 |
| 192 | 24 | 22.725100385496624 | 8448.807562695574  | 1056.1009453369468 | 227.25098077603616 | 1921536  | 254552539648 |
| 224 | 28 | 24.21575831020018  | 9250.174912162323  | 1156.2718640202904 | 242.1573626450263  | 2241792  | 255003353088 |
| 256 | 32 | 26.251939422101714 | 9751.660472920015  | 1218.957559115002  | 262.51926388201537 | 2562048  | 255453225472 |
| 288 | 36 | 28.91812167810276  | 9959.15306000244   | 1244.894132500305  | 289.1811864929623  | 2882304  | 255903830016 |
| 320 | 40 | 30.771934552298625 | 10399.086201621223 | 1299.885775202653  | 307.71934552298626 | 3202560  | 256355343360 |

`max_memory_allocated_bytes` grows only from 253.45 GB (C=48) to 256.36 GB (C=320) — +2.9 GB across
the whole **EP8** sweep on 288 GB cards (this span was only measured on the EP8 sweep; the same
check was not done on the EP1 sweep, so no claim is made about EP1 device memory here). C=320 was
nowhere near a device-memory wall on EP8; the tight quantity was the KV-pool *reservation*
(`reserved_tokens` 3,202,560 of the EP8 pool's 3,448,128 = 92.9%), a different budget from device
memory. This is arithmetic from measured numbers, not a performance prediction, but it bounds where
the sweep would next fail on reservation rather than on device memory: C=352 would need
`44 x 80,064 = 3,522,816` tokens reserved. There are two measured pool sizes in this workspace —
EP8's 3,448,128 and EP1's 3,471,936 (see the Phase 3 section below) — and 3,522,816 exceeds **both**,
so the conclusion holds either way. "C=320 passed" should not be read as "there is lots of headroom
above 320" under either topology.

Reported as measured only: TPOT rises from 12.083 ms at C=48 to 30.772 ms at C=320, and per-GPU
throughput rises from 496.56 to 1299.89 tok/s over the same range. No mechanism for this shape is
asserted here beyond the measurement itself.

Full per-point columns (including `load_pool_capture_seconds`, `point` dir name) are in
`results/sweep_summary_yihou.csv`.

---

## Phase 3: EP8 vs EP1 (expert parallelism off) comparison

Same C sweep (48, 64, 96, 128, 160, 192, 224, 256, 288, 320) re-run with `--ep-size 1` in place of
`--ep-size 8`, every other flag byte-for-byte identical. Both sweeps ran **on the same node, in the
same container (`yihou-glm52-tp8ep8-0914`, never recreated between phases), with the same image
digest and warm AITER JIT cache** — EP8 vs EP1 is a single-variable comparison. Gate-checked by
`scripts/collect_sweep_yihou.py iterations --prefix tp8_noep_dpa_on_c --expect-ep 1 --output
results/sweep_summary_noep_yihou.csv`: **`exit=0`, all 10 rows `verdict=pass`**.

Each point above and below is a single run with no repeats — treat differences of a few percent as
within unmeasured run-to-run variation, not as attributed effects.

| C | EP8 TPOT (ms) | EP1 TPOT (ms) | TPOT delta | EP8 tok/s | EP1 tok/s | tok/s delta |
|---|---|---|---|---|---|---|
| 48  | 12.0832 | 10.5585 | -12.62% | 3972.46  | 4546.11  | +14.44% |
| 64  | 13.1428 | 11.9746 | -8.89%  | 4869.57  | 5344.65  | +9.76%  |
| 96  | 16.2379 | 14.9122 | -8.16%  | 5912.10  | 6437.67  | +8.89%  |
| 128 | 17.8478 | 17.3231 | -2.94%  | 7171.77  | 7388.97  | +3.03%  |
| 160 | 20.8425 | 19.7278 | -5.35%  | 7676.61  | 8110.39  | +5.65%  |
| 192 | 22.7251 | 21.9950 | -3.21%  | 8448.81  | 8729.24  | +3.32%  |
| 224 | 24.2158 | 24.2623 | **+0.19%** | 9250.17  | 9232.45  | **-0.19%** |
| 256 | 26.2519 | 25.6740 | -2.20%  | 9751.66  | 9971.17  | +2.25%  |
| 288 | 28.9181 | 28.3732 | -1.88%  | 9959.15  | 10150.44 | +1.92%  |
| 320 | 30.7719 | 30.4108 | -1.17%  | 10399.09 | 10522.58 | +1.19%  |

(delta = (EP1 - EP8) / EP8 x 100; negative TPOT delta / positive tok/s delta means EP1 is faster)

**Open question, not explained:** at C=224 the sign flips — EP1 is marginally *slower* than EP8
(+0.19% TPOT), the only point in the sweep where this happens; every other C has EP1 faster. No
per-stage attribution was collected to explain this sign flip, and each point is a single
unrepeated run, so this could be either a real effect specific to C=224 or run-to-run noise near a
region where the two curves are close. Flagging it rather than asserting either.

A secondary observation, also unexplained: the EP1 sweep's KV pool consistently allocated
`#tokens=3,471,936` per rank, versus EP8's `#tokens=3,448,128` — a small, consistent difference
across every EP1 point that was not investigated further.

Full per-point EP1 columns are in `results/sweep_summary_noep_yihou.csv`.

---

## Phase 1 point: C=128 (measured first, reused unchanged in the sweep)

### Result summary

Measured point (from `iterations/tp8_ep8_dpa_on_c128_yihou/result_yihou.json`, gate-checked by
`scripts/verify_point_yihou.py` — `verdict: "pass"`, `exit=0`, `problems: []`):

| Field | Value |
|---|---|
| `complete` | `true` |
| driver `exit_code` | `0` (`launch_wall_seconds=296`) |
| `useful_output_tokens` | `1280000` (= 128 x 10000) |
| topology | `tp_size=8, ep_size=8, dp_size=8, enable_dp_attention=true` |
| `local_batch_size` | `16` (= 128 / 8) |
| `realized_accept_length` | `3.6134393063583814` |
| `verify_iterations` | `2768` |
| `expected_accept_length` | `3.61` |

## Command

```
docker exec -w / yihou-glm52-tp8ep8-0914 bash -lc \
  "python3 bench/profile_decode.py \
     --model-path /perf_apps/data/models/GLM-5.2-MXFP4 \
     --result-dir <WS>/iterations/tp8_ep8_dpa_on_c128_yihou \
     --tp-size 8 --ep-size 8 --enable-dp-attention --batch-size 128 \
     --input-len 70000 --output-len 10000 --accept-length 3.61 \
     --warmup-steps 10 --enable-aiter-allreduce-fusion \
     --enable-fused-qk-norm-rope --mem-fraction-static 0.85 \
     --max-running-requests 128 \
   2>&1 | tee <WS>/iterations/tp8_ep8_dpa_on_c128_yihou/runtime.log"
```

launched via `scripts/run_tp8_ep8_dpa_yihou.sh 128` -> `scripts/run_decode.sh`, detached
(`setsid nohup ... &`), inside container `yihou-glm52-tp8ep8-0914`.

Full resolved SGLang `server_cli` (from `config_yihou.json`):
```
--model-path /perf_apps/data/models/GLM-5.2-MXFP4 --tp-size 8 --ep-size 8 --dp-size 8
--moe-a2a-backend none --max-running-requests 128 --enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope --mem-fraction-static 0.85 --enable-dp-attention
--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-num-draft-tokens 6
--speculative-eagle-topk 1 --kv-cache-dtype fp8_e4m3 --dsa-decode-backend flydsl
--dsa-prefill-backend flydsl --dsa-topk-backend aiter --cuda-graph-bs-decode 16
--cuda-graph-max-bs-decode 16 --random-seed 1234 --disable-radix-cache --skip-tokenizer-init
--disable-overlap-schedule --trust-remote-code
```

## Measured performance

| Metric | Value |
|---|---|
| TPOT (`effective_token_latency_ms_per_user`) | **17.847756853699686 ms** |
| Throughput (`output_tokens_per_second`) | **7171.769598231988 tok/s** |
| Per-GPU throughput (`output_tokens_per_second_per_gpu`) | **896.4711997789985 tok/s/GPU** |
| `verify_iterations` | 2768 |
| `raw_accept_tokens` | 1280256 |
| `num_correct_drafts` | 925952 |
| `accept_histogram` | `{"3": 136960, "4": 217344}` |

`tok/s = C x 1000 / TPOT` is one measurement in two units, not two independent results
(128000/17.8478 ~= 7171.8, consistent).

## Phase timing (`phase_seconds`, seconds)

| Phase | Value |
|---|---|
| `load_pool_capture` | 65.30290102696745 |
| `physical_initialization` | 0.3234621210140176 |
| `bootstrap` | 10.282247716968413 |
| `warmup_and_reset` | 13.339999316027388 |
| `decode` (the measured window) | 178.47753150801873 |
| `rank_total_to_result` | 267.72983751201537 |

## Memory

| Field | Value |
|---|---|
| `memory.reserved_tokens` | 1281024 |
| `memory.reserved_tokens_per_request` | 80064 |
| `memory.page_size` | 64 |
| `memory.speculative_reserve` | 12 |
| `memory.target_initialized_bytes` | 164478758400 |
| `memory.draft_initialized_bytes` | 2441319936 |
| `max_memory_allocated_bytes` | 253638683648 |
| KV cache (`runtime.log`, per rank) | `#tokens: 3448128`, dtype `fp8_e4m3fn`, `KV size: 153.18 GB` (main) + `2.27 GB` (secondary pool) |

`reserved_tokens=1281024` matches the mission's stated expected value for local batch 16 exactly
(`16 x 80064`).

## Environment

| Item | Value |
|---|---|
| Host | `smci355-ccs-aus-n06-25.prov.aus.ccs.cpe.ice.amd.com` (bare metal, 8x MI355X) |
| OS | Ubuntu 22.04.5 LTS, kernel `6.8.0-107-generic` |
| Container image digest | `sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0` (local tag `rocm-llm-bench:latest`) |
| SGLang commit | `402df1e1e453e1e85ec0f5ac4052d36598cc691a` (pinned; matches packup) |
| AITER commit | `2c71811b32c8ce2e1266aedaec199df7d90f597d` (matches packup) |
| torch | `2.9.1+rocm7.2.0` (matches packup) |
| Model path | `/perf_apps/data/models/GLM-5.2-MXFP4` (NFS, 408 G, mounted `:ro`) |
| GPU visibility | `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` (8 of 8 GPUs) |
| Container user | non-root, `--user "$(id -u):$(id -g)"` |

## Deviations from the spur packup

This run reproduces the packup's method on new hardware; the following are the only differences,
none of which change the benchmark parameters or measurement semantics:

1. **Container image digest.** The spur packup used `sha256:b9a83742f631...`; that archive is not
   present on this host. Substituted `sha256:b5aa5bd3d828...` (local tag `rocm-llm-bench:latest`),
   probed first-hand to carry the identical pinned SGLang commit, identical AITER commit, and
   identical torch build. Only the image build digest differs, not the software stack.
2. **Non-root container.** The workspace lives on an NFS export with `root_squash`; a root-owned
   container process cannot write result files there (`touch` returns `EPERM`, verified). The
   container instead runs as the host user (`--user "$(id -u):$(id -g)"`), with:
   - a host-owned bind-mount of `/tmp/aiter_configs` (root-owned in the image) at
     `/var/tmp/yihou-glm52-aiter-configs`, so AITER's tuned-GEMM CSVs stay writable and
     byte-identical to the image's own copy;
   - a synthesized `/etc/passwd` / `/etc/group` bind-mount at `/var/tmp/yihou-glm52-nss`, because
     the host uid is LDAP/sssd-only and absent from the image's NSS files, which otherwise crashes
     `getpass.getuser()` at torch-inductor import time;
   - an explicit container `PATH` (`/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:
     /usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin`) that drops `/root/.cargo/bin`. `/root` is
     mode `drwx------`; as a non-root user, a PATH search that walks into it for `nvcc` raises
     `EACCES`, which CPython's `_posixsubprocess` reports as `PermissionError` instead of the
     `FileNotFoundError` that torch-inductor's CUDA-version-info probe (`_cuda_system_info_comment`,
     called from its repro-dump helper, not a real compile step) tolerates on a ROCm-only image.
     Confirmed first-hand by reproducing both outcomes with and without `/root/.cargo/bin` on PATH
     via a direct `subprocess.check_output(["nvcc","--version"])` call inside the container.
3. **`--max-running-requests 128` added to `server_cli`.** Not present in the packup's `server_cli`
   at all -- `max_running_requests` auto-derives from EAGLE speculative decoding defaults to 48 when
   unset. Confirmed first-hand (`sglang/srt/mem_cache/kv_cache_configurator.py:1880-1883`) that the
   per-worker `ReqToTokenPool` is sized as `max_running_requests // attn_dp_size`. With
   `attn_dp_size=8` this gave only `48 // 8 = 6` slots/worker, short of the `local_batch_size=16`
   needed, and the run failed with `alloc_req_slots runs out of memory,
   req_to_token_pool.available_size()=6, num_reqs=16` before reaching the decode loop. Passing
   `--max-running-requests 128` gives `128 // 8 = 16` slots/worker -- exactly the requested
   concurrency, no slack. This is a request-slot-table sizing fix, not a performance tuning knob:
   `--mem-fraction-static`, `--cuda-graph-bs-decode`, and every other benchmark parameter are
   unchanged from the packup's `server_cli`. Note the spur packup's own TP4/EP4 control at C=48 sat
   exactly at this same limit (`attn_dp_size=4` -> `48 // 4 = 12` = its local batch there) without
   ever crossing it -- a coincidence of that operating point, not evidence the limit doesn't exist.

## Failed attempts kept as evidence (not deleted)

All 9 remaining EP8 sweep points (48/64/96/160/192/224/256/288/320) and all 10 EP1 sweep points
(48/64/96/128/160/192/224/256/288/320) launched sequentially in two detached driver runs and
succeeded on the first attempt each, with no new failures of any kind at any C in either sweep. The
failures below are all from Phase 1 (getting the C=128 EP8 point running for the first time):

- `iterations/aborted_rootsquash_c128_yihou/` -- root-owned container, NFS `root_squash` blocked
  result writes.
- `iterations/aborted_aiterconfigs_c128_yihou/` -- `/tmp/aiter_configs` root-owned, non-root write
  failure.
- `iterations/aborted_getpwuid_c128_yihou/` -- `getpass.getuser()` `KeyError` (LDAP-only uid absent
  from image NSS files).
- `iterations/aborted_nvcc_path_c128_yihou/` -- CUDA-graph capture crashed with
  `PermissionError: [Errno 13] Permission denied: 'nvcc'` (see deviation #2 above; fixed by dropping
  `/root/.cargo/bin` from `PATH`).
- `iterations/aborted_reqpool_c128_yihou/` -- passed CUDA-graph capture, then failed at
  `alloc_req_slots` (see deviation #3 above; fixed by adding `--max-running-requests 128`).

## Not investigated / open questions

- The exact mechanism by which a missing PATH-search executable surfaces as `PermissionError`
  rather than `FileNotFoundError` was root-caused via CPython's `_posixsubprocess` errno-priority
  behavior (keeps the first errno that is neither `ENOENT` nor `ENOTDIR`); this was verified by a
  decisive before/after test, not merely inferred.
- No other deviations from the packup's benchmark parameters were made. `--mem-fraction-static`
  was never lowered despite SGLang's generic OOM-troubleshooting message suggesting it on the first
  failure; that failure was not an OOM (KV-cache allocation succeeded both times it was reached).

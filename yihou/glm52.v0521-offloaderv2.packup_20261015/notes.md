# Notes — what / why / how / context

## 1. The `is_shuffled` attribute patch (YIHOU-OFFLOAD-ATTRS)

- **What:** `_CpuParamOffloader.create_device_tensor` now copies `is_shuffled=True` from the host parameter onto the device tensor it creates (`patches/0001-offloader-carry-is_shuffled.diff`).
- **Why:** aiter fused_moe picks its kernel through `getattr(w, "is_shuffled")`, and the device tensor that OffloaderV2 creates on each onload drops that attribute. This comes from the code reading in `analysis/research/offv2_impl.md`.
- **How:** it is a bind mount over `/sgl-workspace/sglang/python/sglang/srt/utils/offloader.py` on the prefill leg only, built by `patches/build_offpatch.yihou.py` from the image's own file.
- **Context:** no unpatched offload round was run. This kit therefore does not demonstrate the failure the patch prevents. The patched rounds pass the 23-test suite at 1/8, 1/3 and 1/2.

## 2. Why K=1 and step=1, ratio raised through G

- `OffloaderV2.wrap_modules` offloads module `i` when `i % G >= G - K` (`offloader.py:257`).
- The forward hook of offloaded module `idx` starts onloading `offloaders[(idx + step) % n]` (`offloader.py:297`).
- With K>1, offloaded layers are adjacent. With step 1, the next copy then overlaps only one layer's compute.
- Spreading them with K=1 and a smaller G gives G−1 non-offloaded layers between copies. That is G4K1 / G3K1 / G2K1 for 1/4, 1/3, 1/2.
- The assessment (§D) recommended `G=2 K=1 step=2` for 50 %. The sweep used step=1, and step=2 was not tried.

## 3. H2D bandwidth (gate), 089, `results/h2d/h2d_bench.089.json`

| case | per-GPU GB/s |
|---|---|
| each GPU alone | 45.2 / 46.0 / 45.8 / 45.4 |
| 4 GPUs concurrent | 37.3 / 36.4 / 46.3 / 34.6 |
| 4 concurrent H2D + D2H | 18.3 / 17.8 / 19.0 / 19.0 |
| 4 concurrent, with 64 GiB extra pinned host memory allocated | 29.9 / 30.3 / 30.0 / 31.2 |

- The assessment gate read: "≥ 40 GB/s → 50 %; 25-40 → cap at 25 %; < 25 → abort". The concurrent cases fall in 25-40, and at the time this led to "cap at 25 %".
- On 10-15 the user asked to raise the ratio anyway and observe ("提高offload率观察性能"), so 1/3 and 1/2 were launched.

## 4. Memory accounting observed

- Each offloaded layer reduces the prefill weight footprint by about 1.0-1.15 GB per rank:
  - 9 layers → −9.07 GB;
  - 19 → −20.44;
  - 26 → −28.34;
  - 39 → −43.06 (vs 119.84).
- `avail mem` after the KV pool stays at 27.7-27.8 GB in every round. So the freed HBM went into the KV pool: post-pool free = pre-load × (1 − mem_fraction), as in `kv_pool_runtime.py`.
- Note on units: the logs print "GB" here; earlier notes used GiB for the same lines.

## 5. Bring-up times are noisy across nodes

Prefill `Engine startup timings`:

| round | load_weight (s) | scheduler_e2e (s) |
|---|---|---|
| ofs0c (089) | 56 | 276 |
| ofs0 (089) | 56 | 687 |
| ofg2c (288) | 392 | 685 |
| ofg3c (266) | 370 | 894 |
| ofg4c (252) | 389 | 2064 |

- The ~370-390 s weight load was seen on the first boot on 252/266/288 (vs 56 s on 089). Its cause was not measured.
- 252's PD warm-up took 20 min 45 s (02:21:24 → 02:42:09), against 60 s / 85 s on 266/288. GPU use was 0 % on GPU0-4 while it waited, and the logs show no error.
- The cause is unknown. The next step if it repeats is a `py-spy dump` of the prefill scheduler while it waits.

## 6. Incidents

- **Node 136 taken** by another user (~10-14 10:35) before S0 ran there. S0 moved to 089 behind tw90 (`after_tw90_s0.yihou.sh`). The 136 env files are kept for record.
- **Node 266 released** (10-15 ~02:50, user: "管理员指定266"). Steps taken:
  - The driver PIDs were killed by number.
  - `run.v3 stop` was run. It reported "cleanup completed with errors", yet all containers were exited and 0 KFD processes remained.
  - Our seven named, exited containers were removed with `docker rm` by name.
  - `scancel 215368`.
  - Ledger entry 02:45-02:55.
- **Hold sleep length:** `hold_node.sh` sleeps `HOLD_SECONDS` (default 28800 = 8 h) regardless of `-t`. Pass `HOLD_SECONDS=86400 --export=ALL` for a 24 h hold. The replacement hold 217052 was resubmitted that way.
- **Single replicates:** every score here is n = 1, and no same-config replicate was run, so the run-to-run spread is not measured. The S0 −1.3 % (A2) and +2.4 % (A1) deltas have opposite signs.

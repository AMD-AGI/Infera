# Environment

Per-node snapshots in `env/env_chi2835.txt` and `env/env_chi2879.txt`, captured
**2026-08-03 10:51 UTC** — 16 minutes after the run ended, with both legs still
live, so the recorded command lines are the ones that served the run.

## Digest

| | |
|---|---|
| cluster | **vultr** (not spur — the spur mlx5 block silently drops to TCP here) |
| access | jump host `root@149.28.124.225` (= slurm login node chi2866), then `ssh chi2835` / `ssh chi2879` |
| **prefill node** | **chi2835**, data-plane `10.2.122.78`, kernel `6.8.0-107-generic` |
| **decode node** | **chi2879**, data-plane `10.2.122.10`, kernel `6.8.0-124-generic` |
| GPUs | 8 × AMD Instinct MI355X `gfx950`, 288 GB/card, per node |
| CPU / RAM | AMD EPYC 9575F 64-core (256 thr) / ~3.0 TB, both nodes |
| GPU driver | **6.16.13**, both nodes |
| image | **`infera/engine-sglang:merged-e`** |
| digest chi2835 | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| digest chi2879 | `sha256:bfcb6462fa306743e0bf43b32ac0263ce9094e13591f6f748263e5348bf97e41` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| sglang / ROCm | 0.5.15.post1 / 7.2.0 |
| repo | `AMD-AGI/Infera`, branch **`yihou.dev.glm52.merged.experiment`** |
| slurm holders | `yeandy-debug` on **both** nodes — **not ours, never `scancel`** |

> The two image IDs differ **by design** — each node built independently from the
> same branch head. The digests are the binding artifact. chi2835 carries the
> same ID as chi2867/chi2878; chi2879 carries the one from the solo kits.

## RDMA fabric — an asymmetry, recorded

| node | ACTIVE rails | note |
|---|---|---|
| chi2835 (prefill) | **8/8** `ionic_0..7` | full |
| chi2879 (decode) | **7/8** — `ionic_5` **PORT_DOWN** | see below |

`ionic_5` on chi2879 is physically down. `glm52_leg.sh:75-81` enumerates only
`PORT_ACTIVE` ionic devices, so the decode leg came up with
`--disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_6,ionic_7`
— **7 rails, automatically, not a misconfiguration.** The prefill leg used all 8.

This is a real asymmetry in the transport and is **not** controlled for. It was
present for the whole run. Its effect on KV transfer time is **unmeasured**.

`MC_GID_INDEX` is **discovered per node**, never hardcoded (`_gid_discover()`,
`glm52_leg.sh:108-120`) — it picks the first GID that is neither empty nor
`fe80::`. Both nodes resolved to **index 1**.

## The deployment delta, exactly

Live prefill command line (`env/env_chi2835.txt`), diffed against Case A's:

```
 --model-path /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --tp-size 8
 --mem-fraction-static 0.80 --context-length 262144
---chunked-prefill-size 65536          --dp-size 8 --enable-dp-attention
-  --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000
+--chunked-prefill-size 16384          <- CHUNK_PASSTHROUGH, see patches/
 --ep-size 8                           <- HELD FIXED (EP_DECOUPLE)
 --hicache-size 16 --enable-cache-report --kv-cache-dtype fp8_e4m3
 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake
```

Resolved server args, read from the boot log rather than assumed:

| | dp_size | ep_size | `chunked_prefill_size` (resolved) | max_prefill | max_running |
|---|---|---|---|---|---|
| **prefill (DPA OFF)** | **1** | **8** | **16384** (not divided) | 16384 | 2048 |
| **decode (DPA ON)** | 8 | 8 | 8192 (65536 ÷ 8) | 16384 | 256 (=2048/8) |

Three consequences, each verified rather than assumed:

1. **MoE unchanged** — `ep_size` stays 8 only because of `EP_DECOUPLE`.
2. **Per-forward prefill work is 2× Case A's**, deliberately. sglang divides by
   `dp_size` *only* under DPA (`server_args.py:4902`): Case A resolved
   65536÷8 = 8192; this run resolves 16384÷1 = **16384**.
3. **The prefill delayer is gone.** `--enable-prefill-delayer` is scoped to the
   DPA branch, so DPA-off drops it. A real config difference, **not controlled
   for**; unlike the solo runs (where a queue never forms at concurrency 1),
   this run *does* build queues, so the delayer's absence is a live variable.
   **Its effect is unmeasured.**

## KV memory structure

```
prefill (TP8)   [TP0 EP0]     max_total_num_tokens=3263680  max_running_requests=2048  <- ONE scheduler
decode  (DPA8)  [DP0 TP0 EP0] max_total_num_tokens=3085248  max_running_requests=256   <- 2048/8, per-rank
```

Under pure TP the prefill leg has **one** scheduler and **one** KV pool
replicated across ranks; under DPA the decode leg has **8** schedulers each
owning a distinct shard. Reading `max_total_num_tokens` as "the KV pool" without
asking *how many ranks hold a distinct copy* produces a confident wrong story —
see `../solo.glm52.dpaoff.packup_20260802/notes.md` for the full correction.

## Legs: what was started when

| leg | node | TAG | started | notes |
|---|---|---|---|---|
| prefill | chi2835 | q1 | **2026-08-02 15:34:07** | ran 18 h before the measurement; untouched during it |
| decode | chi2879 | **q4** | **2026-08-03 09:22:12** | 4th attempt — see `notes.md` |

The decode leg was relaunched three times before the run: `q2` on chi2878 (killed
externally when that node went `resv`), `q3` on chi2879 (loaded **unpatched**
bytecode — the `.pyc` trap), `q4` on chi2879 (patched, verified, served the run).

## The driver runs from the JUMP HOST

| | |
|---|---|
| bench repo | `Optimus-AgenticBench`, branch **`fix/realistic-profile-session-driver`** @ **`1cf01cb`** — **not `main`** |
| staged at | `/mnt/vast/c_huggingface/bench_20260801/agbench/` |
| venv | `/mnt/vast/c_huggingface/bench_20260801/venv/`, Python 3.12.3, `pip install -e .` |
| workload | `/mnt/vast/c_huggingface/bench_20260801/par8.yaml` = this kit's `spec/par8.yaml` (md5 `78e4badf107178f64c6d45a85674f2cb`) |
| target | `http://10.2.122.78:8100` (the **router** — never a leg's own port) |
| patch | `SOLO_M1` — persists per-request E2E and TPOT; inherited from `../solo.glm52.latencyfloor.packup_20260802/patches/` |

`main`'s closed-loop session driver is unfixed and silently under-loads.

> **`SOLO_M1` is why this kit has E2E/TPOT ladders and the Case A kit does not.**
> Case A ran before that patch existed, so its `metrics.jsonl` carries
> `new_ttfts` but **zero** `new_e2es` / `new_tpots`. Any E2E or TPOT comparison
> against Case A is therefore impossible from the raw samples — Case A's TPOT
> p50 of 14.8 ms comes from its own summary line, not from per-request data.

## kvd state across the run

| counter | prefill before | prefill after | delta | decode (before→after) |
|---|---|---|---|---|
| entries | 0 | 47,677 | +47,677 | 0 → 0 |
| gets | 0 | **0** | **+0** | 0 → 0 |
| hits | 0 | 0 | +0 | 0 → 0 |
| misses | 0 | 0 | +0 | 0 → 0 |
| sets | 0 | 57,870 | **+57,870** | 0 → 0 |
| evictions | 0 | 10,193 | +10,193 | 0 → 0 |
| host_bytes | 0 | 84.4 GB | | 0 |
| long_bytes | 0 | 68.7 GB | | 0 |

Both daemons started **cold** (all-zero) — this is a clean baseline, unlike the
solo kits which inherited a warm tier.

**`gets = 0` is the notable number.** kvd wrote 57,870 entries and read back
**none**. Contrast Case A: +452 gets / +452 hits. Decode is all-zero **by
design** — the image deliberately skips kvd wiring on a PD decode leg.

**Why prefill never read back is NOT ESTABLISHED.** The measurement that would
answer it: correlate kvd `gets` against the engine's HiCache prefetch decisions
over the window, which requires per-request HiCache tier accounting this run did
not capture.

## External dependencies (absolute paths, not in any repo)

| what | where |
|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST NFS). `generation_config.json` temp 1.0 / top_p 0.95 is **load-bearing** — see `notes.md` on the driver's hardcoded `temperature: 0.0` |
| host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1` → bind-mounted `/host-libionic/libionic.so`; must match the host `ionic_rdma` kmod |
| scratch / logs | `/mnt/vast/c_huggingface/bench_20260801/` |
| bench repo | `/home/yihou/dev/git.16-19/Optimus-AgenticBench` @ `1cf01cb` |
| kvd L3 tier | `/tmp/kvd-long` **inside the container** — node root disk, **not** `/mnt/vast`. `--long-bytes 64G` (lowered from 512G after it filled a node's root disk during the fixlen sweep) |

## Secrets required (names and sources only — no values here)

| secret | source |
|---|---|
| cluster SSH | key-based access to `root@149.28.124.225`, then node-to-node as root. Arrange your own; nothing in this kit contains a key. |
| docker registry | **not needed** — images already present on both nodes. A cold node would need the team registry login for `lmsysorg/sglang`. |
| etcd | **unauthenticated** on the prefill node's private data-plane IP. |
| router / engine | no API key (`api_key=None`, `admin_api_key=None`). |

**No secret value appears anywhere in this kit** — env snapshots, scripts, the
router log and the driver log were checked before packing.

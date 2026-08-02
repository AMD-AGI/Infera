# Environment

Per-node snapshots in `env/env_chi2879.txt` and `env/env_chi2867.txt`, captured
2026-08-01 11:21 UTC with the packup skill's `collect_env.sh`.

**Hardware, fabric, image and repo state are identical to the DPA-on baseline
kit** (`../solo.glm52.latencyfloor.packup_20260802/environment.md`). This file
records the digest, states precisely **what was and was not restarted**, and
lists the deltas.

## What changed between the baseline and this run

| | baseline (DPA-on) | this run (DPA-off) |
|---|---|---|
| prefill leg | chi2879 TAG **p4**, PID 1176352, up 2026-08-01 11:47:40 | chi2879 TAG **p7**, PID **1508523**, up **2026-08-02 05:42:33** |
| decode leg | chi2867 TAG p6, PID **2420132**, up 2026-08-01 13:29:28 | chi2867 TAG p6, PID **2420132** — **UNCHANGED** |
| driver | `1cf01cb` + `SOLO_M1` | same, untouched |
| router | rust, 10.2.122.10:8100 | same, not restarted |
| kvd daemon | running | same daemon, survived the prefill restart |

**Only the prefill leg was relaunched.** This matters: `GLM52_P1V3`
(`patches/0004`) is applied to the decode container's bytecode **at runtime**
and is lost on restart. It stayed loaded throughout — verified `P1V3: 3` in the
loaded module, and the PID is provably the same one that served the baseline.

That the kvd daemon and its contents survived is visible in the counters: the
`kvd_before.json` for this run is byte-identical to the baseline's
`kvd_after.json` (entries 47,648 / gets 1,260 / sets 109,876), proving nothing
ran in between.

## The deployment delta, exactly

Prefill leg command line, diffed against the baseline's:

```
 --model-path /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --tp-size 8
 --mem-fraction-static 0.80 --context-length 262144
---chunked-prefill-size 65536          --dp-size 8 --enable-dp-attention
-  --enable-prefill-delayer --prefill-delayer-max-delay-ms 5000
+--chunked-prefill-size 8192
 --ep-size 8                            <- HELD FIXED (patches/0006)
 --hicache-size 16 --enable-cache-report --kv-cache-dtype fp8_e4m3
 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake
```

Resolved server args (checked offline *before* the window was spent):

| | dp_size | ep_size | chunked_prefill (resolved) | max_prefill | aiter AR fusion |
|---|---|---|---|---|---|
| DPA-ON | 8 | 8 | 8192 | 16384 | False |
| DPA-OFF | **1** | **8** | 8192 | 16384 | False |

Three consequences, each verified rather than assumed:

1. **MoE unchanged** — `ep_size` stays 8 only because of `patches/0006`.
2. **Per-pass prefill work unchanged** — sglang divides `chunked_prefill_size`
   by `dp_size`, so 65536÷8 and 8192÷1 both resolve to 8192.
3. **aiter all-reduce fusion stays OFF.** DPA-off makes sglang *log*
   `Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM`, but the assignment
   behind it is commented out in this image (`# self.enable_aiter_allreduce_fusion = True`).
   Resolved value is `False`, so the gfx950 EAGLE-verify deadlock path is not
   entered. **The log line is misleading; the flag is what counts.**

### The prefill delayer is gone, and it does not matter

`--enable-prefill-delayer` is scoped to the DPA branch, so DPA-off drops it.
This is a real config difference, but not a confound: the baseline's prefill log
shows the delayer triggering **0 times** in its measured window (concurrency 1
never builds a queue). Both runs therefore had an inactive delayer.

### KV memory structure — the mechanism of this experiment

```
DPA-on   [DP0 TP0 EP0] max_total_num_tokens=2829952  max_running_requests=256   <- 2048/8, per-rank scheduler
DPA-off  [TP0 EP0]     max_total_num_tokens=3263680  max_running_requests=2048  <- ONE scheduler
```

| | per rank | KV size/rank | distinct shards | **aggregate KV** |
|---|---|---|---|---|
| DPA-on | 2,829,952 tok | 145.55 GB | 8 | **22,639,616 tok** |
| DPA-off | 3,263,680 tok | 167.86 GB | 1 (replicated) | **3,263,680 tok** |

Per-rank +15.3 %, aggregate **−85.6 %**. The host (L2) tier is **356,160 tokens
in both runs** — `--hicache-size 16` was not changed, and the
"host pool smaller than device pool" warning appears in *both* logs.

## Digest

| | |
|---|---|
| cluster | **vultr** (not spur — the spur mlx5 block silently drops to TCP here) |
| access | jump host `root@149.28.124.225` (= slurm login node chi2866), then `ssh chi2879` |
| prefill node | **chi2879**, data-plane `10.2.122.10` |
| decode node | **chi2867**, data-plane `10.2.122.44` |
| GPUs | 8 × **AMD Instinct MI355X** `gfx950`, 288 GB/card, per node |
| CPU / RAM | AMD EPYC 9575F 64-core (256 thr) / ~3.0 TB |
| GPU driver / kernel | 6.16.13 / `6.8.0-124-generic` |
| fabric | 8 × **ionic** RoCE v2 per node, all `PORT_ACTIVE` |
| `MC_GID_INDEX` | **node-dependent: chi2879 → 1, chi2867 → 2.** Discovered, never hardcoded |
| image | **`infera/engine-sglang:merged-e`** |
| digest chi2879 | `sha256:bfcb6462fa306743e0bf43b32ac0263ce9094e13591f6f748263e5348bf97e41` |
| digest chi2867 | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`<br>`sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| sglang / ROCm | 0.5.15.post1 / 7.2.0 |
| repo | `AMD-AGI/Infera`, branch **`yihou.dev.glm52.merged.experiment`** |
| slurm holder | job name `yeandy-debug` on **both** nodes — **not ours, never `scancel`** |

> The two node image ids differ by design — each node built independently from
> the same branch head. The image digests are the binding artifact.

## Deployment under test

    two-node PD over mooncake RDMA   ionic RoCE, MC_GID_INDEX discovered per node
    DP-attention                     PREFILL OFF (pure TP8)  /  DECODE ON (8/8)
    expert parallel (MoE)            ep_size 8 on BOTH legs   <- held fixed
    kv-aware routing                 ON, RUST backend, w_prefill 20.0 / w_decode 2.0
                                     NOTE: prefill now exposes 1 target, not 8
    kvd (infera HiCacheStorage)      prefill ON (--hicache-size 16), decode skipped by design
    MTP                              decode leg only, EAGLE steps=3 topk=1 draft=4
    --context-length                 262144
    --cuda-graph-max-bs              128
    --enable-cache-report            ON          (else cache-hit reads 0)
    --kv-cache-dtype                 fp8_e4m3
    mem-fraction-static              prefill 0.80 / decode 0.85

## The driver runs from the JUMP HOST

| | |
|---|---|
| bench repo | `Optimus-AgenticBench`, branch **`fix/realistic-profile-session-driver`** @ **`1cf01cb`** — **not `main`** |
| staged at | `/mnt/vast/c_huggingface/bench_20260801/agbench/` |
| venv | `/mnt/vast/c_huggingface/bench_20260801/venv/`, Python 3.12.3, `pip install -e .` |
| workload | `/mnt/vast/c_huggingface/bench_20260801/solo.yaml` = this kit's `spec/solo.yaml` |
| target | `http://10.2.122.10:8100` (the router — **never a leg's own port**) |
| patch | `SOLO_M1` (`patches/0005`) — inherited, unchanged, `.orig` preserved |

`main`'s closed-loop session driver is unfixed and silently under-loads.

## kvd state across the run

| counter | before (05:46) | after (06:23) | delta |
|---|---|---|---|
| entries | 47,648 | 47,781 | +133 |
| gets | 1,260 | 1,692 | **+432** |
| hits | 1,260 | 1,692 | **+432** (100 %) |
| misses | 0 | 0 | +0 |
| sets | 109,876 | 120,952 | **+11,076** |
| evictions | 54,782 | 65,031 | **+10,249** |

Contrast the baseline: **+0 gets, +1,122 sets, +1,206 evictions**. See
`analysis/` for why — this is the capacity side of the DPA trade, not a
configuration accident.

## External dependencies (absolute paths, not in any repo)

| what | where |
|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST NFS). `generation_config.json` temp 1.0 / top_p 0.95 is **load-bearing** — `temperature: 0` with MTP is indistinguishable from KV corruption |
| host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1` → bind-mounted `/host-libionic/libionic.so`; must match the host `ionic_rdma` kmod |
| scratch / logs | `/mnt/vast/c_huggingface/bench_20260801/` |
| bench repo | `/home/yihou/dev/git.16-19/Optimus-AgenticBench` @ `1cf01cb` |
| kvd L3 tier | `/tmp/kvd-long` **inside the container** — node root disk, not `/mnt/vast` |

## Secrets required (names and sources only — no values here)

| secret | source |
|---|---|
| cluster SSH | key-based access to `root@149.28.124.225`, then node-to-node as root. Arrange your own; nothing in this kit contains a key. |
| docker registry | **not needed** — images already present on both nodes. A cold node would need the team registry login for `lmsysorg/sglang`. |
| etcd | **unauthenticated** on the prefill node's private data-plane IP. |
| router / engine | no API key (`api_key=None`, `admin_api_key=None`). |

**No secret value appears anywhere in this kit** — env snapshots, scripts and
the driver log were checked before packing.

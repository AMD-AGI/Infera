# Environment

Per-node snapshots (read-only, no secret values) in `env/env_chi2879.txt` and
`env/env_chi2867.txt`, captured 2026-08-01 11:21 UTC with the packup skill's
`collect_env.sh`.

**The hardware, fabric, image, repo state and running processes are identical to
the Case A kit** — the same deployment served both, which is what makes the
comparison in `analysis/solo_latency.md` valid. This file records the digest,
**proves the no-restart claim**, and lists the two deltas.

Full detail: `../caseA.glm52.fullfeature.packup_20260801/environment.md`.

## ⚠️ Why the env snapshot predates the run, and why that is fine

The snapshots are stamped **11:21 UTC**; this run started **16:05 UTC**. That
gap would normally be a reproducibility hole. It is closed by direct evidence
that the serving processes never restarted in between:

```
chi2879 (prefill)  container bench_run  StartedAt 2026-08-01T09:18:57Z  running=true
                   sglang.launch_server started  Sat Aug  1 11:47:40 2026
chi2867 (decode)   container bench_run  StartedAt 2026-08-01T09:18:56Z  running=true
                   sglang.launch_server started  Sat Aug  1 13:29:28 2026
```

Both `launch_server` processes predate the **Case A** window (13:34–14:41) and
were still the same PIDs when the solo window opened at 16:05. Same containers,
same processes, same loaded weights, same patched bytecode. Re-check with:

```bash
ssh root@149.28.124.225 "ssh chi2879 'docker inspect -f \"{{.State.StartedAt}}\" bench_run; \
  ps -eo lstart,cmd | grep -m1 \"[s]glang.launch_server\"'"
```

## Digest

| | |
|---|---|
| cluster | **vultr** (not spur — configured oppositely; the spur mlx5 block silently drops to TCP here) |
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
| repo | `AMD-AGI/Infera`, branch **`yihou.dev.glm52.merged.experiment`** @ **`e56e975`** |
| slurm holder | job name `yeandy-debug` on **both** nodes — **not ours**, never `scancel` |

> The two node image ids differ by design — each node built independently from
> the same branch head.
>
> The repo SHA advanced from Case A's `b92a1e8` to `e56e975` between the two
> runs. **This did not affect the serving processes** — they had been running
> since 11:47 / 13:29 off the already-built images; the commits were
> documentation. The image digests are the binding artifact and are unchanged.

## Delta 1 — the DRIVER is patched (new to this run)

```
driver under test = Optimus-AgenticBench @ 1cf01cb  +  SOLO_M1
```

`patches/0005`, applied by `scripts/apply_solo_metrics.py` to
`$W/agbench/agent/agent_throughput.py` on the **jump host** (the driver runs
there, not in a container). Idempotent, anchors on exact source text, exits
loudly if the driver has drifted.

    original preserved at: $W/agbench/agent/agent_throughput.py.orig
    md5 BEFORE patch:      2aa74d1d983984c1b53a3f27d51ebbaa

**Without it there is no per-request E2E and no per-request TPOT on disk**, and
the two headline ladders of this experiment cannot be produced. `solo_run.sh`
hard-gates on the `SOLO_M1` marker and refuses to start without it.

Stale bytecode is a live trap here (it has invalidated an experiment in this
tree before): the applier is followed by deleting `agbench/**/__pycache__` and
re-importing to confirm `SOLO_M1` appears in the **loaded** module.

## Delta 2 — the ENGINE is patched (inherited from Case A)

```
image under test = infera/engine-sglang:merged-e  +  GLM52_P1V3
```

`patches/0004`, applied at runtime inside the **decode** container to
`/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`.
Already live on leg `p6` when this run started — verified `P1V3: 3` in the
loaded module immediately before the window opened.

Stock `merged-e` **cannot run this request shape at all**; it dies on
`Expected lengths.size(0) == B` in the DSA indexer under MTP draft-extend.
Analysis: `notes/notes.dsa.mtp.crash.md`.

The prefill leg is **unpatched** — the bug is on the MTP draft path, decode-only.

    file md5 BEFORE patch: 632f17acd38737459b43f830ee60ee89
    original preserved at: /tmp/dsa_indexer.py.orig (inside the decode container)

## Leg lineage

| leg | node | TAG | MTP | gmu | engine patch | started |
|---|---|---|---|---|---|---|
| prefill | chi2879 | p4 | off | 0.80 | none (not needed) | 11:47:40 |
| decode | chi2867 | p6 | **on** | 0.85 | **GLM52_P1V3** | 13:29:28 |

`SGLANG_DEBUG_DSA_ROWS=1` was left **on** for leg p6 (inherited from the Case A
crash hunt). It costs one log line per indexer call. Off by default in
`start_leg.sh` (`DSA_ROWS=0`).

## The driver runs from the JUMP HOST

| | |
|---|---|
| bench repo | `Optimus-AgenticBench`, branch **`fix/realistic-profile-session-driver`** @ **`1cf01cb`** — **not `main`** |
| staged at | `/mnt/vast/c_huggingface/bench_20260801/agbench/` (copied without `.git`) |
| venv | `/mnt/vast/c_huggingface/bench_20260801/venv/`, Python 3.12.3, `pip install -e .` |
| workload | `/mnt/vast/c_huggingface/bench_20260801/solo.yaml` = this kit's `spec/solo.yaml`, verbatim |
| target | `http://10.2.122.10:8100` (the router — **never a leg's own port**) |

`main`'s closed-loop session driver is unfixed and silently under-loads; the
branch is not optional.

## kvd state across the run

| counter | before (16:05) | after (16:42) | delta |
|---|---|---|---|
| entries | 47,732 | 47,648 | −84 |
| gets | 1,260 | 1,260 | **+0** |
| hits | 1,260 | 1,260 | **+0** |
| misses | 0 | 0 | +0 |
| sets | 108,754 | 109,876 | +1,122 |
| evictions | 53,576 | 54,782 | +1,206 |
| host_bytes | 84.4 GB | 84.4 GB | −0.06 GB |

**Zero gets.** kvd started warm from the Case A runs and a single stream nesting
in one prefix never needed to read the spill tier. Correct behaviour, and it
confirms this shape does not exercise tiering.

Root disk at run time: **80 % (162 GB free)** on chi2879.

## Deployment under test

    two-node PD over mooncake RDMA   ionic RoCE, MC_GID_INDEX discovered per node
    DP-attention 8/8 both legs       --dp-size 8 --enable-dp-attention --ep-size 8
    kv-aware routing                 ON, RUST backend, w_prefill 20.0 / w_decode 2.0
    kvd (infera HiCacheStorage)      prefill ON (--hicache-size 16), decode skipped by design
    MTP                              decode leg only, EAGLE steps=3 topk=1 draft=4
    --context-length                 262144
    --chunked-prefill-size           65536       (= 8192/rank at dp8)
    --cuda-graph-max-bs              128
    --max-running-requests           2048
    --enable-cache-report            ON          (else cache-hit reads 0)
    --kv-cache-dtype                 fp8_e4m3
    mem-fraction-static              prefill 0.80 / decode 0.85
    kvd --max-bytes / --long-bytes   64G / 64G

Frozen since Phase 1 and **not retuned** for this run — deliberately, since
retuning would break comparability with Case A.

## External dependencies (absolute paths, not in any repo)

| what | where |
|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (shared VAST NFS). `generation_config.json` temp 1.0 / top_p 0.95 is **load-bearing** — `temperature: 0` with MTP is indistinguishable from KV corruption |
| host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1` → bind-mounted `/host-libionic/libionic.so`; must match the host `ionic_rdma` kmod |
| scratch / logs | `/mnt/vast/c_huggingface/bench_20260801/` |
| bench repo | `/home/yihou/dev/git.16-19/Optimus-AgenticBench` @ `1cf01cb` |
| kvd L3 tier | `/tmp/kvd-long` **inside the container** — on the node root disk, not `/mnt/vast` |

## Secrets required (names and sources only — no values here)

| secret | source |
|---|---|
| cluster SSH | key-based access to `root@149.28.124.225`, then node-to-node as root. Arrange your own; nothing in this kit contains a key. |
| docker registry | **not needed** — both images already present on both nodes. A cold node would need the team registry login for `lmsysorg/sglang`. |
| etcd | **unauthenticated** on the prefill node's private data-plane IP. |
| router / engine | no API key (`api_key=None`, `admin_api_key=None`). |

**No secret value appears anywhere in this kit** — env snapshots, scripts and the
driver log were checked before packing.

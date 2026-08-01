# Environment

Per-node snapshots (read-only, no secret values) in `env/env_chi2879.txt` and
`env/env_chi2867.txt`, captured 2026-08-01 11:21 UTC with the packup skill's
`collect_env.sh`.

**The hardware, fabric, image and repo state are identical to Phase 1** — the same
deployment served both phases, which was the point (one server across the whole
load range). This file records the digest plus **the four things that differ**.

Full detail: `../fixlen.glm52.fullfeature.packup_20260801/environment.md`.

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
| repo | `AMD-AGI/Infera`, branch **`yihou.dev.glm52.merged.experiment`** @ **`b92a1e8`** |
| slurm holder | job name `yeandy-debug` on **both** nodes — **not ours**, never `scancel` |

> The two node image ids differ by design — each node built independently from the
> same branch head.

## ⚠️ Delta 1 — the image is PATCHED for this phase

**This is the most important line in this file.** Phase 1 ran stock `merged-e`.
Phase 2 could not: the decode leg crashes under Case A within minutes.

    image under test = infera/engine-sglang:merged-e  +  GLM52_P1V3

Applied at runtime inside the decode container, to
`/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, by
`scripts/apply_p1v3.py` (idempotent; verifies its own anchors and fails loudly if
the image differs). See `patches/0004-*.txt` and `notes/notes.dsa.mtp.crash.md`.

    file md5 BEFORE patch: 632f17acd38737459b43f830ee60ee89
    original preserved at: /tmp/dsa_indexer.py.orig (inside the container)

The prefill leg is **unpatched** — the bug is in the MTP draft path, which only the
decode leg runs.

**Stale bytecode is a real trap here** (it has invalidated an experiment in this
tree before). `apply_p1v3.py` is followed by deleting
`.../dsa/__pycache__/dsa_indexer*` and re-importing to confirm `GLM52_P1V3`
appears in the *loaded* module, not just on disk.

## Delta 2 — decode leg TAG lineage

Three decode legs were launched. Which one produced a result matters:

| TAG | image state | diagnostic | outcome |
|---|---|---|---|
| p3 | stock | off | crashed 125 s into full attempt 1 |
| p5 | stock | `SGLANG_DEBUG_DSA_ROWS=1` | crashed 766 s into attempt 2 — **root cause captured here** |
| **p6** | **+ P1V3** | `SGLANG_DEBUG_DSA_ROWS=1` | **PASS — the 4,006 s run this kit reports** |

Prefill was `p4` throughout (gmu 0.80, unpatched, never restarted).

> `SGLANG_DEBUG_DSA_ROWS=1` was left **on** for the passing run. It costs one log
> line per indexer call — 144 MB over 67 minutes. It is off by default in
> `start_leg.sh` (`DSA_ROWS=0`); enable only when chasing this bug class.

## Delta 3 — the driver runs from the JUMP HOST

Phase 1 ran `sglang.bench_serving` **inside** the prefill container. Case A's
driver runs **on the jump host (chi2866)** against the router over the data plane.

| | |
|---|---|
| bench repo | `Optimus-AgenticBench`, branch **`fix/realistic-profile-session-driver`** @ **`1cf01cb`** — **not `main`** |
| staged at | `/mnt/vast/c_huggingface/bench_20260801/agbench/` (copied without `.git`) |
| venv | `/mnt/vast/c_huggingface/bench_20260801/venv/`, Python 3.12.3, `pip install -e .` |
| workload | `/mnt/vast/c_huggingface/bench_20260801/caseA.yaml` = repo's `glm52_crxx_caseA.fix.yaml` with **only** the tokenizer path substituted |
| target | `http://10.2.122.10:8100` (the router — **never a leg's own port**) |

`main`'s closed-loop session driver is unfixed and silently under-loads; the branch
is not optional.

## Delta 4 — kvd started warm, and hit its cap

Phase 1 ended with kvd counters cleared (the L3 disk fix wiped `/tmp/kvd-long`).
Case A's probe then warmed it, so the **full run started warm**:

| counter | at full-run start | at full-run end |
|---|---|---|
| entries | 29,750 | 47,732 |
| host_bytes | 52.6 GB | 84.4 GB |
| long_bytes | 54.2 GB | 68.7 GB |
| evictions | 0 | **53,576** |

Evictions began once the L3 tier reached its **64 G cap** (`KVD_LONG_GB`, patch
0003 from Phase 1). That is correct, deliberate behaviour: the cap is what keeps
the tier off the node's 838 GB root disk. Run-level cache efficiency stayed at
100.3 % with 0.0 % of the *expected* prefix evicted, so eviction pressure fell on
cold entries only.

Root disk finished at **83 % (148 GB free)** — no repeat of the Phase 1 disk fill.

## Deployment under test

    two-node PD over mooncake RDMA   ionic RoCE, MC_GID_INDEX discovered per node
    DP-attention 8/8 both legs       --dp-size 8 --enable-dp-attention --ep-size 8
    kv-aware routing                 ON, RUST backend, w_prefill 20.0 / w_decode 2.0
    kvd (infera HiCacheStorage)      prefill ON (--hicache-size 16), decode skipped by design
    MTP                              decode leg only, EAGLE steps=3 topk=1 draft=4
    --context-length                 262144      (covers Case A's 260000 clamp)
    --chunked-prefill-size           65536       (= 8192/rank at dp8)
    --cuda-graph-max-bs              128
    --max-running-requests           2048
    --enable-cache-report            ON          (else cache-hit reads 0)
    --kv-cache-dtype                 fp8_e4m3
    mem-fraction-static              prefill 0.80 / decode 0.85
    kvd --max-bytes / --long-bytes   64G / 64G

Frozen from Phase 1 and **not retuned for Case A** — one deployment measured across
both workloads, per the mission.

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

**No secret value appears anywhere in this kit** — env snapshots and logs were
checked before packing.

# Reproduction kit — Exp 3 (both #32209 ports together)

Goal: reproduce **4/4 probe + 32/32 at conc=32 + 64/64 at conc=64** on a cross-node PD
deployment of GLM-5.2-MXFP4 with DP-attention 8 and MTP, with **both** halves of upstream
PR #32209 ported on top of our patch 1 + 2a + 3.

Estimated time: **~25 min** (~2 min weight load per leg, ~6–8 min JIT + graph capture,
~2 min of tests).

> Cold start is slow and looks like a hang. Eight live `sglang::scheduler_DP*` processes
> mean it is still working.

## 0. Prerequisites

**Machines.** Two nodes, 8 × MI355X, spur cluster `crsuse2-m2m`:

```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00   # twice, one per leg
```

Expect `JobHoldMaxRequeue` bounces; `scontrol release <jobid>` and retry. This run used:

| role | job | node | IP | router |
|---|---|---|---|---|
| prefill | 14320 | crsuse2-m2m-074 | 10.245.154.156 | 8130 |
| decode | 14321 | crsuse2-m2m-072 | 10.245.144.119 | — |

Substitute your own ids/IPs below **and** in `scripts/boot.sh` + `scripts/router.sh`
(hard-coded node tables — see `scripts/NODES.md`).

**Secrets** (values not here; see `environment.md`): docker registry login if you rebuild
the image, spur cluster account. `export DOCKER_CONFIG=/tmp/dockercfg` before **every**
docker call.

**External paths:** model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

> The image tar named in earlier kits (`/home/yihou/infera.yihou.sglang.1.0.tar`) was
> **gone** on 2026-07-30 — `/home` had filled to 100 %. A freshly held node without the
> image fails with `pull access denied`; move it from a node that has it via
> `docker save` → shared storage → `docker load`.

**Repo state:** `yihou.dev.glm5.2.mxfp4.experiment` @ `0d3e374`.

## 1. Stage the workspace

```bash
export W=/shared_nfs/yihou_exp3way
mkdir -p $W/common $W/kit_patches $W/e3
cp scripts/{pd_leg_exp.sh,probe.py,stress.py}                     $W/
cp scripts/{apply_arm.sh,boot.sh,router.sh,start_ctr.sh,NODES.md} $W/common/
cp patches/*.diff                                                 $W/kit_patches/
cp patches/{patch2b_32209_style.py,patch4_32209_style.py,strip_patch2b_v1.py} $W/e3/
```

> **Read this before applying.** `patches/patch4_32209_style.py` in this kit is the
> **corrected** version, which is *not* what this arm ran — it was fixed after this arm
> completed. Applying it reproduces arm **e3b**'s behaviour (draft graph used ~97 %), not
> this arm's (draft graph never used). See `notes.md`. `patch2b_32209_style.py` is
> unchanged since this arm ran it.

Keep the JIT caches off `/home` — `boot.sh` already points them at `$W`.

## 2. Health-gate and start containers

```bash
export DOCKER_CONFIG=/tmp/dockercfg
bash $W/common/start_ctr.sh 14320
bash $W/common/start_ctr.sh 14321
```

Both must print `GPUGATE True 8`. A node printing `False` must be abandoned.

## 3. Apply the arm, and prove it in bytecode

```bash
for J in 14320 14321; do
  spur exec $J bash -c 'docker exec dbg2 bash /shared_nfs/yihou_exp3way/common/apply_arm.sh e3'
done
```

Both must end with `ARM e3 OK`. The arm is:

| # | what | source |
|---|---|---|
| 1 | HIP/aiter padded rows | `dsa_indexer_hip_dp_padded_rows.diff` (ours) |
| 2a | `max_seqlen_k` when `needs_cpu_seq_lens=False` | `dsa_backend_dp_sync_and_page_table_rows.diff` (ours) |
| **2b** | **trim/restore around the DSA decode call** | `patch2b_32209_style.py` (**#32209 shape**) |
| 3 | nextn `eh_proj` bf16 | `deepseek_nextn_glm52_mtp_bf16.diff` (ours) |
| **4** | **draft-graph vote via MLP-sync all-gather** | `patch4_32209_style.py` (**#32209 shape**) |

The kit's `dsa_backend` diff carries **both** our 2a and our 2b, so `apply_arm.sh` applies
it and then runs `strip_patch2b_v1.py` to remove our 2b, leaving 2a. The 2b-v2 script
refuses to run while our v1 helper is still present — that refusal is the guard proving the
strip worked.

Verification greps the **bytecode**, not the source: a stale `.pyc` silently reverts a
patch. Markers are *identifiers* — the compiler discards comments, so a comment marker is a
guaranteed false negative.

## 4. Boot both legs

```bash
bash $W/common/boot.sh e3 prefill
bash $W/common/boot.sh e3 decode
for r in prefill decode; do
  echo -n "$r ready="; strings $W/e3/$r.log | grep -c 'ready to roll'
done
```

Logs contain binary bytes — **plain `grep` silently returns 0**. Always `strings` or
`grep -a`.

## 5. Router

```bash
bash $W/common/router.sh e3
sleep 25
```

## 6. Probe

```bash
export DOCKER_CONFIG=/tmp/dockercfg
spur exec 14320 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/probe.py \
  http://10.245.154.156:8130 4 24 180'
```

## 7. Stress

```bash
spur exec 14320 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/stress.py \
  http://10.245.154.156:8130 32 512 /shared_nfs/yihou_exp3way/e3/stress_c32.jsonl 900'
```

As run here, also conc=64 into `stress_c64.jsonl`.

## Expected output

```
conc=32 maxtok=512 elapsed=10.4s
ok      : 32/32
full tok: 30/32
dp ranks: [0, 1, 2, 3, 4, 5, 6, 7]
acc_len : min=2.27 mean=2.74 max=3.94
```

Check, in order: `ok : 32/32` (the criterion); `acc_len` > 1 (else MTP was silently
bypassed); `dp ranks: [0..7]`; zero `Traceback` in either server log. `full tok` below the
request count is expected — EOS before the 512-token cap.

## Recommended addition: measure draft-graph usage

This arm shipped **without** graph-usage instrumentation, which is why the patch-4 eager
fallback went unnoticed here. If you want the complete picture, add
`scripts/instr_graph_usage.py` from the sibling kit `..._exp3b_patch4_32209` and read the
`GLM52_GUSE` lines from `decode.log`. Note that doing so — together with the corrected
patch 4 — means you are effectively running arm e3b.

## If it doesn't reproduce

See `notes.md`. In particular: a 503 returned in ~0.4 s is a stale router circuit breaker,
not a dead backend; a 503 taking ~12 s is a real backend failure.

# Reproduction kit — Exp 3b (#32209-style patch 4, graph usage measured)

Goal: reproduce **4/4 probe + 32/32 at conc=32 + 97 % draft-graph usage** on a cross-node
PD deployment of GLM-5.2-MXFP4 with DP-attention 8 and MTP, where the draft graph/eager
decision is unified through the MLP-sync all-gather instead of an extra collective.

The graph-usage number is **the point**. Steps 1–7 reproduce the functional pass; **step 8
is what distinguishes this from the Variant-B workaround.** Do not skip it.

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

| role | job | node | IP |
|---|---|---|---|
| prefill | 14317 | crsuse2-m2m-234 | 10.245.152.243 |
| decode | 14318 | crsuse2-m2m-259 | 10.245.155.111 |

Substitute your own ids/IPs below **and** in `scripts/boot.sh` + `scripts/router.sh`
(hard-coded node tables — see `scripts/NODES.md`).

**Secrets** (values not here; see `environment.md`): docker registry login if you rebuild
the image, spur cluster account. `export DOCKER_CONFIG=/tmp/dockercfg` before **every**
docker call.

**External paths:** model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`.

> The image tar referenced by earlier kits (`/home/yihou/infera.yihou.sglang.1.0.tar`) was
> **not present** on 2026-07-30 — `/home` had filled to 100 %. If your node lacks the
> image, move it from a node that has it: `docker save` → shared storage → `docker load`.
> A freshly held node will otherwise fail with `pull access denied`.

**Repo state:** `yihou.dev.glm5.2.mxfp4.experiment` @ `0d3e374`.

## 1. Stage the workspace

```bash
export W=/shared_nfs/yihou_exp3way
mkdir -p $W/common $W/kit_patches $W/e3 $W/e3b
cp scripts/{pd_leg_exp.sh,probe.py,stress.py}                     $W/
cp scripts/{apply_arm.sh,boot.sh,router.sh,start_ctr.sh,NODES.md} $W/common/
cp scripts/instr_graph_usage.py                                   $W/common/
cp patches/*.diff                                                 $W/kit_patches/
cp patches/patch4_32209_style.py                                  $W/e3/
```

Keep the JIT caches off `/home` — `boot.sh` already points them at `$W`.

## 2. Health-gate and start containers

```bash
export DOCKER_CONFIG=/tmp/dockercfg
bash $W/common/start_ctr.sh 14317
bash $W/common/start_ctr.sh 14318
```

Both must print `GPUGATE True 8`. A node printing `False` must be abandoned.

## 3. Apply the arm, and prove it in bytecode

```bash
for J in 14317 14318; do
  spur exec $J bash -c 'docker exec dbg2 bash /shared_nfs/yihou_exp3way/common/apply_arm.sh e3b'
done
```

Both must end with `ARM e3b OK`. The arm is:

| # | what | source |
|---|---|---|
| 1 | HIP/aiter padded rows | `dsa_indexer_hip_dp_padded_rows.diff` |
| 2a + 2b | dp sync + page-table rows | `dsa_backend_dp_sync_and_page_table_rows.diff` (**ours**) |
| 3 | nextn `eh_proj` bf16 | `deepseek_nextn_glm52_mtp_bf16.diff` |
| **4** | **draft-graph vote via MLP-sync all-gather** | `patch4_32209_style.py` (**#32209 shape**) |
| — | graph-usage counters | `instr_graph_usage.py` |

and the anti-markers that must count **0** — exactly one draft-graph mechanism may be live:

```
WANT=0  eagle_worker_v2.py :: _needs_eager_local              (our patch 4)
WANT=0  dsa_backend.py     :: _p2bv2_trim_decode_dp_padding   (#32209 patch 2b)
```

Bytecode, not source: a stale `.pyc` silently reverts a patch. Markers are *identifiers* —
the compiler discards comments, so a comment marker is a guaranteed false negative.

## 4. Boot both legs

```bash
bash $W/common/boot.sh e3b prefill
bash $W/common/boot.sh e3b decode
for r in prefill decode; do
  echo -n "$r ready="; strings $W/e3b/$r.log | grep -c 'ready to roll'
done
```

Logs contain binary bytes — **plain `grep` silently returns 0**. Always `strings` or
`grep -a`.

## 5. Router

```bash
bash $W/common/router.sh e3b
sleep 25
```

## 6. Probe

```bash
export DOCKER_CONFIG=/tmp/dockercfg
spur exec 14317 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/probe.py \
  http://10.245.152.243:8150 4 24 180'
```

## 7. Stress

```bash
spur exec 14317 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/stress.py \
  http://10.245.152.243:8150 32 512 /shared_nfs/yihou_exp3way/e3b/stress_c32.jsonl 900'
```

As run here: repeat into `stress_c32_r2.jsonl`, plus conc=64 into `stress_c64.jsonl`.

## 8. THE decisive check — is the draft graph actually replayed?

```bash
strings $W/e3b/decode.log | grep -a "GLM52_GUSE periodic" | tail -8
strings $W/e3b/decode.log | grep -a "GLM52_GUSE_WHY"      | tail -8
```

Expected:

```
rank=0 calls=800 graph=777 (97.1%) refused_bs=0 refused_dp=0 refused_draftvote=23
...                                          (identical on all 8 ranks)

rank=0 total=600 future_seed_missing=9  (1.5%) future_seed_ok=591 (98.5%)
rank=3 total=600 future_seed_missing=11 (1.8%) future_seed_ok=589 (98.2%)
...                                          (NOT identical -- 8..11)
```

**How to read it:**

| observation | meaning |
|---|---|
| `graph=` **> 0** | the draft graph is genuinely replayed — the run is a fix, not Variant B |
| `graph=` **identical across ranks** | the vote reconciles the group; this is the fix working |
| `future_seed_missing` **differs across ranks** | the underlying divergence is still live — the bug is latent, not absent, so the vote is doing real work |
| `graph=0 (0.0%)` | **the port has degenerated into Variant B.** Functional tests will still pass. See `notes.md`. |

The counters are emitted every 200 calls and once at exit, so a crash cannot swallow them.

## Expected output (functional)

```
conc=32 maxtok=512 elapsed=9.5s
ok      : 32/32
full tok: 29/32
dp ranks: [0, 1, 2, 3, 4, 5, 6, 7]
acc_len : min=2.23 mean=2.86 max=3.91
```

`full tok` below the request count is expected (EOS before the 512-token cap). `acc_len`
must be > 1 or MTP was silently bypassed.

## If it doesn't reproduce

See `notes.md` — especially the 0 %-graph-usage failure, which passes every functional
test while disabling the thing under test.

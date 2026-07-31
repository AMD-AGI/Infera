# Reproduction kit — Exp 2 (IndexShare off, no patch 2/4)

Goal: reproduce **4/4 probe + 32/32 at conc=32** on a cross-node PD deployment of
GLM-5.2-MXFP4 with DP-attention 8 and MTP, where the PD deadlock is avoided by turning
**IndexShare off** instead of by applying patches 2 and 4.

Estimated time: **~25 min** — ~2 min weight load per leg, then ~6–8 min of tilelang JIT
plus DP CUDA-graph capture per leg (legs boot in parallel), ~1 min of tests.

> Cold start is slow and looks like a hang. Eight live `sglang::scheduler_DP*` processes
> mean it is still working. Be patient; do not kill it.

## 0. Prerequisites

**Machines.** Two nodes, 8 × MI355X each, on the spur cluster `crsuse2-m2m`:

```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00   # run twice, one per leg
```

This run used:

| role | job | node | IP |
|---|---|---|---|
| prefill | 14317 | crsuse2-m2m-234 | 10.245.152.243 |
| decode | 14318 | crsuse2-m2m-259 | 10.245.155.111 |

Substitute your own job ids and IPs below **and** in `scripts/boot.sh` +
`scripts/router.sh` (both carry a hard-coded node table — see `scripts/NODES.md`).

**Secrets** (values not included; see `environment.md`): docker registry login if you
rebuild the image, and a spur cluster account. `export DOCKER_CONFIG=/tmp/dockercfg`
before **every** docker call.

**External paths** (not in this repo):
- model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
- image tar: `/home/yihou/infera.yihou.sglang.1.0.tar`

**Repo state:** `yihou.dev.glm5.2.mxfp4.experiment` @ `0d3e374`.
**Image:** `infera.yihou.sglang.1.0` (digest in `environment.md`).

## 1. Stage the workspace on shared storage

```bash
export W=/shared_nfs/yihou_exp3way
mkdir -p $W/common $W/kit_patches $W/e2
cp scripts/{pd_leg_exp.sh,probe.py,stress.py}          $W/
cp scripts/{apply_arm.sh,boot.sh,router.sh,start_ctr.sh,NODES.md} $W/common/
cp patches/*.diff                                       $W/kit_patches/
```

Do **not** put the JIT caches on `/home` — see `notes.md`.

## 2. Health-gate the nodes and start the container

```bash
export DOCKER_CONFIG=/tmp/dockercfg
bash $W/common/start_ctr.sh 14317      # prefill node
bash $W/common/start_ctr.sh 14318      # decode node
```

Each must print `GPUGATE True 8`. If a node prints `False`, abandon it and hold another.

If the image is not on the node yet: `docker load -i /home/yihou/infera.yihou.sglang.1.0.tar`.

## 3. Apply the Exp-2 patch set — and prove patches 2 and 4 are ABSENT

```bash
for J in 14317 14318; do
  spur exec $J bash -c 'docker exec dbg2 bash /shared_nfs/yihou_exp3way/common/apply_arm.sh e2'
done
```

Both must end with `ARM e2 OK`. For this arm the script:

| # | what | how |
|---|---|---|
| 1 | HIP/aiter padded-row trim+restore | `dsa_indexer_hip_dp_padded_rows.diff` |
| 3 | nextn `eh_proj` bf16 | `deepseek_nextn_glm52_mtp_bf16.diff` |
| 2a / 2b / 4 | **not applied** | asserted absent |

The absence checks are the point of this arm, so they are enforced, not assumed. The
script resets the tree to pristine first (so a previous arm's edits cannot be inherited),
then greps the **bytecode** for anti-markers that must count **0**:

```
WANT=0  dsa_backend.py     :: _glm52_match_page_table_rows  -> pyc=0
WANT=0  eagle_worker_v2.py :: _needs_eager_local            -> pyc=0
```

Patch 3 is required by **every** arm: without it the server dies at weight load with a
3072-vs-6144 shape mismatch on the draft head.

**Why bytecode and not source.** A stale `.pyc` silently reverts a patch and has already
invalidated a full experiment on this stack. Verification also uses *identifiers*, never
`#` comment markers — the compiler discards comments, so a comment marker is a guaranteed
false negative.

## 4. Boot both legs

```bash
bash $W/common/boot.sh e2 prefill
bash $W/common/boot.sh e2 decode
```

`boot.sh` gives arm `e2` two settings the other arms do not get:

```bash
export PREFILL_MTP=1
export EXTRA_ARGS='--json-model-override-args {"index_share_for_mtp_iteration":false}'
```

Both are essential. See `notes.md` for why prefill MTP is not optional here.

The environment is written to a **file** (`$W/e2/env_prefill.sh`) and sourced inside the
container rather than interpolated through the host → spur → docker → sh quoting chain —
the JSON argument contains double quotes and braces, and a file is the only way to be sure
what the server receives.

Watch progress — logs contain binary bytes, so **plain `grep` will not work**:

```bash
for r in prefill decode; do
  echo -n "$r ready="; strings $W/e2/$r.log | grep -c 'ready to roll'
done
```

## 5. Verify the configuration actually took effect

Do this **before** running the tests. A silently-ignored override would make this arm
measure nothing.

```bash
strings $W/e2/decode.log  | grep -o "json_model_override_args='[^']*'"
strings $W/e2/prefill.log | grep -o "speculative_algorithm='[^']*'"
```

Expected:

```
json_model_override_args='{"index_share_for_mtp_iteration":false}'
speculative_algorithm='EAGLE'
```

The second line is the one people forget: it confirms MTP is live on the **prefill** leg.

## 6. Start the router (on the prefill node)

```bash
bash $W/common/router.sh e2
sleep 25
```

## 7. Run the two acceptance tests

```bash
export DOCKER_CONFIG=/tmp/dockercfg
# criterion 1 -- 4-prompt sequential probe
spur exec 14317 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/probe.py \
  http://10.245.152.243:8120 4 24 180'

# criterion 2 -- conc=32 x 512 tokens
spur exec 14317 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/stress.py \
  http://10.245.152.243:8120 32 512 /shared_nfs/yihou_exp3way/e2/stress_c32.jsonl 900'
```

Optional, as run here: repeat conc=32 into `stress_c32_r2.jsonl`, and conc=64 into
`stress_c64.jsonl`.

## Expected output

```
conc=32 maxtok=512 elapsed=20.4s
ok      : 32/32
full tok: 30/32
dp ranks: [0, 1, 2, 3, 4, 5, 6, 7]
acc_len : min=2.45 mean=2.98 max=3.97
```

**What to check, in order of importance:**

1. `ok : 32/32` — the criterion.
2. `acc_len` present and **> 1** — otherwise MTP was silently bypassed.
3. `dp ranks: [0..7]` — traffic reached the whole DP group.
4. Zero `Traceback` in either server log.
5. The step-5 config check passed — otherwise the arm tested the default build.

`full tok` below the request count is expected (EOS before the 512-token cap).

## If it doesn't reproduce

See `notes.md` — in particular the 503 that looks like a dead backend but is a stale
router circuit breaker.

# Reproduction kit — Exp 1 (patch 1 v2, #32762 shape, full patch set)

Goal: reproduce **4/4 probe + 32/32 at conc=32** on a cross-node PD deployment of
GLM-5.2-MXFP4 with DP-attention 8 and MTP, with the full patch set live and patch 1 in
its reworked form.

Estimated time: **~25 min** — ~2 min weight load per leg, then ~6–8 min of tilelang JIT
plus DP CUDA-graph capture per leg (legs boot in parallel), ~1 min of tests.

> Cold start is slow and looks like a hang. Eight live `sglang::scheduler_DP*` processes
> mean it is still working. Be patient; do not kill it.

## 0. Prerequisites

**Machines.** Two nodes, 8 × MI355X each, on the spur cluster `crsuse2-m2m`:

```bash
sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00   # run twice, one per leg
```

Note each job id and the node's `ens3` IP. This run used:

| role | job | node | IP |
|---|---|---|---|
| prefill | 14315 | crsuse2-m2m-118 | 10.245.159.138 |
| decode | 14316 | crsuse2-m2m-003 | 10.245.157.171 |

Substitute your own job ids and IPs everywhere below, **and** in
`scripts/boot.sh` + `scripts/router.sh` (both carry a hard-coded node table — see
`scripts/NODES.md`).

**Secrets** (values not included; see `environment.md`): docker registry login if you
rebuild the image, and a spur cluster account. `export DOCKER_CONFIG=/tmp/dockercfg`
before **every** docker call.

**External paths** (not in this repo):
- model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
- image tar: `/home/yihou/infera.yihou.sglang.1.0.tar`

**Repo state:** `yihou.dev.glm5.2.mxfp4.experiment` @ `0d3e374`.
**Image:** `infera.yihou.sglang.1.0` (digest in `environment.md`).

## 1. Stage the workspace on shared storage

Every script here expects to live at `/shared_nfs/yihou_exp3way`, visible from both nodes.

```bash
export W=/shared_nfs/yihou_exp3way
mkdir -p $W/common $W/kit_patches $W/e1
cp scripts/{pd_leg_exp.sh,probe.py,stress.py}          $W/
cp scripts/{apply_arm.sh,boot.sh,router.sh,start_ctr.sh,NODES.md} $W/common/
cp patches/*.diff                                       $W/kit_patches/
cp patches/patch1_v2_32762_style.py                     $W/e1/
```

Do **not** put the JIT caches on `/home` — see `notes.md`.

## 2. Health-gate the nodes and start the container

```bash
export DOCKER_CONFIG=/tmp/dockercfg
bash $W/common/start_ctr.sh 14315      # prefill node
bash $W/common/start_ctr.sh 14316      # decode node
```

Each must print `GPUGATE True 8`. If a node prints `False`, abandon it and hold another —
spur has nodes that enumerate 8 GPUs but cannot use them.

If the image is not on the node yet: `docker load -i /home/yihou/infera.yihou.sglang.1.0.tar`.

## 3. Apply the Exp-1 patch set, and prove it reached the bytecode

```bash
for J in 14315 14316; do
  spur exec $J bash -c 'docker exec dbg2 bash /shared_nfs/yihou_exp3way/common/apply_arm.sh e1'
done
```

Both must end with `ARM e1 OK`. The script resets the tree to pristine first, then applies:

| # | what | how |
|---|---|---|
| 1 v2 | HIP/aiter padded-row trim+restore, **#32762 shape** | kit diff, then `patch1_v2_32762_style.py` rewrites its aiter block |
| 2a | `max_seqlen_k = req_to_token.shape[1]` when `needs_cpu_seq_lens=False` | `dsa_backend_dp_sync_and_page_table_rows.diff` |
| 2b | page-table row match | same diff |
| 3 | nextn `eh_proj` bf16 | `deepseek_nextn_glm52_mtp_bf16.diff` |
| 4 | uniform draft-graph decision (DP-group vote) | `eagle_worker_v2_uniform_draft_graph.diff` |

**Why the verification step is not optional.** A stale `.pyc` silently reverts a patch and
has already invalidated a full experiment on this stack. `apply_arm.sh` therefore purges
`__pycache__`, recompiles, and greps the **bytecode** for each marker — using
*identifiers*, never `#` comments (the compiler discards comments, so a comment marker is
a guaranteed false negative). Patch 2a introduces no new identifier and is checked at
source level; the script says so where it does that.

## 4. Boot both legs

```bash
bash $W/common/boot.sh e1 prefill
bash $W/common/boot.sh e1 decode
```

Boot is asynchronous (`docker exec -d`). Watch progress — the logs contain binary bytes,
so **plain `grep` will not work**; use `strings` or `grep -a`:

```bash
for r in prefill decode; do
  echo -n "$r ready="; strings $W/e1/$r.log | grep -c 'ready to roll'
done
```

Wait until both print `1`. Check for errors with
`strings $W/e1/decode.log | grep -aE "Traceback|RuntimeError|ValueError"`.

Key launcher settings for this arm (all in `scripts/pd_leg_exp.sh`): DPA=1, MTP=1
(**decode leg only** — `PREFILL_MTP=0`), EAGLE steps=3 / topk=1 / draft-tokens=4,
`--kv-cache-dtype fp8_e4m3`, ctx 32768, `--disable-custom-all-reduce` (the aiter custom
all-reduce kernel deadlocks on gfx950 during EAGLE verify), mooncake pinned to `mlx5_0`
GID 3 with dma-buf on.

## 5. Start the router (on the prefill node)

```bash
bash $W/common/router.sh e1
sleep 25
```

## 6. Run the two acceptance tests

```bash
export DOCKER_CONFIG=/tmp/dockercfg
# criterion 1 -- 4-prompt sequential probe
spur exec 14315 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/probe.py \
  http://10.245.159.138:8110 4 24 180'

# criterion 2 -- conc=32 x 512 tokens
spur exec 14315 bash -c 'docker exec dbg2 python3 /shared_nfs/yihou_exp3way/stress.py \
  http://10.245.159.138:8110 32 512 /shared_nfs/yihou_exp3way/e1/stress_c32.jsonl 900'
```

Optional, as run here: repeat the conc=32 line into `stress_c32_r2.jsonl`, and a
conc=64 run into `stress_c64.jsonl`.

## Expected output

Probe:

```
[0] OK  0.90s dp=1 acc_len=2.18 tok=24 text=' Paris. France is a country located in Western Europe. It is'
...
4/4 ok
```

Stress:

```
conc=32 maxtok=512 elapsed=19.4s
ok      : 32/32
full tok: 31/32
dp ranks: [0, 1, 2, 3, 4, 5, 6, 7]
acc_len : min=2.06 mean=2.85 max=3.91
```

**What to check, in order of importance:**

1. `ok : 32/32` — the criterion.
2. `acc_len` present and **> 1** — otherwise MTP was silently bypassed and the run proves
   nothing about spec-dec.
3. `dp ranks: [0..7]` — traffic reached the whole DP group.
4. Zero `Traceback` in either server log.

`full tok` below the request count is expected (EOS before the 512-token cap).

## If it doesn't reproduce

See `notes.md` — it lists the traps that cost time on this stack, including the 503 that
looks like a server failure but is a stale router circuit breaker.

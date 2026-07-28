# Reproduce — GLM-5.2 DSA "DPA + MTP" crash and its fix

Every command below was actually run on 2026-07-28. Paths/IPs are the real ones; swap the
job IDs and node IPs for yours. Cluster mechanics: see the `spur-cluster-usage` skill.

```bash
KIT=/home/yihou/dev/git/infera.yihou.glm5.2.mxfp4/glm52.mxfp4.spur.mooncake.packup_20260728/dpa_mtp_fix
IMG=infera.yihou.sglang.1.0
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
export DOCKER_CONFIG=/tmp/dockercfg      # REQUIRED before every docker call
```

## 0. Hold nodes and gate their GPUs

One node is enough for the fix itself (§2-§4); the PD section (§5) needs two.

```bash
JOB=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00 ~/hold_node.sh)
spur exec "$JOB" true                     # must exit 0 — a RUNNING poll is not a real hold
# GPU HEALTH GATE (spur has nodes that enumerate 8 GPUs but report is_available()=False):
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker load -i /home/yihou/$IMG.tar
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --entrypoint '' $IMG python3 -c 'import torch;print(torch.cuda.is_available(), torch.cuda.device_count())'"
# want: True 8
```
Expect `JobHoldMaxRequeue` bounces — cancel and resubmit (a retry loop landed a node
after ~10 tries during this run).

## 1. Start the container and apply the patch

```bash
CTR=dbg
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker rm -f $CTR 2>/dev/null
  docker run -d --name $CTR --network=host --ipc=host --shm-size=32G \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou --entrypoint '' $IMG sleep infinity
  # MTP needs the nextn eh_proj patch (else a 3072-vs-6144 shape crash on the draft head):
  docker cp $KIT/../patches/deepseek_nextn.unified_patch.py \
    $CTR:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py
  docker cp $KIT/scripts/mix_leg.sh $CTR:/mix_leg.sh
  docker cp $KIT/scripts/probe.py   $CTR:/tmp/probe.py
  docker cp $KIT/scripts/sweep_dpa.sh $CTR:/sweep_dpa.sh
  docker cp $KIT/patches/apply_fix.py $CTR:/tmp/apply_fix.py
  # THE FIX (idempotent, verifies each anchor matches exactly once):
  docker exec $CTR python3 /tmp/apply_fix.py
  docker exec $CTR python3 -m py_compile \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py && echo COMPILE_OK"
```
Backup of the stock file is left at `dsa_indexer.py.orig` inside the container, so
`diff -u dsa_indexer.py.orig dsa_indexer.py` regenerates the patch.

## 2. Reproduce the ORIGINAL crash (skip the patch step above)

```bash
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec -d $CTR env MY_IP=<node ens3 IP> PORT=30000 DPA=1 MTP=1 \
    LOG=/home/yihou/glm52_fix/logs/mix_dpa_mtp_baseline.log bash /mix_leg.sh"
```
~9 min later the schedulers die on the first real batch:
```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```
Full traceback: `evidence/crash_traceback_baseline.txt`. Note this is a **single node with
no PD** — contrary to the original todo, PD is not required to trigger it.

## 3. Run the FIXED single-node mix (the headline result)

```bash
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec -d $CTR env MY_IP=<node ens3 IP> PORT=30000 DPA=1 MTP=1 \
    SGLANG_DEBUG_DSA_ROWS=1 \
    LOG=/home/yihou/glm52_fix/logs/mix_dpa_mtp_fix1.log bash /mix_leg.sh"
```
`SGLANG_DEBUG_DSA_ROWS=1` (added by the patch) logs the row bookkeeping that proves the
root cause. Wait for `ready to roll` (~9 min: 2 min weights + tilelang JIT + DP graph
capture). **Be patient** — 8 live `sglang::scheduler_DP*` processes means it is working,
not hung.

```bash
# correctness (want 4/4):
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR python3 /tmp/probe.py http://127.0.0.1:30000 glm5.2-mxfp4"
# the measured proof of the root cause:
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR bash -c 'tr \"\\r\" \"\\n\" < /home/yihou/glm52_fix/logs/mix_dpa_mtp_fix1.log \
    | grep -a dsa-rows | sed \"s/.*\\[dsa-rows\\]/[dsa-rows]/\" | sort | uniq -c | sort -rn'"
# spec-dec actually firing (drive some load first, then):
#   grep -aoE 'accept len: [0-9.]+'   -> want median ~3.8 of 4 under load
# conc=64 stress:
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec -d $CTR env CONCS=64 BASE=http://127.0.0.1:30000 MODEL=glm5.2-mxfp4 \
    TOK=$MODEL ISL=1024 OSL=1024 NUM_PROMPTS=256 \
    OUTDIR=/home/yihou/glm52_fix/sweep TAG=mixfix bash /sweep_dpa.sh"
```

## 4. Regression checks (must stay green)

Same container, only the switches change. **Recreate the container between runs** — see
`NOTES_rootcause_and_fix.md` §6(a): `pkill` orphans the scheduler tree and leaks VRAM.

```bash
# DPA-only:
docker exec -d $CTR env MY_IP=<IP> PORT=30000 DPA=1 MTP=0 LOG=.../regr_dpa_only.log bash /mix_leg.sh
# MTP-only:
docker exec -d $CTR env MY_IP=<IP> PORT=30000 DPA=0 MTP=1 LOG=.../regr_mtp_only.log bash /mix_leg.sh
```

## 5. PD (2 nodes, mooncake mlx5 + dmabuf)

Uses the parent kit's `pd_leg_spur.sh` unchanged — it already encodes the whole spur
transport recipe (`DMABUF=1`, `--disaggregation-ib-device mlx5_0`, `MC_GID_INDEX=3`,
`MC_MS_AUTO_DISC=0 MC_MS_FILTERS=mlx5_0`, `MC_DISABLE_HIP_TRANSPORT=1`, NIC=ens3).

```bash
P_IP=10.245.158.91   # prefill node (197)
D_IP=10.245.156.172  # decode  node (207)

# prefill leg (DPA8, no MTP):
spur exec "$JOB2" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec -d pd_spur env ROLE=prefill MY_IP=$P_IP P_IP=$P_IP MODEL=$MODEL PORT=30000 \
    DPA=1 MTP=0 DMABUF=1 CTX=32768 GMU=0.85 LOG=.../pd_prefill_30000.log bash /pd_leg_spur.sh"

# decode leg — DPA8 ONLY (this PASSES):
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec -d $CTR env ROLE=decode MY_IP=$D_IP P_IP=$P_IP MODEL=$MODEL PORT=30001 \
    DPA=1 MTP=0 DMABUF=1 CTX=32768 GMU=0.80 TORCHINDUCTOR_COMPILE_THREADS=4 \
    LOG=.../pd_decode_dpaonly.log bash /pd_leg_spur.sh"

# router (fresh --port AND --prometheus-port on every restart; kill stale routers first):
spur exec "$JOB2" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec pd_spur bash -c 'pkill -9 -f launch_router; sleep 5'
  docker exec -d pd_spur bash -c 'python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://$P_IP:30000 8998 --decode http://$D_IP:30001 \
    --host 0.0.0.0 --port 8006 --prometheus-port 29107 > /tmp/router5.log 2>&1'"
sleep 25
spur exec "$JOB2" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec pd_spur python3 /tmp/probe.py http://$P_IP:8006 glm5.2-mxfp4"   # -> 4/4
```

`TORCHINDUCTOR_COMPILE_THREADS=4` is **required on a cold Inductor cache** — without it
the 8 DP ranks spawn 264 compile workers and deadlock during warmup
(`NOTES_rootcause_and_fix.md` §6(b)).

Setting `MTP=1` on the decode leg brings the server up fine (`ready to roll`, warmup 200
on all 8 DP ranks, spec-dec active) but **hangs on the first routed request** — a second,
independent defect. See `RESULTS.md` §"Not passing".

Verify transport is really RDMA (not TCP, not hip, not ionic):
```bash
grep -c 'installTransport, type=rdma' <leg log>   # want 8
grep -ci ionic <leg log>                          # want 0
grep -c KVTransferError <leg log>                 # want 0
```

## Success criteria

| Check | Target |
|---|---|
| Single-node mix DPA8+MTP | 4/4 probe, no `lengths.size(0) == B` |
| spec-dec active | `accept len` median ~3.8 of 4 under load |
| DP flags took | `enable_dp_attention=True, dp_size=8, ep_size=8` |
| conc=64 (1k/1k, 256 prompts) | 256/256, 0 failed |
| DPA-only / MTP-only regression | still 4/4 |
| PD decode DPA8-only | 4/4, 8× rdma, 0 ionic, 0 KVTransferError |

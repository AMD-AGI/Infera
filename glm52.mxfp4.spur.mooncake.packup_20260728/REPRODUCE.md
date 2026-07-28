# Reproduce — GLM-5.2-MXFP4 sglang mooncake PD (mlx5+dmabuf) on crsuse spur

All commands run from the spur **login** node; `spur exec <job>` enters a held compute node
(ssh to compute nodes is banned). `export DOCKER_CONFIG=/tmp/dockercfg` is required for every
`docker` call on a node (docker 29 buildx plugin discovery fails on the default config path).
Account/QOS/held-node mechanics: see the `spur-cluster-usage` skill. Model + image paths: environment.md.

```bash
KIT=/home/yihou/dev/git/infera.yihou.glm5.2.mxfp4/glm52.mxfp4.spur.mooncake.packup_20260728
IMG=infera.yihou.sglang.1.0
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
```

## 0. Hold two 8-GPU nodes (verify GPU health!)

```bash
JOB=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 10:00:00 ~/hold_node.sh)
spur exec "$JOB" true          # must exit 0 (real hold)
# GPU HEALTH GATE — some nodes enumerate GPUs but torch.cuda.is_available()=False (bad node).
# Test it before using; if False, scancel and --exclude that node, resubmit.
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --entrypoint '' $IMG \
  python3 -c 'import torch;print(torch.cuda.is_available(), torch.cuda.device_count())'"   # want: True 8
```

## A. Build the image (network-free; ~15 min) — once, then reuse

docker build backgrounded in `spur exec` gets killed by the exec-namespace teardown. Run the
mooncake rebuild inside a **host-owned detached container**, then commit. See scripts/build_dmabuf.sh
for the buildx form; the reliable path actually used was:

```bash
REPO=/home/yihou/dev/git/infera.yihou.dev
spur exec "$JOB" bash -c '
  export DOCKER_CONFIG=/tmp/dockercfg
  docker run -d --name mcbuild -v /home/yihou:/home/yihou --entrypoint "" \
    lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x sleep infinity
  docker cp '"$REPO"'/deploy/docker/patches/mooncake_cpp mcbuild:/tmp/mooncake_cpp
  docker cp '"$REPO"'/deploy/docker/scripts/build_mooncake_dmabuf.sh mcbuild:/tmp/build_mooncake_dmabuf.sh
  docker cp '"$REPO"'/deploy/docker/scripts/infera_inject_host_ionic.sh mcbuild:/usr/local/bin/infera-inject-host-ionic
  docker exec mcbuild chmod +x /usr/local/bin/infera-inject-host-ionic
  docker exec -d mcbuild bash -c "MC_CPP_PATCH_DIR=/tmp/mooncake_cpp bash /tmp/build_mooncake_dmabuf.sh > /home/yihou/glm52_spur/logs/mcbuild.log 2>&1; echo EXIT=\$? >> /home/yihou/glm52_spur/logs/mcbuild.log"
'
# wait for: DMABUF_COMPILED_IN=yes / LINKS_HSA=yes / MOONCAKE_DMABUF_BUILD_DONE / EXIT=0
spur exec "$JOB" bash -c 'export DOCKER_CONFIG=/tmp/dockercfg; docker commit \
  --change "ENTRYPOINT [\"/usr/local/bin/infera-inject-host-ionic\"]" --change "CMD [\"/bin/bash\"]" \
  mcbuild infera.yihou.sglang.1.0; docker rm -f mcbuild'
# save to NFS + load on the 2nd node (spur bans node->node ssh):
spur exec "$JOB"  bash -c 'export DOCKER_CONFIG=/tmp/dockercfg; docker save infera.yihou.sglang.1.0 -o /home/yihou/infera.yihou.sglang.1.0.tar'
spur exec "$JOB2" bash -c 'export DOCKER_CONFIG=/tmp/dockercfg; docker load -i /home/yihou/infera.yihou.sglang.1.0.tar'
```

## B. Bring up 2-node PD (mooncake RDMA over mlx5 + dmabuf)

prefill=node1, decode=node2. Start a container on each, stage scripts, launch legs.

```bash
CTR=pd_spur
P_IP=<node1 ens3 IP>;  D_IP=<node2 ens3 IP>
start_ctr(){ spur exec "$1" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker rm -f $CTR 2>/dev/null; docker run -d --name $CTR --network=host --ipc=host --shm-size=32G \
   --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --group-add render \
   --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
   -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou --entrypoint '' $IMG sleep infinity"; }
# decode container also bind-mounts the MTP nextn patch (for config B):
#   -v $KIT/patches/deepseek_nextn.unified_patch.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py:ro
start_ctr "$JOB"; start_ctr "$JOB2"
for J in "$JOB" "$JOB2"; do spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker cp $KIT/scripts/pd_leg_spur.sh $CTR:/pd_leg_spur.sh; docker cp $KIT/scripts/probe.py $CTR:/tmp/probe.py
  docker cp $KIT/scripts/sweep_dpa.sh $CTR:/sweep_dpa.sh"; done
```

### Config A — mooncake RDMA + DP-attention (both legs symmetric DP8), no MTP  [scales to conc≥128]
```bash
spur exec "$JOB"  bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec -d $CTR env \
  ROLE=prefill MY_IP=$P_IP P_IP=$P_IP MODEL=$MODEL PORT=30000 DPA=1 MTP=0 DMABUF=1 \
  LOG=/home/yihou/glm52_spur/logs/pd_prefill_30000.log bash /pd_leg_spur.sh"
spur exec "$JOB2" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec -d $CTR env \
  ROLE=decode  MY_IP=$D_IP P_IP=$P_IP MODEL=$MODEL PORT=30001 DPA=1 MTP=0 DMABUF=1 \
  LOG=/home/yihou/glm52_spur/logs/pd_decode_30001_dpa.log bash /pd_leg_spur.sh"
```

### Config B — mooncake RDMA + MTP (EAGLE steps=3 on decode), TP8, no DPA  [1.6x faster decode]
```bash
# prefill DPA=0 MTP=0 CTX=400000 MAX_RUNNING=128 GMU=0.85 ; decode DPA=0 MTP=1 GMU=0.80 (nextn patch mounted)
```
(⚠️ **DPA + MTP together does NOT work** — DSA indexer topk crashes under EAGLE draft-extend + DP.
See notes.md. Run A or B, not both fused.)

Wait for both legs to print `ready to roll` (cold start ~7-8 min: weights ~2min + DP cudagraph).

### Router + correctness + stress
```bash
# router in the prefill container (fresh prometheus-port each restart; pkill stale routers first):
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec $CTR bash -c 'cat > /run_router.sh <<EOF
#!/bin/bash
exec python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://$P_IP:30000 8998 --decode http://$D_IP:30001 \
  --host 0.0.0.0 --port 8002 --prometheus-port 29077 > /tmp/router.log 2>&1
EOF
chmod +x /run_router.sh'; docker exec -d $CTR bash /run_router.sh"
sleep 16
# correctness (want 4/4):
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec $CTR python3 /tmp/probe.py http://$P_IP:8002 glm5.2-mxfp4"
# conc=128 (1k/1k, 512 prompts):
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec -d $CTR env \
  CONCS=128 BASE=http://$P_IP:8002 MODEL=glm5.2-mxfp4 TOK=$MODEL ISL=1024 OSL=1024 \
  OUTDIR=/home/yihou/glm52_spur/sweep TAG=spur bash /sweep_dpa.sh > /home/yihou/glm52_spur/logs/sweep_c128.log 2>&1"
```
Verify RDMA (not TCP): `grep 'installTransport, type=rdma' pd_prefill...log` (on mlx5_0 GID3),
0 `MC_FORCE_TCP`, 0 `KVTransferError`.

## C. infera kvd + kv-aware routing (single-node mix)

Needs the infera package in the container (pip-install from repo source) + etcd + kvd daemon.

```bash
# 1. install infera into the container (deps from pypi):
spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker cp <repo>/infera $CTR:/opt/inferapkg/infera; docker cp <repo>/pyproject.toml <repo>/README.md $CTR:/opt/inferapkg/
  docker exec $CTR bash -c 'cd /opt/inferapkg && SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 pip install --no-build-isolation .'"
# 2. etcd (v3 HTTP/JSON gateway is what infera uses):
spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker run -d --name etcd0 --network=host quay.io/coreos/etcd:v3.5.14 \
  /usr/local/bin/etcd --name n0 --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://$D_IP:2379 \
  --listen-peer-urls http://0.0.0.0:2380 --initial-advertise-peer-urls http://$D_IP:2380 --initial-cluster n0=http://$D_IP:2380"
# 3. kvd daemon (L3 cache; dockerd-managed so it survives exec teardown):
spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec $CTR bash -c 'mkdir -p /run/infera-kvd /run/kvd-long'
  docker exec -d $CTR bash -c 'python3 -m infera.kvd --socket /run/infera-kvd/infera-kvd.sock --max-bytes 8G --long-path /run/kvd-long --long-bytes 32G > /home/yihou/glm52_spur/logs/kvd.log 2>&1'"
# 4. infera sglang worker (mix, kv-events on, kvd wired). See scripts/infera_worker.sh.
#    KEY: --hicache-size 40 (fixed, small) --hicache-io-backend direct --hicache-write-policy write_through_selective
#    (a big hicache ratio OOMs/hangs the host alloc; the kernel io backend GPU-faults on gfx950 write-back — see notes).
spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker cp $KIT/scripts/infera_worker.sh $CTR:/infera_worker.sh
  docker exec -d $CTR env MY_IP=$D_IP PORT=30000 ETCD=$D_IP:2379 bash /infera_worker.sh"
# wait for 'ready to roll'; confirm: worker registered in etcd (/infera/workers/<ip>:30000),
#   8x 'Creating dynamic storage backend infera-kvd' in the worker log (kvd wired).
# 5. infera kv-aware router:
spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec -d $CTR bash -c \
  'python3 -m infera.server --host 0.0.0.0 --port 8100 --discovery-backend etcd --etcd-endpoint $D_IP:2379 \
   --etcd-prefix /infera/workers/ --request-transport http --router-policy kv-aware --kv-event-transport zmq \
   --router-tokenizer-path $MODEL > /home/yihou/glm52_spur/logs/infera_router.log 2>&1'"
# test: probe http://$D_IP:8100 ; router log shows 'pick policy=kv-aware ... picked=<worker> cache_hits=N'.
```

## Success criteria
- Build: `DMABUF_COMPILED_IN=yes` + `LINKS_HSA=yes`.
- PD A: 4/4 probe; conc=128 512/512; `installTransport type=rdma` on mlx5_0 GID3; 0 TCP/KVTransferError.
- PD B: 4/4 probe; draft head loads (no 3072/6144 crash); accept_len ~2.7-3.0; conc=128 512/512.
- kv-aware: router log `pick policy=kv-aware`; worker in etcd with `kv=yes`; kvd backend created ×8.

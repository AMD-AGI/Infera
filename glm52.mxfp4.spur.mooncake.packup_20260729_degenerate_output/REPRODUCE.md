# Exact reproduction

Everything below is copy-pasteable. No patches are applied anywhere — all
three results were produced on **clean upstream sglang**.

## 0. Prerequisites

| | |
|---|---|
| cluster | crsuse spur, partition `amd-spur`, QoS `amd-burst-qos` |
| image tar | `/home/yihou/infera.yihou.sglang.1.0.tar` (NFS, ~30 GB) |
| models | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB)<br>`/shared_nfs/huggingface_models/zai-org/GLM-5.2-FP8` (704 GB) |
| scripts | `scripts/` in this kit; the launch script must be reachable **inside** the container — it is bind-mounted via `/home/yihou` |
| secrets | none |

Copy `scripts/mix_leg.sh` to a path under `/home/yihou` so the container sees
it (the examples below assume `/home/yihou/glm52_fix/mix_leg.sh`).

## 1. Hold two nodes

```bash
# hold_node.sh must be `sleep infinity`, NOT `sleep 36000`.
# A 10h sleep under a 12h wall makes the job exit on its own, mid-experiment.
cat > /home/yihou/hold_node.sh <<'EOS'
#!/bin/bash
echo "held: $(hostname) job=$SLURM_JOB_ID"
sleep infinity
EOS
chmod +x /home/yihou/hold_node.sh

for i in 1 2; do
  sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00 \
         /home/yihou/hold_node.sh
done
squeue -u $USER          # note the two job ids -> $J_MXFP4, $J_FP8
```

**Health-gate every node** — spur has nodes that enumerate 8 GPUs but report
`torch.cuda.is_available() == False`:

```bash
spur exec $J bash -c 'hostname; ip -4 addr show ens3 | grep -oP "inet \K[0-9.]+"'
```

## 2. Load the image and start a container (both nodes)

```bash
# docker load must run in the FOREGROUND. Backgrounding a docker client inside
# `spur exec` gets it killed when the exec namespace tears down.
spur exec $J bash -c '
  export DOCKER_CONFIG=/tmp/dockercfg
  docker load -i /home/yihou/infera.yihou.sglang.1.0.tar'

spur exec $J bash -c '
  export DOCKER_CONFIG=/tmp/dockercfg
  docker rm -f dbg2 2>/dev/null
  docker run -d --name dbg2 --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --group-add video --group-add render \
    --cap-add CAP_IPC_LOCK --cap-add CAP_SYS_PTRACE \
    --security-opt seccomp=unconfined --security-opt label=disable \
    -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou \
    infera.yihou.sglang.1.0 sleep infinity
  sleep 3
  docker exec dbg2 python3 -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"'
# must print: True 8
```

## 3. Confirm the tree is clean upstream

```bash
spur exec $J bash -c 'docker exec dbg2 bash /home/yihou/glm52_fix/envcap.sh'
```

All five patch markers must read `0`. If not, restore from the `.orig` /
`.bug2fix_orig` backups and purge `__pycache__` before continuing — a stale
`.pyc` will silently run patched bytecode from an unpatched-looking source.

## 4. Launch the two servers

Single-node mix, DP-attention on, **no MTP**. `mix_leg.sh` adds
`--disable-custom-all-reduce` independently of MTP (the aiter custom
all-reduce deadlocks on gfx950 under concurrency).

```bash
# --- MXFP4 (node A) ---
spur exec $J_MXFP4 bash -c 'docker exec -d dbg2 bash -c "
  MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 \
  SERVED=glm5.2-mxfp4 MY_IP=<A_IP> DPA=1 MTP=0 PORT=30000 \
  LOG=/tmp/mxfp4_mix.log bash /home/yihou/glm52_fix/mix_leg.sh"'

# --- FP8 (node B). GMU=0.90 because the weights are 704 GB / 88 GB per GPU ---
spur exec $J_FP8 bash -c 'docker exec -d dbg2 bash -c "
  MODEL=/shared_nfs/huggingface_models/zai-org/GLM-5.2-FP8 \
  SERVED=glm5.2-fp8 MY_IP=<B_IP> DPA=1 MTP=0 PORT=30000 GMU=0.90 \
  LOG=/tmp/fp8_mix.log bash /home/yihou/glm52_fix/mix_leg.sh"'
```

Cold start is **8–12 min** (weights ~1.5 min, then a long silent tilelang JIT +
CUDA-graph capture). 8 live `sglang::scheduler` processes means it is working,
not hung.

```bash
spur exec $J bash -c 'docker exec dbg2 curl -s -m3 -o /dev/null -w "%{http_code}\n" \
  http://127.0.0.1:30000/health'   # wait for 200
```

## 5. The three measurements

```bash
# ---- RESULT 1: MXFP4, chat template, model-recommended sampling ----
spur exec $J_MXFP4 bash -c '
  docker cp /home/yihou/glm52_fix/chat_stress.py dbg2:/tmp/cs.py
  docker exec dbg2 python3 /tmp/cs.py \
    --url http://127.0.0.1:30000 --model glm5.2-mxfp4 \
    --n 128 --ntok 512 --temp 1.0 --top-p 0.95 --conc 128 --out /tmp/cs1'

# ---- RESULT 2: FP8, same ----
spur exec $J_FP8 bash -c '
  docker cp /home/yihou/glm52_fix/chat_stress.py dbg2:/tmp/cs.py
  docker exec dbg2 python3 /tmp/cs.py \
    --url http://127.0.0.1:30000 --model glm5.2-fp8 \
    --n 128 --ntok 512 --temp 1.0 --top-p 0.95 --conc 128 --out /tmp/cs2'

# ---- RESULT 3: FP8, raw /generate, greedy (the flawed original setup) ----
spur exec $J_FP8 bash -c '
  docker cp /home/yihou/glm52_fix/probe_onset.py dbg2:/tmp/po.py
  docker exec dbg2 python3 /tmp/po.py \
    --url http://127.0.0.1:30000 --n 128 --ntok 512 --temp 0 \
    --tag fp8raw --out /tmp/cs3'
```

### Expected output

```
RESULT 1  chat t=1.0 p=0.95 conc=128   ok=128 err=0 looping=0 (0.0%)  ~18s
RESULT 2  chat t=1.0 p=0.95 conc=128   ok=128 err=0 looping=0 (0.0%)  ~33s
RESULT 3  raw  t=0        conc=128     ok=128        looping=1 (0.8%)
            LOOP i=117 onset=0 uniq=3 dp=7
            '4.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1...'
```

The loop rate is low and stochastic; RESULT 3 needs several rounds to be more
than anecdotal. Earlier rounds of the same configuration produced 1–6 per 128.

## 6. Retrieve the data

```bash
for pair in "$J_MXFP4:cs1" "$J_FP8:cs2" "$J_FP8:cs3"; do
  J=${pair%%:*}; d=${pair##*:}
  spur exec $J bash -c "mkdir -p /home/yihou/out/$d;
                        docker cp dbg2:/tmp/$d/. /home/yihou/out/$d/"
done
```

`docker cp` into `/tmp` on the host does **not** survive the node — copy into
`/home/yihou` (NFS) instead.

## 7. Verify the sampling knobs (optional, ~2 min)

```bash
spur exec $J_FP8 bash -c '
  docker cp /home/yihou/glm52_fix/answer_three.py dbg2:/tmp/a3.py
  docker exec dbg2 python3 /tmp/a3.py --url http://127.0.0.1:30000 \
    --model glm5.2-fp8 --only 2'
```

`t=2.0` must degrade into noise while `t=1.0,p=0.95` stays fluent. That is the
proof the knobs are wired through.

## 8. Release

```bash
scancel $J_MXFP4 $J_FP8
```

Kill the server by explicit PID rather than a broad `pkill -f` (which can match
your own shell), and confirm `rocm-smi` reads 0 % VRAM before reusing a node.

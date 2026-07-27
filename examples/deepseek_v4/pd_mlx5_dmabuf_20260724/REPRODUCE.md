# 严格复现 — DSv4 1P1D mlx5 + dmabuf

从零到 smoke 通过 + bench。全程在 spur login 节点操作,通过 `spur exec` 进计算节点
(**spur 禁 ssh 计算节点**,必须用 `spur exec`)。账户/QOS/坏节点见 spur-cluster-usage skill。

前置:你在 spur login 节点,属于某账户(本例 amd-primus),模型已在
`/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`。脚本在本目录 `scripts/`
(在 NFS home 下,计算节点可见)。

---

## 步骤 0 — 占两个 8 卡节点(prefill P + decode D)

```bash
cat > ~/hold_node.sh <<'EOF'
#!/bin/bash
echo "held: $(hostname) job=$SLURM_JOB_ID gpus=$ROCR_VISIBLE_DEVICES"
sleep 28800
EOF
# 各占一个(-q amd-burst-qos,坏节点 --exclude;必须 spur exec <job> true 验证真占住)
JOB_P=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 06:00:00 ~/hold_node.sh)
JOB_D=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 06:00:00 ~/hold_node.sh)
spur exec "$JOB_P" true && spur exec "$JOB_D" true   # 都要退出 0
# 记下两节点的 mlx5 IP(= ens3 的 IP):
spur exec "$JOB_P" bash -c 'ip -4 -o addr show dev ens3 | awk "{print \$4}"'   # -> P_IP
spur exec "$JOB_D" bash -c 'ip -4 -o addr show dev ens3 | awk "{print \$4}"'   # -> D_IP
```

## 步骤 1 — 在 P 上从 base 镜像编出 dmabuf 镜像(★核心 fix)

```bash
IMG_BASE=lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x
KIT=/home/yihou/dev/git/infera.yihou.dev/examples/deepseek_v4/pd_mlx5_dmabuf_20260724
spur exec "$JOB_P" bash -c "docker pull $IMG_BASE"
# 起一个临时容器编译
spur exec "$JOB_P" bash -c "docker rm -f mc_build 2>/dev/null; docker run -d --name mc_build \
  --network=host --ipc=host --shm-size=32g --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou --entrypoint '' $IMG_BASE sleep infinity"
# 跑重编脚本(见 scripts/build_mc_dmabuf.sh;应打印 DMABUF_PRESENT=yes / HIP_GATE 存在)
spur exec "$JOB_P" bash -c "docker exec mc_build bash $KIT/scripts/build_mc_dmabuf.sh"
# commit 成镜像
spur exec "$JOB_P" bash -c "docker commit mc_build dsv4-sgl-dmabuf:mlx5 && docker rm -f mc_build"
```

**这一步做了什么**(详见 GOTCHAS §1):base 镜像的 Mooncake 把 `ibv_reg_dmabuf_mr` 分支
**编译掉了**(`USE_HIP_DMABUF` 只加到 transfer_engine target,没加到真正含 `rdma_context.cpp`
的 rdma_transport target)。脚本:①往 `rdma_transport/CMakeLists.txt` 追加 dmabuf 传播 +
`hsa-runtime64` 链接;②`-DUSE_HIP_DMABUF=ON` 重编;③把 `installTransport("hip")` gate 到
`MC_ENABLE_HIP_TRANSPORT`(默认关,保证跨节点走 RDMA)。验证 `rdma_context.cpp.o` 里确有
`ibv_reg_dmabuf_mr` + `hsa_amd_portable_export_dmabuf`。

## 步骤 2 — 把镜像搬到 D 节点(NFS 中转,ssh 被禁)

```bash
TAR=/home/yihou/dev/git/infera.yihou.dev/temp/dsv4-sgl-dmabuf.tar
spur exec "$JOB_P" bash -c "docker save dsv4-sgl-dmabuf:mlx5 -o $TAR"   # ~27GB, 存到 NFS
spur exec "$JOB_D" bash -c "docker load -i $TAR"                        # D 节点 load
```

## 步骤 3 — 两节点起 PD 容器

```bash
KIT=/home/yihou/dev/git/infera.yihou.dev/examples/deepseek_v4/pd_mlx5_dmabuf_20260724
for J in "$JOB_P" "$JOB_D"; do
  spur exec "$J" bash -c "IMG=dsv4-sgl-dmabuf:mlx5 bash $KIT/scripts/start_ctr_mlx5.sh pd_mlx5"
done
# 验证:容器内 mlx5 可见 + mooncake 带 dmabuf
spur exec "$JOB_P" bash -c "docker exec pd_mlx5 bash -c 'ibv_devinfo -d mlx5_0 | grep -c PORT_ACTIVE; \
  ldd \$(python3 -c \"import mooncake.engine as e;print(e.__file__)\") | grep -c hsa-runtime64'"
```

## 步骤 4 — 起 prefill + decode server(强制 mlx5,冷启动 ~22min)

```bash
P_IP=10.245.156.178; D_IP=10.245.150.73   # 换成步骤0 查到的
KIT=/home/yihou/dev/git/infera.yihou.dev/examples/deepseek_v4/pd_mlx5_dmabuf_20260724
# prefill(bootstrap host = 自己)
spur exec "$JOB_P" bash -c "docker exec -d pd_mlx5 bash -c \
  'ROLE=prefill MY_IP=$P_IP P_IP=$P_IP CONC=128 LOG=/home/yihou/…/prefill.log bash $KIT/scripts/pd_server_mlx5.sh'"
# decode
spur exec "$JOB_D" bash -c "docker exec -d pd_mlx5 bash -c \
  'ROLE=decode  MY_IP=$D_IP P_IP=$P_IP CONC=128 LOG=/home/yihou/…/decode.log bash $KIT/scripts/pd_server_mlx5.sh'"
# 等两侧 'The server is fired up and ready to roll!'
#   冷启动:权重加载两阶段 ~1300s(evidence/*.log 有 elapsed=1302s)+ warmup。
#   判活:看 VRAM 持续上涨(rocm-smi),别急着判死。
```

强制 mlx5 的关键(`scripts/pd_server_mlx5.sh` 已内置):
- `--disaggregation-ib-device mlx5_0`(官方 per-GPU NIC 指定,全 8 卡都用 mlx5_0)
- `MC_MS_AUTO_DISC=0 MC_MS_FILTERS=mlx5_0`(Mooncake 层双保险,别让它抓 8 张 ionic)
- `SGLANG_LOCAL_IP_NIC=ens3 GLOO_SOCKET_IFNAME=ens3 MC_GID_INDEX=3 NCCL_IB_DISABLE=1`
- **不设** `MC_ENABLE_HIP_TRANSPORT`(跨节点必须走 RDMA,不能 hip IPC)

## 步骤 5 — 起 router + smoke

```bash
# router 在 P 节点,port 8200(prefill 带 bootstrap 8998)
spur exec "$JOB_P" bash -c "docker exec -d pd_mlx5 bash -c \
  'python3 -m sglang_router.launch_router --pd-disaggregation \
   --prefill http://$P_IP:30000 8998 --decode http://$D_IP:30000 \
   --host 0.0.0.0 --port 8200 --policy round_robin > /home/yihou/…/router.log 2>&1'"
# smoke(必须经 router;直连 prefill 会报 "without bootstrap room id")
spur exec "$JOB_P" bash -c "docker exec pd_mlx5 curl -s http://127.0.0.1:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro\",\
       \"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One word.\"}],\
       \"max_tokens\":16,\"temperature\":0}'"
# 期望: ...\"content\":\"Paris\"...  → results/smoke_result.json
```

## 步骤 6 — bench_serving 扫并发

```bash
KIT=/home/yihou/dev/git/infera.yihou.dev/examples/deepseek_v4/pd_mlx5_dmabuf_20260724
spur exec "$JOB_P" bash -c "docker exec -d pd_mlx5 bash -c \
  'ISL=4096 OSL=512 NP_CAP=512 bash $KIT/scripts/bench_sweep_mlx5.sh \"1 16 32 64 128 256\" 8200 <outdir>'"
# ★bench 入口:脚本内用 python3 -m sglang.benchmark.serving + PYTHONPATH=/sgl-workspace/sglang/python
#   (镜像里 sglang.bench_serving 那个 shim 是坏的,见 GOTCHAS §4)
# 结果 <outdir>/mlx5_c*.jsonl。控制 context:ISL+OSL 必须 < server context_length(9472)。
```

## 步骤 7 — 清理

```bash
for J in "$JOB_P" "$JOB_D"; do spur exec "$J" bash -c 'docker rm -f pd_mlx5 2>/dev/null'; done
scancel "$JOB_P" "$JOB_D"   # 容器不随 job 自动清,务必先 docker rm 再 scancel
```

## 验证成功的判据

- 两侧 server 各打印 `The server is fired up and ready to roll!`
- decode.log 里 `installTransport, type=rdma` on `RDMA device: mlx5_0 ... GID_Index 3`,
  **且全程 grep ionic = 0**(证明真的只走 mlx5)。
- smoke 返回 "Paris"(以及 "2+2"→4, "largest planet"→Jupiter)。
- decode 稳态 VRAM ~237/288 GiB(= 159 权重 + 78 KV,**没翻倍/没 OOM**)。
- bench:conc 单调升到 128 峰值 ~26.6k tok/s。

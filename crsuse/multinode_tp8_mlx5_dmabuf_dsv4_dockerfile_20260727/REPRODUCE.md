# 严格复现 — SGLang dma-buf Dockerfile

全程在 spur login 节点操作,`spur exec` 进计算节点(**spur 禁 ssh 计算节点**)。
账户/QOS/坏节点见 spur-cluster-usage skill。注意:`amd-burst-qos` 一度失效,可用**默认 QOS**
(`sbatch` 不带 `-q`)提交。

前置:模型已在 `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`;本仓 checkout 到
分支 `yihou.dev.sglang.mooncake.experiment`(含 `Dockerfile.sglang.dmabuf`)。KIT 指本交付目录。

```bash
KIT=/home/yihou/dev/git/infera.yihou.dev/examples/deepseek_v4/sglang_dmabuf_dockerfile_20260727
REPO=/home/yihou/dev/git/infera.yihou.dev
```

---

## A. 从 Dockerfile 严格 build 镜像

```bash
# 占一个节点(默认 QOS;坏节点 --exclude;spur exec <job> true 验证真占住)
JOB=$(sbatch --parsable -p amd-spur -N1 -G8 -t 05:00:00 --exclude=crsuse2-m2m-226 ~/hold_node.sh)
spur exec "$JOB" true
# docker build(context=repo 根;Dockerfile 会 COPY patches + build 脚本)
spur exec "$JOB" bash -c "docker build -f $REPO/deploy/docker/Dockerfile.sglang.dmabuf -t dsv4-sgl-dmabuf:v1 $REPO"
# 期望构建日志出现: [transfer_engine_impl.diff] applied / [rdma_auto_chunk_mr_2017.diff] applied
#                    rdma_transport: HIP dmabuf MR registration enabled
#                    DMABUF_COMPILED_IN=yes ... MOONCAKE_DMABUF_BUILD_DONE
# 成品验证:
spur exec "$JOB" bash -c "docker run --rm --entrypoint '' dsv4-sgl-dmabuf:v1 bash -c \
  'ldd \$(python3 -c \"import mooncake.engine as e;print(e.__file__)\") | grep -c hsa-runtime64; \
   python3 -c \"from mooncake.engine import TransferEngine; print(1)\"'"
```

**Dockerfile 做了什么**(详见 GOTCHAS §1):
1. apply 仓内两个 patch(`apply_mooncake_cpp_patches.sh`):HIP-transport gate + 超 max_mr_size 分块。
2. 追加 `USE_HIP_DMABUF` 到 `rdma_transport` target(仓内 patch 没做的一步,dmabuf 核心)。
3. `cmake -DUSE_HIP_DMABUF=ON` 重编 engine.so,自校验 `ibv_reg_dmabuf_mr` 确实编进 `rdma_context.o`。

## B. 2 节点 mlx5 + dmabuf PD 跑通

```bash
# 占第 2 个节点(decode);把 v1 镜像 save->NFS tar->第2节点 load
JOB_P=$JOB   # build 的那台当 prefill
JOB_D=$(sbatch --parsable -p amd-spur -N1 -G8 -t 05:00:00 --exclude=crsuse2-m2m-226,$(squeue -j $JOB_P -h -o %N) ~/hold_node.sh)
spur exec "$JOB_D" true
TAR=/home/yihou/dev/git/infera.yihou.dev/temp/dsv4-sgl-dmabuf-v1.tar
spur exec "$JOB_P" bash -c "docker save dsv4-sgl-dmabuf:v1 -o $TAR"
spur exec "$JOB_D" bash -c "docker load -i $TAR"
# 各节点 mlx5 IP + GID:
P_IP=$(spur exec "$JOB_P" bash -c 'ip -4 -o addr show dev ens3|awk "{print \$4}"|cut -d/ -f1')
D_IP=$(spur exec "$JOB_D" bash -c 'ip -4 -o addr show dev ens3|awk "{print \$4}"|cut -d/ -f1')
# 起容器(带 host libionic 注入 entrypoint)
for J in "$JOB_P" "$JOB_D"; do spur exec "$J" bash -c "IMG=dsv4-sgl-dmabuf:v1 bash $KIT/scripts/start_ctr_mlx5.sh pd_mlx5"; done
# 起 PD leg(强制 mlx5,GID3,不开 hip);冷启动 ~22min(两阶段权重加载)
spur exec "$JOB_P" bash -c "docker exec -d pd_mlx5 bash -c 'ROLE=prefill MY_IP=$P_IP P_IP=$P_IP CONC=128 GID=3 LOG=/tmp/p.log bash $KIT/scripts/pd_server_mlx5.sh'"
spur exec "$JOB_D" bash -c "docker exec -d pd_mlx5 bash -c 'ROLE=decode  MY_IP=$D_IP P_IP=$P_IP CONC=128 GID=3 LOG=/tmp/d.log bash $KIT/scripts/pd_server_mlx5.sh'"
# 等两侧 'ready to roll'。起 router(唯一 prometheus-port 避冲突):
spur exec "$JOB_P" bash -c "docker exec -d pd_mlx5 bash -c 'python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://$P_IP:30000 8998 --decode http://$D_IP:30000 --host 0.0.0.0 --port 8200 --prometheus-port 29055 --policy round_robin > /tmp/r.log 2>&1'"
# smoke(必须经 router):
spur exec "$JOB_P" bash -c "docker exec pd_mlx5 curl -s http://127.0.0.1:8200/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{\"model\":\"/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro\",\
  \"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One word.\"}],\"max_tokens\":16,\"temperature\":0}'"
# 期望 "Paris"。验证不翻倍:decode 稳态 ~237/288 GiB;grep ionic=0;installTransport type=rdma。
```

## C. 单机 TP4 P↔D — 三种传输方式对比(同节点)

一个 8 卡节点,P=GPU0-3、D=GPU4-7。三个脚本区别只在传输:

```bash
JOB=<8卡节点>; CTR=tp4
spur exec "$JOB" bash -c "IMG=dsv4-sgl-dmabuf:v1 bash $KIT/scripts/start_ctr_mlx5.sh $CTR"
MY_IP=$(spur exec "$JOB" bash -c 'ip -4 -o addr show dev ens3|awk "{print \$4}"|cut -d/ -f1')

# --- C1: mlx5 RDMA loopback (不开 hip) —— 唯一稳定 ---
spur exec "$JOB" bash -c "docker exec -d $CTR bash -c 'ROLE=prefill MY_IP=$MY_IP PORT=30000 BASE_GPU=0 MEMFRAC=0.85 NIC_DEV=mlx5_0 GID=3 BOOTSTRAP=8998 LOG=/tmp/p.log bash $KIT/scripts/pd_server_tp4_rdma.sh'"
spur exec "$JOB" bash -c "docker exec -d $CTR bash -c 'ROLE=decode  MY_IP=$MY_IP PORT=30100 BASE_GPU=4 MEMFRAC=0.85 NIC_DEV=mlx5_0 GID=3 LOG=/tmp/d.log bash $KIT/scripts/pd_server_tp4_rdma.sh'"
# router prefill :30000 + decode :30100，smoke → Paris/4/Jupiter 全对。

# --- C2: ionic RDMA loopback (不开 hip) —— 不稳(1次后 session 死) ---
#   同 C1，把 NIC_DEV=ionic_0 GID=1（脚本已带 RDMAV_FORK_SAFE=1）。

# --- C3: hip transport (MC_ENABLE_HIP_TRANSPORT=1) —— 不通 ---
spur exec "$JOB" bash -c "docker exec -d $CTR bash -c 'ROLE=prefill MY_IP=$MY_IP PORT=30000 BASE_GPU=0 MEMFRAC=0.85 BOOTSTRAP=8998 LOG=/tmp/p.log bash $KIT/scripts/pd_server_tp4_hip.sh'"
#   (pd_server_tp4_hip.sh 内置 MC_ENABLE_HIP_TRANSPORT=1)。smoke 会 500: hipIpc address not found。
```

**★ 关键坑**(否则复现失败,详见 GOTCHAS):
- TP4 每卡 210GB 权重,`--mem-fraction-static` 必须 **≥0.76**(用 0.85);0.42 会报 "no GPU memory for KV"。
- router 反复起会 `Address already in use`(host 网络端口 + prometheus port 都要换新的;彻底 kill 旧 router)。
- 同机双实例 aiter 抢 `/tmp/aiter_configs/*.lock`,冷启动很慢(TP4 ~1200s)但**不是死锁**,会推进。

## D. ionic dmabuf 崩溃复现(2 节点 TP8,反证 mlx5 方案价值)

```bash
# probe_and_repro.sh: 每 tick 抢一个新节点跑 repro_ionic_oom.sh,判定是否在 KV 注册步崩。
# 用 cron 每 10min 跑一次(集群 NODE_FAIL 抽风时靠它反复试)。已复现:
#   node 215: KV 注册步 NODE_FAIL;  node 322: KV 注册后 mooncake conn.py SIGSEGV(evidence/)。
bash $KIT/scripts/probe_and_repro.sh   # 见 evidence/ionic_repro_watch.log
```

## E. 清理

```bash
for J in $JOB_P $JOB_D; do spur exec "$J" bash -c 'docker rm -f pd_mlx5 tp4 2>/dev/null'; done
scancel $JOB_P $JOB_D
```

## 验证成功判据

- build: `DMABUF_COMPILED_IN=yes` + 成品 `ldd engine.so | grep hsa-runtime64` = 1。
- 2节点 PD: 两侧 ready；smoke Paris；decode 237/288（不翻倍）；`installTransport type=rdma` on mlx5，ionic=0。
- 单机 TP4: mlx5 三 smoke 全对；ionic 1次后 `session not alive`；hip `address not found`。

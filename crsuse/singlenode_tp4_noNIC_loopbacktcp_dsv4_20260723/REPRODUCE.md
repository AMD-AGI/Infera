# 严格复现 — DSv4 + SGLang PD 分离 on spur

节点内 1P1D(✅ 完整可复现)+ 跨节点(⏸️ 卡点已标注)。全程 login 节点操作,`spur exec`
进计算节点。基础集群姿势(占卡/burst-qos/绝对路径挂载)见上一级 `../../spur_repro/REPRODUCE.md` 和
`../../spur_repro/GOTCHAS.md`;本文件只讲 PD 增量。

前置:你在 spur login 节点,用户属某账户(本例 amd-primus),
模型在 `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`。

---

# A. 节点内 1P1D(✅ MVP,推荐先跑通这个)

一个 8 卡节点上:prefill=TP4 GPU0-3、decode=TP4 GPU4-7,KV 走 loopback(`mooncake_tcp`,
零 RDMA)。这是验证整套 PD wiring 的最小闭环。

## 步骤 1 — 占一个 8 卡节点

```bash
JOB=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 08:00:00 \
       --exclude=crsuse2-m2m-036,crsuse2-m2m-149 scripts/hold_node.sh)
spur exec "$JOB" true && echo "held job=$JOB node=$(squeue -j $JOB -h -o %N)"
# JobHoldMaxRequeue 见 ../GOTCHAS.md #1(--exclude 坏节点 + burst-qos)
```

## 步骤 2 — 起容器(★挂载用绝对路径,含 /dev/infiniband + IPC_LOCK)

```bash
IMG=rocm/sgl-dev:sglang-0.5.13.post1-rocm720-mi35x-mori-0615
spur exec "$JOB" bash -c "
  docker rm -f pd_probe 2>/dev/null || true
  docker run -d --name pd_probe --network=host --ipc=host --shm-size=32g \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render \
    --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v /shared_nfs:/shared_nfs -v /home/\$USER:/home/\$USER \
    --entrypoint '' $IMG sleep infinity
"
# /dev/infiniband + IPC_LOCK + memlock=-1 为 PD transfer engine 备着(节点内 tcp 其实不需要,
# 但跨节点需要,统一带上无害)。
```

## 步骤 3 — 起 prefill(TP4 GPU0-3)+ decode(TP4 GPU4-7),各 ~13min 冷启动

```bash
WORK=/home/$USER/dev/git/infera.yihou.dev/examples/deepseek_v4/spur_repro/pd_disagg
# prefill 先起(GPU0-3, :30000, bootstrap 8998)
spur exec "$JOB" bash -c "docker exec -d pd_probe bash -c 'cd $WORK && ROLE=prefill BASE_GPU=0 PORT=30000 CONC=32 bash scripts/pd_server.sh'"
# 等 prefill 'ready to roll'(权重加载 staggered,慢 rank ~580s),再起 decode
spur exec "$JOB" bash -c "docker exec -d pd_probe bash -c 'cd $WORK && ROLE=decode  BASE_GPU=4 PORT=30100 CONC=32 bash scripts/pd_server.sh'"
# 判活:看 xxx.log 的 'Load weight end' 计数 + rocm-smi VRAM 上涨(见 ../GOTCHAS.md #5)
```

`pd_server.sh` 内固化的关键点(一个都不能少):
- R4 recipe:`--attention-backend dsv4` + 全套 `SGLANG_OPT_USE_FUSED_COMPRESS*` env
- PD wiring:`--disaggregation-mode <role>` + `--disaggregation-transfer-backend mooncake_tcp`
- prefill 带 `--disaggregation-bootstrap-port 8998`;P_IP=D_IP=127.0.0.1;decode 用 :30100

## 步骤 4 — 起 router :8100(mini-lb,1P1D loopback)

```bash
spur exec "$JOB" bash -c "docker exec -d pd_probe bash -c 'cd $WORK && P_IPS=127.0.0.1:30000 D_IPS=127.0.0.1:30100 bash scripts/pd_router.sh'"
```

## 步骤 5 — smoke(过 router,走完整 P→D 路径)

```bash
spur exec "$JOB" bash -c 'docker exec pd_probe curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=? Answer in one word.\"}],\"max_tokens\":16,\"temperature\":0}"'
# 期望: content:"Two."  → 见 results/pd_smoke_result.json
```

## 步骤 6 — 压测(可选,过 router)

```bash
spur exec "$JOB" bash -c "docker exec -d pd_probe bash -c 'cd $WORK && bash scripts/pd_sweep.sh \"2 32\" 8100 $WORK/results pd_intranode > $WORK/results/pd_sweep.log 2>&1'"
# ★ docker exec -d(detached),否则 spur exec 会话结束会杀掉压测(见 ../GOTCHAS.md #4)
# 结果 results/pd_intranode_c{2,32}.jsonl
```

## 步骤 7 — 清理

```bash
spur exec "$JOB" bash -c 'docker rm -f pd_probe'   # 容器不随 job 自动清
scancel "$JOB"
```

---

# B. 跨节点 1P1D(⏸️ 未完成 — wiring 全验证,卡 mooncake RDMA 注册)

真 2 机 PD:P=TP8 全在节点 A、D=TP8 全在节点 B,KV 走 ionic RoCE。**本轮到 server
加载阶段被用户叫停,且 mooncake RDMA 撞 ENOMEM(见 GOTCHAS.md)。以下步骤 + 卡点如实记录。**

## B.1 抢第 2 节点 + 拉镜像

```bash
JOB2=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 08:00:00 \
        --exclude=crsuse2-m2m-036,crsuse2-m2m-149,<node-A> scripts/hold_node.sh)
spur exec "$JOB2" true
# 第2节点常缺 mori-0615 镜像;ssh 被封不能 save|load,用 retry-loop 从 docker.io 拉:
spur exec "$JOB2" bash -c "setsid bash $WORK/scripts/do_pull.sh < /dev/null >/dev/null 2>&1 &"
# 等 docker images 出现 mori-0615(cloudfront 偶发 timeout,do_pull 自动重试)
```

## B.2 取双平面 IP + 验 fabric（起 server 前必做）

```bash
# 控制面 ens3 IP(两节点):
spur exec "$JOB"  bash -c "ip -o -4 addr show ens3 | awk '{print \$4}'"   # A 的 ens3
spur exec "$JOB2" bash -c "ip -o -4 addr show ens3 | awk '{print \$4}'"   # B 的 ens3
# 起容器(两节点,同步骤2);然后 rail test（★证明 RDMA 通）:
#   B 起 server: docker exec -d pd_probe ib_write_bw -d ionic_0 -x 1 -F --report_gbits
#   A 跑 client: docker exec pd_probe ib_write_bw -d ionic_0 -x 1 -F --report_gbits <B_ens3_ip>
# 期望 ~200+ Gb/s（本轮实测 213.7）。
```

## B.3 起 P(节点A,TP8)+ D(节点B,TP8)

```bash
P_IP=<A_ens3>; D_IP=<B_ens3>
spur exec "$JOB"  bash -c "docker exec -d pd_probe bash -c 'cd $WORK && ROLE=prefill TP=8 P_IP=$P_IP D_IP=$D_IP bash scripts/pd_server_xnode.sh'"
spur exec "$JOB2" bash -c "docker exec -d pd_probe bash -c 'cd $WORK && ROLE=decode  TP=8 P_IP=$P_IP D_IP=$D_IP bash scripts/pd_server_xnode.sh'"
```

`pd_server_xnode.sh` 关键点:`--disaggregation-transfer-backend mooncake` +
`--disaggregation-ib-device ionic_0` + `MC_GID_INDEX=1` + `SGLANG_LOCAL_IP_NIC=ens3`
(GLOO/MORI_SOCKET_IFNAME 同)。

### ★★ 卡点(本轮实测,见 GOTCHAS.md #1)
backend=mooncake 时每 rank 报 `rdma_context.cpp: Failed to register memory ...:
Cannot allocate memory [12]`(ENOMEM),尽管 memlock 已 unlimited。权重能加载,但两端
卡在 KV 池 init,不 ready。**绕过**:把 `BACKEND=mooncake_tcp` 传给 `pd_server_xnode.sh`
→ 0 报错、TP8 干净加载(KV 改走 ens3 TCP;功能可用,非 RDMA 快路)。
```bash
# 绕过版:
spur exec "$JOB"  bash -c "docker exec -d pd_probe bash -c 'cd $WORK && ROLE=prefill TP=8 BACKEND=mooncake_tcp P_IP=$P_IP D_IP=$D_IP bash scripts/pd_server_xnode.sh'"
spur exec "$JOB2" bash -c "docker exec -d pd_probe bash -c 'cd $WORK && ROLE=decode  TP=8 BACKEND=mooncake_tcp P_IP=$P_IP D_IP=$D_IP bash scripts/pd_server_xnode.sh'"
```

## B.4 router（2 机）+ smoke + 清理
router 指向 `P_IPS=<A_ens3>:30000 D_IPS=<B_ens3>:30000`,其余同 A 节。清理两节点都要
`docker rm -f pd_probe` + `scancel` 两个 job。

## If it doesn't reproduce
见 `GOTCHAS.md`:mooncake RDMA ENOMEM(未解,待调 MC_SLICE_SIZE/dmabuf);login-node
NSS 抽风阻塞 submit/scancel;冷启动 staggered 加载"看起来卡住"看 VRAM 判活。

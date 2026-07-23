# 严格复现 — DSv4 + SGLang 单机 on spur

从零到 smoke 通过 + 对拍。全程在 login 节点操作,通过 `spur exec` 进计算节点。
账户/QOS/节点等细节见 `GOTCHAS.md`(**先读它,能省几小时**)。

前置:你在 spur login 节点(如 crs-m2m-cpu-spur-009),用户属于某账户(本例 `amd-primus`),
模型已在 `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`。

---

## 步骤 1 — 占一个 8 卡节点(关键姿势:burst-qos + 排除坏节点)

```bash
# hold_node.sh: 睡 8h 的占位任务(见 scripts/hold_node.sh)
cat > ~/hold_node.sh <<'EOF'
#!/bin/bash
echo "held: $(hostname) job=$SLURM_JOB_ID gpus=$ROCR_VISIBLE_DEVICES"
sleep 28800
EOF

# 提交:不带 -A(默认自己账户)、-q amd-burst-qos、-G8
JOB=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 08:00:00 ~/hold_node.sh)

# ★必须验证真的占住了(不是一闪即 requeue):
spur exec "$JOB" true && echo "稳! job=$JOB node=$(squeue -j $JOB -h -o %N)"
# 若 JobHoldMaxRequeue:抓失败节点名,加 --exclude 重试。见 GOTCHAS #1。
# 本次踩坑黑名单示例: --exclude=crsuse2-m2m-036,crsuse2-m2m-149
```

## 步骤 2 — 起容器(★ 挂载用绝对路径,不要用 $HOME)

```bash
IMG=rocm/sgl-dev:sglang-0.5.13.post1-rocm720-mi35x-mori-0615
spur exec "$JOB" bash -c "
  docker pull $IMG   # 首次约 100GB+,节点本地缓存后秒起
  docker rm -f dsv4_r4 2>/dev/null || true
  docker run -d --name dsv4_r4 --network=host --ipc=host --shm-size=32g \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v /shared_nfs:/shared_nfs -v /home/\$USER:/home/\$USER \
    --entrypoint '' $IMG sleep infinity
"
# ★坑#2: spur exec 环境是 root@/,$HOME 会展开成 /root。容器挂载必须写绝对路径
#   /home/yihou:/home/yihou,否则容器里看不到你的脚本。/shared_nfs 用绝对路径所以没问题。
```

## 步骤 3 — 启动 R4 no-DP server(冷启动 ~23min)

```bash
# 把 scripts/r4_server.sh 放到 NFS 下能被容器读到的路径,然后:
WORK=/home/$USER/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round1_server_smoke
spur exec "$JOB" bash -c "
  docker exec -d dsv4_r4 bash -c 'cd $WORK && bash r4_server.sh nodp 30000 32 > $WORK/server_nodp.log 2>&1'
"
# 等 ready(权重加载 ~700s + cuda-graph 捕获,总 ~23min):
#   tail -f $WORK/server_nodp.log   → 出现 "The server is fired up and ready to roll!"
# ★坑#3: cuda-graph 捕获阶段日志静默数分钟、GPU 低利用率,不是卡死(显存会持续上涨)。
```

R4 recipe 的两个杠杆(`scripts/r4_server.sh` 已内置,一个都不能少):
- `--attention-backend dsv4`
- 全套 `SGLANG_OPT_USE_FUSED_COMPRESS*` 等 env(见脚本)
- no-DP: `--chunked-prefill-size 8192`(适用 conc≤128);conc≥256 用 `dp` 模式。

## 步骤 4 — smoke 测试

```bash
spur exec "$JOB" bash -c 'docker exec dsv4_r4 curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=? Answer in one word.\"}],\"max_tokens\":32,\"temperature\":0}"'
# 期望: {"choices":[{"message":{"content":"Two."...}}]}  → 见 results/smoke_result.json
```

## 步骤 5 — 对拍压测(可选,验证性能与 legacy 一致)

```bash
# scripts/r4_sweep.sh:与 legacy 完全相同的 8k/1k、random-range 1.0、num-prompts=10*conc
spur exec "$JOB" bash -c "docker exec -d dsv4_r4 bash -c \
  'cd $WORK/../round2_bench && bash r4_sweep.sh \"2 32\" 30000 \$PWD/out nodp > bench.log 2>&1'"
# ★注意用 docker exec -d(detached),否则 spur exec 会话结束会杀掉后台压测(坑#4)。
# 结果 out/r4_nodp_c{2,32}.jsonl,对比 results/legacy_baseline/。
```

## 步骤 6 — 清理(见 CLEANUP)

```bash
spur exec "$JOB" bash -c 'docker rm -f dsv4_r4'   # 容器不随 job 自动清,务必手动
scancel "$JOB"                                     # 释放节点
```

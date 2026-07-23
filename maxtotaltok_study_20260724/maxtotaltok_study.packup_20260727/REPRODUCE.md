# Reproduction kit — DSv4-Pro 1P1D KV 用量 + max-total-tokens 扫描

目标：从干净的两台 MI355X 节点复现 KV 真实用量测量 + max-total-tokens 扫描 + 5态队列分析。
预计耗时：环境+MVP ~15min；每个 launch 点冷启 ~2-9min（NFS 冷载不定）+ cuda-graph ~50s；bench conc=128 ~7min。

## 0. 前置（开跑前准备好）

- **机器**：2 台 8×MI355X。本研究 prefill=chi2832(10.2.122.79) / decode=chi2878(10.2.122.3)。
  （2026-07-24 首轮 prefill 用 chi2879=10.2.122.10；节点可换，改脚本 IP 即可。）Step 全跨节点 PD。
- **需要的 secret（值不含在此，自行获取）**：
  - Docker registry 登录 —— 团队 vault（镜像 `lmsysorg/sglang-rocm` 是公开的，但 infera registry 需登录）。
  - 集群 SSH —— 跳板 `root@149.28.124.225` ProxyJump 到 chiXXXX。
- **仓外依赖（绝对路径）**：
  - 模型：`/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro`（真权重, symlink→john；容器须 `-v /mnt/vast:/mnt/vast`）。
    起测前 `stat -c %s <shard>` 确认非 LFS stub：`model-00001-of-00064.safetensors` = 1,853,358,176 B。
  - host libionic：`/usr/lib/x86_64-linux-gnu/libionic.so.1`（容器内 provider 不匹配，必须 overlay，见 §1）。
- **镜像**：`lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612`（digest sha256:9365...a95b，见 environment.md）。
- **共享盘**：`/mnt/vast/c_huggingface`（脚本+产物落地；compute 节点看不到 /tmp）。

## 1. 起容器 + libionic 修复 + RDMA MVP（两节点）

```bash
# 每节点 host 上
docker run -d --name mtt_pd --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined -v /mnt/vast:/mnt/vast \
  --entrypoint "" lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612 sleep infinity

# libionic overlay（否则 ibv_devinfo PORT_ACTIVE=0 → 静默退 TCP）
HL=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=$(basename "$HL")
docker cp "$HL" mtt_pd:/usr/lib/x86_64-linux-gnu/$B
docker exec mtt_pd bash -lc "cd /usr/lib/x86_64-linux-gnu && ln -sf $B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../$B libionic-rdmav34.so && ldconfig; ibv_devinfo | grep -c PORT_ACTIVE"
# 期望 8

# RDMA MVP（先证 fabric）：decode 节点起 server，prefill 节点当 client
# on decode:  docker exec -d mtt_pd bash -lc "ib_write_bw -d ionic_0 -x 1 -F --report_gbits"
# on prefill: docker exec mtt_pd ib_write_bw -d ionic_0 -x 1 -F --report_gbits <decode_ip>
# 期望 BW avg ~338-339 Gb/s（健康同 rail）。跑完 pkill ib_write_bw。
```

放脚本到共享盘：把本 packup `scripts/*.sh` 拷到 `/mnt/vast/c_huggingface/mtt_scripts/`（容器可见）。

## 2. 起 1P1D（mooncake）

```bash
OUT=/mnt/vast/c_huggingface/mtt_study
# prefill leg (chi2832)
docker exec -d mtt_pd bash -lc "ROLE=prefill MY_IP=10.2.122.79 OUT=$OUT LOG=$OUT/prefill.log bash /mnt/vast/c_huggingface/mtt_scripts/launch_leg.sh"
# decode leg (chi2878) — 加 MAX_TOTAL_TOKENS 扫不同池；不加=默认大池
docker exec -d mtt_pd bash -lc "ROLE=decode MY_IP=10.2.122.3 MAX_TOTAL_TOKENS=163840 OUT=$OUT LOG=$OUT/decode.log bash /mnt/vast/c_huggingface/mtt_scripts/launch_leg.sh"
# 等两腿 "fired up and ready"（读数看 "DSV4 pool sizes: full=N" / "Memory pool end"）

# router（原生 sglang_router，在 prefill 节点）
docker exec -d mtt_pd bash -lc "P_IP=10.2.122.79 D_IP=10.2.122.3 RPORT=8100 OUT=$OUT LOG=$OUT/router.log bash /mnt/vast/c_huggingface/mtt_scripts/launch_router.sh"
# 等 log 出现 2 次 "Activated"；smoke:
curl -s http://10.2.122.79:8100/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro","prompt":"1+1=","max_tokens":8,"temperature":0}'
# 期望返回含 "2"（PD 配对+KV 手递成功）
```

## 3. 跑 conc=128 bench + 抓运行时 KV 峰值

```bash
docker exec -d mtt_pd bash -lc "P_IP=10.2.122.79 RPORT=8100 CONC=128 NPROMPT=1280 TAG=c128 OUT=$OUT LOG=$OUT/bench.log bash /mnt/vast/c_huggingface/mtt_scripts/bench.sh"
# 稳态抓 decode 峰值：
docker exec mtt_pd bash /mnt/vast/c_huggingface/mtt_scripts/mtt_peaks.sh $OUT/decode.log
# 期望 gmu0.90 大池: peak_#full_token≈166K, retract=0, running≈16
# bench 完读吞吐:
grep -E "Total token throughput|Successful requests" $OUT/bench.log
```

## 4. 扫 max-total-tokens（换池换轮）

```bash
# 每换一个 max-total-tokens：kill decode + router → 等 VRAM<10GB → 重起 decode(新池) + router
bash scripts/reset.sh   # 从本地驱动两节点 kill + 等 VRAM（PN/DN 改成你的节点）
# 扫描点：262144 / 163840 / 147456 / 131072（131072 会崩，验证 DSv4 无 retract）
# 每点重复 §2 decode + §2 router + §3 bench
```

## 5. 5态队列分析（出交付1/2 的表）

```bash
# 把 decode + prefill log 拉到本地，跑分析脚本：
python3 scripts/extract_pd_stats.py --prefill <prefill.log> --decode <decode.log> --label "<config>"
# 读 state5 running / state4 transfer-in / state1 queue-req → 判 P/D bound + 算真实 KV
```

## Expected output（对齐 README 结论）

- decode 大池 conc=128：peak #full token ≈166K/rank（4%）、retract=0 → 真实 KV ≈5.6 GiB/卡。
- max-total-tokens=163840：峰值 93%、retract=0、吞吐 29.6K（持平大池）→ 5.0 GiB/卡地板。
- max-total-tokens=131072：撑到 100% → **decode 崩** `retract_decode NotImplementedError`（验证 DSv4 无 retract）。
- prefill：full token usage 峰值 2% → 真实 KV ≈2.1 GiB/卡（瞬态）。
- 5态：prefill queue-req 堆积 + decode running 只 24% → **prefill-bound**。

## If it doesn't reproduce

见 `notes.md`。最常见：① PORT_ACTIVE=0 → libionic overlay 没做；② 换轮没等 VRAM 回落 → decode
`Not enough memory` 死 + 留 KFD 僵尸（host 层按 pid kill）；③ router 只 Activate 1 个 worker / circuit-breaker
flapping → kill decode+router 重起（router 进程名 `sglang::router`，用 `mtt_kill_router.sh` 清端口）；
④ bench_serving import 错 → 需 `PYTHONPATH=/sgl-workspace/sglang/python`（已在 bench.sh）。

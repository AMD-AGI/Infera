# Reproduction kit — DSv4-Pro KV pool × gmu study

目标：从一台干净的 MI355X 节点复现"每卡 KV pool = f(gmu)"曲线（Step 1/2）+ 运行时不超池验证（Step 3）。
预计耗时：单点 launch 到 "Memory pool end" ~110s（暖缓存）；5+2 点扫描 ~15min；真跑验证 ~10min。总 ~30min。

## 0. 前置（开跑前准备好）

- **机器**：1 台 8×MI355X (gfx950) 节点（本研究用 chi2879）。Step 1/2/3 全单机，**无需跨节点 RDMA**。
- **需要的 secret（值不含在此，自行获取）**：
  - Docker registry `infera` 登录 —— 团队 vault。
  - 集群 SSH —— 跳板 `root@149.28.124.225`（chi2866）ProxyJump 到 chiXXXX。
- **仓外依赖（绝对路径）**：
  - 模型：`/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro`（真权重, symlink→john；容器须 `-v /mnt/vast:/mnt/vast`）。
    起测前 `stat -c %s <shard>` + `tokenizer.json` 确认非 LFS stub（见 environment.md 校验值）。
- **镜像**：`infera/engine-sglang:pd-mcgate`（digest 见 environment.md）。内含 sglang 0.5.15.post1。
- **共享盘**：`/mnt/vast/c_huggingface`（脚本 + 产物落地处；compute 节点看不到 /tmp）。

## 1. 起容器（若节点上还没有）

```bash
# host 上，节点已有则复用现成的 dsv4_pd_sgl 容器
docker run -d --name dsv4_pd_sgl --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined -v /mnt/vast:/mnt/vast \
  --entrypoint "" infera/engine-sglang:pd-mcgate sleep infinity
```

## 2. 放脚本到共享盘

把本 packup 的 `scripts/kv_gmu_sweep.sh`、`scripts/run_mix_verify.sh` 拷到
`/mnt/vast/c_huggingface/`（容器内可见）。改脚本头部 `MY_IP` 为本节点 data-plane IP（本研究 10.2.122.10）。

## 3. Step 1 — decode gmu 扫描（核心曲线）

```bash
# 容器内，纯 launch 读数（不发请求）；每点起 decode 单腿→读 KV pool 分配行→kill→等 VRAM 回落
docker exec -d dsv4_pd_sgl bash -lc '
  cd /mnt/vast/c_huggingface
  ROLE=decode GMUS="0.80 0.85 0.88 0.90 0.92" \
  OUT=/mnt/vast/c_huggingface/kvrepro bash kv_gmu_sweep.sh'
# 看结果：
cat /mnt/vast/c_huggingface/kvrepro/decode_summary.csv
grep -hE "DSV4 memory calculation|DSV4 pool sizes" /mnt/vast/c_huggingface/kvrepro/decode_gmu0.90.log | grep "DP0 "
```

## 4. Step 2 — prefill 抽 2 点对照

```bash
docker exec -d dsv4_pd_sgl bash -lc '
  cd /mnt/vast/c_huggingface
  ROLE=prefill GMUS="0.85 0.90" \
  OUT=/mnt/vast/c_huggingface/kvrepro bash kv_gmu_sweep.sh'
# 期望：同 gmu 下 prefill 与 decode 的 full/swa/c4/c128 逐字节一致（池公式与 role 无关）
```

## 5. Step 3 — 运行时不超池验证（单机 mix 真跑）

```bash
# 起 mix server @gmu0.90 → 发 128 请求 @conc64 @8k/1k → 抓运行时 token usage + retract
docker exec -d dsv4_pd_sgl bash /mnt/vast/c_huggingface/run_mix_verify.sh
# 完成后看：
L=/mnt/vast/c_huggingface/kvcache_gmu_study_*/mix_verify_gmu0.90.log
grep -c retract $L                    # 期望 0
grep -oE "swa token usage: [0-9.]+" $L | sort -n | tail -1   # 期望 ~0.05 (5%)
```

## 6. 解析 + 出表

```bash
# 本地：把各 gmu 点的 DSV4 memory calculation / DSV4 pool sizes 行填进 results/pool_breakdown.py 的 data dict
python3 results/pool_breakdown.py      # 打印 gmu→KVpool GB→token 容量 主表 + 线性系数
```

## Expected output（对齐 REPORT.md 结论）

| gmu | 每卡 KV pool | full_token/卡 |
|-----|-------------|--------------|
| 0.80 | 91.24 GB | 7,053,056 |
| 0.85 | 105.08 GB | 8,135,424 |
| 0.90 | 118.93 GB | 9,217,792 |
| 0.92 | 124.47 GB | 9,650,944 |

- 线性：`KVpool_GB ≈ 276.9×gmu − 130.3`，每 +0.01 gmu → **+2.77 GB / +216,470 token**。
- prefill==decode 同 gmu 逐字节一致。
- Step 3：retract=0，swa usage 峰值 5%（远在池内）。

## If it doesn't reproduce

见 `notes.md`。最常见：① `--no-enable-kv-events` 报 unrecognized（那是 infera wrapper 旗标，
裸 sglang 不需要，本脚本已不含）；② 绝对数字与上表不同 → 检查 sglang 版本（0.5.13 vs 0.5.15 语义不同）
+ 权重是否 stub；③ 换 gmu 点之间 VRAM 未回落 → kill 不干净，等 VRAM≈baseline 再起。

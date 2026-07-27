# DSv4 + SGLang 单机 on spur 集群 — 实验交付

**时间**: 2026-07-23 (UTC)
**执行**: yihou @ AMD crsuse2-m2m (amd-spur)
**结论**: ✅ 成功。DeepSeek-V4-Pro 用 SGLang 在单个 8×MI355X 节点上跑通,smoke 正确推理,
性能与 legacy 已知good结果**对拍一致(差异 <1%)**。

---

## 任务

在 spur 集群上找一台机器,学习 legacy 经验,成功调试运行 **DeepSeek-V4-Pro + SGLang**。
- 任务下发:用户口头 spec(参见本仓库 `CLAUDE.md` 的 "CURRENT MAIN TASK")。
- 参考经验(对拍源):`/home/yihou/dev/git/legacy.infera/infera/sglang_single_r4_20260707_080726/`
  (legacy 2026-07-07 在 chi2865 的成功实验)。
- 模型:`/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`(fp8/fp4, 806G)。

## 成果一句话

legacy 的 R4 recipe(stock SGLang TP8 + 两个杠杆:`--attention-backend dsv4` +
全套 `SGLANG_OPT_USE_FUSED_COMPRESS*` env)**在 spur 上原样可复现**,用完全相同的镜像
(mori-0615)。conc=2 对拍 tot/GPU = **154.0 vs legacy 154.2**(<1% 差异)。

## 交付目录结构

```
deliverable_dsv4_sglang_spur_20260723/
├── README.md          ← 本文件(索引+总结)
├── ENVIRONMENT.md     ← 硬件/软件/镜像sha/模型路径/密钥(环境快照)
├── REPRODUCE.md       ← 从零到 smoke+对拍 的严格复现步骤
├── GOTCHAS.md         ← ★spur 集群使用潜规则/踩坑(本次最有价值产出)
├── scripts/
│   ├── hold_node.sh   ← 8卡占位睡眠任务
│   ├── r4_server.sh   ← R4 recipe 启动器(nodp/dp 双模式,模型路径已改spur)
│   ├── r4_sweep.sh    ← bench_serving 压测(与legacy同配置)
│   ├── grab_node.sh   ← 坏节点重试抢占脚本(�window紧张时用)
│   └── smi_job.sh     ← amd-smi 冒烟job模板
└── results/
    ├── smoke_result.json      ← smoke: 1+1=? → "Two."
    ├── r4_nodp_c2.jsonl       ← 本次 c=2 压测原始结果
    ├── r4_nodp_c32.jsonl      ← 本次 c=32(见 COMPARISON)
    ├── COMPARISON.md          ← 对拍汇总表 spur vs legacy
    └── legacy_baseline/       ← legacy c2/c32 jsonl(对拍基线)
```

## 阅读顺序建议

1. **先读 `GOTCHAS.md`** —— spur 集群不写在文档里的潜规则,能省几小时(尤其 `JobHoldMaxRequeue`)。
2. `ENVIRONMENT.md` —— 确认你的环境(镜像 sha、模型路径、驱动版本)。
3. `REPRODUCE.md` —— 照着 6 步复现。
4. `results/COMPARISON.md` —— 看对拍数据。

## 关键环境(详见 ENVIRONMENT.md)

- 节点: crsuse2-m2m-292, 8× **MI355X** (gfx950), ROCm 7.2.0, amdgpu 6.14.14
- 镜像: `rocm/sgl-dev:sglang-0.5.13.post1-rocm720-mi35x-mori-0615`
  (`sha256:976831ec7f1976bb0ff4d469600e38546549e60a4dec7e5148e853694976e387`)
- SGLang `0.0.0.dev14036+g19c78552d.d20260615`, torch `2.9.1+rocm7.2.0`
- **不需要任何密钥**(公开镜像 + 共享 NFS 模型 + 集群自带身份)

## 复现要点(潜规则速记)

| # | 坑 | 一句话解法 |
|---|----|-----------|
| 1 | `JobHoldMaxRequeue`(部分 idle 节点 NODE_FAIL) | `--exclude` 坏节点 + `spur exec <job> true` 验证真占住 |
| 2 | 提交姿势 | `-p amd-spur -q amd-burst-qos`,能不带 `-A` 就不带 |
| 3 | 容器挂载 | 用绝对路径 `-v /home/$USER:/home/$USER`,别用 `$HOME`(spur exec 是 root@/) |
| 4 | 后台长任务 | `docker exec -d`,别用裸 `&`(随会话被杀) |
| 5 | 冷启动~23min 中途"卡住" | 看显存上涨判活,不是死;等 "ready to roll" |

## 未提交/依赖(见 ENVIRONMENT.md)

- 模型 `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`(806G,不入 git)
- 原始 workspace(未删除,按规矩保留): `temp/dsv4_sglang_spur/`
  含各 round 的 RESULT/NOTES 和完整 `working_process.md` 调试索引。
- server/压测日志见原始 workspace 的 `round*/`(大日志未拷入交付,需要可另取)。

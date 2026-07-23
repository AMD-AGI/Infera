# DSv4 + SGLang PD 分离 on spur 集群 — 实验交付

**时间**: 2026-07-23 (UTC)
**执行**: yihou @ AMD crsuse2-m2m (amd-spur)
**结论**: ✅ **节点内 1P1D PD 分离端到端跑通**(smoke + 压测过 router)。
⏸️ 跨节点(真 2 机 RDMA PD)wiring 全验证 + 脚本化,卡在 mooncake RDMA 内存注册
`ENOMEM`,`mooncake_tcp` 可绕过 —— 用户叫停多机,留作后续。

> 这是 spur DSv4+SGLang 系列的**第二个实验(PD 分离)**。第一个是单机 R4
> serving(见 `../../spur_repro/README.md`)。两者共享集群/镜像/模型,PD 在其上加 PD wiring。

---

## 任务

借鉴 legacy PD 分离成功经验 + 单机 R4 的 mix 踩坑,在 spur 上调通 DSv4 PD 分离。
- 任务下发:用户口头 spec(参见本仓库 `CLAUDE.md` 的 "CURRENT MAIN TASK" + 后续对话)。
- 参考经验(对拍源):`/home/yihou/dev/git/legacy.infera/infera/pd_1p1d_dpa_8k1k_20260714_235121/`
  (legacy 2026-07-14 chi2774↔chi2800 的 1P1D/2P1D 成功实验)。
- 模型:`/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`(fp8/fp4, 806G)。

## 成果一句话

用 `mooncake_tcp` 让**节点内 1P1D**(P=TP4 GPU0-3、D=TP4 GPU4-7,KV 走 loopback)
完全免 RDMA/ionic 注入就把整套 PD wiring(disaggregation-mode + bootstrap +
mini-lb router)在单个 8×MI355X 节点上验证通过。smoke 过 router → `"Two."`。

## 关键结论:两个通信层要分清

| 层 | 用什么 | 走哪 | 备注 |
|----|--------|------|------|
| **TP 组内**(prefill 4 rank / decode 4 rank 的 all-reduce/gather) | **RCCL**(日志 `nccl==2.27.7`,ROCm 上即 RCCL) | XGMI 卡间直连 | 与 transfer-backend 无关,一直是 RCCL |
| **PD 的 KV 搬运**(prefill→decode handoff) | **mooncake**(独立 transfer engine,非集合通信库) | 本轮走 `mooncake_tcp` = loopback TCP | 换 `mooncake` 即 RDMA,但仍非 RCCL |

**为啥节点内选 tcp**:MVP 求正确避坑 —— `mooncake` RDMA 要向 ionic 注册内存区,
跨节点撞 `ENOMEM`;`mooncake_tcp` 走 127.0.0.1 零 RDMA 配置就通。
**够快吗**:功能够,非最优。loopback TCP 有 GPU→host→GPU 拷贝开销,但 **DSv4 是 MLA、
KV 极小**(~43KB/token,8k 请求 ~352MB,handoff 只传一次),所以这条慢路对吞吐影响很小
(legacy 已证换传输后端 mori≈mooncake <2%)。真压性能应上 RDMA loopback 或 GPU P2P。

## 交付目录结构

```
pd_disagg/
├── README.md          ← 本文件(索引+成果+通信层解释)
├── ENVIRONMENT.md     ← 硬件/软件/镜像sha/模型/fabric(ionic RoCE)/密钥
├── REPRODUCE.md       ← 节点内 1P1D 严格复现 + 跨节点(未完成)步骤
├── GOTCHAS.md         ← ★PD 特有坑(RDMA ENOMEM、NSS 抽风、免注入等)
├── scripts/
│   ├── hold_node.sh          ← 8卡占位睡眠任务
│   ├── pd_server.sh          ← 节点内 PD server(TP4,mooncake_tcp,loopback)
│   ├── pd_router.sh          ← mini-lb router :8100(1P1D loopback)
│   ├── pd_sweep.sh           ← bench_serving 压测(过 router)
│   ├── pd_server_xnode.sh    ← 跨节点 PD server(TP8,mooncake/tcp,ionic GID1,ens3)
│   └── do_pull.sh            ← 节点本地镜像 retry-loop 拉取(跨节点第2机用)
├── results/
│   ├── pd_smoke_result.json  ← smoke 过 router:1+1=? → "Two."
│   ├── pd_intranode_c2.jsonl ← 节点内 PD 压测 c=2
│   ├── pd_intranode_c32.jsonl← 节点内 PD 压测 c=32
│   └── pd_sweep.log          ← 压测汇总
└── logs/
    ├── prefill.log / decode.log / router.log   ← 节点内 1P1D(成功)
    ├── xtcp_prefill.log / xtcp_decode.log      ← 跨节点 mooncake_tcp(干净加载)
    └── xnode_prefill.log / xnode_decode.log    ← 跨节点 mooncake(★含 RDMA ENOMEM 证据)
```

## 阅读顺序建议

1. **先读 `GOTCHAS.md`** —— PD 特有的坑(尤其 mooncake RDMA `ENOMEM` 和 NSS 抽风)。
   基础 spur 潜规则见上一级 `../../spur_repro/GOTCHAS.md`(JobHoldMaxRequeue、绝对路径挂载等)。
2. `ENVIRONMENT.md` —— fabric(ionic RoCE)拓扑、镜像、模型。
3. `REPRODUCE.md` —— 节点内 6 步复现;跨节点步骤(卡点已标注)。

## 结果速记(节点内 1P1D,TP4+TP4,8k/1k,过 router)

| conc | tot tok/s | out tok/s | median TTFT | median ITL |
|------|-----------|-----------|-------------|------------|
| 2    | 1110      | 123       | 1853 ms     | 14.6 ms    |
| 32   | 4252      | 472       | 48277 ms    | 18.8 ms    |

> 注:这是 **TP4/角色 + tcp handoff** 的功能数,非调优数,也**不是对拍目标**
> (与上一级单机 TP8 R4 是不同 shape)。c=32 TTFT 高是 TP4 prefill 在 c=32 快速饱和,
> 属预期。`#transfer-req` 出现在 decode.log 证明 KV 真在 P→D 传。

## 未提交/依赖

- 模型 `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro`(806G,不入 git)
- 原始 workspace(未删,按规矩保留):`temp/dsv4_sglang_spur/round3_pd_disagg/`
  含 `NOTES.md` / `RECON.md` 全过程记录 + `working_process.md` 全局索引。
- **本实验不需要任何密钥**(公开镜像 + 共享 NFS 模型 + 集群自带身份)。
